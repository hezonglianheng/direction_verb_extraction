# encoding: utf8

"""本地部署模型的 OpenAI 兼容服务生命周期。

背景：
    `extract` 与 `replace` 的 LLM 筛选原本只面向远程 API。换成本地小模型时，除了把
    `model_config/api_keys.json` 里该模型的 `provider` / `base_url` 指到本地服务，还需要
    在程序启动时把服务拉起来、程序结束时关掉。本模块只负责这件事，不碰提示语，也不碰
    筛选逻辑。

配置（`model_config/api_keys.json` 的某个模型条目）：

    "local-qwen": {
        "model_name": "Qwen/Qwen3.8-27B",
        "provider": "openai",
        "base_url": "http://127.0.0.1:8000/v1",   # 本地小模型地址
        "api_key": "EMPTY",
        "temperature": 0,
        "timeout": 600,
        "max_retries": 6,
        "proxy": false,
        "disable_thinking": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}
        },
        "local_service": {
            "enabled": true,
            "service_dir": "/home/qinyuhang/model_service",
            "start_command": ["bash", "service.sh", "start", "--wait"],
            "stop_command": ["bash", "service.sh", "stop"],
            "status_command": ["bash", "service.sh", "status"]
        }
    }

    没有 `local_service` 块（或 `enabled` 为 false）的模型一律按远程 API 处理，本模块
    不会启动或关闭任何进程。`base_url` 是唯一的请求地址来源；如果 `local_service` 里
    也写了 `base_url`，必须与它一致，避免“服务起在 A 地址、模型连到 B 地址”。

生命周期：
    `for_model(...)` 是推荐入口。只有本次程序真正拉起服务时才登记 atexit，并在
    上下文结束时停止；如果启动前服务就已经在运行（用户手动起的），只连接、不关闭。
    启动命令返回后还会轮询健康检查，确保“程序已启动”与“模型可连接”不是两件事。
"""

from __future__ import annotations

import atexit
import os
import shlex
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

import config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_PREFIX = "[本地模型服务]"
_MISSING = object()
_HEALTH_OPENER = build_opener(ProxyHandler({}))
"""健康检查强制直连：本地地址不应该被 HTTP(S)_PROXY 绕到代理上。

模型请求的代理由 agents.get_model 的 proxy 字段控制；健康检查若走环境代理，
会出现“模型已经连上、健康检查却一直失败”的假故障。
"""


def _log(message: str) -> None:
    """生命周期日志不能反过来阻断清理：stdout 被关闭/管道断开时静默跳过。"""
    try:
        print(f"{_LOG_PREFIX} {message}", flush=True)
    except (BrokenPipeError, OSError):
        pass


def _resolve_service_dir(value: str | os.PathLike[str]) -> Path:
    """相对路径按项目根目录解析，避免从子目录启动时找错 service.sh。"""
    if value is None or not str(value).strip():
        raise ValueError("local_service.service_dir 不能为空")
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


def _validate_http_url(value: str, field: str) -> str:
    parsed = urlsplit(str(value))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field} 不是合法的 http(s) 地址：{value!r}")
    return str(value)


def _command(value: Any, default: tuple[str, ...], field: str) -> tuple[str, ...]:
    """把字符串命令或列表命令统一成参数元组。"""
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        parts = shlex.split(value)
    elif isinstance(value, (list, tuple)):
        parts = [str(part) for part in value]
    else:
        raise ValueError(f"local_service.{field} 必须是字符串或字符串列表，收到 {type(value).__name__}")
    if not parts or not parts[0].strip():
        raise ValueError(f"local_service.{field} 不能为空")
    return tuple(parts)


def _optional_command(value: Any, default: tuple[str, ...], field: str) -> tuple[str, ...] | None:
    """同 `_command`，但显式 null 表示“没有这条命令”。"""
    if value is None:
        return None
    return _command(value, default, field)


def _positive_seconds(value: Any, default: float, field: str) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"local_service.{field} 必须是正数，收到 {value!r}")
    return float(value)


def _env_mapping(value: Any, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"local_service.{field} 必须是对象，收到 {type(value).__name__}")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"local_service.{field} 的键必须是非空字符串")
        result[key] = str(item)
    return result


def _with_local_proxy_bypass(env: dict[str, str]) -> dict[str, str]:
    """把本地回环地址加入 NO_PROXY/no_proxy。

    model_service/service.sh 用 curl 访问 ``BASE_URL/health``；curl 会读环境代理。
    如果 HTTP(S)_PROXY 存在但 NO_PROXY 没覆盖 127.0.0.1，service.sh 的 --wait 会
    一直判健康失败，即使本模块自己的直连健康检查已经通过。
    """
    loopback = ("127.0.0.1", "localhost", "::1", "[::1]")
    for key in ("NO_PROXY", "no_proxy"):
        existing = [item.strip() for item in str(env.get(key, "")).split(",") if item.strip()]
        if "*" in existing:
            continue
        for item in loopback:
            if item not in existing:
                existing.append(item)
        env[key] = ",".join(existing)
    return env


def derive_health_url(base_url: str, health_path: str = "/health") -> str:
    """由模型 base_url 推导健康检查地址。

    ``http://127.0.0.1:8000/v1`` -> ``http://127.0.0.1:8000/health``；
    ``http://127.0.0.1:8000/api/v1`` -> ``http://127.0.0.1:8000/api/health``。
    """
    parsed = urlsplit(str(base_url))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"本地模型 base_url 不是合法的 http(s) 地址：{base_url!r}")

    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    suffix = "/" + str(health_path).lstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path + suffix, "", ""))


@dataclass(frozen=True)
class LocalServiceConfig:
    """一个模型对应的本地服务启动/停止参数。"""

    model_name: str
    base_url: str
    service_dir: Path
    health_url: str
    start_command: tuple[str, ...]
    stop_command: tuple[str, ...]
    status_command: tuple[str, ...] | None
    start_timeout: float
    health_timeout: float
    stop_timeout: float
    status_timeout: float
    extra_env: dict[str, str]

    @classmethod
    def from_model_config(cls, model_cfg: Mapping[str, Any], model_name: str) -> "LocalServiceConfig | None":
        """从模型配置里解析本地服务块；没有或显式关闭时返回 None。"""
        raw = model_cfg.get("local_service")
        if raw is None or raw is False:
            return None
        if raw is True:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"模型 {model_name!r} 的 local_service 必须是对象、true 或 false，"
                f"收到 {type(raw).__name__}"
            )

        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"模型 {model_name!r} 的 local_service.enabled 必须是布尔值，收到 {enabled!r}")
        if not enabled:
            return None

        model_base_url = model_cfg.get("base_url")
        if not model_base_url:
            raise ValueError(f"模型 {model_name!r} 使用 local_service 时必须提供 base_url")
        model_base_url = _validate_http_url(str(model_base_url), "base_url")

        declared_base_url = raw.get("base_url")
        if declared_base_url is not None:
            declared = str(declared_base_url).rstrip("/")
            actual = model_base_url.rstrip("/")
            if declared != actual:
                raise ValueError(
                    f"模型 {model_name!r} 的 local_service.base_url={declared_base_url!r} "
                    f"与模型条目 base_url={model_base_url!r} 不一致；请只保留一处地址。"
                )

        health_path = raw.get("health_path", config.LOCAL_MODEL_SERVICE_HEALTH_PATH)
        health_url = _validate_http_url(
            raw.get("health_url") or derive_health_url(model_base_url, str(health_path)),
            "local_service.health_url",
        )
        status_raw = raw.get("status_command", _MISSING)
        status_command = (
            _command(None, config.LOCAL_MODEL_SERVICE_STATUS_COMMAND, "status_command")
            if status_raw is _MISSING
            else _optional_command(status_raw, config.LOCAL_MODEL_SERVICE_STATUS_COMMAND, "status_command")
        )

        return cls(
            model_name=model_name,
            base_url=model_base_url,
            service_dir=_resolve_service_dir(raw.get("service_dir", config.LOCAL_MODEL_SERVICE_DIR)),
            health_url=str(health_url),
            start_command=_command(
                raw.get("start_command"), config.LOCAL_MODEL_SERVICE_START_COMMAND, "start_command"
            ),
            stop_command=_command(
                raw.get("stop_command"), config.LOCAL_MODEL_SERVICE_STOP_COMMAND, "stop_command"
            ),
            status_command=status_command,
            start_timeout=_positive_seconds(
                raw.get("start_timeout"),
                config.LOCAL_MODEL_SERVICE_START_TIMEOUT_SECONDS,
                "start_timeout",
            ),
            health_timeout=_positive_seconds(
                raw.get("health_timeout"),
                config.LOCAL_MODEL_SERVICE_HEALTH_TIMEOUT_SECONDS,
                "health_timeout",
            ),
            stop_timeout=_positive_seconds(
                raw.get("stop_timeout"),
                config.LOCAL_MODEL_SERVICE_STOP_TIMEOUT_SECONDS,
                "stop_timeout",
            ),
            status_timeout=_positive_seconds(
                raw.get("status_timeout"),
                config.LOCAL_MODEL_SERVICE_STATUS_TIMEOUT_SECONDS,
                "status_timeout",
            ),
            extra_env={
                **{str(k): str(v) for k, v in config.LOCAL_MODEL_SERVICE_EXTRA_ENV.items()},
                **_env_mapping(raw.get("extra_env"), "extra_env"),
            },
        )


class LocalModelService:
    """管理一个本地 OpenAI 兼容服务的启动、健康检查与关闭。

    只管理“本实例启动的进程”：启动前 `/health` 已通过，或状态命令显示进程已存在时，
    `owned` 保持 False，退出时不执行 stop，避免误关用户手动启动的服务。
    """

    def __init__(self, service_config: LocalServiceConfig):
        self.config = service_config
        self._started = False
        self._owned = False
        self._previous_sigterm: Any = _MISSING

    @property
    def owned(self) -> bool:
        """本次程序是否负责关闭该服务。"""
        return self._owned

    def health_ok(self) -> bool:
        """访问健康检查地址；任何网络/HTTP 异常都按“尚未就绪”处理。"""
        request = Request(self.config.health_url, headers={"Accept": "application/json"})
        try:
            with _HEALTH_OPENER.open(request, timeout=3) as response:
                return 200 <= int(response.status) < 300
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            return False

    def wait_until_healthy(self, timeout: float | None = None) -> bool:
        """轮询到健康检查通过或超时；启动命令自带 --wait 时通常一次就通过。"""
        limit = self.config.health_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + limit
        while True:
            if self.health_ok():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(2)

    def _service_status(self) -> int | None:
        """执行状态命令，返回退出码；没有状态命令时返回 None。"""
        command = self.config.status_command
        if command is None:
            return None
        try:
            completed = self._run(command, self.config.status_timeout, capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"无法执行本地服务状态命令 {command!r}：{exc}") from exc
        if completed.stdout:
            print(completed.stdout.rstrip(), flush=True)
        if completed.stderr:
            print(completed.stderr.rstrip(), flush=True)
        if completed.returncode not in (0, 1, 3):
            raise RuntimeError(
                f"本地服务状态命令返回未知退出码 {completed.returncode}；"
                "请检查 status_command，或在该模型配置里写 \"status_command\": null"
            )
        return completed.returncode

    def _run(
        self,
        command: tuple[str, ...],
        timeout: float,
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        # service.sh 用自己的 STARTUP_TIMEOUT 控制等待窗口；配置是唯一默认来源，
        # 但 local_service.extra_env 仍可显式覆盖。
        env["STARTUP_TIMEOUT"] = str(int(self.config.start_timeout))
        env.update(self.config.extra_env)
        env = _with_local_proxy_bypass(env)
        return subprocess.run(
            list(command),
            cwd=str(self.config.service_dir),
            env=env,
            timeout=timeout,
            check=False,
            text=True,
            capture_output=capture_output,
        )

    def _install_sigterm_handler(self) -> None:
        """兜底处理 SIGTERM：先停服务，再按原 handler 继续退出。

        普通退出与 Ctrl-C 已经由上下文管理器/atexit 覆盖；这里只补“被调度器
        SIGTERM”这一种不会触发 atexit 的路径。只在主线程安装，且不改写其他线程。
        """
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            previous = signal.getsignal(signal.SIGTERM)
            if previous == signal.SIG_IGN:
                # 调用方明确要求忽略 SIGTERM；恢复后保持忽略，不擅自改成退出。
                handler = signal.SIG_IGN
            else:
                def handler(signum, frame):  # type: ignore[no-redef]
                    self.stop()
                    if callable(previous):
                        previous(signum, frame)
                    else:
                        raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, handler)
            self._previous_sigterm = previous
        except (ValueError, OSError, AttributeError):
            # 非主线程/嵌入解释器等环境装不上信号处理器时，仍有 atexit 兜底。
            self._previous_sigterm = _MISSING

    def _restore_sigterm_handler(self) -> None:
        if self._previous_sigterm is _MISSING:
            return
        try:
            signal.signal(signal.SIGTERM, self._previous_sigterm)
        except (ValueError, OSError, AttributeError):
            pass
        self._previous_sigterm = _MISSING

    def ensure_started(self) -> "LocalModelService":
        """确保服务可用；已启动时直接复用，不重复拉起。"""
        if self._started:
            return self

        if os.name == "nt":
            raise RuntimeError(
                "本地模型服务生命周期管理目前只支持 POSIX；默认 service.sh 面向 Linux"
            )

        _log(f"准备本地模型 {self.config.model_name}：{self.config.base_url}")

        if self.health_ok():
            _log(f"服务已在运行，直接连接：{self.config.health_url}")
            self._started = True
            self._owned = False
            return self

        if not self.config.service_dir.is_dir():
            raise RuntimeError(
                f"本地模型服务目录不存在：{self.config.service_dir}；"
                "请检查 local_service.service_dir"
            )

        status = self._service_status()
        if status not in (None, 0, 1, 3):
            raise RuntimeError(
                f"本地服务状态命令返回未知退出码 {status}；"
                "请检查 status_command，或在该模型配置里写 \"status_command\": null"
            )
        if status == 0:
            # status 的 0 来自 service.sh 自己配置的 BASE_URL，不一定等于模型配置里的
            # base_url/health_url；再确认一次，避免“状态说就绪、模型请求却连不上”。
            if not self.wait_until_healthy(timeout=min(10.0, self.config.health_timeout)):
                raise RuntimeError(
                    "状态命令显示服务已就绪，但模型配置的健康检查未通过："
                    f"{self.config.health_url}；请确认模型 base_url 与 "
                    "model_service/config.py 的 HOST/PORT 一致"
                )
            _log("状态命令显示服务已就绪，直接连接")
            self._started = True
            self._owned = False
            return self

        # 0/1 说明进程已在（0 的健康检查可能刚好在两次调用之间恢复）；
        # 3/None 说明没有运行中的服务，由本次程序负责启动并关闭。
        self._owned = status in (None, 3)
        if not self._owned:
            _log("服务进程已存在但尚未就绪，等待其完成启动（退出时不会关闭它）")
        else:
            _log("服务未运行，开始启动")

        self._started = True
        try:
            # 先登记 atexit/SIGTERM，再开始可能持续数小时的启动命令与健康等待：
            # 否则这段时间收到 SIGTERM 会直接终止进程，已拉起的服务无人关闭。
            if self._owned:
                atexit.register(self.stop)
                self._install_sigterm_handler()

            # 给 service.sh 自己的 STARTUP_TIMEOUT 留一点收尾余量：否则外层先超时、
            # 内层还以为在等待，stop 会被迫在启动子进程未退出时介入。
            completed = self._run(self.config.start_command, self.config.start_timeout + 30)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"本地服务启动命令返回退出码 {completed.returncode}："
                    f"{' '.join(self.config.start_command)}"
                )
            if not self.wait_until_healthy():
                raise RuntimeError(
                    f"本地服务启动后 {self.config.health_timeout:g}s 内健康检查仍未通过："
                    f"{self.config.health_url}"
                )
        except BaseException:
            # 本次已经开始的启动失败时，尽量把可能残留的进程停掉，不把脏状态留给下一轮。
            if self._owned:
                self.stop()
            else:
                self._started = False
            raise

        if self._owned:
            _log(f"服务已启动并就绪：{self.config.health_url}")
        else:
            _log(f"服务已由其他进程启动并就绪：{self.config.health_url}")
        return self

    def stop(self) -> None:
        """关闭本次程序启动的服务；重复调用安全，非本程序启动的服务不处理。"""
        if not self._started:
            return
        self._started = False
        if not self._owned:
            return
        self._owned = False
        try:
            atexit.unregister(self.stop)
        except Exception:  # noqa: BLE001 —— 解释器退出阶段 unregister 失败不应阻断清理
            pass

        _log(f"关闭本地模型服务：{' '.join(self.config.stop_command)}")
        try:
            try:
                completed = self._run(self.config.stop_command, self.config.stop_timeout)
            except subprocess.TimeoutExpired:
                _log(f"停止命令超过 {self.config.stop_timeout:g}s，未完成；请手动检查服务进程")
                return
            except (OSError, subprocess.SubprocessError) as exc:
                _log(f"停止命令执行失败：{exc!r}")
                return
            if completed.returncode != 0:
                _log(f"停止命令返回退出码 {completed.returncode}，请手动检查服务进程")
            else:
                _log("本地模型服务已关闭")
        finally:
            # 停止命令跑完后再恢复原 SIGTERM handler；提前恢复会让第二个 SIGTERM
            # 在服务真正停掉之前就按默认行为终止进程。
            self._restore_sigterm_handler()

    def __enter__(self) -> "LocalModelService":
        return self.ensure_started()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()


def from_model_key(model_key: str) -> LocalModelService | None:
    """按 api_keys.json 里的配置名构造服务管理器；远程模型返回 None。

    延迟 import agents：只有真的需要读模型配置时才加载 langchain，
    提示语为空、LLM 筛选整体跳过时不会多付这份启动成本。
    """
    import agents as agent_factory

    model_cfg = agent_factory.load_model_config(model_key)
    service_config = LocalServiceConfig.from_model_config(model_cfg, model_key)
    return LocalModelService(service_config) if service_config is not None else None


@contextmanager
def for_model(model_key: str, *, enabled: bool = True):
    """按模型配置自动管理本地服务；远程模型或 enabled=False 时是空上下文。

    用法：
        with local_model_service.for_model(screen_model, enabled=screen_llm):
            ...  # 这里可以安全地建模型、发筛选请求

    服务只有在本次程序负责启动时才会在退出上下文时关闭。
    """
    if not enabled:
        yield None
        return
    service = from_model_key(model_key)
    if service is None:
        yield None
        return
    with service:
        yield service
