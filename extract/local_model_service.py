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

并发：
    多个 extract/replace 进程可能同时使用同一个本地服务。所有客户端会用共享
    `flock` 登记“仍在使用”，启动决策再用独占锁串行化；启动者在退出时先释放自己的
    共享锁，只有能拿到独占锁（说明没有其他客户端）才立即 stop。若还有其他客户端，
    则拉起一个 detached reaper 阻塞在独占锁上，等最后一个客户端退出后再 stop。
    这样后启动的任务不会因为前一个任务退出而被中断，服务也不会永远留在后台；
    两个任务同时发现服务未启动时也只会拉起一份。stop 前还会用启动回执里的 PID
    核对服务代际，服务被人工/其他实例替换后不会误停新进程。
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import ProxyHandler, Request, build_opener

try:  # 本地服务生命周期只支持 POSIX；Windows 在 ensure_started 里直接报错。
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

import config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_PREFIX = "[本地模型服务]"
_MISSING = object()
_START_RECEIPT_ENV = "LOCAL_MODEL_SERVICE_START_RECEIPT"
"""service.sh 用这个环境变量把“本次 start 是否真的拉起进程 + PID”写回临时文件。"""
_STOP_REAPER = r"""
import fcntl
import subprocess
import sys

path, cwd, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
pid_file, expected_pid = sys.argv[4], sys.argv[5]
command = sys.argv[6:]
lock = open(path, "a+")
try:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    if expected_pid and pid_file:
        try:
            current_pid = open(pid_file, encoding="utf-8").read().strip()
        except OSError:
            current_pid = ""
        if current_pid != expected_pid:
            sys.exit(0)
    subprocess.run(command, cwd=cwd, timeout=timeout, check=False)
finally:
    lock.close()
"""
"""最后一个客户端退出后执行 stop 的 detached 进程。

启动者在退出时若发现还有别的 extract/replace 进程在用服务，就把“关闭”交给这个
进程：它阻塞在客户端独占锁上，最后一个共享锁释放后拿到锁并执行 stop。这样既不会
中断并发任务，也不会让服务永远留在后台。
"""
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


def _resolve_pid_file(value: Any, service_dir: Path) -> Path | None:
    """PID 文件默认取 <service_dir>/run/server.pid；显式 null 表示没有 PID 文件。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return service_dir / "run" / "server.pid"
    path = Path(os.path.expanduser(text))
    if not path.is_absolute():
        path = service_dir / path
    return path


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
    pid_file: Path | None
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

        service_dir = _resolve_service_dir(raw.get("service_dir", config.LOCAL_MODEL_SERVICE_DIR))
        pid_file = _resolve_pid_file(
            raw.get("pid_file", config.LOCAL_MODEL_SERVICE_PID_FILE), service_dir
        )

        return cls(
            model_name=model_name,
            base_url=model_base_url,
            service_dir=service_dir,
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
            pid_file=pid_file,
            extra_env={
                **{str(k): str(v) for k, v in config.LOCAL_MODEL_SERVICE_EXTRA_ENV.items()},
                **_env_mapping(raw.get("extra_env"), "extra_env"),
            },
        )


class LocalModelService:
    """管理一个本地 OpenAI 兼容服务的启动、健康检查与关闭。

    只管理“本实例启动的进程”：启动前 `/health` 已通过，或状态命令显示进程已存在时，
    `owned` 保持 False，退出时不执行 stop，避免误关用户手动启动的服务。

    并发场景下，`owned=True` 的实例退出前还会尝试独占客户端锁；如果其他 extract/replace
    进程仍在用同一服务，则不立即 stop，而是安排 reaper 在最后一个客户端退出后关闭，
    避免一个任务把另一个任务的模型连接停掉，也避免服务长期占用显存。
    """

    def __init__(self, service_config: LocalServiceConfig):
        self.config = service_config
        self._started = False
        self._owned = False
        self._previous_sigterm: Any = _MISSING
        self._client_lock: Any = None
        self._start_lock: Any = None
        self._lock_base: Path | None = None
        self._service_pid: str | None = None
        self._cleanup_registered = False
        self._start_in_progress = False
        self._pid_pending = False

    @property
    def owned(self) -> bool:
        """本次程序是否负责关闭该服务。"""
        return self._owned

    # ---- 跨进程生命周期锁 ----
    # 锁文件放在系统临时目录，按 service_dir + base_url 区分服务，不往被管理的
    # 服务目录里写额外状态。取不到锁时退回旧行为（只打日志），不让锁故障阻断跑批。
    def _lock_path(self, kind: str) -> Path:
        if self._lock_base is None:
            uid = os.getuid() if hasattr(os, "getuid") else 0
            base = Path(tempfile.gettempdir()) / f"direction-verb-local-model-{uid}"
            base.mkdir(mode=0o700, parents=True, exist_ok=True)
            digest = hashlib.sha256(
                f"{self.config.service_dir}\0{self.config.base_url}".encode("utf-8")
            ).hexdigest()[:16]
            self._lock_base = base / digest
        return self._lock_base.with_suffix(f".{kind}.lock")

    def _lock_file(self, kind: str) -> Any:
        try:
            path = self._lock_path(kind)
            return path.open("a+")
        except OSError as exc:
            _log(f"无法创建生命周期锁（{kind}）：{exc}；并发保护已降级")
            return None

    def _acquire_client_lock(self, *, exclusive: bool = False, blocking: bool = True) -> bool:
        """登记本进程是服务客户端；独占模式用于确认没有其他客户端。"""
        if fcntl is None:
            return True
        stream = self._lock_file("client")
        if stream is None:
            return True
        flags = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(stream.fileno(), flags)
        except BlockingIOError:
            stream.close()
            return False
        except OSError as exc:
            stream.close()
            _log(f"客户端锁获取失败：{exc}；按未加锁处理")
            return True
        self._client_lock = stream
        return True

    def _release_client_lock(self) -> None:
        stream = self._client_lock
        self._client_lock = None
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except (OSError, AttributeError):
            pass
        stream.close()

    def _acquire_start_lock(self) -> None:
        """启动决策串行化：避免两个进程同时 status=3 各拉起一份 server。"""
        if fcntl is None:
            return
        stream = self._lock_file("start")
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            stream.close()
            _log(f"启动锁获取失败：{exc}；按未加锁处理")
            return
        self._start_lock = stream

    def _release_start_lock(self) -> None:
        stream = self._start_lock
        self._start_lock = None
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except (OSError, AttributeError):
            pass
        stream.close()

    # ---- atexit/SIGTERM 登记（幂等，避免重复 unregister）----
    def _register_cleanup(self) -> None:
        if self._cleanup_registered:
            return
        atexit.register(self.stop)
        self._install_sigterm_handler()
        self._cleanup_registered = True

    def _unregister_cleanup(self) -> None:
        if not self._cleanup_registered:
            return
        try:
            atexit.unregister(self.stop)
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._cleanup_registered = False
            self._restore_sigterm_handler()

    # ---- 服务代际（PID）与启动回执 ----
    def _receipt_path(self) -> Path:
        self._lock_path("start")  # 确保 _lock_base 已初始化
        assert self._lock_base is not None
        return self._lock_base.with_suffix(".start-receipt")

    def _read_pid_file(self) -> str | None:
        path = self.config.pid_file
        if path is None:
            return None
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def _read_start_receipt(self, path: Path) -> tuple[bool | None, str | None]:
        """返回 (本次调用是否真的启动了服务, 启动后的 PID)；无回执时状态为 None。"""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None, None
        if not lines:
            return None, None
        state = lines[0].strip()
        pid = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None
        if state == "started":
            return True, pid
        if state == "reused":
            return False, pid
        return None, pid

    def _apply_start_receipt(self, started_by_call: bool | None, pid: str | None) -> None:
        """按回执修正所有权；没有回执时沿用 status 命令的判断。"""
        if pid:
            self._service_pid = pid
            self._pid_pending = False
        elif self._service_pid is None:
            self._service_pid = self._read_pid_file()

        if started_by_call is True:
            self._owned = True
            # service.sh 在 spawn 前就写了 started 回执，PID 可能稍后才补；stop 时
            # 需要短暂等 PID 文件出现，才能停掉刚拉起的服务。
            self._pid_pending = self._service_pid is None and self.config.pid_file is not None
        elif started_by_call is False:
            self._owned = False
            self._pid_pending = False
            self._unregister_cleanup()

    def _service_pid_changed(self) -> bool:
        """要停的服务是否已被其他实例替换/移除。"""
        expected = self._service_pid
        if not expected or self.config.pid_file is None:
            return False
        return self._read_pid_file() != expected

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

    def _service_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # service.sh 用自己的 STARTUP_TIMEOUT 控制等待窗口；配置是唯一默认来源，
        # 但 local_service.extra_env 仍可显式覆盖。
        env["STARTUP_TIMEOUT"] = str(int(self.config.start_timeout))
        env.update(self.config.extra_env)
        return _with_local_proxy_bypass(env)

    def _run(
        self,
        command: tuple[str, ...],
        timeout: float,
        *,
        capture_output: bool = False,
        extra_env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        env = self._service_env()
        if extra_env:
            env.update({str(key): str(value) for key, value in extra_env.items()})
        return subprocess.run(
            list(command),
            cwd=str(self.config.service_dir),
            env=env,
            timeout=timeout,
            check=False,
            text=True,
            capture_output=capture_output,
        )

    def _spawn_stop_reaper(self) -> bool:
        """安排最后一个客户端退出后关闭服务；返回是否成功拉起 reaper。"""
        if fcntl is None:
            return False
        env = self._service_env()
        if self._service_pid:
            # reaper 执行 stop_command 时，service.sh 在它自己的锁内再核对一次 PID。
            env["LOCAL_MODEL_SERVICE_EXPECT_PID"] = self._service_pid
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _STOP_REAPER,
                    str(self._lock_path("client")),
                    str(self.config.service_dir),
                    str(self.config.stop_timeout),
                    str(self.config.pid_file or ""),
                    self._service_pid or "",
                    *self.config.stop_command,
                ],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _log(f"无法安排延迟关闭进程：{exc!r}；请手动检查服务")
            return False
        return True

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
                    if self._start_in_progress:
                        # 启动命令还没返回、回执/所有权未定；此时直接 stop 会按旧
                        # status 误停人工启动的服务。让异常路径先读回执，再决定是否
                        # stop（subprocess.run 会同时终止启动命令子进程）。
                        raise SystemExit(128 + signum)
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

        # 先登记客户端，再决定是否启动：启动者退出时靠这把共享锁判断是否还有别人在用。
        self._acquire_client_lock()
        try:
            self._acquire_start_lock()
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
                # 即使 status 暂时判定为非 owner，也先登记；回执确认是复用别人的服务后
                # 再撤销，避免处置过程中出现无人负责的窗口。
                self._register_cleanup()

                # service.sh 会写回执：started=本次真的拉起了进程，reused=已有进程。
                # 回执缺失时（自定义命令）沿用 status 命令的所有权判断。
                receipt = self._receipt_path()
                receipt.unlink(missing_ok=True)
                self._start_in_progress = True
                try:
                    try:
                        # 给 service.sh 自己的 STARTUP_TIMEOUT 留一点收尾余量：否则外层先超时、
                        # 内层还以为在等待，stop 会被迫在启动子进程未退出时介入。
                        completed = self._run(
                            self.config.start_command,
                            self.config.start_timeout + 30,
                            extra_env={_START_RECEIPT_ENV: str(receipt)},
                        )
                    finally:
                        started_by_call, receipt_pid = self._read_start_receipt(receipt)
                        receipt.unlink(missing_ok=True)
                        self._apply_start_receipt(started_by_call, receipt_pid)
                finally:
                    self._start_in_progress = False

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
                if not self._owned:
                    # 复用别人的服务：启动期间临时登记的清理钩子不再需要。
                    self._unregister_cleanup()
            except BaseException:
                # 本次已经开始的启动失败时，尽量把可能残留的进程停掉，不把脏状态留给下一轮。
                if self._owned:
                    self.stop()
                else:
                    self._started = False
                    self._unregister_cleanup()
                    self._release_client_lock()
                raise

            if self._owned:
                _log(f"服务已启动并就绪：{self.config.health_url}")
            else:
                _log(f"服务已由其他进程启动并就绪：{self.config.health_url}")
            return self
        except BaseException:
            if not self._started:
                self._release_client_lock()
            raise
        finally:
            self._release_start_lock()

    def stop(self) -> None:
        """关闭本次程序启动的服务；重复调用安全，非本程序启动的服务不处理。"""
        if not self._started:
            self._release_client_lock()
            return
        self._started = False
        if not self._owned:
            self._owned = False
            self._unregister_cleanup()
            self._release_client_lock()
            return
        self._owned = False
        if self._cleanup_registered:
            try:
                atexit.unregister(self.stop)
            except Exception:  # noqa: BLE001 —— 解释器退出阶段 unregister 失败不应阻断清理
                pass
            self._cleanup_registered = False

        # 启动命令如果在写回执 PID 之前被强杀，PID 文件可能由子 shell 稍后补写。
        # 在释放客户端锁之前先等它出现：这样无论下面走直接 stop 还是 reaper，
        # 都带着同一个 expected PID，不会在“PID 待定”时生成无代际保护的 reaper。
        if self._pid_pending and self._service_pid is None and self.config.pid_file is not None:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                pid = self._read_pid_file()
                if pid:
                    self._service_pid = pid
                    self._pid_pending = False
                    break
                time.sleep(0.05)
        pid_unknown = self._pid_pending and self._service_pid is None

        # 先释放自己的共享锁，再尝试独占：能拿到独占说明没有其他任务还在用。
        # 拿不到时保留服务，并安排一个 detached reaper 阻塞在独占锁上；最后一个
        # 客户端退出后由它执行 stop，避免中断并发任务，也避免服务永远留在后台。
        self._release_client_lock()
        if not self._acquire_client_lock(exclusive=True, blocking=False):
            if pid_unknown:
                # 连 PID 都还没出现时不安排 reaper，避免它以后无代际核对地误停新服务。
                _log("启动回执缺少 PID，且 PID 文件未出现；为避免误停，暂不安排延迟关闭")
            elif self._spawn_stop_reaper():
                _log("检测到其他任务仍在使用本地模型服务；已安排最后一个任务退出后关闭")
            else:
                _log("检测到其他任务仍在使用本地模型服务，保留服务不关闭")
            self._restore_sigterm_handler()
            return

        if pid_unknown:
            _log("启动回执缺少 PID，且 PID 文件未出现；为避免误停，跳过 stop")
            self._release_client_lock()
            self._restore_sigterm_handler()
            return

        # 独占锁只说明没有“本模块的客户端”，不能防止服务被人工重启。用启动回执里的
        # PID 再核对一次：对象变了就不 stop，避免误停后来启动的新服务。
        if self._service_pid_changed():
            _log("本地模型服务已被其他实例替换或停止，跳过 stop")
            self._release_client_lock()
            self._restore_sigterm_handler()
            return

        _log(f"关闭本地模型服务：{' '.join(self.config.stop_command)}")
        try:
            try:
                completed = self._run(
                    self.config.stop_command,
                    self.config.stop_timeout,
                    extra_env=(
                        {"LOCAL_MODEL_SERVICE_EXPECT_PID": self._service_pid}
                        if self._service_pid
                        else None
                    ),
                )
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
            self._release_client_lock()
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

    服务只有在本次程序负责启动、且退出时没有其他任务仍在使用时才会关闭。
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
