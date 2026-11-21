# encoding: utf8

"""同步路径的成功节点结果缓存（可选启用，不伪装成原生 Batch）。

为什么单独一层：
    原生 Batch 已有 ``BatchSession`` 的结果 sidecar、受理账与续跑身份；纯同步链路
    目前只有进程内 ``BaseAgent._verdicts``，进程退出或报告重跑后，已经成功的节点
    结果全部丢失，下一次会重新付费。本模块只保存**已通过现有业务校验的成功结果**，
    让中断后能按节点续跑；失败、空结果与未完成请求一律不写入。

边界：
    - 键由调用方用 ``batch_inference.agent_item_key`` 生成（agent 名 + system prompt
      + 输出 schema 指纹 + 完整 user message）；模型身份、生成参数、提示词与配置表
      指纹放在 sidecar 的身份文件里。任一变化都拒绝复用旧结果。
    - 不是 exactly-once：响应成功但尚未落盘时进程崩溃，仍可能重复付费。
    - 不与原生 Batch 的 ``.batch-cache.jsonl`` 混用；两者可以同时存在，但各自校验。
    - 文件只存业务结果与脱敏后的身份结构；保留非敏感配置细节便于排查续跑，
      但 api_key、认证头、URL 账号与查询参数一律不落盘。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import tempfile
import threading
import time
from pathlib import Path

from file_paths import atomic_write_text

if os.name != "nt":
    import fcntl

SCHEMA_VERSION = 2
"""sidecar 语义版本；2 起不再把 Batch 结果写入同步缓存，并收紧身份/行边界口径。"""


def sidecar_paths(path: str | Path) -> tuple[Path, Path, Path]:
    """同步缓存会写出的结果、身份与锁三件套。"""
    target = Path(path)
    return (
        target,
        target.with_suffix(target.suffix + ".meta.json"),
        target.with_suffix(target.suffix + ".lock"),
    )


def validate_sidecar_paths(path: str | Path) -> tuple[Path, Path, Path]:
    """缓存、身份和锁都必须是文件路径，且在创建任何 sidecar 前检查。"""
    paths = sidecar_paths(path)
    for target in paths:
        if target.exists() and not target.is_file():
            raise ValueError(f"同步缓存路径不是文件：{target}")
    return paths


def _sort_key(value) -> str:
    return json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    )


def _canonical(value):
    """把任意配置值折成稳定可哈希的 JSON 形状。

    判定表的 `cells` 用 tuple 作键（如 ``("里1", "外1")``），``json.dumps`` 的
    ``default`` 只处理值、不处理字典键；直接序列化会在默认 pair 图上抛 TypeError。
    这里把非字符串键的映射改成有序 ``[[key, value], ...]``，集合同样排序，避免
    ``PYTHONHASHSEED`` 让身份摘要跨进程漂移。
    """
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _canonical(item) for key, item in value.items()}
        pairs = [[_canonical(key), _canonical(item)] for key, item in value.items()]
        pairs.sort(key=lambda pair: _sort_key(pair[0]))
        return pairs
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=_sort_key)
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _digest(value) -> str:
    text = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SENSITIVE_KEYWORDS = frozenset({
    "api_key", "apikey", "api_secret", "api_token", "secret", "password", "passwd",
    "passphrase", "token", "access_token", "accesstoken", "refresh_token",
    "refreshtoken", "auth", "auth_token", "authtoken", "id_token",
    "authorization", "authentication", "credential", "credentials",
    "private_key", "privatekey", "client_secret",
    "clientsecret", "access_key", "accesskey", "secret_key", "secretkey",
})
"""身份摘要里必须脱敏的键名（按规范化后整词匹配）。"""

_URL_USERINFO = re.compile(r"(?i)((?:https?|socks5h?)://)[^/@\s'\"]+@")
_URL_QUERY = re.compile(r"(?i)((?:https?|socks5h?)://[^\s'\"<>?]+)\?[^\s'\"<>]*")
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


def _normalized_key(key: object) -> str:
    # 拆 camelCase 与大写缩写（accessToken -> access_Token，IDToken -> ID_Token），
    # 再统一成下划线小写。
    text = str(key)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_sensitive_key(key: object, value=None) -> bool:
    normalized = _normalized_key(key)
    if normalized == "max_token" and type(value) is int and value >= 0:
        return False  # 数字 token 上限不是凭据；同名的不透明字符串仍遮盖
    if normalized in _SENSITIVE_KEYWORDS:
        return True
    parts = normalized.split("_")
    return (
        "authorization" in parts
        or "authentication" in parts
        or normalized.endswith((
            "_api_key", "_apikey", "_token", "_secret", "_password",
            "_credential", "_private_key", "_access_key",
        ))
    )


def redact_url(value: object) -> str:
    """去掉 URL 里的账号、密码与查询参数；不解析普通文本。"""
    text = str(value or "")
    parsed = text
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(text)
        if parts.scheme and parts.hostname:
            netloc = parts.hostname
            try:
                if parts.port:
                    netloc = f"{netloc}:{parts.port}"
            except ValueError:
                pass
            parsed = urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except ValueError:
        parsed = text
    parsed = _URL_USERINFO.sub(r"\1***@", parsed)
    parsed = _URL_QUERY.sub(r"\1?***", parsed)
    parsed = _BEARER.sub("Bearer ***", parsed)
    return _SK_KEY.sub("sk-***", parsed)


def redact(value, *, key: object | None = None):
    """把任意配置值折成可落盘的详细身份：保留结构与所有非敏感值。

    敏感键（api_key / token / authentication / password 等）的值统一替换为
    ``"<redacted>"``，因此换密钥不会让旧同步结果失效；URL 只保留协议、主机、
    端口与路径。返回的结构仍可逐字段比较，便于续跑时判断非敏感配置是否变化。
    """
    if key is not None and _is_sensitive_key(key, value):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(item_key): redact(item, key=item_key) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((redact(item) for item in value), key=lambda item: json.dumps(
            item, ensure_ascii=False, sort_keys=True, default=str))
    if isinstance(value, str):
        return redact_url(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)


def redact_config(cfg: dict) -> dict:
    """同步配置的详细脱敏视图；不保存密钥、Batch 或本地服务启停配置。"""
    return {
        str(key): redact(value, key=key)
        for key, value in (cfg or {}).items()
        if _normalized_key(key) not in {"api_key", "apikey", "batch", "local_service"}
    }


def _all_strings(value) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_all_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_all_strings(item))
    return found


_LOCAL_SERVICE_COMMANDS = ("start_command", "stop_command", "status_command")


def _command_secrets(command) -> list[str]:
    """只收集命令中显式标记的凭据，不执行命令或遮盖普通参数。"""
    if isinstance(command, str):
        try:
            args = shlex.split(command)
        except ValueError:
            return []
    elif isinstance(command, (list, tuple)):
        args = [str(item) for item in command]
    else:
        return []

    found = []
    for index, arg in enumerate(args):
        if arg == "--":
            break
        name, separator, value = arg.partition("=")
        if not _is_sensitive_key(name):
            continue
        if separator and (name.startswith("--") or name.isidentifier()):
            found.append(value)
        elif name.startswith("--") and index + 1 < len(args) and not args[index + 1].startswith("--"):
            found.append(args[index + 1])
    return found


def sensitive_values(value, key: object | None = None) -> list[str]:
    """收集配置中的已知秘密，包括本地服务启停命令的显式凭据参数。"""
    found: list[str] = []
    if key is not None and _is_sensitive_key(key, value):
        # 命中敏感父键后，整个子树都视为凭据，不能再按子键重新判断。
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return [str(value)]
        return _all_strings(value)
    if isinstance(value, dict):
        if key == "local_service":
            for field in _LOCAL_SERVICE_COMMANDS:
                found.extend(_command_secrets(value.get(field)))
        for item_key, item in value.items():
            found.extend(sensitive_values(item, item_key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(sensitive_values(item))
    return found


def redact_text(text: object, secrets=()) -> str:
    """脱敏任意文本：已知秘密、URL 账号/查询参数、Bearer 与 sk- 形式。"""
    result = str(text or "")
    for secret in sorted(
        {str(item) for item in secrets if len(str(item)) >= 4},
        key=len,
        reverse=True,
    ):
        result = result.replace(secret, "***")
    result = _URL_USERINFO.sub(r"\1***@", result)
    result = _URL_QUERY.sub(r"\1?***", result)
    result = _BEARER.sub("Bearer ***", result)
    return _SK_KEY.sub("sk-***", result)


def sanitize_result(value, secrets=()):
    """递归脱敏业务结果里的字符串；键名与非字符串值保持不变。"""
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, dict):
        return {key: sanitize_result(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_result(item, secrets) for item in value]
    return value


def model_fingerprint(model_key: str, cfg: dict, *, temperature=None, generation=None) -> str:
    """同步 sidecar 的模型身份摘要：不落盘 base_url/密钥原文。"""
    return _digest({
        "model_key": model_key,
        "model": cfg.get("model") or cfg.get("model_name"),
        "provider": cfg.get("model_provider") or cfg.get("provider"),
        "base_url": _digest(cfg.get("base_url") or ""),
        "temperature": temperature,
        "generation": generation,
        "prompt_cache": cfg.get("prompt_cache"),
    })


def materials_fingerprint(agents) -> str:
    """把本次 agent 的提示语、输出 schema 与配置表折成一个不可逆摘要。

    身份文件不能存完整配置（可能很大，也可能夹带端点信息）；这里只存摘要。
    表内容变化、system prompt 变化、输出 schema 变化都会让旧同步缓存失效。
    """
    items: dict[str, dict] = {}
    for name, agent in agents.items():
        prompt_paths = getattr(agent, "prompt_paths", {}) or {}
        prompts = getattr(agent, "prompts", {}) or {}
        tables = {}
        for attr in (
            "table", "schema_table", "judge_table", "feature_classes", "feature_keys",
        ):
            if hasattr(agent, attr):
                value = getattr(agent, attr)
                if isinstance(value, (set, frozenset)):
                    value = sorted(value, key=str)
                tables[attr] = _digest(value)
        try:
            tools = agent.tools()
        except Exception:  # noqa: BLE001 —— 工具枚举失败时宁可让缓存更保守地失效
            tools = ()
        try:
            schema = agent.schema_key()
        except Exception:  # noqa: BLE001 —— 纯规则 agent（如 AlterAgent）没有输出 schema
            schema = ""
        items[name] = {
            "class": f"{type(agent).__module__}.{type(agent).__qualname__}",
            "system_prompt": _digest(getattr(agent, "system_prompt", "")),
            "schema": _digest(schema),
            "tables": tables,
            "tools": _digest([
                {
                    "name": getattr(tool, "name", type(tool).__name__),
                    "description": getattr(tool, "description", ""),
                }
                for tool in tools
            ]),
            "prompt_paths": {key: str(value) for key, value in prompt_paths.items()},
            "prompts": {key: _digest(value) for key, value in prompts.items()},
        }
    return _digest(items)


class SyncResultStore:
    """一个模型/一次实验独立的同步成功结果 sidecar。

    目录与文件锁口径沿用 Batch sidecar：同一路径同时只允许一个进程持有；进程退出
    时操作系统自动释放。``resume=False`` 只清空**可识别为本工具缓存**的旧结果并
    重写身份；无身份的非空文件会拒绝，避免 --sync-cache 误指配置表或语料。
    """

    def __init__(
        self, path: str | Path, identity: dict, *, resume: bool = False,
        logger=None, sanitize=None,
    ):
        self.path, self.meta_path, self._file_lock = validate_sidecar_paths(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = copy.deepcopy(identity)
        self.resume = bool(resume)
        self.logger = logger or (lambda message: None)
        self.sanitize = sanitize
        self.hits = 0
        self.writes = 0
        self.invalid_lines = 0
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._file_handle = None
        self._acquire_file_lock()
        try:
            self._prepare()
            self._load()
        except BaseException:
            self.close()
            raise

    # -- 进程锁 ------------------------------------------------------------
    def _acquire_file_lock(self) -> None:
        self._file_lock.parent.mkdir(parents=True, exist_ok=True)
        handle = self._file_lock.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise ValueError(
                f"同步缓存 {self.path} 正被另一个进程使用；请等它结束后重试，或换一个 --sync-cache"
            ) from None
        self._file_handle = handle

    def close(self) -> None:
        handle, self._file_handle = self._file_handle, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- 身份与加载 --------------------------------------------------------
    def _write_meta(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": time.time(),
            "identity": self.identity,
        }
        fd, temp_name = tempfile.mkstemp(
            dir=self.meta_path.parent, prefix=f".{self.meta_path.name}.", suffix=".tmp"
        )
        os.close(fd)
        temp = Path(temp_name)
        try:
            temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.meta_path)
        finally:
            temp.unlink(missing_ok=True)

    def _read_meta(self) -> dict | None:
        try:
            actual = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return actual if isinstance(actual, dict) else None

    def _prepare(self) -> None:
        if self.resume:
            actual = self._read_meta()
            if actual is None:
                raise ValueError(
                    f"同步缓存 {self.path} 缺少有效身份文件；拒绝旧版或损坏缓存续跑"
                )
            if actual.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"同步缓存 {self.path} 身份版本不匹配，拒绝续跑")
            if actual.get("identity") != self.identity:
                raise ValueError(
                    f"同步缓存 {self.path} 的模型/提示词/配置身份与本次不一致，拒绝续跑；"
                    "请换一个 --sync-cache，或删掉旧缓存后重跑"
                )
            return
        # fresh 只允许清空本工具可识别的旧缓存。没有 meta 的非空文件可能是用户
        # 误指过来的配置表/语料/任意文件，必须先拒绝，绝不能先截断再报错。
        if self.path.exists() and self.path.stat().st_size > 0:
            actual = self._read_meta()
            owned = (
                actual is not None
                and isinstance(actual.get("schema_version"), int)
                and isinstance(actual.get("identity"), dict)
            )
            if not owned:
                raise ValueError(
                    f"同步缓存 {self.path} 已存在内容，但没有可识别的同步缓存身份；"
                    "拒绝清空。请换一个 --sync-cache 路径，或确认后手工删除该文件"
                )
        self.path.write_text("", encoding="utf-8")
        self._write_meta()

    def _load(self) -> None:
        if not self.resume or not self.path.exists():
            return
        raw = self.path.read_bytes()
        truncated_tail = False
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            # 进程可能在追加一行中文结果时崩在半个多字节字符上。只有错误发生在文件
            # 末尾附近才按「不完整尾行」处理；中间损坏仍然拒绝续跑。
            if exc.start < len(raw) - 4:
                raise ValueError(f"同步缓存 {self.path} 不是完整 UTF-8，拒绝续跑") from None
            text = raw[:exc.start].decode("utf-8")
            truncated_tail = True
        # 只按物理换行切分：JSON 字符串里允许出现 U+2028/U+2029，splitlines()
        # 会把它们当成记录边界，从而把一条合法记录误判成损坏。
        lines = text.split("\n")
        last_nonempty = max(
            (index for index, line in enumerate(lines) if line.strip()), default=-1
        )
        valid_lines: list[str] = []
        for index, line in enumerate(lines):
            text = line.strip()
            if not text:
                continue
            try:
                entry = json.loads(text)
            except json.JSONDecodeError:
                if index == last_nonempty:
                    # 进程在追加一行写到一半时崩溃：最后一截不完整，丢掉并截断文件，
                    # 否则下一次 append 会把新行接在这截后面，永久损坏整个 sidecar。
                    self.invalid_lines += 1
                    truncated_tail = True
                    continue
                raise ValueError(f"同步缓存 {self.path} 第 {index + 1} 行损坏，拒绝续跑")
            if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
                self.invalid_lines += 1
                continue
            self._entries[entry["key"]] = {"dumps": entry.get("dumps")}
            valid_lines.append(text)
        if truncated_tail:
            # 保持可用的原文件，直到修复稿完整写好；锁文件不会随数据文件替换。
            with atomic_write_text(self.path.resolve()) as handle:
                for text in valid_lines:
                    handle.write(text + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        elif raw and not raw.endswith(b"\n"):
            # 完整末行只补一个 LF；二进制追加也能处理 Windows 中断留下的孤立 CR。
            with self.path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    # -- 读写 --------------------------------------------------------------
    def lookup(self, key: str) -> dict | None:
        if not key:
            return None
        with self._lock:
            hit = self._entries.get(key)
            if hit is None or not isinstance(hit.get("dumps"), dict):
                return None
            self.hits += 1
            return copy.deepcopy(hit["dumps"])

    def put(self, key: str, dumps: dict) -> None:
        if not key or not isinstance(dumps, dict):
            return
        if self.sanitize is not None:
            dumps = self.sanitize(copy.deepcopy(dumps))
        with self._lock:
            if key in self._entries:
                return
            self._entries[key] = {"dumps": copy.deepcopy(dumps)}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "dumps": dumps}, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.writes += 1

    def summary(self) -> dict:
        return {
            "enabled": True,
            "path": str(self.path),
            "resume": self.resume,
            "entries": len(self._entries),
            "hits": self.hits,
            "writes": self.writes,
            "invalid_lines": self.invalid_lines,
        }
