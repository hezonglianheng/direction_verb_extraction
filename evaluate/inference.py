# encoding: utf8

"""评测推理层：提示词渲染、模型调用、原生 Batch 与同步回退。

本模块只做「把一条句对送出去、把结构化结论收回来」；输入筛选在 data.py，
报告与指标在 runner.py / metrics.py。模型构造、超时、重试、代理与思考模式
全部经 ``agents.get_agent``，不在评测组件里另造一套配置。

同步与 Batch 使用同一份 system/user message 和同一个 ``EvaluationVerdict``
结构。BatchSession 在这里始终以 strict_native=True 运行：batch 一旦受理，
失败/缺失项只记错误，绝不在评测层同步重发；auto 的同步回退只发生在
「一个 batch 都还没受理」时。结果缺失会如实记成失败，留给
``--batch-resume`` 继续收。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

import file_paths
import retry
import utils
from evaluate.data import LABEL_DIFFERENT, LABEL_SAME, EvalSample

REQUEST_AGENT = "evaluate_v1"
"""请求缓存键里的 agent 名；改结构化输出 schema 时同步改版本号。"""

SYSTEM_PROMPT_FILE = "system.txt"
USER_PROMPT_FILE = "user.txt"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_DIR = _PROJECT_ROOT / "system_prompt" / "evaluate"

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_SYSTEM_PLACEHOLDERS = frozenset()
_USER_PLACEHOLDERS = frozenset(
    {
        "sentence_a",
        "sentence_b",
        "focus_word",
        "focus_word_b",
        "focus_word_a",
        "focus_word_text",
        "focus_word_b_text",
        "record_id",
        "line_no",
        "input_line_no",
    }
)
_REQUIRED_USER_PLACEHOLDERS = frozenset({"sentence_a", "sentence_b"})


class EvaluationVerdict(BaseModel):
    """待测模型对一条句对的同义判断。"""

    same_meaning: bool = Field(description="两句话是否表达相同的意义")
    reason: str = Field(default="", description="判断依据，简要说明语义差别")


@dataclass(frozen=True)
class Prompts:
    """已校验的两份提示词模板及其路径/哈希。"""

    system: str
    user: str
    system_path: str
    user_path: str
    system_sha256: str
    user_sha256: str


@dataclass(frozen=True)
class ModelSpec:
    """一个待测模型的配置快照（含密钥，只在本进程内使用，绝不进报告）。"""

    key: str
    cfg: dict
    policy: dict
    proxy_mode: str
    batch: Any | None = None


@dataclass(frozen=True)
class RequestSpec:
    key: str
    user_message: str


@dataclass
class RequestPlan:
    requests: dict[str, RequestSpec]
    sample_to_request: dict[str, str]


@dataclass
class ModelOutcome:
    plan: RequestPlan
    predictions: dict[str, dict]
    unique_requests: int = 0
    sync_retries: int = 0
    batch: dict | None = None
    batch_error: str | None = None
    elapsed_seconds: float = 0.0


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompts(prompt_dir: str | Path | None = None) -> Prompts:
    """读取并校验 system_prompt/evaluate/ 下的系统/用户提示词。

    系统提示词必须与样本无关（不接受占位符）；随样本变化的内容放在 user.txt。
    用户提示词必须包含 ``{{sentence_a}}`` 与 ``{{sentence_b}}``，否则模型看不到句对。
    """
    directory = Path(prompt_dir) if prompt_dir else DEFAULT_PROMPT_DIR
    system_path = directory / SYSTEM_PROMPT_FILE
    user_path = directory / USER_PROMPT_FILE
    system = utils.read_prompt_path(system_path)
    user = utils.read_prompt_path(user_path)

    if not system.strip():
        raise ValueError(f"系统提示词不能为空：{system_path}；请填写 {SYSTEM_PROMPT_FILE} 的正文")
    if not user.strip():
        raise ValueError(f"用户提示词不能为空：{user_path}；请填写 {USER_PROMPT_FILE} 的正文")

    system_placeholders = set(_PLACEHOLDER.findall(system))
    if system_placeholders:
        raise ValueError(
            f"系统提示词 {system_path} 不能包含占位符 "
            f"{'、'.join(sorted(system_placeholders))}；随样本变化的内容请放进 {USER_PROMPT_FILE}"
        )

    user_placeholders = set(_PLACEHOLDER.findall(user))
    unknown = user_placeholders - _USER_PLACEHOLDERS
    if unknown:
        raise ValueError(
            f"用户提示词 {user_path} 有未知占位符：{'、'.join(sorted(unknown))}；"
            f"可用：{'、'.join(sorted(_USER_PLACEHOLDERS))}"
        )
    missing = _REQUIRED_USER_PLACEHOLDERS - user_placeholders
    if missing:
        raise ValueError(
            f"用户提示词 {user_path} 缺少必需占位符：{'、'.join(sorted(missing))}；"
            "模型必须能在消息里看到原句与改写句"
        )

    return Prompts(
        system=system,
        user=user,
        system_path=str(system_path),
        user_path=str(user_path),
        system_sha256=_sha256(system),
        user_sha256=_sha256(user),
    )


def _focus_json(value: object | None) -> str:
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _focus_text(value: object | None) -> str:
    if isinstance(value, dict):
        word = value.get("word")
        return str(word) if word is not None else ""
    return "" if value is None else str(value)


def render_values(sample: EvalSample) -> dict[str, str]:
    """用户提示词可用的占位符取值；关注词对象以 JSON 文本给出。"""
    focus = _focus_json(sample.focus_word)
    focus_b = _focus_json(sample.focus_word_b)
    return {
        "sentence_a": sample.sentence_a,
        "sentence_b": sample.sentence_b,
        "focus_word": focus,
        "focus_word_b": focus_b,
        "focus_word_a": focus,
        "focus_word_text": _focus_text(sample.focus_word),
        "focus_word_b_text": _focus_text(sample.focus_word_b),
        "record_id": sample.record_id or "",
        "line_no": "" if sample.line_no is None else str(sample.line_no),
        "input_line_no": str(sample.input_line_no),
    }


def render_messages(sample: EvalSample, prompts: Prompts) -> tuple[str, str]:
    """渲染 (system, user) 两条消息；未知占位符在这里直接报错。"""
    values = render_values(sample)
    system = utils.render_prompt(prompts.system, values, source=prompts.system_path)
    user = utils.render_prompt(prompts.user, values, source=prompts.user_path)
    return system, user


def request_key(agent_name: str, content: str, system_prompt: str) -> str:
    """与 batch_inference.item_key 同一算法；同步路径不必因此导入 Batch 模块。"""
    digest = hashlib.sha256(
        f"{agent_name}\0{system_prompt}\0{content}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def prepare_requests(samples: list[EvalSample], prompts: Prompts) -> RequestPlan:
    """把所有样本渲染成请求；相同消息只保留一条，结果再映射回全部样本。"""
    requests: dict[str, RequestSpec] = {}
    sample_to_request: dict[str, str] = {}
    for sample in samples:
        _, user_message = render_messages(sample, prompts)
        key = request_key(REQUEST_AGENT, user_message, prompts.system)
        if key not in requests:
            requests[key] = RequestSpec(key=key, user_message=user_message)
        sample_to_request[sample.key] = key
    return RequestPlan(requests=requests, sample_to_request=sample_to_request)


def build_agent(model_key: str, system_prompt: str):
    """按现有模型配置构造 agent；延迟导入避免 `--help` 加载整套 langchain。"""
    from langchain.agents.structured_output import ToolStrategy

    from agents import get_agent

    return get_agent(
        model_key,
        [],
        system_prompt,
        response_format=ToolStrategy(EvaluationVerdict, handle_errors=False),
    )


def _ok(same_meaning: bool, reason: str = "") -> dict:
    return {
        "same_meaning": bool(same_meaning),
        "label": LABEL_SAME if same_meaning else LABEL_DIFFERENT,
        "reason": reason or "",
        "error": None,
        "error_kind": None,
    }


def _error(kind: str, message: str) -> dict:
    return {
        "same_meaning": None,
        "label": None,
        "reason": "",
        "error": message or kind,
        "error_kind": kind,
    }


def call_once(agent, user_message: str, policy: dict, *, sleep=time.sleep) -> tuple[dict, int]:
    """一次逻辑调用；返回 (预测, 传输层重试次数)。

    传输层重试由 ``retry.call_with_retry`` 统一负责（模型配置里的 max_retries /
    退避参数已经由 request_policy 读入 policy）。结构化输出缺失或校验失败不算
    可重试错误，直接记成该条失败。
    """
    retries = 0

    def on_retry(exc, attempt, delay):
        nonlocal retries
        retries += 1

    try:
        result = retry.call_with_retry(
            lambda: agent.invoke({"messages": [{"role": "user", "content": user_message}]}),
            policy=policy,
            sleep=sleep,
            on_retry=on_retry,
        )
    except Exception as exc:  # noqa: BLE001 —— 单条失败只影响这一条
        return _error("call_failed", f"调用失败：{exc!r}"), retries

    response = result.get("structured_response") if isinstance(result, dict) else None
    if response is None:
        return _error("no_structured_output", "模型没有给出结构化输出（structured_response 为空）"), retries
    try:
        if isinstance(response, EvaluationVerdict):
            verdict = response
        else:
            verdict = EvaluationVerdict.model_validate(response)
    except Exception as exc:  # noqa: BLE001
        return _error("invalid_response", f"结构化输出校验失败：{exc!r}"), retries
    return _ok(bool(verdict.same_meaning), verdict.reason or ""), retries


def sanitize_text(text: str, cfg: dict | None = None) -> str:
    """日志/缓存/报告统一脱敏：API key、Bearer、URL 账号与查询参数。"""
    cfg = cfg or {}
    text = str(text or "")
    secret = str(cfg.get("api_key") or "")
    if len(secret) >= 4:
        text = text.replace(secret, "***")
    proxy = cfg.get("proxy")
    if isinstance(proxy, str) and "://" in proxy:
        text = text.replace(proxy, redact_url(proxy))
    text = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)\S+", r"\1***", text)
    text = re.sub(r"(?i)\bbearer\s+\S+", "Bearer ***", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "sk-***", text)
    # URL 里的账号/密码（含 http/socks5 代理）与查询参数（token/api_key 常在这里）。
    text = re.sub(r"(?i)((?:https?|socks5h?)://)[^/@\s'\"]+@", r"\1***@", text)
    text = re.sub(
        r"(?i)((?:https?|socks5h?)://[^\s'\"<>]+)\?[^\s'\"<>]*", r"\1?***", text
    )
    text = re.sub(r"(?i)\b(api[_-]?key|token|secret)=([^&\s,;]+)", r"\1=***", text)
    return text


def sanitize_exception(exc: BaseException, cfg: dict | None = None) -> str:
    return sanitize_text(f"{type(exc).__name__}: {exc}", cfg)[:2000]


def redact_url(value: object) -> str:
    """去掉 URL 里的账号、密码与查询参数，只保留协议/主机/端口/路径。"""
    text = str(value or "")
    try:
        parts = urlsplit(text)
    except ValueError:
        return "<invalid-url>"
    if not parts.scheme or not parts.hostname:
        return text.split("?", 1)[0]
    netloc = parts.hostname
    try:
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
    except ValueError:
        pass
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _sanitize_prediction(prediction: dict, cfg: dict | None) -> dict:
    if prediction.get("error"):
        prediction["error"] = sanitize_text(prediction["error"], cfg)
    return prediction


def _batch_error_kind(message: str | None) -> str:
    text = (message or "").lower()
    if "timeout" in text or "超时" in text:
        return "batch_timeout"
    return "batch_error"


def _split_error(message: str | None) -> tuple[str, str]:
    """同步回退把 error_kind 编码成 ``[kind] message``；原生错误没有前缀。"""
    text = message or ""
    match = re.match(r"^\[(\w+)\]\s*(.*)$", text, re.S)
    if match:
        return match.group(1), match.group(2)
    return _batch_error_kind(text), text


def _run_sync(
    spec: ModelSpec,
    plan: RequestPlan,
    outcome: ModelOutcome,
    *,
    agent,
    workers: int,
    requests: dict[str, RequestSpec] | None = None,
) -> None:
    todo = requests if requests is not None else plan.requests

    def work(item: RequestSpec) -> tuple[dict, int]:
        return call_once(agent, item.user_message, spec.policy)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {key: pool.submit(work, item) for key, item in todo.items()}
        for key, future in futures.items():
            try:
                prediction, retries = future.result()
            except Exception as exc:  # noqa: BLE001 —— 线程内兜底，别让一条拖垮整批
                prediction, retries = _error("internal_error", f"内部错误：{exc!r}"), 0
            outcome.predictions[key] = prediction
            outcome.sync_retries += retries


def _sanitize_batch_cache(cache_path: str | Path, cfg: dict | None) -> bool:
    """把旧 sidecar 里已经落盘的错误文本再洗一遍，返回 False 表示读不到文件。

    新结果的错误已在 ``BatchSession`` 落盘前脱敏（sanitize_error 钩子）；这里
    只是给 --batch-resume 的旧缓存兜底。缓存是本地文件，但用户可能把它连同
    报告一起分享；只改 ``error`` 字段，成功结果与键保持原样。
    """
    path = Path(cache_path)
    if not path.exists() or not path.is_file():
        return True
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    cleaned: list[str] = []
    changed = False
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            cleaned.append(line)
            continue
        if isinstance(entry, dict) and entry.get("error"):
            new_error = sanitize_text(entry["error"], cfg)
            if new_error != entry["error"]:
                entry["error"] = new_error
                changed = True
        cleaned.append(json.dumps(entry, ensure_ascii=False))
    if changed:
        with file_paths.atomic_write_text(path) as handle:
            for line in cleaned:
                handle.write(line + "\n")
    return True


def _accepted_job_keys(session) -> set[str]:
    """恢复的 jobs 覆盖了哪些请求键；用于判断 auto 是否还允许同步回退。"""
    keys: set[str] = set()
    for job in (getattr(session, "jobs", {}) or {}).values():
        keys.update(getattr(job, "keys", ()) or ())
    return keys


def _run_batch(
    spec: ModelSpec,
    plan: RequestPlan,
    outcome: ModelOutcome,
    prompts: Prompts,
    *,
    agent,
    workers: int,
    cache_path: str | Path,
    poll_seconds: float,
    timeout_seconds: float,
    chunk_size: int,
    resume: bool,
    allow_sync_fallback: bool,
    log: Callable[[str], None],
) -> None:
    import batch_inference as bi

    retry_count = 0
    lock = threading.Lock()

    def sync_call(user_message: str):
        def call():
            nonlocal retry_count
            prediction, retries = call_once(agent, user_message, spec.policy)
            with lock:
                retry_count += retries
            if prediction.get("error"):
                kind = prediction.get("error_kind") or "call_failed"
                message = sanitize_text(prediction["error"], spec.cfg)
                return None, f"[{kind}] {message}"
            return {
                "same_meaning": prediction["same_meaning"],
                "reason": prediction.get("reason", ""),
            }, None

        return call

    items = [
        bi.BatchItem(
            key=item.key,
            agent_name=REQUEST_AGENT,
            system_prompt=prompts.system,
            content=item.user_message,
            response_model=EvaluationVerdict,
            sync_call=sync_call(item.user_message),
            native=True,
        )
        for item in plan.requests.values()
    ]
    # strict_native=True 刻意与 batch_mode 解耦：一旦 batch 被受理，失败/缺失项
    # 只记错误，绝不在评测层偷偷同步重发（会改变口径、重复计费）。auto 的同步
    # 回退只允许发生在「一个 batch 都还没受理」时，见下面的 except 分支。
    session = bi.BatchSession(
        spec.batch,
        cache_path=cache_path,
        poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds,
        chunk_size=chunk_size,
        retry_policy=spec.policy,
        logger=lambda message: log(sanitize_text(message, spec.cfg)),
        sanitize_error=lambda message: sanitize_text(message, spec.cfg),
        resume=resume,
        strict_native=True,
    )
    synced_keys: set[str] = set()
    try:
        session.submit_direct(items, fallback_workers=max(1, int(workers)))
    except bi.BatchTimeout as exc:
        # 建批请求可能已经被受理，只是响应没回来；此时绝不能同步重发。
        outcome.batch_error = sanitize_exception(exc, spec.cfg)
    except Exception as exc:  # noqa: BLE001 —— 已受理的 batch 不能重发，部分结果仍要收
        outcome.batch_error = sanitize_exception(exc, spec.cfg)
        # 恢复的旧 job 同样算「已受理」：即使本轮 session.batches 还是 0，
        # 只要它的 keys 覆盖当前请求，就不能同步重发。
        restored = _accepted_job_keys(session)
        accepted = getattr(session, "batches", 0) > 0 or any(
            key in restored for key in plan.requests
        )
        if allow_sync_fallback and not accepted:
            pending = {
                key: item
                for key, item in plan.requests.items()
                if key not in session.results
            }
            if pending:
                log(f"原生批量受理前不可用，回退同步执行：{outcome.batch_error[:200]}")
                _run_sync(
                    spec, plan, outcome, agent=agent, workers=workers, requests=pending
                )
                synced_keys = set(pending)
    with lock:
        outcome.sync_retries += retry_count

    if not _sanitize_batch_cache(cache_path, spec.cfg):
        log("警告：无法读取 Batch 缓存，旧错误文本可能尚未脱敏；请检查文件权限")
    try:
        outcome.batch = session.summary()
    except Exception as exc:  # noqa: BLE001
        outcome.batch_error = outcome.batch_error or sanitize_exception(exc, spec.cfg)

    for item in plan.requests.values():
        if item.key in synced_keys:
            continue
        hit = session.results.get(item.key)
        if hit is None:
            kind = _batch_error_kind(outcome.batch_error) if outcome.batch_error else "batch_missing"
            message = outcome.batch_error or "批量结果里缺少这一项"
            outcome.predictions[item.key] = _error(kind, message)
            continue
        dumps, error = hit.get("dumps"), hit.get("error")
        if dumps is None:
            kind, message = _split_error(error)
            outcome.predictions[item.key] = _error(
                kind, message or outcome.batch_error or "批量结果为空"
            )
            continue
        try:
            verdict = (
                dumps
                if isinstance(dumps, EvaluationVerdict)
                else EvaluationVerdict.model_validate(dumps)
            )
        except Exception as exc:  # noqa: BLE001
            outcome.predictions[item.key] = _error("invalid_response", f"批量结果校验失败：{exc!r}")
            continue
        outcome.predictions[item.key] = _ok(bool(verdict.same_meaning), verdict.reason or "")


def run_model(
    spec: ModelSpec,
    samples: list[EvalSample],
    prompts: Prompts,
    *,
    workers: int,
    batch_mode: str,
    cache_path: str | Path | None,
    poll_seconds: float,
    timeout_seconds: float,
    chunk_size: int,
    resume: bool,
    log: Callable[[str], None] = print,
) -> ModelOutcome:
    """评测一个模型；每个模型独立 agent、独立缓存、独立报告。"""
    started = time.perf_counter()
    plan = prepare_requests(samples, prompts)
    outcome = ModelOutcome(plan=plan, predictions={}, unique_requests=len(plan.requests))
    if not plan.requests:
        outcome.elapsed_seconds = round(time.perf_counter() - started, 3)
        return outcome

    agent = build_agent(spec.key, prompts.system)
    use_batch = batch_mode in ("auto", "native") and spec.batch is not None and spec.batch.enabled
    if batch_mode == "native" and not use_batch:
        description = spec.batch.describe() if spec.batch is not None else "没有批处理配置"
        raise RuntimeError(f"--batch native 不可用：{description}")

    if use_batch:
        if cache_path is None:
            raise ValueError("批处理需要 cache_path")
        _run_batch(
            spec,
            plan,
            outcome,
            prompts,
            agent=agent,
            workers=workers,
            cache_path=cache_path,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
            chunk_size=chunk_size,
            resume=resume,
            allow_sync_fallback=batch_mode == "auto",
            log=log,
        )
    else:
        if batch_mode != "off":
            description = spec.batch.describe() if spec.batch is not None else "当前模型没有原生批量入口"
            log(f"批处理：{description}（回退同步执行）")
        _run_sync(spec, plan, outcome, agent=agent, workers=workers)

    for prediction in outcome.predictions.values():
        _sanitize_prediction(prediction, spec.cfg)
    outcome.elapsed_seconds = round(time.perf_counter() - started, 3)
    return outcome
