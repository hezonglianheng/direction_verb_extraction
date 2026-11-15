# encoding: utf8

"""五档评测共用的同步/严格 Batch 编排、去重、身份校验和脱敏。

业务判断只在 evaluate.annotation 及已有 annotate 节点中执行。这里不渲染旧
evaluate 模板，不读取参考标注，也不在原生 Batch 受理后同步补发。
缓存锁由入口持有（<cache>.lock）；新增身份文件同属该锁保护的 sidecar 集合。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import file_paths
import retry
import utils
from evaluate.data import EvalSample

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_DIR = _PROJECT_ROOT / "system_prompt" / "evaluate"
"""仅保留资源路径兼容常量；新推理路径不读取此目录。"""
IDENTITY_SCHEMA_VERSION = 2


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
    """一条逻辑句对；state 只含 pipeline.initial_state 清理后的输入。"""

    key: str
    state: dict


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
    pipeline: dict = field(default_factory=dict)
    agent_stats: dict = field(default_factory=dict)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _error(kind: str, message: str) -> dict:
    return {
        "same_meaning": None,
        "label": None,
        "reason": message or kind,
        "source": "error",
        "error": message or kind,
        "error_kind": kind,
        "features": {},
        "schema": {},
        "history": [],
    }


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


def _sanitize_value(value, cfg: dict | None):
    """节点 reason/history、agent first_error 与 Batch summary 也可能携带异常。"""
    if isinstance(value, str):
        return sanitize_text(value, cfg)
    if isinstance(value, dict):
        return {key: _sanitize_value(item, cfg) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item, cfg) for item in value]
    return value


def _batch_error_kind(message: str | None) -> str:
    text = (message or "").lower()
    if "timeout" in text or "超时" in text:
        return "batch_timeout"
    return "batch_error"


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


def _canonical(value):
    """稳定序列化有效配置（判定表有 tuple 键；特征键有 set）。"""
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _canonical(item) for key, item in value.items()}
        pairs = [[_canonical(key), _canonical(item)] for key, item in value.items()]
        return {"entries": sorted(pairs, key=lambda pair: _json(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=_json)
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _json(value) -> str:
    return json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value) -> str:
    return _sha256(_json(value))


def _public_config(value, cfg):
    """配置快照只输出非凭据字段；URL 与任意嵌套字符串沿用现有脱敏口径。"""
    if isinstance(value, dict):
        return {
            key: _public_config(item, cfg) for key, item in value.items()
            if not re.fullmatch(
                r"(?i)(?:.*[-_])?(?:api[-_]?key|token|secret|password|authorization|credentials)"
                r"|(?:access|refresh|auth)[-_]?token",
                str(key),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_public_config(item, cfg) for item in value]
    if isinstance(value, str):
        return redact_url(value) if "://" in value else sanitize_text(value, cfg)
    return value


def pipeline_provenance(pipeline, spec: ModelSpec) -> dict:
    """实际 agent 实例的可报告快照；不读取模板/表第二遍，也不接收样本。

    指纹含有效 system prompt、加载的原模板/表/示例、输出 schema、预算/重试策略，
    以及生成 user message / 查表 / 解析的实现文件哈希。路径用于溯源，不替代内容哈希。
    """
    import batch_inference as bi
    from evaluate import annotation

    agents = {}
    for name, agent in pipeline.agents.items():
        config = {}
        for attr in ("feature_classes", "feature_keys", "table", "schema_table", "judge_table"):
            if hasattr(agent, attr):
                table_name = (
                    "schema_table" if name == "schema" else "judge_table"
                ) if attr == "table" else attr
                path = (getattr(agent, "table_paths", {}) or {}).get(table_name)
                if path is None:
                    path = getattr(agent, "feature_table_path" if attr == "feature_keys" else "table_path", "")
                config[attr] = {"path": path, "sha256": _digest(getattr(agent, attr))}
        groups = getattr(agent, "example_groups", None)
        examples = {}
        if groups is not None:
            stages = groups if isinstance(groups, dict) else {name: groups}
            examples = {stage: {"groups": len(items), "sha256": _digest(items)}
                        for stage, items in stages.items()}
        agents[name] = {
            "class": f"{type(agent).__module__}.{type(agent).__qualname__}",
            "name": agent.NAME,
            "model_key": agent.model_key,
            "corpus_type": agent.corpus_type,
            "system_prompt": {"sha256": _sha256(agent.system_prompt)},
            "templates": {
                prompt: {"path": agent.prompt_paths.get(prompt, ""), "sha256": _sha256(text)}
                for prompt, text in agent.prompts.items()
            },
            "config": config,
            "examples": examples,
            "class_prompts_sha256": _digest(getattr(agent, "class_prompts", [])),
            "output_schema": {"sha256": _digest(agent.response_model().model_json_schema())},
            "policy": {
                "retry": _public_config(agent.retry_policy, spec.cfg),
                "allow_structured_retry": agent.allow_structured_retry,
                "max_requests": agent.max_requests,
                "batch_max_requests": getattr(agent.batch_budget, "max_requests", None),
                "deep": getattr(agent, "deep", False),
                "supports_batch": agent.supports_batch(),
                "llm": agent.llm,
            },
        }
    # 除提示语文件外，user message 和本地解析还受这些共享实现影响；版本变化宁可拒绝续跑。
    modules = (annotation, annotation.sa, annotation.sg, annotation.st, annotation.tables,
               annotation.pe, annotation.sa.agent_factory, bi, retry, utils)
    implementation = {
        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in modules
    }
    implementation[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    batch_profile = None
    if spec.batch is not None:
        batch_profile = spec.batch.fingerprint()
        # fingerprint() 将 extra 编码成字符串；先取实际字典，才能递归去凭据而不丢请求选项。
        batch_profile["extra"] = spec.batch.extra
    provenance = {
        "agents": agents,
        "model_config": _public_config(spec.cfg, spec.cfg),
        "request_policy": _public_config(spec.policy, spec.cfg),
        "proxy_mode": spec.proxy_mode,
        "batch_profile": _public_config(batch_profile, spec.cfg),
        "native_output_policy": {"tool_name": bi.TOOL_NAME, "strict_native": True},
        "implementation": implementation,
    }
    metadata = _sanitize_value({**pipeline.metadata(), "provenance": provenance}, spec.cfg)
    return {**metadata, "fingerprint": _digest(metadata)}


def prepare_requests(samples: list[EvalSample], pipeline) -> RequestPlan:
    """只按干净句对输入去重；来源身份/参考标注/中间结果均不参与逻辑键。"""
    requests, sample_to_request = {}, {}
    for sample in samples:
        state = pipeline.initial_state(sample)
        key = _digest({"pipeline": pipeline.metadata(), "input": state})[:32]
        requests.setdefault(key, RequestSpec(key, state))
        sample_to_request[sample.key] = key
    return RequestPlan(requests, sample_to_request)


def _request_sample(item: RequestSpec) -> EvalSample:
    # pipeline 的公共契约接收 EvalSample；来源/参考字段不给它真实值。
    return EvalSample(
        key=item.key, input_file="", input_line_no=0, record_id=None, line_no=None,
        sentence_a=item.state["sentence_a"], sentence_b=item.state["sentence_b"],
        focus_word=item.state["focus_word"], focus_word_b=item.state["focus_word_b"],
        reference_label="", reference_source="", reference_model=None, annotation_status=None,
        voting_status=None, candidate=None,
    )


def cache_identity_path(cache_path: str | Path) -> Path:
    """评测身份文件；与共享 Batch 的 .meta.json（模型身份）分开。"""
    path = Path(cache_path)
    return path.with_suffix(path.suffix + ".evaluate.json")


def cache_sidecar_paths(cache_path: str | Path) -> tuple[Path, Path, Path, Path]:
    """入口输出冲突检查应包含四件套；共用 <cache>.lock，不新增嵌套文件锁。"""
    return (*file_paths.sidecar_paths(cache_path), cache_identity_path(cache_path))


def _write_cache_identity(cache_path, identity):
    """提交前先原子落盘并刷新 guard；不能依赖正常返回或异常收尾才记录。"""
    with file_paths.atomic_write_text(cache_identity_path(cache_path)) as handle:
        handle.write(json.dumps(identity, ensure_ascii=False, indent=2))
        handle.flush()
        os.fsync(handle.fileno())


def _valid_request_keys(keys):
    return (
        isinstance(keys, list) and bool(keys)
        and all(isinstance(key, str) and key for key in keys)
        and len(set(keys)) == len(keys)
    )


def _validate_cache_identity(cache_path, metadata, spec, *, resume):
    expected = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "cache_path": str(Path(cache_path).resolve()),
        "pipeline": metadata,
    }
    if not resume:
        return {**expected, "native_lifecycle": {
            "state": "never_submitted", "accepted_jobs": {}, "pending_keys": [],
        }}
    else:
        path = cache_identity_path(cache_path)
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("Batch 缓存缺少有效评测身份，拒绝旧版或损坏缓存续跑") from None
        if not isinstance(actual, dict) or actual.get("schema_version") != IDENTITY_SCHEMA_VERSION:
            raise ValueError("Batch 评测身份为旧版或无有效提交生命周期，拒绝不安全缓存续跑")
        if {key: value for key, value in actual.items()
            if key not in ("native_lifecycle", "resume_blocked")} != expected:
            raise ValueError("Batch 缓存路径或评测身份不一致，拒绝续跑；请换一个缓存路径")
        lifecycle = actual.get("native_lifecycle")
        if not isinstance(lifecycle, dict) or set(lifecycle) != {
            "state", "accepted_jobs", "pending_keys",
        }:
            raise ValueError("Batch 缓存提交生命周期身份损坏，拒绝续跑")
        accepted, pending = lifecycle["accepted_jobs"], lifecycle["pending_keys"]
        if (
            not isinstance(accepted, dict)
            or not all(isinstance(job, str) and job and _valid_request_keys(keys)
                       for job, keys in accepted.items())
            or not isinstance(pending, list) or (pending and not _valid_request_keys(pending))
        ):
            raise ValueError("Batch 缓存提交生命周期记录损坏，拒绝续跑")
        state = "submitting" if pending else "accepted" if accepted else "never_submitted"
        if lifecycle["state"] != state:
            raise ValueError("Batch 缓存提交生命周期记录不一致，拒绝续跑")
        if pending or actual.get("resume_blocked"):
            raise ValueError("Batch 建批受理状态未知，禁止自动续跑；请先在供应商控制台确认，避免重复计费")
        # BatchSession 在初始化时可能写文件；在此之前验证其旧模型身份。
        native_meta = file_paths.sidecar_paths(cache_path)[2]
        try:
            actual_model = json.loads(native_meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("Batch 缓存缺少有效模型身份，拒绝续跑") from None
        if actual_model != spec.batch.fingerprint():
            raise ValueError("Batch 缓存模型身份不一致，拒绝续跑")
        jobs_path = file_paths.sidecar_paths(cache_path)[1]
        recorded = {}
        if jobs_path.exists():
            try:
                jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
                if not isinstance(jobs, list) or not all(
                    isinstance(job, dict) and isinstance(job.get("batch_id"), str)
                    and bool(job["batch_id"]) and _valid_request_keys(job.get("keys"))
                    for job in jobs
                ):
                    raise ValueError("invalid jobs")
                recorded = {job["batch_id"]: sorted(job["keys"]) for job in jobs}
                if len(recorded) != len(jobs):
                    raise ValueError("duplicate jobs")
            except (OSError, ValueError):
                raise ValueError("Batch jobs 记录损坏，拒绝续跑以免重复提交已受理请求") from None
        if recorded != {job: sorted(keys) for job, keys in accepted.items()}:
            raise ValueError("Batch jobs 记录缺失或与持久化受理记录不一致，拒绝续跑以免重复提交")
    return actual


def _run_sync(pipeline, outcome, *, workers, cfg, items=None, session=None, scopes=None):
    todo = list(outcome.plan.requests.values()) if items is None else list(items)

    def work(item):
        def invoke():
            if scopes is None:
                return pipeline.run(_request_sample(item))
            with pipeline.agents.use_budget_scope(scopes[item.key]):
                return pipeline.run(_request_sample(item))
        if session is None:
            return invoke()
        # 保留 active session：消费原先登记的预算预留，不能另建 agent 或直接 _invoke_sync。
        with session.active():
            return invoke()

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(work, item): item.key for item in todo}
        for future in as_completed(futures):
            key = futures[future]
            try:
                prediction = future.result()
            except Exception as exc:
                prediction = _error("internal_error", sanitize_exception(exc, cfg))
            outcome.predictions[key] = _sanitize_value(prediction, cfg)


def _run_batch(spec, pipeline, outcome, *, workers, cache_path, poll_seconds,
               timeout_seconds, chunk_size, resume, allow_sync_fallback, log):
    import batch_inference as bi

    # 共享 BatchSession 会把原始 extra 完整写入 .meta.json。不能仅清理评测
    # provenance，也不能悄悄删掉真实请求的凭据/选项；必须在所有 sidecar 写入前拒绝。
    if _json(_public_config(spec.batch.extra, spec.cfg)) != _json(spec.batch.extra):
        raise ValueError(
            "原生 Batch extra 含凭据字段或需脱敏内容，拒绝执行以免写入身份缓存；"
            "请使用独立的模型 API key 配置，不要把凭据放入 extra"
        )

    identity = _validate_cache_identity(cache_path, outcome.pipeline, spec, resume=resume)
    lifecycle = identity["native_lifecycle"]

    def persist_lifecycle():
        lifecycle["state"] = (
            "submitting" if lifecycle["pending_keys"]
            else "accepted" if lifecycle["accepted_jobs"] else "never_submitted"
        )
        _write_cache_identity(cache_path, identity)

    class SubmissionGuard:
        """保留真实 client.run 的分片/回调；整个调用开始前登记可能提交的键。"""

        def __init__(self, client):
            self.client = client

        def __getattr__(self, name):
            return getattr(self.client, name)

        def run(self, items, **kwargs):
            lifecycle["pending_keys"] = sorted({item.key for item in items})
            persist_lifecycle()
            try:
                results = self.client.run(items, **kwargs)
            except bi.BatchUnavailable:
                # 真实 NativeBatchClient.run 串行分片：该异常明确表示本片未提交；
                # 此前片的受理键已由 _job_submitted 单独登记，后续片尚未执行。
                lifecycle["pending_keys"] = []
                persist_lifecycle()
                raise
            except bi.BatchTimeout as exc:
                if exc.job is not None and exc.job.batch_id in lifecycle["accepted_jobs"]:
                    # 已知 job 的 wait/collect 停止了串行 run，余下键确定未提交。
                    lifecycle["pending_keys"] = []
                    persist_lifecycle()
                raise
            if lifecycle["pending_keys"]:
                raise bi.BatchTimeout("原生 Batch 未提供完整受理 job 记录，提交状态未知，禁止重发")
            return results

    class ReplaySession(bi.BatchSession):
        """异常收尾只消费缓存，不登记、扣预算或发送任何新请求。"""
        replay_only = False

        def add(self, agent, content):
            if self.replay_only:
                raise bi.BatchPending(agent.NAME, bi.item_key(agent.NAME, content, agent.system_prompt))
            return super().add(agent, content)

        def _job_submitted(self, job):
            # 先记独立受理账，再写共享 jobs 快照。两次写之间中断时，续跑会因不一致
            # fail closed；绝不把「已受理但 jobs 尚未落盘」误认为从未提交。
            lifecycle["accepted_jobs"][job.batch_id] = sorted(job.keys)
            lifecycle["pending_keys"] = sorted(set(lifecycle["pending_keys"]) - set(job.keys))
            persist_lifecycle()
            super()._job_submitted(job)

    session = ReplaySession(
        spec.batch, cache_path=cache_path, poll_seconds=poll_seconds,
        timeout_seconds=timeout_seconds, chunk_size=chunk_size, retry_policy=spec.policy,
        logger=lambda message: log(sanitize_text(message, spec.cfg)),
        sanitize_error=lambda message: sanitize_text(message, spec.cfg),
        resume=resume, strict_native=True,
    )
    if not resume:
        # Session 初始化只重置本地缓存，不发请求；重置完成前不发布 never_submitted。
        persist_lifecycle()
    if session.client is not None:
        session.client = SubmissionGuard(session.client)
    states = [{"key": item.key} for item in outcome.plan.requests.values()]
    scopes = {state["key"]: pipeline.agents.new_budget_scope() for state in states}

    def invoke(state):
        key = state["key"]
        with pipeline.agents.use_budget_scope(scopes[key]):
            prediction = pipeline.run(_request_sample(outcome.plan.requests[key]))
        return {"key": key, "prediction": prediction}

    def completed(state):
        outcome.predictions[state["key"]] = _sanitize_value(state["prediction"], spec.cfg)

    try:
        bi.run_waves(states, invoke, session, fallback_workers=workers, on_complete=completed)
    except Exception as exc:
        outcome.batch_error = sanitize_exception(exc, spec.cfg)
        accepted = bool(session.batches or session.jobs or lifecycle["accepted_jobs"])
        uncertain = bool(lifecycle["pending_keys"])
        if uncertain:
            # 这只是补充可读诊断；即使 BaseException/进程中断绕过此分支，
            # 提交前落盘的 submitting guard 也已经禁止自动续跑。
            identity["resume_blocked"] = {
                "reason": "uncertain_acceptance", "request_keys": lifecycle["pending_keys"],
                "error": outcome.batch_error,
            }
            persist_lifecycle()
        # BatchTimeout 包括建批响应丢失的受理不确定状态，永远不能回退。
        if allow_sync_fallback and not accepted and not uncertain and not isinstance(exc, bi.BatchTimeout):
            log(f"原生批量受理前不可用，回退同步执行：{outcome.batch_error[:200]}")
            session.fallback_mode = True
            session.mode = "fallback"
            session.disabled_reason = outcome.batch_error
            before = sum(agent.stats()["requests"] for agent in pipeline.agents.values())
            _run_sync(
                pipeline, outcome, workers=workers, cfg=spec.cfg, session=session, scopes=scopes,
                items=(item for key, item in outcome.plan.requests.items() if key not in outcome.predictions),
            )
            session.fallback_items = sum(agent.stats()["requests"] for agent in pipeline.agents.values()) - before
        else:
            # on_complete 已记录之前完成的逻辑句对。还有一片已落盘但 execute 抛错时，
            # 只回放这些结果，让可完成的句对经过同一 pipeline 正常校验后立即保留。
            session.replay_only = True
            with session.active():
                for state in states:
                    if state["key"] in outcome.predictions:
                        continue
                    try:
                        completed(invoke(state))
                    except bi.BatchPending:
                        pass
    for key in outcome.plan.requests:
        if key not in outcome.predictions:
            outcome.predictions[key] = _error(
                _batch_error_kind(outcome.batch_error) if outcome.batch_error else "batch_missing",
                outcome.batch_error or "批量结果缺失；请使用 --batch-resume 继续收取",
            )
    if not _sanitize_batch_cache(cache_path, spec.cfg):
        log("警告：无法读取 Batch 缓存，旧错误文本可能尚未脱敏；请检查文件权限")
    outcome.batch = _sanitize_value(session.summary(), spec.cfg)
    outcome.batch["identity_file"] = str(cache_identity_path(cache_path))
    outcome.batch["resume_blocked"] = identity.get("resume_blocked")


def run_model(
    spec: ModelSpec,
    samples: list[EvalSample],
    *,
    step: int,
    corpus_type: str,
    workers: int,
    batch_mode: str,
    cache_path: str | Path | None,
    poll_seconds: float,
    timeout_seconds: float,
    chunk_size: int,
    resume: bool,
    log: Callable[[str], None] = print,
) -> ModelOutcome:
    """每个模型独立 pipeline；五档同步/Batch 共用 Task 1 的完整业务节点。"""
    from evaluate.annotation import build_pipeline

    started = time.perf_counter()
    if batch_mode not in ("off", "auto", "native"):
        raise ValueError(f"非法 batch_mode {batch_mode!r}")
    pipeline = build_pipeline(step, spec.key, corpus_type, llm=True, policy=spec.policy)
    plan = prepare_requests(samples, pipeline)
    outcome = ModelOutcome(plan=plan, predictions={}, unique_requests=len(plan.requests),
                           pipeline=pipeline_provenance(pipeline, spec))
    use_batch = batch_mode in ("auto", "native") and spec.batch is not None and spec.batch.enabled
    if batch_mode == "native" and not use_batch:
        description = spec.batch.describe() if spec.batch is not None else "没有批处理配置"
        raise RuntimeError(sanitize_text(f"--batch native 不可用：{description}", spec.cfg))
    if resume and not use_batch:
        raise ValueError("Batch 续跑需要启用的原生批处理配置")

    if use_batch:
        if cache_path is None:
            raise ValueError("批处理需要 cache_path")
        _run_batch(
            spec,
            pipeline,
            outcome,
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
        if batch_mode != "off" and plan.requests:
            description = spec.batch.describe() if spec.batch is not None else "当前模型没有原生批量入口"
            log(sanitize_text(f"批处理：{description}（回退同步执行）", spec.cfg))
        _run_sync(pipeline, outcome, workers=workers, cfg=spec.cfg)

    outcome.agent_stats = _sanitize_value(
        {name: agent.stats() for name, agent in pipeline.agents.items()}, spec.cfg)
    outcome.sync_retries = sum(stats.get("retries", 0) for stats in outcome.agent_stats.values())
    outcome.elapsed_seconds = round(time.perf_counter() - started, 3)
    return outcome
