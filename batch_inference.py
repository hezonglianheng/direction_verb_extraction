# encoding: utf8

"""LLM 标注的批量推理入口：provider 原生 Batch API + 明确的同步回退。

为什么单独一层：
    七个 agent 的同步链路已经很长（消息构造、缓存、预算、重试、结构化输出），把
    provider 的「上传 JSONL → 建 batch → 轮询 → 下载结果」塞进各个 agent 里，会让
    每一处都各写一份协议细节。本模块只负责**把一批已经拼好的请求送出去、再把结果按
    稳定 ID 收回来**，业务提示语与结果校验仍留在 spatial_agents/all_in_one。

两种模式，口径必须分清：
    native      原生 Batch API，按 profile.protocol 分两套：
                openai     上传 JSONL（`/files`）→ 建批 → 从结果文件下载；
                openrouter 内联 requests 数组 → 结果直接随 `GET /batches/{id}` 返回。
    fallback    没有原生入口、或原生入口提交失败时，用有界线程池逐条跑现有同步调用。
                它**不是**批量推理，只在汇总里如实记成 fallback，不能拿它冒充原生批处理。

如何与 langgraph 配合（波次执行，不改图的语义）：
    原生 Batch 是异步长任务，不能让一个图节点阻塞等待。所以按「波次」重放同一张图：
    第一次重放会在第一个需要模型的节点处抛出 BatchPending，收集这一波所有待发请求；
    整批提交并收齐结果后，从初始状态重新重放——已拿到的结果由 BatchSession 直接回放，
    图会走到下一个需要模型的节点，再收集下一波。闸门、查表、路由仍由原图决定，
    批处理层不复制一套阶段判断，因此不会出现「批处理口径与图口径」两套逻辑。

    代价是纯规则步骤会按波次重复执行（毫秒级、无副作用）；换来的是图怎么改，批处理
    都不用跟着改。save 节点只在状态真正走完时触发，不会重复落盘。

结果缓存与恢复：
    每条结果按 (agent 名, system prompt, 完整 user message) 的哈希写进 sidecar
    `.batch-cache.jsonl`。
    同一次运行里它是跨波次的回放来源；加 `--batch-resume` 时也跨进程复用。
    batch 提交后另写 `.batch-jobs.json`：轮询超时不会丢 batch ID，下次可先收旧批，
    避免「不知道上次有没有受理」时重复提交、重复计费。

不做的事：
    - 不把 LangChain 的 `.batch()` 或「一次 prompt 多条 item」称作原生 Batch API；
    - 不在 worker 里补算 LTP（句法证据由 spatial_tools.prepare_syntax 一次准备）；
    - 不改查表优先级、闸门放行政策、结果 ID/CSV 与投票规则。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests
from pydantic import BaseModel

import agents as agent_factory
import config
import retry

TOOL_NAME = "submit_verdict"
"""原生 Batch 请求里强制调用的结构化输出工具名（与 ToolStrategy 的机制对齐）"""

_NATIVE_BATCH_PROVIDERS = frozenset({"openai", "openrouter"})
"""默认支持 OpenAI 兼容 Batch API 的 provider 名。

不在表里的 provider（例如 deepseek）不会凭 provider 名猜测能力：要启用原生批处理，
必须在模型配置里显式写 `"batch": {...}`。猜错的代价是整批跑不起来，而不是慢一点。
"""

_TERMINAL_OK = frozenset({"completed"})
_TERMINAL_BAD = frozenset({"failed", "expired", "cancelled"})
"""cancelling 不算终态：取消请求还在处理时远端可能继续产出结果，继续轮询到 cancelled"""


class BatchUnavailable(RuntimeError):
    """当前模型/端点没有可用的原生批量入口，调用方应回退同步执行"""


class BatchTimeout(RuntimeError):
    """等待原生 batch 完成超时；batch ID 已落盘，可带 --batch-resume 继续收"""

    def __init__(self, message: str, job: "BatchJob | None" = None):
        super().__init__(message)
        self.job = job


class BatchPending(BaseException):
    """图重放到需要模型的节点时抛出：这一问已登记进当前波次，等整批结果回来再重放

    刻意继承 BaseException：各 agent 的 run() 用 `except Exception` 兜底单条失败，
    BatchPending 若被它们吞掉，就会变成一条假的 error 结果，图也走不到下一波。
    """

    def __init__(self, agent_name: str, key: str):
        super().__init__(f"{agent_name} 的请求已进入批处理波次（{key}）")
        self.agent_name = agent_name
        self.key = key


class BatchPendingGroup(BatchPending):
    """一次节点里 a/b 两侧都待发时，把两个 pending 一起抛出去

    agent_node 用它对两侧各跑一次：只收集、不返回结果，避免「先 pending 的那一侧
    把另一侧挡住」，否则 semantic/feature/schema 每个节点都要多花一波。
    """

    def __init__(self, pending: list[BatchPending]):
        super().__init__(pending[0].agent_name if pending else "batch", pending[0].key if pending else "")
        self.items = list(pending)


_CURRENT_BATCH: ContextVar["BatchSession | None"] = ContextVar("spatial_batch_session", default=None)
"""当前波次使用的 BatchSession（None = 不在批处理里，走同步调用）"""


def current_session() -> "BatchSession | None":
    return _CURRENT_BATCH.get()


def item_key(agent_name: str, content: str, system_prompt: str = "") -> str:
    """一条请求的稳定键：agent 名 + system prompt + 完整 user message

    system prompt 也进键，是为了让 `--batch-resume` 在提示语/类别表改过之后不会把旧结果
    回放进来：同一句话配不同系统提示语，本来就不是同一个问题。模型名由会话隔离（一个模型
    一个 BatchSession），所以不重复放进键里。
    """
    digest = hashlib.sha256(
        f"{agent_name}\0{system_prompt}\0{content}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


@dataclass
class BatchProfile:
    """一个模型配置对应的批处理能力与请求参数"""

    model_key: str
    provider: str
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    completion_window: str = "24h"
    protocol: str = "openai"
    """批量协议：openai = 上传 JSONL 文件再建批；openrouter = 直接内联 requests 数组

    两者的差别不只是请求体：OpenAI 从 `/files/{id}/content` 下载结果，OpenRouter 的
    `GET /batches/{id}` 直接返回 results。把协议显式记在 profile 上，客户端不按 provider
    名猜。
    """
    enabled: bool = False
    disabled_reason: str = ""
    extra: dict = field(default_factory=dict)
    proxy_mode: str = "env"
    """原生 Batch 的代理口径，与 agents.get_model 的 proxy 字段同一套语义：

    env = 沿用 requests 的环境代理；direct = 强制直连；url = 使用 proxy_url。
    原生 Batch 走裸 requests，不经过 langchain 的 http_client，所以必须在这里显式带上。
    """
    proxy_url: str | None = None

    def describe(self) -> str:
        if self.enabled:
            return f"{self.model_key}（{self.provider}，{self.protocol}，{self.base_url}/batches）"
        return f"{self.model_key}：原生批处理不可用（{self.disabled_reason or '未配置'}）"

    def fingerprint(self) -> dict:
        """跨进程续跑要核对的模型身份（不含 api_key）"""
        return {
            "model_key": self.model_key,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "protocol": self.protocol,
            "temperature": self.temperature,
            "completion_window": self.completion_window,
            "extra": json.dumps(self.extra, ensure_ascii=False, sort_keys=True),
        }


def _default_protocol(provider: str) -> str:
    """按 provider 选默认批量协议；显式 batch.protocol 可覆盖

    OpenAI 是文件式 Batch API；OpenRouter 是内联 requests 数组（没有文件上传）。
    把两者分开而不是硬套一种，是因为请求体与结果获取方式都不同。
    """
    return "openrouter" if provider == "openrouter" else "openai"


def batch_profile(model_key: str) -> BatchProfile:
    """读取并规范化某个模型的批处理配置（写错在启动时抛，不把错误留到跑批中段）

    优先级：
        1. 模型配置里的 `"batch"` 字段（false = 明确关闭；dict = 覆盖端点/模型等）；
        2. provider 属于 OpenAI/OpenRouter 时默认启用，端点取 base_url；
        3. 其余 provider 默认关闭，理由写进 disabled_reason。
    """
    cfg = agent_factory.load_model_config(model_key)
    # 与 agents.get_model 同一套取法：优先 model_provider，其次 provider。
    provider = str(cfg.get("model_provider") or cfg.get("provider") or "")
    # 原生 Batch 不走 langchain，必须把模型的 proxy 口径显式带过来，否则
    # api_keys.json 里的 proxy=false / 显式代理不会作用于批量请求。
    proxy_mode, proxy_url = agent_factory.proxy_policy(cfg, model_key)
    batch = cfg.get("batch")
    if batch is False:
        return BatchProfile(
            model_key=model_key,
            provider=provider,
            model=str(cfg.get("model") or cfg.get("model_name") or ""),
            base_url=str(cfg.get("base_url") or ""),
            api_key=str(cfg.get("api_key") or ""),
            protocol=_default_protocol(provider),
            disabled_reason="模型配置里 batch 显式关闭",
            proxy_mode=proxy_mode,
            proxy_url=proxy_url,
        )
    if isinstance(batch, dict):
        enabled = bool(batch.get("enabled", True))
        base_url = str(batch.get("base_url") or cfg.get("base_url") or "")
        model = str(batch.get("model") or cfg.get("model") or cfg.get("model_name") or "")
        reason = "" if enabled else str(batch.get("reason") or "batch.enabled=false")
        completion = str(batch.get("completion_window") or "24h")
        api_key = str(batch.get("api_key") or cfg.get("api_key") or "")
        protocol = str(batch.get("protocol") or _default_protocol(provider))
    elif provider in _NATIVE_BATCH_PROVIDERS:
        enabled = True
        base_url = str(cfg.get("base_url") or "")
        model = str(cfg.get("model") or cfg.get("model_name") or "")
        reason = ""
        completion = "24h"
        api_key = str(cfg.get("api_key") or "")
        protocol = _default_protocol(provider)
    else:
        return BatchProfile(
            model_key=model_key,
            provider=provider,
            model=str(cfg.get("model") or cfg.get("model_name") or ""),
            base_url=str(cfg.get("base_url") or ""),
            api_key=str(cfg.get("api_key") or ""),
            protocol=_default_protocol(provider),
            disabled_reason=f"provider {provider!r} 未声明 batch 配置；如该端点兼容 OpenAI Batch，"
            '请在模型配置里加 "batch": {"enabled": true, "base_url": ...}',
            proxy_mode=proxy_mode,
            proxy_url=proxy_url,
        )
    if enabled and not base_url:
        raise ValueError(f"模型 {model_key!r} 启用了批处理但 base_url 为空")
    if enabled and not api_key:
        raise ValueError(f"模型 {model_key!r} 启用了批处理但 api_key 为空")
    if protocol not in ("openai", "openrouter"):
        raise ValueError(
            f"模型 {model_key!r} 的 batch.protocol 只能是 openai 或 openrouter，收到 {protocol!r}"
        )
    if enabled and protocol == "openrouter" and completion != "24h":
        raise ValueError(
            f"模型 {model_key!r} 用 OpenRouter 批量接口时 completion_window 只能是 24h，"
            f"收到 {completion!r}"
        )
    extra = _flatten_extra(agent_factory.disable_thinking_kwargs(cfg))
    return BatchProfile(
        model_key=model_key,
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        temperature=float(cfg.get("temperature", 0) or 0),
        completion_window=completion,
        protocol=protocol,
        enabled=enabled,
        disabled_reason=reason,
        extra=extra,
        proxy_mode=proxy_mode,
        proxy_url=proxy_url,
    )


def _flatten_extra(raw: dict | None) -> dict:
    """把 disable_thinking 的 {extra_body: {...}} / {reasoning: {...}} 摊平进请求体

    原生 HTTP 请求没有 langchain 的 extra_body 包装：deepseek 要 `{"thinking": ...}`，
    OpenRouter 要 `{"reasoning": ...}`，都直接放在 body 顶层。
    """
    out: dict = {}
    for key, value in (raw or {}).items():
        if key == "extra_body" and isinstance(value, dict):
            out.update(value)
        else:
            out[key] = value
    return out


@dataclass
class BatchItem:
    """一条待发请求（与具体 provider 无关）"""

    key: str
    agent_name: str
    system_prompt: str
    content: str
    response_model: type[BaseModel] | None = None
    sync_call: Callable[[], tuple[dict | None, str | None]] | None = None
    native: bool = True
    """False = 这条只能同步跑（例如 deep feature agent 的工具回合），不送原生 Batch"""

    def request_body(self, profile: BatchProfile) -> dict:
        """OpenAI 兼容 chat.completions 请求体（结构与同步链路的 ToolStrategy 对齐）"""
        body: dict[str, Any] = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.content},
            ],
            "temperature": profile.temperature,
        }
        if self.response_model is not None:
            schema = self.response_model.model_json_schema()
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "description": "按给定的结构化字段返回本次判断结果",
                        "parameters": schema,
                    },
                }
            ]
            body["tool_choice"] = {"type": "function", "function": {"name": TOOL_NAME}}
        body.update(profile.extra)
        return body

    def parse_response(self, body: dict) -> dict:
        """从一条 chat.completion 结果里取出结构化判定并校验

        优先读强制工具调用的 arguments（与同步 ToolStrategy 同一形状）；
        个别兼容网关会忽略 tool_choice 而把 JSON 放在 content 里，也接受。
        """
        if not isinstance(body, dict):
            raise ValueError(f"批量结果不是 JSON 对象：{body!r}")
        if body.get("error"):
            raise RuntimeError(f"接口返回错误：{body['error']}")
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("批量结果里没有 choices")
        message = choices[0].get("message") or {}
        payload: Any = None
        calls = message.get("tool_calls") or []
        if calls:
            function = (calls[0] or {}).get("function") or {}
            payload = function.get("arguments")
        if payload is None:
            legacy = message.get("function_call") or {}
            payload = legacy.get("arguments")
        if payload is None:
            payload = message.get("content")
        if isinstance(payload, str):
            text = payload.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip()
            payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"结构化输出不是 JSON 对象：{payload!r}")
        if self.response_model is None:
            return payload
        return self.response_model.model_validate(payload).model_dump()


@dataclass
class BatchJob:
    """一个已受理的原生 batch（落盘时只存这几个字段，不存 API key）"""

    batch_id: str
    keys: list[str]
    input_file_id: str = ""
    output_file_id: str = ""
    error_file_id: str = ""
    status: str = "submitted"
    created_at: float = 0.0
    protocol: str = "openai"
    results: list | None = field(default=None, repr=False)
    """OpenRouter 的 results 内联在 GET /batches/{id} 里；不落盘（可能很大），续跑时重新拉"""

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "keys": list(self.keys),
            "input_file_id": self.input_file_id,
            "output_file_id": self.output_file_id,
            "error_file_id": self.error_file_id,
            "status": self.status,
            "created_at": self.created_at,
            "protocol": self.protocol,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatchJob":
        return cls(
            batch_id=str(data.get("batch_id") or ""),
            keys=[str(key) for key in data.get("keys") or []],
            input_file_id=str(data.get("input_file_id") or ""),
            output_file_id=str(data.get("output_file_id") or ""),
            error_file_id=str(data.get("error_file_id") or ""),
            status=str(data.get("status") or "submitted"),
            created_at=float(data.get("created_at") or 0.0),
            protocol=str(data.get("protocol") or "openai"),
        )


class NativeBatchClient:
    """原生批量接口的最小客户端（不引入额外 SDK，复用 requests）

    两套协议（见 BatchProfile.protocol）：

    openai（OpenAI 及兼容网关）：
        POST {base}/files                 上传 JSONL（purpose=batch）
        POST {base}/batches               建批（input_file_id + endpoint）
        GET  {base}/batches/{id}          轮询
        GET  {base}/files/{id}/content    下载结果/错误文件

    openrouter：
        POST {base}/batches               内联 {endpoint, model, requests:[{custom_id, body}]}
        GET  {base}/batches/{id}          轮询；完成后 results 直接返回，没有文件下载

    路径不通（404/405）会抛 BatchUnavailable，由会话层回退同步执行。
    """

    def __init__(
        self,
        profile: BatchProfile,
        *,
        poll_seconds: float = 30,
        timeout_seconds: float = 86400,
        chunk_size: int = 5000,
        retry_policy: dict | None = None,
        logger: Callable[[str], None] = print,
    ):
        self.profile = profile
        self.poll_seconds = max(1.0, float(poll_seconds))
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.chunk_size = max(1, int(chunk_size))
        self.retry_policy = dict(retry_policy or {"max_retries": 0})
        self.logger = logger
        # 原生 Batch 走裸 requests；这里复刻 agents.get_model 的 proxy 口径：
        # env 交给 requests 自己读环境变量，direct/url 则关掉环境代理、显式设置。
        self.session = requests.Session()
        if profile.proxy_mode == "direct":
            self.session.trust_env = False
            self.session.proxies.clear()
        elif profile.proxy_mode == "url":
            if not profile.proxy_url:
                raise ValueError("原生 Batch 的 proxy_mode=url 但 proxy_url 为空")
            self.session.trust_env = False
            self.session.proxies.update(
                {"http": profile.proxy_url, "https": profile.proxy_url}
            )

    # -- HTTP --------------------------------------------------------------
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.profile.api_key}"}

    def _request(self, method: str, path: str, *, idempotent: bool = True, **kwargs) -> requests.Response:
        url = f"{self.profile.base_url}{path}"
        timeout = float(self.retry_policy.get("timeout", config.REQUEST_TIMEOUT_SECONDS))
        kwargs.setdefault("timeout", timeout)
        extra_headers = kwargs.pop("headers", {}) or {}
        headers = {**self._headers(), **extra_headers}
        policy = self.retry_policy if idempotent else {**self.retry_policy, "max_retries": 0}

        def call():
            response = self.session.request(method, url, headers=headers, **kwargs)
            if response.status_code in (404, 405):
                raise BatchUnavailable(f"{method} {url} 不是可用的批量接口（HTTP {response.status_code}）")
            if response.status_code >= 400:
                error = requests.HTTPError(
                    f"{method} {url} 失败：HTTP {response.status_code} {response.text[:500]}",
                    response=response,
                )
                raise error
            return response

        return retry.call_with_retry(call, policy=policy)

    # -- 生命周期 ----------------------------------------------------------
    def submit(self, items: list[BatchItem]) -> BatchJob:
        if not items:
            raise ValueError("没有待提交的批量请求")
        if self.profile.protocol == "openrouter":
            return self._submit_openrouter(items)
        return self._submit_openai(items)

    def _submit_openrouter(self, items: list[BatchItem]) -> BatchJob:
        """OpenRouter：内联 requests 数组，上传文件与结果下载都不存在

        字段顺序有要求（docs：endpoint/model/provider/completion_window 必须在 requests
        之前），Python dict 保插入序，所以这里按顺序构造。每条 request 的 body 与同步
        chat.completions 完全一致，结构化输出仍靠 tools/tool_choice。
        """
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": self.profile.model,
            "completion_window": self.profile.completion_window,
            "requests": [
                {"custom_id": item.key, "body": item.request_body(self.profile)} for item in items
            ],
        }
        # 创建 batch 不是幂等操作：响应丢了也不能自动重发，否则可能重复计费。
        # 同时带一个由请求内容算出的 Idempotency-Key，支持该头的网关会返回同一个批。
        idem = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        try:
            response = self._request(
                "POST",
                "/batches",
                json=payload,
                idempotent=False,
                headers={"Idempotency-Key": idem},
            )
        except Exception as exc:
            if retry.is_retryable_error(exc):
                raise BatchTimeout(
                    "OpenRouter 建批请求可能已被受理，但响应没有收全；为避免重复计费，本轮停止。"
                    "请先在 OpenRouter 控制台确认，再决定是否重跑",
                ) from exc
            raise
        body = response.json()
        batch_id = body.get("id")
        if not batch_id:
            raise RuntimeError(f"创建 OpenRouter batch 后没有拿到 id：{response.text[:500]}")
        job = BatchJob(
            batch_id=str(batch_id),
            keys=[item.key for item in items],
            status=str(body.get("status") or "submitted"),
            created_at=time.time(),
            protocol="openrouter",
            results=body.get("results"),
        )
        self.logger(f"  OpenRouter batch 已提交：{batch_id}（{len(items)} 项）")
        return job

    def _submit_openai(self, items: list[BatchItem]) -> BatchJob:
        """OpenAI 兼容文件式 Batch：上传 JSONL -> 建批 -> 结果从 files 下载"""
        lines = [
            json.dumps(
                {
                    "custom_id": item.key,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": item.request_body(self.profile),
                },
                ensure_ascii=False,
            )
            for item in items
        ]
        data = ("\n".join(lines) + "\n").encode("utf-8")
        upload = self._request(
            "POST",
            "/files",
            data={"purpose": "batch"},
            files={"file": ("batch.jsonl", data, "application/jsonl")},
        )
        file_id = upload.json().get("id")
        if not file_id:
            raise RuntimeError(f"上传批量文件后没有拿到 id：{upload.text[:500]}")
        create_body = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": self.profile.completion_window,
        }
        try:
            created = self._request(
                "POST",
                "/batches",
                idempotent=False,
                json=create_body,
                headers={
                    "Idempotency-Key": hashlib.sha256(
                        json.dumps(create_body, ensure_ascii=False, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                },
            )
        except Exception as exc:
            if retry.is_retryable_error(exc):
                raise BatchTimeout(
                    "输入文件已上传，但建批请求可能已被受理、只是响应没收到；为避免重复计费，"
                    "本轮停止。请先在供应商控制台确认，再决定是否重跑",
                ) from exc
            raise
        payload = created.json()
        batch_id = payload.get("id")
        if not batch_id:
            raise RuntimeError(f"创建 batch 后没有拿到 id：{created.text[:500]}")
        job = BatchJob(
            batch_id=str(batch_id),
            keys=[item.key for item in items],
            input_file_id=str(file_id),
            status=str(payload.get("status") or "submitted"),
            created_at=time.time(),
            protocol="openai",
        )
        self.logger(f"  原生 batch 已提交：{batch_id}（{len(items)} 项）")
        return job

    def wait(self, job: BatchJob) -> BatchJob:
        started = time.time()
        while True:
            response = self._request("GET", f"/batches/{job.batch_id}")
            payload = response.json()
            job.status = str(payload.get("status") or job.status)
            job.output_file_id = str(payload.get("output_file_id") or job.output_file_id or "")
            job.error_file_id = str(payload.get("error_file_id") or job.error_file_id or "")
            if job.protocol == "openrouter":
                job.results = payload.get("results")
                # completed 但 results 还没挂上（列表接口或竞态）时继续等，不能提前收
                if job.status in _TERMINAL_OK and job.results is None:
                    if self.timeout_seconds and time.time() - started > self.timeout_seconds:
                        raise BatchTimeout(
                            f"batch {job.batch_id} 已 completed 但一直没返回 results，"
                            f"等待超过 {self.timeout_seconds:.0f} 秒；可加 --batch-resume 继续收",
                            job=job,
                        )
                    time.sleep(self.poll_seconds)
                    continue
            if job.status in _TERMINAL_OK or job.status in _TERMINAL_BAD:
                return job
            if self.timeout_seconds and time.time() - started > self.timeout_seconds:
                raise BatchTimeout(
                    f"等待 batch {job.batch_id} 超过 {self.timeout_seconds:.0f} 秒仍未结束"
                    f"（当前状态 {job.status}）；batch ID 已记入 .batch-jobs.json，可加 --batch-resume 继续收",
                    job=job,
                )
            time.sleep(self.poll_seconds)

    def collect(self, job: BatchJob) -> dict[str, tuple[dict | None, str | None]]:
        """取回结果，返回 {key: (body, error)}；缺项按错误处理，不猜答案"""
        if job.protocol == "openrouter":
            return self._collect_openrouter(job)
        return self._collect_openai(job)

    def _collect_openrouter(self, job: BatchJob) -> dict[str, tuple[dict | None, str | None]]:
        """OpenRouter 的 results 直接在 GET /batches/{id} 里，没有文件下载"""
        raws: dict[str, dict] = {}
        errors: dict[str, str] = {}
        for result in job.results or []:
            if not isinstance(result, dict):
                continue
            custom_id = str(result.get("custom_id") or "")
            if not custom_id:
                continue
            response = result.get("response")
            if isinstance(response, dict) and isinstance(response.get("body"), dict):
                raws[custom_id] = response["body"]
            elif result.get("error"):
                errors[custom_id] = str(result["error"])
        result_map = {
            key: (raws.get(key), errors.get(key) if key not in raws else None) for key in job.keys
        }
        if job.status in _TERMINAL_BAD:
            for key in job.keys:
                if key not in raws and key not in errors:
                    result_map[key] = (None, f"batch 状态为 {job.status}")
        return result_map

    def _collect_openai(self, job: BatchJob) -> dict[str, tuple[dict | None, str | None]]:
        """OpenAI 兼容文件式结果：下载 output/error 文件"""
        raws: dict[str, dict] = {}
        errors: dict[str, str] = {}
        if job.output_file_id:
            content = self._request("GET", f"/files/{job.output_file_id}/content").text
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                custom_id = str(item.get("custom_id") or "")
                if not custom_id:
                    continue
                response = item.get("response")
                if isinstance(response, dict) and isinstance(response.get("body"), dict):
                    raws[custom_id] = response["body"]
                elif item.get("error"):
                    errors[custom_id] = str(item["error"])
        if job.error_file_id:
            content = self._request("GET", f"/files/{job.error_file_id}/content").text
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                custom_id = str(item.get("custom_id") or "")
                if custom_id:
                    errors[custom_id] = str(item.get("error") or item.get("response") or line[:500])
        result = {
            key: (raws.get(key), errors.get(key) if key not in raws else None) for key in job.keys
        }
        if job.status in _TERMINAL_BAD:
            for key in job.keys:
                if key not in raws and key not in errors:
                    result[key] = (None, f"batch 状态为 {job.status}")
        return result

    def run(
        self,
        items: list[BatchItem],
        *,
        on_job: Callable[[BatchJob], None] | None = None,
        on_status: Callable[[BatchJob], None] | None = None,
        on_results: Callable[[dict[str, tuple[dict | None, str | None]]], None] | None = None,
    ) -> dict[str, tuple[dict | None, str | None]]:
        """提交 + 等待 + 收集（按 chunk_size 分片，逐片串行）

        on_results 在**每一片收完后立即**回调，调用方据此落缓存：后面某一分片失败时，
        前面的结果不会因为异常而丢失、被整批重跑。
        """
        results: dict[str, tuple[dict | None, str | None]] = {}
        for start in range(0, len(items), self.chunk_size):
            chunk = items[start : start + self.chunk_size]
            job = self.submit(chunk)
            if on_job is not None:
                on_job(job)
            try:
                job = self.wait(job)
                if on_status is not None:
                    on_status(job)
                raw = self.collect(job)
            except BatchTimeout:
                raise
            except Exception as exc:  # noqa: BLE001 —— 已受理后不能当作没发生，交给续跑
                raise BatchTimeout(
                    f"batch {job.batch_id} 已受理，但等待/下载结果失败：{exc!r}；"
                    "不会自动重新提交，可加 --batch-resume 继续收",
                    job=job,
                ) from exc
            chunk_results: dict[str, tuple[dict | None, str | None]] = {}
            for item in chunk:
                body, error = raw.get(item.key, (None, "批量结果里缺少这一项"))
                if body is None:
                    chunk_results[item.key] = (None, error or "批量结果为空")
                    continue
                try:
                    chunk_results[item.key] = (item.parse_response(body), None)
                except Exception as exc:  # noqa: BLE001 —— 单条解析失败只影响这一条
                    chunk_results[item.key] = (None, f"批量结果解析失败：{exc!r}")
            results.update(chunk_results)
            if on_results is not None:
                on_results(chunk_results)
        return results


class BatchSession:
    """一次跑批的批量会话：收集、提交、回放、持久化、统计

    一个会话对应**一个模型配置**（多模型投票时每个模型各一份），但句法证据是共享的。
    """

    def __init__(
        self,
        profile: BatchProfile,
        *,
        cache_path: str | Path,
        poll_seconds: float = 30,
        timeout_seconds: float = 86400,
        chunk_size: int = 5000,
        retry_policy: dict | None = None,
        logger: Callable[[str], None] = print,
        client: NativeBatchClient | None = None,
        resume: bool = False,
        strict_native: bool = False,
    ):
        self.profile = profile
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs_path = self.cache_path.with_suffix(self.cache_path.suffix + ".jobs.json")
        self.meta_path = self.cache_path.with_suffix(self.cache_path.suffix + ".meta.json")
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.chunk_size = chunk_size
        self.retry_policy = dict(retry_policy or {"max_retries": 0})
        self.logger = logger
        self.resume = bool(resume)
        self.strict_native = bool(strict_native)
        """True = --batch native：需要回退时直接报错，不静默走同步链路"""
        self.client = client
        if self.client is None and profile.enabled:
            self.client = NativeBatchClient(
                profile,
                poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
                chunk_size=chunk_size,
                retry_policy=self.retry_policy,
                logger=logger,
            )
        self.results: dict[str, dict] = {}
        self.pending: dict[str, BatchItem] = {}
        self.jobs: dict[str, BatchJob] = {}
        self._seen: set[str] = set()
        """已经登记过的请求键（统计 items_collected，避免波次重放时重复计数）"""
        self._reserved: set[str] = set()
        """已扣过预算、但还没真正发出的键；同步回退首发时据此避免二次扣减"""
        self.collecting = False
        self.fallback_mode = False
        """原生批量不可用/失败后置位：下一波不再收集请求，直接走同步 _invoke_sync

        这样同步回退仍经过各 agent 的 run() → _ask 业务校验，而不是只拿裸结构化输出。
        """
        self.mode = "native" if profile.enabled else "fallback"
        self.waves = 0
        self.items_collected = 0
        self.items_completed = 0
        self.items_failed = 0
        self.cache_hits = 0
        self.fallback_items = 0
        self.batches = 0
        self.disabled_reason = profile.disabled_reason
        self._lock = threading.Lock()
        self._load_results()
        if not self.resume:
            # 新一轮跑批不能留下旧模型的 job：否则先退出、再以 resume 启动时，
            # 新 meta 会通过校验，却把旧 job 当成自己的收进来。
            self.jobs_path.unlink(missing_ok=True)
            self.jobs = {}
        self._check_or_write_meta()
        if self.resume:
            # 只有显式续跑才收旧 batch；新一轮跑批不能把旧 job 的结果混进来
            self._load_jobs()

    # -- 持久化 ------------------------------------------------------------
    def _load_results(self) -> None:
        if not self.cache_path.exists():
            return
        if not self.resume:
            # 新一轮跑批：旧 sidecar 不能混进本轮结果（与 ResultWriter 的覆盖政策同口径）
            self.cache_path.write_text("", encoding="utf-8")
            return
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(entry.get("key") or "")
            if key:
                self.results[key] = {
                    "dumps": entry.get("dumps"),
                    "error": entry.get("error"),
                }
                self.cache_hits += 1

    def _check_or_write_meta(self) -> None:
        """校验/写入 sidecar 的模型身份，防止换模型后续跑串结果

        缓存键只覆盖 agent/system/user，不覆盖模型名；同一条消息换一个模型重跑，
        旧结果必须作废。--batch-resume 时身份不一致直接拒绝，而不是静默回放。
        """
        expected = self.profile.fingerprint()
        if self.resume and not self.meta_path.exists() and (self.results or self.jobs_path.exists()):
            raise ValueError(
                f"批量 sidecar {self.cache_path} 有结果但没有模型身份记录（.meta.json），"
                "无法确认是否同一模型；请换一个 --batch-cache，或删掉旧 sidecar 后重跑"
            )
        if self.meta_path.exists():
            try:
                actual = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                actual = None
            if self.resume and actual != expected:
                raise ValueError(
                    f"批量 sidecar {self.cache_path} 的模型身份与本次不一致，拒绝续跑：\n"
                    f"  上次 {actual}\n  本次 {expected}\n"
                    "请换一个 --batch-cache，或删掉旧 sidecar 后重跑"
                )
        if not self.resume or not self.meta_path.exists():
            self.meta_path.write_text(
                json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def _load_jobs(self) -> None:
        if not self.jobs_path.exists():
            return
        try:
            data = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, list):
            return
        for item in data:
            job = BatchJob.from_dict(item)
            if job.batch_id:
                self.jobs[job.batch_id] = job

    def _save_jobs(self) -> None:
        data = [job.to_dict() for job in self.jobs.values()]
        fd, temp_name = tempfile.mkstemp(dir=self.jobs_path.parent, prefix=f".{self.jobs_path.name}.", suffix=".tmp")
        os.close(fd)
        temp = Path(temp_name)
        try:
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(self.jobs_path)
        finally:
            temp.unlink(missing_ok=True)

    def _append_result(self, key: str, dumps: dict | None, error: str | None) -> None:
        with self._lock:
            self.results[key] = {"dumps": dumps, "error": error}
            if dumps is not None:
                self.items_completed += 1
            else:
                self.items_failed += 1
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"key": key, "dumps": dumps, "error": error},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _append_native_results(self, results: dict[str, tuple[dict | None, str | None]]) -> None:
        """每片结果收完立即落盘：成功项入缓存；失败项只在 strict 模式下入缓存

        非 strict 的失败项留给同步回退，不能写进结果缓存（否则续跑会永久复用错误）。
        """
        for key, (dumps, error) in results.items():
            if dumps is not None:
                self._append_result(key, dumps, None)
            elif self.strict_native:
                self._append_result(key, None, error or "批量结果为空")

    # -- 会话状态 ----------------------------------------------------------
    @contextmanager
    def active(self):
        """当前波次：spatial_agents._invoke 从这里取会话并登记/回放请求"""
        token = _CURRENT_BATCH.set(self)
        # 回退模式仍把会话挂上：已拿到的 batch 结果继续回放，缺的项由 _invoke_sync 补，
        # 业务校验（schema 候选、特征闭集等）因此与同步链路完全一致。
        self.collecting = not self.fallback_mode
        try:
            yield self
        finally:
            self.collecting = False
            _CURRENT_BATCH.reset(token)

    def begin_wave(self) -> None:
        """开始一波图重放：清空上一波收集的请求（真正提交时再计一波）"""
        self.pending.clear()

    def lookup(self, agent, content: str) -> tuple[dict | None, str | None] | None:
        """回放一条已有结果；没有返回 None（与「有结果但失败」区分开）"""
        key = item_key(agent.NAME, content, getattr(agent, "system_prompt", ""))
        with self._lock:
            hit = self.results.get(key)
        if hit is None:
            return None
        self.cache_hits += 1
        return hit.get("dumps"), hit.get("error")

    def add(self, agent, content: str) -> str:
        """登记一条待发请求（同一 key 只登记一次）；超整批预算时按异常抛出"""
        key = item_key(agent.NAME, content, getattr(agent, "system_prompt", ""))
        if key in self.results or key in self.pending:
            return key
        # 预算在登记时就扣（候选作用域 + 整批），口径见 BaseAgent.reserve_batch_request
        agent.reserve_batch_request()
        self._reserved.add(key)
        item = BatchItem(
            key=key,
            agent_name=agent.NAME,
            system_prompt=agent.system_prompt,
            content=content,
            response_model=agent.response_model(),
            sync_call=lambda: self._sync_call(agent, content),
            native=bool(agent.supports_batch()),
        )
        self.pending[key] = item
        return key

    def remember(self, agent, content: str, dumps: dict) -> None:
        """把一条**局部同步**的结果写进 sidecar（deep feature 这类不能原生批量的请求）

        它不占 batch 名额，也不进 pending；只是让后续波次和 --batch-resume 能按同一
        key 回放，避免下游 batch 超时后把这条同步请求重跑一遍。
        """
        key = item_key(agent.NAME, content, getattr(agent, "system_prompt", ""))
        if key not in self.results:
            self._append_result(key, dumps, None)

    def consume_reservation(self, agent, content: str) -> bool:
        """登记时扣过预算的请求改走同步首发时，复用那次预留（返回 True）

        只在同步回退路径调用：原生批量的请求由 add 扣一次；回退首发不该再扣一次，
        否则候选预算为 1 时会在「登记后接口 404」这种路径上直接报预算耗尽。
        """
        key = item_key(agent.NAME, content, getattr(agent, "system_prompt", ""))
        if key in self._reserved:
            self._reserved.discard(key)
            return True
        return False

    @staticmethod
    def _sync_call(agent, content: str) -> tuple[dict | None, str | None]:
        try:
            verdict = agent._invoke_sync(content, check_budget=False)
            return verdict.model_dump(), None
        except Exception as exc:  # noqa: BLE001 —— 单条失败只影响这一条
            return None, f"{exc!r}"

    # -- 执行 --------------------------------------------------------------
    def execute(self, items: list[BatchItem] | None = None, *, fallback_workers: int = 1) -> int:
        """把一批请求送出去并收结果（原生优先，失败/不支持则同步回退）

        items=None 是图波次（结果由 run_waves 重放消费）；给了 items 是 all_in_one 的直接
        提交。两者的回退方式不同：图波次把 fallback_mode 置位，下一波重放时缺的项由
        `_invoke_sync` 补（仍经过 agent.run 的业务校验）；直接提交没有下一波，立即用
        `sync_call` 逐条补齐。
        """
        direct = items is not None
        if items is None:
            items = list(self.pending.values())
            self.pending.clear()
        else:
            items = list(items)
        if not items:
            return 0
        self.waves += 1
        for item in items:
            if item.key not in self._seen:
                self._seen.add(item.key)
                self.items_collected += 1
        # 先把上次超时遗留、而本波次又要用的 batch 收掉，避免重复提交
        self._resume_jobs(items)
        todo = [item for item in items if item.key not in self.results]
        if not todo:
            return 0
        # 有只能同步跑的项（deep feature 的工具回合）时，整个波次回退，避免一半批一半同步
        if any(not item.native for item in todo):
            self._enter_fallback("本波次含只能同步执行的请求（deep feature）")
        if self.client is None:
            self._enter_fallback("没有可用的原生批量客户端")
        native = [item for item in todo if item.native and not self.fallback_mode]
        results: dict[str, tuple[dict | None, str | None]] | None = None
        if native and self.client is not None:
            batches_before = self.batches
            try:
                results = self.client.run(
                    native,
                    on_job=self._job_submitted,
                    on_status=self._job_status,
                    on_results=self._append_native_results,
                )
            except BatchTimeout:
                raise
            except BatchUnavailable as exc:
                if self.batches > batches_before:
                    raise BatchTimeout(
                        f"部分 batch 已受理，但后续提交遇到不可用接口：{exc!r}；"
                        "请加 --batch-resume 继续收，避免重复计费",
                    ) from exc
                self._enter_fallback(f"原生批量入口不可用（{exc}）")
            except Exception as exc:  # noqa: BLE001
                if self.batches > batches_before:
                    # 已经受理过至少一个 batch：远端可能还在跑，不能再同步重发。
                    raise BatchTimeout(
                        f"批量提交后发生故障：{exc!r}；已受理的 batch 记录在 {self.jobs_path}，"
                        "请加 --batch-resume 继续收，避免重复计费",
                    ) from exc
                self._enter_fallback(f"原生批量调用失败（{exc!r}）")
        if results is not None:
            retry_keys: list[str] = []
            completed_any = False
            for item in native:
                if item.key not in results:
                    if self.strict_native:
                        self._append_result(item.key, None, "原生批量结果缺少这一项")
                    else:
                        retry_keys.append(item.key)
                    continue
                dumps, error = results[item.key]
                if dumps is not None:
                    completed_any = True
                    continue  # 成功结果已由 on_results 分片落盘
                if self.strict_native:
                    continue  # 失败结果已由 on_results 以 error 落盘
                # 单项失败（限流、过期、批次级失败等）再给同步链路一次机会；
                # 不把失败写进结果缓存，否则续跑会永久复用这个错误。
                retry_keys.append(item.key)
            if retry_keys:
                self._enter_fallback(
                    f"原生批量有 {len(retry_keys)} 项失败或缺失，改走同步补一次"
                )
                self.mode = "mixed" if completed_any else "fallback"
        if not direct:
            # 图波次：把缺的项留给下一波重放时的 _invoke_sync（业务校验仍走 run/_ask）
            if self.fallback_mode:
                self.fallback_items += len([item for item in todo if item.key not in self.results])
            return len(todo)
        fallback = [item for item in todo if item.key not in self.results]
        if fallback:
            # 同一 key 只跑一次：重复提交既多花钱，也可能让 custom_id 撞车
            unique_fallback: dict[str, BatchItem] = {item.key: item for item in fallback}
            self.fallback_items += len(unique_fallback)
            with ThreadPoolExecutor(max_workers=max(1, int(fallback_workers))) as pool:
                futures = {
                    key: pool.submit(item.sync_call)
                    for key, item in unique_fallback.items()
                    if item.sync_call is not None
                }
                for key, item in unique_fallback.items():
                    future = futures.get(key)
                    if future is None:
                        self._append_result(key, None, "该请求没有同步回退实现")
                        continue
                    dumps, error = future.result()
                    self._append_result(key, dumps, error)
        return len(todo)

    def _enter_fallback(self, reason: str) -> None:
        """切到同步回退：下一波重放不再收集请求，缺项由 _invoke_sync 补"""
        if self.strict_native:
            raise RuntimeError(f"--batch native 要求原生批量，但不能继续：{reason}")
        if not self.fallback_mode:
            self.logger(f"警告：{reason}；本波次起回退同步执行")
        self.fallback_mode = True
        self.disabled_reason = reason
        if self.mode == "native":
            self.mode = "fallback"

    def submit_direct(self, items: list[BatchItem], *, fallback_workers: int = 1) -> int:
        """all_in_one 用：请求已经由调用方拼好，直接送这一批"""
        return self.execute(items, fallback_workers=fallback_workers)

    def _job_submitted(self, job: BatchJob) -> None:
        self.jobs[job.batch_id] = job
        self.batches += 1
        # 这些项已经真的发出去了：预算算过一次；之后若失败改同步重试，应再算一次
        for key in job.keys:
            self._reserved.discard(key)
        self._save_jobs()

    def _job_status(self, job: BatchJob) -> None:
        self.jobs[job.batch_id] = job
        self._save_jobs()

    def _resume_jobs(self, items: list[BatchItem]) -> None:
        """收掉上次留下、仍未终结的 batch（只处理本波次还会用到的 key）"""
        if self.client is None:
            return
        wanted = {item.key for item in items}
        by_key = {item.key: item for item in items}
        for job in list(self.jobs.values()):
            needed = set(job.keys) & wanted
            if not needed:
                continue
            if all(key in self.results for key in needed):
                continue
            if job.status == "cancelling":
                # 还在取消中，任务可能仍在跑；先不动，避免与远端抢着处理同一批
                continue
            # completed/failed/expired/cancelled 都先收一次：expired/partial 也可能
            # 带回了部分成功结果；completed 更要再 GET 一次拿 output_file_id/results。
            try:
                job = self.client.wait(job)
            except BatchTimeout:
                raise
            raw = self.client.collect(job)
            for key, (body, error) in raw.items():
                item = by_key.get(key)
                if item is None or key in self.results:
                    continue
                if body is None:
                    # 失败项不写缓存，留给本轮重新提交/同步回退；strict 才如实记错误
                    if self.strict_native:
                        self._append_result(key, None, error or "恢复的 batch 没有返回这一项")
                    continue
                try:
                    self._append_result(key, item.parse_response(body), None)
                except Exception as exc:  # noqa: BLE001
                    if self.strict_native:
                        self._append_result(key, None, f"批量结果解析失败：{exc!r}")
                    # 非 strict：解析失败的项同样留给同步回退
            self._job_status(job)

    # -- 汇总 --------------------------------------------------------------
    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "provider": self.profile.provider,
            "model": self.profile.model,
            "protocol": self.profile.protocol,
            "native_enabled": bool(self.profile.enabled),
            "endpoint": f"{self.profile.base_url}/batches" if self.profile.enabled else "",
            "disabled_reason": self.disabled_reason,
            "waves": self.waves,
            "items_collected": self.items_collected,
            "items_completed": self.items_completed,
            "items_failed": self.items_failed,
            "fallback_items": self.fallback_items,
            "cache_hits": self.cache_hits,
            "batches": self.batches,
            "cache_file": str(self.cache_path),
            "jobs_file": str(self.jobs_path),
        }


def run_waves(
    states: list[dict],
    invoke: Callable[[dict], dict],
    session: BatchSession,
    *,
    fallback_workers: int = 1,
    max_waves: int = 64,
    on_complete: Callable[[dict], None] | None = None,
) -> list[dict]:
    """按波次重放图，直到所有状态走完

    Args:
        states: 图的初始状态（每个候选一条）
        invoke: 跑一条状态的可调用对象（通常是 graph.invoke 的薄包装）
        session: 批会话
        on_complete: 状态真正走完时的回调（save 节点已经落盘，一般不需要）
        max_waves: 防御上限；每一波都必须有进展，否则抛错而不是死循环

    Returns:
        走完的最终状态列表（顺序可能因完成波次不同而变；落盘顺序由 ResultWriter 重排）
    """
    pending = list(states)
    completed: list[dict] = []
    while pending:
        session.begin_wave()
        remaining: list[dict] = []
        with session.active():
            for state in pending:
                try:
                    final = invoke(state)
                except BatchPending:
                    remaining.append(state)
                else:
                    completed.append(final)
                    if on_complete is not None:
                        on_complete(final)
        if not remaining:
            break
        if not session.pending:
            raise RuntimeError(
                "批处理波次没有收集到任何请求，但仍有状态未走完："
                "通常是 agent 抛了非 BatchPending 的异常，或结果键不稳定"
            )
        session.execute(fallback_workers=fallback_workers)
        pending = remaining
        if session.waves >= max_waves:
            raise RuntimeError(f"批处理超过 {max_waves} 波仍未收敛，停止以免死循环")
    return completed
