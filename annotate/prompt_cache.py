# encoding: utf8

"""提示词缓存：把 user message 切成「稳定前缀」与「本条动态内容」，按模型家族加显式断点。

为什么要有这一层：
    各 agent 拼出来的 user message 既有跨记录稳定的部分（按关注词选出的示例、图式候选表），
    也有每条语料特有的部分（关注词下标、句子、上游判断）。供应商的提示词缓存只复用**从第 0
    个 token 起完全相同**的前缀，因此稳定部分必须排在前面，README 里的「静态→动态」顺序才
    真的成立。

    Claude 与千问（ZenMux / 千问官方显式缓存）需要在稳定前缀末尾显式标注缓存断点
    （`cache_control`）。其他模型要么自动缓存（OpenAI / Gemini / DeepSeek / Kimi / GLM），
    要么完全不支持这个字段；字段不能无差别发给所有模型。本模块按实际送出的模型标识
    决定是否注入，非显式家族一律返回普通字符串。

怎么判断模型家族：
    只看真正发给网关的模型标识（`batch.model` 覆盖值优先，其次 `model`、`model_name`），
    不看配置里的 `provider`：
    本项目的 ZenMux 模型 `provider` 都写 `"openai"`，那只是客户端协议，底层可能是 Claude、
    GPT 或 Gemini。模型配置里可写 `"prompt_cache"` 关闭注入：
        "auto"（默认）   按实际送出的模型标识判断；
        "explicit"        显式启用（用于已验证支持、但不在自动允许列表里的模型）；
        "off"/"implicit" 不注入任何缓存字段；
        false            同上。
    自动识别只覆盖 Claude 与 config.PROMPT_CACHE_QWEN_EXPLICIT_MODELS 里的千问
    模型；未列入的模型可显式写 ``"prompt_cache": "explicit"``，但不按「模型名含
    qwen」全家族放行。全局开关见 config.PROMPT_CACHE_ENABLED，它排在单个模型配置
    之前。

为什么不用字符串分隔符：
    PromptParts 直接是 str 的子类：既能在任何期望字符串的地方（本地结果缓存键、Batch 键、
    --print-prompt、日志）原样工作，又带着 stable/dynamic 两段边界。这样「拼给模型的文本」与
    「切分依据」永远同源，不会出现两份实现漂移、也不会因为语料里恰好含分隔符而切错。
"""

from __future__ import annotations

import hashlib
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import config

CACHE_CONTROL: dict = {"type": "ephemeral"}
"""Claude / Qwen 5 分钟缓存的断点标记（与 ZenMux 文档一致）"""

EXPLICIT_FAMILIES = frozenset({"anthropic", "qwen", "explicit"})
"""需要显式 `cache_control` 的家族；其余家族保持纯字符串，不携带缓存字段。"""

_DISABLED_VALUES = frozenset({"off", "none", "disabled", "implicit"})
"""显式关闭注入的取值；`implicit` 表示交给供应商自动缓存，也不注入断点"""

_FAMILY_CACHE: dict[tuple[str, bool], str] = {}
_FAMILY_LOCK = threading.Lock()


class PromptParts(str):
    """完整 user message（str）＋稳定/动态两段边界

    构造后就是 `stable + dynamic` 的普通字符串：`==`、`hash`、`json.dumps`、`.endswith()`
    都与手拼的字符串完全一致，因此既有的本地结果缓存与 Batch 结果缓存键不需要改口径。
    `stable` 为空表示这条消息没有可复用的前缀（此时不注入断点）。
    """

    stable: str
    dynamic: str

    def __new__(cls, stable: str = "", dynamic: str = ""):
        obj = super().__new__(cls, f"{stable}{dynamic}")
        obj.stable = stable
        obj.dynamic = dynamic
        return obj

    def parts(self) -> tuple[str, str]:
        return self.stable, self.dynamic


def _model_id(model_cfg: dict, model_override: str | None = None) -> str:
    """真正发给网关的那个模型标识：覆盖值 > model > model_name

    只取**一个**而不是把三个拼起来：同步配置写 claude、batch.model 覆盖成 gpt 时，拼起来
    仍会命中 "claude"，把 `cache_control` 发给一个不收这个字段的模型。判断必须按实际送达的
    那个标识来。
    """
    for value in (model_override, model_cfg.get("model"), model_cfg.get("model_name")):
        if value:
            return str(value).lower()
    return ""


def is_qwen_explicit_model(model_id: str) -> bool:
    """模型标识是否属于千问显式缓存白名单（按 ``/`` 或 ``:`` 后的模型名匹配）。

    白名单集中在 ``config.PROMPT_CACHE_QWEN_EXPLICIT_MODELS``，便于手动维护：
    普通条目匹配本名或「本名-后缀」；末尾带 ``*`` 的条目按前缀匹配（如
    ``qwen4*`` 覆盖 qwen4-max、qwen4-max-preview、qwen4.1-max 等）。

    不直接用子串：``local-qwen3.8-max`` 这类本地方案也可能带 cache 字段而被网关拒绝。
    白名单只覆盖官方明确支持显式缓存的模型；新模型用 ``prompt_cache: explicit``
    或把本名/前缀加进白名单。
    """
    candidate = str(model_id or "").lower().rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    for marker in config.PROMPT_CACHE_QWEN_EXPLICIT_MODELS:
        marker = str(marker).lower()
        if marker.endswith("*"):
            prefix = marker[:-1]
            if prefix and candidate.startswith(prefix):
                return True
        elif candidate == marker or candidate.startswith(marker + "-"):
            return True
    return False


def family_for_config(model_cfg: dict, model_override: str | None = None) -> str:
    """按模型配置判断缓存家族：``"anthropic"`` / ``"qwen"`` / ``"explicit"`` / ``"none"``

    ``prompt_cache`` 可写：

        ``"auto"``（默认）  按实际送出的模型标识自动识别；
        ``"explicit"``      显式启用；用于已验证支持 `cache_control`、但不在自动
                            允许列表里的新模型；
        ``"off"``/``"implicit"``/``false``  不注入任何断点。

    自动识别只覆盖 Claude 与 config.PROMPT_CACHE_QWEN_EXPLICIT_MODELS 里的千问
    模型；不按 ``provider`` 猜，也不按「模型名含 qwen」放行——本地 Qwen 与未验证
    端点不保证接受该字段。
    """
    if not config.PROMPT_CACHE_ENABLED:
        return "none"
    override = model_cfg.get("prompt_cache")
    if override is False:
        return "none"
    if isinstance(override, str):
        value = override.strip().lower()
        if value in _DISABLED_VALUES:
            return "none"
        if value == "explicit":
            return "explicit"
    model_id = _model_id(model_cfg, model_override)
    if "claude" in model_id:
        return "anthropic"
    if is_qwen_explicit_model(model_id):
        return "qwen"
    return "none"


def is_explicit(family: str) -> bool:
    """该家族是否需要显式 `cache_control` 断点。"""
    return family in EXPLICIT_FAMILIES


def family_for_model_key(model_key: str, *, disable_thinking: bool = False) -> str:
    """按 api_keys.json 的配置名判断缓存家族（结果按配置名 + 是否关思考缓存）

    ``disable_thinking`` 为 True 时，它的 ``model`` 会在同步链路中覆盖实际送出的
    模型标识，因此按覆盖后的标识判断；否则基础模型是 Claude、实际换成非显式缓存
    模型时，请求仍会带上 ``cache_control`` 而被网关拒绝。两个开关状态分开缓存，
    避免结构化输出与非结构化输出共用同一结论。

    读配置失败时返回 ``"none"`` 而不是抛错：缓存只是省钱手段，不能因为一份可选配置
    读不到就让整批标注起不来。真正的配置错误会在 ``agents.request_policy`` 那一步先报。
    """
    cache_key = (model_key, bool(disable_thinking))
    with _FAMILY_LOCK:
        cached = _FAMILY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        import agents as agent_factory

        cfg = agent_factory.load_model_config(model_key)
        override = None
        extra = cfg.get("disable_thinking") if isinstance(cfg, dict) else None
        if disable_thinking and isinstance(extra, dict):
            # langchain 的 extra_body 会合并进请求体，嵌套的 model 同样能覆盖实际模型；
            # 它比顶层同名字段更接近最终请求，优先采用。
            candidates = []
            body = extra.get("extra_body")
            if isinstance(body, dict):
                candidates.append(body.get("model"))
            candidates.append(extra.get("model"))
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    override = candidate.strip()
                    break
        family = family_for_config(cfg, model_override=override)
    except Exception:  # noqa: BLE001 —— 见 docstring
        family = "none"
    with _FAMILY_LOCK:
        _FAMILY_CACHE[cache_key] = family
    return family


def user_content(content, family: str):
    """构造发给模型的 user content：显式缓存家族返回带断点的内容块，其余返回字符串

    断点只加在稳定块末尾，动态块永不带缓存字段。稳定块为空时退回普通字符串——
    没有可复用的前缀却标一个断点，只会让供应商在每次请求上做一次无用的缓存查找。
    """
    text = str(content)
    if not is_explicit(family) or not isinstance(content, PromptParts):
        return text
    stable, dynamic = content.parts()
    if not stable.strip():
        return text
    blocks = [{"type": "text", "text": stable, "cache_control": dict(CACHE_CONTROL)}]
    if dynamic.strip():
        blocks.append({"type": "text", "text": dynamic})
    return blocks


def system_blocks(system_prompt: str, family: str):
    """构造 system content：显式缓存家族返回带断点的文本块，其余返回原字符串

    system 与 user 稳定块是两个更新频率不同的断点：改示例组时 system 缓存仍然有效，
    改 system 时示例缓存作废。Anthropic 与 ZenMux 都明确说明断点本身不额外收费，
    费用只按实际写入/读取的内容计算（每个请求最多 4 个断点）。
    """
    if not is_explicit(family) or not isinstance(system_prompt, str) or not system_prompt.strip():
        return system_prompt
    return [{"type": "text", "text": system_prompt, "cache_control": dict(CACHE_CONTROL)}]


def system_message(system_prompt: str, family: str):
    """给 ``create_agent(system_prompt=...)`` 用的 system 参数（str 或 SystemMessage）"""
    blocks = system_blocks(system_prompt, family)
    if isinstance(blocks, list):
        from langchain_core.messages import SystemMessage

        return SystemMessage(content=blocks)
    return blocks


def warmup_key(model_key: str, system_prompt: str, content, family: str) -> str:
    """同一模型、同一 system、同一稳定前缀的调度键；无可复用前缀时返回空串

    供应商的缓存只在「从第 0 个 token 起逐字节相同」时命中，且缓存写入发生在
    响应之后。并发首问如果一起发出去，会全部未命中并各写一份缓存。这个键只用于
    让同一前缀的**首个真实请求**先完成，再放行其余请求；它不是结果缓存键，也不
    参与业务去重。

    普通字符串（例如 evaluate step 0 的 SynonymAgent）没有 PromptParts，此时退回
    system-only 键：同一模型/同一 system 仍共享 `tools → system` 前缀。system 也
    为空才返回空串，避免把所有无关请求串行化。
    """
    if not is_explicit(family):
        # 关闭显式缓存的模型没有可写的远端前缀；退回 system-only 键只会把同一
        # system 下互不相关的请求串行化，且 leader 失败时还会拖住其他 worker。
        return ""
    stable = ""
    if isinstance(content, PromptParts):
        stable = content.stable if content.stable.strip() else ""
    if stable:
        scope = "user\0" + stable
    elif str(system_prompt or "").strip():
        scope = "system"
    else:
        return ""
    digest = hashlib.sha256(
        f"{model_key}\0{family}\0{system_prompt}\0{scope}".encode("utf-8")
    ).hexdigest()
    return digest


@dataclass
class _PrefixState:
    """一个稳定前缀的冷启动状态。event 未置位 = 首个请求尚未完成。"""

    event: threading.Event
    warmed_at: float = 0.0
    success: bool = False
    failed_until: float = 0.0


class WarmupGate:
    """同前缀冷启动门控：leader 先发，等响应回来后再放行其余请求。

    只影响**首次**发同一前缀的并发请求，不把整批请求永久串行化。缓存 TTL 过后
    重新发生一次冷启动，避免把「曾经写过缓存」误当成「现在仍然命中」。leader
    失败后进入短暂冷却（最多 10 秒）：冷却期内等待者直接放行，避免端点级故障时
    把 --workers N 退化成逐个串行重试；冷却结束后的第一个请求再尝试做 leader。
    本地结果命中不会经过这里，因此不会刷新远端 TTL。
    """

    def __init__(self, ttl_seconds: float = 300.0, clock=time.monotonic):
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, _PrefixState] = {}
        self.waits = 0

    @contextmanager
    def acquire(self, key: str):
        if not key:
            yield
            return
        while True:
            leader = False
            ready = False
            with self._lock:
                state = self._states.get(key)
                now = self._clock()
                if state is None or (
                    state.event.is_set()
                    and (
                        now - state.warmed_at >= self.ttl_seconds
                        or (not state.success and now >= state.failed_until)
                    )
                ):
                    state = _PrefixState(threading.Event())
                    self._states[key] = state
                    leader = True
                elif state.event.is_set():
                    # leader 刚失败时不要立刻把等待者逐个变成新 leader：端点级故障
                    # （鉴权/宕机/持续 429）下那会把 --workers N 退化成串行重试。
                    # 在短暂冷却期内直接放行，让请求并发失败或并发成功；冷却后再试。
                    ready = state.success or now < state.failed_until
                else:
                    self.waits += 1
            if ready:
                yield
                return
            if leader:
                try:
                    yield
                except BaseException:
                    with self._lock:
                        state.success = False
                        state.warmed_at = self._clock()
                        cooldown = max(1.0, min(self.ttl_seconds, 10.0))
                        state.failed_until = state.warmed_at + cooldown
                        state.event.set()
                    raise
                else:
                    with self._lock:
                        state.success = True
                        state.warmed_at = self._clock()
                        state.event.set()
                    return
            state.event.wait()

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self.waits = 0


_GATE = WarmupGate(config.PROMPT_CACHE_WARMUP_TTL_SECONDS)


@contextmanager
def warmup_gate(key: str):
    """进程内共享的前缀冷启动门控（见 ``WarmupGate``）。"""
    with _GATE.acquire(key):
        yield
