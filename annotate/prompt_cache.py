# encoding: utf8

"""提示词缓存：把 user message 切成「稳定前缀」与「本条动态内容」，并只为 Claude 加显式断点。

为什么要有这一层：
    各 agent 拼出来的 user message 既有跨记录稳定的部分（按关注词选出的示例、图式候选表），
    也有每条语料特有的部分（关注词下标、句子、上游判断）。供应商的提示词缓存只复用**从第 0
    个 token 起完全相同**的前缀，因此稳定部分必须排在前面，README 里的「静态→动态」顺序才
    真的成立。

    Claude 需要在稳定前缀末尾显式标注缓存断点（`cache_control`）。其他模型要么自动缓存，
    要么完全不支持这个字段：DeepSeek 的 Anthropic 兼容端点会**忽略** `cache_control`，
    Gemini 4 系列不支持显式缓存。所以字段绝对不能无差别地发给所有模型——本模块按模型家族
    决定是否注入，非 Claude 一律返回普通字符串。

怎么判断模型家族：
    只看真正发给网关的模型标识（`batch.model` 覆盖值优先，其次 `model`、`model_name`），
    不看配置里的 `provider`：
    本项目的 ZenMux 模型 `provider` 都写 `"openai"`，那只是客户端协议，底层可能是 Claude、
    GPT 或 Gemini。模型配置里可写 `"prompt_cache"` 关闭注入：
        "auto"（默认）   按实际送出的模型标识判断；
        "off"/"implicit" 不注入任何缓存字段；
        false            同上。
    不提供「强制注入」档位：把 `cache_control` 发给非 Claude 模型没有收益，还可能被网关拒绝。
    全局开关见 config.PROMPT_CACHE_ENABLED，它排在单个模型配置之前。

为什么不用字符串分隔符：
    PromptParts 直接是 str 的子类：既能在任何期望字符串的地方（本地结果缓存键、Batch 键、
    --print-prompt、日志）原样工作，又带着 stable/dynamic 两段边界。这样「拼给模型的文本」与
    「切分依据」永远同源，不会出现两份实现漂移、也不会因为语料里恰好含分隔符而切错。
"""

from __future__ import annotations

import threading

import config

CACHE_CONTROL: dict = {"type": "ephemeral"}
"""Claude 5 分钟缓存的断点标记（与 ZenMux / Anthropic 文档一致）"""

_DISABLED_VALUES = frozenset({"off", "none", "disabled", "implicit"})
"""显式关闭注入的取值；`implicit` 表示交给供应商自动缓存，也不注入断点"""

_FAMILY_CACHE: dict[str, str] = {}
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


def family_for_config(model_cfg: dict, model_override: str | None = None) -> str:
    """按模型配置判断缓存家族：``"anthropic"`` 或 ``"none"``

    模型配置里的 ``prompt_cache`` 只用于**关闭**（``off``/``implicit``/``false``）；不提供
    「强制注入」档位——把 `cache_control` 发给非 Claude 模型既没有收益，还可能被网关拒绝，
    要支持同构网关请让模型标识里带上 claude，或另开显式能力字段。
    """
    if not config.PROMPT_CACHE_ENABLED:
        return "none"
    override = model_cfg.get("prompt_cache")
    if override is False:
        return "none"
    if isinstance(override, str) and override.strip().lower() in _DISABLED_VALUES:
        return "none"
    return "anthropic" if "claude" in _model_id(model_cfg, model_override) else "none"


def family_for_model_key(model_key: str) -> str:
    """按 api_keys.json 的配置名判断缓存家族（结果按配置名缓存）

    读配置失败时返回 ``"none"`` 而不是抛错：缓存只是省钱手段，不能因为一份可选配置
    读不到就让整批标注起不来。真正的配置错误会在 ``agents.request_policy`` 那一步先报。
    """
    with _FAMILY_LOCK:
        cached = _FAMILY_CACHE.get(model_key)
    if cached is not None:
        return cached
    try:
        import agents as agent_factory

        family = family_for_config(agent_factory.load_model_config(model_key))
    except Exception:  # noqa: BLE001 —— 见 docstring
        family = "none"
    with _FAMILY_LOCK:
        _FAMILY_CACHE[model_key] = family
    return family


def user_content(content, family: str):
    """构造发给模型的 user content：Claude 返回带断点的内容块，其余返回字符串

    断点只加在稳定块末尾，动态块永不带缓存字段。稳定块为空时退回普通字符串——
    没有可复用的前缀却标一个断点，只会让供应商在每次请求上做一次无用的缓存查找。
    """
    text = str(content)
    if family != "anthropic" or not isinstance(content, PromptParts):
        return text
    stable, dynamic = content.parts()
    if not stable.strip():
        return text
    blocks = [{"type": "text", "text": stable, "cache_control": dict(CACHE_CONTROL)}]
    if dynamic.strip():
        blocks.append({"type": "text", "text": dynamic})
    return blocks


def system_blocks(system_prompt: str, family: str):
    """构造 system content：Claude 返回带断点的文本块，其余返回原字符串

    system 与 user 稳定块是两个更新频率不同的断点：改示例组时 system 缓存仍然有效，
    改 system 时示例缓存作废。Anthropic 与 ZenMux 都明确说明断点本身不额外收费，
    费用只按实际写入/读取的内容计算（每个请求最多 4 个断点）。
    """
    if family != "anthropic" or not isinstance(system_prompt, str) or not system_prompt.strip():
        return system_prompt
    return [{"type": "text", "text": system_prompt, "cache_control": dict(CACHE_CONTROL)}]


def system_message(system_prompt: str, family: str):
    """给 ``create_agent(system_prompt=...)`` 用的 system 参数（str 或 SystemMessage）"""
    blocks = system_blocks(system_prompt, family)
    if isinstance(blocks, list):
        from langchain_core.messages import SystemMessage

        return SystemMessage(content=blocks)
    return blocks
