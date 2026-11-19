# encoding: utf8

"""模型用量归一化与累计：把各家的 usage 形状折成同一套 token 账。

为什么要单独一层：
    同步链路看到的是 LangChain 的 ``usage_metadata``，原生 Batch 看到的是网关返回的原始
    ``usage``，两者字段名不同；同一家供应商还会用不同键表达缓存（OpenAI 的
    ``prompt_tokens_details.cached_tokens``、DeepSeek 的 ``prompt_cache_hit_tokens``、
    Anthropic 的 ``cache_read_input_tokens``）。把口径写在每个 agent / 每条批量结果里，
    必然出现「有的算了缓存写、有的没算」。

三个原则：
    1. **缺失不等于零**：没拿到 usage 记 ``usage_missing``，不猜 token；拿到了但没有缓存
       明细，只说明「这次没观察到缓存读写」，不能据此断言供应商没收费。
    2. **不重复相加**：``reasoning_tokens`` 通常是输出 token 的子集，只单列、不再加进
       ``output_tokens``；``usage_metadata`` 与 ``response_metadata`` 同时存在时优先前者，
       只在缓存明细缺失时用后者补，绝不两份相加。
    3. **只统计本进程真正发出去并拿到回复的调用**：本地结果复用与 Batch 回放不产生新用量，
       由调用方决定不记；这样 summary 里的 token 和「本次新增花费」才对得上。
"""

from __future__ import annotations

TOKEN_FIELDS = (
    "input_tokens",
    "uncached_input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)
"""可累加的 token 字段；``uncached_input_tokens`` 是总输入里未命中缓存的那部分"""

COUNTER_FIELDS = TOKEN_FIELDS + (
    "usage_observed",
    "usage_missing",
    "cache_hit_requests",
)
"""进 _stats / summary 的全部计数字段"""


def empty_tally() -> dict:
    """一份全新的空账本"""
    return {name: 0 for name in COUNTER_FIELDS}


def _nested(mapping, *keys):
    """安全地取嵌套字典里的整数（缺任何一层都返回 None）"""
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _int(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(int(value), 0)


def _blank(fields: dict) -> dict | None:
    if all(value is None for value in fields.values()):
        return None
    return {name: _int(value) for name, value in fields.items()}


def _finish(input_total: int, uncached: int, cache_read: int, cache_write: int,
            output: int, reasoning: int) -> dict:
    return {
        "input_tokens": max(input_total, 0),
        "uncached_input_tokens": max(uncached, 0),
        "cache_read_tokens": max(cache_read, 0),
        "cache_write_tokens": max(cache_write, 0),
        "output_tokens": max(output, 0),
        "reasoning_tokens": max(reasoning, 0),
        "cache_hit": max(cache_read, 0) > 0,
    }


def normalize_metadata(meta) -> dict | None:
    """归一 LangChain 的 ``usage_metadata``

    LangChain 把 total input 统一放在 ``input_tokens``，缓存明细在
    ``input_token_details.cache_read`` / ``cache_creation``；Anthropic 的
    ``input_tokens`` 在这里已经是「总输入」，与原生 Messages API 的「未命中部分」不同。
    """
    if not isinstance(meta, dict):
        return None
    input_total = meta.get("input_tokens")
    output = meta.get("output_tokens")
    details = meta.get("input_token_details") or {}
    cache_read = details.get("cache_read") if isinstance(details, dict) else None
    cache_write = details.get("cache_creation") if isinstance(details, dict) else None
    out_details = meta.get("output_token_details") or {}
    reasoning = out_details.get("reasoning") if isinstance(out_details, dict) else None
    base = _blank({
        "input_total": input_total, "output": output,
        "cache_read": cache_read, "cache_write": cache_write, "reasoning": reasoning,
    })
    if base is None:
        return None
    total = base["input_total"]
    read = base["cache_read"]
    write = base["cache_write"]
    return _finish(total, max(total - read - write, 0), read, write,
                   base["output"], base["reasoning"])


def normalize_usage(usage) -> dict | None:
    """归一网关返回的原始 ``usage`` 对象

    兼容三种输入口径：
        OpenAI / ZenMux：``prompt_tokens`` 是总输入，缓存明细在 ``prompt_tokens_details``；
        DeepSeek：        额外给 ``prompt_cache_hit_tokens``；
        Anthropic 原生：  ``input_tokens`` 只算未命中部分，命中/写入另列。
    """
    if not isinstance(usage, dict):
        return None

    has_prompt_total = usage.get("prompt_tokens") is not None
    prompt = usage.get("prompt_tokens")
    if prompt is None:
        prompt = usage.get("input_tokens")
    output = usage.get("completion_tokens")
    if output is None:
        output = usage.get("output_tokens")

    cache_read = (
        _nested(usage, "prompt_tokens_details", "cached_tokens")
        or _nested(usage, "input_tokens_details", "cached_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or usage.get("cache_read_input_tokens")
    )
    cache_write = (
        _nested(usage, "prompt_tokens_details", "cache_write_tokens")
        or _nested(usage, "input_tokens_details", "cache_write_tokens")
        or usage.get("prompt_cache_write_tokens")
        or usage.get("cache_creation_input_tokens")
    )
    reasoning = (
        _nested(usage, "completion_tokens_details", "reasoning_tokens")
        or _nested(usage, "output_tokens_details", "reasoning_tokens")
        or _nested(usage, "output_tokens_details", "thinking_tokens")
        or usage.get("thoughts_token_count")
        or usage.get("thinking_tokens")
    )
    base = _blank({
        "prompt": prompt, "output": output, "cache_read": cache_read,
        "cache_write": cache_write, "reasoning": reasoning,
    })
    if base is None:
        return None

    read = base["cache_read"]
    write = base["cache_write"]
    if has_prompt_total:
        total = base["prompt"]
        uncached = max(total - read - write, 0)
    else:
        uncached = base["prompt"]
        total = uncached + read + write
    return _finish(total, uncached, read, write, base["output"], base["reasoning"])


def from_chat_message(message) -> dict | None:
    """从一条 LangChain 消息取用量：``usage_metadata`` 优先，原始 usage 只补缓存明细

    部分网关只把缓存字段放在 ``response_metadata.token_usage``，而 LangChain 的
    ``usage_metadata`` 仍是全零。此时用原始 usage 的缓存明细补齐，但输入/输出总数不重复相加。
    """
    if message is None:
        return None
    meta = getattr(message, "usage_metadata", None)
    normalized = normalize_metadata(meta) if isinstance(meta, dict) else None
    response_metadata = getattr(message, "response_metadata", None) or {}
    raw = None
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage"):
            candidate = response_metadata.get(key)
            if isinstance(candidate, dict):
                raw = normalize_usage(candidate)
                if raw is not None:
                    break
    if normalized is None:
        return raw
    if raw is not None:
        for name in ("cache_read_tokens", "cache_write_tokens", "reasoning_tokens"):
            if not normalized.get(name) and raw.get(name):
                normalized[name] = raw[name]
        normalized["uncached_input_tokens"] = max(
            normalized["input_tokens"] - normalized["cache_read_tokens"]
            - normalized["cache_write_tokens"], 0)
        normalized["cache_hit"] = normalized["cache_read_tokens"] > 0
    return normalized


def from_llm_result(result) -> dict | None:
    """从回调的 ``LLMResult`` 取一次模型调用的用量（多次生成则相加）

    没有 ``usage_metadata`` 时退回 ``llm_output.token_usage``；都没有返回 None，
    由调用方记一条 ``usage_missing``。
    """
    if result is None:
        return None
    found = []
    for batch in getattr(result, "generations", None) or []:
        for generation in batch or []:
            usage = from_chat_message(getattr(generation, "message", None))
            if usage is not None:
                found.append(usage)
    if not found:
        llm_output = getattr(result, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            for key in ("token_usage", "usage"):
                usage = normalize_usage(llm_output.get(key))
                if usage is not None:
                    return usage
        return None
    merged = merge(found)
    return merged


def merge(usages: list[dict]) -> dict:
    """把同一逻辑调用里的多份用量相加（返回形状与 normalize_* 一致）"""
    merged = {name: 0 for name in TOKEN_FIELDS}
    merged["cache_hit"] = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for name in TOKEN_FIELDS:
            merged[name] += _int(usage.get(name))
        merged["cache_hit"] = merged["cache_hit"] or bool(usage.get("cache_hit"))
    return merged


def accumulate(tally: dict, usage: dict | None) -> None:
    """把一次调用的用量记进累计账（就地修改；缺失记 usage_missing）"""
    if usage is None:
        tally["usage_missing"] = tally.get("usage_missing", 0) + 1
        return
    tally["usage_observed"] = tally.get("usage_observed", 0) + 1
    for name in TOKEN_FIELDS:
        tally[name] = tally.get(name, 0) + _int(usage.get(name))
    if usage.get("cache_hit"):
        tally["cache_hit_requests"] = tally.get("cache_hit_requests", 0) + 1


def totals(items) -> dict:
    """把多个 agent 的 stats 合并成一份总量，并给出缓存读占输入的比例"""
    out = empty_tally()
    for item in items:
        if not isinstance(item, dict):
            continue
        for name in COUNTER_FIELDS:
            out[name] += _int(item.get(name))
    out["cache_hit_rate"] = (
        round(out["cache_read_tokens"] / out["input_tokens"], 4) if out["input_tokens"] else 0.0
    )
    return out
