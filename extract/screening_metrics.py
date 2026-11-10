# encoding: utf8

"""LLM 筛选的性能指标：延迟摘要、token usage 提取与日志渲染。

设计要点：
    - 延迟分布用有界 reservoir。请求数少时百分位精确；超过容量后只保留样本，
      并在摘要里标 approximate=True，避免把几万条浮点数一直留在内存/日志里。
    - token usage 只从 LangChain 返回的消息里取，不按字符数估算。provider 没给
      usage 时如实记 missing，不伪造 token/s。
    - 本模块不导入 langchain / agents，保持可被日志模块和测试独立复用。
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Iterable

DEFAULT_RESERVOIR_SIZE = 4096
"""延迟样本上限；count/min/mean/max 始终精确，百分位在超限后是近似值。"""

DEFAULT_HISTOGRAM_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
"""elapsed_s 直方图边界（秒）；每个计数在 add() 时精确累加，不受 reservoir 影响。"""


class LatencyTracker:
    """有界延迟采样器；调用方负责在并发环境下加锁。"""

    def __init__(self, capacity: int = DEFAULT_RESERVOIR_SIZE):
        if capacity < 1:
            raise ValueError(f"capacity 必须是正整数，收到 {capacity!r}")
        self.capacity = int(capacity)
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self._samples: list[float] = []
        self._rng = random.Random(0)
        self._approximate = False
        self._histogram = [0] * (len(DEFAULT_HISTOGRAM_BUCKETS) + 1)

    @staticmethod
    def _bucket_index(value: float) -> int:
        for index, upper in enumerate(DEFAULT_HISTOGRAM_BUCKETS):
            if value <= upper:
                return index
        return len(DEFAULT_HISTOGRAM_BUCKETS)

    def _histogram_labels(self) -> list[str]:
        labels: list[str] = []
        lower = 0.0
        for upper in DEFAULT_HISTOGRAM_BUCKETS:
            labels.append(f"{lower:g}-{upper:g}s")
            lower = upper
        labels.append(f">{lower:g}s")
        return labels

    def add(self, elapsed_s: float) -> None:
        value = max(0.0, float(elapsed_s))
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self._histogram[self._bucket_index(value)] += 1

        if len(self._samples) < self.capacity:
            self._samples.append(value)
            return
        # 经典 reservoir sampling：每个新样本以 capacity/count 的概率替换旧样本。
        self._approximate = True
        index = self._rng.randrange(self.count)
        if index < self.capacity:
            self._samples[index] = value

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "min": None,
                "mean": None,
                "p50": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "max": None,
                "sample_size": 0,
                "approximate": False,
                "histogram": {},
            }
        samples = sorted(self._samples)
        histogram = {
            label: count
            for label, count in zip(self._histogram_labels(), self._histogram)
            if count
        }
        return {
            "count": self.count,
            "min": round(self.minimum or 0.0, 4),
            "mean": round(self.total / self.count, 4),
            "p50": _percentile(samples, 50),
            "p90": _percentile(samples, 90),
            "p95": _percentile(samples, 95),
            "p99": _percentile(samples, 99),
            "max": round(self.maximum or 0.0, 4),
            "sample_size": len(samples),
            "approximate": self._approximate,
            "histogram": histogram,
        }


def _percentile(sorted_values: list[float], percent: float) -> float | None:
    """线性插值百分位；空样本返回 None。"""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)
    percent = min(100.0, max(0.0, float(percent)))
    position = (len(sorted_values) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = min(len(sorted_values) - 1, lower + 1)
    fraction = position - lower
    value = sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    return round(value, 4)


def summary_from_values(values: Iterable[float], capacity: int = DEFAULT_RESERVOIR_SIZE) -> dict[str, Any]:
    """便捷入口：直接从一组延迟值生成摘要（测试/一次性脚本使用）。"""
    tracker = LatencyTracker(capacity)
    for value in values:
        tracker.add(value)
    return tracker.summary()


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        role = message.get("role") or message.get("type") or ""
    else:
        role = getattr(message, "type", None) or type(message).__name__
    return str(role).lower()


def _message_usage(message: Any) -> dict[str, int] | None:
    """从一条 LangChain 消息里取 token usage；没有可用值返回 None。

    以 ``response_metadata.token_usage`` 的原始 usage 为准：langchain-openai /
    langchain-openrouter 会在原始 usage 缺少 completion_tokens 时把
    ``usage_metadata.output_tokens`` 补成 0。若先信 usage_metadata，会把这个 0 当成
    “真实输出 0 token”。原始 usage 缺失时才回退 usage_metadata，并对其中可能的
    补零做保守处理。
    """
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        usage = {}

    metadata = getattr(message, "response_metadata", None)
    raw: dict[str, Any] | None = None
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage") or metadata.get("usage")
        if isinstance(token_usage, dict) and token_usage:
            raw = token_usage

    def raw_value(*keys: str) -> tuple[Any, bool]:
        if raw is None:
            return None, False
        for key in keys:
            if key in raw and raw[key] is not None:
                return raw[key], True
        return None, False

    def usage_value(*keys: str) -> Any:
        for key in keys:
            if key in usage and usage[key] is not None:
                return usage[key]
        return None

    prompt, prompt_reported = raw_value("prompt_tokens", "input_tokens")
    completion, completion_reported = raw_value("completion_tokens", "output_tokens")
    total, total_reported = raw_value("total_tokens")

    # 原始 usage 存在但缺某字段时，usage_metadata 的同名字段只作补全：
    # 补 0 不可信（适配器会补 0），非 0 才可能是另一个来源的真实值。
    if not prompt_reported:
        candidate = usage_value("input_tokens", "prompt_tokens")
        if candidate not in (None, 0):
            prompt = candidate
    if not completion_reported:
        candidate = usage_value("output_tokens", "completion_tokens")
        if raw is None and candidate == 0:
            # OpenRouter 适配器只写 usage_metadata，缺 completion 时补 0，并把
            # total 算成 input+0。此时 total==input 是可识别的“补零”形状；
            # 真正的 0 输出应同时有 total==input 的原始 usage，二者无法区分时
            # 宁可记 missing，也不伪造 output_token/s=0。
            total_candidate = usage_value("total_tokens")
            prompt_candidate = usage_value("input_tokens", "prompt_tokens")
            if total_candidate is None or total_candidate == (prompt_candidate or 0):
                candidate = None
        if raw is None or candidate not in (None, 0):
            completion = candidate
    if not total_reported and raw is None:
        total = usage_value("total_tokens")

    return _normalize_usage(prompt, completion, total)


def _normalize_usage(prompt: Any, completion: Any, total: Any) -> dict[str, int] | None:
    """只返回实际报告的字段；缺失的 completion 不用 0 冒充。"""
    values = []
    for value in (prompt, completion, total):
        if value is None:
            values.append(None)
        elif isinstance(value, bool) or not isinstance(value, int):
            values.append(None)
        else:
            values.append(max(0, value))
    prompt_value, completion_value, total_value = values
    if all(value is None for value in (prompt_value, completion_value, total_value)):
        return None

    result: dict[str, int] = {}
    if prompt_value is not None:
        result["prompt_tokens"] = prompt_value
    if completion_value is not None:
        result["completion_tokens"] = completion_value
    if total_value is not None:
        result["total_tokens"] = total_value
    elif prompt_value is not None and completion_value is not None:
        result["total_tokens"] = prompt_value + completion_value
    return result


def usage_from_result(result: Any) -> dict[str, int] | None:
    """从 ``agent.invoke`` 的返回中汇总 token usage。

    只汇总可见 AI 消息；工具调用产生的多条 AI 消息各算一次。返回 dict 只包含
    实际报告过的字段，因此“只给了 prompt/total”不会被记成 completion=0。
    """
    if not isinstance(result, dict):
        return None
    messages = result.get("messages")
    if not isinstance(messages, (list, tuple)):
        return None

    totals: dict[str, int] = {}
    for message in messages:
        role = _message_role(message)
        if role not in {"ai", "assistant", "aimessage", "chatmodel"}:
            continue
        usage = _message_usage(message)
        if usage is None:
            continue
        for key, value in usage.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals or None


def merge_usage(*usages: dict[str, int] | None) -> dict[str, int] | None:
    """把多次尝试的 usage 逐字段相加；全部缺失时返回 None。

    ``call_with_retry`` 可能在一次逻辑请求内调用模型多次。只保留最后一次的 usage
    会漏掉前面尝试已经消耗的 token，因此调用方应在每次尝试后收集 usage，再用本函数
    合并。非法字段（bool、非 int）按缺失处理，不伪造 0。
    """
    totals: dict[str, int] = {}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            totals[key] = totals.get(key, 0) + value
    return totals or None


def _format_ratio(value: Any) -> str:
    if value is None:
        return "null"
    ratio = float(value)
    return f"{ratio:.4f} ({ratio * 100:.2f}%)"


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def format_stats_lines(stats: dict[str, Any], *, indent: str = "    ") -> list[str]:
    """把 ScreeningScreener.stats() 渲染成便于 grep/对比的多行文本。"""
    performance = (stats or {}).get("performance") or {}
    latency = performance.get("elapsed_s") or {}
    lines = [
        f"{indent}请求: completed={_format_number(performance.get('completed_requests'))}，"
        f"failed={_format_number(performance.get('failed_requests'))}，"
        f"invalid_batches={_format_number(performance.get('invalid_batches'))}，"
        f"wall_s={_format_number(performance.get('request_wall_s'))}，"
        f"requests/s={_format_number(performance.get('requests_per_s'), 3)}",
        f"{indent}输出 token: output={_format_number(performance.get('output_tokens'))}，"
        f"output_token/s={_format_number(performance.get('output_tokens_per_s'), 3)}，"
        f"output_usage_reported={_format_number(performance.get('output_usage_reported_requests'))}，"
        f"output_usage_missing={_format_number(performance.get('output_usage_missing_requests'))}，"
        f"usage_reported={_format_number(performance.get('usage_reported_requests'))}，"
        f"usage_missing={_format_number(performance.get('usage_missing_requests'))}",
        f"{indent}失败率: request={_format_ratio(performance.get('request_failure_rate'))}，"
        f"invalid_batch={_format_ratio(performance.get('invalid_batch_rate'))}，"
        f"item_error={_format_ratio(performance.get('item_error_rate'))}",
        f"{indent}elapsed_s: count={_format_number(latency.get('count'))}，"
        f"min={_format_number(latency.get('min'))}，"
        f"mean={_format_number(latency.get('mean'))}，"
        f"p50={_format_number(latency.get('p50'))}，"
        f"p90={_format_number(latency.get('p90'))}，"
        f"p95={_format_number(latency.get('p95'))}，"
        f"p99={_format_number(latency.get('p99'))}，"
        f"max={_format_number(latency.get('max'))}，"
        f"sample={_format_number(latency.get('sample_size'))}，"
        f"approximate={str(bool(latency.get('approximate'))).lower()}",
        f"{indent}elapsed_s 直方图: {json.dumps(latency.get('histogram') or {}, ensure_ascii=False, sort_keys=True)}",
    ]
    return lines


def stats_json(stats: dict[str, Any]) -> str:
    """稳定、可 grep 的 JSON 行（不含原始延迟数组）。

    ``default=str`` 是为了兼容旧测试桩/自定义 screener 传入的非标准对象；
    日志是附属产物，不应因为一个路径对象或自定义计数类型而整份写不出来。
    """
    return json.dumps(stats or {}, ensure_ascii=False, sort_keys=True, default=str)
