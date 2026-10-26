# encoding: utf8

"""模型请求的可重试错误识别与指数退避重试。

本模块只做**一次逻辑调用**内部的传输层重试，不改变业务语义：
    - 只有超时、连接失败、限流与可恢复的 5xx 才重试；
    - 鉴权失败（401/403）、参数错误（400/422）、模型名不存在（404）等一律直接抛出；
    - 每次重试都在原调用外面重新发一次请求，成功后返回同一形状的结果。

为什么要单独一层，而不是只依赖各家 SDK 的 ``max_retries``：
    1. provider 适配层不统一，有的字段名不同、有的默认值不进运行清单；
    2. SDK 的重试次数与业务层重试叠在一起时会乘起来（6 × 6 = 49 次），
       一次超时可能把整批跑批拖到不可控；
    3. 本层把重试次数、退避时间、是否耗尽都记进 agent 的调用账，
       事后能从 .summary.json 里判断「慢」是模型慢还是重试多。

``max_retries`` 的语义与 OpenAI/Anthropic SDK 一致：**首次请求之外**最多再试几次。
默认 6 表示最多 7 次请求（1 次首发 + 6 次重试）。
"""

import random
import time

RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
"""值得重试的 HTTP 状态码：超时/冲突/限流/可恢复的服务端错误。

401/403（鉴权）、400/422（参数）、404（模型或路径不存在）都不在这里：
重试不会让它们变好，只会把一次明确的配置错误拖成七次。
"""

_NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 405, 422})
"""明确不重试的状态码。

异常链上可能同时出现外层包装器的 401/403 与内层文本里的 "timeout"；
这组状态码在 ``is_retryable_error`` 里**先于**其他信号检查，避免把配置错误误判成超时。
"""


def _status_code(exc) -> int | None:
    """从异常或它的 response 上取 HTTP 状态码（取不到返回 None）"""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _exception_chain(exc):
    """异常本身 + 它的 __cause__/__context__ 链，防止循环引用"""
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_retryable_error(exc: BaseException) -> bool:
    """这个异常是否值得重发一次请求

    判定顺序：
        1. 明确不可重试的 HTTP 状态码（401/403/400/404/422 等）先否决——
           外层包装器常把真正的错误塞进一个通用异常里，文本里可能刚好带 "timeout"；
        2. 超时/连接/限流/可恢复 5xx 的类名或状态码；
        3. ``TimeoutError`` / ``ConnectionError`` 等内建异常。
    """
    chain = list(_exception_chain(exc))
    for item in chain:
        status = _status_code(item)
        if status in _NON_RETRYABLE_STATUS_CODES:
            return False

    for item in chain:
        status = _status_code(item)
        if status in RETRYABLE_STATUS_CODES:
            return True

        if isinstance(item, (TimeoutError, ConnectionError)):
            return True

        name = type(item).__name__.lower()
        for marker in (
            "timeout",
            "timedout",
            "connection",
            "connecterror",
            "ratelimit",
            "internalserver",
            "serviceunavailable",
            "resourceexhausted",
            "deadlineexceeded",
            "unavailable",
            "overloaded",
            "transport",
            "protocol",
        ):
            if marker in name:
                return True

        text = str(item).lower()
        if "timed out" in text or "timeout" in text:
            return True
        if "connection reset" in text or "connection aborted" in text:
            return True
    return False


def _non_negative_int(policy: dict, key: str, default: int) -> int:
    value = policy.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"重试策略的 {key!r} 必须是非负整数，收到 {value!r}")
    return value


def _non_negative_float(policy: dict, key: str, default: float) -> float:
    value = policy.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"重试策略的 {key!r} 必须是非负数，收到 {value!r}")
    return float(value)


def backoff_delay(
    attempt: int,
    *,
    backoff_seconds: float = 1.0,
    backoff_max_seconds: float = 60.0,
    jitter_seconds: float = 0.5,
    rng: random.Random | None = None,
) -> float:
    """第 ``attempt`` 次重试前要等多久（attempt 从 1 起）

    指数退避：1、2、4、8…，封顶 ``backoff_max_seconds``，再叠加 [0, jitter] 的抖动。
    抖动是为了避免多个模型/多条候选在同一时刻一起重试，把限流打得更死。
    """
    attempt = max(1, int(attempt))
    delay = min(float(backoff_max_seconds), float(backoff_seconds) * (2 ** (attempt - 1)))
    jitter = float(jitter_seconds)
    if jitter > 0:
        delay += (rng or random).uniform(0.0, jitter)
    return delay


def call_with_retry(
    fn,
    *,
    policy: dict | None = None,
    sleep=time.sleep,
    rng: random.Random | None = None,
    on_retry=None,
):
    """执行 ``fn()``；遇到可重试错误时按策略重试，最终失败原样抛出最后一个异常

    Args:
        fn: 无参可调用对象，通常是一次 ``agent.invoke(...)``
        policy: 重试策略，键：
            max_retries            首次之外最多重试几次（默认 0，即不重试）
            backoff_seconds        首次退避秒数（默认 1）
            backoff_max_seconds    单次退避上限（默认 60）
            jitter_seconds         随机抖动上限（默认 0.5）
        sleep: 注入用（测试里换成不真睡的假函数）
        rng: 注入用（测试里固定抖动）
        on_retry: 每次决定重试时回调 ``on_retry(exc, attempt, delay)``，
            attempt 从 1 起。只统计，不返回值。

    Returns:
        ``fn()`` 的返回值。

    Raises:
        最后一次尝试的异常；不可重试的异常在第一次就抛出。
    """
    policy = policy or {}
    max_retries = _non_negative_int(policy, "max_retries", 0)
    backoff_seconds = _non_negative_float(policy, "backoff_seconds", 1.0)
    backoff_max_seconds = _non_negative_float(policy, "backoff_max_seconds", 60.0)
    jitter_seconds = _non_negative_float(policy, "jitter_seconds", 0.5)

    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_error(exc):
                raise
            attempt += 1
            delay = backoff_delay(
                attempt,
                backoff_seconds=backoff_seconds,
                backoff_max_seconds=backoff_max_seconds,
                jitter_seconds=jitter_seconds,
                rng=rng,
            )
            if on_retry is not None:
                on_retry(exc, attempt, delay)
            sleep(delay)
