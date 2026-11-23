# encoding: utf8

"""温度参数（temperature）的总开关与统一出站策略。

唯一开关是 :data:`config.LLM_TEMPERATURE_ENABLED`：

- **关闭（默认）**：所有模块都不向接口发送 ``temperature``。模型配置里写了也
  忽略；``disable_thinking`` / ``extra_body`` / ``model_kwargs`` / headers 等间接
  容器里的同名字段同样会被清掉，避免绕过总开关。
- **打开**：模型条目顶层有可解析的 ``temperature`` 就发送（保留 0）；缺失或非法
  时不补默认值，而是向 stderr 告警一次并省略该字段，请求照常继续。其他容器里的
  同名字段仍会被清除，不作为取值来源。

同步链路（``agents.get_model``）与原生 Batch（``batch_inference``）共用本模块，
评测报告也只按同一口径展示“实际是否发送”，避免三处各写一套默认值。
"""

from __future__ import annotations

import math
import sys
import threading

import config

_MISSING = object()

_NESTED_CONTAINERS = ("extra_body", "model_kwargs", "extra_headers", "default_headers", "headers")
"""可能夹带 temperature 的包装层；只清理这些已知容器，不递归碰业务内容。"""

_WARNED: set[tuple[str, str]] = set()
"""按（模型配置名, 原因）去重，保证重试/多波次不会刷屏。"""

_LOCK = threading.Lock()


def enabled() -> bool:
    """当前是否允许把 temperature 传给接口（读 config 总开关）。"""
    return bool(getattr(config, "LLM_TEMPERATURE_ENABLED", False))


def coerce(value) -> float | None:
    """把配置值转成有限浮点数；bool、None、空串、非法字符串一律视为无效。"""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (OverflowError, ValueError):
            # 合法 JSON 大整数（如 10**400）转 float 会溢出；按非法值省略，
            # 不能因为一个温度配置让整条请求失败。
            return None
    elif isinstance(value, str) and value.strip():
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def resolve(model_cfg: dict, *, model_key: str, warn: bool = True) -> float | None:
    """返回本次请求应发送的温度；``None`` 表示不发送该字段。

    只认模型条目**顶层**的 ``temperature``。``disable_thinking`` 等额外容器里的
    同名字段不作为取值来源（由 :func:`strip_temperature` 清掉），避免开关打开时
    仍存在两条互相矛盾的取值路径。开关关闭时直接返回 ``None``，不读取也不校验
    配置值。

    开关打开但配置缺失/非法时，按原因告警一次并返回 ``None``；调用方必须继续
    发请求，不能因为缺温度而失败。
    """
    if not enabled():
        return None

    raw = model_cfg.get("temperature", _MISSING) if isinstance(model_cfg, dict) else _MISSING
    if raw is _MISSING:
        if warn:
            _warn(model_key, "missing",
                  "已开启温度传入，但配置中没有 temperature；本次及后续请求省略该字段")
        return None

    value = coerce(raw)
    if value is None:
        if warn:
            _warn(model_key, "invalid",
                  "的 temperature 不是有效数值，已忽略；本次及后续请求省略该字段")
        return None
    return value


def strip_temperature(container: dict | None) -> dict:
    """返回去掉 temperature 的浅拷贝；用于切断额外参数里的旁路。

    只在已知包装层删键（大小写不敏感的 headers 也一并处理），不递归修改
    messages、tool schema 等业务内容。
    """
    if not isinstance(container, dict):
        return {}
    cleaned = dict(container)
    cleaned.pop("temperature", None)
    for name in _NESTED_CONTAINERS:
        nested = cleaned.get(name)
        if not isinstance(nested, dict):
            continue
        kept = {key: value for key, value in nested.items()
                if str(key).strip().lower() != "temperature"}
        if kept:
            cleaned[name] = kept
        else:
            cleaned.pop(name, None)
    return cleaned


def _warn(model_key: str, reason: str, message: str) -> None:
    token = (str(model_key), reason)
    with _LOCK:
        if token in _WARNED:
            return
        _WARNED.add(token)
    print(f"[温度参数] 警告：模型 {model_key!r} {message}", file=sys.stderr)


def reset_warnings() -> None:
    """清空告警去重表（测试用；生产进程无需调用）。"""
    with _LOCK:
        _WARNED.clear()
