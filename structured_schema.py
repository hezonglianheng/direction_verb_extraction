# encoding: utf8

"""结构化输出的 JSON Schema 契约：对象必须闭合，自由特征以条目数组传输。

为什么需要这一层：
    ZenMux 的 Anthropic 协议要求每个 ``type: object`` 节点显式写
    ``additionalProperties: false``（见 Claude structured outputs 文档）。Pydantic
    默认不写这个字段，裸 ``dict[str, Any]`` 还会生成 ``additionalProperties: true``，
    于是同步、Batch 两条链路都会在发请求时被后端拒绝。

两类对象分开处理：
    - 字段固定的判定模型继承 ``ClosedModel``：``extra="forbid"`` 会让 Pydantic 在
      schema 的每一层（含 ``$defs``、数组元素）写入 ``additionalProperties: false``；
    - 键名未知的自由特征映射不能直接闭合（闭合后只能返回 ``{}``），改用
      ``FeatureMap``：出站是闭合的 ``{key, value_json}`` 条目数组，验证时把
      ``value_json`` 解码回原值。业务代码仍然拿到 ``dict[str, Any]``，落盘形状不变。

``value_json`` 用 JSON 文本而不是 ``Any``：后者只是把不受约束的对象挪进数组元素，
仍会被 Claude 拒绝；缩窄成 ``list[str]`` 又会丢掉现有 ``Any`` 支持的标量、布尔、
嵌套对象等取值。JSON 文本是唯一能同时满足闭合约束与原语义的表示。
"""

import json
import math
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, WithJsonSchema


class ClosedModel(BaseModel):
    """所有结构化输出模型的基类：禁止额外字段，并让 schema 显式闭合每个对象"""

    model_config = ConfigDict(extra="forbid")


_FEATURE_MAP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "特征名，键名用简短的汉语名词短语",
            },
            "value_json": {
                "type": "string",
                "description": "该特征取值的 JSON 文本：字符串写成 \"值\"，"
                "列表写成 [\"值1\", \"值2\"]，布尔/数字/null 按 JSON 写法",
            },
        },
        "required": ["key", "value_json"],
        "additionalProperties": False,
    },
}
"""自由特征映射的出站表示：闭合条目数组（键名未知时唯一符合 Claude 约束的形状）"""


def _reject_nonstandard_constant(token: str) -> Any:
    """拒绝 JSON 标准之外的 NaN / Infinity / -Infinity

    Python 的 ``json.loads`` 默认接受它们，但 ``json.dumps`` 写出的裸 ``NaN`` /
    ``Infinity`` 不是合法 JSON，落盘后严格读取器（以及部分下游工具）会读不了整行。
    这里在解码入口就拒绝，避免把非法值带进业务字典。
    """
    raise ValueError(f"{token} 不是标准 JSON 数值")


def _parse_finite_float(text: str) -> float:
    """解析浮点数并拒绝溢出（例如 1e400 会变成 inf）"""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"数值 {text} 超出 JSON 可表示范围")
    return value


def _feature_entries_to_mapping(value: Any) -> dict[str, Any]:
    """把 ``{key, value_json}`` 条目数组还原成业务字典

    已经是字典的输入原样接受：本地缓存、Batch sidecar 与历史落盘保存的都是业务形状，
    续跑回放时不能要求它们先改写成传输形状。重复键与非法 JSON 必须显式失败，不能
    静默覆盖或丢值。
    """
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        raise ValueError(
            "自由特征必须是 {key, value_json} 条目数组或特征字典，"
            f"收到 {type(value).__name__}"
        )

    mapping: dict[str, Any] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"自由特征条目必须是对象，收到 {type(entry).__name__}")
        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"自由特征条目的 key 必须是非空字符串，收到 {key!r}")
        if key in mapping:
            raise ValueError(f"自由特征映射里出现重复键 {key!r}")
        raw = entry.get("value_json")
        if not isinstance(raw, str):
            raise ValueError(f"自由特征 {key!r} 的 value_json 必须是字符串，收到 {type(raw).__name__}")
        try:
            mapping[key] = json.loads(
                raw,
                parse_constant=_reject_nonstandard_constant,
                parse_float=_parse_finite_float,
            )
        except ValueError as exc:
            raise ValueError(f"自由特征 {key!r} 的 value_json 不是合法 JSON：{exc}") from exc
    return mapping


FeatureMap = Annotated[
    dict[str, Any],
    BeforeValidator(_feature_entries_to_mapping),
    WithJsonSchema(_FEATURE_MAP_SCHEMA, mode="validation"),
]
"""业务上是 ``dict[str, Any]``，出站 schema 是闭合的 ``{key, value_json}`` 数组"""
