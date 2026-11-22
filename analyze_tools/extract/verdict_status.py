# encoding: utf8

"""逐词/整句筛选状态的只读镜像（仅标准库）。

分析脚本不导入 ``extract.sentence_screening``：那条导入会连带加载 pydantic/config，
让纯离线比较工具多花约 2 秒启动，也破坏“只依赖标准库”的定位。这里按生产口径
复刻三个判据：

- ``strict_verdict_state`` 对应 ``sentence_screening.verdict_status``；
- ``record_screening_status`` 对应 ``sentence_screening.screening_status``；
- ``judgement_eligible`` 对应 ``sentence_screening.is_judgement_eligible``。

复刻不是永久承诺：``tests/test_extract_analysis.py`` 有逐组合 parity 测试，生产规则
一旦变化，该测试会失败，提醒两边同步。

严格五态中只有 ``true``/``false`` 是有效判定；``error``/``limit``/``no_llm`` 分别表示
调用或输出异常、额度未送、未启用 LLM，三者绝不互相等同。
"""

from __future__ import annotations

STRICT_TRUE = "true"
STRICT_FALSE = "false"
STRICT_ERROR = "error"
STRICT_LIMIT = "limit"
STRICT_NO_LLM = "no_llm"

STRICT_STATES = (STRICT_TRUE, STRICT_FALSE, STRICT_ERROR, STRICT_LIMIT, STRICT_NO_LLM)
"""严格状态；顺序固定，用于矩阵和表头。"""

STRICT_EFFECTIVE = frozenset({STRICT_TRUE, STRICT_FALSE})
"""能算作“完成了有效判定”的状态；不包含 error/limit/no_llm。"""

STRICT_BLOCKING_SOURCES = (STRICT_ERROR, STRICT_LIMIT, STRICT_NO_LLM)
"""与生产 ``verdict_status`` 一致：这些来源优先于三项布尔值。"""

SCREENING_STATUSES = frozenset(STRICT_STATES)
JUDGEMENT_ELIGIBLE = frozenset({STRICT_TRUE, STRICT_NO_LLM})
"""与生产 ``JUDGEMENT_SCREENING_STATUSES`` 一致。"""

_BOOLEAN_FIELDS = ("extractable", "grammatical", "spatial_coherent")


def strict_verdict_state(verdict: dict | None) -> str:
    """一个逐词 verdict 的严格状态。

    与生产 ``verdict_status`` 同口径：来源为 error/limit/no_llm 时直接返回该来源；
    否则三项中任一为 False → false，全为 True → true，其余（含缺字段、混合值）→ error。

    注意 ``extractable=False`` 且其他字段缺失仍算 ``false``，因为生产口径先看
    “任一 False”，再看“全部 True”。
    """
    if not isinstance(verdict, dict):
        return STRICT_ERROR
    source = verdict.get("source")
    if source in STRICT_BLOCKING_SOURCES:
        return source
    values = [verdict.get(key) for key in _BOOLEAN_FIELDS]
    if any(value is False for value in values):
        return STRICT_FALSE
    if all(value is True for value in values):
        return STRICT_TRUE
    return STRICT_ERROR


def record_screening_status(record: dict) -> str:
    """一条原句记录的整句筛选状态，兼容没有 screening 的旧语料。

    与生产 ``screening_status`` 同口径：``screening.status`` 有效时以它为准；
    旧记录没有 status 时退回 extractable/source 推导；形状无法识别一律 error。
    """
    if not isinstance(record, dict):
        return STRICT_ERROR
    if "screening" not in record:
        return STRICT_NO_LLM
    block = record.get("screening")
    if not isinstance(block, dict):
        return STRICT_ERROR
    if "status" in block:
        status = block.get("status")
        # 生产实现先做 isinstance(status, str)：数组/对象/数字等不可哈希或非字符串
        # 的值应归为 error，而不是让 `in frozenset` 抛 TypeError 中断整批。
        return status if isinstance(status, str) and status in SCREENING_STATUSES else STRICT_ERROR
    extractable = block.get("extractable")
    if extractable is True:
        return STRICT_TRUE
    if extractable is False:
        return STRICT_FALSE
    source = block.get("source")
    if source in ("no_llm", "unfilled_prompt"):
        return STRICT_NO_LLM
    if source == "limit":
        return STRICT_LIMIT
    # 生产把 error 与 no_target 都归为 error；其余未知来源同样按 error。
    return STRICT_ERROR


def judgement_eligible(record: dict) -> bool:
    """判定工作流是否可使用这条原句：只接受 true 与 no_llm。"""
    return record_screening_status(record) in JUDGEMENT_ELIGIBLE


def raw_extractable_text(value: bool | None) -> str:
    """原始 ``extractable`` 的三态文本：true / false / unknown。"""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"
