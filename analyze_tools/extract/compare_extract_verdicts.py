#!/usr/bin/env python3
# encoding: utf8

"""对比 extract 模块的方位词筛选结果，输出两两 CSV 与总体一致性矩阵。

只支持 extract 原句；replace 句对请使用
``analyze_tools/replace/compare_replace_pairs.py``，本脚本会拒绝句对 JSONL。

比较对象是 ``results/extract_result`` 下同名的
``location_corpus_with_locative_split_*.screening_verdicts.csv``（或原始 ``.jsonl``）。
一行 CSV / 一个逐词 verdict 表示模型对某个方位词候选的一次判定，其
``(目标物, 参照物, 动词)`` 就是本项目所说的三元组。

比较分三层，全部保留：

1. **原始 extractable 三态**（兼容旧口径）：只看 ``extractable`` 的
   true/false/unknown。这是旧报告里的“一致率”，单看它不能说明下游可用性。
2. **严格逐词状态**：按生产 ``verdict_status`` 口径分成
   ``true / false / error / limit / no_llm`` 五态。``error``、``limit``、``no_llm``
   **分别独立**，绝不互算相同；只有 ``true`` 与 ``false`` 内部允许包含不同的底层
   字段组合（例如 true 可以由不同理由/来源得到）。严格 true 要求
   ``extractable``、``grammatical``、``spatial_coherent`` 全为 true，且来源不是
   error/limit/no_llm。
3. **整句状态与下游准入**（仅 JSONL）：按 ``screening_status`` 统计整句
   true/false/error/limit/no_llm，并比较两侧“可进入判定工作流”的句集
   （true 或 no_llm）。CSV 导出会漏掉 ``verdicts=[]`` 的 no_llm 句，无法还原
   全量句级指标，因此 CSV 的句级指标记 ``N/A（需 JSONL）``。

旧版 CSV 没有 ``extractable``、``grammatical``、``spatial_coherent``、
``逐词判定来源`` 四列，无法反推严格状态：此时仍输出原始 extractable 一致率，
严格指标记 ``N/A（缺列，请重新导出）``，不把缺列伪装成生产 ``error``。

对齐口径（默认）::

    base = (源文件, 原句, 方位词, 方位词字符偏移)
    key  = base + (同一 base 在该文件中按原始顺序出现的序号,)

字符偏移是方位词在“原句拼接文本”中的起始位置（从 0 开始的 Unicode 字符数），
可由 token 列表 ``sum(len(token) for token in tokens[:index])`` 得到。它比 token
下标稳定：不同批次即使重新分词，只要原句文本不变，同一个方位词的偏移就不变。

``JSONL行号`` **不参与** 对齐：不同批次的结果会因记录增删而整体平移，用它对齐会把
大量相同项误判为“仅单侧”。三元组本身也不参与 key：角色可能为空、grounding 也可能
变化；三元组文本是否变化单独统计，不混进判定翻转。

``screening.verdicts`` 中实际存在的 verdict 若 ``extractable=None``（典型为
limit/error；旧数据里也可能有逐词 ``no_llm`` verdict），表示该方位词没有拿到
true/false 判定。脚本**不再跳过**这些行，而是作为 ``unknown`` 参与对齐、明细和统计；
``unknown`` 不是 ``false``，也不会被当成 ``false``。真实 ``annotate_no_llm`` 记录是
记录级状态、``verdicts=[]``，不产生比较行，也不会计入 ``unknown``。

同一 ``base`` 在一侧重复出现时，按原始顺序第 1 对第 1、第 2 对第 2 配对；若一侧多出
或缺少其中一条，序号配对只能保证确定性，无法消除歧义，因此输出会单独打印“重复 base
组数”和相同判断数可能范围。

输入 2 个或更多文件时，脚本对每一对文件生成一个逐项明细 CSV，并额外写一份
``summary.md``，其中包含每个文件的统计、两两汇总表和一致性矩阵。一致性矩阵的单元格
是 ``一致率（相同/并集）``，分母是两侧 key 的并集
``common + only_left + only_right``，仅单侧项也进入分母；对角线写 ``—``，并集为 0
时写 ``N/A (0/0)``。共同项中三态判定相等（含 ``unknown`` 对 ``unknown``）记为相同。

严格口径另给“有效判定一致率”：分母只算两侧都是严格 true/false 的共同项，
error/limit/no_llm 不进入分母；同时报告有效判定覆盖率，避免只看一个高一致率。
句级另给“整句状态一致率”和“下游准入句集重合率”。

脚本只读输入；差异本身不算失败，返回 0。只有至少一对的并集为 0（两侧都没有判定行）
才返回 1；无法开始比较（少于两个文件、重复输入、格式混合、文件损坏、输出冲突）返回 2。

用法::

    # 不传参数：自动发现 results/extract_result 下全部筛选 CSV
    .venv/bin/python analyze_tools/extract/compare_extract_verdicts.py

    # 只比较最新两份 CSV
    .venv/bin/python analyze_tools/extract/compare_extract_verdicts.py --latest 2

    # 显式指定多个 CSV / JSONL
    .venv/bin/python analyze_tools/extract/compare_extract_verdicts.py \
        results/extract_result/location_corpus_with_locative_split_20261029.screening_verdicts.csv \
        results/extract_result/location_corpus_with_locative_split_20261031.screening_verdicts.csv \
        results/extract_result/location_corpus_with_locative_split_20261103.screening_verdicts.csv

    # 指定输出目录（必须尚不存在）
    .venv/bin/python analyze_tools/extract/compare_extract_verdicts.py \
        --output-dir /tmp/verdict_comparison
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

if __package__:
    from .verdict_status import (
        STRICT_EFFECTIVE,
        STRICT_STATES,
        judgement_eligible,
        record_screening_status,
        strict_verdict_state,
    )
else:
    # 直接执行脚本时 Python 只把脚本目录放进 sys.path；状态镜像与脚本同目录。
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verdict_status import (  # noqa: E402
        STRICT_EFFECTIVE,
        STRICT_STATES,
        judgement_eligible,
        record_screening_status,
        strict_verdict_state,
    )

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results" / "extract_result"
ANALYZE_RESULTS_DIR = ROOT_DIR / "analyze_results"
CSV_GLOB = "location_corpus_with_locative_split_*.screening_verdicts.csv"
COMPARISON_DIR_NAME = "compare_verdicts"
SUMMARY_FILENAME = "summary.md"
DEFAULT_MAX_FILES = 20
DATE_RE = re.compile(r"_(\d{8})\.screening_verdicts\.csv$")
LABEL_PREFIX_RE = re.compile(r"^location_corpus_with_locative_split_")

CSV_COLUMNS = [
    "JSONL行号",
    "原句",
    "源文件",
    "方位词",
    "方位词下标",
    "方位词字符偏移",
    "目标物",
    "参照物",
    "动词",
    "模型判定",
    "判定理由",
    "判定来源",
    "整句筛选状态",
]
# 只要求比较所必需的列；判定来源、判定理由、整句筛选状态缺失时按空串处理，
# 避免旧版 CSV 仅因多/少一个展示列就无法比较。
CSV_REQUIRED = frozenset(
    {
        "JSONL行号",
        "原句",
        "源文件",
        "方位词",
        "方位词下标",
        "方位词字符偏移",
        "目标物",
        "参照物",
        "动词",
        "模型判定",
    }
)
CSV_STRICT_REQUIRED = frozenset(
    {"extractable", "grammatical", "spatial_coherent", "逐词判定来源"}
)
"""新版导出用于复算严格状态的四列。

四列全无 = 旧版 CSV，严格指标记 N/A；只出现一部分 = 导出被截断/手工改坏，
直接报错，避免把“缺列”静默当成“字段缺失的 error”。
"""

DETAIL_COLUMNS = [
    "比较结果",
    "左侧行号",
    "右侧行号",
    "源文件",
    "原句",
    "方位词",
    "方位词字符偏移",
    "重复序号",
    "左方位词下标",
    "右方位词下标",
    "左目标物",
    "左参照物",
    "左动词",
    "右目标物",
    "右参照物",
    "右动词",
    "左判定",
    "右判定",
    "左严格状态",
    "右严格状态",
    "严格比较结果",
    "左extractable",
    "左grammatical",
    "左spatial_coherent",
    "右extractable",
    "右grammatical",
    "右spatial_coherent",
    "左逐词来源",
    "右逐词来源",
    "左整句状态",
    "右整句状态",
    "三元组是否变化",
    "左理由",
    "右理由",
]

LOCATIVE_COLUMNS = [
    "方位词",
    "共同",
    "原始相同",
    "原始不同",
    "严格共同",
    "严格相同",
    "严格不同",
    "有效共同",
    "有效相同",
    "仅左",
    "仅右",
]
LOCATIVE_ALIGNS = ["---"] + ["---:"] * 10


def verdict_text(value: bool | None) -> str:
    """原始 ``extractable`` 的三态文本：true / false / unknown。"""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def strict_state_text(value: str | None) -> str:
    """严格状态的展示文本；不可计算时用 ``n/a``。"""
    return value if value else "n/a"


def strict_state_label(value: str | None) -> str:
    """严格状态的中文表头标签；只有 true/false 是有效判定。"""
    if value is None:
        return "n/a"
    return {"true": "严格true", "false": "严格false"}.get(value, value)


def is_effective_state(value: str | None) -> bool:
    return value in STRICT_EFFECTIVE


def _ratio_text(numerator: int, denominator: int, *, unavailable: str = "N/A (0/0)") -> str:
    if not denominator:
        return unavailable
    return f"{numerator / denominator * 100:.2f}% ({numerator}/{denominator})"


def _format_strict_counts(counts: Counter[str]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{state}={counts.get(state, 0)}" for state in STRICT_STATES)


@dataclass(frozen=True)
class VerdictRow:
    """一个可比较的逐方位词判定投影。

    ``verdict`` 是三态：``True`` / ``False`` / ``None``（unknown，没拿到判定）。
    """

    key: tuple[str, str, str, int, int]
    base: tuple[str, str, str, int]
    line_no: str
    source_file: str
    sentence: str
    locative: str
    locative_index: str
    locative_offset: int
    target: str
    reference: str
    verb: str
    verdict: bool | None
    reason: str
    status: str
    grammatical: bool | None = None
    spatial_coherent: bool | None = None
    verdict_source: str = ""
    record_status: str = ""
    strict_state: str | None = None
    strict_available: bool = False

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.target, self.reference, self.verb)

    @property
    def raw_state(self) -> str:
        return verdict_text(self.verdict)

    @property
    def strict_state_text(self) -> str:
        return strict_state_text(self.strict_state)

    @property
    def effective(self) -> bool:
        return is_effective_state(self.strict_state)

    @property
    def complete(self) -> bool:
        """目标物、参照物、动词三者都不为空。"""
        return bool(self.target and self.reference and self.verb)

    @property
    def verdict_text(self) -> str:
        return verdict_text(self.verdict)


def _state_counts(rows: Sequence[VerdictRow]) -> tuple[int, int, int]:
    """返回 (true, false, unknown) 三态计数。"""
    true_count = sum(1 for row in rows if row.verdict is True)
    false_count = sum(1 for row in rows if row.verdict is False)
    return true_count, false_count, len(rows) - true_count - false_count


def _max_cross_pairs(
    left_counts: tuple[int, int, int], right_counts: tuple[int, int, int]
) -> int:
    """最多能配出多少对不同状态的 pair（小型三态最大流）。"""
    source, sink = 0, 7
    size = 8
    capacity = [[0] * size for _ in range(size)]
    for state, count in enumerate(left_counts):
        capacity[source][1 + state] = count
    for left_state in range(3):
        for right_state in range(3):
            if left_state != right_state:
                capacity[1 + left_state][4 + right_state] = 10**9
    for state, count in enumerate(right_counts):
        capacity[4 + state][sink] = count

    total = 0
    while True:
        parent = [-1] * size
        parent[source] = source
        queue = [source]
        for node in queue:
            for nxt in range(size):
                if parent[nxt] == -1 and capacity[node][nxt] > 0:
                    parent[nxt] = node
                    queue.append(nxt)
                    if nxt == sink:
                        break
            if parent[sink] != -1:
                break
        if parent[sink] == -1:
            return total
        amount = 10**9
        node = sink
        while node != source:
            prev = parent[node]
            amount = min(amount, capacity[prev][node])
            node = prev
        node = sink
        while node != source:
            prev = parent[node]
            capacity[prev][node] -= amount
            capacity[node][prev] += amount
            node = prev
        total += amount


@dataclass
class Comparison:
    left: list[VerdictRow]
    right: list[VerdictRow]
    common_keys: list[tuple]
    only_left_keys: list[tuple]
    only_right_keys: list[tuple]
    left_by_key: dict[tuple, VerdictRow]
    right_by_key: dict[tuple, VerdictRow]
    left_sentences: dict[tuple[str, str, int], SentenceRecord] | None = None
    right_sentences: dict[tuple[str, str, int], SentenceRecord] | None = None
    left_strict_available: bool | None = None
    right_strict_available: bool | None = None

    def __post_init__(self) -> None:
        self.left_total = len(self.left)
        self.right_total = len(self.right)
        self.left_true, self.left_false, self.left_unknown = _state_counts(self.left)
        self.right_true, self.right_false, self.right_unknown = _state_counts(self.right)
        self.left_complete = sum(row.complete for row in self.left)
        self.right_complete = sum(row.complete for row in self.right)

        self.common = len(self.common_keys)
        self.only_left = len(self.only_left_keys)
        self.only_right = len(self.only_right_keys)
        # 一致性矩阵的分母：两侧 key 的并集。仅单侧项也进入分母。
        self.union = self.common + self.only_left + self.only_right

        # ---- 第一层：原始 extractable 三态（兼容旧口径） ----
        self.matrix: Counter[tuple[bool | None, bool | None]] = Counter()
        self.same = 0
        self.different = 0
        self.common_complete = 0
        self.same_complete = 0
        self.different_complete = 0
        self.triple_changed = 0
        self.triple_changed_same_verdict = 0
        self.triple_changed_diff_verdict = 0
        self.line_mismatch = 0

        # ---- 第二层：逐词严格五态 ----
        # 文件级可用性优先由读取器给出：空的旧版 CSV 没有行，不能靠 all([])
        # 误判成“严格可用”。直接调用 compare_rows 时没有文件级标志，才退回逐行检查。
        left_strict = (
            self.left_strict_available
            if self.left_strict_available is not None
            else all(row.strict_available for row in self.left)
        )
        right_strict = (
            self.right_strict_available
            if self.right_strict_available is not None
            else all(row.strict_available for row in self.right)
        )
        self.strict_available = bool(left_strict and right_strict)
        self.strict_matrix: Counter[tuple[str, str]] = Counter()
        self.strict_comparable = 0
        self.strict_same = 0
        self.strict_different = 0
        self.common_effective = 0
        self.same_effective = 0
        self.left_strict_counts = Counter(
            row.strict_state for row in self.left if row.strict_state
        )
        self.right_strict_counts = Counter(
            row.strict_state for row in self.right if row.strict_state
        )

        for key in self.common_keys:
            left = self.left_by_key[key]
            right = self.right_by_key[key]
            pair = (left.verdict, right.verdict)
            self.matrix[pair] += 1
            is_same = left.verdict == right.verdict
            if is_same:
                self.same += 1
            else:
                self.different += 1
            if left.complete and right.complete:
                self.common_complete += 1
                if is_same:
                    self.same_complete += 1
                else:
                    self.different_complete += 1
            if left.triple != right.triple:
                self.triple_changed += 1
                if is_same:
                    self.triple_changed_same_verdict += 1
                else:
                    self.triple_changed_diff_verdict += 1
            if left.line_no != right.line_no:
                self.line_mismatch += 1

            if left.strict_state and right.strict_state:
                self.strict_comparable += 1
                self.strict_matrix[(left.strict_state, right.strict_state)] += 1
                if left.strict_state == right.strict_state:
                    self.strict_same += 1
                else:
                    self.strict_different += 1
            if left.effective and right.effective:
                self.common_effective += 1
                if left.strict_state == right.strict_state:
                    self.same_effective += 1

        self.only_left_strict_counts = Counter(
            self.left_by_key[key].strict_state
            for key in self.only_left_keys
            if self.left_by_key[key].strict_state
        )
        self.only_right_strict_counts = Counter(
            self.right_by_key[key].strict_state
            for key in self.only_right_keys
            if self.right_by_key[key].strict_state
        )
        self.left_effective = sum(row.effective for row in self.left)
        self.right_effective = sum(row.effective for row in self.right)

        self.only_left_true, self.only_left_false, self.only_left_unknown = _state_counts(
            [self.left_by_key[key] for key in self.only_left_keys]
        )
        self.only_right_true, self.only_right_false, self.only_right_unknown = (
            _state_counts([self.right_by_key[key] for key in self.only_right_keys])
        )

        self.common_unknown = sum(
            1
            for key in self.common_keys
            if self.left_by_key[key].verdict is None
            or self.right_by_key[key].verdict is None
        )
        self.same_unknown = self.matrix[(None, None)]
        self.different_unknown = self.common_unknown - self.same_unknown

        left_by_base: dict[tuple[str, str, str, int], list[VerdictRow]] = {}
        for row in self.left:
            left_by_base.setdefault(row.base, []).append(row)
        right_by_base: dict[tuple[str, str, str, int], list[VerdictRow]] = {}
        for row in self.right:
            right_by_base.setdefault(row.base, []).append(row)
        self.duplicate_groups_left = sum(
            1 for rows in left_by_base.values() if len(rows) > 1
        )
        self.duplicate_groups_right = sum(
            1 for rows in right_by_base.values() if len(rows) > 1
        )

        # 重复 base 的配对歧义：序号配对是确定性的，但一侧多出/缺少一条时无法知道
        # 究竟对应哪一条。这里把受影响的共同项和 same 的可能范围都量化出来。
        self.duplicate_common = 0
        self.duplicate_unequal = 0
        self.duplicate_unequal_both = 0
        self.order_ambiguous_groups = 0
        self.order_ambiguous_common = 0
        self.same_in_duplicate_base = 0
        duplicate_same_min = 0
        duplicate_same_max = 0
        for base in set(left_by_base) | set(right_by_base):
            left_rows = left_by_base.get(base, [])
            right_rows = right_by_base.get(base, [])
            if len(left_rows) <= 1 and len(right_rows) <= 1:
                continue
            if len(left_rows) != len(right_rows):
                self.duplicate_unequal += 1
            if left_rows and right_rows:
                paired = min(len(left_rows), len(right_rows))
                self.duplicate_common += paired
                self.same_in_duplicate_base += sum(
                    left_rows[i].verdict == right_rows[i].verdict
                    for i in range(paired)
                )
                left_counts = _state_counts(left_rows)
                right_counts = _state_counts(right_rows)
                duplicate_same_max += min(
                    paired, sum(min(a, b) for a, b in zip(left_counts, right_counts))
                )
                cross_possible = _max_cross_pairs(left_counts, right_counts)
                duplicate_same_min += paired - min(paired, cross_possible)
                if len(left_rows) != len(right_rows):
                    self.duplicate_unequal_both += 1
            if left_rows and right_rows and len(left_rows) == len(right_rows):
                left_order = [row.verdict for row in left_rows]
                right_order = [row.verdict for row in right_rows]
                if (
                    Counter(left_order) == Counter(right_order)
                    and left_order != right_order
                ):
                    self.order_ambiguous_groups += 1
                    self.order_ambiguous_common += len(left_order)
        self.same_min_total = (
            self.same - self.same_in_duplicate_base + duplicate_same_min
        )
        self.same_max_total = (
            self.same - self.same_in_duplicate_base + duplicate_same_max
        )

        # ---- 第三层：整句状态与下游准入（仅 JSONL） ----
        self.sentence_available = (
            self.left_sentences is not None and self.right_sentences is not None
        )
        self.sentence_matrix: Counter[tuple[str, str]] = Counter()
        self.sentence_common = 0
        self.sentence_only_left = 0
        self.sentence_only_right = 0
        self.sentence_union = 0
        self.sentence_same = 0
        self.sentence_left_counts: Counter[str] = Counter()
        self.sentence_right_counts: Counter[str] = Counter()
        self.eligible_left = 0
        self.eligible_right = 0
        self.eligible_both = 0
        self.eligible_left_only = 0
        self.eligible_right_only = 0
        self.eligible_union = 0
        if self.sentence_available:
            left_sentences = self.left_sentences or {}
            right_sentences = self.right_sentences or {}
            left_keys = set(left_sentences)
            right_keys = set(right_sentences)
            common_sentences = left_keys & right_keys
            self.sentence_common = len(common_sentences)
            self.sentence_only_left = len(left_keys - right_keys)
            self.sentence_only_right = len(right_keys - left_keys)
            self.sentence_union = len(left_keys | right_keys)
            for key in common_sentences:
                left_status = left_sentences[key].status
                right_status = right_sentences[key].status
                self.sentence_matrix[(left_status, right_status)] += 1
                if left_status == right_status:
                    self.sentence_same += 1
            self.sentence_left_counts = Counter(
                value.status for value in left_sentences.values()
            )
            self.sentence_right_counts = Counter(
                value.status for value in right_sentences.values()
            )
            eligible_left_keys = {
                key for key, value in left_sentences.items() if value.eligible
            }
            eligible_right_keys = {
                key for key, value in right_sentences.items() if value.eligible
            }
            self.eligible_left = len(eligible_left_keys)
            self.eligible_right = len(eligible_right_keys)
            self.eligible_both = len(eligible_left_keys & eligible_right_keys)
            self.eligible_left_only = len(eligible_left_keys - eligible_right_keys)
            self.eligible_right_only = len(eligible_right_keys - eligible_left_keys)
            self.eligible_union = len(eligible_left_keys | eligible_right_keys)

        self.by_locative = self._by_locative()

    @property
    def agreement(self) -> float | None:
        """原始 extractable：相同 / 两侧并集；仅单侧项也进入分母。"""
        return self.same / self.union if self.union else None

    @property
    def common_agreement(self) -> float | None:
        """原始 extractable：相同 / 共同项，仅用于对照。"""
        return self.same / self.common if self.common else None

    @property
    def strict_agreement(self) -> float | None:
        """严格五态：相同 / 两侧并集；error/limit/no_llm 也参与分母。"""
        if not self.strict_available:
            return None
        return self.strict_same / self.union if self.union else None

    @property
    def strict_common_agreement(self) -> float | None:
        """严格五态：相同 / 可比较共同项。"""
        if not self.strict_available or not self.strict_comparable:
            return None
        return self.strict_same / self.strict_comparable

    @property
    def effective_agreement(self) -> float | None:
        """有效判定：两侧都为严格 true/false 的共同项内，状态相同的比例。"""
        return self.same_effective / self.common_effective if self.common_effective else None

    @property
    def left_effective_coverage(self) -> float | None:
        return self.left_effective / self.left_total if self.left_total else None

    @property
    def right_effective_coverage(self) -> float | None:
        return self.right_effective / self.right_total if self.right_total else None

    @property
    def sentence_agreement(self) -> float | None:
        if not self.sentence_available:
            return None
        return self.sentence_same / self.sentence_union if self.sentence_union else None

    @property
    def eligibility_overlap(self) -> float | None:
        if not self.sentence_available:
            return None
        return self.eligible_both / self.eligible_union if self.eligible_union else None

    def _by_locative(self) -> list[dict[str, Any]]:
        stats: dict[str, Counter[str]] = {}
        for key in self.common_keys:
            left = self.left_by_key[key]
            right = self.right_by_key[key]
            item = stats.setdefault(left.locative, Counter())
            item["common"] += 1
            if left.verdict == right.verdict:
                item["same"] += 1
                item["raw_same"] += 1
            else:
                item["different"] += 1
                item["raw_different"] += 1
            if left.verdict is True and right.verdict is False:
                item["tf"] += 1
            elif left.verdict is False and right.verdict is True:
                item["ft"] += 1
            if left.verdict is None and right.verdict is None:
                item["uu"] += 1
            elif left.verdict is None or right.verdict is None:
                item["udiff"] += 1
            if left.strict_state and right.strict_state:
                item["strict_common"] += 1
                if left.strict_state == right.strict_state:
                    item["strict_same"] += 1
                else:
                    item["strict_different"] += 1
            if left.effective and right.effective:
                item["effective_common"] += 1
                if left.strict_state == right.strict_state:
                    item["effective_same"] += 1
        for key in self.only_left_keys:
            item = stats.setdefault(self.left_by_key[key].locative, Counter())
            item["only_left"] += 1
        for key in self.only_right_keys:
            item = stats.setdefault(self.right_by_key[key].locative, Counter())
            item["only_right"] += 1
        rows = []
        for locative, counter in stats.items():
            rows.append({"locative": locative, **counter})
        rows.sort(key=lambda item: (-item.get("common", 0), item["locative"]))
        return rows

    def difference_examples(self, limit: int) -> list[tuple[VerdictRow, VerdictRow]]:
        if limit <= 0:
            return []
        examples = []
        for key in self.common_keys:
            left = self.left_by_key[key]
            right = self.right_by_key[key]
            if left.verdict != right.verdict:
                examples.append((left, right))
                if len(examples) >= limit:
                    break
        return examples

    def only_examples(
        self, limit: int
    ) -> tuple[list[VerdictRow], list[VerdictRow]]:
        if limit <= 0:
            return [], []
        return (
            [self.left_by_key[key] for key in self.only_left_keys[:limit]],
            [self.right_by_key[key] for key in self.only_right_keys[:limit]],
        )


@dataclass
class LoadedFile:
    """一次运行中已加载的输入文件。"""

    index: int
    path: Path
    fmt: str
    label: str
    rows: list[VerdictRow]
    strict_available: bool = False
    sentences: dict[tuple[str, str, int], SentenceRecord] | None = None

    def __post_init__(self) -> None:
        self.strict_counts: Counter[str] = Counter(
            row.strict_state for row in self.rows if row.strict_state
        )
        self.sentence_status_counts: Counter[str] = Counter(
            value.status for value in (self.sentences or {}).values()
        )
        self.sentence_eligible_count = sum(
            1 for value in (self.sentences or {}).values() if value.eligible
        )

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def true_count(self) -> int:
        return sum(1 for row in self.rows if row.verdict is True)

    @property
    def false_count(self) -> int:
        return sum(1 for row in self.rows if row.verdict is False)

    @property
    def unknown_count(self) -> int:
        return self.total - self.true_count - self.false_count

    @property
    def effective_count(self) -> int:
        return sum(row.effective for row in self.rows)

    @property
    def complete_count(self) -> int:
        return sum(row.complete for row in self.rows)

    @property
    def duplicate_groups(self) -> int:
        return sum(
            1 for count in Counter(row.base for row in self.rows).values() if count > 1
        )

    def strict_count(self, state: str) -> int:
        return self.strict_counts.get(state, 0)

    @property
    def sentence_total(self) -> int:
        return len(self.sentences) if self.sentences is not None else 0


@dataclass
class PairSummary:
    """一对比较的轻量汇总；不持有 Comparison，避免 N 文件时累积 O(N^2) 内存。"""

    left: LoadedFile
    right: LoadedFile
    csv_name: str
    common: int
    union: int
    same: int
    different: int
    only_left: int
    only_right: int
    triple_changed: int
    duplicate_unequal_both: int
    order_ambiguous_groups: int
    order_ambiguous_common: int
    same_min_total: int
    same_max_total: int
    strict_available: bool
    strict_same: int
    strict_different: int
    strict_comparable: int
    strict_transitions: dict[tuple[str, str], int]
    common_effective: int
    same_effective: int
    sentence_available: bool
    sentence_common: int
    sentence_union: int
    sentence_same: int
    eligible_both: int
    eligible_union: int
    eligible_left_only: int
    eligible_right_only: int

    @property
    def agreement(self) -> float | None:
        return self.same / self.union if self.union else None

    @property
    def agreement_text(self) -> str:
        """原始 extractable 一致率（兼容旧报告的名字）。"""
        return _ratio_text(self.same, self.union)

    @property
    def strict_agreement(self) -> float | None:
        if not self.strict_available:
            return None
        return self.strict_same / self.union if self.union else None

    @property
    def strict_agreement_text(self) -> str:
        if not self.strict_available:
            return "N/A（缺列，请重新导出）"
        return _ratio_text(self.strict_same, self.union)

    @property
    def effective_agreement(self) -> float | None:
        if not self.common_effective:
            return None
        return self.same_effective / self.common_effective

    @property
    def effective_agreement_text(self) -> str:
        if not self.strict_available:
            return "N/A（缺列，请重新导出）"
        if not self.common_effective:
            return "N/A (0/0)"
        return f"{self.effective_agreement * 100:.2f}% ({self.same_effective}/{self.common_effective})"

    @property
    def sentence_agreement(self) -> float | None:
        if not self.sentence_available or not self.sentence_union:
            return None
        return self.sentence_same / self.sentence_union

    @property
    def sentence_agreement_text(self) -> str:
        if not self.sentence_available:
            return "N/A（需 JSONL）"
        return _ratio_text(self.sentence_same, self.sentence_union)

    @property
    def eligibility_overlap(self) -> float | None:
        if not self.sentence_available or not self.eligible_union:
            return None
        return self.eligible_both / self.eligible_union

    @property
    def eligibility_overlap_text(self) -> str:
        if not self.sentence_available:
            return "N/A（需 JSONL）"
        return _ratio_text(self.eligible_both, self.eligible_union)


def pair_summary_from_comparison(
    left: LoadedFile, right: LoadedFile, csv_name: str, comparison: Comparison
) -> PairSummary:
    return PairSummary(
        left=left,
        right=right,
        csv_name=csv_name,
        common=comparison.common,
        union=comparison.union,
        same=comparison.same,
        different=comparison.different,
        only_left=comparison.only_left,
        only_right=comparison.only_right,
        triple_changed=comparison.triple_changed,
        duplicate_unequal_both=comparison.duplicate_unequal_both,
        order_ambiguous_groups=comparison.order_ambiguous_groups,
        order_ambiguous_common=comparison.order_ambiguous_common,
        same_min_total=comparison.same_min_total,
        same_max_total=comparison.same_max_total,
        strict_available=comparison.strict_available,
        strict_same=comparison.strict_same,
        strict_different=comparison.strict_different,
        strict_comparable=comparison.strict_comparable,
        strict_transitions=dict(comparison.strict_matrix),
        common_effective=comparison.common_effective,
        same_effective=comparison.same_effective,
        sentence_available=comparison.sentence_available,
        sentence_common=comparison.sentence_common,
        sentence_union=comparison.sentence_union,
        sentence_same=comparison.sentence_same,
        eligible_both=comparison.eligible_both,
        eligible_union=comparison.eligible_union,
        eligible_left_only=comparison.eligible_left_only,
        eligible_right_only=comparison.eligible_right_only,
    )


def sentence_text(value: Any) -> str:
    """还原原句：新结果存 token 列表，旧结果可能直接存字符串。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(token for token in value if isinstance(token, str))
    if value is None:
        return ""
    return str(value)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _bool_cell(value: Any) -> str:
    """布尔字段写成小写 true/false；None 写空串，非布尔值明确标出。

    JSONL 里可能混入字符串 "false" 之类的脏值。生产 ``verdict_status`` 不会把它
    当成 False（最终归 error）；明细也不能按 Python 真值写成 true/false，否则会与
    严格结论自相矛盾。
    """
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return ""
    return f"非布尔:{value!r}"


def _offset_value(value: Any) -> int:
    """把字符偏移规范化为非负整数；bool 不能冒充整数。"""
    if isinstance(value, bool):
        raise ValueError(f"方位词字符偏移非法：{value!r}")
    if isinstance(value, int):
        offset = value
    else:
        text = _cell(value).strip()
        if not text:
            raise ValueError("方位词字符偏移为空；请用新版导出脚本重新生成 CSV")
        try:
            offset = int(text)
        except ValueError as exc:
            raise ValueError(f"方位词字符偏移非法：{text!r}（只接受非负整数）") from exc
    if offset < 0:
        raise ValueError(f"方位词字符偏移非法：{offset}（只接受非负整数）")
    return offset


def _node_word(node: Any) -> str:
    if isinstance(node, dict):
        return _cell(node.get("word"))
    return ""


def _make_row(
    *,
    line_no: Any,
    source_file: Any,
    sentence: Any,
    locative: Any,
    locative_index: Any,
    locative_offset: Any,
    target: Any,
    reference: Any,
    verb: Any,
    verdict: bool | None,
    reason: Any = "",
    status: Any = "",
    grammatical: bool | None = None,
    spatial_coherent: bool | None = None,
    verdict_source: Any = "",
    record_status: Any = "",
    strict_state: str | None = None,
    strict_available: bool = False,
) -> VerdictRow:
    source = _cell(source_file)
    text = sentence_text(sentence)
    loc = _cell(locative)
    index = _cell(locative_index)
    offset = _offset_value(locative_offset)
    base = (source, text, loc, offset)
    return VerdictRow(
        key=(*base, 0),
        base=base,
        line_no=_cell(line_no),
        source_file=source,
        sentence=text,
        locative=loc,
        locative_index=index,
        locative_offset=offset,
        target=_cell(target),
        reference=_cell(reference),
        verb=_cell(verb),
        verdict=verdict,
        reason=_cell(reason),
        status=_cell(status),
        grammatical=grammatical,
        spatial_coherent=spatial_coherent,
        verdict_source=_cell(verdict_source),
        record_status=_cell(record_status) or _cell(status),
        strict_state=strict_state,
        strict_available=strict_available,
    )


def assign_keys(rows: Sequence[VerdictRow]) -> list[VerdictRow]:
    """给同一 base 的多行按原始顺序编号，保证 key 在文件内唯一。"""
    seen: Counter[tuple[str, str, str, int]] = Counter()
    result = []
    for row in rows:
        ordinal = seen[row.base]
        seen[row.base] += 1
        result.append(replace(row, key=(*row.base, ordinal)))
    return result


def _progress(stream, every: int, count: int, unit: str) -> None:
    if every and count % every == 0:
        print(f"已读取 {count} {unit}…", file=stream or sys.stderr, flush=True)


def _csv_bool(value: Any, field: str, source: Path, line_no: int) -> bool | None:
    """CSV 里的布尔字段：空串表示 None，只接受 true/false（大小写不敏感）。"""
    text = _cell(value).strip().lower()
    if not text:
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(
        f"{source} 第 {line_no} 行「{field}」非法：{text!r}（只接受 true/false/空串）"
    )


def read_csv_rows(
    path: str | Path,
    *,
    progress_every: int = 50_000,
    progress_stream=None,
    capabilities: dict | None = None,
) -> list[VerdictRow]:
    """读新版或旧版逐词 CSV。

    ``capabilities["strict_available"]`` 会写回严格状态是否可复算：
    四列全有 = True；四列全无 = 旧版 False；只出现一部分直接报错。
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    rows: list[VerdictRow] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            fieldnames = reader.fieldnames or []
            missing = sorted(CSV_REQUIRED - set(fieldnames))
            if missing:
                hint = (
                    "；旧版 CSV 不含字符偏移，请用新版 "
                    "export_extract_verdicts_csv.py 重新导出"
                    if "方位词字符偏移" in missing
                    else ""
                )
                raise ValueError(f"{source} 缺列：{', '.join(missing)}{hint}")
            strict_present = CSV_STRICT_REQUIRED & set(fieldnames)
            if strict_present and strict_present != CSV_STRICT_REQUIRED:
                missing_strict = sorted(CSV_STRICT_REQUIRED - strict_present)
                raise ValueError(
                    f"{source} 严格状态列不完整，缺：{', '.join(missing_strict)}；"
                    "请用新版 export_extract_verdicts_csv.py 重新导出，"
                    "或删除全部四个严格状态列后按旧版 CSV 比较"
                )
            strict_available = strict_present == CSV_STRICT_REQUIRED
            if capabilities is not None:
                capabilities["strict_available"] = strict_available
            for raw in reader:
                # DictReader 对行尾缺失字段返回 None，对多余字段用 None 作键。
                # 合法空单元格是空串；None 说明这一行的物理列数与表头不一致，
                # 不能把它当成“来源缺失/字段缺失”继续算严格状态。
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError(
                        f"{source} 第 {reader.line_num} 行字段数与表头不一致"
                        "（行被截短或有多余列）"
                    )
                raw_verdict = _cell(raw.get("模型判定")).strip().lower()
                if raw_verdict == "true":
                    verdict = True
                elif raw_verdict == "false":
                    verdict = False
                elif raw_verdict == "unknown":
                    verdict = None
                else:
                    raise ValueError(
                        f"{source} 第 {reader.line_num} 行「模型判定」非法："
                        f"{_cell(raw.get('模型判定'))!r}"
                        "（只接受 true/false/unknown）"
                    )
                try:
                    offset = _offset_value(raw.get("方位词字符偏移"))
                except ValueError as exc:
                    raise ValueError(
                        f"{source} 第 {reader.line_num} 行：{exc}"
                    ) from exc
                sentence_value = _cell(raw.get("原句"))
                locative_value = _cell(raw.get("方位词"))
                if (
                    offset + len(locative_value) > len(sentence_value)
                    or sentence_value[offset : offset + len(locative_value)]
                    != locative_value
                ):
                    raise ValueError(
                        f"{source} 第 {reader.line_num} 行「方位词字符偏移」"
                        f"与「原句/方位词」不一致：offset={offset}, "
                        f"方位词={locative_value!r}"
                    )
                grammatical = spatial_coherent = None
                verdict_source = ""
                strict_state = None
                if strict_available:
                    csv_extractable = _csv_bool(
                        raw.get("extractable"),
                        "extractable",
                        source,
                        reader.line_num,
                    )
                    if csv_extractable != verdict:
                        raise ValueError(
                            f"{source} 第 {reader.line_num} 行「extractable」与"
                            f"「模型判定」不一致：{csv_extractable!r} != {verdict!r}"
                        )
                    grammatical = _csv_bool(
                        raw.get("grammatical"), "grammatical", source, reader.line_num
                    )
                    spatial_coherent = _csv_bool(
                        raw.get("spatial_coherent"),
                        "spatial_coherent",
                        source,
                        reader.line_num,
                    )
                    # 不 strip：生产 verdict_status 按精确字符串判断 limit/error/no_llm，
                    # " limit " 与 "limit" 不是同一状态；往返必须保真。
                    verdict_source = _cell(raw.get("逐词判定来源"))
                    strict_state = strict_verdict_state(
                        {
                            "extractable": verdict,
                            "grammatical": grammatical,
                            "spatial_coherent": spatial_coherent,
                            "source": verdict_source or None,
                        }
                    )
                rows.append(
                    _make_row(
                        line_no=raw.get("JSONL行号"),
                        source_file=raw.get("源文件"),
                        sentence=raw.get("原句"),
                        locative=raw.get("方位词"),
                        locative_index=raw.get("方位词下标"),
                        locative_offset=offset,
                        target=raw.get("目标物"),
                        reference=raw.get("参照物"),
                        verb=raw.get("动词"),
                        verdict=verdict,
                        reason=raw.get("判定理由"),
                        status=raw.get("整句筛选状态"),
                        grammatical=grammatical,
                        spatial_coherent=spatial_coherent,
                        verdict_source=verdict_source,
                        record_status=raw.get("整句筛选状态"),
                        strict_state=strict_state,
                        strict_available=strict_available,
                    )
                )
                _progress(progress_stream, progress_every, len(rows), "行判定")
        except csv.Error as exc:
            raise ValueError(
                f"{source} 第 {reader.line_num} 行 CSV 解析失败：{exc}"
            ) from exc
    return assign_keys(rows)


def _jsonl_locative_offset(
    sentence: Any, index: Any, word: Any, source: Path, line_no: int
) -> int:
    """从 JSONL 的 token 列表计算方位词字符偏移，并校验下标/词形。"""
    if isinstance(sentence, list):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(sentence)
        ):
            raise ValueError(
                f"{source} 第 {line_no} 行 verdict index 非法：{index!r}"
            )
        token = sentence[index]
        if not isinstance(token, str):
            raise ValueError(
                f"{source} 第 {line_no} 行 index={index} 的 token 不是字符串：{token!r}"
            )
        if token != _cell(word):
            raise ValueError(
                f"{source} 第 {line_no} 行 index={index} 的 token={token!r} "
                f"与 verdict.word={_cell(word)!r} 不一致"
            )
        return sum(len(item) for item in sentence[:index] if isinstance(item, str))
    if isinstance(sentence, str):
        text = sentence
        needle = _cell(word)
        positions = []
        start = 0
        while needle:
            found = text.find(needle, start)
            if found < 0:
                break
            positions.append(found)
            start = found + 1
        if len(positions) == 1:
            return positions[0]
        raise ValueError(
            f"{source} 第 {line_no} 行无法从字符串原句确定方位词字符偏移："
            f"word={needle!r} 出现 {len(positions)} 次；请提供 token 列表或字符偏移"
        )
    raise ValueError(
        f"{source} 第 {line_no} 行 sentence 类型非法：{type(sentence).__name__}"
    )


def _jsonl_verdict_rows(
    record: dict[str, Any], line_no: int, source: Path
) -> Iterator[VerdictRow]:
    if "pair_schema_version" in record or "pair_screening" in record:
        raise ValueError(
            f"{source} 第 {line_no} 行是 replace 句对，不能按 extract 原句比较；"
            "请使用 analyze_tools/replace/compare_replace_pairs.py"
        )
    screening = record.get("screening")
    if not isinstance(screening, dict):
        return
    verdicts = screening.get("verdicts")
    if not isinstance(verdicts, list):
        return
    grounding = record.get("localization_grounding")
    candidates = grounding.get("candidates") if isinstance(grounding, dict) else None
    if not isinstance(candidates, list):
        candidates = []
    try:
        candidates_by_index = {
            candidate["index"]: candidate
            for candidate in candidates
            if isinstance(candidate, dict) and "index" in candidate
        }
    except TypeError as exc:
        raise ValueError(
            f"{source} 第 {line_no} 行 candidate index 非法（不可哈希）：{exc}"
        ) from exc
    record_status = record_screening_status(record)
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        extractable = verdict.get("extractable")
        # None 表示这个实际存在的 verdict 没拿到判定（典型 limit/error；旧数据里
        # 可能有逐词 no_llm），按 unknown 保留并参与统计。真实 annotate_no_llm 记录
        # 的 verdicts 为空，不会走到这里。其余非布尔值属于损坏数据，必须报错。
        if extractable is True:
            extractable_value = True
        elif extractable is False:
            extractable_value = False
        elif extractable is None:
            extractable_value = None
        else:
            raise ValueError(
                f"{source} 第 {line_no} 行 verdict extractable 非法："
                f"{extractable!r}（只接受 true/false/None）"
            )
        index = verdict.get("index")
        offset = _jsonl_locative_offset(
            record.get("sentence"), index, verdict.get("word"), source, line_no
        )
        try:
            candidate = candidates_by_index.get(index)
        except TypeError as exc:
            raise ValueError(
                f"{source} 第 {line_no} 行 verdict index 非法（不可哈希）：{exc}"
            ) from exc
        if not isinstance(candidate, dict):
            candidate = {}
        verdict_source = verdict.get("source")
        yield _make_row(
            line_no=line_no,
            source_file=record.get("source_file"),
            sentence=record.get("sentence"),
            locative=verdict.get("word"),
            locative_index=index,
            locative_offset=offset,
            target=_node_word(candidate.get("target")),
            reference=_node_word(candidate.get("reference")),
            verb=_node_word(candidate.get("verb")),
            verdict=extractable_value,
            reason=verdict.get("reason"),
            status=record_status,
            grammatical=verdict.get("grammatical"),
            spatial_coherent=verdict.get("spatial_coherent"),
            verdict_source=_cell(verdict_source),
            record_status=record_status,
            strict_state=strict_verdict_state(verdict),
            strict_available=True,
        )


@dataclass(frozen=True)
class SentenceRecord:
    """一条原句的整句状态；用于句级统计和下游准入比较。"""

    status: str
    eligible: bool


def read_jsonl_rows(
    path: str | Path,
    *,
    progress_every: int = 50_000,
    progress_stream=None,
    sentence_index: dict[tuple[str, str, int], SentenceRecord] | None = None,
) -> list[VerdictRow]:
    """流式读取 JSONL；只保留判定投影，不把整份记录留在内存里。

    传入 ``sentence_index`` 时同时收集**每条记录**（含 ``verdicts=[]`` 的 no_llm）
    的整句状态与准入标志。句键是 ``(源文件, 原句, 同句出现序号)``，与逐词对齐
    同一套“重复按出现顺序编号”的思路；同句重复时仍存在配对歧义。
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    rows: list[VerdictRow] = []
    records = 0
    sentence_seen: Counter[tuple[str, str]] = Counter()
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source} 第 {line_no} 行不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source} 第 {line_no} 行不是 JSON 对象")
            records += 1
            if sentence_index is not None:
                text = sentence_text(record.get("sentence"))
                document = _cell(record.get("source_file"))
                base = (document, text)
                occurrence = sentence_seen[base]
                sentence_seen[base] += 1
                sentence_index[(*base, occurrence)] = SentenceRecord(
                    status=record_screening_status(record),
                    eligible=judgement_eligible(record),
                )
            rows.extend(_jsonl_verdict_rows(record, line_no, source))
            _progress(progress_stream, progress_every, records, "条记录")
    return assign_keys(rows)


def compare_rows(
    left: Sequence[VerdictRow],
    right: Sequence[VerdictRow],
    *,
    left_sentences: dict[tuple[str, str, int], SentenceRecord] | None = None,
    right_sentences: dict[tuple[str, str, int], SentenceRecord] | None = None,
    left_strict_available: bool | None = None,
    right_strict_available: bool | None = None,
    keyed: bool = False,
) -> Comparison:
    # 读取器产出的行已经编号；keyed=True 时不再为每一对比较复制整份投影。
    if not keyed:
        left = assign_keys(list(left))
        right = assign_keys(list(right))
    else:
        left = list(left)
        right = list(right)
    left_by_key = {row.key: row for row in left}
    right_by_key = {row.key: row for row in right}
    if len(left_by_key) != len(left) or len(right_by_key) != len(right):
        raise ValueError("存在重复 key；请确认输入行是否经过 assign_keys 处理")
    common_keys = [row.key for row in left if row.key in right_by_key]
    only_left_keys = [row.key for row in left if row.key not in right_by_key]
    only_right_keys = [row.key for row in right if row.key not in left_by_key]
    return Comparison(
        left=list(left),
        right=list(right),
        common_keys=common_keys,
        only_left_keys=only_left_keys,
        only_right_keys=only_right_keys,
        left_by_key=left_by_key,
        right_by_key=right_by_key,
        left_sentences=left_sentences,
        right_sentences=right_sentences,
        left_strict_available=left_strict_available,
        right_strict_available=right_strict_available,
    )


def _md_cell(value: Any) -> str:
    """转义 Markdown 表格单元格：反斜杠、竖线，换行转成 ``<br>``。"""
    text = _cell(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return text


def _md_row(columns: Sequence[str], row: dict[str, Any]) -> str:
    return "| " + " | ".join(_md_cell(row.get(column, "")) for column in columns) + " |"


def _md_table(
    columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
    aligns: Sequence[str] | None = None,
) -> list[str]:
    align_values = list(aligns) if aligns is not None else ["---"] * len(columns)
    lines = [
        _md_row(columns, {column: column for column in columns}),
        "| " + " | ".join(align_values) + " |",
    ]
    lines.extend(_md_row(columns, row) for row in rows)
    return lines


def _locative_rows(comparison: Comparison) -> list[dict[str, Any]]:
    return [
        {
            "方位词": item["locative"],
            "共同": item.get("common", 0),
            "原始相同": item.get("raw_same", 0),
            "原始不同": item.get("raw_different", 0),
            "严格共同": item.get("strict_common", 0),
            "严格相同": item.get("strict_same", 0),
            "严格不同": item.get("strict_different", 0),
            "有效共同": item.get("effective_common", 0),
            "有效相同": item.get("effective_same", 0),
            "仅左": item.get("only_left", 0),
            "仅右": item.get("only_right", 0),
        }
        for item in comparison.by_locative
    ]


def format_summary(comparison: Comparison, left_path: Path, right_path: Path) -> str:
    c = comparison
    lines = [
        f"左结果: {left_path}",
        f"右结果: {right_path}",
        "对齐口径: (源文件, 原句, 方位词, 方位词字符偏移, 同 base 序号)；"
        "比较字段: 原始 extractable 三态 + 严格五态",
        "",
        f"左侧: 判定={c.left_total} "
        f"(原始 true={c.left_true}, false={c.left_false}, unknown={c.left_unknown})，"
        f"严格={_format_strict_counts(c.left_strict_counts)}，"
        f"完整三元组={c.left_complete}",
        f"右侧: 判定={c.right_total} "
        f"(原始 true={c.right_true}, false={c.right_false}, unknown={c.right_unknown})，"
        f"严格={_format_strict_counts(c.right_strict_counts)}，"
        f"完整三元组={c.right_complete}",
        "",
        f"共同判定项: {c.common}；两侧并集（矩阵分母）: {c.union}",
        f"严格指标可用: {'是' if c.strict_available else '否（缺 strict 列，请重新导出）'}",
    ]
    if c.common:
        common_rate = (c.common_agreement or 0.0) * 100
        union_rate = (c.agreement or 0.0) * 100
        lines.extend(
            [
                f"  相同判断: {c.same}（共同项内 {common_rate:.2f}%）；"
                f"并集口径 {union_rate:.2f}% ({c.same}/{c.union})",
                f"    true→true : {c.matrix[(True, True)]}",
                f"    false→false: {c.matrix[(False, False)]}",
                f"    unknown→unknown: {c.matrix[(None, None)]}",
                f"  不同判断: {c.different}（共同项内 {100 - common_rate:.2f}%）",
                f"    true→false: {c.matrix[(True, False)]}",
                f"    false→true: {c.matrix[(False, True)]}",
                f"    unknown≠bool: {c.different_unknown}",
                f"  其中两侧三元组都完整: {c.common_complete} "
                f"(相同={c.same_complete}, 不同={c.different_complete})",
            ]
        )
    else:
        lines.append("  相同判断: N/A（没有可对齐的判定项）")

    if c.strict_available:
        lines.extend(
            [
                "",
                f"严格逐词状态: 可比较={c.strict_comparable} "
                f"(相同={c.strict_same}, 不同={c.strict_different})；"
                f"并集口径 {_ratio_text(c.strict_same, c.union)}",
                "严格状态转移（非零）:",
            ]
        )
        transitions = [
            f"    {left_state}→{right_state}: {count}"
            for (left_state, right_state), count in sorted(c.strict_matrix.items())
            if count
        ]
        lines.extend(transitions or ["    （无）"])
        lines.extend(
            [
                f"有效判定一致率: {_ratio_text(c.same_effective, c.common_effective, unavailable='N/A (0/0)')}"
                f"（共同有效项={c.common_effective}；error/limit/no_llm 不进入分母）",
                f"有效判定覆盖率: 左={_ratio_text(c.left_effective, c.left_total, unavailable='N/A')}，"
                f"右={_ratio_text(c.right_effective, c.right_total, unavailable='N/A')}",
            ]
        )
    else:
        lines.append("严格指标: N/A（缺 strict 列，请重新导出）")

    if c.sentence_available:
        lines.extend(
            [
                "",
                f"整句状态: 共同={c.sentence_common} 并集={c.sentence_union} "
                f"相同={c.sentence_same}；一致率 {_ratio_text(c.sentence_same, c.sentence_union)}",
                f"下游准入句集: 双方={c.eligible_both}、仅左={c.eligible_left_only}、"
                f"仅右={c.eligible_right_only}、并集={c.eligible_union}；"
                f"重合率 {_ratio_text(c.eligible_both, c.eligible_union)}",
            ]
        )

    union_text = (
        "N/A (0/0)"
        if not c.union
        else f"{(c.agreement or 0.0) * 100:.2f}% ({c.same}/{c.union})"
    )
    lines.extend(
        [
            "",
            f"仅左侧: {c.only_left} "
            f"(true={c.only_left_true}, false={c.only_left_false}, "
            f"unknown={c.only_left_unknown})",
            f"仅右侧: {c.only_right} "
            f"(true={c.only_right_true}, false={c.only_right_false}, "
            f"unknown={c.only_right_unknown})",
            f"一致率（相同/并集）: {union_text}",
            f"三元组文本变化（共同项）: {c.triple_changed} "
            f"(判定相同={c.triple_changed_same_verdict}, "
            f"判定不同={c.triple_changed_diff_verdict})",
            f"JSONL行号不一致（共同项）: {c.line_mismatch}",
            f"重复 base 组数: 左={c.duplicate_groups_left}, 右={c.duplicate_groups_right}",
            f"  涉及共同项: {c.duplicate_common}；次数不等: {c.duplicate_unequal} 组"
            f"（两侧都有={c.duplicate_unequal_both}）",
            f"  顺序歧义: {c.order_ambiguous_groups} 组/{c.order_ambiguous_common} 项；"
            f"若所有重复 base 组内任意重配对，相同判断总数范围="
            f"[{c.same_min_total}, {c.same_max_total}]（当前={c.same}）",
            "",
            "按方位词（Markdown 表格）:",
        ]
    )
    lines.extend(_md_table(LOCATIVE_COLUMNS, _locative_rows(c), LOCATIVE_ALIGNS))
    return "\n".join(lines)


def _short(value: str, limit: int = 60) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _format_example(index: int, left: VerdictRow, right: VerdictRow | None) -> str:
    left_verdict = f"{left.verdict_text}/{left.strict_state_text}"
    triple = " / ".join(part or "∅" for part in left.triple)
    head = (
        f"[{index}] 左行 {left.line_no} / 右行 "
        f"{right.line_no if right else '∅'} | "
        f"{Path(left.source_file).name} | {left.locative}[{left.locative_index}] | "
        f"三元组: {triple}"
    )
    if right is None:
        return f"{head}\n    仅左侧判定: {left_verdict} | 理由: {_short(left.reason)}"
    right_triple = " / ".join(part or "∅" for part in right.triple)
    return (
        f"{head}\n"
        f"    左: {left_verdict} | {_short(left.reason)}\n"
        f"    右: {right.verdict_text}/{right.strict_state_text} | "
        f"三元组: {right_triple} | {_short(right.reason)}"
    )


def format_examples(comparison: Comparison, limit: int) -> str:
    if limit <= 0:
        return ""
    lines = []
    differences = comparison.difference_examples(limit)
    if differences:
        lines.append("\n判定不同样例:")
        lines.extend(
            _format_example(i, left, right)
            for i, (left, right) in enumerate(differences, 1)
        )
    only_left, only_right = comparison.only_examples(limit)
    if only_left:
        lines.append("\n仅左侧样例:")
        lines.extend(
            _format_example(i, row, None) for i, row in enumerate(only_left, 1)
        )
    if only_right:
        lines.append("\n仅右侧样例:")
        lines.extend(
            f"[{i}] 右行 {row.line_no} | {Path(row.source_file).name} | "
            f"{row.locative}[{row.locative_index}] | "
            f"三元组: {' / '.join(part or '∅' for part in row.triple)} | "
            f"判定: {row.verdict_text} | 理由: {_short(row.reason)}"
            for i, row in enumerate(only_right, 1)
        )
    return "\n".join(lines)


def _detail_row(comparison: Comparison, key: tuple, kind: str) -> dict[str, str]:
    left = comparison.left_by_key.get(key)
    right = comparison.right_by_key.get(key)
    source = left or right
    assert source is not None
    if left and right and left.strict_state and right.strict_state:
        strict_kind = "相同" if left.strict_state == right.strict_state else "不同"
    elif left and right:
        strict_kind = "N/A"
    elif left:
        strict_kind = "仅左侧"
    else:
        strict_kind = "仅右侧"
    return {
        "比较结果": kind,
        "左侧行号": left.line_no if left else "",
        "右侧行号": right.line_no if right else "",
        "源文件": source.source_file,
        "原句": source.sentence,
        "方位词": source.locative,
        "方位词字符偏移": str(source.locative_offset),
        "重复序号": str(key[-1]),
        "左方位词下标": left.locative_index if left else "",
        "右方位词下标": right.locative_index if right else "",
        "左目标物": left.target if left else "",
        "左参照物": left.reference if left else "",
        "左动词": left.verb if left else "",
        "右目标物": right.target if right else "",
        "右参照物": right.reference if right else "",
        "右动词": right.verb if right else "",
        "左判定": left.verdict_text if left else "",
        "右判定": right.verdict_text if right else "",
        "左严格状态": left.strict_state_text if left else "",
        "右严格状态": right.strict_state_text if right else "",
        "严格比较结果": strict_kind,
        "左extractable": _bool_cell(left.verdict) if left else "",
        "左grammatical": _bool_cell(left.grammatical) if left else "",
        "左spatial_coherent": _bool_cell(left.spatial_coherent) if left else "",
        "右extractable": _bool_cell(right.verdict) if right else "",
        "右grammatical": _bool_cell(right.grammatical) if right else "",
        "右spatial_coherent": _bool_cell(right.spatial_coherent) if right else "",
        "左逐词来源": left.verdict_source if left else "",
        "右逐词来源": right.verdict_source if right else "",
        "左整句状态": left.record_status if left else "",
        "右整句状态": right.record_status if right else "",
        "三元组是否变化": (
            ("是" if left.triple != right.triple else "否") if left and right else ""
        ),
        "左理由": left.reason if left else "",
        "右理由": right.reason if right else "",
    }


def _same_file(left: Path, right: Path) -> bool:
    try:
        if left.resolve() == right.resolve():
            return True
        return left.samefile(right)
    except (FileNotFoundError, OSError, RuntimeError):
        # resolve() 可能遇到符号链接环（RuntimeError），samefile 可能遇到悬空链接。
        return False


def latest_two_csv(results_dir: str | Path | None = None) -> tuple[Path, Path]:
    """按文件名日期取最新两份 CSV；日期相同再按文件名兜底。"""
    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    candidates = list(directory.glob(CSV_GLOB))

    def sort_key(path: Path) -> tuple[str, str]:
        match = DATE_RE.search(path.name)
        return (match.group(1) if match else "", path.name)

    candidates.sort(key=sort_key)
    if len(candidates) < 2:
        raise FileNotFoundError(
            f"在 {directory} 下找到 {len(candidates)} 份 {CSV_GLOB}，至少需要两份；"
            "请显式传入左右两个结果文件"
        )
    return candidates[-2], candidates[-1]


def discover_csv_files(results_dir: str | Path | None = None) -> list[Path]:
    """发现结果目录下全部筛选 CSV，按文件名日期、再按文件名排序。"""
    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    candidates = list(directory.glob(CSV_GLOB))

    def sort_key(path: Path) -> tuple[str, str]:
        match = DATE_RE.search(path.name)
        return (match.group(1) if match else "", path.name)

    return sorted(candidates, key=sort_key)


def select_latest_csvs(
    count: int, results_dir: str | Path | None = None
) -> list[Path]:
    """取最新 ``count`` 份 CSV；不足时报错而不是静默少选。"""
    if count < 2:
        raise ValueError("--latest 至少为 2")
    candidates = discover_csv_files(results_dir)
    if len(candidates) < count:
        raise FileNotFoundError(
            f"只找到 {len(candidates)} 份 {CSV_GLOB}，少于 --latest {count}"
        )
    return candidates[-count:]


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节数截断，且不切断多字节字符。"""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    raw = raw[:max_bytes]
    while raw:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raw = raw[:-1]
    return ""


def _file_label(path: Path, index: int) -> str:
    """生成稳定、可读、可作为文件名的输入标签。

    标签按 UTF-8 字节数截断：文件名上限是 255 字节而不是 255 个字符，中文
    文件名按字符截断仍可能让 ``pair_csv_name()`` 超过上限。
    """
    name = path.name
    for suffix in (".screening_verdicts.csv", ".jsonl", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = LABEL_PREFIX_RE.sub("", name)
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._-")
    # 左右标签 + "__vs__" + ".csv" 控制在 220 字节以内，给文件系统留足余量。
    name = _truncate_utf8(name, 100)
    if not name:
        name = f"file_{index:02d}"
    return f"{index:02d}_{name}"


def pair_csv_name(left: LoadedFile, right: LoadedFile) -> str:
    return f"{left.label}__vs__{right.label}.csv"


def _unique_default_output_dir() -> Path:
    base = ANALYZE_RESULTS_DIR / COMPARISON_DIR_NAME
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = base / stamp
    suffix = 1
    while target.exists():
        target = base / f"{stamp}_{suffix:02d}"
        suffix += 1
    return target


def plan_output_dir(output_dir: str | Path | None) -> Path:
    """确定输出目录；显式目录必须尚不存在，避免覆盖已有结果。"""
    target = Path(output_dir) if output_dir else _unique_default_output_dir()
    if target.exists() or target.is_symlink():
        raise ValueError(f"输出目录已存在：{target}；请换一个新目录或删除旧目录")
    return target


def make_staging_dir(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )


def _rename_noreplace(staging_dir: Path, target_dir: Path) -> bool:
    """Linux 下用 renameat2(RENAME_NOREPLACE) 原子发布目录。

    返回 False 表示当前平台/文件系统不支持该标志，调用方需走降级路径。
    目标已存在（包括空目录、悬空符号链接）时原子失败，不会替换。
    """
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            return False
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            os.fsencode(staging_dir),
            -100,
            os.fsencode(target_dir),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return True
        error = ctypes.get_errno()
        if error in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            return False
        if error == errno.EEXIST:
            raise FileExistsError(
                error, os.strerror(error), str(target_dir)
            )
        raise OSError(error, os.strerror(error), str(target_dir))
    except OSError:
        raise
    except Exception:
        # ctypes 不可用或平台差异：退回可移植路径。
        return False


def publish_output_dir(staging_dir: Path, target_dir: Path) -> None:
    """发布输出目录，确保不会替换已有目录。

    优先使用 Linux ``renameat2(RENAME_NOREPLACE)`` 一次原子重命名。平台不支持时
    降级为：先用 ``os.mkdir(target)`` 占位，再逐个移入文件，最后移动
    ``summary.md`` 作为完成标志；中途失败会删除本次创建的目标目录和暂存目录。
    """
    if _rename_noreplace(staging_dir, target_dir):
        return
    os.mkdir(target_dir)
    try:
        entries = sorted(staging_dir.iterdir(), key=lambda item: item.name)
        # summary.md 最后移动：外部看到它时，其余 pair CSV 已经就位。
        entries.sort(key=lambda item: item.name == SUMMARY_FILENAME)
        for entry in entries:
            os.replace(entry, target_dir / entry.name)
        staging_dir.rmdir()
    except BaseException:
        shutil.rmtree(target_dir, ignore_errors=True)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def resolve_input_paths(
    inputs: Sequence[str],
    latest: int | None,
    max_files: int,
) -> tuple[list[Path], str]:
    if inputs:
        if latest is not None:
            raise ValueError("--latest 只能与“不传位置参数”的自动发现模式一起使用")
        paths = [Path(item) for item in inputs]
    elif latest is not None:
        paths = select_latest_csvs(latest)
    else:
        paths = discover_csv_files()

    if len(paths) < 2:
        raise ValueError(
            f"至少需要两个输入文件；当前只有 {len(paths)} 个"
            "（不传位置参数时会自动发现 results/extract_result 下的全部 CSV）"
        )
    if max_files and len(paths) > max_files:
        raise ValueError(
            f"输入文件数为 {len(paths)}，超过 --max-files {max_files}；"
            "请显式提高上限或改为分批比较"
        )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{path}")
    formats = {detect_format(path) for path in paths}
    if len(formats) != 1:
        raise ValueError(
            "输入格式不一致："
            + ", ".join(f"{path.suffix}={detect_format(path)}" for path in paths)
            + "；请比较同一种结果文件"
        )
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if _same_file(left, right):
                raise ValueError(f"存在重复输入文件：{left} 与 {right}")
    return paths, formats.pop()


def load_input_files(
    paths: Sequence[Path],
    fmt: str,
    *,
    progress_every: int = 50_000,
) -> list[LoadedFile]:
    files: list[LoadedFile] = []
    for index, path in enumerate(paths, 1):
        rows, _, strict_available, sentences = load_rows(
            path, progress_every=progress_every
        )
        files.append(
            LoadedFile(
                index=index,
                path=path,
                fmt=fmt,
                label=_file_label(path, index),
                rows=rows,
                strict_available=strict_available,
                sentences=sentences,
            )
        )
    return files


def write_pair_csv(comparison: Comparison, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=DETAIL_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for key in comparison.common_keys:
            left = comparison.left_by_key[key]
            right = comparison.right_by_key[key]
            kind = "相同" if left.verdict == right.verdict else "不同"
            writer.writerow(_detail_row(comparison, key, kind))
        for key in comparison.only_left_keys:
            writer.writerow(_detail_row(comparison, key, "仅左侧"))
        for key in comparison.only_right_keys:
            writer.writerow(_detail_row(comparison, key, "仅右侧"))


def compare_all_pairs(
    files: Sequence[LoadedFile],
    staging_dir: Path,
    *,
    progress_stream=None,
) -> list[PairSummary]:
    summaries: list[PairSummary] = []
    pair_total = len(files) * (len(files) - 1) // 2
    pair_no = 0
    for index, left in enumerate(files):
        for right in files[index + 1 :]:
            pair_no += 1
            comparison = compare_rows(
                left.rows,
                right.rows,
                left_sentences=left.sentences,
                right_sentences=right.sentences,
                left_strict_available=left.strict_available,
                right_strict_available=right.strict_available,
                keyed=True,
            )
            csv_name = pair_csv_name(left, right)
            write_pair_csv(comparison, staging_dir / csv_name)
            summary = pair_summary_from_comparison(
                left, right, csv_name, comparison
            )
            summaries.append(summary)
            # 及时释放本对的 Comparison（含左右行列表与两张 5x5 表），避免下一对
            # 构造期间旧对象仍驻留，抬高多文件比较的峰值内存。
            del comparison
            print(
                f"[{pair_no}/{pair_total}] {left.label} vs {right.label}: "
                f"共同={summary.common}, 并集={summary.union}, "
                f"原始一致率={summary.agreement_text}, "
                f"严格一致率={summary.strict_agreement_text}, "
                f"有效判定={summary.effective_agreement_text}, "
                f"句级={summary.sentence_agreement_text}, "
                f"准入重合={summary.eligibility_overlap_text}",
                file=progress_stream or sys.stdout,
            )
    return summaries


def format_rate_matrix(
    files: Sequence[LoadedFile],
    summaries: Sequence[PairSummary],
    value_attr: str,
) -> list[str]:
    """按 ``PairSummary`` 的某个 ``*_text`` 属性生成文件间一致率矩阵。"""
    lookup = {
        (summary.left.index, summary.right.index): summary
        for summary in summaries
    }
    header = ["文件"] + [item.label for item in files]
    lines = [
        "| " + " | ".join(_md_cell(cell) for cell in header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(files)) + " |",
    ]
    for left in files:
        cells = [_md_cell(left.label)]
        for right in files:
            if left.index == right.index:
                cells.append("—")
                continue
            summary = lookup.get((min(left.index, right.index), max(left.index, right.index)))
            cells.append(getattr(summary, value_attr) if summary else "N/A")
        lines.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    return lines


def format_matrix_lines(
    files: Sequence[LoadedFile], summaries: Sequence[PairSummary]
) -> list[str]:
    """原始 extractable 一致率矩阵（兼容旧调用名）。"""
    return format_rate_matrix(files, summaries, "agreement_text")


def format_overall_markdown(
    files: Sequence[LoadedFile], summaries: Sequence[PairSummary]
) -> str:
    lines = [
        "# extract 方位词筛选结果总体对比",
        "",
        "本报告分三层对比筛选结果：",
        "",
        "- **原始 extractable**：只看该词“是否空间可提取”的单项判断（与旧报告同口径，不代表能进入下游）。",
        "- **严格逐词状态**：结合语法通顺、空间自洽和调用来源后的最终逐词状态；仅 `true` / `false` 为有效判定，`error`、`limit`、`no_llm` 各自独立。",
        "- **整句与下游准入**（仅 JSONL）：整条语料只有未出错且通过（或未启用 LLM 的 `no_llm`）时，才允许进入后续 replace/标注。",
        "",
        "## 输入文件",
        "",
        "以下标签（如 `01_...`）用于下文各对比表；`完整三元组` 指目标物、参照物、动词均非空；`同词位重复组` 指同一文件内相同 `(源文件, 原句, 方位词, 字符偏移)` 出现多次的组数。",
        "",
    ]
    lines.extend(
        _md_table(
            [
                "编号",
                "标签",
                "文件",
                "格式",
                "逐词判定数",
                "完整三元组",
                "同词位重复组",
            ],
            [
                {
                    "编号": f"{item.index:02d}",
                    "标签": item.label,
                    "文件": item.path,
                    "格式": item.fmt,
                    "逐词判定数": item.total,
                    "完整三元组": item.complete_count,
                    "同词位重复组": item.duplicate_groups,
                }
                for item in files
            ],
            ["---", "---", "---", "---", "---:", "---:", "---:"],
        )
    )
    lines.extend(
        [
            "",
            "### 逐词状态分布",
            "",
            "`有效判定 = 严格true + 严格false`；`有效覆盖率 = 有效判定 / 逐词判定数`。",
            "",
        ]
    )
    lines.extend(
        _md_table(
            [
                "标签",
                "原始true",
                "原始false",
                "原始unknown",
                "严格true",
                "严格false",
                "error",
                "limit",
                "no_llm",
                "有效判定",
                "有效覆盖率",
            ],
            [
                {
                    "标签": item.label,
                    "原始true": item.true_count,
                    "原始false": item.false_count,
                    "原始unknown": item.unknown_count,
                    "严格true": item.strict_count("true") if item.strict_available else "—",
                    "严格false": item.strict_count("false") if item.strict_available else "—",
                    "error": item.strict_count("error") if item.strict_available else "—",
                    "limit": item.strict_count("limit") if item.strict_available else "—",
                    "no_llm": item.strict_count("no_llm") if item.strict_available else "—",
                    "有效判定": item.effective_count if item.strict_available else "—",
                    "有效覆盖率": (
                        _ratio_text(item.effective_count, item.total, unavailable="N/A")
                        if item.strict_available
                        else "—"
                    ),
                }
                for item in files
            ],
            ["---"] + ["---:"] * 10,
        )
    )
    lines.extend(
        [
            "",
            "## 句级状态与下游准入",
            "",
            "按每条原句统计；`准入句` 指整句状态为 `true` 或 `no_llm`、可进入后续 replace/标注的句子（CSV 不含无逐词判定的记录，无法统计整句全量）。",
            "",
        ]
    )
    sentence_rows = []
    for item in files:
        if item.sentences is None:
            sentence_rows.append(
                {
                    "编号": f"{item.index:02d}",
                    "标签": item.label,
                    "句数": "N/A（需 JSONL）",
                    "true": "—",
                    "false": "—",
                    "error": "—",
                    "limit": "—",
                    "no_llm": "—",
                    "准入句数": "—",
                    "准入率": "—",
                }
            )
            continue
        sentence_rows.append(
            {
                "编号": f"{item.index:02d}",
                "标签": item.label,
                "句数": item.sentence_total,
                "true": item.sentence_status_counts.get("true", 0),
                "false": item.sentence_status_counts.get("false", 0),
                "error": item.sentence_status_counts.get("error", 0),
                "limit": item.sentence_status_counts.get("limit", 0),
                "no_llm": item.sentence_status_counts.get("no_llm", 0),
                "准入句数": item.sentence_eligible_count,
                "准入率": _ratio_text(
                    item.sentence_eligible_count,
                    item.sentence_total,
                    unavailable="N/A (0/0)",
                ),
            }
        )
    lines.extend(
        _md_table(
            ["编号", "标签", "句数", "true", "false", "error", "limit", "no_llm", "准入句数", "准入率"],
            sentence_rows,
            ["---", "---"] + ["---:"] * 8,
        )
    )
    lines.extend(
        [
            "",
            "## 两两对比：词级",
            "",
            "`共同` 为两侧按同一词位对齐的判定数；`并集 = 共同 + 仅左 + 仅右`；`有效共同` 仅包含两侧均为严格 `true`/`false` 的共同项。",
            "",
        ]
    )
    lines.extend(
        _md_table(
            [
                "#",
                "左",
                "右",
                "共同",
                "并集",
                "原始相同",
                "原始一致率",
                "严格相同",
                "严格一致率",
                "有效共同",
                "有效相同",
                "有效一致率",
                "仅左",
                "仅右",
                "三元组变化",
                "明细 CSV",
            ],
            [
                {
                    "#": index,
                    "左": summary.left.label,
                    "右": summary.right.label,
                    "共同": summary.common,
                    "并集": summary.union,
                    "原始相同": summary.same,
                    "原始一致率": summary.agreement_text,
                    "严格相同": summary.strict_same if summary.strict_available else "—",
                    "严格一致率": summary.strict_agreement_text,
                    "有效共同": summary.common_effective if summary.strict_available else "—",
                    "有效相同": summary.same_effective if summary.strict_available else "—",
                    "有效一致率": summary.effective_agreement_text,
                    "仅左": summary.only_left,
                    "仅右": summary.only_right,
                    "三元组变化": summary.triple_changed,
                    "明细 CSV": f"[{summary.csv_name}]({summary.csv_name})",
                }
                for index, summary in enumerate(summaries, 1)
            ],
            ["---", "---", "---"] + ["---:"] * 12 + ["---"],
        )
    )
    lines.extend(
        [
            "",
            "## 两两对比：句级",
            "",
            "按 `(源文件, 原句, 同句出现序号)` 对齐原句；`准入并集 = 双方准入 + 仅左准入 + 仅右准入`。",
            "",
        ]
    )
    lines.extend(
        _md_table(
            [
                "#",
                "左",
                "右",
                "共同句",
                "句并集",
                "句状态相同",
                "句状态一致率",
                "双方准入",
                "仅左准入",
                "仅右准入",
                "准入并集",
                "准入重合率",
            ],
            [
                {
                    "#": index,
                    "左": summary.left.label,
                    "右": summary.right.label,
                    "共同句": summary.sentence_common if summary.sentence_available else "—",
                    "句并集": summary.sentence_union if summary.sentence_available else "—",
                    "句状态相同": summary.sentence_same if summary.sentence_available else "—",
                    "句状态一致率": summary.sentence_agreement_text,
                    "双方准入": summary.eligible_both if summary.sentence_available else "—",
                    "仅左准入": summary.eligible_left_only if summary.sentence_available else "—",
                    "仅右准入": summary.eligible_right_only if summary.sentence_available else "—",
                    "准入并集": summary.eligible_union if summary.sentence_available else "—",
                    "准入重合率": summary.eligibility_overlap_text,
                }
                for index, summary in enumerate(summaries, 1)
            ],
            ["---", "---", "---"] + ["---:"] * 9,
        )
    )
    lines.extend(
        [
            "",
            "## 逐词严格状态转移（非零）",
            "",
            "仅统计两侧均存在且严格状态可计算的共同项；`error`、`limit`、`no_llm` 各自独立，不互算相同。",
            "",
        ]
    )
    transition_rows = []
    for summary in summaries:
        for left_state in STRICT_STATES:
            for right_state in STRICT_STATES:
                count = summary.strict_transitions.get((left_state, right_state), 0)
                if count:
                    transition_rows.append(
                        {
                            "对比": f"{summary.left.label} vs {summary.right.label}",
                            "左状态": left_state,
                            "右状态": right_state,
                            "数量": count,
                            "占共同项": (
                                f"{count / summary.strict_comparable * 100:.2f}%"
                                if summary.strict_comparable
                                else "N/A"
                            ),
                        }
                    )
    if transition_rows:
        lines.extend(
            _md_table(
                ["对比", "左状态", "右状态", "数量", "占共同项"],
                transition_rows,
                ["---", "---", "---", "---:", "---:"],
            )
        )
    else:
        lines.append("无可统计的严格状态转移（可能缺列或没有共同项）。")
    lines.extend(
        [
            "",
            "## 一致性矩阵",
            "",
            "对角线 `—` 表示不与自身比较；分母为 0 时记 `N/A (0/0)`；旧版 CSV 缺严格列时记 `N/A（缺列，请重新导出）`；CSV 不计算句级指标时记 `N/A（需 JSONL）`。",
            "",
            "### 原始 extractable 一致率",
            "",
            "口径：只看 `extractable` 是否一致（`相同数 / 两侧词键并集`；`unknown→unknown` 记为相同）。",
            "",
        ]
    )
    lines.extend(format_rate_matrix(files, summaries, "agreement_text"))
    lines.extend(
        [
            "",
            "### 严格逐词状态一致率",
            "",
            "口径：严格五态完全一致才算相同（`相同数 / 两侧词键并集`；`error`、`limit`、`no_llm` 各自独立，不互算相同）。",
            "",
        ]
    )
    lines.extend(format_rate_matrix(files, summaries, "strict_agreement_text"))
    lines.extend(
        [
            "",
            "### 有效判定一致率",
            "",
            "口径：只在双方都给出严格 `true`/`false` 的共同项内比较（`有效相同 / 有效共同`；未完成项不进入分母）。",
            "",
        ]
    )
    lines.extend(format_rate_matrix(files, summaries, "effective_agreement_text"))
    lines.extend(
        [
            "",
            "### 句级状态一致率",
            "",
            "口径：整句筛选状态是否一致（`相同句数 / 两侧句键并集`；仅 JSONL）。",
            "",
        ]
    )
    lines.extend(format_rate_matrix(files, summaries, "sentence_agreement_text"))
    lines.extend(
        [
            "",
            "### 下游准入句集重合率",
            "",
            "口径：双方都能进入下游的句子在至少一侧准入句子中的占比（`双方准入 / 准入并集`；准入指整句为 `true` 或 `no_llm`；仅 JSONL）。",
            "",
        ]
    )
    lines.extend(format_rate_matrix(files, summaries, "eligibility_overlap_text"))
    lines.extend(
        [
            "",
            "## 原始 extractable 重复 base 歧义警告",
            "",
            "同一源文件里，相同的 `(原句, 方位词, 字符偏移)` 在单文件内出现多次时，只能按出现顺序第 1 对第 1、第 2 对第 2 配对。",
            "本节只量化**原始 extractable** 的相同数可能范围；严格五态与句级指标同样按出现顺序配对、存在相同歧义，但未单独计算其范围。",
            "",
        ]
    )
    warnings = [
        summary
        for summary in summaries
        if summary.duplicate_unequal_both
        or summary.order_ambiguous_groups
        or summary.same_min_total != summary.same_max_total
    ]
    if warnings:
        lines.extend(
            _md_table(
                ["对比", "两侧次数不等组", "顺序歧义组", "顺序歧义项", "原始相同数可能范围"],
                [
                    {
                        "对比": f"{item.left.label} vs {item.right.label}",
                        "两侧次数不等组": item.duplicate_unequal_both,
                        "顺序歧义组": item.order_ambiguous_groups,
                        "顺序歧义项": item.order_ambiguous_common,
                        "原始相同数可能范围": f"[{item.same_min_total}, {item.same_max_total}]"
                        f"（按顺序={item.same}）",
                    }
                    for item in warnings
                ],
                ["---", "---:", "---:", "---:", "---:"],
            )
        )
    else:
        lines.append("未发现原始 extractable 的重复词位配对歧义。")
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- **对齐方式**：词级按 `(源文件, 原句, 方位词, 字符偏移, 同位置出现序号)` 对齐；句级按 `(源文件, 原句, 同句出现序号)` 对齐。JSONL 行号、三元组文本与整句状态均不参与词级对齐。",
            "- **严格判定规则**：严格 `true` 要求 `extractable`、`grammatical`、`spatial_coherent` 全为 true 且来源不是 `error`/`limit`/`no_llm`；来源不是上述三类且三项中任一为 false 时为严格 `false`；其余（如缺少字段、类型异常）为 `error`。",
            "- **三元组变化**：仅统计共同项中 `(目标物, 参照物, 动词)` 文本是否不同；混合真值句中被判 `false` 的词位三元组可能已被剪除。",
            "- **无逐词判定的记录**：未启用 LLM 且没有逐词判定的 `no_llm` 句子只计入句级统计，不会伪造词级行。",
            "- **旧版 CSV 与明细**：缺少严格四列的旧版 CSV 仍可输出原始一致率，严格指标记 `N/A（缺列，请重新导出）`；同目录下的两两明细 CSV 保留全部共同项和仅单侧项。",
            "",
        ]
    )
    return "\n".join(lines)


def write_overall_markdown(
    path: Path, files: Sequence[LoadedFile], summaries: Sequence[PairSummary]
) -> None:
    path.write_text(format_overall_markdown(files, summaries), encoding="utf-8")


def print_terminal_summary(
    files: Sequence[LoadedFile], summaries: Sequence[PairSummary], output_dir: Path
) -> None:
    print(f"已加载 {len(files)} 个文件，完成 {len(summaries)} 对比较")
    print()
    print(
        "| 编号 | 文件 | 判定数 | 原始true | 原始false | 原始unknown | "
        "严格true | 严格false | error | limit | no_llm | 有效判定 | 完整三元组 | 重复 base 组 |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in files:
        print(
            f"| {item.index:02d} | {item.path} | {item.total} | "
            f"{item.true_count} | {item.false_count} | {item.unknown_count} | "
            f"{item.strict_count('true') if item.strict_available else '—'} | "
            f"{item.strict_count('false') if item.strict_available else '—'} | "
            f"{item.strict_count('error') if item.strict_available else '—'} | "
            f"{item.strict_count('limit') if item.strict_available else '—'} | "
            f"{item.strict_count('no_llm') if item.strict_available else '—'} | "
            f"{item.effective_count if item.strict_available else '—'} | "
            f"{item.complete_count} | {item.duplicate_groups} |"
        )
    print()
    print(
        "| 对比 | 共同 | 并集 | 原始一致率 | 严格一致率 | 有效一致率 | "
        "句级一致率 | 准入重合率 | 仅左 | 仅右 | 明细 CSV |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in summaries:
        print(
            f"| {summary.left.label} vs {summary.right.label} | "
            f"{summary.common} | {summary.union} | "
            f"{summary.agreement_text} | {summary.strict_agreement_text} | "
            f"{summary.effective_agreement_text} | {summary.sentence_agreement_text} | "
            f"{summary.eligibility_overlap_text} | {summary.only_left} | "
            f"{summary.only_right} | {summary.csv_name} |"
        )
    print()
    print("原始 extractable 一致率矩阵:")
    print("\n".join(format_rate_matrix(files, summaries, "agreement_text")))
    print()
    print("严格逐词状态一致率矩阵:")
    print("\n".join(format_rate_matrix(files, summaries, "strict_agreement_text")))
    print()
    print("有效判定一致率矩阵:")
    print("\n".join(format_rate_matrix(files, summaries, "effective_agreement_text")))
    print()
    print("句级状态一致率矩阵:")
    print("\n".join(format_rate_matrix(files, summaries, "sentence_agreement_text")))
    print()
    print(f"输出目录: {output_dir}")
    print(f"总体报告: {output_dir / SUMMARY_FILENAME}")


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    raise ValueError(f"无法识别输入格式（只支持 .csv / .jsonl）：{path}")


def load_rows(
    path: str | Path,
    *,
    progress_every: int = 50_000,
    progress_stream=None,
) -> tuple[
    list[VerdictRow],
    str,
    bool,
    dict[tuple[str, str, int], SentenceRecord] | None,
]:
    """读一个输入文件，返回 ``(rows, fmt, strict_available, sentences)``。

    ``sentences`` 只有 JSONL 才非 None；CSV 会漏掉 ``verdicts=[]`` 的记录，
    无法还原全量句级指标，因此不能用词行反推句级统计。
    """
    fmt = detect_format(path)
    if fmt == "csv":
        capabilities: dict = {}
        rows = read_csv_rows(
            path,
            progress_every=progress_every,
            progress_stream=progress_stream,
            capabilities=capabilities,
        )
        return rows, fmt, bool(capabilities.get("strict_available")), None
    sentences: dict[tuple[str, str, int], SentenceRecord] = {}
    rows = read_jsonl_rows(
        path,
        progress_every=progress_every,
        progress_stream=progress_stream,
        sentence_index=sentences,
    )
    return rows, fmt, True, sentences


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 extract 原句的筛选判定，输出两两 CSV 和一致性矩阵（不支持 replace）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法:")[-1],
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        metavar="INPUT",
        help="两个或更多 .csv/.jsonl 结果文件；省略时自动发现全部筛选 CSV",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help=f"输出目录（必须尚不存在；默认 analyze_results/"
        f"{COMPARISON_DIR_NAME}/<时间戳>）",
    )
    parser.add_argument(
        "--latest",
        type=int,
        metavar="N",
        help="仅在不传位置参数时使用：只比较最新的 N 份 CSV",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help=f"最多加载多少个输入文件（默认 {DEFAULT_MAX_FILES}；0 表示不限制）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="每读取多少行/记录打印一次进度（默认 50000；0 关闭）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.max_files < 0:
            raise ValueError("--max-files 不能为负数")
        paths, fmt = resolve_input_paths(
            args.inputs, args.latest, args.max_files
        )
        target_dir = plan_output_dir(args.output_dir)
        files = load_input_files(
            paths, fmt, progress_every=max(0, args.progress_every)
        )
        staging_dir = make_staging_dir(target_dir)
        try:
            summaries = compare_all_pairs(files, staging_dir)
            write_overall_markdown(
                staging_dir / SUMMARY_FILENAME, files, summaries
            )
            publish_output_dir(staging_dir, target_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        print_terminal_summary(files, summaries, target_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if any(summary.union == 0 for summary in summaries):
        print(
            "\n警告：至少一对两侧都没有判定行，矩阵中对应单元格为 N/A；"
            "其他对比和输出文件仍然有效。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
