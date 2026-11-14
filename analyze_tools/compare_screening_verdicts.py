#!/usr/bin/env python3
# encoding: utf8

"""对比多个方位词筛选结果，输出两两 CSV 与总体一致性矩阵。

比较对象是 ``results/extract_result`` 下同名的
``location_corpus_with_locative_split_*.screening_verdicts.csv``（或原始 ``.jsonl``）。
一行 CSV / 一个布尔 verdict 表示模型对某个方位词候选的一次 true/false 判定，其
``(目标物, 参照物, 动词)`` 就是本项目所说的三元组。

对齐口径（默认）::

    base = (源文件, 原句, 方位词, 方位词下标)
    key  = base + (同一 base 在该文件中按原始顺序出现的序号,)

``JSONL行号`` **不参与** 对齐：不同批次的结果会因记录增删而整体平移，用它对齐会把
大量相同项误判为“仅单侧”。三元组本身也不参与 key：角色可能为空、grounding 也可能
变化；三元组文本是否变化单独统计，不混进 true/false 翻转。

同一 ``base`` 在一侧重复出现时，按原始顺序第 1 对第 1、第 2 对第 2 配对；若一侧多出
或缺少其中一条，序号配对只能保证确定性，无法消除歧义，因此输出会单独打印“重复 base
组数”和一致率可能范围。

输入 2 个或更多文件时，脚本对每一对文件生成一个逐项明细 CSV，并额外写一份
``summary.md``，其中包含每个文件的统计、两两汇总表和一致性矩阵。一致性矩阵的单元格
是 ``一致率（相同/共同）``；对角线写 ``—``，无共同项写 ``N/A (0/0)``。

脚本只读输入；差异本身不算失败，返回 0。只有至少一对完全没有共同判定项才返回 1；
无法开始比较（少于两个文件、重复输入、格式混合、文件损坏、输出冲突）返回 2。

用法::

    # 不传参数：自动发现 results/extract_result 下全部筛选 CSV
    .venv/bin/python analyze_tools/compare_screening_verdicts.py

    # 只比较最新两份 CSV
    .venv/bin/python analyze_tools/compare_screening_verdicts.py --latest 2

    # 显式指定多个 CSV / JSONL
    .venv/bin/python analyze_tools/compare_screening_verdicts.py \
        results/extract_result/location_corpus_with_locative_split_20261029.screening_verdicts.csv \
        results/extract_result/location_corpus_with_locative_split_20261031.screening_verdicts.csv \
        results/extract_result/location_corpus_with_locative_split_20261103.screening_verdicts.csv

    # 指定输出目录（必须尚不存在）
    .venv/bin/python analyze_tools/compare_screening_verdicts.py \
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

ROOT_DIR = Path(__file__).resolve().parents[1]
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
        "目标物",
        "参照物",
        "动词",
        "模型判定",
    }
)

DETAIL_COLUMNS = [
    "比较结果",
    "左侧行号",
    "右侧行号",
    "源文件",
    "原句",
    "方位词",
    "方位词下标",
    "左目标物",
    "左参照物",
    "左动词",
    "右目标物",
    "右参照物",
    "右动词",
    "左判定",
    "右判定",
    "三元组是否变化",
    "左理由",
    "右理由",
]

LOCATIVE_COLUMNS = [
    "方位词",
    "共同",
    "相同",
    "不同",
    "true→false",
    "false→true",
    "仅左",
    "仅右",
]
LOCATIVE_ALIGNS = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]


@dataclass(frozen=True)
class VerdictRow:
    """一个可比较的逐方位词判定投影。"""

    key: tuple[str, str, str, str, int]
    base: tuple[str, str, str, str]
    line_no: str
    source_file: str
    sentence: str
    locative: str
    locative_index: str
    target: str
    reference: str
    verb: str
    verdict: bool
    reason: str
    status: str

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.target, self.reference, self.verb)

    @property
    def complete(self) -> bool:
        """目标物、参照物、动词三者都不为空。"""
        return bool(self.target and self.reference and self.verb)


@dataclass
class Comparison:
    left: list[VerdictRow]
    right: list[VerdictRow]
    common_keys: list[tuple]
    only_left_keys: list[tuple]
    only_right_keys: list[tuple]
    left_by_key: dict[tuple, VerdictRow]
    right_by_key: dict[tuple, VerdictRow]

    def __post_init__(self) -> None:
        self.left_total = len(self.left)
        self.right_total = len(self.right)
        self.left_true = sum(row.verdict for row in self.left)
        self.left_false = self.left_total - self.left_true
        self.right_true = sum(row.verdict for row in self.right)
        self.right_false = self.right_total - self.right_true
        self.left_complete = sum(row.complete for row in self.left)
        self.right_complete = sum(row.complete for row in self.right)

        self.common = len(self.common_keys)
        self.only_left = len(self.only_left_keys)
        self.only_right = len(self.only_right_keys)

        self.matrix: Counter[tuple[bool, bool]] = Counter()
        self.same = 0
        self.different = 0
        self.common_complete = 0
        self.same_complete = 0
        self.different_complete = 0
        self.triple_changed = 0
        self.triple_changed_same_verdict = 0
        self.triple_changed_diff_verdict = 0
        self.line_mismatch = 0
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

        self.only_left_true = sum(self.left_by_key[key].verdict for key in self.only_left_keys)
        self.only_left_false = self.only_left - self.only_left_true
        self.only_right_true = sum(self.right_by_key[key].verdict for key in self.only_right_keys)
        self.only_right_false = self.only_right - self.only_right_true

        left_by_base: dict[tuple[str, str, str, str], list[VerdictRow]] = {}
        for row in self.left:
            left_by_base.setdefault(row.base, []).append(row)
        right_by_base: dict[tuple[str, str, str, str], list[VerdictRow]] = {}
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
                left_true = sum(row.verdict for row in left_rows)
                left_false = len(left_rows) - left_true
                right_true = sum(row.verdict for row in right_rows)
                right_false = len(right_rows) - right_true
                duplicate_same_max += min(left_true, right_true) + min(
                    left_false, right_false
                )
                cross_possible = min(left_true, right_false) + min(
                    left_false, right_true
                )
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

        self.by_locative = self._by_locative()

    @property
    def agreement(self) -> float | None:
        return self.same / self.common if self.common else None

    def _by_locative(self) -> list[dict[str, Any]]:
        stats: dict[str, Counter[str]] = {}
        for key in self.common_keys:
            left = self.left_by_key[key]
            right = self.right_by_key[key]
            item = stats.setdefault(left.locative, Counter())
            item["common"] += 1
            if left.verdict == right.verdict:
                item["same"] += 1
            else:
                item["different"] += 1
            if left.verdict and not right.verdict:
                item["tf"] += 1
            elif not left.verdict and right.verdict:
                item["ft"] += 1
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

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def true_count(self) -> int:
        return sum(row.verdict for row in self.rows)

    @property
    def false_count(self) -> int:
        return self.total - self.true_count

    @property
    def complete_count(self) -> int:
        return sum(row.complete for row in self.rows)

    @property
    def duplicate_groups(self) -> int:
        return sum(
            1 for count in Counter(row.base for row in self.rows).values() if count > 1
        )


@dataclass
class PairSummary:
    """一对比较的轻量汇总；不持有 Comparison，避免 N 文件时累积 O(N^2) 内存。"""

    left: LoadedFile
    right: LoadedFile
    csv_name: str
    common: int
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

    @property
    def agreement(self) -> float | None:
        return self.same / self.common if self.common else None

    @property
    def agreement_text(self) -> str:
        if not self.common:
            return "N/A (0/0)"
        return f"{self.agreement * 100:.2f}% ({self.same}/{self.common})"


def pair_summary_from_comparison(
    left: LoadedFile, right: LoadedFile, csv_name: str, comparison: Comparison
) -> PairSummary:
    return PairSummary(
        left=left,
        right=right,
        csv_name=csv_name,
        common=comparison.common,
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
    target: Any,
    reference: Any,
    verb: Any,
    verdict: bool,
    reason: Any = "",
    status: Any = "",
) -> VerdictRow:
    source = _cell(source_file)
    text = sentence_text(sentence)
    loc = _cell(locative)
    index = _cell(locative_index)
    base = (source, text, loc, index)
    return VerdictRow(
        key=(*base, 0),
        base=base,
        line_no=_cell(line_no),
        source_file=source,
        sentence=text,
        locative=loc,
        locative_index=index,
        target=_cell(target),
        reference=_cell(reference),
        verb=_cell(verb),
        verdict=verdict,
        reason=_cell(reason),
        status=_cell(status),
    )


def assign_keys(rows: Sequence[VerdictRow]) -> list[VerdictRow]:
    """给同一 base 的多行按原始顺序编号，保证 key 在文件内唯一。"""
    seen: Counter[tuple[str, str, str, str]] = Counter()
    result = []
    for row in rows:
        ordinal = seen[row.base]
        seen[row.base] += 1
        result.append(replace(row, key=(*row.base, ordinal)))
    return result


def _progress(stream, every: int, count: int, unit: str) -> None:
    if every and count % every == 0:
        print(f"已读取 {count} {unit}…", file=stream or sys.stderr, flush=True)


def read_csv_rows(
    path: str | Path,
    *,
    progress_every: int = 50_000,
    progress_stream=None,
) -> list[VerdictRow]:
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
                raise ValueError(f"{source} 缺列：{', '.join(missing)}")
            for raw in reader:
                raw_verdict = _cell(raw.get("模型判定")).strip().lower()
                if raw_verdict == "true":
                    verdict = True
                elif raw_verdict == "false":
                    verdict = False
                else:
                    raise ValueError(
                        f"{source} 第 {reader.line_num} 行「模型判定」非法："
                        f"{_cell(raw.get('模型判定'))!r}（只接受 true/false）"
                    )
                rows.append(
                    _make_row(
                        line_no=raw.get("JSONL行号"),
                        source_file=raw.get("源文件"),
                        sentence=raw.get("原句"),
                        locative=raw.get("方位词"),
                        locative_index=raw.get("方位词下标"),
                        target=raw.get("目标物"),
                        reference=raw.get("参照物"),
                        verb=raw.get("动词"),
                        verdict=verdict,
                        reason=raw.get("判定理由"),
                        status=raw.get("整句筛选状态"),
                    )
                )
                _progress(progress_stream, progress_every, len(rows), "行判定")
        except csv.Error as exc:
            raise ValueError(
                f"{source} 第 {reader.line_num} 行 CSV 解析失败：{exc}"
            ) from exc
    return assign_keys(rows)


def _jsonl_verdict_rows(
    record: dict[str, Any], line_no: int, source: Path
) -> Iterator[VerdictRow]:
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
    status = screening.get("status")
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        extractable = verdict.get("extractable")
        # None 表示未送模型（limit/error/no_llm），按导出器口径跳过；
        # 其余非布尔值属于损坏数据，必须报错而不是静默当成“未判定”。
        if extractable is None:
            continue
        if extractable is True:
            extractable_value = True
        elif extractable is False:
            extractable_value = False
        else:
            raise ValueError(
                f"{source} 第 {line_no} 行 verdict extractable 非法："
                f"{extractable!r}（只接受 true/false/None）"
            )
        try:
            candidate = candidates_by_index.get(verdict.get("index"))
        except TypeError as exc:
            raise ValueError(
                f"{source} 第 {line_no} 行 verdict index 非法（不可哈希）：{exc}"
            ) from exc
        if not isinstance(candidate, dict):
            candidate = {}
        yield _make_row(
            line_no=line_no,
            source_file=record.get("source_file"),
            sentence=record.get("sentence"),
            locative=verdict.get("word"),
            locative_index=verdict.get("index"),
            target=_node_word(candidate.get("target")),
            reference=_node_word(candidate.get("reference")),
            verb=_node_word(candidate.get("verb")),
            verdict=extractable_value,
            reason=verdict.get("reason"),
            status=status,
        )


def read_jsonl_rows(
    path: str | Path,
    *,
    progress_every: int = 50_000,
    progress_stream=None,
) -> list[VerdictRow]:
    """流式读取 JSONL；只保留判定投影，不把整份记录留在内存里。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    rows: list[VerdictRow] = []
    records = 0
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
            rows.extend(_jsonl_verdict_rows(record, line_no, source))
            _progress(progress_stream, progress_every, records, "条记录")
    return assign_keys(rows)


def compare_rows(
    left: Sequence[VerdictRow], right: Sequence[VerdictRow]
) -> Comparison:
    # 允许调用方传入未编号的行（例如测试里直接构造的 VerdictRow）；已编号的行重算后不变。
    left = assign_keys(list(left))
    right = assign_keys(list(right))
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
            "相同": item.get("same", 0),
            "不同": item.get("different", 0),
            "true→false": item.get("tf", 0),
            "false→true": item.get("ft", 0),
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
        "对齐口径: (源文件, 原句, 方位词, 方位词下标, 同 base 序号)；比较字段: 模型判定",
        "",
        f"左侧: 判定={c.left_total} "
        f"(true={c.left_true}, false={c.left_false})，完整三元组={c.left_complete}",
        f"右侧: 判定={c.right_total} "
        f"(true={c.right_true}, false={c.right_false})，完整三元组={c.right_complete}",
        "",
        f"共同判定项: {c.common}",
    ]
    if c.common:
        rate = (c.agreement or 0.0) * 100
        lines.extend(
            [
                f"  相同判断: {c.same} ({rate:.2f}%)",
                f"    true→true : {c.matrix[(True, True)]}",
                f"    false→false: {c.matrix[(False, False)]}",
                f"  不同判断: {c.different} ({100 - rate:.2f}%)",
                f"    true→false: {c.matrix[(True, False)]}",
                f"    false→true: {c.matrix[(False, True)]}",
                f"  其中两侧三元组都完整: {c.common_complete} "
                f"(相同={c.same_complete}, 不同={c.different_complete})",
            ]
        )
    else:
        lines.append("  相同判断: N/A（没有可对齐的判定项）")

    lines.extend(
        [
            "",
            f"仅左侧: {c.only_left} (true={c.only_left_true}, false={c.only_left_false})",
            f"仅右侧: {c.only_right} (true={c.only_right_true}, false={c.only_right_false})",
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
    left_verdict = "true" if left.verdict else "false"
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
        f"    右: {'true' if right.verdict else 'false'} | 三元组: {right_triple} | "
        f"{_short(right.reason)}"
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
            f"判定: {'true' if row.verdict else 'false'} | 理由: {_short(row.reason)}"
            for i, row in enumerate(only_right, 1)
        )
    return "\n".join(lines)


def _detail_row(comparison: Comparison, key: tuple, kind: str) -> dict[str, str]:
    left = comparison.left_by_key.get(key)
    right = comparison.right_by_key.get(key)
    source = left or right
    assert source is not None
    return {
        "比较结果": kind,
        "左侧行号": left.line_no if left else "",
        "右侧行号": right.line_no if right else "",
        "源文件": source.source_file,
        "原句": source.sentence,
        "方位词": source.locative,
        "方位词下标": source.locative_index,
        "左目标物": left.target if left else "",
        "左参照物": left.reference if left else "",
        "左动词": left.verb if left else "",
        "右目标物": right.target if right else "",
        "右参照物": right.reference if right else "",
        "右动词": right.verb if right else "",
        "左判定": ("true" if left.verdict else "false") if left else "",
        "右判定": ("true" if right.verdict else "false") if right else "",
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
        rows, _ = load_rows(path, progress_every=progress_every)
        files.append(
            LoadedFile(
                index=index,
                path=path,
                fmt=fmt,
                label=_file_label(path, index),
                rows=rows,
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
            comparison = compare_rows(left.rows, right.rows)
            csv_name = pair_csv_name(left, right)
            write_pair_csv(comparison, staging_dir / csv_name)
            summary = pair_summary_from_comparison(
                left, right, csv_name, comparison
            )
            summaries.append(summary)
            print(
                f"[{pair_no}/{pair_total}] {left.label} vs {right.label}: "
                f"共同={summary.common}, 相同={summary.same}, "
                f"不同={summary.different}, 一致率={summary.agreement_text}",
                file=progress_stream or sys.stdout,
            )
    return summaries


def format_matrix_lines(
    files: Sequence[LoadedFile], summaries: Sequence[PairSummary]
) -> list[str]:
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
            cells.append(summary.agreement_text if summary else "N/A")
        lines.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    return lines


def format_overall_markdown(
    files: Sequence[LoadedFile], summaries: Sequence[PairSummary]
) -> str:
    lines = [
        "# 方位词筛选结果总体对比",
        "",
        "## 输入文件",
        "",
    ]
    lines.extend(
        _md_table(
            ["编号", "文件", "格式", "判定数", "true", "false", "完整三元组", "重复 base 组"],
            [
                {
                    "编号": f"{item.index:02d}",
                    "文件": item.path,
                    "格式": item.fmt,
                    "判定数": item.total,
                    "true": item.true_count,
                    "false": item.false_count,
                    "完整三元组": item.complete_count,
                    "重复 base 组": item.duplicate_groups,
                }
                for item in files
            ],
            ["---", "---", "---", "---:", "---:", "---:", "---:", "---:"],
        )
    )
    lines.extend(["", "## 两两对比汇总", ""])
    lines.extend(
        _md_table(
            ["#", "左", "右", "共同", "相同", "不同", "一致率", "仅左", "仅右", "三元组变化", "明细 CSV"],
            [
                {
                    "#": index,
                    "左": summary.left.label,
                    "右": summary.right.label,
                    "共同": summary.common,
                    "相同": summary.same,
                    "不同": summary.different,
                    "一致率": summary.agreement_text,
                    "仅左": summary.only_left,
                    "仅右": summary.only_right,
                    "三元组变化": summary.triple_changed,
                    "明细 CSV": f"[{summary.csv_name}]({summary.csv_name})",
                }
                for index, summary in enumerate(summaries, 1)
            ],
            ["---"] * 7 + ["---:"] * 3 + ["---"],
        )
    )
    lines.extend(
        [
            "",
            "## 一致性矩阵",
            "",
            "单元格为 `一致率（相同/共同）`；只统计两侧都存在的共同判定项，"
            "仅单侧项不进入分母。对角线不与自己比较，写 `—`；无共同项写 `N/A (0/0)`。",
            "",
        ]
    )
    lines.extend(format_matrix_lines(files, summaries))
    lines.extend(["", "## 重复 base 歧义警告", ""])
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
                ["对比", "次数不等（两侧都有）", "顺序歧义组", "顺序歧义项", "相同判断范围"],
                [
                    {
                        "对比": f"{item.left.label} vs {item.right.label}",
                        "次数不等（两侧都有）": item.duplicate_unequal_both,
                        "顺序歧义组": item.order_ambiguous_groups,
                        "顺序歧义项": item.order_ambiguous_common,
                        "相同判断范围": f"[{item.same_min_total}, {item.same_max_total}]"
                        f"（当前={item.same}）",
                    }
                    for item in warnings
                ],
                ["---", "---:", "---:", "---:", "---:"],
            )
        )
    else:
        lines.append("未发现重复 base 配对歧义。")
    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 对齐 key: `(源文件, 原句, 方位词, 方位词下标, 同 base 序号)`。",
            "- 比较字段: `模型判定` 的 true/false；三元组文本变化单独统计。",
            "- 两两 CSV 保留全部共同相同、共同不同、仅左侧、仅右侧明细。",
            "- 差异本身不是错误；只有至少一对完全没有共同判定项时才返回 1。",
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
        "| 编号 | 文件 | 判定数 | true | false | 完整三元组 | 重复 base 组 |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for item in files:
        print(
            f"| {item.index:02d} | {item.path} | {item.total} | "
            f"{item.true_count} | {item.false_count} | {item.complete_count} | "
            f"{item.duplicate_groups} |"
        )
    print()
    print("| 对比 | 共同 | 相同 | 不同 | 一致率 | 仅左 | 仅右 | 明细 CSV |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in summaries:
        print(
            f"| {summary.left.label} vs {summary.right.label} | "
            f"{summary.common} | {summary.same} | {summary.different} | "
            f"{summary.agreement_text} | {summary.only_left} | "
            f"{summary.only_right} | {summary.csv_name} |"
        )
    print()
    print("一致性矩阵:")
    print("\n".join(format_matrix_lines(files, summaries)))
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
) -> tuple[list[VerdictRow], str]:
    fmt = detect_format(path)
    if fmt == "csv":
        rows = read_csv_rows(
            path, progress_every=progress_every, progress_stream=progress_stream
        )
    else:
        rows = read_jsonl_rows(
            path, progress_every=progress_every, progress_stream=progress_stream
        )
    return rows, fmt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比多个方位词筛选结果，输出两两 CSV 和总体一致性矩阵",
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

    if any(summary.common == 0 for summary in summaries):
        print(
            "\n警告：至少一对没有共同判定项，矩阵中对应单元格为 N/A；"
            "其他对比和输出文件仍然有效。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
