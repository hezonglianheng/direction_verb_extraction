#!/usr/bin/env python3
# encoding: utf8
"""对比多份 annotate 结果 JSONL，输出两两明细 CSV 与 summary.md。

支持两种正式产物，可任意混比：

- ``python -m annotate --model <配置名>`` 生成的单模型结果（无 ``voting`` 块）；
- ``python -m annotate --models <配置名...>`` 生成的最终投票结果（带 ``voting`` 块），
  以及各模型原始结果 ``<最终文件主名>.models/00_<模型>.jsonl``（与单模型同形）。

对齐口径：唯一键 ``candidate_key``（``句子ID|关注词位置|替换词``）。
``id`` / ``line_no`` / 文件行序都**不参与**对齐：它们会随 ``--offset``、抽样和批次变化。
没有候选的记录（``bad_record`` / ``alter_missed``）合法地没有 key，单独计数，不强行配对。

主比较字段是**最终结论标签**：

- 单模型结果按 ``annotate.voting.ballot_label`` 的同一套口径还原
  （``same`` / ``different`` / ``not_spatial`` / ``not_fluent_*`` / 无结论）；
- 投票结果取 ``voting.winner_label``。``tie`` / ``no_valid_votes`` 记为无结论，
  **不会**退回代表模型继承的 ``judgement.same_meaning``；采纳 ``not_spatial`` 时
  ``judgement.same_meaning`` 也可能是 null，那不是投票共识。

``judgement.same_meaning`` 另列三态矩阵（true/false/未判定）；空值不当作 false，
"三态状态一致率" 与 "双方都给出结论时的一致率" 分开报告。
``voting.status``、准入、复核、两侧图式、句子上下文变化单列；理由只展示，不进一致率。

用法::

    python analyze_tools/annotate/compare_annotate_results.py run_a.jsonl run_b.jsonl
    python analyze_tools/annotate/compare_annotate_results.py \
        results/spatial_pairs/vote.jsonl \
        results/spatial_pairs/vote.models/00_model-a.jsonl \
        results/spatial_pairs/vote.models/01_model-b.jsonl
    python analyze_tools/annotate/compare_annotate_results.py run_a.jsonl run_b.jsonl \
        --output-dir analyze_results/annotate_comparison

退出码：0=每对都有共同 key；1=至少一对没有共同 key（仍写出报告）；
2=输入/输出错误（重复输入、格式不符、损坏 JSON、重复 key、输出目录已存在）。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Sequence

if __package__:
    from . import _records as records
else:
    import _records as records


ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYZE_RESULTS_DIR = ROOT_DIR / "analyze_results"
COMPARISON_DIR_NAME = "compare_annotate_results"
SUMMARY_FILENAME = "summary.md"
DEFAULT_MAX_FILES = 20

CHANGE_LABELS = (
    ("label", "最终结论"),
    ("same_meaning", "同义判定字段"),
    ("status", "最终状态"),
    ("schema_a", "a图式"),
    ("schema_b", "b图式"),
    ("admitted", "是否准入"),
    ("review", "需复核"),
    ("context", "句子上下文"),
    ("voting_status", "投票状态"),
)

DETAIL_COLUMNS = [
    "比较结果",
    "candidate_key",
    "变化字段",
    "句子ID",
    "原句",
    "替换句",
    "关注词a",
    "关注词b",
    "左记录类型",
    "右记录类型",
    "左模型",
    "右模型",
    "左标注图",
    "右标注图",
    "左最终结论",
    "右最终结论",
    "左同义判定",
    "右同义判定",
    "左最终状态",
    "右最终状态",
    "左a图式",
    "右a图式",
    "左b图式",
    "右b图式",
    "左是否准入",
    "右是否准入",
    "左需复核",
    "右需复核",
    "左投票状态",
    "右投票状态",
    "左获胜标签",
    "右获胜标签",
    "左获胜模型",
    "右获胜模型",
    "左投票不完整",
    "右投票不完整",
    "左缺票模型",
    "右缺票模型",
    "左无效票模型",
    "右无效票模型",
    "左行号",
    "右行号",
    "左JSONL行号",
    "右JSONL行号",
    "左判定理由",
    "右判定理由",
    "左失败原因",
    "右失败原因",
    "左复核原因",
    "右复核原因",
]


def _context(row: records.RecordRow) -> tuple[str, str, str, str, str]:
    return (
        row.source_file,
        row.sentence_a,
        row.sentence_b,
        row.focus_a,
        row.focus_b,
    )


def _changes(
    left: records.RecordRow,
    right: records.RecordRow,
    *,
    schema_applicable: bool,
    voting_applicable: bool,
) -> dict[str, bool]:
    result = {
        "label": left.label != right.label,
        "same_meaning": left.same_meaning != right.same_meaning,
        "status": left.status != right.status,
        "schema_a": schema_applicable and left.schema_a != right.schema_a,
        "schema_b": schema_applicable and left.schema_b != right.schema_b,
        "admitted": left.admitted != right.admitted,
        "review": left.review_required != right.review_required,
        "context": _context(left) != _context(right),
        "voting_status": voting_applicable
        and left.voting_status_text != right.voting_status_text,
    }
    return result


def agreement_text(same: int, denominator: int) -> str:
    if not denominator:
        return "N/A (0/0)"
    return f"{same / denominator * 100:.2f}% ({same}/{denominator})"


def _schema_text(same: int, comparable: int, one_side: int) -> str:
    return f"{agreement_text(same, comparable)}；仅单侧={one_side}"


def _supports_schema(rows: Sequence[records.RecordRow]) -> bool:
    modes = {row.mode for row in rows}
    return bool(modes) and modes <= {"pair", "all_in_one"}


def _supports_voting(rows: Sequence[records.RecordRow]) -> bool:
    return bool(rows) and all(row.is_voting for row in rows)


def _index_rows(
    rows: Sequence[records.RecordRow],
) -> dict[str, records.RecordRow]:
    result: dict[str, records.RecordRow] = {}
    for row in rows:
        if row.key is None:
            continue
        if row.key in result:
            raise ValueError(
                f"candidate_key 重复：{row.key!r}；不能按出现顺序配对"
            )
        result[row.key] = row
    return result


@dataclass
class Comparison:
    """一对文件的逐项对比；只保留计数，不保留 N² 份记录。"""

    left_by_key: dict[str, records.RecordRow]
    right_by_key: dict[str, records.RecordRow]
    schema_applicable: bool
    voting_applicable: bool

    def __post_init__(self) -> None:
        self.common_keys = [
            key for key in self.left_by_key if key in self.right_by_key
        ]
        self.only_left_keys = [
            key for key in self.left_by_key if key not in self.right_by_key
        ]
        self.only_right_keys = [
            key for key in self.right_by_key if key not in self.left_by_key
        ]
        self.common = len(self.common_keys)
        self.only_left = len(self.only_left_keys)
        self.only_right = len(self.only_right_keys)
        self.union = self.common + self.only_left + self.only_right

        self.same = 0
        self.different = 0
        self.defined_comparable = 0
        self.defined_same = 0
        self.unresolved = 0
        self.label_transitions: Counter[tuple[str | None, str | None]] = Counter()
        self.same_meaning_matrix: Counter[tuple[bool | None, bool | None]] = Counter()
        self.same_meaning_comparable = 0
        self.same_meaning_same = 0
        self.status_same = 0
        self.status_transitions: Counter[tuple[str, str]] = Counter()
        self.schema_a_comparable = 0
        self.schema_a_same = 0
        self.schema_a_one_side = 0
        self.schema_a_both_empty = 0
        self.schema_b_comparable = 0
        self.schema_b_same = 0
        self.schema_b_one_side = 0
        self.schema_b_both_empty = 0
        self.admitted_comparable = 0
        self.admitted_same = 0
        self.review_comparable = 0
        self.review_same = 0
        self.context_changed = 0
        self.voting_same = 0
        self.voting_comparable = 0
        self.voting_transitions: Counter[tuple[str, str]] = Counter()
        self.changes: Counter[str] = Counter()

        for key in self.common_keys:
            left = self.left_by_key[key]
            right = self.right_by_key[key]

            same_label = left.label == right.label
            self.same += same_label
            self.different += not same_label
            self.label_transitions[(left.label, right.label)] += 1
            if left.label is not None and right.label is not None:
                self.defined_comparable += 1
                self.defined_same += same_label
            else:
                self.unresolved += 1

            meaning = (left.same_meaning, right.same_meaning)
            self.same_meaning_matrix[meaning] += 1
            if left.same_meaning is not None and right.same_meaning is not None:
                self.same_meaning_comparable += 1
                self.same_meaning_same += left.same_meaning == right.same_meaning

            self.status_transitions[(left.status, right.status)] += 1
            self.status_same += left.status == right.status

            if self.schema_applicable:
                for side, prefix in (("a", "schema_a"), ("b", "schema_b")):
                    left_name = getattr(left, prefix)
                    right_name = getattr(right, prefix)
                    if left_name and right_name:
                        setattr(self, prefix + "_comparable",
                                getattr(self, prefix + "_comparable") + 1)
                        setattr(self, prefix + "_same",
                                getattr(self, prefix + "_same") + (left_name == right_name))
                    elif left_name or right_name:
                        setattr(self, prefix + "_one_side",
                                getattr(self, prefix + "_one_side") + 1)
                    else:
                        setattr(self, prefix + "_both_empty",
                                getattr(self, prefix + "_both_empty") + 1)

            if left.admitted is not None and right.admitted is not None:
                self.admitted_comparable += 1
                self.admitted_same += left.admitted == right.admitted

            if left.review_required is not None and right.review_required is not None:
                self.review_comparable += 1
                self.review_same += left.review_required == right.review_required

            self.context_changed += _context(left) != _context(right)

            if self.voting_applicable:
                voting = (left.voting_status_text, right.voting_status_text)
                self.voting_transitions[voting] += 1
                self.voting_comparable += 1
                self.voting_same += voting[0] == voting[1]

            flags = _changes(
                left,
                right,
                schema_applicable=self.schema_applicable,
                voting_applicable=self.voting_applicable,
            )
            self.changes.update(
                {name: int(changed) for name, changed in flags.items()}
            )


def compare_rows(
    left_rows: Sequence[records.RecordRow],
    right_rows: Sequence[records.RecordRow],
    *,
    schema_applicable: bool | None = None,
    voting_applicable: bool | None = None,
) -> Comparison:
    if schema_applicable is None:
        schema_applicable = _supports_schema(left_rows) and _supports_schema(right_rows)
    if voting_applicable is None:
        voting_applicable = _supports_voting(left_rows) and _supports_voting(right_rows)
    return Comparison(
        left_by_key=_index_rows(left_rows),
        right_by_key=_index_rows(right_rows),
        schema_applicable=schema_applicable,
        voting_applicable=voting_applicable,
    )


@dataclass
class PairSummary:
    """一对比较的轻量汇总，供 summary.md 与终端输出使用。"""

    left: records.LoadedFile
    right: records.LoadedFile
    csv_name: str
    common: int
    union: int
    same: int
    different: int
    defined_comparable: int
    defined_same: int
    unresolved: int
    same_meaning_comparable: int
    same_meaning_same: int
    only_left: int
    only_right: int
    status_same: int
    schema_applicable: bool
    schema_a_comparable: int
    schema_a_same: int
    schema_a_one_side: int
    schema_a_both_empty: int
    schema_b_comparable: int
    schema_b_same: int
    schema_b_one_side: int
    schema_b_both_empty: int
    admitted_comparable: int
    admitted_same: int
    review_comparable: int
    review_same: int
    context_changed: int
    voting_applicable: bool
    voting_comparable: int
    voting_same: int
    changes: Counter[str]
    label_transitions: Counter[tuple[str | None, str | None]]
    same_meaning_matrix: Counter[tuple[bool | None, bool | None]]
    status_transitions: Counter[tuple[str, str]]
    voting_transitions: Counter[tuple[str, str]]

    @classmethod
    def from_comparison(
        cls,
        left: records.LoadedFile,
        right: records.LoadedFile,
        csv_name: str,
        comparison: Comparison,
    ) -> "PairSummary":
        return cls(
            left=left,
            right=right,
            csv_name=csv_name,
            common=comparison.common,
            union=comparison.union,
            same=comparison.same,
            different=comparison.different,
            defined_comparable=comparison.defined_comparable,
            defined_same=comparison.defined_same,
            unresolved=comparison.unresolved,
            same_meaning_comparable=comparison.same_meaning_comparable,
            same_meaning_same=comparison.same_meaning_same,
            only_left=comparison.only_left,
            only_right=comparison.only_right,
            status_same=comparison.status_same,
            schema_applicable=comparison.schema_applicable,
            schema_a_comparable=comparison.schema_a_comparable,
            schema_a_same=comparison.schema_a_same,
            schema_a_one_side=comparison.schema_a_one_side,
            schema_a_both_empty=comparison.schema_a_both_empty,
            schema_b_comparable=comparison.schema_b_comparable,
            schema_b_same=comparison.schema_b_same,
            schema_b_one_side=comparison.schema_b_one_side,
            schema_b_both_empty=comparison.schema_b_both_empty,
            admitted_comparable=comparison.admitted_comparable,
            admitted_same=comparison.admitted_same,
            review_comparable=comparison.review_comparable,
            review_same=comparison.review_same,
            context_changed=comparison.context_changed,
            voting_applicable=comparison.voting_applicable,
            voting_comparable=comparison.voting_comparable,
            voting_same=comparison.voting_same,
            changes=comparison.changes,
            label_transitions=comparison.label_transitions,
            same_meaning_matrix=comparison.same_meaning_matrix,
            status_transitions=comparison.status_transitions,
            voting_transitions=comparison.voting_transitions,
        )

    @property
    def common_agreement_text(self) -> str:
        return agreement_text(self.same, self.common)

    @property
    def defined_agreement_text(self) -> str:
        return agreement_text(self.defined_same, self.defined_comparable)

    @property
    def status_agreement_text(self) -> str:
        return agreement_text(self.status_same, self.common)


def _record_kind(row: records.RecordRow) -> str:
    return "投票汇总" if row.is_voting else "单模型"


def _side_cells(
    prefix: str, row: records.RecordRow
) -> dict[str, str]:
    return {
        f"{prefix}记录类型": _record_kind(row),
        f"{prefix}模型": row.model_name,
        f"{prefix}标注图": row.mode,
        f"{prefix}最终结论": records.label_text(row.label),
        f"{prefix}同义判定": records.boolean_text(row.same_meaning),
        f"{prefix}最终状态": row.status,
        f"{prefix}a图式": row.schema_a,
        f"{prefix}b图式": row.schema_b,
        f"{prefix}是否准入": records.boolean_text(row.admitted),
        f"{prefix}需复核": records.boolean_text(row.review_required),
        f"{prefix}投票状态": row.voting_status_text,
        f"{prefix}获胜标签": row.winner_label_text,
        f"{prefix}获胜模型": row.winner_model_text,
        f"{prefix}投票不完整": row.incomplete_text,
        f"{prefix}缺票模型": row.missing_text,
        f"{prefix}无效票模型": row.invalid_text,
        f"{prefix}行号": row.line_no,
        f"{prefix}JSONL行号": str(row.input_line),
        f"{prefix}判定理由": row.judgement_reason,
        f"{prefix}失败原因": row.status_reason,
        f"{prefix}复核原因": ", ".join(row.reasons),
    }


def _detail_row(
    left: records.RecordRow | None,
    right: records.RecordRow | None,
    *,
    schema_applicable: bool,
    voting_applicable: bool,
) -> dict[str, str]:
    source = left or right
    assert source is not None
    result: dict[str, str] = {
        "candidate_key": source.key or "",
        "变化字段": "",
        "句子ID": source.sentence_id,
        "原句": source.sentence_a,
        "替换句": source.sentence_b,
        "关注词a": source.focus_a,
        "关注词b": source.focus_b,
    }
    if left is not None and right is not None:
        result["比较结果"] = "相同" if left.label == right.label else "不同"
        flags = _changes(
            left,
            right,
            schema_applicable=schema_applicable,
            voting_applicable=voting_applicable,
        )
        result["变化字段"] = ", ".join(
            label for name, label in CHANGE_LABELS if flags[name]
        )
    else:
        result["比较结果"] = "仅左侧" if left is not None else "仅右侧"
    if left is not None:
        result.update(_side_cells("左", left))
    if right is not None:
        result.update(_side_cells("右", right))
    return result


def write_pair_csv(comparison: Comparison, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=DETAIL_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for keys in (
            comparison.common_keys,
            comparison.only_left_keys,
            comparison.only_right_keys,
        ):
            for key in keys:
                writer.writerow(
                    _detail_row(
                        comparison.left_by_key.get(key),
                        comparison.right_by_key.get(key),
                        schema_applicable=comparison.schema_applicable,
                        voting_applicable=comparison.voting_applicable,
                    )
                )


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def _table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(_md(item) for item in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *[
            "| " + " | ".join(_md(item) for item in row) + " |"
            for row in rows
        ],
    ]


def _file_label(path: Path, index: int) -> str:
    name = path.name
    if name.lower().endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._-")
    # 两个标签 + "__vs__" + ".csv" 控制在常见文件系统 255 字节上限以内。
    name = name.encode("utf-8")[:90].decode("utf-8", errors="ignore") or "result"
    return f"{index:02d}_{name}"


def format_matrix_lines(
    files: Sequence[records.LoadedFile], summaries: Sequence[PairSummary]
) -> list[str]:
    lookup = {
        (summary.left.index, summary.right.index): summary
        for summary in summaries
    }
    header = ["文件"] + [item.label for item in files]
    lines = [
        "| " + " | ".join(_md(cell) for cell in header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(files)) + " |",
    ]
    for left in files:
        cells = [_md(left.label)]
        for right in files:
            if left.index == right.index:
                cells.append("—")
                continue
            summary = lookup.get(
                (min(left.index, right.index), max(left.index, right.index))
            )
            cells.append(
                summary.common_agreement_text if summary else "N/A (0/0)"
            )
        lines.append("| " + " | ".join(_md(cell) for cell in cells) + " |")
    return lines


def _transition_rows(
    counter: Counter[tuple[Any, Any]],
) -> list[list[Any]]:
    return [
        [left, right, count]
        for (left, right), count in sorted(
            counter.items(), key=lambda item: (-item[1], str(item[0]))
        )
    ]


def _label_value(value: str | None) -> str:
    return records.label_text(value)


def _tri_value(value: bool | None) -> str:
    if value is None:
        return "未判定"
    return "true" if value else "false"


def _pair_section(summary: PairSummary) -> list[str]:
    left, right = summary.left, summary.right
    lines = [
        f"### {left.label} vs {right.label}",
        "",
        f"- 对比文件：`{left.path}` ↔ `{right.path}`",
        f"- 标注图：`{left.mode or '—'}` ↔ `{right.mode or '—'}`；"
        f"记录类型：{left.kind} ↔ {right.kind}",
        f"- 共同 key：{summary.common}；仅左：{summary.only_left}；"
        f"仅右：{summary.only_right}；并集：{summary.union}",
        f"- 结论一致率（含未判定，分母=共同 key）：{summary.common_agreement_text}",
        f"- 有效结论一致率（双方都有结论）：{summary.defined_agreement_text}；"
        f"涉及无结论：{summary.unresolved}",
        f"- 最终状态一致率：{summary.status_agreement_text}；"
        f"句子/关注词上下文变化：{summary.context_changed}",
        "",
        "#### 结论转移",
        "",
    ]
    lines.extend(
        _table(
            ["左结论", "右结论", "数量"],
            [
                [_label_value(left_label), _label_value(right_label), count]
                for (left_label, right_label), count in sorted(
                    summary.label_transitions.items(),
                    key=lambda item: (-item[1], str(item[0])),
                )
            ],
        )
    )

    lines.extend(["", "#### 同义判定字段矩阵", ""])
    matrix = summary.same_meaning_matrix
    matrix_rows = []
    for left_value in (True, False, None):
        matrix_rows.append(
            [
                _tri_value(left_value),
                *[matrix[(left_value, right_value)] for right_value in (True, False, None)],
            ]
        )
    lines.extend(
        _table(
            ["左 \\ 右", "true", "false", "未判定"],
            matrix_rows,
        )
    )
    lines.extend(
        [
            "",
            f"双方都给出布尔判定：{summary.same_meaning_comparable}；"
            f"相同：{summary.same_meaning_same}；"
            f"布尔一致率："
            f"{agreement_text(summary.same_meaning_same, summary.same_meaning_comparable)}。"
            "未判定不当作 false，也不进入这个分母。"
            "投票记录中的该字段来自代表模型，不是投票共识；结论请看上方最终标签。",
        ]
    )

    lines.extend(["", "#### 最终状态转移", ""])
    lines.extend(
        _table(
            ["左状态", "右状态", "数量"],
            _transition_rows(summary.status_transitions),
        )
    )

    lines.extend(["", "#### 图式", ""])
    if summary.schema_applicable:
        lines.extend(
            _table(
                ["侧", "相同", "双方均有", "一致率", "仅单侧有图式", "双方为空"],
                [
                    [
                        "a",
                        summary.schema_a_same,
                        summary.schema_a_comparable,
                        agreement_text(
                            summary.schema_a_same, summary.schema_a_comparable
                        ),
                        summary.schema_a_one_side,
                        summary.schema_a_both_empty,
                    ],
                    [
                        "b",
                        summary.schema_b_same,
                        summary.schema_b_comparable,
                        agreement_text(
                            summary.schema_b_same, summary.schema_b_comparable
                        ),
                        summary.schema_b_one_side,
                        summary.schema_b_both_empty,
                    ],
                ],
            )
        )
    else:
        lines.append(
            "不适用：至少一侧是 synonym 结果（不写图式），或文件为空；"
            "图式只在双方都有名字时统计，不会把不适用当成不同。"
        )

    lines.extend(["", "#### 准入与复核", ""])
    lines.extend(
        _table(
            ["指标", "双方均有", "相同", "一致率", "涉及空值"],
            [
                [
                    "是否准入",
                    summary.admitted_comparable,
                    summary.admitted_same,
                    agreement_text(
                        summary.admitted_same, summary.admitted_comparable
                    ),
                    summary.common - summary.admitted_comparable,
                ],
                [
                    "需复核",
                    summary.review_comparable,
                    summary.review_same,
                    agreement_text(
                        summary.review_same, summary.review_comparable
                    ),
                    summary.common - summary.review_comparable,
                ],
            ],
        )
    )

    lines.extend(["", "#### 投票", ""])
    if summary.voting_applicable:
        lines.extend(
            _table(
                ["左投票状态", "右投票状态", "数量"],
                [
                    [left_status, right_status, count]
                    for (left_status, right_status), count in sorted(
                        summary.voting_transitions.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
            )
        )
        lines.extend(
            [
                "",
                f"投票状态一致率："
                f"{agreement_text(summary.voting_same, summary.voting_comparable)}",
            ]
        )
    else:
        lines.append(
            "不适用：至少一侧不是投票汇总结果（单模型结果没有 voting 块），"
            "不把「没有投票」当成缺票。"
        )
    lines.append("")
    return lines


def format_overall_markdown(
    files: Sequence[records.LoadedFile], summaries: Sequence[PairSummary]
) -> str:
    lines = [
        "# annotate 结果对比",
        "",
        "## 口径说明",
        "",
        "- 唯一对齐键：`candidate_key`（`句子ID|关注词位置|替换词`）；"
        "`id`、`line_no`、文件行序和出现顺序都不参与对齐。",
        "- 无候选记录（`bad_record` / `alter_missed`）合法地没有 key，"
        "单独计入「无候选」，不强行配对。",
        "- 主比较字段是最终结论标签：单模型结果按 `annotate.voting.ballot_label` "
        "同一套口径还原；投票结果取 `voting.winner_label`。",
        "- 投票 `tie` / `no_valid_votes` 记为 `（无结论）`，"
        "**不退回**代表模型继承的 `judgement.same_meaning`；"
        "采纳 `not_spatial` / `not_fluent_*` 时该字段可能为 null，不是投票共识。",
        "- 投票记录的 `schema` / `features` 等是代表模型继承下来的字段，"
        "不是投票共识；本工具只把 `voting.winner_label` 当结论。",
        "- `judgement.same_meaning` 另列三态矩阵；空值不当作 false。"
        "「结论一致率（含未判定）」与「有效结论一致率」分开统计。",
        "- 图式只在双方都有名字时统计；synonym 不写图式，记为不适用而不是不同。",
        "- 理由与失败原因只展示，不进入一致率；差异本身不是错误。",
        "",
        "## 输入文件",
        "",
    ]
    lines.extend(
        _table(
            [
                "编号",
                "文件",
                "类型",
                "标注图",
                "模型",
                "记录数",
                "有候选键",
                "无候选",
                "结论分布",
                "投票状态",
                "投票不完整",
                "缺票模型",
                "无效票模型",
            ],
            [
                [
                    f"{item.index:02d}",
                    item.path,
                    item.kind,
                    item.mode or "—",
                    ", ".join(item.models) or "—",
                    item.total,
                    item.keyed,
                    item.unkeyed,
                    records.file_distribution(item.label_counts),
                    records.file_distribution(item.voting_statuses),
                    item.incomplete_count if item.is_voting else "—",
                    records.file_distribution(item.missing_by_model),
                    records.file_distribution(item.invalid_by_model),
                ]
                for item in files
            ],
        )
    )
    lines.extend(["", "## 两两对比汇总", ""])
    lines.extend(
        _table(
            [
                "#",
                "左",
                "右",
                "共同",
                "并集",
                "结论一致率（含未判定）",
                "有效结论一致率",
                "仅左",
                "仅右",
                "状态一致率",
                "a图式",
                "b图式",
                "投票状态一致率",
                "明细 CSV",
            ],
            [
                [
                    index,
                    summary.left.label,
                    summary.right.label,
                    summary.common,
                    summary.union,
                    summary.common_agreement_text,
                    summary.defined_agreement_text,
                    summary.only_left,
                    summary.only_right,
                    summary.status_agreement_text,
                    _schema_text(
                        summary.schema_a_same,
                        summary.schema_a_comparable,
                        summary.schema_a_one_side,
                    )
                    if summary.schema_applicable
                    else "不适用",
                    _schema_text(
                        summary.schema_b_same,
                        summary.schema_b_comparable,
                        summary.schema_b_one_side,
                    )
                    if summary.schema_applicable
                    else "不适用",
                    agreement_text(summary.voting_same, summary.voting_comparable)
                    if summary.voting_applicable
                    else "不适用",
                    f"[{summary.csv_name}]({summary.csv_name})",
                ]
                for index, summary in enumerate(summaries, 1)
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 一致性矩阵",
            "",
            "单元格为 `结论一致率（相同/共同 key）`，未判定对未判定计为相同状态；"
            "仅单侧 key 不进入分母。对角线写 `—`，无共同 key 写 `N/A (0/0)`。",
            "",
        ]
    )
    lines.extend(format_matrix_lines(files, summaries))
    lines.extend(["", "## 逐对明细", ""])
    for summary in summaries:
        lines.extend(_pair_section(summary))
    return "\n".join(lines)


def _unique_default_output_dir() -> Path:
    base = ANALYZE_RESULTS_DIR / COMPARISON_DIR_NAME
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = base / stamp
    suffix = 1
    while target.exists() or target.is_symlink():
        target = base / f"{stamp}_{suffix:02d}"
        suffix += 1
    return target


def plan_output_dir(value: str | None) -> Path:
    target = Path(value) if value else _unique_default_output_dir()
    if target.exists() or target.is_symlink():
        raise ValueError(f"输出目录已存在：{target}；请换一个新目录或删除旧目录")
    return target


def publish_output_dir(staging: Path, target: Path) -> None:
    """独占创建目标目录；``summary.md`` 最后发布，已有目录绝不覆盖。"""
    os.mkdir(target)
    try:
        for path in sorted(
            staging.iterdir(),
            key=lambda item: (item.name == SUMMARY_FILENAME, item.name),
        ):
            os.replace(path, target / path.name)
        staging.rmdir()
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def _same_file(left: Path, right: Path) -> bool:
    try:
        if left.resolve() == right.resolve():
            return True
        return left.samefile(right)
    except (FileNotFoundError, OSError, RuntimeError):
        return False


def _input_paths(inputs: Sequence[str], max_files: int) -> list[Path]:
    if len(inputs) < 2:
        raise ValueError("至少需要两份 annotate 结果 JSONL")
    if max_files < 0:
        raise ValueError("--max-files 不能为负数")
    if max_files and len(inputs) > max_files:
        raise ValueError(
            f"输入文件数为 {len(inputs)}，超过 --max-files {max_files}；"
            "请提高上限或分批比较（0 表示不限制）"
        )
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            raise ValueError(
                f"输入是目录：{path}；--models 各模型的原始结果位于 "
                "<最终文件主名>.models/*.jsonl，请逐个传入 JSONL 文件"
            )
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            raise ValueError(
                f"不支持 annotate 配套 CSV：{path}；CSV 只是拍平视图，"
                "缺少 voting 与完整记录，请改用同名 JSONL"
            )
        if path.name.endswith(".summary.json"):
            raise ValueError(
                f"这是运行汇总，不是逐条标注结果：{path}；请传入同名的结果 JSONL"
            )
        if suffix != ".jsonl":
            raise ValueError(f"只支持 annotate 结果 JSONL：{path}")
        for other in paths:
            if _same_file(path, other):
                raise ValueError(f"存在重复输入文件：{other} 与 {path}")
        paths.append(path)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比多份 annotate 结果 JSONL（支持 --model 单模型与 --models 投票结果）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法::")[-1],
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="两份或更多 annotate 结果 JSONL；投票汇总与各模型原始结果可混比",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help=f"输出目录（必须尚不存在；默认 analyze_results/"
        f"{COMPARISON_DIR_NAME}/<时间戳>）",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=DEFAULT_MAX_FILES,
        help=f"最多输入文件数（默认 {DEFAULT_MAX_FILES}；0 表示不限制）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="每读取多少条记录打印一次进度（默认 50000；0 关闭）",
    )
    return parser.parse_args(argv)


def print_terminal_summary(
    files: Sequence[records.LoadedFile],
    summaries: Sequence[PairSummary],
    output_dir: Path,
) -> None:
    print(f"已加载 {len(files)} 个文件，完成 {len(summaries)} 对比较")
    print()
    print("| 编号 | 文件 | 类型 | 标注图 | 记录数 | 有候选键 | 无候选 |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for item in files:
        print(
            f"| {item.index:02d} | {item.path} | {item.kind} | "
            f"{item.mode or '—'} | {item.total} | {item.keyed} | {item.unkeyed} |"
        )
    print()
    print(
        "| 对比 | 共同 | 结论一致（含未判定） | 有效结论一致 | "
        "仅左 | 仅右 | 明细 CSV |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in summaries:
        print(
            f"| {summary.left.label} vs {summary.right.label} | "
            f"{summary.common} | {summary.common_agreement_text} | "
            f"{summary.defined_agreement_text} | {summary.only_left} | "
            f"{summary.only_right} | {summary.csv_name} |"
        )
    print()
    print("一致性矩阵（结论一致率，分母=共同 key）:")
    print("\n".join(format_matrix_lines(files, summaries)))
    print()
    print(f"输出目录：{output_dir}")
    print(f"总体报告：{output_dir / SUMMARY_FILENAME}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = _input_paths(args.inputs, args.max_files)
        target = plan_output_dir(args.output_dir)
        files = [
            records.load_file(
                path,
                index=index,
                label=_file_label(path, index),
                progress_every=max(0, args.progress_every),
            )
            for index, path in enumerate(paths, 1)
        ]
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.")
        )
        summaries: list[PairSummary] = []
        try:
            pair_total = len(files) * (len(files) - 1) // 2
            pair_no = 0
            for index, left in enumerate(files):
                for right in files[index + 1:]:
                    pair_no += 1
                    comparison = compare_rows(
                        left.rows,
                        right.rows,
                        schema_applicable=left.has_schema and right.has_schema,
                        voting_applicable=(
                            left.is_voting is True and right.is_voting is True
                        ),
                    )
                    csv_name = f"{left.label}__vs__{right.label}.csv"
                    write_pair_csv(comparison, staging / csv_name)
                    summary = PairSummary.from_comparison(
                        left, right, csv_name, comparison
                    )
                    summaries.append(summary)
                    print(
                        f"[{pair_no}/{pair_total}] {left.label} vs {right.label}："
                        f"共同={summary.common}, 相同={summary.same}, "
                        f"不同={summary.different}, "
                        f"结论一致率={summary.common_agreement_text}, "
                        f"有效结论一致率={summary.defined_agreement_text}"
                    )
            (staging / SUMMARY_FILENAME).write_text(
                format_overall_markdown(files, summaries), encoding="utf-8"
            )
            publish_output_dir(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print_terminal_summary(files, summaries, target)
    if any(summary.common == 0 for summary in summaries):
        print(
            "\n警告：至少一对没有共同 candidate_key；对应一致率为 N/A (0/0)，"
            "报告仍已保存。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
