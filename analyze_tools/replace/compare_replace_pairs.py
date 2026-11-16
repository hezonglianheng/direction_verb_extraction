#!/usr/bin/env python3
# encoding: utf8
"""按 candidate_key 比较 replace 句对，输出两两明细 CSV 与 summary.md。

输入必须显式指定，支持 replace JSONL 与 export_replace_pairs_csv.py 导出的 CSV 混用。
主一致性比较 A.extractable、B.extractable 和 pair_screening.status；语法、空间自洽、
三元组及文本变化单列统计，不把未判定当成 false，也不按句对出现顺序强行配对。

用法：
    python analyze_tools/replace/compare_replace_pairs.py run_a.jsonl run_b.jsonl
    python analyze_tools/replace/compare_replace_pairs.py run_a.jsonl run_b.replace_pairs.csv -o report

退出码：0=各文件对均有共同 key；1=至少一对无共同 key（仍写出报告）；2=输入或输出错误。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Sequence

if __package__:
    from . import _pairs as pairs
else:
    import _pairs as pairs


ROOT_DIR = Path(__file__).resolve().parents[2]
SUMMARY_FILENAME = "summary.md"
DEFAULT_MAX_FILES = 20
BOOLEAN_FIELDS = {
    "a_extractable": ("A判定", "a", "extractable"),
    "b_extractable": ("B判定", "b", "extractable"),
    "a_grammatical": ("A语法正确", "a", "grammatical"),
    "b_grammatical": ("B语法正确", "b", "grammatical"),
    "a_spatial_coherent": ("A空间自洽", "a", "spatial_coherent"),
    "b_spatial_coherent": ("B空间自洽", "b", "spatial_coherent"),
}
FIELD_LABELS = {
    **{name: values[0] for name, values in BOOLEAN_FIELDS.items()},
    "status": "句对状态", "a_triple": "A三元组", "b_triple": "B三元组",
    "a_sentence": "A句子", "b_sentence": "B句子", "a_tokens": "A切片", "b_tokens": "B切片",
    "context": "标识上下文",
}
DETAIL_COLUMNS = [
    "比较结果", "candidate_key", "变化字段", *[label + "变化" for label in FIELD_LABELS.values()],
    *[prefix + column for prefix in ("左", "右") for column in pairs.PAIR_COLUMNS if column != "candidate_key"],
]


@dataclass
class FieldAgreement:
    comparable: int = 0
    same: int = 0
    true_to_false: int = 0
    false_to_true: int = 0
    null_involved: int = 0
    changed: int = 0

    def observe(self, left: bool | None, right: bool | None) -> None:
        self.changed += left != right
        if left is None or right is None:
            self.null_involved += 1
            return
        self.comparable += 1
        self.same += left == right
        self.true_to_false += left is True and right is False
        self.false_to_true += left is False and right is True


def _context(row: pairs.PairRow) -> tuple:
    return (row.source_file, row.corpus_type, row.replacement, row.a.word, row.a.index, row.b.word, row.b.index)


def _changes(left: pairs.PairRow, right: pairs.PairRow) -> dict[str, bool]:
    result = {
        name: getattr(getattr(left, side), attr) != getattr(getattr(right, side), attr)
        for name, (_, side, attr) in BOOLEAN_FIELDS.items()
    }
    result.update(status=left.status != right.status, context=_context(left) != _context(right))
    for side in ("a", "b"):
        for attr in ("triple", "sentence", "tokens"):
            result[f"{side}_{attr}"] = getattr(getattr(left, side), attr) != getattr(getattr(right, side), attr)
    return result


def _index_rows(rows: Sequence[pairs.PairRow]) -> dict[str, pairs.PairRow]:
    result: dict[str, pairs.PairRow] = {}
    for row in rows:
        if row.key in result:
            raise ValueError(f"candidate_key 重复：{row.key!r}；不能按出现顺序配对")
        result[row.key] = row
    return result


@dataclass
class Comparison:
    left_by_key: dict[str, pairs.PairRow]
    right_by_key: dict[str, pairs.PairRow]

    def __post_init__(self) -> None:
        self.common_keys = [key for key in self.left_by_key if key in self.right_by_key]
        self.only_left_keys = [key for key in self.left_by_key if key not in self.right_by_key]
        self.only_right_keys = [key for key in self.right_by_key if key not in self.left_by_key]
        self.common = len(self.common_keys)
        self.only_left = len(self.only_left_keys)
        self.only_right = len(self.only_right_keys)
        self.same = 0
        self.changes: Counter[str] = Counter()
        self.field_stats = {name: FieldAgreement() for name in BOOLEAN_FIELDS}
        self.status_transitions: Counter[tuple[str, str]] = Counter()
        self.by_replacement: dict[tuple[str, str], Counter[str]] = {}
        for key in self.common_keys:
            left, right = self.left_by_key[key], self.right_by_key[key]
            is_same = left.signature == right.signature
            self.same += is_same
            self.changes.update({name: int(changed) for name, changed in _changes(left, right).items()})
            self.status_transitions[(left.status, right.status)] += 1
            for name, (_, side, attr) in BOOLEAN_FIELDS.items():
                self.field_stats[name].observe(getattr(getattr(left, side), attr), getattr(getattr(right, side), attr))
            group = self.by_replacement.setdefault((left.a.word, left.replacement), Counter())
            group["common"] += 1
            group["same"] += is_same
            group["different"] += not is_same
        self.different = self.common - self.same
        for rows, keys, label in ((self.left_by_key, self.only_left_keys, "only_left"),
                                   (self.right_by_key, self.only_right_keys, "only_right")):
            for key in keys:
                row = rows[key]
                group = self.by_replacement.setdefault((row.a.word, row.replacement), Counter())
                group[label] += 1


def compare_rows(left: Sequence[pairs.PairRow], right: Sequence[pairs.PairRow]) -> Comparison:
    return Comparison(_index_rows(left), _index_rows(right))


@dataclass
class LoadedPairs:
    path: Path
    label: str
    rows: list[pairs.PairRow]
    by_key: dict[str, pairs.PairRow] = field(init=False)
    stats: Counter[str] = field(init=False)

    def __post_init__(self) -> None:
        self.by_key = _index_rows(self.rows)
        self.stats = Counter()
        for row in self.rows:
            pairs.update_stats(self.stats, row)


@dataclass
class PairSummary:
    """只保留轻量计数，不为 N² 份报告保留 Comparison 的逐项索引。"""
    left: LoadedPairs
    right: LoadedPairs
    csv_name: str
    common: int
    same: int
    different: int
    only_left: int
    only_right: int
    changes: Counter[str]
    field_stats: dict[str, FieldAgreement]
    status_transitions: Counter[tuple[str, str]]
    by_replacement: dict[tuple[str, str], Counter[str]]

    @property
    def agreement_text(self) -> str:
        return agreement_text(self.same, self.common)


def agreement_text(same: int, common: int) -> str:
    return f"{same / common * 100:.2f}% ({same}/{common})" if common else "N/A (0/0)"


def _file_label(path: Path, index: int) -> str:
    name = path.name
    for suffix in (".replace_pairs.csv", ".jsonl", ".csv"):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._-")
    # 两个标签拼接后仍小于常见文件系统 255 字节上限，且不截断 UTF-8 字符。
    name = name.encode("utf-8")[:90].decode("utf-8", errors="ignore") or "pairs"
    return f"{index:02d}_{name}"


def _detail_row(left: pairs.PairRow | None, right: pairs.PairRow | None) -> dict[str, str]:
    source = left or right
    assert source is not None
    result = {"candidate_key": source.key, "变化字段": ""}
    if left is not None and right is not None:
        flags = _changes(left, right)
        result["比较结果"] = "相同" if left.signature == right.signature else "不同"
        result["变化字段"] = ", ".join(label for name, label in FIELD_LABELS.items() if flags[name])
        result.update({label + "变化": "是" if flags[name] else "否" for name, label in FIELD_LABELS.items()})
    else:
        result["比较结果"] = "仅左侧" if left is not None else "仅右侧"
    for prefix, row in (("左", left), ("右", right)):
        if row is not None:
            result.update({prefix + column: value for column, value in pairs.pair_to_csv_row(row).items()
                           if column != "candidate_key"})
    return result


def write_pair_csv(comparison: Comparison, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for keys in (comparison.common_keys, comparison.only_left_keys, comparison.only_right_keys):
            for key in keys:
                writer.writerow(_detail_row(comparison.left_by_key.get(key), comparison.right_by_key.get(key)))


def _md(value: Any) -> str:
    return pairs.cell(value).replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return ["| " + " | ".join(_md(item) for item in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(_md(item) for item in row) + " |" for row in rows]]


def format_matrix_lines(files: Sequence[LoadedPairs], summaries: Sequence[PairSummary]) -> list[str]:
    lookup = {frozenset((item.left.label, item.right.label)): item.agreement_text for item in summaries}
    return _table(["文件", *[item.label for item in files]], [
        [left.label, *["—" if left is right else lookup.get(frozenset((left.label, right.label)), "N/A")
                       for right in files]] for left in files
    ])


def format_overall_markdown(files: Sequence[LoadedPairs], summaries: Sequence[PairSummary]) -> str:
    lines = ["# replace 句对结果对比", "", "## 口径说明", "",
             "- 唯一对齐键：`candidate_key`；不同替换词不会按顺序强行配对，缺失或重复 key 会报错。",
             "- 主比较字段：`pair_screening.a.extractable`、`pair_screening.b.extractable`、`pair_screening.status`。",
             "- 主一致率是共同 key 上三项字段全部相同的比例，含空值相同；不代表句对合格，也不代表模型已判定。",
             "- A/B 布尔一致率只统计两份结果中均为布尔值的项；未判定不当作 false，不进入该分母。",
             "- 语法、空间自洽、三元组、句子、切片和标识上下文变化单列；理由与来源仅保留明细，不进入一致率。",
             "- 只读句对侧数据，不读继承的顶层 screening/grounding；角色缺失保持为空，趋向动词允许动词角色为空。",
             "- 标识上下文指源文件、语料类型、替换词和两侧关注词/下标；相同 key 的上下文或文本变化需要人工核查。",
             "", "## 输入文件", ""]
    lines.extend(_table(
        ["文件", "路径", "格式", "句对数", "A true", "A false", "A 未判定", "B true", "B false", "B 未判定"],
        [[item.label, item.path, item.path.suffix.lower().lstrip("."), len(item.rows),
          *[item.stats[key] for key in ("a_true", "a_false", "a_null", "b_true", "b_false", "b_null")]]
         for item in files]))
    lines.extend(["", "### 句对状态分布", ""])
    lines.extend(_table(["文件", *pairs.PAIR_STATUSES], [
        [item.label, *[item.stats["pair_status_" + status] for status in pairs.PAIR_STATUSES]] for item in files]))
    lines.extend(["", "### 两侧检查项分布", ""])
    lines.extend(_table(["文件", "检查项", "true", "false", "未判定"], [
        [item.label, label, *[item.stats[name.replace("_extractable", "") + "_" + state]
                              for state in ("true", "false", "null")]]
        for item in files for name, (label, _, _) in BOOLEAN_FIELDS.items()]))
    lines.extend(["", "## 两两对比汇总", ""])
    lines.extend(_table(
        ["左", "右", "共同", "相同", "不同", "主字段一致率", "仅左", "仅右", "明细 CSV"],
        [[s.left.label, s.right.label, s.common, s.same, s.different, s.agreement_text,
          s.only_left, s.only_right, f"[{s.csv_name}]({s.csv_name})"] for s in summaries]))
    lines.extend(["", "## 一致性矩阵", "", "只统计共同 key；无共同项写 N/A，单侧项不进入分母。", ""])
    lines.extend(format_matrix_lines(files, summaries))
    for summary in summaries:
        lines.extend(["", f"## {summary.left.label} vs {summary.right.label}", "", "### 字段变化（共同 key）", ""])
        lines.extend(_table(["字段", "变化数"], [[label, summary.changes[name]] for name, label in FIELD_LABELS.items()]))
        lines.extend(["", "### A/B 布尔判定一致率", ""])
        lines.extend(_table(["字段", "双方均布尔", "相同", "布尔一致率", "true→false", "false→true", "涉及未判定"], [
            [BOOLEAN_FIELDS[name][0], stats.comparable, stats.same, agreement_text(stats.same, stats.comparable),
             stats.true_to_false, stats.false_to_true, stats.null_involved]
            for name, stats in summary.field_stats.items()]))
        lines.extend(["", "### 句对状态转移", ""])
        lines.extend(_table(["左状态", "右状态", "数量"], [[left, right, count] for (left, right), count
                                                       in sorted(summary.status_transitions.items())]))
        lines.extend(["", "### 按替换组合", ""])
        lines.extend(_table(["A关注词", "替换词", "共同", "相同", "不同", "仅左", "仅右"], [
            [word, replacement, *[counts[name] for name in ("common", "same", "different", "only_left", "only_right")]]
            for (word, replacement), counts in sorted(summary.by_replacement.items())]))
    return "\n".join([*lines, ""])


def _output_path(value: str | None) -> Path:
    if value:
        path = Path(value)
    else:
        base = ROOT_DIR / "analyze_results" / "compare_replace_pairs"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base / stamp
        index = 1
        while path.exists() or path.is_symlink():
            path = base / f"{stamp}_{index:02d}"
            index += 1
    if path.exists() or path.is_symlink():
        raise ValueError(f"输出目录已存在：{path}；请指定尚不存在的目录")
    return path


def publish_output_dir(staging: Path, target: Path) -> None:
    """独占创建目标目录；summary.md 最后发布，已有目录绝不覆盖。"""
    os.mkdir(target)
    try:
        for path in sorted(staging.iterdir(), key=lambda item: (item.name == SUMMARY_FILENAME, item.name)):
            os.replace(path, target / path.name)
        staging.rmdir()
    except BaseException:
        # 只清理本函数成功独占创建的目录，不触碰竞争者先创建的目标。
        shutil.rmtree(target, ignore_errors=True)
        raise


def _input_paths(inputs: Sequence[str], max_files: int) -> list[Path]:
    if len(inputs) < 2:
        raise ValueError("至少需要两个 replace 句对 JSONL / CSV")
    if max_files < 0:
        raise ValueError("--max-files 不能为负数")
    if max_files and len(inputs) > max_files:
        raise ValueError(f"输入文件数超过 --max-files {max_files}；0 表示不限制")
    paths = [Path(value) for value in inputs]
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{path}")
        if path.suffix.lower() not in (".jsonl", ".csv"):
            raise ValueError(f"只支持 replace JSONL / CSV：{path}")
        for other in paths[:index]:
            if pairs.same_file(path, other):
                raise ValueError(f"重复输入文件：{other} 与 {path}")
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", metavar="INPUT", help="至少两个 replace JSONL / 本工具导出的 CSV")
    parser.add_argument("-o", "--output-dir", help="尚不存在的目录；默认 analyze_results/compare_replace_pairs/<时间戳>")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES, help="最多输入文件数（默认 20，0 不限制）")
    parser.add_argument("--progress-every", type=int, default=50_000, help="每多少条句对显示进度；0 关闭")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = _input_paths(args.inputs, args.max_files)
        target = _output_path(args.output_dir)
        files = [LoadedPairs(path, _file_label(path, index), list(pairs.iter_pair_rows(
            path, progress_every=max(0, args.progress_every)))) for index, path in enumerate(paths, 1)]
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
        summaries: list[PairSummary] = []
        try:
            for index, left in enumerate(files):
                for right in files[index + 1:]:
                    comparison = Comparison(left.by_key, right.by_key)
                    csv_name = f"{left.label}__vs__{right.label}.csv"
                    write_pair_csv(comparison, staging / csv_name)
                    summary = PairSummary(left, right, csv_name, comparison.common, comparison.same,
                                          comparison.different, comparison.only_left, comparison.only_right,
                                          comparison.changes, comparison.field_stats, comparison.status_transitions,
                                          comparison.by_replacement)
                    summaries.append(summary)
                    print(f"{left.label} vs {right.label}：共同={summary.common}, 相同={summary.same}, "
                          f"不同={summary.different}, 主字段一致率={summary.agreement_text}")
            (staging / SUMMARY_FILENAME).write_text(format_overall_markdown(files, summaries), encoding="utf-8")
            publish_output_dir(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print("\n".join(format_matrix_lines(files, summaries)))
    print(f"总体报告：{target / SUMMARY_FILENAME}")
    if any(summary.common == 0 for summary in summaries):
        print("警告：至少一对没有共同 candidate_key；对应一致率为 N/A，报告仍已保存。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
