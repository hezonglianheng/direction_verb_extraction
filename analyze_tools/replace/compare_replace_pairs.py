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
    lines = [
        "# replace 句对结果对比报告",
        "",
        "这份报告用来对比**多份「句对替换初筛（replace）」结果**，看看不同模型、不同参数或不同批次在同一个「原句 vs 改写句」句对上，给出的判断是否一致。",
        "",
        "## 一分钟看懂这份报告",
        "",
        "### 1. 先分清两对方向词（最容易看混）",
        "",
        "- **左 / 右**：指的是**参与对比的两个文件**（例如 `01_run_a` 和 `02_run_b`）。",
        "- **A / B**：指的是**同一个句对里的两句话**：**A = 原句**，**B = 把关注词换成替换词后的改写句**。",
        "  - 举例：明细里的「左B判定」意思就是“**左文件**对**改写句（B）**的判定”。",
        "",
        "### 2. 推荐阅读顺序",
        "",
        "1. 先看 **「输入文件」和「句对状态分布」**：确认每份文件一共多少条句对，有多少条通过（`true`），有多少条根本没拿到模型结果（`error` / `limit` / `no_llm`）。",
        "2. 再看 **「两两对比汇总」和「一致性矩阵」**：一眼看出每对文件在核心判定上的整体一致率是多少。",
        "3. 最后看 **「逐对详细分析」**：如果一致率不理想，顺着下面的「字段变化」「布尔一致率方向（true→false / false→true）」「状态转移」和「按替换组合」，就能定位差异到底出在语法、空间自洽，还是某个具体替换词上。",
        "",
        "### 3. 看数字前先避开这 4 个坑",
        "",
        "- **主字段一致率高 ≠ 句对合格率高**：主一致率只代表“两次结果写得一模一样”，不代表句对合格，哪怕两边都判不通过或都没判定，也会算成“相同”。",
        "- **两边都「未判定（null）」也会算进主一致率**：所以如果 `未判定` 很多，一定要看后面的「A/B 布尔判定一致率」——那张表会把未判定剔除，只比双方都明确给出 true/false 的项。",
        "- **「仅左 / 仅右」不参与一致率分母**：两份文件只按各自共同拥有的句对算一致率；如果两边候选句对重合很少，一致率只代表交集那一部分。",
        "- **只比较“句对自身”的筛查结果**：不读从原句继承的整句旧结论；语法、空间自洽、三元组、句子文本或上下文的变化不计入主一致率，会在详细部分单独列出来提醒核查。",
        "",
        "## 白话名词对照表",
        "",
    ]
    lines.extend(_table(
        ["报告里的说法", "对应的代码/原字段", "大白话解释"],
        [
            ["唯一对齐键", "`candidate_key`", "由“句子 ID + 关注词下标 + 替换词”组成；相同 key 才算同一个候选句对，绝不按文件行号强行配对。"],
            ["句对状态：通过", "`pair_screening.status = true`", "原句（A）和改写句（B）三项检查全部明确为 true，句对才算真正通过，可进入后续标注。"],
            ["句对状态：不通过", "`pair_screening.status = false`", "没有报错或跳过，且至少一侧有一项检查明确为 false（可能是空间信息不行，也可能是语法或空间自洽没过）。"],
            ["句对状态：调用异常", "`pair_screening.status = error`", "调用模型失败、返回格式异常或字段不完整，没能得到可靠判断（不是“不通过”）。"],
            ["句对状态：超额未送", "`pair_screening.status = limit`", "达到了设定的发送条数上限，这条句对根本没发给模型（不是“不通过”）。"],
            ["句对状态：未启用模型", "`pair_screening.status = no_llm`", "运行时关闭了 LLM 筛选，直接落盘占位（不是“不通过”）。"],
            ["A判定 / B判定", "`pair_screening.a/b.extractable`", "该侧句子的关注词是否表达本类语料认定的空间信息、是否值得提取（true / false / 未判定）。"],
            ["A/B 语法正确", "`pair_screening.a/b.grammatical`", "该侧句子语序语法是否通顺（true / false / 未判定）。"],
            ["A/B 空间自洽", "`pair_screening.a/b.spatial_coherent`", "该侧句子的空间语义是否合乎常理（true / false / 未判定）。"],
            ["未判定", "`null`（空值）", "模型没返回或未送审该字段；统计时不会当成 false。"],
            ["主比较字段", "`(A判定, B判定, 句对状态)`", "两两对比时判断“相同 / 不同”只看这三项是否完全一致。"],
            ["三元组", "`(target, reference, verb)`", "关注词对应的目标物、参照物、动词文本；趋向动词允许动词角色为空，角色缺失保持为空。"],
            ["标识上下文", "源文件、语料类型、替换词、关注词/下标", "用来核对同 key 是否真的来自同一句和同一个替换位置；如果发生变化说明数据来源或版本可能有异。"],
        ],
    ))
    lines.extend(["", "## 输入文件概览", ""])
    lines.extend([
        "- **这张表看什么**：先摸清每个输入文件有多少条句对，以及 A（原句）/ B（改写句）在“是否表达空间信息（`extractable`）”上的分布。",
        "- **怎么读**：`A true / false / 未判定` 和 `B true / false / 未判定` 各自相加都等于该文件的 `句对数`。",
        "- **注意事项**：这里的 true/false 只看单侧 `extractable`，不代表整个句对通过（即使两侧 `extractable` 都是 true，只要语法或空间自洽为 false，整条句对状态依然是 false）。",
        "",
    ])
    lines.extend(_table(
        [
            "文件标签", "路径", "格式", "句对总数",
            "A 通过 (A true)", "A 不通过 (A false)", "A 未判定",
            "B 通过 (B true)", "B 不通过 (B false)", "B 未判定",
        ],
        [[item.label, item.path, item.path.suffix.lower().lstrip("."), len(item.rows),
          *[item.stats[key] for key in ("a_true", "a_false", "a_null", "b_true", "b_false", "b_null")]]
         for item in files]))
    lines.extend(["", "### 句对最终状态分布 (`pair_screening.status`)", ""])
    lines.extend([
        "- **这张表看什么**：看每个文件最终能放行进入下一阶段的句对有多少。",
        "- **怎么读**：状态按优先级 `error > limit > no_llm > false > true` 汇总——只要任意一侧出现异常或未送审，句对状态就会保留为对应提示状态，不会被掩盖成 false。",
        "",
    ])
    lines.extend(_table(["文件标签", *pairs.PAIR_STATUSES], [
        [item.label, *[item.stats["pair_status_" + status] for status in pairs.PAIR_STATUSES]] for item in files]))
    lines.extend(["", "### 两侧细项检查分布", ""])
    lines.extend([
        "- **这张表看什么**：把每个文件的原句（A）和改写句（B）拆成 6 个细项（判定、语法正确、空间自洽），看卡住句对的主要原因。",
        "- **怎么读**：若 B 侧大量出现 `B语法正确=false` 或 `B空间自洽=false`，说明很多替换词造成了病句或空间逻辑不通。",
        "",
    ])
    lines.extend(_table(["文件标签", "检查项", "是 (true)", "否 (false)", "未判定 (null)"], [
        [item.label, label, *[item.stats[name.replace("_extractable", "") + "_" + state]
                              for state in ("true", "false", "null")]]
        for item in files for name, (label, _, _) in BOOLEAN_FIELDS.items()]))
    lines.extend(["", "## 两两对比汇总", ""])
    lines.extend([
        "- **这张表看什么**：总览任意两个文件在共同句对上的核心一致率。",
        "- **怎么读（分母口径）**：`主字段一致率 = 相同 / 共同`（只统计两侧都有的 `candidate_key`；完全无共同项写 `N/A (0/0)`）。",
        "- **注意事项**：`仅左` / `仅右` 表示只有一边出现的句对，不进入一致率分母；如果数量很大，说明两次运行的输入或筛选范围不一致。",
        "",
    ])
    lines.extend(_table(
        [
            "左文件", "右文件", "共同句对数", "主字段相同", "主字段不同",
            "主字段一致率 (相同/共同)", "仅左文件有", "仅右文件有", "逐条明细 CSV",
        ],
        [[s.left.label, s.right.label, s.common, s.same, s.different, s.agreement_text,
          s.only_left, s.only_right, f"[{s.csv_name}]({s.csv_name})"] for s in summaries]))
    lines.extend([
        "",
        "## 一致性矩阵（主字段一致率）",
        "",
        "- **这张表看什么**：多个文件互相比较时的速查网格，单元格数字与上表的 `主字段一致率` 相同。",
        "- **怎么读**：横竖交叉点就是对应两个文件在共同 key 上 `(A判定, B判定, 句对状态)` 全相同的比例；对角线 `—` 表示不与自己比较，无共同项写 `N/A (0/0)`，仅单侧项不进入分母。",
        "",
    ])
    lines.extend(format_matrix_lines(files, summaries))
    lines.extend(["", "## 逐对详细分析", ""])
    for summary in summaries:
        lines.extend(["", f"### {summary.left.label} vs {summary.right.label}", "", "#### 1. 字段变化统计（共同 key）", ""])
        lines.extend([
            "- **这张表看什么**：在两侧共有的句对里，各个字段分别有多少条发生了变化（含从布尔变成未判定）。",
            "- **怎么读**：分母是这一对的 `共同句对数`；一条句对可以同时有多个字段变化。若 `A/B句子`、`A/B切片` 或 `标识上下文` 变化数不为 0，说明相同 key 下的文本或来源对不上，建议打开明细 CSV 核查。",
            "",
        ])
        lines.extend(_table(["字段", "变化条数"], [[label, summary.changes[name]] for name, label in FIELD_LABELS.items()]))
        lines.extend(["", "#### 2. A/B 布尔判定一致率（剔除未判定）", ""])
        lines.extend([
            "- **这张表看什么**：分别看 6 个细项在“双方都真正给出了 true/false”的句对里有多一致，用来避免大量 `未判定 (null)` 把一致率虚抬或拉低。",
            "- **怎么读**：`布尔一致率 = 相同 / 双方均布尔`（`双方均布尔` 为 0 时写 `N/A (0/0)`）。`true→false` 表示左文件为 true、右文件为 false；`false→true` 相反。如果一边明显偏多，说明其中一份判定标准更严格或更宽松。`涉及未判定` 只单独列出，不进分母。",
            "",
        ])
        lines.extend(_table(["字段", "双方均布尔 (分母)", "相同 (分子)", "布尔一致率", "左true→右false", "左false→右true", "涉及未判定 (不进分母)"], [
            [BOOLEAN_FIELDS[name][0], stats.comparable, stats.same, agreement_text(stats.same, stats.comparable),
             stats.true_to_false, stats.false_to_true, stats.null_involved]
            for name, stats in summary.field_stats.items()]))
        lines.extend(["", "#### 3. 句对状态转移（`pair_screening.status`）", ""])
        lines.extend([
            "- **这张表看什么**：看共同句对从左文件的最终状态变成右文件最终状态的流向。",
            "- **怎么读**：`左状态 = 右状态`（例如 `true → true`）代表稳定；`true → false` 或 `false → true` 代表结论反转；出现 `error` / `limit` / `no_llm` 转成 `true` / `false`，通常说明其中一次跑出了异常或限额。",
            "",
        ])
        lines.extend(_table(["左文件状态", "右文件状态", "句对数量"], [[left, right, count] for (left, right), count
                                                           in sorted(summary.status_transitions.items())]))
        lines.extend(["", "#### 4. 按替换组合分组统计", ""])
        lines.extend([
            "- **这张表看什么**：把差异按“原句关注词 + 替换词”拆开看，定位是不是某几个特定替换词特别不稳定。",
            "- **怎么读**：样本太少（比如只有 1~2 条）的组合容易波动，优先关注 `共同` 较多且 `不同` 比例较高的替换组合。",
            "",
        ])
        lines.extend(_table(["A关注词 (原词)", "替换词 (B词)", "共同", "主字段相同", "主字段不同", "仅左文件有", "仅右文件有"], [
            [word, replacement, *[counts[name] for name in ("common", "same", "different", "only_left", "only_right")]]
            for (word, replacement), counts in sorted(summary.by_replacement.items())]))
    lines.extend([
        "",
        "## 明细 CSV 怎么看",
        "",
        "- 每对文件在同目录下都生成了对应的 `.csv`（可用 Excel 直接打开），里面逐条列出了所有共同句对以及 `仅左侧` / `仅右侧` 的句对。",
        "- **常用筛选建议**：",
        "  1. 在 `比较结果` 列筛选 `不同`：直接看主判定不一致的句对，对照 `左A理由` / `左B理由` 与 `右A理由` / `右B理由` 检查原因。",
        "  2. 在 `变化字段` 列筛选非空行：即使 `比较结果=相同`，也能快速揪出“语法、空间自洽、三元组或文本”其实发生了变化的条目。",
        "  3. 记住字段前缀含义：**左/右**指文件，**A/B**指原句侧与改写侧。",
        "",
    ])
    return "\n".join(lines)


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
