#!/usr/bin/env python3
# encoding: utf8
"""比较多个 evaluate details JSONL 之间的逐条标注一致性。

输入是 ``python -m evaluate`` 写出的 ``<模型>_step<N>_<时间>.details.jsonl``。
对齐键优先用 ``(line_no, candidate.index, candidate.focus_index, replacement)``，
并带上 details 里的 ``input_file``；没有 candidate 时退回 ``id``。
JSONL 行号与记录出现顺序不参与对齐。

用法::

    python analyze_tools/evaluate/compare_evaluate_details.py a.details.jsonl b.details.jsonl
    python -m analyze_tools.evaluate.compare_evaluate_details a.details.jsonl b.details.jsonl \\
        --output-dir analyze_results/evaluate_details_check

退出码：0=每对都有共同身份；1=至少一对没有共同身份（报告仍生成）；
2=输入/输出错误（格式不符、重复身份、输出目录已存在等）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if __package__:
    from . import _common as common
    from . import _details as details_mod
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _common as common  # type: ignore
    import _details as details_mod  # type: ignore


COMPARISON_DIR_NAME = "compare_evaluate_details"


def _input_table(files: list[details_mod.LoadedDetails]) -> list[str]:
    rows = [
        [
            f"{item.index:02d}",
            item.path,
            item.route_text,
            item.total,
            item.keyed,
            item.unkeyed,
            common.distribution(item.prediction_counts),
            common.distribution(item.error_counts),
            common.distribution(item.reference_counts),
        ]
        for item in files
    ]
    return common._table(
        [
            "编号",
            "文件路径",
            "评测路径 (route)",
            "记录总数",
            "有身份",
            "无身份",
            "模型预测分布 (prediction)",
            "无结论原因",
            "参考标签分布 (reference)",
        ],
        rows,
    )


def _pair_section(comparison: common.Comparison) -> list[str]:
    lines = [
        f"### {comparison.left_label} vs {comparison.right_label}",
        "",
        f"- **共同身份**：{comparison.common}；**仅左文件有**：{comparison.only_left}；"
        f"**仅右文件有**：{comparison.only_right}；**两侧并集**：{comparison.union}",
        f"- 身份内容冲突（句对不同，不计入分母）：{comparison.identity_conflicts}",
        f"- **结论一致率（含未判定，分母 = 共同身份）**："
        f"{comparison.common_agreement_text}",
        f"- **有效结论一致率（只比双方都有标签）**："
        f"{comparison.defined_agreement_text}；**涉及无结论**：{comparison.unresolved}",
        f"- **参考标签一致率 (`reference`)**："
        f"{common.agreement_text(comparison.reference_same, comparison.reference_comparable)}",
        "",
        "#### 结论转移表（`prediction` 变化明细）",
        "",
        "- **这张表看什么**：把共同身份记录中，左文件预测变成右文件预测的各种组合计数列出来。",
        "- **怎么读**：`same → same`、`different → different` 表示判断一致；`same ↔ different` 是实质性结论分歧；出现 `（无结论）` 表示一侧或两侧跑失败/超限（如 `model_error` / `invalid_json`）。",
        "",
    ]
    lines.extend(
        common._table(
            ["左侧预测 (prediction)", "右侧预测 (prediction)", "条数"],
            common.transition_rows(comparison.label_transitions),
        )
    )
    lines.append("")
    return lines


def format_summary(
    files: list[details_mod.LoadedDetails],
    comparisons: list[common.Comparison],
    lookup: dict[tuple[int, int], common.Comparison],
) -> str:
    labels = [item.label for item in files]
    lines = [
        "# evaluate details 多文件一致性对比报告",
        "",
        "这份报告用来对比**多份「评测明细（evaluate `.details.jsonl`）」**，看看不同模型、提示词或实验批次在相同测试句对上的预测结果有多一致。",
        "",
        "## 一分钟看懂这份报告",
        "",
        "### 1. 推荐阅读顺序",
        "",
        "1. 先看 **「输入文件概览」**：确认每个文件有多少条记录、评测的是哪条路（`route`），以及有多少「无身份」占位行或「无结论原因」（如 `model_error`、`invalid_json`）。",
        "2. 再看 **「两两对比汇总」和「一致性矩阵」**：先看整体一致性，注意对比「结论一致率（含未判定）」和「有效结论一致率」。",
        "3. 最后看 **「逐对明细」**：看具体有多少条从 `same` 变成了 `different`，以及有没有发生「身份内容冲突」。",
        "",
        "### 2. 看数字前先避开这 4 个坑",
        "",
        "- **主一致率比的是 `prediction`（模型预测），不是 `reference`（参考答案）**：`reference` 只作为对照上下文展示，不会用来代替模型预测。",
        "- **两边都出错/无结论也会算进「含未判定一致率」**：如果两边同一条都返回 `null`，含未判定一致率会记为相同；想看双方真正给出判断时有多像，请看 **「有效结论一致率」**。",
        "- **跨文件不会“硬对齐”不同来源或不同内容**：只有 `input_file` 和候选身份都相同、且句对文本一致才进入共同分母；如果同身份但句子文本不同，会被记为 **「身份内容冲突」** 并剔除出分母。",
        "- **「无身份」占位行不参与配对**：例如补跑失败或 JSON 解析损坏留下的占位记录（`invalid_json`），只在输入文件表里统计个数。",
        "",
        "## 白话名词对照表",
        "",
    ]
    lines.extend(
        common._table(
            ["报告里的说法", "对应的代码/原字段", "大白话解释"],
            [
                ["对齐身份", "`(input_file, line_no, candidate.index, focus_index, replacement)`", "没有 candidate 时退回 `id`；物理行号和记录顺序不参与跨语料对齐。"],
                ["模型预测 (`prediction`)", "`same` / `different` / `null`", "评测模型认为原句与改写句空间语义是否相同；`null` 记为 `（无结论）`。"],
                ["参考标签 (`reference`)", "输入自带标准/参考答案", "仅用于核对两边测试集参考标签是否一致，不代替模型预测。"],
                ["评测路径 (`route`)", "`step` / `mode` / `corpus_type`", "标注这批评测跑的是第几步、哪种标注图和哪种语料类型。"],
                ["有身份 / 无身份", "能否算出对齐身份", "正常记录为「有身份」；损坏/补跑失败占位行（如 `invalid_json`）没有候选信息，记为「无身份」。"],
                ["身份内容冲突", "同身份但原句/替换句不同", "可能两份文件指向了不同版本的源语料；为防误统计，该条不进一致率分母。"],
                ["含未判定 vs 有效结论一致率", "一致率双口径", "前者分母=共同身份（含 `null=null`）；后者只比双方都给出了有效标签的记录。"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 输入文件概览",
            "",
            "- **这张表看什么**：确认每份评测文件的规模、配置路径，以及报错无结论的分布。",
            "- **怎么读**：`记录总数 = 有身份 + 无身份`；若 `无结论原因` 里有很多 `model_error` 或 `timeout`，应优先补跑失败条目。",
            "",
        ]
    )
    lines.extend(_input_table(files))
    lines.extend(
        [
            "",
            "## 两两对比汇总",
            "",
            "- **这张表看什么**：总览任意两份评测文件在共同身份上的各种一致率与异常计数。",
            "- **怎么读（分母口径）**：`结论一致率（含未判定） = 相同 / 共同`；`有效结论一致率` 剔除了任意一侧为 `null` 的条；`同义二分类一致率` 只统计双方均为 `same` 或 `different` 的条；`仅左`、`仅右`、`身份内容冲突` 均不进一致率分母。",
            "",
        ]
    )
    lines.extend(
        common._table(
            common.COMPARISON_COLUMNS,
            common.comparison_summary_rows(comparisons),
        )
    )
    lines.extend(
        [
            "",
            "## 一致性矩阵（多文件横向速查）",
            "",
            "- **这张表看什么**：当对比 3 个及以上评测文件时，速查任意两份文件的总体一致率。",
            "- **怎么读**：单元格为 `结论一致率（含未判定，相同/共同身份）`；对角线写 `—`，无共同身份写 `N/A (0/0)`。",
            "",
        ]
    )
    lines.extend(common.format_matrix_lines(labels, lookup))
    lines.extend(["", "## 逐对明细", ""])
    for comparison in comparisons:
        lines.extend(_pair_section(comparison))
    lines.extend(
        [
            "## 明细 CSV 怎么看",
            "",
            "- 同目录下为每对文件生成了对应的 `.csv`（可用 Excel 直接打开），逐条保存了所有成功对齐的记录（发生「身份内容冲突」的记录不会混入）。",
            "- **常用排查建议**：在 `比较结果` 列筛选 `不同`（或 `仅左侧` / `仅右侧`），再看 `左最终标签`、`右最终标签` 以及两侧原句/替换句与原因字段即可定位差异。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较多个 evaluate details JSONL 之间的逐条标注一致性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法::")[-1],
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="两份或更多 evaluate details JSONL",
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
        default=common.DEFAULT_MAX_FILES,
        help=f"最多输入文件数（默认 {common.DEFAULT_MAX_FILES}；0 表示不限制）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="每读取多少条记录打印一次进度（默认 50000；0 关闭）",
    )
    return parser.parse_args(argv)


def _print_terminal(
    files: list[details_mod.LoadedDetails],
    comparisons: list[common.Comparison],
    target: Path,
) -> None:
    print(f"已加载 {len(files)} 份 details，完成 {len(comparisons)} 对比较")
    print()
    print("| 编号 | 文件 | 记录数 | 有身份 | 无身份 | prediction 分布 |")
    print("| --- | --- | ---: | ---: | ---: | --- |")
    for item in files:
        print(
            f"| {item.index:02d} | {item.path} | {item.total} | "
            f"{item.keyed} | {item.unkeyed} | "
            f"{common.distribution(item.prediction_counts)} |"
        )
    print()
    print("| 对比 | 共同 | 结论一致（含未判定） | 有效结论一致 | 明细 CSV |")
    print("| --- | ---: | ---: | ---: | --- |")
    for comparison in comparisons:
        print(
            f"| {comparison.left_label} vs {comparison.right_label} | "
            f"{comparison.common} | {comparison.common_agreement_text} | "
            f"{comparison.defined_agreement_text} | {comparison.csv_name} |"
        )
    print()
    print(f"输出目录：{target}")
    print(f"总体报告：{target / common.SUMMARY_FILENAME}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = common.input_paths(
            args.inputs,
            args.max_files,
            minimum=2,
            kind="evaluate details JSONL",
        )
        target = common.plan_output_dir(args.output_dir, COMPARISON_DIR_NAME)
        files = [
            details_mod.load_details(
                path,
                index=index,
                label=common.file_label(path, index),
                progress_every=max(0, args.progress_every),
            )
            for index, path in enumerate(paths, 1)
        ]
        staging = common.make_staging_dir(target)
        comparisons: list[common.Comparison] = []
        lookup: dict[tuple[int, int], common.Comparison] = {}
        try:
            pair_total = len(files) * (len(files) - 1) // 2
            pair_no = 0
            for index, left in enumerate(files):
                for right in files[index + 1:]:
                    pair_no += 1
                    comparison = common.compare_maps(
                        left.rows_by_key,
                        right.rows_by_key,
                        left_label=left.label,
                        right_label=right.label,
                        check_context=True,
                    )
                    csv_name = f"{left.label}__vs__{right.label}.csv"
                    comparison.csv_name = csv_name
                    common.write_pair_csv(comparison, staging / csv_name)
                    comparison.release_maps()
                    comparisons.append(comparison)
                    lookup[(left.index - 1, right.index - 1)] = comparison
                    print(
                        f"[{pair_no}/{pair_total}] {left.label} vs {right.label}："
                        f"共同={comparison.common}, 相同={comparison.same}, "
                        f"不同={comparison.different}, "
                        f"结论一致率={comparison.common_agreement_text}, "
                        f"有效结论一致率={comparison.defined_agreement_text}"
                    )
            (staging / common.SUMMARY_FILENAME).write_text(
                format_summary(files, comparisons, lookup), encoding="utf-8"
            )
            common.publish_output_dir(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    _print_terminal(files, comparisons, target)
    if any(comparison.common == 0 for comparison in comparisons):
        print(
            "\n警告：至少一对没有共同身份；对应一致率为 N/A (0/0)，"
            "报告仍已保存。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
