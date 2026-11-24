#!/usr/bin/env python3
# encoding: utf8
"""把 evaluate details 的逐条标注与 annotate 投票结果做一致性比较。

输入一份或多份 ``evaluate`` details JSONL，另加一份 annotate 结果 JSONL
（通常是 ``python -m annotate --models ...`` 的投票最终结果）。比较对象：

- **annotate 总结果**：只取 ``voting.winner_label``；``tie`` /
  ``no_valid_votes`` 记无结论，不退回代表模型继承的 ``judgement.same_meaning``；
- **voting 各模型结果**：直接取 ``voting.ballots[model].label``，
  这是本次投票实际采用的票，不重新投票。``missing_models`` 中的模型记缺票，
  不计入共同项分母；有 ballot 但 label 为 null 记弃权，仍计入共同项。

details 与 annotate 用同一候选身份对齐：
``(line_no, candidate.index, candidate.focus_index, replacement)``；没有
candidate 时退回 ``id``。句对内容不一致的身份按「身份内容冲突」单列，不强行配对。

用法::

    python analyze_tools/evaluate/compare_evaluate_annotate.py a.details.jsonl \\
        --annotate results/spatial_pairs/vote.jsonl
    python -m analyze_tools.evaluate.compare_evaluate_annotate \\
        a.details.jsonl b.details.jsonl --annotate vote.jsonl \\
        --output-dir analyze_results/evaluate_annotate_check

退出码：0=所有比较都有共同身份；1=至少一个比较没有共同身份（报告仍生成）；
2=输入/输出错误。
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


COMPARISON_DIR_NAME = "compare_evaluate_annotate"


def _details_input_table(files: list[details_mod.LoadedDetails]) -> list[str]:
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
        ],
        [
            [
                f"{item.index:02d}",
                item.path,
                item.route_text,
                item.total,
                item.keyed,
                item.unkeyed,
                common.distribution(item.prediction_counts),
                common.distribution(item.error_counts),
            ]
            for item in files
        ],
    )


def _annotate_input_table(annotate: details_mod.LoadedAnnotate) -> list[str]:
    return common._table(
        ["项目", "统计值"],
        [
            ["文件路径", str(annotate.path)],
            ["记录类型", annotate.kind],
            ["标注图 (mode)", annotate.mode or "—"],
            ["记录总数", annotate.total],
            ["总结果分布", common.distribution(annotate.total_counts)],
            ["状态分布", common.distribution(annotate.status_counts)],
            ["投票状态", common.distribution(annotate.voting_status_counts)],
            ["参与模型", ", ".join(annotate.models) or "—"],
            ["缺票模型", common.distribution(annotate.missing_by_model)],
            ["无效票模型", common.distribution(annotate.invalid_by_model)],
        ],
    )


def _comparison_section(comparison: common.Comparison) -> list[str]:
    lines = [
        f"### {comparison.left_label} vs {comparison.right_label}",
        "",
        f"- **共同身份**：{comparison.common}；**仅左（details 有）**：{comparison.only_left}；"
        f"**仅右（annotate 有）**：{comparison.only_right}；**两侧并集**：{comparison.union}",
        f"- 右侧缺票/不适用：{comparison.right_unavailable}；"
        f"左侧身份不唯一：{comparison.left_ambiguous}；"
        f"身份内容冲突：{comparison.identity_conflicts}",
        f"- **结论一致率（含未判定，分母 = 共同身份）**："
        f"{comparison.common_agreement_text}",
        f"- **有效结论一致率（双方都有标签）**："
        f"{comparison.defined_agreement_text}；**涉及无结论**：{comparison.unresolved}",
        f"- **同义二分类一致率（双方均为 `same`/`different`）**："
        f"{comparison.binary_agreement_text}",
        "",
        "#### 标签转移表（`details prediction` → `annotate 标签`）",
        "",
        "- **这张表看什么**：看评测模型的 `same`/`different` 在正式标注（总结果或单模型投票）里具体对应成了哪些标签。",
        "- **怎么读**：除了 `same ↔ different` 之外，如果出现大量 `→ not_spatial` 或 `→ not_fluent_*`，说明正式标注阶段过滤掉了非空间或不顺畅句对，而评测模型仍当作二分类输出了。",
        "",
    ]
    lines.extend(
        common._table(
            ["左侧预测 (details prediction)", "右侧标签 (annotate)", "条数"],
            common.transition_rows(comparison.label_transitions),
        )
    )
    lines.append("")
    return lines


def _matrix(
    files: list[details_mod.LoadedDetails],
    columns: list[str],
    lookup: dict[tuple[int, int], common.Comparison],
) -> list[str]:
    header = ["details 文件"] + columns
    lines = [
        "| " + " | ".join(common._md(cell) for cell in header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(columns)) + " |",
    ]
    for row_index, item in enumerate(files):
        cells = [common._md(item.label)]
        for column_index, _ in enumerate(columns):
            comparison = lookup.get((row_index, column_index))
            cells.append(
                comparison.common_agreement_text if comparison else "N/A (0/0)"
            )
        lines.append("| " + " | ".join(common._md(cell) for cell in cells) + " |")
    return lines


def format_summary(
    files: list[details_mod.LoadedDetails],
    annotate: details_mod.LoadedAnnotate,
    comparisons: list[common.Comparison],
    lookup: dict[tuple[int, int], common.Comparison],
    columns: list[str],
) -> str:
    lines = [
        "# evaluate details vs annotate 跨层一致性报告",
        "",
        "这份报告用来把**小模型/评测集跑出的 `evaluate details`**，与**正式标注流水线 `annotate` 的结果（投票总结果，以及各模型单票）**逐条对账，看它们对同一批句对的判断是否一致。",
        "",
        "## 一分钟看懂这份报告",
        "",
        "### 1. 推荐阅读顺序",
        "",
        "1. 先看 **「evaluate details 输入」和「annotate 输入」**：核对两边文件、记录数、标注类型（投票汇总还是单模型），以及投票有没有出现大量平票（`tie`）或缺票模型。",
        "2. 再看 **「跨层对比汇总」和「一致性矩阵」**：如果 `annotate` 是多模型投票，矩阵里会同时列出「annotate 总结果」以及每个参与投票的「模型 XXX」，一眼就能看出评测结果更像哪位裁判、或和最终共识差多远。",
        "3. 最后看 **「逐项明细」**：用「标签转移表」确认差异到底是 `same ↔ different` 判断相反，还是 `annotate` 判成了 `not_spatial` / `not_fluent_*` / 平票无结论。",
        "",
        "### 2. 看数字前先避开这 4 个坑",
        "",
        "- **同时提供「三种一致率」，选对你要看的口径**：",
        "  1. `结论一致率（含未判定）`：分母=共同身份（连双方同时为 `（无结论）` 也算相同）。",
        "  2. `有效结论一致率`：剔除平票/报错等无结论项后，双方都有标签时的一致率。",
        "  3. `同义二分类一致率`：**只看双方都落在 `same`/`different` 的条**。因为 `evaluate` 通常只输出 `same`/`different`，而 `annotate` 还会输出 `not_spatial` / `not_fluent_*`（它们是有效标签，但不进二分类分母），对比“同义理解像不像”建议重点看这一列。",
        "- **「缺票」和「弃权」处理完全不同**：在对比 `voting.ballots` 各模型单票时，`missing_models` 里的模型叫 **缺票**，统计在「右侧缺票/不适用」里、**不进共同身份分母**；有选票但 `label=null` 叫 **弃权**，记为 `（无结论）` 并**继续计入共同身份分母**。",
        "- **投票总结果不偷看代表模型**：平票（`tie`）或无有效票（`no_valid_votes`）记为 `（无结论）`，**绝不退回**代表模型继承的 `judgement.same_meaning`。",
        "- **身份内容冲突不硬配**：如果同身份下原句/替换句文本对不上，会被单独记为「身份内容冲突」并剔除分母；单模型 `annotate` 输入只有 `annotate 总结果` 一列，没有模型列。",
        "",
        "## 白话名词对照表",
        "",
    ]
    lines.extend(
        common._table(
            ["报告里的说法", "对应的代码/原字段", "大白话解释"],
            [
                ["对齐身份", "`(line_no, candidate.index, focus_index, replacement)`", "没有 candidate 时退回 `id`；必须句对文本也一致才算对齐。"],
                ["左侧预测 (`details`)", "`prediction` (`same`/`different`/`null`)", "评测模型的预测结论。"],
                ["`annotate 总结果`", "`voting.winner_label`（投票）或还原标签（单模型）", "正式标注最终采纳的标签（含 `same`、`different`、`not_spatial`、`not_fluent_*`、`（无结论）`）。"],
                ["各模型单票 (`ballots`)", "`voting.ballots[model].label`", "投票时各参与模型投出的原始单票，直接读取、不重新算票。"],
                ["缺票 / 不适用 (`missing_models`)", "某模型未参与该条投票", "记入「右侧缺票/不适用」，不进入该模型的一致率分母。"],
                ["弃权 (`label=null`)", "有选票但未给出有效标签", "记为 `（无结论）`，仍进入共同身份分母。"],
                ["`not_spatial` / `not_fluent_*`", "正式标注的拦截标签", "属于有效标签（计入「有效结论一致率」分母），但不属于同义二分类（不计入「同义二分类一致率」分母）。"],
                ["左侧身份不唯一", "details 侧多条命中同一身份", "若 details 对同一候选身份有多条记录，会单独提示并谨慎对齐。"],
                ["身份内容冲突", "同身份但原句/替换句不一致", "两份文件可能源自不同语料版本，不计入分母也不进 CSV。"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## evaluate details 输入",
            "",
            "- **这张表看什么**：左侧参与对账的每份评测明细文件基本规模与预测分布。",
            "",
        ]
    )
    lines.extend(_details_input_table(files))
    lines.extend(
        [
            "",
            "## annotate 输入",
            "",
            "- **这张表看什么**：右侧作为对账基准的正式标注文件概况（属于「单模型」还是「投票汇总」、各标签和投票状态分布、有无缺票或无效票模型）。",
            "",
        ]
    )
    lines.extend(_annotate_input_table(annotate))
    lines.extend(
        [
            "",
            "## 跨层对比汇总",
            "",
            "- **这张表看什么**：把每份 `details` 文件分别与 `annotate 总结果` 以及每个投票模型对比的关键数字汇总成一张总表。",
            "- **怎么读**：重点关注 `结论一致率（含未判定）`、`有效结论一致率`、`同义二分类一致率` 三列的变化，以及右侧三列异常统计（`右侧缺票/不适用`、`左侧身份不唯一`、`身份内容冲突`）。",
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
            "## 一致性矩阵（横看评测文件，竖看总结果与各模型）",
            "",
            "- **这张表看什么**：单元格显示 `结论一致率（含未判定，相同/共同身份）`；无共同身份写 `N/A (0/0)`。",
            "- **注意事项**：模型列来自 `voting.ballots`；单模型 annotate 结果没有模型列。",
            "",
        ]
    )
    lines.extend(_matrix(files, columns, lookup))
    lines.extend(["", "## 逐项明细", ""])
    for comparison in comparisons:
        lines.extend(_comparison_section(comparison))
    lines.extend(
        [
            "## 明细 CSV 怎么看",
            "",
            "- 同目录下为每组对比（`annotate 总结果` 以及每个投票模型各自一对）生成了独立 `.csv` 文件，可用 Excel 打开。",
            "- **常用排查建议**：",
            "  1. 在 `比较结果` 列筛选 `不同`，对照 `左最终标签`（`details prediction`）和 `右最终标签`（`annotate` 标签）看差异方向。",
            "  2. 结合 `左参考标签` 和两侧判定理由列，可以快速分辨是评测模型漏判了非空间/不通顺句对，还是两边对空间语义本身看法不同。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比较 evaluate details 与 annotate 总结果 / voting 各模型结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法::")[-1],
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="一份或更多 evaluate details JSONL",
    )
    parser.add_argument(
        "--annotate",
        required=True,
        metavar="PATH",
        help="annotate 结果 JSONL（投票最终结果或单模型结果）",
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
        help=f"最多 details 文件数（默认 {common.DEFAULT_MAX_FILES}；0 表示不限制）",
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
    annotate: details_mod.LoadedAnnotate,
    comparisons: list[common.Comparison],
    target: Path,
) -> None:
    print(
        f"已加载 {len(files)} 份 details、1 份 annotate（{annotate.kind}），"
        f"完成 {len(comparisons)} 项比较"
    )
    print()
    print(
        "| details | 对比层 | 共同 | 右侧缺票 | 结论一致（含未判定） | "
        "有效结论一致 | 二分类一致 |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for comparison in comparisons:
        print(
            f"| {comparison.left_label} | {comparison.right_label} | "
            f"{comparison.common} | {comparison.right_unavailable} | "
            f"{comparison.common_agreement_text} | "
            f"{comparison.defined_agreement_text} | "
            f"{comparison.binary_agreement_text} |"
        )
    print()
    print(f"输出目录：{target}")
    print(f"总体报告：{target / common.SUMMARY_FILENAME}")


def _csv_name_for(details_label: str, model: str | None, model_index: int) -> str:
    if model is None:
        return f"{details_label}__vs__annotate_total.csv"
    return (
        f"{details_label}__vs__model_{model_index:02d}_"
        f"{common.safe_filename(model)}.csv"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = common.input_paths(
            args.inputs,
            args.max_files,
            minimum=1,
            kind="evaluate details JSONL",
        )
        annotate_paths = common.input_paths(
            [args.annotate],
            0,
            minimum=1,
            kind="annotate 结果 JSONL",
        )
        target = common.plan_output_dir(args.output_dir, COMPARISON_DIR_NAME)
        annotate = details_mod.load_annotate(
            annotate_paths[0],
            label=common.safe_filename(annotate_paths[0].stem),
            progress_every=max(0, args.progress_every),
        )
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
        model_layers = list(annotate.models) if annotate.is_voting else []
        columns = ["annotate 总结果"] + [
            f"模型 {model}" for model in model_layers
        ]
        try:
            for row_index, item in enumerate(files):
                layers: list[str | None] = [None, *model_layers]
                for column_index, model in enumerate(layers):
                    comparison = details_mod.build_cross_comparison(
                        item, annotate, model=model
                    )
                    comparison.csv_name = _csv_name_for(
                        item.label, model, column_index
                    )
                    common.write_pair_csv(
                        comparison, staging / comparison.csv_name
                    )
                    comparison.release_maps()
                    comparisons.append(comparison)
                    lookup[(row_index, column_index)] = comparison
                    print(
                        f"{item.label} vs {comparison.right_label}："
                        f"共同={comparison.common}, 相同={comparison.same}, "
                        f"不同={comparison.different}, "
                        f"缺票={comparison.right_unavailable}, "
                        f"结论一致率={comparison.common_agreement_text}, "
                        f"有效结论一致率={comparison.defined_agreement_text}"
                    )
            (staging / common.SUMMARY_FILENAME).write_text(
                format_summary(files, annotate, comparisons, lookup, columns),
                encoding="utf-8",
            )
            common.publish_output_dir(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    _print_terminal(files, annotate, comparisons, target)
    if any(comparison.common == 0 for comparison in comparisons):
        print(
            "\n警告：至少一项比较没有共同身份；对应一致率为 N/A (0/0)，"
            "报告仍已保存。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
