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
            "文件",
            "route",
            "记录数",
            "有身份",
            "无身份",
            "prediction 分布",
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
        ["项目", "值"],
        [
            ["文件", str(annotate.path)],
            ["记录类型", annotate.kind],
            ["标注图", annotate.mode or "—"],
            ["记录数", annotate.total],
            ["总结果分布", common.distribution(annotate.total_counts)],
            ["状态分布", common.distribution(annotate.status_counts)],
            ["投票状态", common.distribution(annotate.voting_status_counts)],
            ["模型", ", ".join(annotate.models) or "—"],
            ["缺票模型", common.distribution(annotate.missing_by_model)],
            ["无效票模型", common.distribution(annotate.invalid_by_model)],
        ],
    )


def _comparison_section(comparison: common.Comparison) -> list[str]:
    lines = [
        f"### {comparison.left_label} vs {comparison.right_label}",
        "",
        f"- 共同身份：{comparison.common}；仅左：{comparison.only_left}；"
        f"仅右：{comparison.only_right}；并集：{comparison.union}",
        f"- 右侧缺票/不适用：{comparison.right_unavailable}；"
        f"左侧身份不唯一：{comparison.left_ambiguous}；"
        f"身份内容冲突：{comparison.identity_conflicts}",
        f"- 结论一致率（含未判定，分母=共同身份）："
        f"{comparison.common_agreement_text}",
        f"- 有效结论一致率（双方都有标签）："
        f"{comparison.defined_agreement_text}；涉及无结论：{comparison.unresolved}",
        f"- 同义二分类一致率（双方均为 same/different）："
        f"{comparison.binary_agreement_text}",
        "",
        "#### 标签转移",
        "",
    ]
    lines.extend(
        common._table(
            ["左侧 prediction", "右侧标签", "数量"],
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
        "# evaluate details vs annotate 一致性",
        "",
        "## 口径说明",
        "",
        "- details 侧主字段是 `prediction`（`same` / `different` / 无结论）；"
        "annotate 总结果只看 `voting.winner_label`。",
        "- `voting.status=tie` / `no_valid_votes` 记无结论，**不退回**代表模型继承的 "
        "`judgement.same_meaning`；采纳 `not_spatial` / `not_fluent_*` 时该字段也可能是 "
        "null，不是投票共识。",
        "- 各模型结果直接取 `voting.ballots[model].label`，不重新投票。"
        "`missing_models` 中的模型记缺票，不计入共同项分母；"
        "有 ballot 但 label 为 null 记弃权，仍计入共同项。",
        "- 对齐身份优先用 `(line_no, candidate.index, candidate.focus_index, "
        "replacement)`；没有 candidate 时退回 `id`。句对内容不一致的身份单独计数，"
        "不强行配对。",
        "- 同时给出三种口径：结论一致率（含未判定）、有效结论一致率（双方都有标签）、"
        "同义二分类一致率（双方均为 same/different）。annotate 侧的 `not_spatial` /"
        "`not_fluent_*` 是有效标签，但不进入二分类分母。",
        "- 差异本身不是错误；参考标签与理由只作审计，不进入一致率。",
        "",
        "## evaluate details 输入",
        "",
    ]
    lines.extend(_details_input_table(files))
    lines.extend(["", "## annotate 输入", ""])
    lines.extend(_annotate_input_table(annotate))
    lines.extend(["", "## 跨层对比汇总", ""])
    lines.extend(
        common._table(
            common.COMPARISON_COLUMNS,
            common.comparison_summary_rows(comparisons),
        )
    )
    lines.extend(
        [
            "",
            "## 一致性矩阵",
            "",
            "单元格为 `结论一致率（含未判定，相同/共同身份）`；"
            "无共同身份写 `N/A (0/0)`。模型列来自 `voting.ballots`；"
            "单模型 annotate 结果没有模型列。",
            "",
        ]
    )
    lines.extend(_matrix(files, columns, lookup))
    lines.extend(["", "## 逐项明细", ""])
    for comparison in comparisons:
        lines.extend(_comparison_section(comparison))
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
