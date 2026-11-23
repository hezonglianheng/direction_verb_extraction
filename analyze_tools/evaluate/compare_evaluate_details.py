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
            "文件",
            "route",
            "记录数",
            "有身份",
            "无身份",
            "prediction 分布",
            "无结论原因",
            "参考标签分布",
        ],
        rows,
    )


def _pair_section(comparison: common.Comparison) -> list[str]:
    lines = [
        f"### {comparison.left_label} vs {comparison.right_label}",
        "",
        f"- 共同身份：{comparison.common}；仅左：{comparison.only_left}；"
        f"仅右：{comparison.only_right}；并集：{comparison.union}",
        f"- 身份内容冲突（句对不同，不计入分母）：{comparison.identity_conflicts}",
        f"- 结论一致率（含未判定，分母=共同身份）："
        f"{comparison.common_agreement_text}",
        f"- 有效结论一致率（双方都有标签）："
        f"{comparison.defined_agreement_text}；涉及无结论：{comparison.unresolved}",
        f"- 参考标签一致率："
        f"{common.agreement_text(comparison.reference_same, comparison.reference_comparable)}",
        "",
        "#### 结论转移",
        "",
    ]
    lines.extend(
        common._table(
            ["左侧 prediction", "右侧 prediction", "数量"],
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
        "# evaluate details 多文件一致性",
        "",
        "## 口径说明",
        "",
        "- 对齐身份：优先 `(line_no, candidate.index, candidate.focus_index, "
        "replacement)`，并带上 details 的 `input_file`；没有 candidate 时退回 `id`。",
        "  物理行号、JSONL 记录顺序与单看 `id` 都不参与跨语料对齐。",
        "- `input_file` 不同的记录即使行号/候选相同也不会强行配对；只在一份文件内"
        "重复身份会直接报错。",
        "- 主比较字段是 `prediction`：`same` / `different` / 无结论（null）。",
        "  `reference` 只作上下文与参考标签一致率，不代替 prediction。",
        "- 结论一致率（含未判定）把 null/null 计为相同状态；有效结论一致率只在"
        "双方都有标签时计算；零分母写 `N/A (0/0)`。",
        "- 差异本身不是错误；工具只报告一致性，不改写输入。",
        "",
        "## 输入文件",
        "",
    ]
    lines.extend(_input_table(files))
    lines.extend(["", "## 两两对比汇总", ""])
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
            "单元格为 `结论一致率（含未判定，相同/共同身份）`；对角线写 `—`，"
            "无共同身份写 `N/A (0/0)`。",
            "",
        ]
    )
    lines.extend(common.format_matrix_lines(labels, lookup))
    lines.extend(["", "## 逐对明细", ""])
    for comparison in comparisons:
        lines.extend(_pair_section(comparison))
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
