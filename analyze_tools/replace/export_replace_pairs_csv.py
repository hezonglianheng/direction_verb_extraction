#!/usr/bin/env python3
# encoding: utf8
"""导出 replace 句对：一行一个 candidate_key，A/B 字段并列。

只读 pair_screening 与对应侧的 grounding，不读取从原句继承的顶层 screening。
保留 false 和未判定句对，null 在 CSV 中为空串；输出使用 UTF-8 BOM。

用法：
    python analyze_tools/replace/export_replace_pairs_csv.py results/location_pairs.jsonl
    python analyze_tools/replace/export_replace_pairs_csv.py input.jsonl --out pairs.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

if __package__:
    from . import _pairs as pairs
else:
    import _pairs as pairs


def default_output_path(input_path: str | Path) -> Path:
    return Path(input_path).with_suffix(".replace_pairs.csv")


def export_jsonl_to_csv(
    input_path: str | Path, output_path: str | Path, *, overwrite: bool = False,
    progress_every: int = 50_000, progress_stream=None,
) -> dict[str, Any]:
    """流式导出，成功后原子发布；查重内存仅与 key 数量成正比。"""
    source, target = Path(input_path), Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    if source.suffix.lower() != ".jsonl":
        raise ValueError(f"导出输入必须是 replace JSONL：{source}")
    if pairs.same_file(source, target):
        raise ValueError(f"输出路径与输入是同一文件：{target}")
    if (target.exists() or target.is_symlink()) and not overwrite:
        raise FileExistsError(f"输出文件已存在：{target}；如需覆盖请显式加 --overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter[str] = Counter(records=0, rows_written=0)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=pairs.PAIR_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for row in pairs.iter_pair_rows(source, progress_every=progress_every, progress_stream=progress_stream):
                writer.writerow(pairs.pair_to_csv_row(row))
                pairs.update_stats(stats, row)
                stats["rows_written"] += 1
        if overwrite:
            os.replace(temporary, target)
        else:
            # link 原子拒绝同名目标，防止检查后才出现的文件被静默覆盖。
            os.link(temporary, target)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dict(stats)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", help="replace 句对 JSONL（必须显式指定，不自动选择旧结果）")
    parser.add_argument("-o", "--out", help="输出 CSV；默认 <输入名>.replace_pairs.csv")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出 CSV")
    parser.add_argument("--progress-every", type=int, default=50_000, help="每多少条句对显示进度；0 关闭")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target = Path(args.out) if args.out else default_output_path(args.input)
    try:
        stats = export_jsonl_to_csv(args.input, target, overwrite=args.overwrite,
                                    progress_every=max(0, args.progress_every))
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    print(f"已写出 {target}：{stats['rows_written']} 行句对")
    for side in ("a", "b"):
        print(f"{side.upper()} 判定：true={stats.get(side + '_true', 0)}, "
              f"false={stats.get(side + '_false', 0)}, 未判定={stats.get(side + '_null', 0)}")
    print("句对状态：" + ", ".join(f"{status}={stats.get('pair_status_' + status, 0)}"
                                    for status in pairs.PAIR_STATUSES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
