# encoding: utf8

"""按「是否同义」分成是、否、留空三类，对成对语料 CSV 随机抽样。

兼容 export_pairs_csv.py 和 spatial_runner.py 导出的 CSV，仅使用 Python 标准库。
采样单位为 CSV 的一条成对记录，不按原句去重；同一条记录不会重复抽取。
保留所有原始列和单元格，默认按输入顺序输出，使用 --shuffle 可打乱输出顺序。
输入默认使用 utf-8-sig（也能读取无 BOM 的 UTF-8），输出为 Excel 可读的 utf-8-sig。

两种采样方式互斥，数量/比例的顺序始终是：是、否、留空。
比例是样本中各类的相对权重：5 3 2、50 30 20、0.5 0.3 0.2 等价。
比例换算数量时先向下取整，再按小数余数由大到小补齐；余数相同时按上述类别顺序。
某类不足时默认报错且不写出；--allow-shortage 取该类全部，不向其他类别补足。
固定输入、参数和随机种子可复现。默认输出 <输入文件名>.sampled.csv。
同名采样结果会覆盖；输出不能指向输入文件。

用法：
    python sample_pairs_csv.py results/pairs.csv --counts 300 200 100
    python sample_pairs_csv.py results/pairs.csv --total 1000 --ratios 5 3 2
    python sample_pairs_csv.py results/pairs.csv --counts 100 0 0 -o results/yes.csv
    python sample_pairs_csv.py results/pairs.csv --total 200 --ratios 1 1 0 --seed 42 --shuffle
    python sample_pairs_csv.py results/pairs.csv --counts 100 100 50 --allow-shortage
    python sample_pairs_csv.py results/pairs.csv --total 1000 --ratios 5 3 2 --dry-run
"""

import argparse
import csv
import random
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


LABELS = ("是", "否", "留空")
DEFAULT_SEED = 0


def nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("数量必须为非负整数") from None
    if number < 0:
        raise argparse.ArgumentTypeError("数量不能为负数")
    return number


def nonnegative_ratio(value: str) -> Fraction:
    try:
        number = Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError("比例必须为有限的非负数字") from None
    if number < 0:
        raise argparse.ArgumentTypeError("比例不能为负数")
    return number


def allocate_counts(total: int, ratios: list[Fraction]) -> dict[str, int]:
    """用最大余数法把相对权重分配为整数数量，保证总数恰好等于 total。"""
    weight = sum(ratios)
    if weight <= 0:
        raise ValueError("三类比例不能全为 0")
    quotas = [total * ratio / weight for ratio in ratios]
    counts = [quota.numerator // quota.denominator for quota in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return dict(zip(LABELS, counts))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按是否同义（是、否、留空）分层随机抽样成对语料 CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument("input", help="export_pairs_csv.py / spatial_runner.py 导出的 CSV")
    parser.add_argument("-o", "--out", help="输出 CSV 路径，默认 <输入文件名>.sampled.csv")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--counts", nargs=3, type=nonnegative_int, metavar=("YES", "NO", "BLANK"),
        help="分别指定是、否、留空的采样条数；0 表示排除该类",
    )
    mode.add_argument("--total", type=nonnegative_int, help="按比例采样时的总条数")
    parser.add_argument(
        "--ratios", nargs=3, type=nonnegative_ratio, metavar=("YES", "NO", "BLANK"),
        help="样本中是、否、留空的相对比例，须配合 --total；默认 1 1 1",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"随机种子，默认 {DEFAULT_SEED}")
    parser.add_argument("--column", default="是否同义", help="分类列的名称，默认 是否同义")
    parser.add_argument("--encoding", default="utf-8-sig", help="输入 CSV 编码，默认 utf-8-sig；可指定 gb18030 等")
    parser.add_argument("--shuffle", action="store_true", help="打乱输出行顺序；默认保留输入顺序")
    parser.add_argument(
        "--allow-shortage", action="store_true",
        help="某类不足时取该类全部，输出总量可能减少、比例可能变化",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅统计和预览采样数量，不写出文件")
    args = parser.parse_args(argv)
    if args.counts is not None:
        if args.ratios is not None:
            parser.error("--ratios 只能配合 --total，不能与 --counts 同时使用")
        args.targets = dict(zip(LABELS, args.counts))
    else:
        args.ratios = args.ratios if args.ratios is not None else [Fraction(1)] * 3
        try:
            args.targets = allocate_counts(args.total, args.ratios)
        except ValueError as exc:
            parser.error(str(exc))
    if not args.column.strip():
        parser.error("--column 不能为空")
    return args


def sample_csv(
    input_path: Path,
    targets: dict[str, int],
    seed: int,
    *,
    column: str = "是否同义",
    encoding: str = "utf-8-sig",
) -> tuple[list[str], list[tuple[int, list[str]]], dict[str, int]]:
    """单次扫描并按类做蓄水池抽样，内存只保留被选中的记录。"""
    counts = dict.fromkeys(LABELS, 0)
    pools: dict[str, list[tuple[int, list[str]]]] = {label: [] for label in LABELS}
    generators = {label: random.Random(f"{seed}:{label}") for label in LABELS}
    with input_path.open("r", encoding=encoding, newline="") as source:
        reader = csv.reader(source, strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError(f"输入 CSV 为空或没有表头：{input_path}")
        matches = [i for i, name in enumerate(header) if name.strip() == column.strip()]
        if not matches:
            raise ValueError(f"输入 CSV 缺少列 {column!r}；现有表头：{'、'.join(header)}")
        if len(matches) > 1:
            raise ValueError(f"输入 CSV 存在多个 {column!r} 列，无法确定分类列")
        column_index = matches[0]
        for record_no, row in enumerate(reader, start=1):
            if not row:  # 跳过真正的空行；有单元格的记录即使全为空，也归入「留空」。
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"CSV 第 {reader.line_num} 行结束的记录有 {len(row)} 列，表头有 {len(header)} 列"
                )
            value = row[column_index].strip()
            if value not in ("是", "否", ""):
                raise ValueError(
                    f"CSV 第 {reader.line_num} 行结束的记录，{column!r} 为 {value!r}，"
                    "只支持「是」「否」或空白"
                )
            label = value if value else "留空"
            counts[label] += 1
            k = targets[label]
            if k == 0:
                continue
            pool = pools[label]
            if len(pool) < k:
                pool.append((record_no, row))
            else:
                index = generators[label].randrange(counts[label])
                if index < k:
                    pool[index] = (record_no, row)
    sampled = [item for label in LABELS for item in pools[label]]
    sampled.sort(key=lambda item: item[0])
    return header, sampled, counts


def write_csv(path: Path, header: list[str], sampled: list[tuple[int, list[str]]]) -> None:
    """先完整写入同目录临时文件，再替换目标，避免写入中断留下半份 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(row for _, row in sampled)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.out) if args.out else input_path.with_name(f"{input_path.stem}.sampled.csv")
    try:
        if not input_path.is_file():
            raise ValueError(f"找不到输入 CSV 文件：{input_path}")
        if output_path.resolve() == input_path.resolve() or (
            output_path.exists() and output_path.samefile(input_path)
        ):
            raise ValueError("输出路径不能与输入文件相同，请通过 -o 指定其他路径")
        # Windows 上 C long 的位数可能小于 sys.maxsize，逐步下调至 csv 模块接受的值。
        limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(limit)
                break
            except OverflowError:
                limit //= 10
        header, sampled, available = sample_csv(
            input_path, args.targets, args.seed, column=args.column, encoding=args.encoding,
        )
        print(f"输入 CSV：{input_path}（共 {sum(available.values())} 条记录）")
        print(f"随机种子：{args.seed}；类别顺序：是、否、留空")
        shortages = []
        for label in LABELS:
            target = args.targets[label]
            selected = min(available[label], target)
            print(f"  {label}：可用 {available[label]} 条，目标 {target} 条，可抽取 {selected} 条")
            if available[label] < target:
                shortages.append(f"{label}需 {target} 条，但只有 {available[label]} 条")
        if shortages:
            message = "；".join(shortages)
            if not args.allow_shortage:
                raise ValueError(f"语料不足：{message}。请降低采样数量，或使用 --allow-shortage")
            print(f"语料不足，按 --allow-shortage 取可用记录：{message}；总量和比例可能变化")
        total = len(sampled)
        if total:
            proportions = "，".join(
                f"{label} {min(available[label], args.targets[label]) / total:.2%}" for label in LABELS
            )
            print(f"实际采样：{total} 条；{proportions}")
        else:
            print("实际采样：0 条（输出仅含表头）")
        if args.dry_run:
            print("--dry-run：仅预览，不写出文件")
            return 0
        if args.shuffle:
            random.Random(args.seed).shuffle(sampled)
        if output_path.exists():
            print(f"覆盖已有采样文件：{output_path}")
        write_csv(output_path, header, sampled)
        print(f"已写出：{output_path}（{total} 条记录，保留全部 {len(header)} 列）")
        return 0
    except (OSError, UnicodeError, ValueError, LookupError, csv.Error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
