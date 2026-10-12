# encoding: utf8

"""把结果 jsonl 里的成对信息拍平成 csv，供 Excel 查看（只读，不改任何输出）。

拍平与写盘的**真源在 spatial_tools.write_pairs_csv**，跑批收尾（spatial_runner.ResultWriter.finish）
用的是同一份，所以本脚本只负责「读哪个文件、写到哪」这层命令行外壳，列序与单元格口径不由这里定
（列序见 spatial_tools.PAIR_CSV_COLUMNS，改列只改那里）。

结果 jsonl（spatial_runner.py / all_in_one.py 的输出，记录形状见 spatial_graphs.serialize_pair）
里每条记录是一对「原句 vs 替换句」，关键信息散在几个嵌套块里：

    pair.a/b.sentence          原句、替换句
    focus_word / focus_word_b  关键词 a、关键词 b
    schema.a/b.name            a、b 的图式
    judgement.*                是否同义、判断理由、判断来源、查表结果

本脚本按固定列序把这些字段拍平成一行，行序保持 jsonl 原序（写盘前已按
(line_no, candidate.index) 排过，键见 spatial_graphs.record_order_key）。

只有走到 semantic_ok 的记录才会有图式与同义判断（judgement 非空），其余记录
（not_spatial、semantic_failed_* 等）这两块是空的，对应格子留空；「阶段」列保留原始
stage 值，用来区分有没有判断——semantic_ok 即有判断，其余为空判断。

跑批（pair 图）默认就会在结果 jsonl 旁导一份同名 .csv，用不到本脚本；这里是给
「事后对某一份已经存在的 jsonl 补导」用的——比如 all_in_one.py 的输出，或 --no-pairs-csv
跑出来、后来又想要 csv 的那些结果。

用法：
    .venv/bin/python export_pairs_csv.py results/loc_test/spatial_pairs_20261003_001802.jsonl
    .venv/bin/python export_pairs_csv.py <结果.jsonl> -o <输出.csv>
"""

import argparse
from pathlib import Path

import spatial_tools as st
from file_paths import validate_output_paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把结果 jsonl 的成对信息（原句/替换句/图式/同义判断）导出为 csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument("input", help="结果 jsonl（spatial_runner.py / all_in_one.py 的输出）")
    parser.add_argument("-o", "--out", help="输出 csv 路径（默认与输入同目录、同名换 .csv 后缀）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"找不到输入文件：{in_path}")
    out_path = Path(args.out) if args.out else in_path.with_suffix(".csv")
    try:
        validate_output_paths([in_path], [out_path])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    records = st.read_jsonl(in_path)
    total, judged = st.write_pairs_csv(records, out_path)
    print(f"已写出 {out_path}：共 {total} 行，其中 {judged} 行有同义判断，{total - judged} 行留空")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
