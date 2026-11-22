# encoding: utf8
"""replace 的轻量 CLI：只导入标准库、配置常量与共享元数据，--help 不加载模型依赖。

参数解析成功后才会导入 replace.sentence_replace 并执行业务流程。
"""

import argparse

import config
from corpus_metadata import (
    CORPUS_TYPE_ALIASES,
    CORPUS_TYPE_DIRS,
    DEFAULT_CORPUS_TYPE,
    PAIR_CORPORA,
    normalize_corpus_type,
)
from filter_funcs import DEFAULT_MODE, DEFAULT_SEED, LIMIT_MODES, REFERENCE_LEVELS

DESCRIPTION = (
    "语料替换阶段：原句及三元组 → 机械替换 → 检查替换句 → 一行一个句对。\n\n"
    "旧语料缺少完整原句判定时先补筛；标注阶段不再负责替换或语法/空间义检查。"
)

EPILOG = """用法：
    python -m replace --corpus results/location_corpus.jsonl --corpus-type 方位词 --out-file results/location_pairs.jsonl

限额抽样（X 为送入替换的可用原句数）：
    python -m replace --corpus ... --limit 500                          # 按顺序取前 500 条
    python -m replace --corpus ... --limit 500 --limit-mode random --seed 0
    python -m replace --corpus ... --limit 500 --limit-mode proportional --seed 0
    python -m replace --corpus ... --reference-file filtered.jsonl      # 按已筛选文件逐条对回
random/proportional 只额外读一遍语料做计数，不把 GB 级语料读进内存；同一 seed 可复现。
aligned（--reference-file）按行号/句子内容严格对回同一份源语料；找不到或源语料状态不可用
（false/error/limit 等）的 reference 行会跳过，并在结束时由控制台和日志汇总报告；一对多歧义仍报错。
record 级跳过整条不可用记录；--reference-level verdict 时按单个目标词位置采样，只跳过不可用的
目标 verdict，同一记录里其他可用 verdict 仍会保留，同一句的多个 verdict 会展开成多条记录。"""


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是大于 0 的整数，收到 {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"必须是大于 0 的整数，收到 {value!r}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是大于等于 0 的整数，收到 {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"必须是大于等于 0 的整数，收到 {value!r}")
    return number


def _corpus_type(value: str) -> str:
    """把 normalize_corpus_type 的 ValueError 转成 argparse 可读的错误提示。"""
    try:
        return normalize_corpus_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _corpus_type_help() -> str:
    groups = []
    for kind in CORPUS_TYPE_DIRS:
        aliases = sorted(alias for alias, target in CORPUS_TYPE_ALIASES.items() if target == kind)
        suffix = f"（默认；别名 {'/'.join(aliases)}）" if kind == DEFAULT_CORPUS_TYPE else f"（别名 {'/'.join(aliases)}）"
        groups.append(f"{kind}{suffix}")
    return "；".join(groups)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m replace",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("--corpus", required=True, help="提取阶段的原句 JSONL（不能已经是句对）")
    parser.add_argument(
        "--corpus-type", default=DEFAULT_CORPUS_TYPE, metavar="类型", type=_corpus_type,
        help=f"语料类型：{_corpus_type_help()}；英文别名不区分大小写。"
             f"决定读取 config_files/ 下哪组替换与检查规则，以及默认输出文件名",
    )
    parser.add_argument(
        "--out-file", default=None,
        help=f"句对 JSONL；默认项目根目录 results/ 下的 {PAIR_CORPORA['方位词']}（方位词）"
             f"或 {PAIR_CORPORA['趋向动词']}（趋向动词）",
    )
    parser.add_argument(
        "--alter-rules", default=None,
        help="覆盖本语料类型的机械替换规则表；省略时读取 "
             "config_files/<localization|direction>/alter_rules.json5",
    )
    parser.add_argument(
        "--model", default=config.SCREEN_MODEL,
        help=f"语法、空间义检查的模型配置名（model_config/api_keys.json 里的键），默认 {config.SCREEN_MODEL}",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"并发检查数；不传时用模型配置 screen_workers，再回退到全局默认 {config.SCREEN_WORKERS}",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"每次 LLM 请求放入多少条判定项；不传时用模型配置 screen_batch_size，"
        f"再回退到全局默认 {config.SCREEN_BATCH_SIZE}",
    )
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="每批读入多少条原句（默认 %(default)s）")
    parser.add_argument(
        "--limit", type=_positive_int, default=None,
        help="只把最多 X 条可用原句送入替换（X>0）；不传则处理全部语料。"
             "可用原句指 true/no_llm 且句子形状合法的记录，坏记录与不可判定记录不占名额。"
             "aligned + --reference-level verdict 时，X 表示最多展开多少个 verdict",
    )
    parser.add_argument(
        "--limit-mode", choices=LIMIT_MODES, default=None,
        help="--limit 的抽取方式：sequential=按文件顺序取前 X 条（默认）；"
             "random=从全部可用原句中随机抽 X 条；"
             "proportional=在带本类关注词的可用原句中按主关注词比例分配 X 条"
             "（配额描述选中记录的分层，句内其他位置仍会展开）；"
             "aligned=按 --reference-file 逐条对回源语料。"
             "除 aligned（由 --reference-file 自动启用）外，必须同时给出 --limit",
    )
    parser.add_argument(
        "--seed", type=_non_negative_int, default=None,
        help="random/proportional 的随机种子，默认 0；同一份语料与种子可复现同一批记录。"
             "仅与 --limit 一起使用；aligned 不使用随机种子",
    )
    parser.add_argument(
        "--reference-file", default=None,
        help="已筛选文件（JSONL）：按行号/句子内容把它逐条对回 --corpus。"
             "给出后自动进入 aligned 模式；--limit 可省略（全部对齐），给了则只取源文件顺序的前 X 条。"
             "reference 中找不到或源语料状态不可用的行会跳过，并在结束时汇总；"
             "同一个 verdict 一对多、一条源记录对多条 reference 等歧义仍报错。"
             "verdict 级同一句的多个目标位置可以并存。不能与 --seed 或"
             "非 aligned 的 --limit-mode 同时使用",
    )
    parser.add_argument(
        "--reference-level", choices=REFERENCE_LEVELS, default="record",
        help="aligned reference 的采样单位：record=整条源记录（默认，保持旧行为）；"
             "verdict=按单个目标词位置采样，并把源记录裁剪成只含该 verdict 的副本。"
             "只在给出 --reference-file 时有效",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="替换日志目录；默认项目根目录下的 log/（不存在会自动创建）",
    )
    parser.add_argument("--no-llm", action="store_true",
                        help="不调用模型（默认会调用模型做语法、空间义检查）；未检查的句对不可进入标注")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="允许覆盖已存在的输出句对文件（默认拒绝；不会覆盖输入文件或参考文件）",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.reference_file is not None:
        if args.limit_mode not in (None, "aligned"):
            parser.error("--reference-file 不能与 --limit-mode sequential/random/proportional 同时使用")
        if args.seed is not None:
            parser.error("--reference-file 不对齐随机过程，不能与 --seed 同时使用")
        args.limit_mode = "aligned"
        args.seed = DEFAULT_SEED
    elif args.limit_mode == "aligned":
        parser.error("--limit-mode aligned 必须同时给出 --reference-file")
    elif args.reference_level != "record":
        parser.error("--reference-level 只能在给出 --reference-file 时使用")
    elif args.limit is None:
        if args.limit_mode is not None or args.seed is not None:
            parser.error("--limit-mode 与 --seed 只能在给出 --limit 时使用")
        args.limit_mode = DEFAULT_MODE
        args.seed = DEFAULT_SEED
    else:
        if args.limit_mode is None:
            args.limit_mode = DEFAULT_MODE
        if args.seed is None:
            args.seed = DEFAULT_SEED
    return args


def main(argv=None):
    args = parse_args(argv)
    # 解析成功后才导入业务模块：--help / 参数错误不会加载 LTP、torch、langchain。
    from replace import sentence_replace
    return sentence_replace.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
