# encoding: utf8
"""annotate 的轻量 CLI：参数、帮助与解析后校验只依赖标准库和纯元数据。

-h/--help 在这里直接由 argparse 打印并退出，不会导入 spatial_runner /
spatial_agents / spatial_graphs / spatial_tools，因此不加载 LTP、torch、
langchain（修复前本机实测 --help 约 181 秒，修复后约 1 秒）。
真正运行时才由 main() 导入 spatial_runner 并执行业务流程。
"""

import argparse

import config
from annotate.cli_args import ConsistentSelection, resolve_graph_selection
from annotate.cli_constants import (
    DEFAULT_MODEL,
    FORMAL_GRAPHS,
    GRAPH_CHOICES,
    GRAPH_DEFAULT_LIMITS,
)
from corpus_metadata import normalize_corpus_type
from filter_funcs import DEFAULT_MODE, DEFAULT_SEED, LIMIT_MODES

EPILOG = """
    python annotate --corpus results/location_pairs.jsonl --limit 5
    python annotate --corpus results/direction_pairs.jsonl --corpus-type 趋向动词 --limit 20
    python annotate --workers 4 --limit 200
    python annotate --limit 200 --limit-mode random --seed 0
    python annotate --limit 200 --limit-mode proportional --seed 0
    python annotate --reference-file results/location_annotated.jsonl
    python annotate --no-llm --batch off --limit 5
    python annotate --model model-a --out-file results/single.jsonl
    python annotate --models model-a model-b --out-file results/vote.jsonl --workers 4
    python annotate --batch native --batch-resume --batch-cache results/run.batch-cache.jsonl
    python annotate --list-graphs
    python annotate --engine all-in-one --limit 1
    python annotate --graph all_in_one --models model-a model-b
    python annotate --graph pair --print-prompt

投票示例中，results/vote.jsonl 是最终结果；各模型的原始结果是
results/vote.models/00_model-a.jsonl 和 results/vote.models/01_model-b.jsonl。
--batch auto 优先使用原生 Batch，不可用时明确回退同步；native 不允许回退。
批处理仅包含特征、图式和判定波次，查表优先级不变。
完整命令与服务器迁移注意事项见 .docs/stage-migration.md。
"""


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是大于等于 0 的整数，收到 {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"必须是大于等于 0 的整数，收到 {value!r}")
    return number


def _corpus_type_arg(value: str) -> str:
    """argparse 的 type：把类型规范化，认不出时报参数错（退出码 2）

    只用 corpus_metadata 的纯函数：spatial_tools 只是转导出它，导入 spatial_tools
    会连带加载 LTP/torch，--help 用不上。
    """
    try:
        return normalize_corpus_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_batch_args(parser: argparse.ArgumentParser) -> None:
    """给跑批入口加上批量推理相关选项（与 evaluate.add_batch_args 同名同义）

    默认 auto：模型配置声明了可用批量入口（OpenAI/OpenRouter，或显式 batch 配置）就
    走原生 Batch；没有就照旧同步执行。native 在不可用时直接报错，避免用户以为用了批量。
    """
    group = parser.add_argument_group("批量推理")
    group.add_argument(
        "--batch",
        choices=("off", "auto", "native"),
        default=config.BATCH_MODE,
        help=f"批量推理：auto（默认 {config.BATCH_MODE}，有原生 Batch 入口就用，否则同步回退）／native（必须用原生，"
        "不可用即报错）／off（完全走原来的同步链路）。原生入口按模型配置的 "
        "provider 或 batch 字段判定，见 batch_inference.batch_profile",
    )
    group.add_argument(
        "--batch-cache",
        default=None,
        help="批量结果 sidecar 路径（默认 <输出文件>.batch-cache.jsonl）；断点续跑时指向同一份",
    )
    group.add_argument(
        "--batch-resume",
        action="store_true",
        help="复用 --batch-cache 里已有的结果，并继续收同一缓存旁的 <缓存名>.jobs.json 里"
        "未终结的 batch。默认每轮重新开始，避免旧结果混进新配置的产物",
    )
    group.add_argument(
        "--batch-poll-seconds",
        type=float,
        default=config.BATCH_POLL_SECONDS,
        help=f"轮询原生 batch 状态的间隔秒数（默认 {config.BATCH_POLL_SECONDS}）",
    )
    group.add_argument(
        "--batch-timeout-seconds",
        type=float,
        default=config.BATCH_TIMEOUT_SECONDS,
        help=f"等待单个原生 batch 完成的上限秒数（默认 {config.BATCH_TIMEOUT_SECONDS}，0 = 不限）；"
        "超时不会重复提交，batch ID 已落盘，可加 --batch-resume 继续收",
    )
    group.add_argument(
        "--batch-chunk-size",
        type=int,
        default=config.BATCH_CHUNK_SIZE,
        help=f"一个原生 batch 最多包含多少条请求（默认 {config.BATCH_CHUNK_SIZE}）；超出会分片提交",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m annotate",
        description="空间语义标注跑批：按指定的图把语料跑一遍，结果写入 jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    add_batch_args(parser)
    parser.add_argument("--graph", default=None, action=ConsistentSelection, choices=GRAPH_CHOICES,
                        help="跑哪张图（默认 pair，见 --list-graphs）")
    parser.add_argument("--engine", choices=("staged", "all-in-one"), action=ConsistentSelection, default=None,
                        help="兼容旧命令：staged 默认 pair，all-in-one 映射为 --graph all_in_one；不能与相反的图组合")
    parser.add_argument("--list-graphs", action="store_true", help="打印可用的图与各自需要的 agent，然后退出")
    parser.add_argument(
        "--corpus-type",
        default=None,
        metavar="类型",
        type=_corpus_type_arg,
        help="语料类型：方位词（别名 localization/location/loc）或 趋向动词（别名 direction/dir/direction_verb）。"
        "决定读取该词性的特征、图式与判定配置，以及 config_files 对应词性目录下的示例。"
        "默认方位词；**刻意不按语料文件名推断**——推断猜错时是静默读错一整组表，"
        "宁可要你显式说一句（语料名像另一类时会提醒一句，但不会替你改）。"
        "正式标注前请先运行 python replace 生成该类型的句对",
    )
    parser.add_argument(
        "--corpus", default=None,
        help="句对 JSONL（pair/synonym/all_in_one 默认 results/location_pairs.jsonl 或 direction_pairs.jsonl）；"
        "single/alter_check 诊断图读原句 JSONL，字段要求不同",
    )
    parser.add_argument(
        "--alter-rules",
        default=None,
        help="仅 alter_check 诊断图使用；正式 pair/synonym 标注不替换，须将此参数传给 python replace",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="最终结果目录，两种模式都默认使用项目根目录下的 results/<图前缀>/。"
        "未给 --out-file 时，文件名为 <图前缀>_<时间戳>.jsonl",
    )
    parser.add_argument(
        "--out-file", default=None,
        help="直接指定最终 JSONL 路径，优先于 --out-dir。单模型时写该模型的标注，投票时写汇总结果",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出文件。默认**拒绝**：已有同名输出直接报错退出（E1）——"
        "旧行为是不拒绝直接追加，于是历史记录、重复 id 与不同配置的结果会混进同一份文件，"
        "收尾重排后与只统计本轮的 summary 对不上。覆盖前会先清空该文件，不会混入历史记录",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help=f"每个 agent **每条候选**的请求预算（默认 {config.CANDIDATE_MAX_REQUESTS}，取自"
        "config.CANDIDATE_MAX_REQUESTS；0 或负表示不限）。一条语料会展开出多个候选，同一位置的"
        "失败若由每个候选各自重试，重试次数会被候选数放大成整批的倍数；超预算即停，"
        "如实记为失败并转入人工复核（A3）",
    )
    parser.add_argument(
        "--batch-max-requests",
        type=int,
        default=None,
        help=f"整批共用的请求预算（默认 {config.BATCH_MAX_REQUESTS}，取自 config.BATCH_MAX_REQUESTS；"
        "0 或负表示不限）。所有 agent 共用一份计数，达到上限后余下的调用一律不再发出，"
        "记录照常落盘并转入人工复核（A3）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理多少条两侧已通过检查的句对（过滤后计数）；正式图必须 >0，"
        "可用数量不足时会报告。未指定时按图默认：pair/synonym=100，all_in_one=1；"
        "aligned 模式可省略（表示 reference 中全部句对）。"
        "诊断图 single/alter_check 沿用旧行为：limit<=0 返回空列表",
    )
    parser.add_argument(
        "--limit-mode",
        choices=LIMIT_MODES,
        default=None,
        help="--limit 的抽取方式（仅正式图 pair/synonym/all_in_one）：sequential=按文件顺序取前 X 条（默认）；"
        "random=从全部合格句对中随机抽 X 条；"
        "proportional=按无向关注词对（上-下 与 下-上 同层）的比例分配 X 条；"
        "aligned=按 --reference-file 的 annotate 结果逐条对回当前句对。"
        "后三种与 replace 的同名模式口径一致；显式给出本参数时必须同时给 --limit"
        "（aligned 由 --reference-file 自动启用，不需要 --limit）",
    )
    parser.add_argument(
        "--seed",
        type=_non_negative_int,
        default=None,
        help="random/proportional 的随机种子（默认 0）；同一份语料与种子可复现同一批句对。"
        "仅与 --limit 一起使用；aligned 不使用随机种子。仅正式图支持",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        help="annotate 结果 JSONL（仅正式图）：读取其中所有句对，按 candidate_key 逐条对回 --corpus。"
        "给出后自动进入 aligned 模式；--limit 可省略（全部对齐），给了则只取源文件顺序的前 X 条。"
        "reference 中任何一条句对找不到或一对多都会报错；不能与 --seed 或"
        "非 aligned 的 --limit-mode 同时使用",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="跳过前多少条可用判定语料（过滤后计数，默认 0）；"
        "三种抽样模式下先跳过再抽样，aligned 按匹配到的源记录顺序跳过",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", default=DEFAULT_MODEL, metavar="配置名",
        help=f"单模型标注：使用 model_config/api_keys.json 中的一个配置（默认 {DEFAULT_MODEL}）。"
        "该模型的标注直接写入最终 JSONL，不投票，也不创建 .models/ 子目录。与 --models 互斥",
    )
    model_group.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="配置名",
        help="只给一个配置名时，--models A 等价于 --model A。给两个及以上不同配置名时，"
        "各模型对同一批输入独立跑完所选图，再按 voting_weight 逐条加权投票。"
        "最终 JSONL 写投票结果；各模型的原始 JSONL 按参数顺序另存于同目录的 <最终文件主名>.models/。"
        "平票或无有效票记为 discarded，缺票转人工复核；个别模型失败前已写出的记录仍参与投票，"
        "全部模型运行失败才中止。投票不能与 --no-llm 同用（--print-prompt 预览除外）",
    )
    parser.add_argument(
        "--model-workers",
        type=int,
        default=config.VOTING_MODEL_WORKERS,
        help=f"多模型投票时同时跑几个模型（默认 {config.VOTING_MODEL_WORKERS}，0 = 全部并行）。"
        "每个模型内部仍按 --workers 并发处理记录，总线程数约为两者相乘",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="并发处理的记录数（默认 %(default)s，LLM 调用是 IO 密集）")
    parser.add_argument(
        "--batch-inference",
        action="store_true",
        help="仅旧诊断图的句法准备使用；正式 pair/synonym 标注无句法输入，此开关无效",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调任何模型：只读取已有句对、查表与落盘；不重新替换或筛选",
    )
    parser.add_argument(
        "--feature-deep",
        action="store_true",
        help="feature_agent 用 deep agent（每类特征一个 subagent、主 agent 汇总）；"
        "默认是单次调用直接产出全部特征字段，一次省下约一个数量级的模型往返。"
        "deep 只在需要对照或单次调用抽得不好时开；不适用于 all_in_one（它会直接报错）",
    )
    parser.add_argument("--print-prompt", action="store_true",
                        help="离线预览正式图的实际提示语及输入，不读模型密钥、不预热、不建日志或输出；"
                        "pair 下游说明依赖；仅正式图支持")
    repair = parser.add_mutually_exclusive_group()
    repair.add_argument("--allow-retry", dest="allow_retry", action="store_true",
                        help="显式允许结构化输出错误补发（不改变传输重试配置）")
    repair.add_argument("--no-allow-retry", dest="allow_retry", action="store_false",
                        help="显式禁止结构化补发；未指定时 pair/synonym 允许，all_in_one 禁止")
    parser.set_defaults(allow_retry=None)
    parser.add_argument("--all-candidates", action="store_true",
                        help="兼容旧 all-in-one 参数：现为一行一个合格句对，不再展开；指定时会提示")
    parser.add_argument("--no-syntax", action="store_true",
                        help="旧兼容参数：正式标注不准备句法，此开关不改变输入或执行"
                        "（保留只为旧命令不报错）")
    parser.add_argument(
        "--no-pairs-csv",
        action="store_true",
        help="不把成对信息（原句/替换句/图式/同义判断）另导一份 csv。"
        "默认**导出**到与结果 jsonl 同名、同目录的 .csv，行序与 jsonl 一致（收尾排序后同一份记录写就，"
        "见 ResultWriter.finish），供 Excel 查看；列序见 spatial_tools.PAIR_CSV_COLUMNS。"
        "正式 pair/synonym/all_in_one 图均可导出；single/alter_check 诊断图不导出",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.graph = resolve_graph_selection(args.graph, args.engine)
    except ValueError as exc:
        parser.error(str(exc))
    if args.graph in FORMAL_GRAPHS:
        # 正式图沿用“X>0”的新口径；诊断图保留旧的 limit<=0 返回空列表兼容行为。
        if args.limit is not None and args.limit <= 0:
            parser.error("--limit 必须是大于 0 的整数")
        if args.offset < 0:
            parser.error("--offset 必须是大于等于 0 的整数")
    if args.graph not in FORMAL_GRAPHS and (
        args.reference_file is not None or args.limit_mode is not None or args.seed is not None
    ):
        parser.error("--limit-mode / --seed / --reference-file 只支持正式图 pair/synonym/all_in_one")
    if args.reference_file is not None:
        if args.limit_mode not in (None, "aligned"):
            parser.error("--reference-file 不能与 --limit-mode sequential/random/proportional 同时使用")
        if args.seed is not None:
            parser.error("--reference-file 不对齐随机过程，不能与 --seed 同时使用")
        args.limit_mode = "aligned"
        args.seed = DEFAULT_SEED
    elif args.limit_mode == "aligned":
        parser.error("--limit-mode aligned 必须同时给出 --reference-file")
    elif args.limit is None:
        if args.limit_mode is not None or args.seed is not None:
            parser.error("--limit-mode 与 --seed 只能在给出 --limit 时使用；aligned 请改用 --reference-file")
        args.limit_mode = DEFAULT_MODE
        args.seed = DEFAULT_SEED
    else:
        if args.limit_mode is None:
            args.limit_mode = DEFAULT_MODE
        if args.seed is None:
            args.seed = DEFAULT_SEED
    # aligned 未显式给 X 时保留 None（用 reference 中全部句对），不能被图的默认 limit 截断。
    if args.limit is None and args.limit_mode != "aligned":
        args.limit = GRAPH_DEFAULT_LIMITS[args.graph]
    if args.graph in FORMAL_GRAPHS and args.alter_rules:
        parser.error("正式标注不再替换句子；请将 --alter-rules 传给 python replace")
    if args.graph == "all_in_one" and args.feature_deep:
        parser.error("--feature-deep 不适用于 all_in_one：它必须一次完成完整标注；请使用 --graph pair")
    if args.print_prompt and args.graph not in FORMAL_GRAPHS:
        parser.error("--print-prompt 只支持正式 pair/synonym/all_in_one 图；诊断图的动态执行不能离线预览")
    if args.graph not in FORMAL_GRAPHS and (args.all_candidates or args.no_syntax):
        parser.error("--all-candidates / --no-syntax 仅为正式标注的旧命令兼容参数，不适用于诊断图")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # 解析成功后才导入业务模块：--help / 参数错误不会加载 LTP、torch、langchain。
    from annotate import spatial_runner

    return spatial_runner.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
