# encoding: utf8
"""extract 的轻量 CLI：只导入标准库与配置常量，--help 不加载 LTP/torch。

参数解析成功后才会在这里导入 extract.sentence_extract 并执行业务流程。
"""

import argparse

import config
from corpus_metadata import SCREEN_PROMPT_FILES
from extract.cli_constants import DEFAULT_FILTER_CONFIG, LEGACY_OUTPUT_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m extract",
        description="句子过滤器：从原句 JSONL 中筛出包含动趋式、方位词等语言现象的句子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m extract input.jsonl -f config_files/direction/direction_expressions.json5 --no-llm-screen\n"
            "  python -m extract input_dir -f config_files/localization/localization_expressions.json5 "
            "-f config_files/direction/direction_expressions.json5 -o multi -t results\n"
        ),
    )
    parser.add_argument("src_path", type=str, help="输入文件或目录路径；传目录时读取其中的 *.jsonl")
    parser.add_argument(
        "-f", "--filter_config_path", action="append", type=str, default=None,
        help="筛选配置文件路径，可重复传入（也可用逗号分隔）以同时运行多套规则；"
        "必须显式指定（两套规则的研究口径不同，不默认替你选）："
        "config_files/direction/direction_expressions.json5（趋向动词）或 "
        "config_files/localization/localization_expressions.json5（方位词）",
    )
    parser.add_argument(
        "-v", "--vocab_config_path", action="append", type=str, default=None,
        help="初筛词表配置文件路径，可重复传入（也可用逗号分隔，取并集）；"
        "省略时从筛选配置自动推导",
    )
    parser.add_argument(
        "-t", "--tgt_path", type=str, default=".",
        help="输出目录路径，默认为当前目录（需已存在，程序不会自动创建）",
    )
    parser.add_argument(
        "-o", "--out_name", type=str, default=None,
        help=f"输出文件名；只有一套规则时即该文件名，多套规则时作为前缀（如 -o out 得到 out_规则名.jsonl）；"
        f"文件名会自动补 .jsonl。省略时单规则写 {LEGACY_OUTPUT_NAME}，"
        f"多规则写 filtered_sentences_<规则文件名>.jsonl",
    )
    parser.add_argument(
        "--no-llm-screen",
        action="store_true",
        help="关闭规则命中后的 LLM 筛选。默认开启，但只有 system_prompt/extract/screen_localization.txt "
        "或 screen_direction.txt 填了正文时才真正调模型；提示语为空时自动跳过、记录原样输出",
    )
    parser.add_argument(
        "--screen-model",
        default=config.SCREEN_MODEL,
        help=f"LLM 筛选使用的模型配置名（model_config/api_keys.json 里的键），默认 {config.SCREEN_MODEL}",
    )
    parser.add_argument(
        "--screen-workers",
        type=int,
        default=None,
        help=f"LLM 筛选的并发线程数；不传时用模型配置 screen_workers，再回退到全局默认 "
        f"{config.SCREEN_WORKERS}（逐词判定是 IO 密集的）",
    )
    parser.add_argument(
        "--screen-batch-size",
        type=int,
        default=None,
        help=f"每次 LLM 请求放入多少条判定项；不传时用模型配置 screen_batch_size，"
        f"再回退到全局默认 {config.SCREEN_BATCH_SIZE}",
    )
    parser.add_argument(
        "--screen-kind",
        choices=sorted(SCREEN_PROMPT_FILES),
        default=None,
        help="强制指定筛选提示语的词类型。默认按规则配置文件名推断（localization→方位词，"
        "direction→趋向动词）；只能用于单套规则",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=config.EXTRACT_WORKERS,
        help=f"文本切句的并发线程数，默认 {config.EXTRACT_WORKERS}（0 = 自动）",
    )
    parser.add_argument(
        "--filter-workers",
        type=int,
        default=config.FILTER_WORKERS,
        help=f"规则筛选的进程数，默认 {config.FILTER_WORKERS}（0 = 自动按 CPU 核数，1 = 串行）。"
        "多进程会各持一份 LTP 模型，单卡上通常 1 更快；需要时再调大",
    )
    parser.add_argument(
        "--sentence-dedup-window",
        type=int,
        default=config.SENTENCE_DEDUP_WINDOW,
        help=f"跨文件句子去重窗口：只记住最近多少个词表初筛后的唯一句哈希，0=关闭。"
        f"默认 {config.SENTENCE_DEDUP_WINDOW}，约 20–30MiB；窗口越大去重越彻底、内存越高",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # 解析成功后才导入业务模块：--help / 参数错误不会加载 LTP、torch 等重依赖。
    from extract import sentence_extract
    return sentence_extract.main(
        args.src_path,
        args.tgt_path,
        args.filter_config_path,
        args.vocab_config_path,
        args.out_name,
        screen_llm=not args.no_llm_screen,
        screen_model=args.screen_model,
        screen_workers=args.screen_workers,
        screen_batch_size=args.screen_batch_size,
        screen_kind=args.screen_kind,
        extract_workers=args.extract_workers,
        filter_workers=args.filter_workers,
        sentence_dedup_window=args.sentence_dedup_window,
    )


if __name__ == "__main__":
    raise SystemExit(main())
