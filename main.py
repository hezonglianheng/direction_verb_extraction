# encoding: utf8

from pathlib import Path
from typing import Any

import sys

import config
import sentenceFilters
from entry_in_linux import run_linux_entry
from entry_in_win import run_windows_entry


DEFAULT_FILTER_CONFIG = "filter_config.json5"
"""筛选配置文件默认路径"""

LEGACY_OUTPUT_NAME = "filtered_sentences.jsonl"
"""只运行一套规则、且未指定 -o 时的输出文件名（与历史行为一致）"""


def _normalize_paths(value: Any) -> list[str]:
    """把命令行参数统一成路径列表

    支持三种写法：单个字符串、逗号分隔的字符串、以及列表（argparse 的 append 动作会给出列表）。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_normalize_paths(item))
        return result
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _resolve_screen_vocab(generators: list[sentenceFilters.FilterGenerator], vocab_config_paths: list[str]) -> list[str]:
    """确定筛选前的初筛词表（子串匹配）

    优先用 -v 指定的词表；未指定时从筛选配置自动推导（见 FilterGenerator.collect_rule_screens）。
    自动推导的好处是词表只有一处真源：换了规则/词表配置文件，初筛自动跟着变，不会因为忘改 -v 而静默漏召。

    Args:
        generators (list[FilterGenerator]): 各套规则的生成器
        vocab_config_paths (list[str]): -v 指定的词表文件路径

    Returns:
        list[str]: 初筛词表；返回空列表表示不做初筛（最慢，但不会漏召）
    """
    if vocab_config_paths:
        given: set[str] = set()
        for path in vocab_config_paths:
            words = sentenceFilters.load_vocab_file(path)
            given |= words
            print(f"从词表配置文件 {path} 中加载了 {len(words)} 个词语", "（" + ", ".join(sorted(words)) + "）" if len(words) <= 20 else "")

        # 校验：某条规则的可用初筛词若与给定词表毫无交集，该规则的句子可能被初筛静默滤掉
        for generator in generators:
            for name, screen in generator.collect_rule_screens():
                if screen and not (screen & given):
                    print(
                        f"[警告] 配置 {generator.config_path} 的规则 {name} 的最小可用初筛词是 "
                        f"{''.join(sorted(screen))}，与 -v 词表无交集，该规则可能被初筛漏掉；"
                        f"建议去掉 -v 改用自动推导"
                    )
        return sorted(given)

    all_words: set[str] = set()
    unscreenable: list[tuple[str, str]] = []
    for generator in generators:
        for name, screen in generator.collect_rule_screens():
            if screen is None:
                unscreenable.append((generator.family_name, name))
            else:
                all_words |= screen

    if unscreenable:
        print("[警告] 下列规则的判定条件里没有任何词表，无法安全推导初筛词表；")
        print("       本次不使用初筛（速度较慢，但不会漏掉任何符合条件的句子）：")
        for family, name in unscreenable:
            print(f"         {family} :: {name}")
        return []

    words = sorted(sentenceFilters.substring_minimize(all_words))
    print(
        f"从筛选配置自动推导出初筛词表：{len(words)} 个词语",
        "（" + "".join(words) + "）" if len(words) <= 40 else "",
    )
    return words


def _resolve_output_paths(tgt_path: str, families: list[str], out_name: str | None) -> list[Path]:
    """按“每套规则一个输出文件”的规则确定输出路径

    - 单套规则且未传 -o：沿用历史文件名 filtered_sentences.jsonl（保证既有流程不受影响）
    - 多套规则且未传 -o：filtered_sentences_<规则文件名>.jsonl
    - 传了 -o：单套规则即该文件名；多套规则以它作为前缀，拼上规则文件名

    Args:
        tgt_path (str): 输出目录
        families (list[str]): 各套规则的名称（取配置文件名）
        out_name (str | None): -o 指定的输出文件名

    Returns:
        list[Path]: 与 families 一一对应的输出文件路径
    """
    tgt = Path(tgt_path)
    if not tgt.exists():
        raise NotADirectoryError(f"输出目录 {tgt} 不存在，请先创建（程序不会自动创建目录）")
    if not tgt.is_dir():
        raise NotADirectoryError(f"输出路径 {tgt} 是一个文件，-t/--tgt_path 需要的是目录")

    if out_name:
        name = out_name if out_name.endswith(".jsonl") else out_name + ".jsonl"
        if len(families) == 1:
            return [tgt / name]
        stem = Path(name).stem
        return [tgt / f"{stem}_{family}.jsonl" for family in families]

    if len(families) == 1:
        return [tgt / LEGACY_OUTPUT_NAME]

    return [tgt / f"filtered_sentences_{family}.jsonl" for family in families]


def main(
    src_path: str,
    tgt_path: str = ".",
    filter_config_path: str | list[str] = DEFAULT_FILTER_CONFIG,
    vocab_config_path: str | list[str] | None = None,
    out_name: str | None = None,
):
    """提取语料中包含指定语言现象的句子

    Args:
        src_path (str): 输入文件或目录（目录则取其中的 *.jsonl）
        tgt_path (str): 输出目录，必须已存在
        filter_config_path (str | list[str]): 一套或多套筛选规则配置文件
        vocab_config_path (str | list[str] | None): 初筛词表文件，None 表示从筛选配置自动推导
        out_name (str | None): 输出文件名（多套规则时作为前缀）
    """
    filter_configs = _normalize_paths(filter_config_path) or [DEFAULT_FILTER_CONFIG]
    vocab_configs = _normalize_paths(vocab_config_path)

    for path in filter_configs:
        if not Path(path).is_file():
            candidates = sorted(p.name for p in (Path(config._PROJECT_ROOT) / "config_files").glob("*.json5"))
            raise FileNotFoundError(f"找不到筛选配置文件 {path}；config_files 下有：{candidates}")

    generators = [sentenceFilters.FilterGenerator(path) for path in filter_configs]
    families = [generator.family_name for generator in generators]
    if len(set(families)) != len(families):
        raise ValueError(f"多个筛选配置的文件名（不含扩展名）重复，无法区分输出文件：{families}")

    vocabs = _resolve_screen_vocab(generators, vocab_configs)
    output_paths = _resolve_output_paths(tgt_path, families, out_name)

    print(f"[筛选计划] 共 {len(filter_configs)} 套规则：")
    for config_path, output_path in zip(filter_configs, output_paths):
        print(f"    {config_path}  ->  {output_path}")
    if vocabs:
        print(f"[筛选计划] 初筛词表 {len(vocabs)} 个词语，未通过初筛的句子不会送入 LTP")
    else:
        print("[筛选计划] 未使用初筛")

    # 输出落在输入目录里时，生成的 jsonl 会被下一轮当作输入读入（届时因缺 content 字段报错）
    src = Path(src_path)
    if src.is_dir() and any(output_path.parent == src for output_path in output_paths):
        print(f"[警告] 输出目录与输入目录相同（{src}），新生成的 jsonl 会被下一次运行的输入扫描读到")

    if sys.platform.startswith("win"):
        run_windows_entry(src_path, filter_configs, output_paths, vocabs)
    else:
        run_linux_entry(src_path, filter_configs, output_paths, vocabs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="句子过滤器，过滤包含动趋式、方位词等语言现象的句子")
    parser.add_argument("src_path", type=str, help="输入文件或目录路径")
    parser.add_argument(
        "-f", "--filter_config_path", action="append", type=str, default=None,
        help=f"筛选配置文件路径，可重复传入（也可用逗号分隔）以同时运行多套规则，默认为 {DEFAULT_FILTER_CONFIG}",
    )
    parser.add_argument(
        "-v", "--vocab_config_path", action="append", type=str, default=None,
        help="初筛词表配置文件路径，可重复传入（取并集），默认为 None（从筛选配置自动推导）",
    )
    parser.add_argument("-t", "--tgt_path", type=str, default=".", help="输出目录路径，默认为当前目录（需已存在）")
    parser.add_argument(
        "-o", "--out_name", type=str, default=None,
        help="输出文件名；只传一个 -f 时即该文件名，传多个 -f 时作为前缀（如 -o out 得到 out_规则名.jsonl）",
    )
    args = parser.parse_args()
    main(args.src_path, args.tgt_path, args.filter_config_path, args.vocab_config_path, args.out_name)
