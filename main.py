# encoding: utf8

import sys

from entry_in_linux import run_linux_entry
from entry_in_win import run_windows_entry


def main(src_path: str, tgt_path: str = ".", filter_config_path: str = "filter_config.json5"):
    if sys.platform.startswith("win"):
        run_windows_entry(src_path, tgt_path, filter_config_path)
    else:
        run_linux_entry(src_path, tgt_path, filter_config_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="句子过滤器，过滤包含动趋式的句子")
    parser.add_argument("src_path", type=str, help="输入文件或目录路径")
    parser.add_argument("-f", "--filter_config_path", type=str, default="filter_config.json5", help="筛选配置文件路径，默认为 filter_config.json5")
    parser.add_argument("-t", "--tgt_path", type=str, default=".", help="输出目录路径，默认为当前目录")
    args = parser.parse_args()
    main(args.src_path, args.tgt_path, args.filter_config_path)