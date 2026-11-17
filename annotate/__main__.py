"""python annotate [--engine staged|all-in-one] [原标注参数]。"""
import sys
from pathlib import Path


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # 先走轻量 CLI：--help 不会导入 spatial_runner 及其 LTP/torch/langchain 依赖。
    from annotate import cli
    return cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
