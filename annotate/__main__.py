"""python annotate [--engine staged|all-in-one] [原标注参数]。"""
import sys
from pathlib import Path


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from annotate import spatial_runner
    return spatial_runner.main(argv)


if __name__ == "__main__":
    main()
