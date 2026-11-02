"""python annotate [--engine staged|all-in-one] [原标注参数]。"""
import argparse
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--engine", choices=("staged", "all-in-one"), default="staged")
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    target = "all_in_one" if args.engine == "all-in-one" else "spatial_runner"
    runpy.run_module("annotate." + target, run_name="__main__")
