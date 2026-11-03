"""python evaluate [评测参数]。"""
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("evaluate.runner", run_name="__main__")
