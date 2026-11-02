"""python extract [原 sentence_extract.py 参数]。"""
import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("extract.sentence_extract", run_name="__main__")
