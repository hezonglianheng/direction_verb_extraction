"""python extract [原 sentence_extract.py 参数]。"""
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from extract import cli
    raise SystemExit(cli.main())
