"""python replace [原 sentence_replace.py 参数]。"""
import sys
from pathlib import Path


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from replace import cli
    raise SystemExit(cli.main())
