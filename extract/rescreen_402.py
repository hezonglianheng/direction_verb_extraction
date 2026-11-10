# encoding: utf8

"""已弃用的旧入口：保留 ``python extract/rescreen_402.py`` 兼容调用。

实现已迁移到通用的 ``extract/rescreen.py``；本文件只负责转发。未显式指定
``--error-type`` 时，新入口默认仍然只处理 402 / Insufficient Balance / 余额不足，
因此旧命令的行为不变。
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract import rescreen


def main(argv: list[str] | None = None) -> int:
    print(
        "[补判] 注意：extract/rescreen_402.py 已更名为 extract/rescreen.py；"
        "本入口仅作兼容转发，默认仍只处理 402 余额不足。",
        flush=True,
    )
    return rescreen.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
