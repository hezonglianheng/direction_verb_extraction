"""旧模块名/脚本的兼容层；实现只在阶段包中保留一份。"""

import importlib
import runpy
import sys


def redirect(name: str, target: str) -> None:
    if name == "__main__":
        runpy.run_module(target, run_name="__main__")
    else:
        sys.modules[name] = importlib.import_module(target)
