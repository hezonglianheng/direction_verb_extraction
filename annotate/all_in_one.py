# encoding: utf8
"""旧单次标注入口，所有执行均转发给 spatial_runner 的 all_in_one 图。

用法（原参数继续可用，也可使用共同 runner 的 --models、预算与补发开关）：
    python -m annotate.all_in_one --corpus results/location_pairs.jsonl --limit 5
    python -m annotate.all_in_one --print-prompt
    python annotate --engine all-in-one --no-llm --batch off
    python annotate --graph all_in_one --models model-a model-b

默认 limit=1；默认禁止结构化补发；传输重试仍按模型配置。
--all-candidates / --no-syntax / --batch-inference 仅兼容旧命令，正式标注不再展开或准备句法。
"""

import sys

from annotate import spatial_agents as sa
from annotate import spatial_runner as sr

_PROJECT_ROOT = sr._PROJECT_ROOT
PROMPT_FILE = sa.ALL_IN_ONE_PROMPT_FILE
DEFAULT_OUT_PREFIX = "all_in_one"
DEFAULT_LIMIT = 1


def _forward_args(argv=None):
    return ["--graph", "all_in_one", "--engine", "all-in-one",
            *(sys.argv[1:] if argv is None else argv)]


def parse_args(argv=None):
    return sr.parse_args(_forward_args(argv))


def main(argv=None):
    return sr.main(_forward_args(argv))


if __name__ == "__main__":
    main()
