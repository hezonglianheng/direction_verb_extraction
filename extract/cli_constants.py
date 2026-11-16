# encoding: utf8
"""extract 阶段 CLI 与实现共用的常量（只依赖标准库，供 --help 轻量加载）。"""

DEFAULT_FILTER_CONFIG = ""
"""筛选配置文件默认路径

刻意**不再**指向一个不存在的文件：旧默认值 filter_config.json5 在仓库里根本没有，按默认跑
必然 FileNotFoundError，而报错只说「找不到文件」，看不出该选哪一个。实际的规则文件是
config_files/<词类>/<词类>_expressions.json5（方位词与趋向动词各一份）——选哪套是研究口径，不由程序替
用户猜，所以默认留空并在 main 里报出可选项（E2）。"""

LEGACY_OUTPUT_NAME = "filtered_sentences.jsonl"
"""只运行一套规则、且未指定 -o 时的输出文件名（与历史行为一致）"""
