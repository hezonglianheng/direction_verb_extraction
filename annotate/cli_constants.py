# encoding: utf8
"""annotate 轻量 CLI 与业务模块共用的常量（只依赖标准库，供 --help 快速加载）。

放在这里而不是 spatial_agents / spatial_graphs：那两个模块导入时会加载
LTP/torch/langchain（本机实测约 3 分钟），而 --help 只需要这些纯数据。
业务模块从这里取值，保证帮助里的默认值与实际执行口径一致。
"""

DEFAULT_MODEL = "deepseek-flash"
"""默认模型配置名（model_config/api_keys.json 里的键）"""

FORMAL_GRAPHS = ("pair", "synonym", "all_in_one")
"""正式标注图；诊断图（single/alter_check）不支持抽样对齐、--print-prompt 等"""

GRAPH_DEFAULT_LIMITS: dict[str, int] = {
    "alter_check": 100,
    "all_in_one": 1,
    "pair": 100,
    "single": 100,
    "synonym": 100,
}
"""未指定 --limit 时各图的默认上限（须与 spatial_graphs.GRAPHS[*].default_limit 一致）"""

GRAPH_CHOICES = tuple(sorted(GRAPH_DEFAULT_LIMITS))
"""--graph 的可选值；排序后与 sorted(spatial_graphs.GRAPHS) 一致"""

ALL_IN_ONE_PROMPT_FILE = "annotate/all_in_one/annotate_all_in_one"
"""all_in_one 单次完整标注的主提示语，与 pair 共享类别片段"""
