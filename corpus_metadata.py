# encoding: utf8
"""跨阶段共享的语料类型、筛选提示语与句对文件元数据。

这里只放纯数据与纯函数，**不导入任何业务模块**：extract/replace 的轻量 CLI 与
业务实现都从这里取同一份定义，避免帮助文案、类型别名和默认文件名各自漂移。
"""

CORPUS_TYPE_DIRS: dict[str, str] = {
    # 语料类型（中文名，与 FOCUS_VOCAB_SPECS、记录里的 kind 同一套字符串）-> 配置子目录名
    # 别名只是命令行入口，见 CORPUS_TYPE_ALIASES；类型枚举的真源是这里与 FOCUS_VOCAB_SPECS
    "方位词": "localization",
    "趋向动词": "direction",
}
"""语料类型 -> config_files/ 下的配置子目录"""

CORPUS_TYPE_ALIASES: dict[str, str] = {
    # 别名（大小写不敏感）-> 中文类型名
    "localization": "方位词",
    "location": "方位词",
    "loc": "方位词",
    "direction": "趋向动词",
    "dir": "趋向动词",
    "direction_verb": "趋向动词",
}
"""命令行上可用的类型别名"""

DEFAULT_CORPUS_TYPE = "方位词"
"""缺省语料类型：与默认语料 results/location_corpus.jsonl 同一口径"""

SCREEN_PROMPT_FILES: dict[str, str] = {
    "方位词": "extract/screen_localization",
    "趋向动词": "extract/screen_direction",
}
"""词类型 -> system_prompt/ 下的提示语文件名（不含 .txt；含模块子目录 extract/）。"""

PAIR_CORPORA: dict[str, str] = {
    "方位词": "location_pairs.jsonl",
    "趋向动词": "direction_pairs.jsonl",
}
"""语料类型 -> 替换阶段的默认句对文件名（results/ 下）。"""


def normalize_corpus_type(value: str | None) -> str:
    """语料类型规范化：别名（大小写不敏感）-> 中文类型名；None/空串 -> DEFAULT_CORPUS_TYPE

    认不出来时抛 ValueError，**绝不当成默认类型静默跑**——那会让人以为跑的是 X、
    实际读的是方位词的四张表，与「绝不静默降级」相违。

    Args:
        value (str | None): 中文类型名或 CORPUS_TYPE_ALIASES 里的别名

    Returns:
        str: 中文类型名（CORPUS_TYPE_DIRS 的键之一）
    """
    if value is None or not str(value).strip():
        return DEFAULT_CORPUS_TYPE
    text = str(value).strip()
    if text in CORPUS_TYPE_DIRS:
        return text
    alias = CORPUS_TYPE_ALIASES.get(text.lower())
    if alias:
        return alias
    available = "、".join(
        f"{kind}（{'/'.join(sorted(a for a, k in CORPUS_TYPE_ALIASES.items() if k == kind))}）"
        for kind in CORPUS_TYPE_DIRS
    )
    raise ValueError(f"认不出语料类型 {value!r}；可用：{available}")
