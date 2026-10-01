# encoding: utf8

"""空间语义改写-判定流水线（spatial_agents.py）用到的工具。

内容分三类：

1. 配置读取：config_files/ 下的四张表——改写规则（alter_rules.json5）、语义特征类别
   （feature_classes.json5）、schema 表（schema_table.json5）、判定表（judge_table.csv，
   唯一一张 CSV：首行首列是 schema 名、格子是结论符号，格式见 load_judge_table）。
   四张表**按语料类型分两组**（见 CORPUS_TYPE_DIRS）：方位词在 config_files/localization/、
   趋向动词在 config_files/direction/，两组里文件名相同，路径一律由 config_path 解析。
   跑哪种类型由 spatial_runner.py 的 --corpus-type 指定（默认方位词），构造期注入 agent。
   哪一组文件缺了或内容为空都按空表跑，对应环节退化为 LLM 兜底或原样返回，并在结果记录的
   source 字段里如实标记来源；跑批入口启动时会逐张打印实际路径与内容摘要，绝不静默降级。
   这两组目录下还各有一份 reality_agent 的提示语（reality.txt，见 CONFIG_PROMPT_FILES）：
   它的判据按词类分，与四张表同一套分组口径，路径与读法见 config_prompt_path /
   utils.read_prompt_path。
2. 纯规则工具：切片改写 alter_sentence（一个原句按其同义词组产出多个待选句）、词组定位
   find_word_groups、schema 查表 lookup_schema（先按关注词收窄、再按语义特征匹配）、
   判定查表 lookup_judgement、结果落盘 write_jsonl —— 不调 LLM，可脱离 agent 单独测试
   （.venv/bin/python spatial_tools.py）。收窄那一步的口径见 _narrowed_groups：一个词条能用
   words 声明它覆盖哪些关注词词形（外面/外边/外头 都归「外」这一类），schema 名仍用组名。
3. 供 LLM agent 调用的 @tool：parse_sentence 提供分词/词性/依存句法证据，让模型基于
   结构判断而不是仅凭语感。

另有一处贯穿全流程的口径：**本次需要重点关注的词**（pick_focus_word）。一条语料是为哪个
方位词/趋向动词抽出来的，改写与五个 LLM agent 都要知道，逐级向下传的只有三样——index、
词本身、属方位词还是趋向动词。取词口径见 pick_focus_word；词表真源是两张筛选配置，
与抽样（word_sampler.py）共用同一个 derive_vocab。

一对句子里的两侧各有各的关注词：a 侧（原句）看的是记录里那个词，b 侧（改写句）看的是规则表
换上去的那个词（见 replacement_focus_word，来源记 FOCUS_SOURCE_ALTER）。所以 b 侧的 schema
收窄到的是替换词所属的词条，判定表里 上12 × 里1 这样的格子才对得上——两侧共用一个词的话，
b 侧会收窄回原词那一组，判定表就永远只在自己那一组里打转。
"""

import csv
import json
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import json5
from langchain.tools import tool

import config
import word_sampler
from sentenceTable import SentenceTaskCache

_PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = _PROJECT_ROOT / "config_files"
"""四张规则表的根目录；每类语料一张同名子目录（见 CORPUS_TYPE_DIRS）"""

TABLE_FILES: dict[str, str] = {
    # 逻辑表名 -> 文件名。各类型子目录里**同名**，于是「表」只有一处定义
    "alter_rules": "alter_rules.json5",
    "feature_classes": "feature_classes.json5",
    "schema_table": "schema_table.json5",
    "judge_table": "judge_table.csv",
}
"""本流水线读的四张表；值同时是各子目录里的文件名（单一真源）"""

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

CORPUS_TYPE_CORPORA: dict[str, str] = {
    # 类型 -> 该类型的默认语料文件名（results/ 下）；键与 word_sampler.CORPUS_SPECS 同名
    "方位词": "location_corpus.jsonl",
    "趋向动词": "direction_corpus.jsonl",
}
"""每类语料的默认输入文件（spatial_runner 不给 --corpus 时用）"""

FOCUS_VOCAB_SPECS: dict[str, tuple[str | Path, tuple[str, ...] | None]] = {
    # 类别 -> (筛选配置, 只取哪些键的词表；None 表示取各顶层规则自己的词表)
    # 与 word_sampler.CORPUS_SPECS 同一口径，别在这里另立一套词表（词表只有一处真源）
    "方位词": (CONFIG_DIR / "localization_expressions.json5", None),
    "趋向动词": (CONFIG_DIR / "direction_expressions.json5", ("direction_verb_",)),
}
"""关注词的两份词表：判一个词属方位词还是趋向动词，就按它落在哪张表里定

刻意不用 config_files/localization_vocab.json5：那是初筛用的超集（会做子串最小化，
出来/上来/过去 会被吞成 出/上/过），且它在 sentenceFilters 自测里是回归基线，不是目标词表。
"""

# 三处都写着「有哪些语料类型」：CORPUS_TYPE_DIRS、CORPUS_TYPE_CORPORA、FOCUS_VOCAB_SPECS。
# 对不上就是改漏了一处，在 import 时就报，别留给跑批跑到一半才暴露。
assert set(CORPUS_TYPE_DIRS) == set(FOCUS_VOCAB_SPECS), (
    "语料类型两处真源对不上：CORPUS_TYPE_DIRS 与 FOCUS_VOCAB_SPECS"
)
assert set(CORPUS_TYPE_CORPORA) == set(CORPUS_TYPE_DIRS), (
    "语料类型两处真源对不上：CORPUS_TYPE_CORPORA 与 CORPUS_TYPE_DIRS"
)
assert set(CORPUS_TYPE_CORPORA.values()) <= set(word_sampler.CORPUS_SPECS), (
    "CORPUS_TYPE_CORPORA 里的语料名不在 word_sampler.CORPUS_SPECS 里（文件名写错了？）"
)


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


def config_path(table: str, corpus_type: str | None = None) -> Path:
    """四张规则表的路径：config_files/<语料类型目录>/<文件名>

    Args:
        table (str): TABLE_FILES 的键（alter_rules / feature_classes / schema_table / judge_table）
        corpus_type (str | None): 语料类型（中文名或别名）；None/空 = DEFAULT_CORPUS_TYPE

    Returns:
        Path: 只拼路径、不检查存在性——缺表按空表跑，由跑批入口在启动时逐张打印并告警
    """
    if table not in TABLE_FILES:
        raise ValueError(f"没有名为 {table!r} 的表；可用：{'、'.join(TABLE_FILES)}")
    return CONFIG_DIR / CORPUS_TYPE_DIRS[normalize_corpus_type(corpus_type)] / TABLE_FILES[table]


CONFIG_PROMPT_FILES: dict[str, str] = {
    # 逻辑名 -> 文件名。与四张表一样按语料类型分两组（见 CORPUS_TYPE_DIRS）
    "reality": "reality.txt",
}
"""与四张表同放在 config_files/<语料类型>/ 下的**提示语**（目前只有 reality_agent 这一份）

为什么不进 system_prompt/（其余五份提示语都在那儿）：判据随词类不同——方位词要挡的是
「会上/意义上/三天内」这类引申与时间用法，趋向动词要挡的是「笑起来/看出来/走出困境」
这类时体、结果与比喻用法，两类词的判据不是一回事。与四张表同一个目录、同一套按类型分组的
口径，改一类的标准碰不到另一类。读与报错见 utils.read_prompt_path。
"""


def config_prompt_path(name: str, corpus_type: str | None = None) -> Path:
    """config_files/<语料类型目录>/ 下的提示语路径（口径与 config_path 一致：只拼路径、不查存在性）

    Args:
        name (str): CONFIG_PROMPT_FILES 的键
        corpus_type (str | None): 语料类型（中文名或别名）；None/空 = DEFAULT_CORPUS_TYPE

    Returns:
        Path: 文件不在时的报错由 utils.read_prompt_path 负责（它会列出同目录下现有哪些 .txt）
    """
    if name not in CONFIG_PROMPT_FILES:
        raise ValueError(f"没有名为 {name!r} 的配置目录提示语；可用：{'、'.join(CONFIG_PROMPT_FILES)}")
    return CONFIG_DIR / CORPUS_TYPE_DIRS[normalize_corpus_type(corpus_type)] / CONFIG_PROMPT_FILES[name]


ALTER_RULES_FILE = config_path("alter_rules")
"""替换规则表：原词 -> 候选替换词（默认类型即方位词那张）

注意它**不再是 load_alter_rules 的默认值**（那个函数的默认值在调用时按语料类型解析，见 config_path）。
保留这个名字只为文档、命令行 help 与外部引用；路径真源是 config_path。
语义与旧版不同：过去它指向一个不存在的根文件（等价于不改写），现在指向方位词的实表。
"""
FEATURE_CLASSES_FILE = config_path("feature_classes")
"""语义特征类别表：特征类别的定义（默认类型即方位词那张，同上不再作 load_* 的默认值）"""
SCHEMA_TABLE_FILE = config_path("schema_table")
"""schema 表：语义特征组合 -> schema（默认类型即方位词那张）"""
JUDGE_TABLE_FILE = config_path("judge_table")
"""判定表：两个 schema -> 意义是否相同（默认类型即方位词那张）

四张表里唯一的 CSV（judge_table.csv）：首行首列是 schema 名、格子写结论符号，格式与匹配
规则见 load_judge_table。改成矩阵是为了能逐格声明「这一对手工裁定」还是「交 LLM」——
旧版 JSON5 的 same: true/false 只有两档，想表达「交 LLM」只能不写，与「忘了写」分不开。
"""
_cache = SentenceTaskCache(max_size=config.CACHE_MAX_SIZE)
"""句法分析结果缓存，与筛选阶段共用同一套 LTP 推理与缓存逻辑"""


# --------------------------------------------------------------------------
# 一、配置读取
# --------------------------------------------------------------------------


def load_json5(path: str | Path) -> Any:
    """读 json5 配置文件（与 sentenceFilters.py 用同一种格式，支持注释与尾逗号）"""
    with open(path, "r", encoding="utf-8") as f:
        return json5.load(f)


def _table_path(path: str | Path | None, table: str, corpus_type: str | None) -> Path:
    """四张表的路径：显式给的 path 优先，否则按语料类型解析（见 config_path）

    默认值刻意不写成 `path=常量`：那样在 import 时就把路径绑死成方位词那张，忘了传类型的
    趋向动词调用会静默读错表——正是「按类型分组」要消灭的东西。
    """
    return Path(path) if path is not None else config_path(table, corpus_type)


def load_alter_rules(path: str | Path | None = None, *, corpus_type: str | None = None) -> dict:
    """读替换规则表，文件不存在时返回空表（等价于“不改写”）

    表形如 {"groups": [["上", "中", "里"], ["前", "后"]]}：一串同义词组，
    组内任意一个词都能换成组内其他词。

    Args:
        path (str | Path | None): 显式表路径；None 时按 corpus_type 解析
        corpus_type (str | None): 语料类型（中文名或别名）；None = DEFAULT_CORPUS_TYPE
    """
    path = _table_path(path, "alter_rules", corpus_type)
    if not path.exists():
        return {"groups": []}
    return load_json5(path) or {"groups": []}


def load_feature_classes(path: str | Path | None = None, *, corpus_type: str | None = None) -> list[dict]:
    """读语义特征类别表，返回其中已填写 key 的类别列表。

    返回空列表表示该类语料的特征类别尚未定义，此时 feature_agent 退化为让模型自由抽取特征。
    """
    path = _table_path(path, "feature_classes", corpus_type)
    if not path.exists():
        return []
    classes = (load_json5(path) or {}).get("classes", [])
    return [c for c in classes if isinstance(c, dict) and c.get("key")]


def load_schema_table(path: str | Path | None = None, *, corpus_type: str | None = None) -> dict:
    """读 schema 表，文件不存在时返回空表（查表必然未命中，走 LLM 兜底）

    表按**词条**分组，一个词条下挂若干编号的图式，schema 名就是「组名 + 编号」：
        {"schemas": {"<组名>": {"words": ["<词形>", ...],
                                "<编号>": {"description": "...", "match": [...]}}}}
    组名是这一类的代表词形；words（可省，与编号同级）列出这一类还覆盖哪些词形，两者合起来
    决定收窄（见 _narrowed_groups）：关注词落在 {组名} ∪ words 里就归这一组。没写 words 的
    词条只认与组名逐字相等的关注词。同形词（「上」既是方位词也是趋向动词）在各自的类型目录里
    各收各的词条，靠 match 与 description 区分用法——这是把表按语料类型分组的另一个理由。
    """
    path = _table_path(path, "schema_table", corpus_type)
    if not path.exists():
        return {"schemas": {}}
    return load_json5(path) or {"schemas": {}}


JUDGE_CELL_SAME = "√"
"""判定表格子：两句意义相同——直接出结论，不调 LLM"""
JUDGE_CELL_DIFFERENT = "×"
"""判定表格子：两句意义不同——直接出结论，不调 LLM"""
JUDGE_CELL_LLM = "○"
"""判定表格子：交大模型做进一步判断——走 LLM 兜底"""
JUDGE_NO_SCHEMA = "未判出"
"""判定表的行/列标签：schema 名为 None（没判出 schema）那一路

那一侧没有 schema，判定表也无从按 schema 断言，故它的对角格自动填 ○（见 load_judge_table）；
它与别的 schema 的组合照常由作者逐格裁定。
"""

JUDGE_CELLS: tuple[str, ...] = (JUDGE_CELL_SAME, JUDGE_CELL_DIFFERENT, JUDGE_CELL_LLM)
"""格子上合法的三种符号。第四类「留空」是非法：说明这张表没填完（见 lookup_judgement）"""

JUDGE_ILLEGAL_SHOWN = 8
"""非法格的汇总最多列出几个：一张表填了一半时非法格可能上百条，全列出来把启动打印淹掉"""

JUDGE_CELL_ALIASES: dict[str, str] = {
    # 输入法/Excel 打出来的异体与顺手敲的 ASCII 替身 -> 规范符号。
    # normalize() 是 NFKC，折不动这几组（〇 U+3007、✓ U+2713、✕ U+2715 都不在兼容区里），
    # 只能显式列；漏掉一个，Excel 里打的 ○ 就会静默变成「非法留空」。
    "✓": JUDGE_CELL_SAME,
    "✔": JUDGE_CELL_SAME,
    "v": JUDGE_CELL_SAME,
    "V": JUDGE_CELL_SAME,
    "✕": JUDGE_CELL_DIFFERENT,
    "✖": JUDGE_CELL_DIFFERENT,
    "x": JUDGE_CELL_DIFFERENT,
    "X": JUDGE_CELL_DIFFERENT,
    "〇": JUDGE_CELL_LLM,
    "◯": JUDGE_CELL_LLM,
    "o": JUDGE_CELL_LLM,
    "O": JUDGE_CELL_LLM,
}
"""格子上认的异体符号（键已规范化），值一律是 JUDGE_CELLS 里的规范符号"""


def _judge_cell_symbol(raw: str) -> str | None:
    """格子的原始文本 -> 规范符号；留空或认不出的符号返回 None（非法）

    两种 None 由调用方按原文区分：「留空」是这一侧没写（对称的另一侧写了就行），
    「写了却认不出」是要报出来的配置错（见 _judge_table_from_rows）。
    """
    text = normalize(raw)
    if text in JUDGE_CELLS:
        return text
    return JUDGE_CELL_ALIASES.get(text)


def _judge_label(name: str | None) -> str:
    """schema 名 -> 表里的标签：None 那一路用 JUDGE_NO_SCHEMA，其余与读表同一口径（normalize）

    两边都用 normalize 是为了对齐「全角/半角、夹了空格」这类手写差异；表里的标签在
    _judge_table_from_rows 里也是这么规整的，于是 (schema 名, 标签) 的比较是同一把尺子。
    """
    return normalize(JUDGE_NO_SCHEMA if name is None else str(name))


def _judge_table_from_rows(raw_rows: Iterable[list[str]]) -> dict:
    """CSV 的原始行 -> 判定表字典（格式与字段含义见 load_judge_table）"""
    # 行号按原始文件算、不按过滤后的位置算：以 # 开头的行虽然跳过，它占掉的表格行不会消失，
    # 而这张表是拿 Excel 打开填的，problems 里的行号要能直接对上所见的行。
    rows = [
        (line_no, list(row))
        for line_no, row in enumerate(raw_rows, start=1)
        if row and not normalize(row[0]).startswith("#")
    ]
    problems: list[str] = []
    if not rows:
        return {"cells": {}, "labels": [], "filled": 0, "problems": problems}

    _, header = rows[0]
    body = rows[1:]

    # 标签：空标签直接丢掉（Excel 存盘常留尾部空列），重复的只认第一次出现的位置。
    # 两张对齐表里 "" 表示「这一行/列不收数据」，后面按标签取格时一并跳过。
    columns: list[str] = [""]
    for index, cell in enumerate(header[1:], start=1):
        label = normalize(cell)
        if not label:
            columns.append("")
        elif label in columns:
            columns.append("")
            problems.append(f"表头第 {index + 1} 列的标签 {label!r} 重复，只按第一次出现的位置取数据")
        else:
            columns.append(label)

    row_labels: list[str] = []
    for line_no, row in body:
        label = normalize(row[0]) if row else ""
        if not label:
            row_labels.append("")
        elif label in row_labels:
            row_labels.append("")
            problems.append(f"第 {line_no} 行首列的标签 {label!r} 重复，只按第一次出现的位置取数据")
        else:
            row_labels.append(label)

    labels: list[str] = []
    for label in columns + row_labels:
        if label and label not in labels:
            labels.append(label)

    # 先按「逻辑对」把格子的原文收齐：对称的两格是同一个逻辑对，行主序先出现的优先。
    # 一格里留空、对称那格写了，是「只写一侧」的正常写法（表是对称的），不算非法；
    # 两侧都没写、或哪一侧写了却认不出（符号打错），都算非法——这正是按对判、而不是
    # 按格判的理由。原文和符号一起留着，「留空」与「写了但认不出」才分得开。
    per_pair: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    for index, (line_no, row) in enumerate(body):
        row_label = row_labels[index]
        for col_index, cell in enumerate(row):
            if col_index == 0:
                continue
            column = columns[col_index] if col_index < len(columns) else ""
            text = normalize(cell)
            if not row_label or not column:
                if text:
                    problems.append(
                        f"第 {line_no} 行第 {col_index + 1} 列的格子 {cell!r} 有一侧没有标签，取不到"
                    )
                continue
            if row_label == column:
                # 对角线由格式定，不归作者写（见 load_judge_table 的说明）
                auto = _judge_diagonal(row_label)
                if text and _judge_cell_symbol(cell) != auto:
                    problems.append(
                        f"对角格 {row_label!r}×{row_label!r} 被写成 {cell!r}，"
                        f"对角线一律按格式自动定为 {auto}，这一格不算数"
                    )
                continue
            pair = (row_label, column) if row_label <= column else (column, row_label)
            per_pair.setdefault(pair, []).append((text, _judge_cell_symbol(cell)))

    cells: dict[tuple[str, str], str] = {}
    illegal: list[str] = []
    # filled 在分类之前算好：分类要按「整表有没有人填过」分档，边循环边累加会受迭代顺序影响
    # （先遇到空对就还是 0，后遇到写过的对才变 1，同一张表换个行序结论就不同）。
    filled = sum(1 for cells_of_pair in per_pair.values() if any(text for text, _ in cells_of_pair))
    for (label_a, label_b), written_cells in per_pair.items():
        # 写了却认不出的符号不能当成「这一侧没写」：另一侧有值时会静默按那一侧出结论，
        # 作者的这一格就被吞了。按非法处理，至少启动时看得见。
        garbled = [text for text, symbol in written_cells if text and symbol is None]
        written = [symbol for _, symbol in written_cells if symbol]
        if garbled or not written:
            # 空模板（filled 为 0）里两侧都没写的对是未命中、不是非法——与 lookup_judgement
            # 同一口径（查表那边也是 filled 为 0 一律 miss），否则启动会先打一条
            # 「N 格留空（非法）」的警告、紧接着又打「表里没有已填内容」的提示，自相矛盾。
            if filled:
                illegal.append(f"{label_a}×{label_b}")
            continue
        symbol = written[0]
        if any(other != symbol for other in written[1:]):
            problems.append(
                f"格子 {label_a}×{label_b} 与它对称的那格对不上（{'、'.join(written)}），"
                f"按行主序先出现的 {symbol} 为准"
            )
        cells[(label_a, label_b)] = symbol
        cells[(label_b, label_a)] = symbol

    # 表头或首列列了名字、却没有那一格（某行短了一截、只写了行没写列）同样是「留空」：
    # 查表时本就按非法处理（见 lookup_judgement），诊断里就不能整类缺席——否则这批对只在
    # 运行期以 judge_lookups["illegal"] 露面，作者却看不到该补哪一行。
    # 作者一格都没填的空模板不在此列（filled 为 0）：那时全表算未命中，不是非法。
    if filled:
        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                pair = (labels[left], labels[right])
                if pair[0] > pair[1]:
                    pair = (pair[1], pair[0])
                if pair not in per_pair:
                    illegal.append(f"{pair[0]}×{pair[1]}")

    if illegal:
        # 插在最前：这一条是「该补哪些格」的汇总，而 config_report 只截前 5 条——
        # 排在末尾时，越是格子缺得多（越需要这句）越容易正好被截掉。
        # 「还有 N 格」从实际列出的条数算，不另写一个 8：两处各写一遍时，改了截断长度
        # 而漏改这一处，就会出现「列了 9 格、还说还有 1 格」的对不上账。
        shown = illegal[:JUDGE_ILLEGAL_SHOWN]
        rest = len(illegal) - len(shown)
        problems.insert(
            0,
            f"{len(illegal)} 格留空或符号认不出（非法）：{'、'.join(shown)}"
            + (f"，还有 {rest} 格" if rest else "")
            + "；这些格查表时如实降级交 LLM",
        )

    for label in labels:
        cells[(label, label)] = _judge_diagonal(label)

    return {"cells": cells, "labels": labels, "filled": filled, "problems": problems}


def _judge_diagonal(label: str) -> str:
    """对角格自动填什么：同一个 schema 按定义同义；「未判出」两侧都没 schema，表无权断言"""
    return JUDGE_CELL_LLM if label == normalize(JUDGE_NO_SCHEMA) else JUDGE_CELL_SAME


def load_judge_table(path: str | Path | None = None, *, corpus_type: str | None = None) -> dict:
    """读判定表（CSV 矩阵），文件不存在时返回空表（查表必然未命中，走 LLM 兜底）

    表长这样（首行首列是 schema 名，格子是结论符号）：

        ,里1,里2,外1,未判出
        里1,,×,○,○
        外1,,,√,○
        未判出,,,,

    格子只能是三种符号之一，第四类「留空」是**非法**（这张表没填完）：

        √  两句意义相同 -> 直接出结论，不调 LLM（lookup_judgement 记 source="table"）
        ×  两句意义不同 -> 直接出结论，不调 LLM（同上）
        ○  交大模型做进一步判断 -> 走 LLM 兜底（source="defer"）
        留空（或认不出的符号）-> 非法：启动打警告，落到处如实降级交 LLM（source="illegal"）

    约定：
      1. 匹配是对称的：(里1, 外1) 与 (外1, 里1) 是同一个逻辑对，写一侧即可；两侧都写且
         不一致算冲突（记 problem，按行主序先出现的为准）。**留空**指这一对没有可用的结论，
         三种情形都算非法、都记进 problems（启动时打警告）：两侧都没写、哪一侧写了却认不出
         （符号打错）、表头或首列列了名字却没有那一格（某行短了一截、只写了行没写列）。
      2. 表里没列出的 schema 名（如运行期 llm_free 自由命名的 里3）算未命中 -> 交 LLM，
         不报警。这是「只列想手工裁定的那些」的口子，也是这张表不必穷举的理由。
      3. schema 名为 None（没判出 schema）那一路的标签写 未判出 常量（JUDGE_NO_SCHEMA）。
      4. 对角线自动填，作者不必写、写了也不算数：真实 schema 结对角格 √，未判出 的对角格 ○。
         自动填的格子不计入 filled，于是「只写了表头」仍算空表（走既有的空表提示与兜底）。
      5. 以 # 开头的行整行跳过，供作者写理由——格子只放符号，没有旧版 JSON5 的 note 字段。
         problems 里的行号仍按文件实际行算，跳过注释行也不会错位。
      6. 文件用 utf-8-sig 读（Excel 存盘会带 BOM），csv.reader 配 newline="" 以正确处理 CRLF。

    格式说明只能写在这里：CSV 放不下注释，而这张表是要用 Excel 打开填的。
    `problems` 里是给人看的诊断（非法格、冲突、重复标签），跑批入口启动时打印。

    Returns:
        dict: {"cells": {(标签 a, 标签 b): 符号}——只有符号认得出的对才进来, "labels": [标签, ...],
               "filled": 作者写过的对数（镜像两格算一格、自动对角线不算；符号认不出的也算写过，
                         否则整表只剩认不出的符号时会被当成空模板，落点从 illegal 滑成 miss）,
               "problems": [诊断字符串, ...]}
    """
    path = _table_path(path, "judge_table", corpus_type)
    if not path.exists():
        return {"cells": {}, "labels": [], "filled": 0, "problems": []}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return _judge_table_from_rows(csv.reader(f))


# --------------------------------------------------------------------------
# 二、语料读取与切片处理
# --------------------------------------------------------------------------


def read_corpus(path: str | Path, limit: int | None = None, offset: int = 0) -> list[dict]:
    """读 results/location_corpus.jsonl 一类语料

    每行形如：
        {"sentence": ["在", "桌子", "上", "。"], "source_file": "...",
         "filter_methods": ["locality_phrase"], "match_tree": [{"index": 2, "word": "上", "children": []}]}

    Args:
        path (str | Path): 语料文件路径
        limit (int, optional): 最多读多少行，None 表示不限
        offset (int): 跳过前多少行

    Returns:
        list[dict]: 每条记录原样返回，另加 "line_no" 字段记录它在文件里的行号（从 1 开始）
    """
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no <= offset or not line.strip():
                continue
            record = json.loads(line)
            record["line_no"] = line_no
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def join_slice(sentence_slice: Iterable[str]) -> str:
    """把分词切片拼回句子字符串

    语料里的标点是独立 token（有的还带尾空格，如 ": "），所以直接无缝拼接，
    不做任何再切分或补空格。
    """
    return "".join(sentence_slice)


def normalize(text: str) -> str:
    """规范化句子文本，用于查表、去重和比较

    统一全角/半角、去掉所有空白，这样 "软件上 说" 与 "软件上说" 会落到同一个键上。
    """
    return "".join(unicodedata.normalize("NFKC", text).split())


def collect_span(node: dict) -> list[int]:
    """收集匹配树节点覆盖的所有 token 下标（含各级 children）

    match_tree 的一个节点形如 {"index": 4, "word": "上", "children": [{"index": 2, "word": "软件"}]}：
    index/word 是方位词本身，children 是它的定语等修饰成分。改写只动方位词那个 token，
    修饰成分原位保留（“硬件上” -> “硬件中”），所以这里把整个 span 记下来供结果留档。
    """
    indices = []
    if isinstance(node.get("index"), int):
        indices.append(node["index"])
    for child in node.get("children") or []:
        indices.extend(collect_span(child))
    return sorted(set(indices))


def pick_word_info(record: dict) -> dict | None:
    """从语料记录里取出待改写的词语信息

    取 match_tree 的第一个节点（该句被哪条筛选规则命中，此处就是哪个词）。
    没有 match_tree（或它不是节点列表）或节点缺 index 时返回 None，调用方据此判为改写成不了对。
    """
    tree = record.get("match_tree") or []
    if not isinstance(tree, list) or not tree:
        return None
    node = tree[0]
    if not isinstance(node, dict) or not isinstance(node.get("index"), int):
        return None
    return node


# --------------------------------------------------------------------------
# 三、关注词：这条语料是为哪个方位词/趋向动词抽出来的
# --------------------------------------------------------------------------


def _method_family(method: str) -> str:
    """筛选规则名 -> 它盯的是哪一族词（判不出来的给空串）

    direction_verb_as_center 与 *_one_complement / *_two_complement 盯的都是趋向动词
    （互补规则的目标词在子节点里，见 word_sampler 模块 docstring），locality_* 盯方位词。
    """
    if method.startswith("direction_verb") or method.endswith("_complement"):
        return "趋向动词"
    if method.startswith("locality"):
        return "方位词"
    return ""


@lru_cache(maxsize=1)
def _focus_vocab_cached() -> tuple[tuple[str, frozenset[str]], ...]:
    """两份词表的缓存：返回元组，调用方拿不到可变对象、改不坏缓存

    derive_vocab 要解析筛选配置（实测 44ms/次），而 pick_focus_word 每条记录都要调，
    每条都推一遍词表是纯浪费。
    """
    return tuple(
        (kind, frozenset(word_sampler.derive_vocab(path, prefixes)[0]))
        for kind, (path, prefixes) in FOCUS_VOCAB_SPECS.items()
    )


def load_focus_vocab() -> dict[str, set[str]]:
    """关注词的两份词表 {"方位词": {...}, "趋向动词": {...}}（浅拷贝，改不到缓存里那份）

    跑批启动时打印一次，顺带把配置问题（筛选配置改名、取词表的前缀对不上）当场暴露出来。
    """
    return {kind: set(words) for kind, words in _focus_vocab_cached()}


def _record_family(filter_methods: Iterable[str]) -> str:
    """这条语料是哪条规则抽出来的 -> 它盯的那一族词（判不出、或两族都沾时给空串）

    用于两处：破「上/下」两表都收时的平局，以及把 level 2 的候选限制在本族的词表里。
    两族同时出现（如一条规则的 filter_methods 里既有 locality_* 又有 *_complement）时给空串，
    不按列表顺序猜——顺序变了结论就变，那不是判据。
    """
    families = {_method_family(str(method)) for method in filter_methods or []}
    families.discard("")
    return families.pop() if len(families) == 1 else ""


def focus_kind(
    word: str, filter_methods: Iterable[str] = (), vocab: dict[str, set[str]] | None = None
) -> str:
    """这个词属方位词还是趋向动词；判不出来给空串

    以词表归属为准。只有两张表都收时才用筛选规则族破平局——实测两表交集只有「上」「下」，
    而两份语料的两族规则互斥（方位词语料 locality_*，趋向动词语料 direction_verb* / *_complement）。

    词不在任何一张表里（如回落 match_tree[0] 时取到的搭配动词 走/带）一律给空串：
    绝不能因为它所在的记录带 _complement 就标成「趋向动词」，那是假信息。
    """
    vocab = load_focus_vocab() if vocab is None else vocab
    kinds = [kind for kind, words in vocab.items() if word in words]
    if len(kinds) == 1:
        return kinds[0]
    if not kinds:
        return ""
    return _record_family(filter_methods)


def _focus_entry(node: dict, source: str, filter_methods: Iterable[str], vocab: dict[str, set[str]]) -> dict:
    """把节点收成向下传的那三样：index、词本身、类别（刻意不带 children 与同义词组）"""
    word = node.get("word")
    return {
        "index": node.get("index"),
        "word": word,
        "kind": focus_kind(word, filter_methods, vocab),
        "source": source,
    }


def pick_focus_word(record: dict, vocab: dict[str, set[str]] | None = None) -> dict | None:
    """这条语料「本次需要重点关注的词」——改写与五个 LLM agent 共用的一处口径

    三级回落，取到的是哪一级记在返回值的 source 里：

    1. "sampled_word"：语料经 word_sampler.py 抽样而带上的字段，有就直接用。
       抽样只落了词、没落位置，同一个词在句子里出现 ≥2 次时无从区分，这里取 index 最小的那个
       （实测 direction 3.3%、location 7.2% 的抽样记录如此）。实测这批记录的 index 与
       sentence[index] 100% 对齐，alter_sentence 的「index 处的词与 word 不一致」守卫不会误伤。
       抽样用的是**那份语料自己的词表**（word_sampler.CORPUS_SPECS），所以换一份 --vocab 抽出来的
       语料，这里的 sampled_word 可能一个都不在当前两张表里，此时会静默回落 level 2、source 记成
       "vocab"——汇总里的 focus_sources 就是这条的判据：抽样语料若一条 "sampled_word" 都没有，
       说明抽样词表与本配置对不上。
       这一级没有 ② 那道补语位偏好——要不要加得先看抽样语料上有没有这个形状：实测两份抽样语料里
       sampled_word 命中 ≥2 处的记录共 54（direction）/ 1274（location）条，其中取到排在搭配动词
       之前的 0 条，所以暂时不必加，加了也是空转。
    2. "vocab"：按词表在 match_tree 里递归找（口径见 word_sampler.target_nodes）。命中多个时
       **取 index 最小的目标词节点，不回落 match_tree[0]**——match_tree[0] 常常是搭配动词
       （走/带/凑），拿它顶掉一个真正的目标词与「本次关注的词」直接冲突。取 index 最小之前有
       两道收窄（顺序即下面两条，缺一不可）：

       ① **候选先按本记录的规则族收窄**：两族取并集会把「被 _complement 规则当补语收进来的
       方位词」选成关注词。实测收窄前后：direction 原始语料 156601 条里 959 条（0.61%）
       从错的方位词改回真目标词，location 原始语料 1530335 条一条都没动（本来就全对）；
       两份语料都 0 条落到「本族零命中」那条退路上。收窄后与 word_sampler 的抽样口径
       （按语料分表）也一致。只有本族一个词都没命中时才退回并集（纯防御，实测触发 0 次）。

       ② **本族候选里再优先「排在父节点之后」的那个**（口径见 word_sampler.pre_head_indices）。
       ①挡不住两种残留：同一个词在动词前后各出现一次（上/下 两张表都收，族收窄筛不掉它）、
       以及动词前那个是另一个本族词。实测 ② 在 direction 语料上改动 91 条（0.058%，含 13 条
       同词、78 条同族异词，如 撒上 取到 上@39 而不是排在「撒」前的 上@37、滚去 取到 去@12
       而不是 向下 的 下@10），location 语料 0 条改动——方位词的候选多是规则树的根（无父节点），
       这条偏好对它们不适用，所以只在本族生效，以后 locality 规则把别的结构收进子节点也不会被它误伤。
       本族候选全都排在父节点之前时退回原样（direction 语料实测 167 条，如「往下去走」的树把
       下@59 去@60 都挂在 走@61 之前——偏好无可偏好，只好按 index 最小取；location 语料 0 条）。
    3. "match_tree"：一个目标词节点都没找到（纯防御，实测两份语料触发 0 次）时，回落
       pick_word_info 的 match_tree[0]。此时词多半不在词表里，kind 会是空串。

    刻意不把 pick_word_info 合并进来：两者语义在 *_complement 规则下本就不同——一个取
    「被规则命中的待改写词」，一个取「词表命中的目标词」（见 word_sampler 模块 docstring），
    合并会同时改错改写口径与抽样口径。

    Returns:
        dict | None: {"index", "word", "kind", "source"}；记录没有 match_tree 时返回 None。
            kind 取 "方位词" / "趋向动词" / ""（判不出来就是空串，不猜）。
    """
    vocab = load_focus_vocab() if vocab is None else vocab
    filter_methods = record.get("filter_methods") or []
    nodes = word_sampler.target_nodes(record, set().union(*vocab.values()) if vocab else set())

    sampled = record.get("sampled_word")
    if isinstance(sampled, str) and sampled:
        matched = [node for node in nodes if node["word"] == sampled]
        if matched:
            return _focus_entry(matched[0], "sampled_word", filter_methods, vocab)

    family = _record_family(filter_methods)
    if family and family in vocab:
        preferred = [node for node in nodes if node["word"] in vocab[family]]
        nodes = preferred or nodes
    if len(nodes) > 1 and family == "趋向动词":
        # 方向补语跟在搭配动词之后：本族候选里优先排在父节点之后的那个（口径见 pre_head_indices）。
        # 只在本族生效——方位词的候选多是规则树的根，这条偏好对它们没有意义。
        after = [node for node in nodes if node["index"] not in word_sampler.pre_head_indices(record)]
        nodes = after or nodes
    if nodes:
        return _focus_entry(nodes[0], "vocab", filter_methods, vocab)

    node = pick_word_info(record)
    if node is None:
        return None
    return _focus_entry(node, "match_tree", filter_methods, vocab)


FOCUS_SOURCE_ALTER = "alter"
"""关注词的来源之一：改写句那侧的关注词，由规则表换上去的那个词重算（见 replacement_focus_word）

与 pick_focus_word 的三个来源（sampled_word / vocab / match_tree）并列，但它不在语料记录里，
是改写之后才有的——判定「这份关注词该按哪一侧的措辞说」就认这个值
（见 spatial_agents.focus_role）。
"""


def replacement_focus_word(
    focus_word: dict | None, replacement: str, filter_methods: Iterable[str] = ()
) -> dict:
    """改写句那一侧的关注词：把关注词换成替换后的词语，其余口径照旧算

    一对句子的两侧各看各的词：a 侧看语料记录里的原词，b 侧看规则表换上去的那个词。这样 b 侧的
    schema 才会收窄到替换词所属的词条（见 _narrowed_groups），判定表里按两组名字写的格子
    （如 上12 × 里1）才对得上；两侧共用一个词的话，b 侧会收窄回原词那一组，判定退化成同组自比。

    index 照搬 a 侧：改写只把切片里 index 处的那个 token 换成同义词组里的另一个（见 alter_sentence），
    改出来的候选切片长度与其余 token 都没动过，**改写句存在状态里的那份切片**（slice_b）在该下标
    处确实是替换词（实测 5310/5310）。但这个下标说的是原句的分词，不是改写句的：模型手上只有改写句
    字符串，它会自己去 parse_sentence，而重新分词在替换点附近可能并词（下+移 并成「下移」，
    实测 4.9%，见 spatial_agents.FOCUS_INDEX_NOTE_ALTER），那时同一下标指的就是别的 token。
    所以 b 侧的措辞只说「下标是原句上的位置」，并让模型按词本身在改写句里定位。

    kind 按**替换词**重算（词表归属 + 规则族破平局，见 focus_kind）：替换词换到另一族、或压根
    不在本语料类型的词表里时如实给出（空串就是不猜），不沿用原词的类别——沿用会把「这个词不是
    方位词/趋向动词」这条配置隐患盖住，而它正是跑批汇总里 replacement_kinds 要暴露的东西。

    Args:
        focus_word (dict | None): a 侧那份关注词（pick_focus_word 的返回值）；只为取 index
        replacement (str): 替换词，即改写句在 index 处实际用的那个词
        filter_methods (Iterable[str]): 该语料记录的筛选规则名，只用于「上」「下」两表都收时破平局

    Returns:
        dict: {"index", "word", "kind", "source"}，形状与 pick_focus_word 的返回值一致，
            source 固定是 FOCUS_SOURCE_ALTER
    """
    return {
        "index": (focus_word or {}).get("index"),
        "word": replacement,
        "kind": focus_kind(replacement, filter_methods),
        "source": FOCUS_SOURCE_ALTER,
    }


# --------------------------------------------------------------------------
# 四、纯规则改写（alter_agent 的实现，不调 LLM）
# --------------------------------------------------------------------------


def find_word_groups(word: str, rules: dict | None = None) -> list[dict]:
    """查出规则表里所有包含该词的词组（“待替换的词落在哪些组里”）

    规则表的 groups 是一串同义词组，组内任意词都能换成组内其他词，所以先要定位待替换词所在的组。
    返回每组的出处与词表，形如 [{"source": "groups[0]", "group": ["上", "中", "里"]}]。
    同一个词出现在多组里时全部返回，调用方把各组里的其他词并起来用。
    """
    rules = load_alter_rules() if rules is None else rules
    found: list[dict] = []
    for index, group in enumerate(rules.get("groups") or []):
        words = [w for w in group if isinstance(w, str)] if isinstance(group, list) else []
        if word in words:
            found.append({"source": f"groups[{index}]", "group": words})
    return found


def alter_sentence(
    sentence_slice: list[str],
    word_info: dict[str, Any],
    rules: dict | None = None,
) -> tuple[list[list[str]], dict]:
    """按规则替换切片中 index 处的词语，返回全部待选切片与改写记录。

    纯规则实现，不调用 LLM。替换词来源按优先级：
        1. word_info["replacement"]（调用方直接指定，便于按需组织不同改写）：只出一个待选句
        2. 规则表里包含该词的所有词组中，除该词以外的其他词：每个词各出一个待选句
    因为一组同义词往往能换好几个词，所以返回的是**候选切片的列表的列表**：
    一个候选切片对应一个待选句，空列表表示没命中任何替换词。

    detail["candidates"] 与返回的切片按下标一一对应，写明每个候选的替换词与它出自哪个词组。

    图里传进来的 word_info 是**关注词**（见 pick_focus_word）：含 index / word / kind，
    **不含 children**。因此 detail["span"] 通常只有它自己那一个下标——旧口径传的是 match_tree
    节点，span 还会带上定语等修饰成分（见 collect_span）。改写本身只动 index 处的那个 token，
    两种口径改出来的句子没有差别，变的只是留档的 span。

    Args:
        sentence_slice (list[str]): 待改写的分词切片
        word_info (dict[str, Any]): 关注词，至少含 "index" 与 "word"（可另带 "replacement"）
        rules (dict, optional): 替换规则表，默认读 config_path("alter_rules") 解析出的那张
            （默认类型方位词，即 config_files/localization/alter_rules.json5）

    Returns:
        tuple[list[list[str]], dict]: (待选切片列表, 改写记录)
    """
    rules = load_alter_rules() if rules is None else rules
    detail: dict[str, Any] = {
        "index": word_info.get("index"),
        "word": word_info.get("word"),
        "span": [],
        "applied": False,
        "reason": "",
        "candidates": [],
    }

    index = word_info.get("index")
    if not isinstance(index, int) or not 0 <= index < len(sentence_slice):
        detail["reason"] = "index 缺失或越界，无法定位待替换的词"
        return [], detail

    original = sentence_slice[index]
    # word 与切片不一致说明上游的 index 与切片没对齐，此时改写会动错词，宁可停下
    if word_info.get("word") not in (None, original):
        detail["reason"] = (
            f"index 处的词与 word 不一致（切片为 {original!r}，word 为 {word_info.get('word')!r}）"
        )
        return [], detail
    detail["word"] = original
    detail["span"] = collect_span(word_info)

    candidates: list[dict] = []
    specified = word_info.get("replacement")
    if specified:
        candidates.append({"replacement": specified, "source": "word_info.replacement", "group": []})
    else:
        seen: set[str] = set()
        for found in find_word_groups(original, rules):
            for word in found["group"]:
                # 组里的原词自己不算替换词；同一个替换词出现在多组里时只取先出现的那组
                if word == original or word in seen:
                    continue
                seen.add(word)
                candidates.append(
                    {"replacement": word, "source": found["source"], "group": found["group"]}
                )

    if not candidates:
        # 不点名具体路径：默认那张表不存在，调用方也可能用 --alter-rules 换了别的表
        detail["reason"] = (
            f"替换规则表里没有包含 {original!r} 的词组，请在本次使用的规则表里补上该组"
        )
        return [], detail

    new_slices: list[list[str]] = []
    for candidate in candidates:
        new_slice = list(sentence_slice)
        new_slice[index] = candidate["replacement"]
        new_slices.append(new_slice)

    detail.update(applied=True, candidates=candidates)
    return new_slices, detail


# --------------------------------------------------------------------------
# 五、查表：语义特征 -> schema，schema 对 -> 判定
# --------------------------------------------------------------------------

WORDS_KEY = "words"
"""词条下声明「这一类还覆盖哪些关注词词形」的保留键（与编号同级）

存在的理由：表的一级键只能写一个词形（「外」），而语料里的关注词还有 外面/外边/外头
（词表见 load_focus_vocab，实测 location 语料 41 个词形里 26 个是双音节）。没有它，
关注词是 外面 的句子在表里取不到「外」那一组，只能落到 LLM 自由命名出「外面1」，
与「外1」成了两类，判定表就得为每个词形各写一遍。

编号一律是数字（见 validate_schema_table），所以这个键不会与编号撞名。
"""


def _numbered_entries(entries: dict) -> dict:
    """一个词条下真正是图式的那部分：摘掉 words 键

    words 与编号同级，是本组的元信息、不是一条图式。凡是「遍历某组的编号条目」的地方
    （收窄、数条数、校验）都必须走这里，否则它会被当成一条编号条目——数条数会把它数进去，
    校验会报出「编号 'words' 要写数字」这种指错方向的错。
    """
    return {key: value for key, value in (entries or {}).items() if str(key) != WORDS_KEY}


def _group_words(group_word: str, entries: dict) -> list[str]:
    """一个词条覆盖的全部关注词词形：组名本身 + words 里列的（组名排在最前）

    组名总在其中，两条理由：schema 名就是「组名 + 编号」，一个不覆盖自己名字的组自相矛盾；
    取并集还保证「给一组补 words」是**纯增量**——原本能命中这组的关注词不会因为漏写了组名
    而静默掉进自由命名（那是配置层面的回归，不报错，只改名）。
    """
    listed = [form for form in (entries or {}).get(WORDS_KEY) or [] if isinstance(form, str)]
    return [group_word, *(form for form in listed if form != group_word)]


def schema_forms(table: dict | None = None) -> list[str]:
    """表里被声明过、因而**可能**当上关注词的全部词形（按表里的书写顺序，去重）

    「这个词形到底存在吗」只能靠语料类型的词表核对（见 load_focus_vocab）：words 里写了错字
    （「外靣」）不会报错，只是那个词形永远当不成关注词、那一条静默失效。跑批入口用它对一次账。
    """
    table = load_schema_table() if table is None else table
    forms: list[str] = []
    for group_word, entries in (table.get("schemas") or {}).items():
        forms.extend(form for form in _group_words(group_word, entries) if form not in forms)
    return forms


def _narrowed_groups(table: dict | None = None, focus_word: dict | None = None) -> dict[str, dict]:
    """按关注词收窄后的词条：{组名: 该组的编号条目}，保持表里的书写顺序

    收窄口径：

    - 关注词落在某组的 {组名} ∪ words 里，就取该组（见 _group_words）。没写 words 的组只有
      组名一个词形，于是逐字相等——与旧版一致：不看词形、不做前缀匹配，不会把「上」的图式
      算给「上面」，反之亦然。
    - 关注词缺失或词形为空时给出表中**全部**词条：这一层要排除的是「与关注词无关的格式」，
      没有关注词就没有可排除的判据；此时宁可不排除（详情里把 focus_word 记成 None），
      也不凭空认定某一族。

    这是收窄的**唯一**实现：候选名单（schema_candidates）与命名前缀（schema_group_key）都基于
    它，两者才不会各算各的——前缀与名单一旦分家，别名那一路的自由命名会被自己的校验全否掉。
    """
    table = load_schema_table() if table is None else table
    schemas = table.get("schemas") or {}
    word = (focus_word or {}).get("word") or ""
    groups = (
        schemas
        if not word
        else {w: entries for w, entries in schemas.items() if word in _group_words(w, entries)}
    )
    return {w: _numbered_entries(entries) for w, entries in groups.items()}


def _schema_count(table: dict) -> int:
    """表里一共有多少条图式（收窄前后各数一次，用来记录被关注词排除了几条）

    只数编号条目：words 是同级的元信息，不是一条图式（漏掉的后果是 excluded 虚高）。
    """
    return sum(len(_numbered_entries(entries)) for entries in (table.get("schemas") or {}).values())


def schema_candidates(table: dict | None = None, focus_word: dict | None = None) -> dict[str, dict]:
    """按关注词收窄后的候选图式：{schema 名: 条目}，保持表里的书写顺序

    schema 名是「组名 + 编号」（如「上12」「下1」）。组名是这一类的代表词形：词条写了 words 时，
    关注词是它列出的任一词形（外面/外邊/外头）都归进同一个名字空间（外1），判定表因此只需
    按类写一遍，不必为每个词形各写一条。

    收窄口径（哪些组算命中）见 _narrowed_groups。
    """
    return {
        f"{group_word}{number}": entry or {}
        for group_word, entries in _narrowed_groups(table, focus_word).items()
        for number, entry in entries.items()
    }


def schema_group_key(table: dict | None = None, focus_word: dict | None = None) -> str | None:
    """关注词归属的**组名**——这一类的图式共用的名字前缀；没有归属时给 None

    自由命名（关注词有归属但那一类还没收图式）那条路要用它：关注词是「里面」而归入「里」这一类
    时，名字必须是「里 + 编号」，命名成「里面1」与判定表里按类写的那套对不上，这一次抽取就白抽了。

    关注词缺失时**直接**给 None，不能写成 next(iter(_narrowed_groups(...)), None)——没有关注词时
    那一层刻意返回全部词条，取第一个会把每一句都钉到表里第一类上。
    """
    if not ((focus_word or {}).get("word") or ""):
        return None
    return next(iter(_narrowed_groups(table, focus_word)), None)


def _as_list(value: Any) -> list:
    """把特征取值统一成列表，便于多值特征（一个类别取多个标签）参与匹配"""
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _match_features(match: dict, features: dict) -> bool:
    """判断**一组** match 条件是否被 features 满足：match 里的每个键都要在 features 里命中取值

    两侧都统一成集合作交集判断：上游一个特征可能传来多个备选值（feature_agent 每类返回的
    就是列表），有交集即算这个键命中。因此一个键写多个值是「该键取其中任一」，多键时就是
    各自取任一的组合；要精确控制组合就写多组条件（见 _schema_alternatives）。
    """
    for key, expected in (match or {}).items():
        if key not in features:
            return False
        actual_values = {str(v) for v in _as_list(features[key])}
        expected_values = {str(v) for v in _as_list(expected)}
        if not actual_values & expected_values:
            return False
    return True


def _schema_alternatives(entry: dict) -> tuple[list[dict], bool]:
    """把一个条目的 match 拆成 (备选条件组列表, 是不是兜底条目)

    match 允许三种写法：一组条件的对象（{"area": [...]}）、多组条件的列表
    （[{"area": [...]}, {"area": [...], "distance": [...]}]）、省略或为空。
    列表里组与组之间是「或」——一个图式往往对应许多组特征值，如「外」的距离特征可以是
    [+紧连] 也可以是 [+近]，就写多组条件，不必为每一组单开一个编号。

    省略、为空、或列出的组全是空对象（{}）时返回 ([], True)：这是兜底条目，
    只有同词的其他图式都没命中时才采用（想让 LLM 去挑最接近的就别写兜底条目）。
    """
    match = (entry or {}).get("match")
    if isinstance(match, dict):
        groups = [match]
    elif isinstance(match, (list, tuple)):
        groups = [g for g in match if isinstance(g, dict)]
    else:
        groups = []
    groups = [g for g in groups if g]
    return groups, not groups


def format_match(match: Any) -> str:
    """把一组 match 条件渲染成一行说明（落盘 reason 与提示语共用这一处措辞）"""
    if isinstance(match, dict):
        if not match:
            return "（兜底条件：同词的其他图式都没命中时才是它）"
        return " 且 ".join(
            f"{key}={'、'.join(str(v) for v in _as_list(value))}" for key, value in match.items()
        )
    return str(match)


def render_schema_candidates(candidates: dict[str, dict]) -> str:
    """把候选图式渲染成 LLM 提示语里的一段：名字、说明、命中条件、例句

    名字、条件、例句都给全，模型才有依据在「一组都没命中」时挑一个最接近的——
    这正是「不命中就让它挑」的前提，条件给不全它就只能在名字上猜。
    """
    lines = []
    for name, entry in candidates.items():
        entry = entry or {}
        groups, is_fallback = _schema_alternatives(entry)
        if is_fallback:
            condition = format_match({})
        else:
            condition = " ／ ".join(format_match(group) for group in groups)
        parts = [f"  {name}：{entry.get('description') or '（未写说明）'}"]
        parts.append(f"      命中条件：{condition}")
        examples = [str(e) for e in (entry.get("examples") or []) if e]
        if examples:
            parts.append(f"      例句：{'；'.join(examples)}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


def is_schema_name(name: str, word: str = "") -> bool:
    """名字是不是「前缀 + 编号」的格式（LLM 自由命名那条路上用来兜住名字的格式）

    word 这里是**前缀**：一般是关注词，关注词归属某个词条（如「里面」归「里」）时传组名，
    否则同一个名字空间会被词形切碎（里1 与 里面1 分成两类）。传空串时只要求非空：
    没有可核对的前半段，此时不硬套格式。
    """
    if not isinstance(name, str) or not name:
        return False
    if not word:
        return True
    return name.startswith(word) and name[len(word) :].isdigit()


def validate_schema_table(
    table: dict,
    feature_keys: Iterable[str] | None = None,
    *,
    source: str | None = None,
    feature_source: str | None = None,
) -> None:
    """校验 schema 表填得能用；写错当场抛 ValueError

    与特征类别表的校验同一口径：表填错属于配置问题，应该在构造 agent（启动）时就报，
    而不是等跑到某条记录才吞掉、把错误混进结果里。

    校验项：
    1. 一级键是组名（非空字符串，且不含数字——schema 名是「组名 + 编号」，
       词里带数字会与别人的编号撞名）；二级键是编号（数字或它的字符串形式，同组内不重复）。
    2. 词条下可选的 words 是「这一类还覆盖哪些词形」的字符串列表（见 WORDS_KEY）：
       非空、无重复、每项都是非空且不含空白的字符串。
    3. **一个词形只能归一个词条**（按 {组名} ∪ words 反查，见 _group_words）：
       两组都收同一个词形时，收窄结果取决于表里的书写顺序，那不是判据。
       这一条对没写 words 的表是空操作（字典键本就不重复），不会影响既有配置。
    4. match 必须是「一组条件的对象」或「多组条件的列表」，取值是字符串或字符串列表。
    5. 给了 feature_keys（本次跑批的特征类别表里已填的类别）时，match 里的特征键必须
       都在其中：写错一个键不会报错，只会让这条图式永远匹配不上（整条静默失效），
       所以在这里挡住。

    words 与 match 一样是「写错了不报错、只是静默失效」的那类配置（词形写错就永远当不成
    关注词），所以与 match 同一口径，全部在这里挡住。

    Args:
        table (dict): 待校验的 schema 表
        feature_keys (Iterable[str] | None): 已填的特征类别 key；None 或空 = 不校验键
        source (str | None): 这张表的路径，只用于报错信息
        feature_source (str | None): 特征类别表的路径，只用于报错信息
    """
    if not isinstance(table, dict):
        raise ValueError('schema 表必须是一个对象，形如 {"schemas": {...}}')
    schemas = table.get("schemas") or {}
    if not isinstance(schemas, dict):
        raise ValueError('schema 表的 schemas 必须是对象：{"<关注词>": {"<编号>": {...}}}')
    known = {str(key) for key in (feature_keys or [])}
    owners: dict[str, list[str]] = {}

    for word, entries in schemas.items():
        if not isinstance(word, str) or not word:
            raise ValueError(f"schema 表的组名 {word!r} 必须是非空字符串（它是 schema 名的前一半）")
        if any(ch.isdigit() for ch in word):
            raise ValueError(
                f"组名 {word!r} 里不能有数字：schema 名是「组名 + 编号」，"
                f"「{word}1」会与编号撞名、也和别的词分不开"
            )
        if not isinstance(entries, dict):
            raise ValueError(f"组名 {word!r} 下要写成 {{编号: 条目}} 的对象（words 与编号同级）")
        for form in _validate_group_words(word, entries):
            owners.setdefault(form, []).append(word)
        numbers: set[str] = set()
        for number, entry in _numbered_entries(entries).items():
            num = str(number)
            if not num.isdigit():
                raise ValueError(f"图式 {word}{num} 的编号 {number!r} 要写数字（如 1、12）")
            if num in numbers:
                # json5 里 1 与 "1" 是两个不同的键，却指向同一个 schema 名
                raise ValueError(f"组名 {word!r} 下有重复编号 {num}（一个写成了数字、另一个写成了字符串？）")
            numbers.add(num)
            name = f"{word}{num}"
            if not isinstance(entry, dict):
                raise ValueError(f"图式 {name} 的条目要写成对象（description / match / examples）")
            if WORDS_KEY in entry:
                raise ValueError(
                    f"图式 {name} 里不该有 {WORDS_KEY}：它要写在组下、与编号同级，"
                    f"写在图式里没人读，等于没写"
                )
            _validate_schema_match(name, entry.get("match"), known, feature_source=feature_source)
            examples = entry.get("examples")
            if examples is not None and not isinstance(examples, (list, tuple)):
                raise ValueError(f"图式 {name} 的 examples 要写成列表（一个例句一项）")

    for form, groups in owners.items():
        if len(groups) > 1:
            raise ValueError(
                f"关注词词形 {form!r} 被多个词条同时收（{'、'.join(groups)}）："
                f"收窄时该取哪一组取决于表里的书写顺序，那不是判据；"
                f"请让它只出现在其中一组的 {WORDS_KEY} 里（组名本身恒属于该组）"
            )


def _validate_group_words(word: str, entries: dict) -> list[str]:
    """校验一个词条的 words（可省略），返回该组覆盖的全部词形（含组名本身）

    写成空列表与不写等价（两者都只覆盖组名），是徒增一处要维护的配置，所以也报出来——
    本仓库对「写了等于没写」的配置一律当场说清，不留给它静默躺着。
    """
    if WORDS_KEY not in entries:
        return _group_words(word, entries)
    listed = entries[WORDS_KEY]
    if not isinstance(listed, (list, tuple)):
        raise ValueError(
            f"组名 {word!r} 的 {WORDS_KEY} 要写成词形的列表（与编号同级），实际是 {listed!r}"
        )
    if not listed:
        raise ValueError(
            f"组名 {word!r} 的 {WORDS_KEY} 是空列表，与不写它等价（只覆盖组名自己），请删掉或填词"
        )
    seen: set[str] = set()
    for form in listed:
        if not isinstance(form, str) or not form or any(ch.isspace() for ch in form):
            raise ValueError(
                f"组名 {word!r} 的 {WORDS_KEY} 里每一项都要是非空且不含空白的字符串，实际有 {form!r}"
            )
        if form in seen:
            raise ValueError(f"组名 {word!r} 的 {WORDS_KEY} 里有重复词形 {form!r}")
        seen.add(form)
    return _group_words(word, entries)


def _validate_schema_match(name: str, match: Any, known: set[str], *, feature_source: str | None = None) -> None:
    """校验一个条目的 match（拆组刻意不共用 _schema_alternatives：那个函数会把写坏的项静默丢掉）"""
    if match is None:
        return
    if isinstance(match, dict):
        groups: list = [match]
    elif isinstance(match, (list, tuple)):
        groups = list(match)
    else:
        raise ValueError(f"图式 {name} 的 match 要写成「一组条件的对象」或「多组条件的列表」，实际是 {match!r}")
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError(f"图式 {name} 的 match 里每一项都要是一组条件（对象），实际是 {group!r}")
        for key, value in group.items():
            if known and str(key) not in known:
                raise ValueError(
                    f"图式 {name} 的 match 用了特征键 {key!r}，但它不在特征类别表"
                    f"（{feature_source or 'feature_classes.json5'}）的类别里"
                    f"（已有：{'、'.join(sorted(known))}）：这个键永远匹配不上，整条图式会静默失效"
                )
            if isinstance(value, str):
                continue
            if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                continue
            raise ValueError(f"图式 {name} 的 match.{key} 取值要写字符串或字符串列表，实际是 {value!r}")


def lookup_schema(
    features: dict, table: dict | None = None, focus_word: dict | None = None
) -> tuple[str | None, dict]:
    """按关注词 + 语义特征查 schema 表。

    schema 名是「组名 + 编号」（如「上12」「下1」），表按词条分组：
        {"schemas": {"<组名>": {"words": ["<词形>", ...],
                                "<编号>": {"description": "...", "match": [...]}}}}

    两级收窄，顺序固定：

    1. **先按关注词排除**别的词条的图式（见 _narrowed_groups）：「上12」与「下1」是两种不同的
       空间结构，别的词条的图式结构上就不可能是答案。词条写了 words 时，关注词是它列出的任一
       词形（外面/外边/外头）都算命中该词条，名字仍用组名（外1）——同一个名字空间，判定表
       因此只需按类写。关注词缺失时不排除。
    2. **再按特征匹配**：条目的 match 是**一组备选条件**，组内是「与」（列出的键都要命中，
       没列出的键不参与判断），组间是「或」。取收窄后第一条命中的图式（按表中书写顺序，
       所以特殊用法写在前面、宽泛的写在后面）。
    3. match 省略或为空的条目是兜底，只有同组其他图式都没命中时才采用。

    Args:
        features (dict): 该句的语义特征
        table (dict, optional): schema 表，默认读 config_path("schema_table") 解析出的那张
            （默认类型方位词，即 config_files/localization/schema_table.json5）
        focus_word (dict, optional): 上游传下来的关注词 {"index", "word", "kind", "source"}

    Returns:
        tuple[str | None, dict]: (schema 名或 None, 命中详情)。详情里始终带着
            candidates（本词的候选名单，未命中时正交给 LLM 从中挑）与 excluded
            （被关注词排除了几条，排查「收窄是不是收错了」的第一现场——经 words 归并进来的
            关注词这里会是 0：图式不是被排除，是经别名够到的）。
    """
    table = load_schema_table() if table is None else table
    candidates = schema_candidates(table, focus_word)
    detail: dict[str, Any] = {
        "focus_word": (focus_word or {}).get("word") or None,
        "candidates": list(candidates),
        "excluded": _schema_count(table) - len(candidates),
    }

    fallback: str | None = None
    for name, entry in candidates.items():
        groups, is_fallback = _schema_alternatives(entry)
        if is_fallback:
            fallback = fallback or name
            continue
        for group in groups:
            if _match_features(group, features):
                return name, {**detail, "source": "table", "matched": name, "match": group}
    if fallback:
        return fallback, {**detail, "source": "fallback", "matched": fallback, "match": {}}
    return None, {**detail, "source": "miss", "matched": None, "match": None}


def lookup_judgement(schema_a: str | None, schema_b: str | None, table: dict | None = None) -> tuple[bool | None, dict]:
    """按两个 schema 查判定表（CSV 矩阵，格式见 load_judge_table），判断两句意义是否相同。

    查到的格子决定下一步：√/× 直接出结论，○ 交 LLM，留空（非法）也如实交 LLM。
    schema 名为 None（没判出 schema）那一路按标签 未判出 查。表里没列出的 schema 名算未命中；
    整张表还没填（filled 为 0）也一律算未命中——空模板不该让每条记录都背一句「这格非法留空」。

    容错刻意写在这里而不是 load_judge_table 里：表是可以注入的（构造 JudgeAgent 时给 table=），
    注入的裸表 {"cells": ...} 或 {} 不经过读表那一步，直接落到这里。

    table 不给就按**默认语料类型**（方位词）读一张：这只是给临时脚本行个方便，跑批路径上一律
    由 agent 传自己那张（JudgeAgent 构造时按 corpus_type 读好存进 self.table），
    所以这里没有 corpus_type 参数——别在这里省 table，那样另一类语料会静默拿到方位词表的结论。

    Returns:
        tuple[bool | None, dict]: (判定结果或 None, 查表详情)。None 表示没有直接结论。
        详情里的 source 把查表结果分成四类，后三类都要 LLM 兜底：
            "table"   格子是 √/×，same 即结论
            "defer"   格子是 ○，交 LLM 做进一步判断
            "miss"    schema 名不在表里，或整张表还没填（空模板）
            "illegal" 格子留空/符号认不出（非法），降级交 LLM
    """
    table = load_judge_table() if table is None else table
    cells = table.get("cells") or {}
    labels = table.get("labels") or []
    label_a, label_b = _judge_label(schema_a), _judge_label(schema_b)
    detail = {"a": schema_a, "b": schema_b, "label_a": label_a, "label_b": label_b, "cell": None}

    symbol = cells.get((label_a, label_b))
    if symbol is None:
        # 表是对称的，读表时两把键都会注册（load_judge_table），但注入的表未必：
        # 只写了 (里1, 外1) 一格时，(外1, 里1) 也该查到它，不该滑成未命中。
        symbol = cells.get((label_b, label_a))
    if symbol == JUDGE_CELL_SAME:
        return True, {**detail, "source": "table", "cell": symbol}
    if symbol == JUDGE_CELL_DIFFERENT:
        return False, {**detail, "source": "table", "cell": symbol}
    if symbol == JUDGE_CELL_LLM:
        return None, {**detail, "source": "defer", "cell": symbol}
    # 没有这一格：两侧标签都在表里、表也填过 -> 这一格是作者漏填的非法空格；否则只是未命中
    if (table.get("filled") or 0) and label_a in labels and label_b in labels:
        return None, {**detail, "source": "illegal"}
    return None, {**detail, "source": "miss"}


# --------------------------------------------------------------------------
# 六、结果落盘
# --------------------------------------------------------------------------


def build_output_path(out_dir: str | Path, prefix: str = "spatial_pairs") -> Path:
    """生成带时间戳的输出文件路径，并确保目录存在"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"{prefix}_{stamp}.jsonl"


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    """把结果记录逐行写成 jsonl，返回写入的条数"""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def append_jsonl(record: dict, path: str | Path) -> None:
    """追加一条结果记录（流式落盘，供长跑时中途查看进度）"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# 七、供 LLM agent 调用的工具
# --------------------------------------------------------------------------


def parse_sentence_text(sentence: str) -> str:
    """对句子做分词、词性标注与依存句法分析，返回文本形式的句法证据

    每个 token 前面都打了它从 0 开始的下标（`3:进/v`）：一是让「第 N 个词」这种说法能直接核对，
    二是分词行按空格分隔，token 本身可能含空白（实测 location 语料的 token 里约 1.2% 带尾空格，
    如 `": "`），照着空格数会整体数错一位。
    """
    cws = _cache.get_task_value(sentence, config.CWS)
    pos = _cache.get_task_value(sentence, config.POS)
    dep = _cache.get_task_value(sentence, config.DEP)

    dep_lines = []
    for idx, (head, label) in enumerate(zip(dep["head"], dep["label"])):
        # LTP 的 head 是从 1 开始的下标，0 表示指向根节点（核心谓词）
        if head == 0:
            dep_lines.append(f"  {idx}:{cws[idx]} —{label}—> ROOT")
        else:
            dep_lines.append(f"  {idx}:{cws[idx]} —{label}—> {head - 1}:{cws[head - 1]}")

    return "\n".join(
        [
            f"分词/词性: {' '.join(f'{i}:{w}/{p}' for i, (w, p) in enumerate(zip(cws, pos)))}",
            "依存句法:",
            *dep_lines,
        ]
    )


def prefill_ltp(sentences: Iterable[str], chunk_size: int | None = None) -> int:
    """批量预填充句法分析缓存

    在跑 agent 之前把整批句子一次性喂给 LTP：一来避免每个句子首次调用时单句前向
    （批量推理快约 4 倍，见 sentenceTable.py），二来让后续多线程跑 agent 时
    不再有并发调用同一份 LTP 模型的窗口。返回实际推理的句子数。
    """
    return _cache.prefill(sentences, chunk_size=chunk_size)


def cache_ready(sentence: str) -> bool:
    """该句的句法分析结果是否都已在缓存里（预填充后的自检用）

    推理是在锁外做的，所以"已经预填充过"这件事必须能查，不能只是假设。
    """
    return _cache.has_base_tasks(sentence)


@tool
def parse_sentence(sentence: str) -> str:
    """对句子做中文分词、词性标注和依存句法分析，返回结构化的句法证据。

    每个 token 前面是它从 0 开始的下标（`3:进/v`），与「第几个词」这种说法可以逐位对上。
    判断句子的语义特征或比较两句意义时，应当先调用本工具查看真实句法结构，
    尤其是确认方位词/趋向动词与哪个成分搭配、承担什么句法角色。

    Args:
        sentence (str): 待分析的单个句子

    Returns:
        str: 分词（词/词性）、依存句法（词 —关系—> 中心词）
    """
    return parse_sentence_text(sentence)


@tool
def lookup_schema_tool(features_json: str, focus_word: str = "") -> str:
    """按语义特征查 schema 表（纯查表，不含 LLM 判断）。

    Args:
        features_json (str): 该句的语义特征，JSON 对象字符串，如 {"area": ["[-容器内]"], "distance": ["[+近]"]}
        focus_word (str): 本次需要关注的方位词/趋向动词；给了就只在收它的那个词条里找，
            词条写了 words 时，写任一列出的词形（外面/外边）都算收进同一个词条（见 lookup_schema）

    Returns:
        str: 命中时返回 schema 名，未命中返回 "未命中"
    """
    try:
        features = json.loads(features_json)
    except json.JSONDecodeError as e:
        return f"features_json 不是合法 JSON：{e}"
    name, detail = lookup_schema(features, focus_word={"word": focus_word} if focus_word else None)
    return name if name else f"未命中（{detail.get('source')}）"


if __name__ == "__main__":
    # 纯规则部分的冒烟测试：不调 LLM，也不需要 GPU
    demo_slice = ["在", "硬件", "上", "说", ",", "它", "将", "涉及", "到", "参数", "设置", "。"]
    demo_info = {"index": 2, "word": "上", "children": [{"index": 1, "word": "硬件"}]}
    demo_rules = {"groups": [["上", "中", "里"], ["前", "后"]]}

    # 关注词：两份词表 + 三级回落（sampled_word -> 按词表 walk match_tree -> match_tree[0]）
    print(f"关注词表：{ {kind: len(words) for kind, words in load_focus_vocab().items()} }")
    demo_records = [
        # 抽样的语料：sampled_word 直接给答案（趋向动词在子节点里，根节点是搭配动词 走）
        {
            "sentence": ["他", "走", "进", "村子", "。"],
            "filter_methods": ["root", "manner_verb_one_complement"],
            "match_tree": [{"index": 1, "word": "走", "children": [{"index": 2, "word": "进"}]}],
            "sampled_word": "进",
        },
        # 未抽样的语料：按词表 walk（方位词就是根节点）
        {
            "sentence": ["在", "硬件", "上", "说", "。"],
            "filter_methods": ["root", "locality_phrase"],
            "match_tree": [{"index": 2, "word": "上", "children": [{"index": 1, "word": "硬件"}]}],
        },
        # 一个目标词都没匹配上：回落到 match_tree[0] 的搭配动词，此时 kind 是空串（不猜）
        {
            "sentence": ["他", "带", "着", "书", "。"],
            "filter_methods": ["root", "accompany_verb_two_complement"],
            "match_tree": [{"index": 1, "word": "带", "children": [{"index": 3, "word": "书"}]}],
        },
        # 同一个词在搭配动词前后各出现一次：取排在动词之后的那个（族收窄挡不住这一类）
        {
            "sentence": ["然后", "我们", "在", "上", "撒", "上", "盐", "。"],
            "filter_methods": ["root", "cause_verb_one_complement"],
            "match_tree": [{"index": 4, "word": "撒", "children": [{"index": 3, "word": "上"}, {"index": 5, "word": "上"}]}],
        },
    ]
    for record in demo_records:
        print(f"  {join_slice(record['sentence'])} -> {pick_focus_word(record)}")
    print(f"  破平局（「上」两张表都收）：locality 规则 {focus_kind('上', ['locality_phrase'])}；"
          f"direction 规则 {focus_kind('上', ['direction_verb_as_center'])}")

    print(f"原句: {join_slice(demo_slice)}")
    print(f"含「上」的词组: {json.dumps(find_word_groups('上', demo_rules), ensure_ascii=False)}")
    print(f"含「里面」的词组: {find_word_groups('里面', demo_rules)}")

    new_slices, detail = alter_sentence(demo_slice, demo_info, demo_rules)
    print(f"待选句 {len(new_slices)} 个：")
    for index, (candidate, new_slice) in enumerate(zip(detail["candidates"], new_slices)):
        print(
            f"  [{index}] {join_slice(new_slice)}"
            f"（{candidate['replacement']}，来自 {candidate['source']}：{'、'.join(candidate['group'])}）"
        )

    # 显式指定替换词时只出一个待选句，便于验证规则表外的用法
    new_slices, detail = alter_sentence(demo_slice, {**demo_info, "replacement": "里"}, demo_rules)
    print(f"指定替换词后: {[join_slice(s) for s in new_slices]}（applied={detail['applied']}）")

    # 空表（显式传入）不得凭空造出待选句——默认那张表读的是 config_files/localization/
    # alter_rules.json5，本地已填内容，跑它验不出空表行为
    new_slices, detail = alter_sentence(demo_slice, demo_info, {})
    print(f"空表（显式传入）: 待选句 {len(new_slices)} 个（applied={detail['applied']}）：{detail['reason']}")

    # schema 查表：先按关注词收窄、再按特征匹配（多组条件是「或」，多值特征是「任一」）
    configured = load_schema_table()
    print(
        f"  实际配置（{SCHEMA_TABLE_FILE.parent.name}/{SCHEMA_TABLE_FILE.name}）："
        f"{len(schema_candidates(configured))} 条图式，"
        f"{len(configured.get('schemas') or {})} 个词条"
    )
    demo_table = {
        "schemas": {
            "外": {
                # 词形写在自己这一组的 words 里（与编号同级）：外面/外边/外头 都归「外」这一类，
                # 名字仍用组名（外1），判定表因此只需按类写一份
                "words": ["外", "外面", "外边", "外头"],
                "1": {
                    "description": "目标物在参照物外部、与之贴近或邻近",
                    "match": [
                        {"area": ["[-容器内]"], "distance": ["[+紧连]"]},
                        {"area": ["[-容器内]"], "distance": ["[+近]"]},
                    ],
                },
                "2": {
                    "description": "目标物在参照物外部、相距很远",
                    "match": [{"area": ["[-容器内]"], "distance": ["[+远]"]}],
                },
            },
            "上": {"1": {"description": "方位短语表处所", "match": {"area": ["[+边界内]"]}}},
        }
    }
    near = {"area": ["[-容器内]"], "distance": ["[+近]"]}
    multi = {**near, "distance": ["[+近]", "[+远]"]}
    print(f"  「外」+ 距离[+近] -> {lookup_schema(near, demo_table, {'word': '外'})[0]}")
    print(f"  「外」+ 多值特征 距离[+近,+远] -> {lookup_schema(multi, demo_table, {'word': '外'})[0]}")
    print(f"  「上」+ 同一组特征（被关注词排除）-> {lookup_schema(near, demo_table, {'word': '上'})[1]}")
    print(f"  「外面」经 words 归入「外」-> {sorted(schema_candidates(demo_table, {'word': '外面'}))}"
          f"，组名 {schema_group_key(demo_table, {'word': '外面'})}")
    print(f"  「桌面」不在任何 words 里 -> {schema_candidates(demo_table, {'word': '桌面'})}"
          f"，组名 {schema_group_key(demo_table, {'word': '桌面'})}")
    print(f"  没有关注词时不收窄 -> {lookup_schema({**near, 'area': ['[+边界内]']}, demo_table)[0]}")
    judge_rows = [
        ["", "位移-进入", "位移-离开", "未判出"],
        ["位移-进入", "", "", "○"],
        ["位移-离开", "", "", ""],
        ["未判出", "", "○", ""],
    ]
    judge_table = _judge_table_from_rows(judge_rows)
    print(f"  判定表：{judge_table['labels']}，已填 {judge_table['filled']} 格，问题 {judge_table['problems']}")
    for a, b in (("位移-进入", "位移-进入"), ("位移-进入", "位移-离开"), ("未判出", "未判出"), ("位移-离开", "未判出"), ("位移-离开", "位移-离开")):
        same, detail = lookup_judgement(a, b, judge_table)
        print(f"    {a} × {b} -> same={same} source={detail['source']} cell={detail['cell']!r}")
