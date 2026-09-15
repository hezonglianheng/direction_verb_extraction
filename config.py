# encoding: utf8

"""
配置文件
"""

import os
from pathlib import Path

SENTENCE_LEN_LOWER_BOUND = 10
"""句子长度下界，单位为字符数"""
SENTENCE_LEN_UPPER_BOUND = 1000
"""句子长度上界，单位为字符数"""

FILTER_BATCH_SIZE = 1000
"""筛选阶段每个批次包含的句子数量"""

# 必须 >= FILTER_BATCH_SIZE：筛选前会把整批句子的 LTP 结果预填充进缓存，
# 容量不足时 LRU 会在读取之前就把本批淘汰掉，导致每句重跑一次模型。
CACHE_MAX_SIZE = FILTER_BATCH_SIZE * 2
"""句子任务缓存的最大容量，超过这个容量时会删除最旧的缓存项"""

LTP_BATCH_CHUNK = 32
"""LTP 批量推理时每组的句子数量。

同一组内会先按长度排序再送入模型：ltp.pipeline 使用 padding="longest"，
组内混入长句会把所有句子都填充到该长度（实测 64 句里混入一句 192 字长句慢 10.8 倍）。
32 在真实语料的长度分布下略优于 64。"""

# FILTER_WORKERS = 0
FILTER_WORKERS = 2
"""筛选阶段进程数，0 表示自动使用 CPU 核数"""

_PROJECT_ROOT = Path(__file__).resolve().parent
"""项目根目录（config.py 所在目录），与运行时 cwd 无关"""

# 优先级：环境变量 > 项目根目录下的本地模型
# 环境变量可填本地目录的绝对路径，也可填 HuggingFace 模型名（如 "LTP/small"）。
# 注意：子进程（spawn）会重新 import 本模块，因此运行期修改该属性不会传播到子进程，请用环境变量。
LTP_MODEL_PATH = os.environ.get("LTP_MODEL_PATH") or str(_PROJECT_ROOT / "LTP_model" / "base2")
"""LTP 模型加载路径。默认加载项目根目录下的 base2 基础模型（LTP_model 在 .gitignore 中，
故新克隆的仓库需自行将模型放置于 config.py 同级目录），如需加载其他本地或线上模型，
可修改此值或设置 LTP_MODEL_PATH 环境变量。必须是 str：LTP() 内部会调用 .split("@")，不支持 Path。"""
# LTP支持的任务：分词 cws、词性 pos、命名实体标注 ner、语义角色标注 srl、依存句法分析 dep、语义依存分析树 sdp、语义依存分析图 sdpg
CWS = "cws"
"""分词"""
POS = "pos"
"""词性标注"""
NER = "ner"
"""命名实体标注"""
SRL = "srl"
"""语义角色标注"""
DEP = "dep"
"""依存句法分析"""
SDP = "sdp"
"""语义依存分析树"""
SDPG = "sdpg"
"""语义依存分析图"""