# encoding: utf8

"""
配置文件
"""

SENTENCE_LEN_LOWER_BOUND = 10
"""句子长度下界，单位为字符数"""
SENTENCE_LEN_UPPER_BOUND = 1000
"""句子长度上界，单位为字符数"""
# CACHE_MAX_SIZE = 10000
CACHE_MAX_SIZE = 1000
"""句子任务缓存的最大容量，超过这个容量时会删除最旧的缓存项"""

FILTER_BATCH_SIZE = 1000
"""筛选阶段每个批次包含的句子数量"""

# FILTER_WORKERS = 0
FILTER_WORKERS = 2
"""筛选阶段进程数，0 表示自动使用 CPU 核数"""

LTP_MODEL_PATH = r"LTP_model\base2"
"""LTP 模型加载路径，默认从本地加载基础模型，如果需要加载本地或线上其他模型，请修改路径"""
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