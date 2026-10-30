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

# ---- 方位词后缀切分（修正 LTP 把「名词+方位词」并成一个 token）----
LOCATIVE_SPLIT_ENABLED = True
"""是否启用方位词后缀切分。默认开启。

开启后，LTP 输出里以方位词结尾、词性为 nl 的 token 会在写缓存前被切开，
并用「已分词」模式重跑 pos/dep/sdp。筛选阶段与空间 agent 阶段的 CWS/POS/DEP/SDP
产物都会随之改变——索引会位移，所以已有产物必须整批重跑，不能增量续跑。

（说明与代码对齐：常量是 True，允许切分的词性只有 LOCATIVE_SPLIT_ALLOWED_POS 里的
nl 一项。旧注释写「默认关闭」「n/nl」，会让人按「没开切分」或「n 也能切」去理解
已有产物的下标口径，故改正。）"""

LOCATIVE_SPLIT_VOCAB_FILE = "config_files/localization_vocab.json5"
"""后缀词表。相对路径按 _PROJECT_ROOT 解析，保证 spawn 出的子进程读到同一份文件。"""

LOCATIVE_SPLIT_ALLOWED_POS = frozenset({"nl"})
"""允许切分的词性。nl 正是并词的主要标签（树上/墙上/门外），但也是误切的主要来源。"""

LOCATIVE_SPLIT_SUFFIX_EXCLUDE = frozenset({"东", "西", "南", "北", "内", "左", "右"})
"""从后缀集合里剔除的单字：这些字作后缀时几乎总是构词成分，而不是独立的方位词。

西/东/南/北 → 避免「东西」（n，事物）被切成 东+西、「南北」同理；
内 → 避免 国内/体内/室内/室外 被切开；
左/右 → 避免「左右」被切开。
代价是 城东/村西/校内 这类少见形式不再切分。开启后最该按真实语料调的就是这一项。"""

LOCATIVE_SPLIT_TOKEN_BLACKLIST = frozenset({
    # 词典化名词：切成方位词是错的
    "东西", "海外", "意外", "左右",
    # 以「内」构词（「内」已被排除，这里是双保险）
    "国内", "体内", "室内", "室外",
    # localization_vocab.json5 注释点名的「LTP 倾向整体标注」案例：
    # 先保守不切，跑真实语料后再逐条裁定是否要拆出方位词
    "身上", "脸上", "心里", "嘴里", "手上", "脚下",
    "乡下", "地下", "手下", "心中", "背后",
})
"""整词命中即不切分。"""

TOKENIZER_NAME = "ltp"
"""分词与句法分析的引擎名，与 LTP_MODEL_PATH 一并记进产物，用来判断旧产物能否复用"""


def segmentation_signature() -> dict:
    """本次分词口径的签名：引擎、模型路径、后缀切分开关与词表

    后缀切分会改变 tokens 与下标（见 LOCATIVE_SPLIT_ENABLED），所以**切分配置变了，旧产物
    的下标一律不可复用**。把这几项写进运行汇总，事后才能判断某份产物是哪个口径跑出来的、
    要不要整批重跑——光看「索引对不上」分不清是代码改了还是配置改了。

    Returns:
        dict: 可直接 json 序列化的签名；值一律取字符串或有序列表，便于与历史产物逐字段比。
    """
    return {
        "tokenizer": TOKENIZER_NAME,
        "model_path": LTP_MODEL_PATH,
        "locative_split_enabled": LOCATIVE_SPLIT_ENABLED,
        "locative_split_vocab_file": LOCATIVE_SPLIT_VOCAB_FILE,
        "locative_split_allowed_pos": sorted(LOCATIVE_SPLIT_ALLOWED_POS),
    }


# ---- 模型请求策略：超时、重试与预算 ----
# 三者集中在配置里，是为了让「这一轮跑批允许花多少」成为可记录、可复现的口径，
# 而不是分散在 SDK 默认值与各 agent 的 try/except 里。
REQUEST_TIMEOUT_SECONDS = 120
"""单个模型请求的超时秒数

底层 SDK 自己有默认超时，但那条默认值不进运行清单、也无法按实验调整；这里显式给一个，
并由 agents.get_model 传下去。超时后由 retry.call_with_retry 按下面的重试参数补发。"""

LLM_MAX_RETRIES = 6
"""单个模型请求在**首次之外**最多重试几次（默认 6，即最多 7 次请求）

只重试超时、连接失败、限流与可恢复的 5xx（见 retry.RETRYABLE_STATUS_CODES）；
401/403/400/404/422 这类配置错误不重试。模型配置里写了 "max_retries" 时以模型配置为准。
"""

LLM_RETRY_BACKOFF_SECONDS = 1.0
"""首次重试前的等待秒数；之后按 2 的幂指数增长"""

LLM_RETRY_BACKOFF_MAX_SECONDS = 60.0
"""单次重试等待的上限（含指数增长后的封顶值）"""

LLM_RETRY_JITTER_SECONDS = 0.5
"""每次退避额外叠加的随机抖动上限，避免多个模型/候选同时重试把限流打得更死"""

VOTING_MODEL_WORKERS = 0
"""多模型投票时同时跑几个模型；0 = 配置了几个模型就同时跑几个

每个模型内部仍按 --workers 并发处理记录，所以总线程数约为
``模型数 × --workers``。模型调用是 IO 密集的，Linux 128 逻辑核下这样用是预期的；
若接口侧有限流，可把本项调小，或同时调小 --workers。
"""

CANDIDATE_MAX_REQUESTS = 8
"""一条候选最多发几次模型请求（0 表示不限）

一条语料会按替换词展开出多个候选，单个候选的失败重试会被候选数放大成整批的倍数。
预算按候选作用域计（一条 graph.invoke = 一条候选，a/b 两侧与重试共用这一份，见
spatial_agents.BaseAgent.budget_scope）；超预算的候选不再继续问，如实记为失败并转入复核
（见 spatial_agents.RequestBudgetExceeded）。"""

BATCH_MAX_REQUESTS = 0
"""整批最多发几次模型请求（0 表示不限）

所有 agent 共用一份计数，不受候选作用域影响（见 spatial_agents.BatchRequestBudget）。
达到上限后余下的调用一律不再发请求，记录照常落盘、admission.review_required 记 True，
留给人工复核。候选预算管「一条候选内别被重试放大」，这条管「候选乘记录别没有上限」。"""

# FILTER_WORKERS = 0
FILTER_WORKERS = 1
"""筛选阶段进程数，0 表示自动使用 CPU 核数。

实测单张 GPU 上开 2 个进程仅比单进程快约 1.2 倍，却要多一份 LTP 模型副本
（约 0.66GB 显存）和约 5s 的进程启动开销；配合 LTP 批量推理后单进程已能打满单卡，
故默认为 1。多进程分支仍保留在 entry_in_linux.py 中，需要时改回即可。"""

EXTRACT_WORKERS = 4
"""文本切句阶段的并发线程数（0 = 自动使用 CPU 核数，最多 8）

这一阶段只做 JSONL 读取、StnSplit 与空白切分，不碰 GPU；用线程池是因为切句是轻量调用，
线程比进程省掉重复 import 与模型副本。默认 4 已经能把 I/O 与切句的重叠做起来；
原始语料很小或磁盘很慢时可调小。"""

EXTRACT_PREFETCH_ITEMS = FILTER_BATCH_SIZE * 2
"""切句预取队列最多囤多少个句子（留给筛选消费的缓冲）

预取线程与规则筛选并行：生产者只比消费者最多快这么多条，避免大语料下把整份语料读进内存。
太小会让两边频繁互相等待，太大则失去内存上界。"""

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