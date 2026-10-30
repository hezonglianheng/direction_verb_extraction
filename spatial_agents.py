# encoding: utf8

"""空间语义改写-判定用到的七个 agent，每个都独立构造、独立调用。

    reality_agent   输入一个句子与它的关注词，判断这个词是不是用于表达现实世界中的空间场景
                    （is_spatial_scene）。全流程的第一道闸门：判为 False 的记录就此收尾，
                    引申用法（会上/理论上）、时间用法（三天内）、时体与结果用法
                    （笑起来/看出来）、比喻用法（走出困境）都不再往下跑。
    alter_agent     输入切片（list[str]）与关注词（dict，含 index/word/kind），按 index 定位
                    该词，用规则表里收它的那组 words 对应的 replacements 逐一替换，产出一批待选句
                    （一个词能换几个词就有几个）。纯规则、不调 LLM（实现在 spatial_tools.alter_sentence）。
    semantic_agent  输入一个句子，分别判断语法是否通顺（grammar_ok）与空间语义是否异常
                    （spatial_ok）：两项在同一次调用里给出、各带一段理由，合出来的
                    is_fluent 由代码算（见 semantic_verdict_result）。
    feature_agent   输入一个句子，输出语义特征。默认单次调用直接产出全部特征字段；
                    也可建 deep agent（deep=True，每类特征一个 subagent、主 agent 汇总），
                    供质量对照用，见 FeatureAgent。
    schema_agent    输入一个句子、它的语义特征与关注词：先按关注词排除别的词条的图式（一个词条
                    可用 words 覆盖一类词形，外面/外边 都归「外」）、再按特征查表；都未命中时
                    才调 LLM，且只让它在该类的候选图式里挑一个最接近的（这一类下一条图式都没有
                    时才退化为自由命名，名字也按组名起）。
    judge_agent     输入两个句子及其 schema，先查判定表（CSV 矩阵）；表里直接给了结论（√/×）
                    就用它，标了 ○、格子留空非法、或这一对没收录进表（未命中）才交 LLM 判断。
    synonym_agent   输入两个句子与两侧各自的关注词，直接由独立 LLM 判断是否同义；
                    不抽特征、不查 schema 或判定表。system_prompt/synonym.txt 由用户填写，
                    正文未填写时不调模型，返回 same_meaning=None、source=unfilled_prompt。

七个 agent 的输入里都有关注词（focus_word 或成对的 focus_word_a / focus_word_b）：
**本记录要替换的那个位置**上的方位词/趋向动词（index、
词本身、属哪一类），由 spatial_tools.pick_focus_words 从语料记录里取（一家语料可能有好几个
位置，每个位置各出几条记录，一条记录只认一个），图从第一个 agent 开始逐级往下传。
它是 user message 的第一行（措辞见 focus_line），六个 LLM agent 都按它展开判断——一句里有多个
方位词/趋向动词时，模型否则无从知道该看哪一个。

一对句子的**两侧各看各的词**：a 侧（原句）看语料记录里那个词，b 侧（改写句）看规则表换上去的
那个词（spatial_tools.replacement_focus_word，来源记 "alter"）。措辞按 source 自动分两侧
（见 focus_role）：原句那侧说下标在原句的分词结果里；改写句那侧只说下标是**原句**上的位置，
并交代改写句要重新分词、不能拿这个下标去数（下标是照搬 a 侧的，见 FOCUS_INDEX_NOTE_ALTER）。
所以 semantic / feature / schema 三步在 a/b 两侧各跑一次时，各自收窄到的是自己那个词所属的词条
——b 侧的 schema 因此落在替换词那一组（判定表就是按两组的名字写的，见 spatial_tools）。

本模块只定义 agent：怎么接成图、单句还是成对、输出什么形状，都属于图的事，
见 spatial_graphs.py；跑批与落盘见 spatial_runner.py。所以这里不 import langgraph。

每个 agent 只依赖自己那张配置表，构造参数都带默认值（默认读默认语料类型「方位词」那一组的
标准路径，见 spatial_tools.config_path），因此既能被图按需建，也能单独建来做对照实验：

    from spatial_agents import SemanticAgent
    focus = {"index": 2, "word": "进", "kind": "趋向动词", "source": "vocab"}
    SemanticAgent().run("他走进屋里。", focus)              # 只看语义通顺性
    JudgeAgent().run(a, b, sa, sb, fa, fb, focus)          # 只做判定

跑批会把语料类型经 build_agents(corpus_type=...) **构造期**注入每个 agent：四张配置表按类型
分成 config_files/localization/（方位词）与 config_files/direction/（趋向动词）两组，
agent 据此选目录；同一条信息也会写进 user message（见 head_block），因为模型要按本类词的表
来判断（schema 名与特征类别都与之对应）。类型不在构造后改——它是跑批级常量。

依赖关系：spatial_runner → spatial_graphs → spatial_agents → spatial_tools / agents。

每个 agent 的 system_prompt 正文外置在 system_prompt/ 下，构造 agent 时读取（见 utils.read_prompt；
含 {{占位符}} 的模板由 utils.render_prompt 渲染）。唯一的例外是 reality_agent：它的判据按语料
类型分，那份提示语与四张表同目录（config_files/<类型目录>/reality.txt，见 PROMPT_SOURCE_CONFIG）。

用法：
    .venv/bin/python spatial_agents.py --list        # 打印七个 agent 的输入输出契约，不调模型
    .venv/bin/python spatial_agents.py               # 纯规则冒烟测试（不调模型，不需要 GPU）
"""

import argparse
import contextvars
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterable

from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field, create_model

import agents as agent_factory
import spatial_tools as st
import utils

DEFAULT_MODEL = "deepseek-flash"
"""默认模型配置名（model_config/api_keys.json 里的键）"""

_FIRST_ERROR_CHARS = 200
"""调用失败时留给第一个例外 repr 的字符数（见 BaseAgent._invoke）

与 _reply_digest 同一个口径：够认出是哪一类失败（403 区域限制、超时、连不上……），
又不至于把整个响应头搬进汇总。完整的那份仍在各条记录的 status_reason 里。
"""


class RequestBudgetExceeded(RuntimeError):
    """请求预算用完，这一问不再发出（A3）

    预算分两层，各防一件事（见 config.CANDIDATE_MAX_REQUESTS 与 config.BATCH_MAX_REQUESTS）：
    候选预算防「同一问按候选数放大重试」，整批预算防「候选乘记录没有上限」。达到上限时
    **抛出而不是无限重试**：各 agent 的 run() 本来就兜异常，于是这条记录如实记成 error/未知
    并转入人工复核（见 spatial_graphs.pair_admission 的 review_required），而不是静默吞掉
    一批请求费。

    它与接口失败的区别只在原因，不在处置：两者都进同一条复核通道，记录里都看得见。
    """


class BatchRequestBudget:
    """整批请求预算：所有 agent 共用一份，按「已发出的调用数」计（见 config.BATCH_MAX_REQUESTS）

    与候选预算（BaseAgent.max_requests + budget_scope）分工不同：候选预算是有作用域的，
    只在一段 graph.invoke（= 一条候选状态）内累计；整批预算没有作用域，跑批从头到尾一份。
    两者都在**发请求之前**检查，且与「占名额」在同一把锁里完成，多个 worker 同时抢最后
    一个名额时不会一起越过去。
    """

    def __init__(self, max_requests: int | None = None):
        # 与 BaseAgent.max_requests 同一套口径：0 或负数表示不限
        self.max_requests = None if not max_requests or max_requests < 0 else int(max_requests)
        self.requests = 0  # 已发出的调用数（成功的、失败的都是发出去的）
        self.exceeded = 0  # 因整批预算而未发出的调用数
        self._lock = threading.Lock()

    def reserve(self, agent_name: str) -> None:
        """占一个整批名额；没有名额了就抛 RequestBudgetExceeded（调用方负责记账）"""
        with self._lock:
            if self.max_requests is not None and self.requests >= self.max_requests:
                self.exceeded += 1
                raise RequestBudgetExceeded(
                    f"整批已达请求预算 {self.max_requests} 次，本次不再发请求（{agent_name} 这一问未发出）"
                )
            self.requests += 1

    def release(self) -> None:
        """回退一次已占用的整批名额（候选预算随后拒绝时用，避免白扣整批预算）"""
        with self._lock:
            if self.requests > 0:
                self.requests -= 1


class _BudgetScope:
    """一条候选的预算作用域：按 agent 实例各记一份已发出的调用数（见 BaseAgent.budget_scope）

    计数挂在**一个跨线程共享的对象**上，而不是线程本地变量：langgraph 的同步节点并不在
    调用 graph.invoke 的那个线程里跑，而是由 BackgroundExecutor 提交到线程池（它只复制
    contextvars，不复制 threading.local）。作用域本身用 ContextVar 下发（见 _CURRENT_BUDGET），
    复制到节点线程的上下文里的是这个对象的引用，于是各节点对计数的增减彼此可见。
    同一候选的几个节点也可能并行跑，所以取名额要过 _lock。
    """

    def __init__(self):
        self.requests: dict[object, int] = {}
        self._lock = threading.Lock()

    def reserve(self, agent, limit: int | None) -> bool:
        """占一个候选名额；limit 为 None 表示不限。返回 False 表示这条候选已超预算"""
        if limit is None:
            return True
        with self._lock:
            used = self.requests.get(agent, 0)
            if used >= limit:
                return False
            self.requests[agent] = used + 1
            return True

    def release(self, agent) -> None:
        """回退一次候选名额（整批预算随后拒绝时用，避免白扣候选预算）"""
        with self._lock:
            used = self.requests.get(agent, 0)
            if used > 0:
                self.requests[agent] = used - 1


_CURRENT_BUDGET: contextvars.ContextVar[_BudgetScope | None] = contextvars.ContextVar(
    "spatial_agents_budget_scope", default=None
)
"""当前线程/节点所处的那条候选的预算作用域（None = 不在任何候选作用域内）"""


class StructuredOutputMissing(RuntimeError):
    """模型这一轮没给出结构化输出，拿不到可用的判定

    这不是「调用失败」，但在本框架里必须当成失败抛出来：langchain 的 ToolStrategy 只在模型
    这一轮**带了 tool call** 时才解析结构化输出，模型只回一段自然语言时它既不报错也不重试，
    而是让子图正常结束、把 structured_response 显式写成 None（见 factory 里 _build_commands
    的 has_structured_output 分支）。也就是说「拿不到判定」是一条**正常返回**的路径。

    而各 agent 的 run() 只兜异常（try 包住 _invoke，返回值却在 try 之外解引用），真让
    _invoke 返回 None 的话 except 永远等不到：AttributeError 会穿透图节点、经 pool.map
    冒回跑批入口，一条记录出事带走整批（见 BaseAgent._invoke）。所以在 _invoke 这一处把它
    转成异常，各个 agent 既有的兜底路径就都生效了——reality 记 error 后放行、semantic 记
    error 后被 semantic 闸门拦下、schema/judge 记 error，各自的下游都早已按 None 处理。
    """


def _reply_digest(messages: list) -> str:
    """模型最后一条回复的摘要，用作 StructuredOutputMissing 的说明

    取结构化输出的正常路径下，最后一条消息就是那次工具调用（content 为空、tool_calls 非空）；
    出岔子时它是模型用自然语言写的一段话——正常路径下这句话落不了盘（落盘的只有解析出来的
    字段），所以出岔子时它是事后唯一能查的东西，值得随异常一起记进记录。截断到 200 字：
    够认出「它在用自然语言作答」，又不至于把整段推理塞进每条记录。
    """
    if not messages:
        return "模型一条消息都没回"
    last = messages[-1]
    calls = [call.get("name") for call in (getattr(last, "tool_calls", None) or [])]
    text = " ".join(str(getattr(last, "content", "") or "").split())
    if len(text) > 200:
        text = text[:200] + "…"
    return f"最后一条回复：工具调用 {calls or '无'}，正文 {text or '（空）'}"


# --------------------------------------------------------------------------
# 一、结构化输出的结果模型
# --------------------------------------------------------------------------


class SemanticVerdict(BaseModel):
    """semantic_agent 的输出：语法与空间语义**分开**判，两项各带自己的理由

    两项判定在同一次调用里给出（多两个字段，不是多一次请求）。合出来的 is_fluent 不在这里：
    那是代码算的（见 semantic_verdict_result）——让模型再多吐一个能从前两项推出的布尔，只会多
    一处自相矛盾的可能（grammar_ok=False 而 is_fluent=True 这种），还要再定谁权威。
    """

    grammar_ok: bool = Field(
        description="该文本的语法是否通顺：成分是否完整、搭配与语序是否合法、虚词是否用得对。"
        "这一项**只看语法**，与句子表达的空间关系合不合理无关——空间再离谱，只要合乎语法即为 true"
    )
    grammar_reason: str = Field(description="语法这一项的判断依据：哪里不合语法，或为什么没问题")
    spatial_ok: bool = Field(
        description="该文本的空间语义是否无异常：描述的空间关系、方向、位置、位移是否自洽，"
        "方位词、趋向词、空间介词、动词等是否用得得当"
    )
    spatial_reason: str = Field(description="空间语义这一项的判断依据：异常在哪，或为什么没问题")


def semantic_verdict_result(
    grammar_ok: bool, grammar_reason: str, spatial_ok: bool, spatial_reason: str
) -> dict:
    """一次成功的语义判定 → 返回 dict（SemanticAgent.run 与 all_in_one 的对照实现共用这一处）

    合出 is_fluent 的算式只写在这里：两侧（图那条链路与 all_in_one 那条）只要有一边自己另算一遍，
    两份输出就没法逐行对账——而逐行对账正是这两条链路同时存在的意义。

    reason 是把两段理由拼起来的一句话，供闸门措辞与人工阅读用；两段原文另存在 grammar_reason /
    spatial_reason 里，落盘后可以分别统计「这条是被语法挡下的还是被空间语义挡下的」。
    """
    return {
        "is_fluent": bool(grammar_ok and spatial_ok),
        "grammar_ok": bool(grammar_ok),
        "spatial_ok": bool(spatial_ok),
        "grammar_reason": grammar_reason,
        "spatial_reason": spatial_reason,
        "reason": f"语法：{grammar_reason}；空间语义：{spatial_reason}",
        "source": "llm",
    }


def semantic_blank(source: str, reason: str) -> dict:
    """没判出来时的同形返回（--no-llm / 调用失败 / 模型没给出字段）

    键集与 semantic_verdict_result 一致，只是两项判定与两段理由留空：落盘形状不随「判没判出来」
    而变，汇总与对账才能按同一套键去读（图里的闸门只看 is_fluent 为 None 这一种）。
    """
    return {
        "is_fluent": None,
        "grammar_ok": None,
        "spatial_ok": None,
        "grammar_reason": "",
        "spatial_reason": "",
        "reason": reason,
        "source": source,
    }


class RealityVerdict(BaseModel):
    """reality_agent 的输出：关注词在这一句里是不是在说现实世界中的空间场景"""

    is_spatial_scene: bool = Field(
        description="句中的关注词（方位词/趋向动词）是否用于表达现实世界中的空间场景；"
        "引申（会上/理论上）、时间（三天内）、时体与结果（笑起来/看出来）、比喻（走出困境）"
        "等用法一律为 false"
    )
    reason: str = Field(description="判断依据，需说清关注词在这一句里的意思，以及为什么算（或不算）现实空间")


class SchemaVerdict(BaseModel):
    """schema_agent 查表未命中时的输出"""

    schema_name: str = Field(
        description="该句所属的 schema 名。给了候选名单时必须从名单里原样照抄一个，不许自己造名字、"
        "也不许改编号（名单为空、即这一类还没收录图式时，按「组名 + 编号」的格式自行命名："
        "消息里给组名就用组名，没给才用关注词本身）"
    )
    reason: str = Field(description="选择依据，需说明关注词在句法上承担的角色，以及句子的语义特征与所选 schema 的命中条件如何对应")


class JudgeVerdict(BaseModel):
    """judge_agent 交 LLM 判断时的输出（表里没给结论才是这条路：格子标了 ○、留空非法、或这一对没收录）"""

    same_meaning: bool = Field(description="两句话的意义是否相同")
    reason: str = Field(description="判断依据，需说明替换词带来的语义差别")


class SynonymVerdict(BaseModel):
    """synonym_agent 的独立结构化输出；判断标准由用户填写的提示语决定。"""

    same_meaning: bool = Field(description="两句话是否同义")
    reason: str = Field(description="判断理由")


class FreeFeatures(BaseModel):
    """特征类别表未填写时 feature_agent 的兜底输出"""

    features: dict[str, Any] = Field(
        default_factory=dict,
        description="从句子中读出的空间语义特征，键为特征名，值为该特征上的取值",
    )


# --------------------------------------------------------------------------
# 二、提示语
#
# 正文全部外置在 system_prompt/ 下（一个 agent 一份 txt，多份时用 <名字>_<用途>.txt），
# 由 BaseAgent 在构造期读入，见 utils.read_prompt / utils.render_prompt。
# 含变动的提示语用 {{占位符}} 标注——不是 {}：正文里有 JSON 花括号，str.format() 会炸。
# --------------------------------------------------------------------------

PROMPT_SOURCE_SYSTEM = "system"
"""提示语在 system_prompt/ 下（默认，见 BaseAgent.PROMPT_SOURCE）"""
PROMPT_SOURCE_CONFIG = "config"
"""提示语与配置表同目录：config_files/<语料类型>/ 下（唯一一份 reality_agent 的提示语）

它按语料类型分两份、判据随词类不同（方位词挡的是引申与时间用法，趋向动词挡的是时体、结果与
比喻用法），所以与那四张表同一个目录、同一套分组口径，见 spatial_tools.CONFIG_PROMPT_FILES。
"""

REALITY_PROMPT_FILE = "reality"
"""reality_agent 的提示语（放 config_files/<语料类型>/ 下，见 PROMPT_SOURCE_CONFIG）"""
FEATURE_PROMPT_FILE = "feature"
"""feature_agent deep 模式（每类一个 subagent）的主提示语（含 {{n}} 与 {{roster}}）"""
FEATURE_FLAT_PROMPT_FILE = "feature_flat"
"""feature_agent 默认（单次调用）的主提示语（含 {{n}} 与 {{classes}}）"""
FEATURE_UNFILLED_PROMPT_FILE = "feature_unfilled"
"""feature_agent 类别表为空时的兜底提示语"""
FEATURE_CLASS_PROMPT_FILE = "feature_class"
"""每类特征的提示语模板（含 {{name}}/{{description}}/{{values}}/{{examples}}/{{usage}}）

{{usage}} 是两种模式的分岔处：deep 那套（调用 parse_sentence、关注词由主 agent 转述）见
feature_class_usage.txt，flat 那套（句法证据与关注词都已在用户消息里）见
feature_class_usage_flat.txt——判据与逐字照抄规则留在本模板一份，两种模式共用。
"""
FEATURE_CLASS_USAGE_FILE = "feature_class_usage"
"""deep 模式下 feature_class.txt 里 {{usage}} 的取值（subagent 的工作方式与输出格式）"""
FEATURE_CLASS_USAGE_FLAT_FILE = "feature_class_usage_flat"
"""flat 模式下 feature_class.txt 里 {{usage}} 的取值（单次调用时各类的工作方式与输出说明）"""


FOCUS_ROLE_RECORD = "record"
"""关注词的角色：这条语料按筛选规则提取出来的词（single 图，以及 pair / alter_check 图的 a 侧）"""
FOCUS_ROLE_ORIGINAL = "original"
"""关注词的角色：一对句子里**原句**那一侧的词（judge 专用：它同时拿着原句与改写句）"""
FOCUS_ROLE_REPLACEMENT = "replacement"
"""关注词的角色：改写句那一侧换上去的词（pair / alter_check 图的 b 侧，见 st.replacement_focus_word）"""

_FOCUS_WHERE = {
    # 角色 -> 「在哪儿、第几个词」那句的措辞。judge 同时拿着两句，所以它那一版必须说清是哪一句的。
    # 三个角色都只说**原句**（record / original）或**本句且不经改写**（record，single 图）的分词结果，
    # 绝不说「在改写句的分词结果里是第 N 个词」：这个下标是原句分词切片上的位置，而改写句是
    # join 成字符串后**重新分词**的，替换点附近可能与相邻字并作一个词，那样下标就不再指着替换词
    # ——措辞不能替它担保（见 FOCUS_ROLE_REPLACEMENT 的第二行）。a 侧没有这个问题：
    # st.parse_sentence_text(原句) 与语料切片逐 token 相同（实测 2000/2000）。
    FOCUS_ROLE_RECORD: "在分词结果里是第 {index} 个词",
    FOCUS_ROLE_ORIGINAL: "在原句的分词结果里是第 {index} 个词",
    FOCUS_ROLE_REPLACEMENT: "占的是原句分词结果里第 {index} 个词的位置",
}

_FOCUS_TAIL = {
    # 角色 -> 第二行（以哪个词为准、要判什么）。提示语里那些「紧接着的下一行」就指着它
    FOCUS_ROLE_RECORD: (
        "它是这条语料按筛选规则提取出来的词；本句若经过同义词替换，"
        "该位置上的词已换成规则表里的另一个词，一律以句子中实际的用词为准。"
    ),
    FOCUS_ROLE_ORIGINAL: (
        "它是本次改写在原句里替换掉的那个词，本次要判的就是把它换成改写句里那个词之后"
        "两句意义是否相同；改写句换成了哪个词见下一行，判断以原句这个词为参照。"
    ),
    # 这一行必须把「下标不适用于改写句」讲明白：b 侧的下标是从 a 侧原样搬过去的
    # （见 st.replacement_focus_word），而改写句的 parse_sentence 会重新分词，
    # 替换点附近可能与相邻的字并成一个词（实测 2000 条改写句样本里 98 条，4.9%：
    # 「对焦点下移」的 下+移 并成 token「下移」，下标 13 于是指到「下移」上），
    # 此时按下标去数会数到别的 token。定位靠词本身是稳的：实测这 98 条里替换词一个不落地
    # 都在改写句字符串里（规则只把 index 处的 token 换成它，逐字仍在原处）。
    FOCUS_ROLE_REPLACEMENT: (
        "它是本次改写替换上去的那个词，改写句里那个位置上写的就是它；"
        "但改写句是把原句重新拼出来的字符串、要重新分词，替换点附近可能与相邻的字并成一个词，"
        "所以不要拿上面的下标去数改写句的分词行，请按这个词本身在改写句里定位它"
        "（句中若不止一处出现这个词，仍以那个位置上的为准）。"
        "这一句的语义（是否通顺、抽什么特征、归哪个 schema）都围绕它展开。"
    ),
}

FOCUS_INDEX_NOTE = "（下标从 0 起，与该句分词结果里 token 前的序号一致）"
"""下标那句注（record / original 共用：这两处的下标就是本句分词的序号）

措辞刻意不提 parse_sentence：默认已不绑这个工具，句法证据（分词/词性那行，token 前同样打着
下标）由各 agent 的消息直接带进去（见 syntax_block）；reality 那条消息没有句法证据，但下标
是上游按同一套分词算好直接给的，这句注在它那里同样成立。"""

FOCUS_INDEX_NOTE_ALTER = "（下标从 0 起，是原句分词结果里的序号，不要在改写句的分词行上数它）"
"""改写句那一侧的下标注：同一个数，但说的是**原句**的分词

与 FOCUS_INDEX_NOTE 只差半句，差别却是要紧的：b 侧的下标是从 a 侧搬过去的（见
st.replacement_focus_word），模型手上只有改写句，照 FOCUS_INDEX_NOTE 去数改写句的分词行、
再数第 index 个 token，替换点并词时就会对到别的词上。

实测（把改写句字符串重新喂给 LTP 分词，2000 条样本）：4.9% 的记录下标对不上，形态都是替换点
并词——「对焦点下移」里 下+移 并成 token「下移」，原下标 13 指到「下移」而非「下」。
同一个测法下 a 侧 2000/2000 全对得上（原句未被改写，切片就是它自己的分词）：

    st.prefill_ltp(sentences); st._cache.get_task_value(sentence, config.CWS)

这两条正是「a 侧沿用 FOCUS_INDEX_NOTE、b 侧另给一条」的依据。
"""

FOCUS_A_ABSENT = (
    "原句那一侧没有可用的关注词（这条记录里没取到关注词），请按句子中实际的用词判断。\n"
    "这一侧的 schema 与语义特征按句子中实际的用词理解。"
)
"""judge 消息里原句那一份关注词缺席时的占位（见 pair_head_block）

与 FOCUS_B_ABSENT 一样是**两行**：focus_line 取不到词时只给一行中性说明，会让后面的行整体上移
一行，而 judge.txt 是按行号指的（第 1 行 / 第 3 行）。judge 那句「相应那一行会明说没有可用的
关注词」也认这个措辞——两份占位都用「没有可用的关注词」，四种缺席组合下行长都是 5。
"""

FOCUS_B_ABSENT = (
    "改写句那一侧没有可用的关注词（本次没记录换上去的词），请按句子中实际的用词判断。\n"
    "这一侧的 schema 与语义特征按句子中实际的用词理解。"
)
"""judge 消息里改写句那一份关注词缺席时的占位（见 pair_head_block）

**两行**：与 focus_line 的产物同高。judge 的消息是「原句那份（2 行）+ 改写句那份（2 行）+ 类型行」，
行数一变，类型行与后面「原句：…」那些行的位置就都跟着挪，而提示语里是按行号指的
（judge.txt 说「第 1 行 / 第 3 行」）。缺席是少见情形，更不该让它成为提示语措辞对不上的那一档。
"""


def focus_role(focus_word: dict | None) -> str:
    """这份关注词是哪一侧的：由它的 source 判——改写句那一侧的来源固定是 alter

    角色只决定措辞（见 focus_line）。由数据自己推、而不是让各个 agent 各接一个形参：同一份
    关注词在各个 agent 的 user message 里必须说成同一回事，而 agent 手上就只有这个 dict。
    手写的关注词（没有 source）一律按 record 算。
    """
    if (focus_word or {}).get("source") == st.FOCUS_SOURCE_ALTER:
        return FOCUS_ROLE_REPLACEMENT
    return FOCUS_ROLE_RECORD


def focus_line(focus_word: dict | None, *, role: str = FOCUS_ROLE_RECORD) -> str:
    """把关注词渲染成 user message 的首行（LLM agent 共用这一处措辞）

    focus_word 是关注词（{"index", "word", "kind", "source"}，见 spatial_tools.pick_focus_word
    与 replacement_focus_word）；没有关注词（None 或取不到词）时给一句中性说明，让 agent 按整句判断。
    role 决定后面两句怎么说（默认按 focus_role 由 source 推）：

    - record：这条语料提取出来的词——single 图，以及 pair / alter_check 图的 a 侧（原句）；
    - replacement：改写句里换上去的那个词——pair / alter_check 图的 b 侧，下标是它**顶替掉的那个词
      在原句**里的位置（不是它在改写句里的位置：改写句要重新分词，下标不保证搬得过去，见 _FOCUS_TAIL）；
    - original：一对句子里原句那一侧的词——judge 专用（它同时拿着原句与改写句，另外那句由
      pair_head_block 的下一行交代），下标是它在**原句**里的位置。

    措辞为什么这么绕，两条都要照顾到：

    - 各个 agent 的 system_prompt 在几张图之间共用，所以措辞里不能出现只对某一张图成立的说法，
      只能用条件句交代，并给出 index 这个可核对锚点（句法证据的分词行里每个 token 前都
      打了下标，能直接对上）。唯一的例外是 replacement：那个下标是原句上的，模型只能按下标
      核对**原句**，改写句上要按词本身找（FOCUS_INDEX_NOTE_ALTER 与 _FOCUS_TAIL 里都说了）。
    - 按 side 生成两套措辞会让 a/b 两侧的特征失去可比性——可比性正是这两个 agent 存在的意义。
      两侧的关注词虽不同（原词与替换词），说法与结构却完全一样，正是为了保住这一点。
    """
    if role not in _FOCUS_WHERE:
        raise ValueError(f"认不出的关注词角色 {role!r}；可用：{'、'.join(_FOCUS_WHERE)}")
    if not isinstance(focus_word, dict) or not focus_word.get("word"):
        return "本次没有需要特别关注的方位词/趋向动词，请按整句判断。"
    kind = f"（{focus_word['kind']}）" if focus_word.get("kind") else ""
    index = focus_word.get("index")
    if isinstance(index, int):
        note = FOCUS_INDEX_NOTE_ALTER if role == FOCUS_ROLE_REPLACEMENT else FOCUS_INDEX_NOTE
        where = _FOCUS_WHERE[role].format(index=index) + note
    else:
        where = "在句中的位置未能定位"
    if role == FOCUS_ROLE_REPLACEMENT:
        lead = f"它是本次改写替换上去的那个词，{where}。"
    elif role == FOCUS_ROLE_ORIGINAL:
        lead = f"它是本次改写在原句里替换掉的那个词，{where}。"
    else:
        lead = f"{where}。"
    return f"本次需要重点关注的词：「{focus_word['word']}」{kind}，{lead}\n{_FOCUS_TAIL[role]}"


CORPUS_TYPE_LABEL = "本次跑批的语料类型"
"""类型那一行的开头；verify_focus_wiring.py 按它定位（改文案要同步改那边）"""


def corpus_type_line(corpus_type: str | None) -> str:
    """user message 里交代本次跑批语料类型的那一行；没有类型（None）时给空串

    语料类型是跑批级常量，它决定的是读哪一组配置表（见 spatial_tools.config_path）；
    写进消息里是让模型知道自己看的是哪一类词的表（schema 名、特征类别都与之对应）。
    """
    return f"{CORPUS_TYPE_LABEL}：{corpus_type}。" if corpus_type else ""


GROUNDING_LABEL = "前序步骤整理的空间角色"
"""user message 里三元组提示那一行的开头（verify 脚本按它定位）"""


def grounding_block(focus_word: dict | None, *, side: str = "a") -> str:
    """把前序「目标物／参照物／动词」三元组渲染成**独立一行**（没有就返回空串）

    数据来自 `spatial_tools.select_grounding`，由 `pick_focus_words` 挂在关注词上，
    所以七个 LLM agent 与 all_in_one 只要走 head_block / pair_head_block 就自动带上，
    不需要每个 `_content()` 各接一个参数。

    几个口径：
    - 只在本位置确实有标注时出现，**不在下游推断**；方位词与趋向动词语料都会带这个块，
      哪一条没有就整行省略。
    - 趋向动词可以合法地没有依附动词（独立作谓语）；此时用 ``verb_optional`` 说明
      「无」而不是「未识别」，避免模型把它当成抽取失败。
    - keep=False/None 也照原样带上，并写明标注状态：它是「前序已经整理出的标注」，
      不是「已确认的事实」，让模型自己判断可信度。
    - b 侧（改写句）照带同一份，但明确说它属于原句、改写后未重新验证；下标也属于原句。
    - 这一行追加在 head_block 已有的关注词行与语料类型行**之后**：提示语里
      「第一行/紧接的下一行」「第三行是语料类型」这些锚点都不会移动。
    """
    grounding = (focus_word or {}).get("grounding")
    if not isinstance(grounding, dict):
        return ""
    target = grounding.get("target") or {}
    reference = grounding.get("reference") or {}
    verb = grounding.get("verb") or {}
    target_word = target.get("word") if isinstance(target, dict) else None
    reference_word = reference.get("word") if isinstance(reference, dict) else None
    verb_word = verb.get("word") if isinstance(verb, dict) else None
    if not (target_word or reference_word):
        return ""
    target_tail = f"（角色 {target.get('role')}）" if target.get("role") else ""
    reference_tail = f"（{reference.get('sdp')}）" if reference.get("sdp") else ""
    if verb_word:
        verb_text = verb_word
    elif grounding.get("verb_optional"):
        verb_text = "无（趋向动词独立作谓语，允许为空）"
    else:
        verb_text = "未识别"
    keep = grounding.get("keep")
    status = "keep=true" if keep is True else "keep=false" if keep is False else "keep=未判定"
    note = "（属于原句；改写后未重新验证）" if side == "b" else "（属于原句）"
    ambiguous = "；同一位置有多条标注，此处取第一条" if grounding.get("ambiguous") else ""
    return (
        f"{GROUNDING_LABEL}{note}：目标物「{target_word or '未识别'}」{target_tail}、"
        f"参照物「{reference_word or '未识别'}」{reference_tail}、"
        f"动词「{verb_text}」；标注状态 {status}{ambiguous}。"
    )


def head_block(focus_word: dict | None, *, corpus_type: str | None = None) -> str:
    """user message 的开头块：关注词那段（1~2 行）+ 语料类型行（有类型时）+ 三元组行（有标注时）

    关注词那句按 source 自动选措辞（见 focus_role）：语料里取来的词与改写句里换上去的词，
    在 semantic / feature / schema 三个 agent 那里都要说成对应的一侧。

    类型行只排在关注词整段**之后**（第 3 行），三处约束逼出来的：

    - 提示语正文（semantic / feature / feature_unfilled / schema）的条件句都写作「用户消息的
      **第一行**若点名了本次需要重点关注的词」，第 1 行必须仍是关注词行；
    - feature.txt 要求「紧接的下一行」交代同义替换以哪个词为准，第 1、2 行之间不能插东西；
    - verify_focus_wiring.py 断言首行以「本次需要重点关注的词：」开头。

    corpus_type=None 时逐字节等于 focus_line(...)：离线接线测试的既有断言靠这条成立。
    judge 那张图不在这里拼——它同时拿着两句的关注词，走 pair_head_block。
    """
    return "\n".join(
        part
        for part in (
            focus_line(focus_word, role=focus_role(focus_word)),
            corpus_type_line(corpus_type),
            # 三元组固定排在既有两行之后：第 1 行仍是关注词、第 3 行仍是语料类型
            grounding_block(
                focus_word,
                side="b" if focus_role(focus_word) == FOCUS_ROLE_REPLACEMENT else "a",
            ),
        )
        if part
    )


def pair_head_block(
    focus_word: dict | None,
    focus_word_b: dict | None,
    *,
    corpus_type: str | None = None,
    role_a: str = FOCUS_ROLE_ORIGINAL,
) -> str:
    """两侧关注词的开头块：a 侧那份 + b 侧那份 + 语料类型行（有类型时）+ 三元组行（有标注时）

    比其余三个 agent 多一段（各 2 行 + 类型行）：这里同时拿着原句与改写句的两份关注词——原句里
    被替换掉的那个，与改写句里换上去的那个（见 st.replacement_focus_word）。第 1 行仍是关注词
    那一行：四份提示语的条件句都按「第一行点名了本次要关注的词」写。

    role_a 决定 a 侧那句怎么说，默认 FOCUS_ROLE_ORIGINAL——judge 走这条：它同时拿着两句，
    第 2 行要点明「本次要判的就是把原句这个词换成改写句那个词之后是否同义」。all_in_one 一次
    作答五个判断，a 侧按 FOCUS_ROLE_RECORD 说（与 reality/semantic/feature/schema 四个 agent
    一致，注入的 reality 判据也说「第二行交代了这个词是怎么被提取出来的」）。b 侧恒为
    FOCUS_ROLE_REPLACEMENT：两侧的关注词是两个词，一侧说错措辞就丢掉可比性。

    两侧各自没有关注词时用 FOCUS_A_ABSENT / FOCUS_B_ABSENT 占位（原句取不到词、没走到改写、
    状态是手工拼的），**行数不变**：focus_line 对空关注词只回一行中性说明，让它顶上去整体就少一行，
    提示语里「第一行 / 紧接着的下一行」这些锚点全错位。两份占位都写成两行，四种组合都是 5 行；
    有前序三元组时再追加第 6 行（grounding_block），前五行不动。
    """
    return "\n".join(
        part
        for part in (
            focus_line(focus_word, role=role_a)
            if isinstance(focus_word, dict) and focus_word.get("word")
            else FOCUS_A_ABSENT,
            focus_line(focus_word_b, role=FOCUS_ROLE_REPLACEMENT)
            if isinstance(focus_word_b, dict) and focus_word_b.get("word")
            else FOCUS_B_ABSENT,
            corpus_type_line(corpus_type),
            grounding_block(
                focus_word if (focus_word or {}).get("grounding") else focus_word_b,
                side="a"
                if (focus_word or {}).get("grounding")
                else "b"
                if focus_role(focus_word_b) == FOCUS_ROLE_REPLACEMENT
                else "a",
            ),
        )
        if part
    )


def syntax_block(sentence: str) -> str | None:
    """该句的句法证据（分词/词性 + 依存句法），取不到时返回 None

    默认不给 agent 绑 parse_sentence 工具了：工具调用就是又一次 API 请求，而这边本地算好
    直接塞进消息（与 all_in_one._syntax_block 同一份文本），每一步的往返就少一次。

    LTP 起不来（没装模型、显存不够）不该带走整批：省掉这一段，模型仍能凭句子作答，只是退化成
    「只凭语感」。真出岔子时由 spatial_runner._prefill 在预填充那一步先报——跑批前整批句子都
    要过一遍 LTP，那里失败是响的，这里的 None 只是单句失败时的兜底。

    做成模块级函数是为了给 verify_focus_wiring.py 留一个接缝：离线测试把它换成确定性假文本，
    「全程不加载 LTP」这条承诺才继续成立（见该脚本的模块 docstring）。
    """
    try:
        return st.parse_sentence_text(sentence)
    except Exception:  # noqa: BLE001 —— 句法分析失败只降级，不让它决定这一条记录跑不跑
        return None


def render_example_lines(feature_class: dict, *, indent: str = "    ", limit: int = 5) -> str:
    """把一类特征填的示例渲染成提示语里的几行（没有可用示例时返回空串）

    每条示例先出一行 JSON（{sentence, <key>: <取值>}）；该条若另填了非空 instruction，
    就在它下面再缩进一行「说明：…」。instruction 先从 dict 里摘掉再 dump，否则同一段话
    会在 JSON 里和说明行里各出现一次。

    instruction 是可选字段：缺字段、空串、类型不对都只是不渲染那一行，不报错——配置里放空串
    占位（见 feature_classes.json5 表头）时，渲染结果与没有这个字段时逐字一致。

    feature_agent（_class_prompt）与 all_in_one（_feature_roster）共用这一份，两边口径不会分叉；
    indent 由调用方给，好让示例缩进到各自提示语里合适的位置。
    """
    lines: list[str] = []
    for example in (feature_class.get("examples") or [])[:limit]:
        if not isinstance(example, dict):
            continue
        body = {k: v for k, v in example.items() if k != "instruction"}
        lines.append(f"{indent}{json.dumps(body, ensure_ascii=False)}")
        instruction = example.get("instruction")
        if isinstance(instruction, str) and instruction.strip():
            lines.append(f"{indent}  说明：{instruction.strip()}")
    return "\n".join(lines)


def _class_prompt(template: str, feature_class: dict, *, usage: str) -> str:
    """按模板给某一类语义特征写一段提示语（模板见 system_prompt/feature_class.txt）

    usage 是模板里 {{usage}} 的取值：deep 时这段给 subagent（它自己去调 parse_sentence、
    关注词由主 agent 转述），flat 时这段跟着整份系统提示语一起给单次调用的那个模型。
    """
    values = "、".join(str(v) for v in (feature_class.get("values") or [])) or "（未限定，按判据自行概括）"
    example_lines = render_example_lines(feature_class)
    return utils.render_prompt(
        template,
        {
            "name": feature_class.get("name") or feature_class["key"],
            "description": feature_class.get("description") or "（未填写）",
            "values": values,
            "usage": usage,
            # 「示例：」这段有没有取决于有没有示例，末尾那个换行也得跟着一起消失，所以整段
            # （含标签与结尾换行）当**一个**占位符传进去，空时给空串。别把它拆成「模板里写死
            # 『示例：\n』+ 只替换行列表」——那样空示例时会多出一行「示例：」，而模板里看不出问题。
            "examples": f"示例：\n{example_lines}\n" if example_lines else "",
        },
        source=f"{FEATURE_CLASS_PROMPT_FILE}.txt",
    )


def _subagent_name(index: int, key: str) -> str:
    """给 subagent 起名

    subagent 的 name 会被当成工具名使用，只能含 ASCII 字母数字与 _-；
    特征 key 多为汉语（如「参照物」），这类回落到序号命名，中文名放进 description。
    """
    safe = "".join(ch for ch in key if ch.isascii() and (ch.isalnum() or ch in "_-"))
    return f"feature_{safe}" if safe else f"feature_{index}"


def _validate_feature_classes(feature_classes: list[dict], *, source: str | None = None) -> None:
    """校验特征类别表填得能用

    校验刻意留在构造期（而不是等跑到模型）：表填错属于配置问题，应该在启动时就报，
    而不是每条记录吞一次、把错误混进结果里。

    source 是这张表的实际路径，只用于报错信息（表按语料类型分组，写死路径会指错文件）。
    """
    where = source or "feature_classes.json5"
    for feature_class in feature_classes:
        key = feature_class.get("key") or ""
        if not key.isidentifier():
            raise ValueError(
                f"特征类别的 key {key!r} 不是合法的字段名（不能含空格/连字符等），"
                f"请修改 {where}。"
            )
    if len({c["key"] for c in feature_classes}) != len(feature_classes):
        raise ValueError(f"特征类别表里有重复的 key，请修改 {where}。")


def validate_feature_values(features: dict, feature_classes: list[dict]) -> dict:
    """结构化输出**之后**的业务校验（A2）：闭集取值、缺失、未定义标签

    _feature_model 建的模型只约束了形状（每类一个 list[str]），**没有**把配置里的闭集 values
    变成枚举约束。于是格式完全合法的输出仍可能含未定义标签、近义拼写或不符合该维度约定的
    取值，后果是 schema 查表静默未命中、或整条记录意外降级到 LLM 兜底——看起来却一切正常。
    这类问题不能只靠提示语约束，必须在产出之后查一遍。

    闭集与开放分开处理：
      - 类别表里 `values` 非空的按**闭集**查：取值必须在里面，否则记进 undefined；
      - 没写 values 的是**开放**类，只查取值本身是不是非空字符串（没有可对照的清单）。

    空列表**不判失败**：某维度在这一句里确实可能不适用（现有测试也把空特征当合法），
    所以只在闭集类里记 missing，由调用方决定要不要复核——不把合法空值误伤成错误。

    Args:
        features (dict): 模型给出的特征对象 {key: [取值, ...]}
        feature_classes (list[dict]): 特征类别表（键与取值清单的真源）

    Returns:
        dict: {"ok": bool, "undefined": {key: [越界取值, ...]}, "missing": [闭集但没取值的 key, ...],
               "open": [开放类的 key, ...], "unfilled": [类别表里没定义的 key, ...]}
               —— flat、deep、all_in_one 三条路共用同一份口径，对同一个特征对象给出同一个结果。
    """
    declared = {str(c.get("key")): c for c in feature_classes or [] if isinstance(c, dict) and c.get("key")}
    closed: dict[str, set[str]] = {}
    for key, feature_class in declared.items():
        values = [str(v) for v in (feature_class.get("values") or []) if str(v)]
        if values:
            closed[key] = set(values)

    undefined: dict[str, list[str]] = {}
    missing: list[str] = []
    unfilled: list[str] = []
    open_keys: list[str] = []

    for key, values in (features or {}).items():
        key = str(key)
        if key not in declared:
            unfilled.append(key)
            continue
        items = [v for v in (values or []) if isinstance(v, str) and v.strip()]
        if key not in closed:
            open_keys.append(key)
            continue
        allowed = closed[key]
        outside = sorted({v for v in items if v not in allowed})
        if outside:
            undefined[key] = outside
        elif not items:
            missing.append(key)

    return {
        "ok": not undefined and not missing and not unfilled,
        "undefined": undefined,
        "missing": sorted(missing),
        "open": sorted(open_keys),
        "unfilled": sorted(unfilled),
    }


def _table_missing(path: str) -> bool:
    """这张表是「路径上确实没有文件」吗

    与「文件在、但内容是空的」分开：两者都降级，但处置不同（前者要去建/改路径，
    后者要往文件里填内容），启动打印要能一眼分清。

    只把看起来是文件路径的当路径判：注入进来的表（只给了 rules/table、没给路径）在
    table_path 里存的是一句说明，不以表文件的后缀结尾，那种情形没有文件可查。
    """
    return path.endswith((".json5", ".csv")) and not Path(path).exists()


def _subagent_specs(feature_classes: list[dict]) -> list[dict]:
    """每类特征对应 subagent 的 name / label / key / description，与 feature_classes 一一对应

    构造期渲染主提示语（要拼 roster）与 _build 时组装 subagents 都要这几样，放一处算，
    免得日后只改了一边、主提示语里报的名字和真正建出来的 subagent 对不上。
    """
    return [
        {
            "name": _subagent_name(index, feature_class["key"]),
            "label": feature_class.get("name") or feature_class["key"],
            "key": feature_class["key"],
            # description 是主 agent 判断何时调用它的唯一依据，要写清"什么时候用"
            "description": f"抽取句子中「{feature_class.get('name') or feature_class['key']}」类语义特征时调用",
        }
        for index, feature_class in enumerate(feature_classes, start=1)
    ]


def render_class_blocks(feature_classes: list[dict], class_template: str, *, usage: str) -> list[str]:
    """flat 提示语里的类别段：每类一段 `### 名称（key：key）` + feature_class.txt 的渲染

    单次调用（flat）要把各类的判据、可选取值与逐字照抄规则一并塞进同一份系统提示语，模型靠每段
    开头这行标题把取值填进正确的字段（deep 时归属由 roster 交代，不需要这行）。

    all_in_one 的 {{roster}} 占位符也复用这一份：两处的类别段（含取值按「、」拼接的口径、
    逐字照抄规则）因此逐字一致，不会各自演化——这正是不必在那边另写一套渲染的原因。
    """
    return [
        f"### {feature_class.get('name') or feature_class['key']}（key：{feature_class['key']}）\n"
        f"{_class_prompt(class_template, feature_class, usage=usage)}"
        for feature_class in feature_classes
    ]


def _render_feature_prompts(
    feature_classes: list[dict],
    head_template: str,
    class_template: str,
    *,
    mode: str,
    usage: str,
) -> tuple[str, list[str]]:
    """渲染 feature_agent 的提示语，返回 (主提示语, [各类的提示语])

    模板由调用方读好传进来（FeatureAgent 在构造期用 system_prompt/ 下的文件渲染）。
    两种模式共用 class_template（判据与逐字照抄规则只有一份），分岔在 {{usage}}：
    mode="flat"（默认，单次调用）的主提示语用 feature_flat.txt（占位符 {{n}}/{{classes}}），
    mode="deep" 用 feature.txt（占位符 {{n}}/{{roster}}，各类的提示语给对应 subagent）。

    flat 的类别段（{{classes}}）由 render_class_blocks 拼（每类前加一行标题），all_in_one 复用同一份。
    """
    class_prompts = [_class_prompt(class_template, feature_class, usage=usage) for feature_class in feature_classes]
    if mode == "flat":
        blocks = render_class_blocks(feature_classes, class_template, usage=usage)
        head = utils.render_prompt(
            head_template,
            {"n": len(feature_classes), "classes": "\n\n".join(blocks)},
            source=f"{FEATURE_FLAT_PROMPT_FILE}.txt",
        )
        return head, class_prompts
    roster = "\n".join(
        f"    {spec['name']}：负责「{spec['label']}」，对应特征字段 {spec['key']}"
        for spec in _subagent_specs(feature_classes)
    )
    head = utils.render_prompt(
        head_template, {"n": len(feature_classes), "roster": roster}, source=f"{FEATURE_PROMPT_FILE}.txt"
    )
    return head, class_prompts


def _feature_model(feature_classes: list[dict]):
    """按类别表建结构化输出模型：每类一个列表字段（flat 与 deep 共用同一个形状）

    key 是汉语也是合法的 Python 标识符，可以直接当 pydantic 字段名（合法性由调用方先校验）；
    含空格、连字符之类的键不行（见 _validate_feature_classes）。
    """
    fields = {
        feature_class["key"]: (
            list[str],
            Field(
                default_factory=list,
                description=f"{feature_class.get('name') or feature_class['key']}："
                f"{feature_class.get('description') or ''}".strip(),
            ),
        )
        for feature_class in feature_classes
    }
    return create_model("SemanticFeatures", **fields)


def _build_deep_feature_agent(
    model_key: str,
    feature_classes: list[dict],
    main_tools: list,
    subagent_tools: list,
    head_prompt: str,
    class_prompts: list[str],
):
    """建特征抽取用的 deep agent：每类特征一个 subagent，主 agent 汇总成结构化特征对象

    head_prompt 与 class_prompts 由 FeatureAgent 构造期按 system_prompt/ 下的模板渲染好传进来
    （本函数是模块级函数，拿不到 agent 实例），class_prompts 与 feature_classes 一一对应。

    工具分两份：主 agent 只负责派发与汇总，默认不给工具；subagent 做实际抽取，保留
    parse_sentence（deep 是唯一还让模型自己调工具的路径），显式注入 tools 时以注入的为准。
    """
    subagents = [
        {
            "name": spec["name"],
            "description": spec["description"],
            "system_prompt": prompt,
            "tools": subagent_tools,
        }
        for spec, prompt in zip(_subagent_specs(feature_classes), class_prompts, strict=True)
    ]

    return agent_factory.get_deep_agent(
        model_key,
        tools=main_tools,
        system_prompt=head_prompt,
        subagents=subagents,
        response_format=ToolStrategy(_feature_model(feature_classes)),
    )


# --------------------------------------------------------------------------
# 三、agent 基类
# --------------------------------------------------------------------------


class BaseAgent:
    """七个 agent 的公共骨架：输入契约、llm 开关、惰性建模型。

    子类要给出 NAME / INPUTS / OUTPUTS，并实现 run() 与 _build()。
    INPUTS 是 run() 的形参名，图的节点适配器拿它与接线声明逐一比对（见 spatial_graphs.agent_node），
    少接一条线在建图时就报错，而不是等跑到那一步才 TypeError。
    """

    NAME: str = ""
    """注册表（AGENTS）里的键，也是 build_agent 取用的名字"""
    INPUTS: tuple[str, ...] = ()
    """run() 的形参名，按顺序"""
    OUTPUTS: tuple[str, ...] = ()
    """run() 返回值必有的键"""
    NEEDS_LLM: bool = True
    """False 表示纯规则 agent：构造与运行都不碰模型"""
    CONFIG_NOTE: str = ""
    """读哪张配置表（提示语文件见 PROMPT_FILES），--list 时打印"""
    PROMPT_FILES: tuple[str, ...] = ()
    """本 agent 的提示语文件（不含 .txt），首个是主提示语，放哪儿见 PROMPT_SOURCE

    构造期全部读进 self.prompts（实际路径记在 self.prompt_paths），主提示语进 self.system_prompt；
    缺一个就在这里报错。空元组表示不用提示语（纯规则的 AlterAgent）。需要按占位符渲染的 agent
    （FeatureAgent）在 __init__ 里覆盖 self.system_prompt——渲染要等它自己的配置表读完。
    """
    PROMPT_SOURCE: str = PROMPT_SOURCE_SYSTEM
    """提示语在哪儿：system = system_prompt/（默认）；config = config_files/<语料类型>/

    按语料类型分两份的提示语走 config（目前只有 reality_agent）：它的判据随词类不同，
    与四张表同一个目录、同一套分组口径（见 spatial_tools.CONFIG_PROMPT_FILES）。
    两种来源的读法与报错文案是同一套（utils.read_prompt_path），只有路径的来源不同。
    """
    CACHEABLE_SOURCES: tuple[str, ...] = ()
    """哪些 source 的 run() 结果值得按 user message 缓存（空元组 = 本 agent 不缓存）

    缓存的前提是「同一串 message ⇒ 同一答案」：同一 agent 实例意味着同一模型、同一 system
    prompt、同一语料类型（三样都在构造期定死），键就是渲染好的 user message 本身。
    只记成功的那几种 source，失败的不入缓存（一次接口抖动被缓存住，同一条语料的候选会成片
    记成同一个错；留着失败按候选重试，代价只是失败的那几个多问一次）。见 _cache_put。
    """

    def __init__(
        self,
        *,
        model_key: str = DEFAULT_MODEL,
        llm: bool = True,
        tools: list | None = None,
        corpus_type: str | None = None,
        max_requests: int | None = None,
        batch_budget: BatchRequestBudget | None = None,
        retry_policy: dict | None = None,
    ):
        self.model_key = model_key
        if retry_policy is not None:
            self.retry_policy = dict(retry_policy)
        else:
            # 生产路径由 build_agents 传入；单独构造 agent 时（测试桩、对照实验）退回不重试。
            # getattr 兜底是为了让只桩了 get_agent 的旧测试仍能构造 agent。
            request_policy = getattr(agent_factory, "request_policy", None)
            self.retry_policy = (
                request_policy(model_key)
                if request_policy is not None and llm and self.NEEDS_LLM
                else {"max_retries": 0}
            )
        """超时/重试策略（见 agents.request_policy 与 retry.call_with_retry）

        由 build_agents 按模型读一次、所有 agent 共用；模型配置里的 timeout /
        max_retries / retry_backoff_seconds 等在这里生效。--no-llm 或纯规则 agent
        不读模型配置，策略置为不重试。
        """
        # 本 agent 的请求预算（A3）：None = 不限。由 build_agents 按 config.CANDIDATE_MAX_REQUESTS
        # 注入；超预算即抛 RequestBudgetExceeded，不再把失败按候选放大成整批的倍数。
        # 它只在 budget_scope() 内累计（一条候选状态一个作用域），见 _invoke。
        self.max_requests = None if not max_requests or max_requests < 0 else int(max_requests)
        # 整批预算：所有 agent 共用 build_agents 建的那一份（None = 不限）
        self.batch_budget = batch_budget
        # 本次跑批的语料类型（规范化后的中文名："方位词" / "趋向动词"）；None = 调用方没交代类型。
        # 刻意不在这里补成默认类型：补了就没法区分「显式指定方位词」与「没指定」，
        # 而两者对提示语的要求不同——只有显式给了类型才在 user message 里交代那一行
        # （没指定就与旧版逐字节一致，离线接线测试靠这条）。选表那一侧则由 config_path 兜默认。
        self.corpus_type = st.normalize_corpus_type(corpus_type) if corpus_type else None
        self.table_path: str = ""
        """本 agent 实际读的那张配置表的路径；表由调用方注入时是一句说明（见 _table_missing）"""
        # 纯规则 agent 永远 self.llm=False：即使调用方传了 llm=True 也不会去建模型
        self.llm = bool(llm) and self.NEEDS_LLM
        self._tools = list(tools) if tools is not None else None
        # 提示语在构造期读：文件名写错、文件缺失属于配置问题，应该在启动时就报，而不是等第一次
        # 调模型（甚至跑到某一条记录）才暴露。--no-llm 下同样会读、会渲染。
        self.prompt_paths: dict[str, str] = {name: str(self._prompt_path(name)) for name in self.PROMPT_FILES}
        self.prompts: dict[str, str] = {
            name: utils.read_prompt_path(path) for name, path in self.prompt_paths.items()
        }
        self.system_prompt: str = self.prompts[self.PROMPT_FILES[0]] if self.PROMPT_FILES else ""
        self._agent = None  # 惰性：第一次 run（或 warm）才建底层 agent
        self._lock = threading.Lock()  # 多线程下避免同时建出好几份（对 deep agent 尤其重要）
        # 按 user message 的结果缓存（见 CACHEABLE_SOURCES）。键就是渲染好的 message 本身——
        # 命中即等价于「问的是逐字节相同的那一句」，所以日后改措辞不会留下陈旧命中
        # （message 变了，键就变）。换一份 agent 就是换一份缓存：模型、system prompt、语料类型
        # 都在构造期定死，这正是「同一串 message ⇒ 同一答案」成立的前提。
        self._verdicts: dict[str, dict] = {}
        self._verdict_lock = threading.Lock()  # 多线程读/写 _verdicts 都过这把锁
        self._flight: dict[str, threading.Lock] = {}
        """按 message 的在途锁：同一串 message 的首问只许一个线程发出去

        --workers N 时同一条语料的多个候选可能同时第一次问到同一句（a 侧正是这样），
        没有这把锁就会同时 miss、各发一次——去重的收益被并发吃掉一半。
        """
        self._flight_guard = threading.Lock()  # 保护 _flight 自身的建表
        # 本次跑批的调用统计：模型往返数（含工具往返）、发出去但没成的调用数、缓存命中数，
        # 跑批入口收尾时汇总。三者缺一不可——只数成功的那部分，会把「整批被接口挡下」显示成
        # 0 次往返，读起来正是一个请求都没发出去（见 _invoke 的失败分支）
        self._stats = {
            "roundtrips": 0,
            "failed": 0,
            "cache_hits": 0,
            # A3：超预算而没发出的请求数。它与 failed 是两件事——failed 是发出去没成，
            # budget_exceeded 是**根本没发**；合起来看才知道「省下了多少」与「漏了多少」。
            "budget_exceeded": 0,
            # A5：统一事件口径里的两项成本指标。requests = roundtrips + failed（成功/最终失败的逻辑调用数），
            # elapsed_seconds 是这些调用花掉的墙上时间，用来算「每个有效句对的成本」。
            "requests": 0,
            "elapsed_seconds": 0.0,
            # 重试账：retries 是决定重发的次数（含最终成功的那几次），
            # retry_exhausted 是重试次数用完仍失败的逻辑调用数。两者与 failed 分开，
            # 才能从汇总里一眼分出「模型慢」与「一直在超时重试」。
            "retries": 0,
            "retry_exhausted": 0,
        }
        self._stats_lock = threading.Lock()

    # -- 子类实现 ----------------------------------------------------------
    def run(self, **kwargs) -> dict:
        """跑一次，返回可直接落盘的 dict（含 source 字段，如实标记结果来源）"""
        raise NotImplementedError

    def _build(self):
        """建底层 langchain / deepagents agent"""
        raise NotImplementedError

    def _prompt_path(self, name: str) -> Path:
        """这份提示语的实际路径：默认在 system_prompt/ 下，config 来源的按语料类型取

        路径在这里算一次、绕给 utils.read_prompt_path：读法与报错文案两种来源完全一致，
        变的只是路径怎么来（见 PROMPT_SOURCE）。路径本身也留着（self.prompt_paths），
        describe() 与 --list 要拿它说明这次读的到底是哪一份。
        """
        if self.PROMPT_SOURCE == PROMPT_SOURCE_CONFIG:
            return st.config_prompt_path(name, self.corpus_type)
        return utils.prompt_path(name)

    # -- 公共设施 ----------------------------------------------------------
    def tools(self) -> list:
        """本 agent 的工具列表，**默认为空**：句法证据已由 syntax_block 直接写进消息

        工具调用就是又一次 API 请求，默认不绑 parse_sentence，每一步少一次往返。显式注入
        tools= 时以注入的为准（deep 模式的 subagent 是唯一还默认拿到这个工具的地方，见
        FeatureAgent._subagent_tools）。
        """
        return list(self._tools) if self._tools is not None else []

    def _model(self):
        """惰性建底层 agent，双重检查加锁保证只建一份"""
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    self._agent = self._build()
        return self._agent

    def warm(self) -> None:
        """提前把底层 agent 建好：配置错在启动时就暴露，且不留给多线程去抢着建"""
        if self.llm:
            self._model()

    @contextmanager
    def budget_scope(self):
        """把这一圈调用算作**同一条候选**的请求预算（见 _invoke）

        跑批入口在每次 graph.invoke 外面套一层（见 AgentSet.budget_scope）：一条候选状态
        展开出的 a/b 两侧、以及失败后的重试，共用 max_requests 这份预算，而不是让每个候选
        各拿一份、再被候选数放大成整批的倍数。

        作用域用 ContextVar 下发而不是 threading.local：langgraph 把同步节点提交到线程池
        执行，threading.local 不会跟过去（见 _BudgetScope）。ContextVar 也不跨候选串联：
        --workers N 时每条候选在各自的 run_state 里进入/退出，互不干扰。

        已在某个作用域内（AgentSet 逐个 agent 进入）就复用那一份，不另起炉灶——计数按
        agent 实例分开，同一条候选里各 agent 的预算各算各的。

        不在作用域内的调用（直接 run()、单独建 agent 做对照实验）不套候选预算——那时没有
        「一条候选」可言，按实例累计反而会误伤一次性调用。整批预算不受作用域影响，始终生效。
        """
        if _CURRENT_BUDGET.get() is not None:
            yield
            return
        token = _CURRENT_BUDGET.set(_BudgetScope())
        try:
            yield
        finally:
            _CURRENT_BUDGET.reset(token)

    def _invoke(self, content: str):
        """跑一次底层 agent 并取结构化输出。异常不吞，由 run() 决定怎么如实记 source

        取不到结构化输出时抛 StructuredOutputMissing 而不是返回 None——这一处是 LLM agent
        共用的关口，理由见该异常类。顺带把模型的原始回复带进异常：它平时落不了盘，出岔子时
        是事后唯一能查的东西（见 _reply_digest）。

        用 .get 而不是 []：键缺失同样该走这条兜底，而不是让 run() 的兜底把 KeyError 记成
        一句「调用失败」——那样就看不出到底是模型没给判定、还是返回形状变了。

        顺带数模型往返数（见 stats）：返回的 messages 里有几条 AI 消息就是几次往返（工具调用
        与结构化输出各算一次）。替身返回形状（只有 structured_response、没有 messages）按 1 次
        算——它毕竟顶替了一次调用。

        注意这是**本层可见**的往返数：deep agent 的 subagent 在 task 工具调用内部各跑各的，
        它们的往返不在主 agent 的 messages 里，这里数不到（deep 路径的数字因此偏小）。

        失败（请求抛异常）另外记一次 failed：往返数只数拿到了回复的那些，而抛异常说明这一问
        发出去了、没成（403、超时、连不上都算）。漏记这一笔的话，一整批被接口挡下时 stats 会
        报「0 次往返」，与事实正好相反——收尾的调用账要能一眼分出「没发过请求」与「发过、
        全被拒」，见 spatial_runner._print_agent_stats。首次失败的原因也留一份（截断），
        免得为了看这一句还得去翻某条记录的 status_reason。
        """
        # 预算检查放在**每次实际尝试之前**（A3）：超预算就不发，而不是发了再记账。
        # 重试也是请求，必须同样扣预算——否则 BATCH_MAX_REQUESTS=1 时一次超时仍会发出 7 次请求。
        # 两层预约要么都成功、要么都不留痕：候选预算先约，整批预算拒绝时把候选名额退回去，
        # 避免“实际没发请求却白扣一个候选名额/整批名额”。
        def reserve_budget() -> None:
            with self._stats_lock:
                scope = _CURRENT_BUDGET.get()
                candidate_reserved = False
                if scope is not None:
                    if not scope.reserve(self, self.max_requests):
                        self._stats["budget_exceeded"] += 1
                        raise RequestBudgetExceeded(
                            f"{self.NAME or self.__class__.__name__} 本候选已达请求预算 {self.max_requests} 次，"
                            "本次不再发请求"
                        )
                    candidate_reserved = True
                if self.batch_budget is not None:
                    try:
                        self.batch_budget.reserve(self.NAME or self.__class__.__name__)
                    except RequestBudgetExceeded:
                        if candidate_reserved:
                            scope.release(self)
                        self._stats["budget_exceeded"] += 1
                        raise

        sent_any = False
        last_error = None

        def invoke_once():
            nonlocal sent_any, last_error
            reserve_budget()
            sent_any = True
            try:
                return self._model().invoke({"messages": [{"role": "user", "content": content}]})
            except Exception as exc:
                last_error = exc
                raise

        started = time.perf_counter()
        call_with_retry = getattr(agent_factory, "call_with_retry", None)
        is_retryable_error = getattr(agent_factory, "is_retryable_error", lambda exc: False)
        try:
            # 重试只覆盖可恢复错误（超时/连接/限流/5xx），鉴权与参数错误直接抛出。
            # SDK 层的 max_retries 已在 agents.get_model 里置 0，避免两层重试相乘。
            if call_with_retry is None:
                result = invoke_once()
            else:
                result = call_with_retry(
                    invoke_once,
                    policy=self.retry_policy,
                    on_retry=lambda exc, attempt, delay: self._note_retry(),
                )
        except RequestBudgetExceeded as e:
            # 如果被拦下的是首发，什么都没发出，只记 budget_exceeded；
            # 如果此前已经发过并失败过（例如超时后重试被预算拦下），那一次逻辑调用
            # 确实发生过，必须保留 failed/requests/elapsed 与首个失败原因，不能整笔漏记。
            if sent_any:
                with self._stats_lock:
                    self._stats["failed"] += 1
                    self._stats["requests"] += 1
                    self._stats["elapsed_seconds"] += time.perf_counter() - started
                    if self._stats.get("first_error") is None:
                        cause = last_error or e
                        self._stats["first_error"] = f"{cause!r}"[:_FIRST_ERROR_CHARS]
            raise
        except Exception as e:  # 记账后原样抛出：怎么处理失败由各 run() 决定（见 RealityAgent._ask）
            with self._stats_lock:
                self._stats["failed"] += 1
                self._stats["requests"] += 1
                self._stats["elapsed_seconds"] += time.perf_counter() - started
                if self.retry_policy.get("max_retries", 0) > 0 and is_retryable_error(e):
                    self._stats["retry_exhausted"] += 1
                if self._stats.get("first_error") is None:
                    # 只留第一条：整批失败时它们通常是同一个原因，留一条就够判案，
                    # 且 403 这类异常的 repr 里裹着整个响应头（几千字符），必须截断
                    self._stats["first_error"] = f"{e!r}"[:_FIRST_ERROR_CHARS]
            raise
        roundtrips = sum(1 for m in (result.get("messages") or []) if getattr(m, "type", "") == "ai") or 1
        with self._stats_lock:
            self._stats["roundtrips"] += roundtrips
            self._stats["requests"] += 1
            self._stats["elapsed_seconds"] += time.perf_counter() - started
        verdict = result.get("structured_response")
        if verdict is None:
            raise StructuredOutputMissing(_reply_digest(result.get("messages") or []))
        return verdict

    # -- 按 message 的缓存 --------------------------------------------------
    def _cache_get(self, content: str) -> dict | None:
        """命中就回一份深拷贝：调用方把它交给落盘与统计，缓存里那一份谁都不许改写

        用 deepcopy 而不是 dict(...)：feature 的结果是嵌套 dict（features 底下还有各类的
        取值列表），浅拷贝会让调用方与缓存共享同一批内层对象。
        """
        with self._verdict_lock:
            hit = self._verdicts.get(content)
        return copy.deepcopy(hit) if hit is not None else None

    def _cache_put(self, content: str, result: dict) -> None:
        """只记成功的结果（见 CACHEABLE_SOURCES 里各 agent 的名单）"""
        if result.get("source") not in self.CACHEABLE_SOURCES:
            return
        with self._verdict_lock:
            self._verdicts[content] = copy.deepcopy(result)

    def _cached_or_ask(self, content: str, ask):
        """查缓存 → 未命中则问一次并（按 source 决定是否）入缓存

        并发保护是两层的：查缓存走 _verdict_lock；未命中时先取该 message 的在途锁再问，
        锁内二次查缓存——同一条语料的多个候选同时首问时，只有一个线程真发请求。
        """
        cached = self._cache_get(content)
        if cached is not None:
            self._note_cache_hit()
            return cached
        with self._flight_guard:
            lock = self._flight.setdefault(content, threading.Lock())
        with lock:
            cached = self._cache_get(content)  # 等锁期间别人可能已经问完
            if cached is not None:
                self._note_cache_hit()
                return cached
            result = ask(content)
            self._cache_put(content, result)
            return result

    def _note_cache_hit(self) -> None:
        with self._stats_lock:
            self._stats["cache_hits"] += 1

    def _note_retry(self) -> None:
        """记一次“决定重发”（含最终成功的那几次），不改变 roundtrips/failed 口径"""
        with self._stats_lock:
            self._stats["retries"] += 1

    def stats(self) -> dict:
        """本次跑批的调用统计（跑批入口收尾汇总用）：往返数、失败数、缓存命中数

        roundtrips 只数本层可见的往返（见 _invoke：deep agent 的 subagent 往返不在主 agent
        的 messages 里，数不到）；cache_hits 是被按 message 缓存省下的重复问，不花往返。

        roundtrips 与 failed 相加才是**发出过的调用数**：前者是拿到回复的，后者是抛异常的。
        只看 roundtrips 会把「请求发出去但全被拒」误读成「什么都没发」——所以 failed 与
        first_error 都在这里如实报出来（没失败过的 agent 不会带 first_error 这个键）。
        requests 就是这两个数之和，省得调用方再算一遍。

        A5 另给两项成本口径：elapsed_seconds（这些调用花掉的墙上时间之和）与
        budget_exceeded（因超预算而没发出的请求数）。拿它们与有效句对数一起看，才能算出
        「每个唯一且可用句对的成本」——只比「往返数」时，不同模式的口径并不可比
        （deep 的子 agent 往返不在主 agent 的 messages 里，见 _invoke）。

        retries 是超时/连接/限流/5xx 的重发次数（含最终成功的那几次），
        retry_exhausted 是重试用完仍失败的逻辑调用数；failed 仍按**逻辑调用**计数，
        不把每次补发都算成一条新失败。
        """
        with self._stats_lock:
            stats = dict(self._stats)
        stats["elapsed_seconds"] = round(stats.get("elapsed_seconds", 0.0), 2)
        return stats

    def describe(self) -> str:
        """启动时打印用的一句话"""
        return f"{self.NAME}：模型 {self.model_key}" if self.llm else f"{self.NAME}：未调用 LLM"

    def config_report(self) -> dict | None:
        """启动时逐张打印配置表用；不读任何配置表的 agent（semantic）返回 None

        由 agent 自己报读了哪张表，而不是跑批入口按语料类型自己算路径：--alter-rules 会把
        改写那张表换掉，只有 agent 知道它实际读了哪张。

        Returns:
            dict | None: {"table", "path", "summary", "empty", "missing", "degrade", "warnings"}
                table    逻辑表名（spatial_tools.TABLE_FILES 的键）
                path     实际路径；表由调用方注入时是一句说明
                summary  内容摘要（一句话）
                empty    表里没有已填内容（空文件、文件不存在，或判定表只写了表头）
                missing  路径上确实没有文件（注入的表给 False）
                degrade  为空时这一环节会怎样（如实交代，便于判断能不能接着跑）
                warnings 表自己报出来的配置诊断（判定表的非法格等）；跑批入口一律按「警告」打印，
                         与 empty 无关——表填了一半时也要看得见，不能静悄悄过去
        """
        return None


# --------------------------------------------------------------------------
# 四、七个 agent
# --------------------------------------------------------------------------


class RealityAgent(BaseAgent):
    """判断句中的关注词是否用于表达**现实世界中的空间场景**——全流程的第一道闸门

    方位词/趋向动词有一大批用法根本不指物理空间：「会议上」「理论上」「三天内」是引申与时间用法，
    「笑起来」「看出来」「走出困境」是时体、结果与比喻用法。那些句子照旧往下走的话，抽出的语义
    特征与 schema 只是把引申义硬套进空间框架（实测多半落到「未判出」或自由命名），白跑一遍
    后面四个 agent。本 agent 只答一件事：关注词在**这一句**里是不是在说现实空间里的位置、方向或位移。

    只判关注词，不判整句：句中没有别的空间信息、只有这一个引申用法时也判 False；反过来，
    句中别处在讲空间也不改变结论——判的是这个词。

    输出 is_spatial_scene（bool）与 reason；判不出来时 is_spatial_scene 为 None，并如实记 source：
    未调 LLM（no_llm）、没有关注词可判（no_focus）、没有句子（no_sentence）、调用失败（error）。
    四种情形都由**图**放行（见 spatial_graphs.reality_gate）：闸门只拦 is_spatial_scene=False
    这一种，它的错法只能是「漏掉几句引申用法」，不能是「把整批记录挡在门外」——一次接口抖动、
    一次 --no-llm 都不该让整轮跑批空掉，何况 --no-llm 的用途正是绕开模型验纯规则环节。

    不问重复的问题：这是一条语料一条判定，而跑批会把一条语料按规则表展开成多个候选状态、
    每个状态各过一遍这道闸门——**问的是逐字相同的一句原话**。所以判定按 user message 缓存
    （见 prefill），跑批前整批判一次、节点读缓存，实测 1782 条记录只对应 200 个原句。
    缓存只存成功判定，失败的照旧按候选重试（见 CACHEABLE_SOURCES）。

    不给工具：判「这个词说的是不是物理空间」靠读句子与常识，句法结构帮不上什么忙（关注词的下标
    已在消息首行给出，不劳模型自己去数），走 BaseAgent.tools() 的默认空表。
    提示语按语料类型分两份、与四张表同目录（见 PROMPT_SOURCE_CONFIG）。
    """

    NAME = "reality"
    INPUTS = ("sentence", "focus_word")
    OUTPUTS = ("is_spatial_scene", "reason", "source")
    CONFIG_NOTE = "config_files/<语料类型>/reality.txt（提示语，与该组四张表同目录）"
    PROMPT_FILES = (REALITY_PROMPT_FILE,)
    PROMPT_SOURCE = PROMPT_SOURCE_CONFIG
    CACHEABLE_SOURCES = ("llm",)
    """成功的判定按 user message 缓存：预填充整批判一次，节点按 message 读（见 prefill）"""

    def _build(self):
        return agent_factory.get_agent(
            self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(RealityVerdict)
        )

    def _content(self, sentence: str, focus_word: dict) -> str:
        """本 agent 的 user message

        单独抽出来是为了让缓存键与真正发出去的请求**同源**：预填充按它去重、run() 按它查缓存，
        两处拼出不同的串就意味着缓存永远不命中（多花的那份钱还看不出来）。
        """
        return (
            f"{head_block(focus_word, corpus_type=self.corpus_type)}\n"
            f"请判断这个句子里的关注词是否用于表达现实世界中的空间场景：{sentence}"
        )

    def _ask(self, content: str) -> dict:
        """发一次请求，转成与 run() 同形的结果 dict（成功与失败两种形状）"""
        try:
            verdict: RealityVerdict = self._invoke(content)
        except Exception as e:  # 单条记录失败不该带走整批，如实记下原因（图会放行，见 reality_gate）
            return {"is_spatial_scene": None, "source": "error", "reason": f"调用失败：{e!r}"}
        return {"is_spatial_scene": verdict.is_spatial_scene, "reason": verdict.reason, "source": "llm"}

    def prefill(self, queries: Iterable[tuple[str, dict | None]], workers: int = 1) -> dict:
        """跑批前把整批原句判一遍（去重），节点随后按 message 读缓存

        与 spatial_tools.prefill_ltp 同一套做法、同一个理由：同一件判定的重复请求应当在进入
        线程池之前去掉。一条语料按规则表展开成多个候选状态、每个状态都要过这道闸门，而它们
        问的是**逐字相同**的一句原话——不去重就是几倍的调用（实测 1782 条记录只对应 200 个原句，
        其中 1283 条记录只对应 157 个原句）。

        queries 是 [(原句, 关注词), ...]，由图自己声明哪些记录会走到闸门（见
        spatial_graphs.GraphSpec.gate_queries）：改写未命中的记录不经过闸门，就不该判。

        Args:
            queries (Iterable[tuple[str, dict | None]]): 去重前的待判清单
            workers (int): 并发数，取跑批的 --workers（模型由跑批入口提前 warm 好）

        Returns:
            dict: {"queries", "unique", "asked", "failed", "skipped"}
                queries  传进来多少条（= 会走到闸门的状态数）
                unique   去重后真正要判的原句数
                asked    实际发出去的请求数（已判过的那部分不再问）
                failed   失败且未入缓存的数量（这些会在各自候选上再问一次）
                skipped  没有句子/关注词、无从判断因而未发的数量
        """
        stat = {"queries": len(queries), "unique": 0, "asked": 0, "failed": 0, "skipped": 0}
        if not self.llm:
            return stat
        contents: dict[str, None] = {}  # 去重且保序（dict 自 3.7 起保插入序）
        for sentence, focus_word in queries:
            if not sentence or not (focus_word or {}).get("word"):
                # run() 对这两种情形本来就不发请求（无句子 / 无关注词），这里也别算进待判清单
                stat["skipped"] += 1
                continue
            contents[self._content(sentence, focus_word)] = None
        stat["unique"] = len(contents)
        todo = [content for content in contents if self._cache_get(content) is None]
        stat["asked"] = len(todo)
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for content, result in zip(todo, pool.map(self._ask, todo)):
                self._cache_put(content, result)
                if result.get("source") != "llm":
                    stat["failed"] += 1
                done += 1
                if done % 10 == 0:
                    print(f"  reality 预填充已判 {done}/{len(todo)}", flush=True)
        return stat

    def run(self, sentence: str, focus_word: dict) -> dict:
        """判这一句里的关注词是不是在说现实空间；判不了就给 None 并记下是哪种判不了

        四种 source 分开记而不是都回 None：汇总里要能看出「这一批有多少条压根没判」
        （见 spatial_graphs._tally_reality），混成一种就分不清「没问模型」与「模型说判不了」了。

        先查缓存（跑批前预填充填的，见 prefill）：命中就不发请求，返回的仍是 source="llm"
        的那份判定——判定本身没变，变的只是「这一问有没有重复问过」。图被单独 invoke
        （状态里没有预填充这一说）时缓存是空的，行为与从前逐字相同。
        """
        # llm 这一关排在最前（同 SemanticAgent）：--no-llm 下每条记录的 source 都该是 no_llm，
        # 那是「没问过模型」这个事实，而不是某条记录的数据有问题——后者会把它盖成 no_sentence。
        if not self.llm:
            return {"is_spatial_scene": None, "source": "no_llm", "reason": "未调用 LLM（--no-llm）"}
        word = (focus_word or {}).get("word")
        if not sentence:
            return {
                "is_spatial_scene": None,
                "source": "no_sentence",
                "reason": "没有可判的句子（语料记录的 sentence 字段不可用）",
            }
        if not word:
            return {
                "is_spatial_scene": None,
                "source": "no_focus",
                "reason": "这条记录没有取到关注词，无从判断是哪个词在表达空间",
            }
        return self._cached_or_ask(self._content(sentence, focus_word), self._ask)

    def describe(self) -> str:
        tail = f"模型 {self.model_key}" if self.llm else "未调用 LLM"
        return f"reality：{tail}（提示语 {self.prompt_paths[REALITY_PROMPT_FILE]}）"


class AlterAgent(BaseAgent):
    """按 index 定位切片中的关注词，用规则表里各组的 replacements 逐一替换，得到一批待选句。

    规则表每组形如 {"words": [...], "replacements": [...]}：关注词落在某组的 words 里，
    就用该组的 replacements 替换（排除关注词本身），见 spatial_tools.alter_sentence。

    句内**每个**目标词位置各出一批候选（见 expand_positions）：一条记录只改一个位置，其余目标词
    原样保留作参照。哪个词算目标词由 spatial_tools.pick_focus_words 定；位置之间不组合。

    纯规则，不调 LLM。rules 可由构造参数注入，于是同一个类能按不同规则表建多份供对照实验
    （spatial_runner.py --alter-rules 走的就是这条路）。构造时按 st.validate_alter_rules 校验，
    填错当场抛错。

    规则表按语料类型取：config_files/<类型目录>/alter_rules.json5（见 st.config_path），
    由 --corpus-type 决定、构造期注入。不给 --alter-rules 时读的就是本组那张表；
    --alter-rules 给了路径则以它为准（换表跑对照实验）。本组表为空（文件缺失或 groups 空）
    时一条都不改写，跑批启动时会打一条告警（见 spatial_runner._print_config_tables）。
    """

    NAME = "alter"
    INPUTS = ("sentence_slice", "focus_word")
    OUTPUTS = ("slices", "sentences", "applied", "detail")
    NEEDS_LLM = False
    CONFIG_NOTE = "config_files/<语料类型>/alter_rules.json5（类型见 --corpus-type，默认 localization）；--alter-rules 可另行指定路径"

    def __init__(self, *, rules: dict | None = None, rules_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if rules is not None:
            # 规则表来源只用于 describe() 与启动期告警：注入了非默认的表时，能一眼看出这次跑的是哪张。
            # 只给了 rules 没给 rules_path 时（注入通道，见 build_agents 的 overrides）不能说成默认路径
            # ——那个文件多半既不存在、也不是这份规则的来处。
            self.rules = rules
            self.table_path = rules_path or "调用方注入的规则表，无路径"
        else:
            self.table_path = str(rules_path or st.config_path("alter_rules", self.corpus_type))
            self.rules = st.load_alter_rules(self.table_path)
        # 表填错要在构造 agent（启动）时当场报，不能等跑到某条记录才吞掉——两条分支都走这里，
        # 默认读表、--alter-rules 注入、verify 注入规则一并覆盖（见 st.validate_alter_rules）。
        st.validate_alter_rules(self.rules, source=self.table_path)

    @property
    def rules_path(self) -> str:
        """兼容旧名（spatial_runner 的告警、文档里都这么叫）：与 table_path 同一个值"""
        return self.table_path

    def run(self, sentence_slice: list[str], focus_word: dict) -> dict:
        """按**一个**位置改写出全部待选句：slices 是"词的列表的列表"，sentences 是对应的句子字符串

        句内多个位置各出一批候选是 expand_positions 的事，本方法只管一个位置（口径见
        spatial_tools.alter_sentence：只动 index 处的那个 token）。
        """
        slices, detail = st.alter_sentence(sentence_slice, focus_word or {}, self.rules)
        return {
            "slices": slices,
            "sentences": [st.join_slice(s) for s in slices],
            "applied": bool(detail["applied"]),
            "detail": detail,
        }

    def expand_positions(
        self, sentence_slice: list[str], focus_words: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """把句内**每个**目标词位置各展开成一批待选句，摊平成一条待选句一项

        位置之间**不组合**：一条记录只改一个位置，句里其余目标词原样保留作参照。所以
        "一条语料展开成几个待选句"= 各位置的候选数之和，`candidate_index` 在这份**全展开**上
        连续编号——排序键 (line_no, candidate.index) 要求同一行内唯一（见
        spatial_graphs.record_order_key），按位置重新从 0 编号会撞键。

        每项形如：
            {"candidate_index": 0, "candidate_total": 4, "word": "上", "span": [2],
             "focus_index": 2, "focus_order": 1, "candidate_order": 3,
             "focus_word": {"index": 2, "word": "上", "kind": "方位词", ...},
             "slice": [...], "sentence": "在硬件中说…", "replacement": "中",
             "source": "groups[0]", "group": ["上", "中", "里"]}   # group 是该组的 words 命中集

        focus_index / focus_word 标明"这一条改的是哪个位置上的哪个词"：同一条语料里
        `上@3` 与 `上@5` 会有各自的一批候选，只有带着位置才分得开（见 spatial_graphs.alter_node）。

        focus_order / candidate_order 是拼记录 id 用的两个**从 1 起**的层级序号（见
        spatial_graphs._record_id）：前者是"句内第几个目标词位置"（按本方法的循环顺序，
        展开不出候选的位置也占号，所以同句 id 会跳号），后者是"该位置自己的第几个替换句"
        （按规则表顺序）。四个编号各是各的口径，别混：candidate_index 是全展开上从 0 起的
        连续号（排序键用），focus_index 是句内第几个 token（0 基下标，"是不是同一位置"看它）
        ——后两个只用来拼 id（候选 dict 上的这两个键不直接落盘，落盘的是它们拼出的顶层 id）。

        returns 的 details 与**位置**一一对应（不是与候选一一对应），形如
        [{"focus_word": ..., "detail": ...}]，detail 是 spatial_tools.alter_sentence 的原样
        改写详情；某位置展开不出候选时它照样在 details 里（调用方靠 detail["reason"] 记原因），
        只是不进候选表——它没有可写的记录，一个坏位置不连坐其它位置。

        跑批前的展开（spatial_graphs.alter_init_states）、节点里取自己那一条、预填充清点句子
        三处都走本方法，保证"这条语料展开成几个待选句"与"记录里选中的是哪个候选"不会对不上。
        """
        candidates: list[dict] = []
        details: list[dict] = []
        focus_order = 0
        for focus_word in focus_words or []:
            if not isinstance(focus_word, dict):
                continue
            # 每个目标词位置都占一个号：第二段要回答「句内第几个目标词位置」，
            # 不是「第几个展开出候选的位置」。空位置照样占号，所以同句 id 会出现跳号。
            focus_order += 1
            result = self.run(sentence_slice, focus_word)
            detail = result["detail"]
            details.append({"focus_word": focus_word, "detail": detail})
            if not detail["candidates"]:
                # 这个位置展开不出候选：不进候选表，但上面已经占过号了
                continue
            for candidate_order, (candidate, slice_, sentence) in enumerate(
                zip(detail["candidates"], result["slices"], result["sentences"]), start=1
            ):
                candidates.append(
                    {
                        "candidate_index": len(candidates),
                        # 全展开完才知道总数，循环后统一回填（只此一处，免得就地算错）
                        "candidate_total": None,
                        "word": detail["word"],
                        "span": detail["span"],
                        "focus_index": focus_word.get("index"),
                        # 从 1 起的两个层级序号，只用来拼记录 id（见 spatial_graphs._record_id）
                        "focus_order": focus_order,
                        "candidate_order": candidate_order,
                        "focus_word": focus_word,
                        "slice": slice_,
                        "sentence": sentence,
                        **candidate,
                    }
                )
        for candidate in candidates:
            candidate["candidate_total"] = len(candidates)
        return candidates, details

    def expand(self, sentence_slice: list[str], focus_word: dict) -> tuple[list[dict], dict]:
        """只按一个位置改写（expand_positions 的单位置薄包装）

        保留给"只关心一个关注词"的调用点；跑批链路走 expand_positions（句内每个位置各一批）。
        返回 (待选句列表, 该位置的 detail)；没有待选句时列表为空，原因在 detail["reason"]。
        """
        candidates, details = self.expand_positions(sentence_slice, [focus_word or {}])
        return candidates, (details[0]["detail"] if details else {})

    def plan_all_positions(self, sentence_slice: list[str], focus_words: list[dict]) -> list[str]:
        """只算出全部位置的全部待选句、无任何副作用

        供图在跑批之前统计"整批会出现哪些句子"，以便一次性预填充 LTP（见 spatial_graphs.*_plan）。
        与 run()/expand_positions() 同源，保证预填充的句子和图里真正会用到的一致：漏掉任何一个
        待选句，它就会在线程池里第一次遇到时才做句法分析，而推理在锁外，多线程下就是并发调同一份 LTP。
        """
        candidates, _ = self.expand_positions(sentence_slice, focus_words)
        return [candidate["sentence"] for candidate in candidates]

    def plan_all(self, sentence_slice: list[str], focus_word: dict) -> list[str]:
        """只算出一个位置的全部待选句（plan_all_positions 的单位置薄包装）"""
        return self.plan_all_positions(sentence_slice, [focus_word or {}])

    def describe(self) -> str:
        groups = self.rules.get("groups") or []
        return f"alter：替换规则 {len(groups)} 组（{self.rules_path}）"

    def config_report(self) -> dict:
        groups = self.rules.get("groups") or []
        return {
            "table": "alter_rules",
            "path": self.rules_path,
            "summary": f"{len(groups)} 组替换规则",
            "empty": not groups,
            "missing": _table_missing(self.table_path),
            "warnings": [],
            "degrade": "所有记录都会在改写环节停下（alter_missed），包括本闸门在内的五个 LLM agent 一次都不调",
        }


class SemanticAgent(BaseAgent):
    """判断单句的语法是否通顺、空间语义是否异常——两项一次问、各带理由

    两道判据合出 is_fluent = grammar_ok and spatial_ok，闸门只读它（见 sg.pair_semantic_gate）；
    拆开存是为了事后能分清这条记录是被语法挡下的还是被空间语义挡下的。两项在**同一次调用**里给出，
    所以调用次数与从前逐字相同（每侧仍是一次），多出来的只是输出里的两个字段。

    句法证据直接拼进消息（见 syntax_block）、不绑工具：判语法通不通、空间语义顺不顺，先读现成的
    分词与依存结构比让模型自己调一次 parse_sentence 又快又省一次往返（措辞与 all_in_one 那条一致）。

    落盘与返回形状由 semantic_verdict_result / semantic_blank 一处合成：all_in_one.py 那条对照
    链路用同一对函数，两边的键集与合出 is_fluent 的算式因此不可能各走各的。

    a 侧的 message 是一条语料所有候选的纯函数，按 message 缓存（见 CACHEABLE_SOURCES）：一条语料
    展开成 ~9.5 个候选时，同一句只问一次。
    """

    NAME = "semantic"
    INPUTS = ("sentence", "focus_word")
    OUTPUTS = (
        "is_fluent",
        "grammar_ok",
        "spatial_ok",
        "reason",
        "grammar_reason",
        "spatial_reason",
        "source",
    )
    CONFIG_NOTE = "无"
    PROMPT_FILES = ("semantic",)
    CACHEABLE_SOURCES = ("llm",)
    """成功的判定按 user message 缓存：a 侧输入只由语料决定，原样复用（b 侧句子不同、键不同）"""

    def _build(self):
        return agent_factory.get_agent(
            self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(SemanticVerdict)
        )

    def _content(self, sentence: str, focus_word: dict) -> str:
        """本 agent 的 user message；syntax_block 取不到证据时那两行整块省略"""
        block = syntax_block(sentence)
        syntax_lines = f"\n句法证据：\n{block}" if block else ""
        return (
            f"{head_block(focus_word, corpus_type=self.corpus_type)}\n"
            f"句子：{sentence}"
            f"{syntax_lines}\n"
            f"请判断这个句子：语法是否通顺，空间语义是否存在异常。"
        )

    def _ask(self, content: str) -> dict:
        """发一次请求，转成与 run() 同形的结果 dict（成功与失败两种形状）"""
        try:
            verdict: SemanticVerdict = self._invoke(content)
        except Exception as e:  # 单条记录失败不该带走整批，如实记下原因
            return semantic_blank("error", f"调用失败：{e!r}")
        return semantic_verdict_result(
            verdict.grammar_ok, verdict.grammar_reason, verdict.spatial_ok, verdict.spatial_reason
        )

    def run(self, sentence: str, focus_word: dict) -> dict:
        if not self.llm:
            return semantic_blank("no_llm", "未调用 LLM（--no-llm）")
        return self._cached_or_ask(self._content(sentence, focus_word), self._ask)


class FeatureAgent(BaseAgent):
    """抽取单句的语义特征

    类别表填了默认走**单次调用**：把各类的判据拼进一份系统提示语，一次调用直接产出全部特征字段
    （每类一个列表字段，见 _feature_model）。deep=True 时改用原来的 deep agent（每类一个 subagent
    各自抽取、主 agent 汇总）——每次约 13 次模型往返，只在需要对照或 flat 抽得不好时开。

    表为空则退化成单个 agent 自由抽取，并把 source 标成 unfilled_config，
    提醒这不是按既定类别抽的（这种情形 deep 无效，一条老路走到底）。

    __init__ 只读表、校验与渲染提示语，不建模型（deep agent 建起来尤其贵）；--no-llm 下一次都不建。
    """

    NAME = "feature"
    INPUTS = ("sentence", "focus_word")
    OUTPUTS = ("features", "source", "reason")
    CONFIG_NOTE = "config_files/<语料类型>/feature_classes.json5（类型见 --corpus-type，默认 localization）"
    PROMPT_FILES = (
        FEATURE_FLAT_PROMPT_FILE,
        FEATURE_PROMPT_FILE,
        FEATURE_UNFILLED_PROMPT_FILE,
        FEATURE_CLASS_PROMPT_FILE,
        FEATURE_CLASS_USAGE_FILE,
        FEATURE_CLASS_USAGE_FLAT_FILE,
    )
    """都在这构造期读入，表为空时用不上的那几份也顺带校验（配置错早暴露）"""
    CACHEABLE_SOURCES = ("config", "unfilled_config")
    """成功抽到的特征按 user message 缓存：a 侧输入只由语料决定，原样复用（b 侧句子不同、键不同）"""

    def __init__(self, *, feature_classes: list[dict] | None = None, deep: bool = False, **kwargs):
        # deep 必须在这里显式声明：build_agents 的 overrides（--feature-deep）按构造参数注入，
        # 让 **kwargs 原样落到 BaseAgent.__init__ 会因未知参数直接 TypeError。
        super().__init__(**kwargs)
        self.deep = bool(deep)
        if feature_classes is not None:
            self.feature_classes = feature_classes
            self.table_path = "调用方注入的类别表，无路径"
        else:
            self.table_path = str(st.config_path("feature_classes", self.corpus_type))
            self.feature_classes = st.load_feature_classes(self.table_path)
        _validate_feature_classes(self.feature_classes, source=self.table_path)
        self.table_source = "config" if self.feature_classes else "unfilled_config"
        # 渲染放构造期而不是 _build()：llm=False 时 _build() 永不调用，放里面就没有任何不碰模型的
        # 路径能覆盖它；而且占位符写错属于配置错误，与「表填错」同一口径，应该启动时就报。
        if self.feature_classes:
            mode = "deep" if self.deep else "flat"
            self.system_prompt, self.class_prompts = _render_feature_prompts(
                self.feature_classes,
                self.prompts[FEATURE_PROMPT_FILE if mode == "deep" else FEATURE_FLAT_PROMPT_FILE],
                self.prompts[FEATURE_CLASS_PROMPT_FILE],
                mode=mode,
                usage=self.prompts[
                    FEATURE_CLASS_USAGE_FILE if mode == "deep" else FEATURE_CLASS_USAGE_FLAT_FILE
                ],
            )
        else:
            self.system_prompt = self.prompts[FEATURE_UNFILLED_PROMPT_FILE]
            self.class_prompts = []

    def _build(self):
        if not self.feature_classes:
            return agent_factory.get_agent(
                self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(FreeFeatures)
            )
        if not self.deep:
            return agent_factory.get_agent(
                self.model_key,
                self.tools(),
                self.system_prompt,
                response_format=ToolStrategy(_feature_model(self.feature_classes)),
            )
        return _build_deep_feature_agent(
            self.model_key,
            self.feature_classes,
            self.tools(),
            self._subagent_tools(),
            self.system_prompt,
            self.class_prompts,
        )

    def _subagent_tools(self) -> list:
        """deep 模式 subagent 的工具：这是唯一还默认给 parse_sentence 的模型面

        deep agent 的 subagent 拿到的不是完整消息，句法证据没地方内联（主 agent 转述的 task
        文本只带关注词与句子），所以保留工具让它自己去取；调用方显式注入 tools 时以注入为准。
        """
        return self.tools() or [st.parse_sentence]

    def _content(self, sentence: str, focus_word: dict) -> str:
        """本 agent 的 user message（flat 与 unfilled 两条路共用；syntax_block 取不到就整块省略）

        deep 路径用不上它：主 agent 的输入是消息里的关注词与句子，句法证据由 subagent 自己取。
        """
        block = syntax_block(sentence)
        syntax_lines = f"\n句法证据：\n{block}" if block else ""
        return (
            f"{head_block(focus_word, corpus_type=self.corpus_type)}\n"
            f"句子：{sentence}"
            f"{syntax_lines}\n"
            f"请抽取这个句子的语义特征。"
        )

    def _ask(self, content: str) -> dict:
        """发一次请求，转成与 run() 同形的结果 dict（成功与失败两种形状）"""
        try:
            data = self._invoke(content).model_dump()
        except Exception as e:
            return {"features": {}, "source": "error", "reason": f"调用失败：{e!r}"}

        # 类别表已填时 data 直接就是特征字段；未填（FreeFeatures）时特征装在 features 键下
        features = data.get("features") if set(data) == {"features"} else data
        features = features or {}
        # 结构化输出之后的业务校验（A2）：形状对了不代表取值对，闭集越界与未定义键都要留痕。
        # 校验结果只**记录**不改值——要不要纠正由人工/复核流程决定，这里不猜模型的意思。
        validation = validate_feature_values(features, self.feature_classes)
        return {
            "features": features,
            "source": self.table_source,
            "reason": "",
            "validation": validation,
        }

    def run(self, sentence: str, focus_word: dict) -> dict:
        if not self.llm:
            return {"features": {}, "source": "no_llm", "reason": "未调用 LLM（--no-llm）"}
        return self._cached_or_ask(self._content(sentence, focus_word), self._ask)

    def describe(self) -> str:
        n = len(self.feature_classes)
        tail = "" if n else "（未填写，自由抽取）"
        mode = "deep" if (n and self.deep) else "单次调用"
        # 模式两种都印（--no-llm 也印）：flat / deep 渲染的是两份不同的 system prompt，
        # 启动时看不出这次读的是哪一份，对照实验就说不清跑的是哪条路
        if not self.llm:
            return f"feature：未调用 LLM；特征类别 {n} 类（{mode}）{tail}（{self.table_path}）"
        return f"feature：模型 {self.model_key}；特征类别 {n} 类（{mode}）{tail}（{self.table_path}）"

    def config_report(self) -> dict:
        n = len(self.feature_classes)
        return {
            "table": "feature_classes",
            "path": self.table_path,
            "summary": f"{n} 类特征",
            "empty": not n,
            "missing": _table_missing(self.table_path),
            "warnings": [],
            "degrade": "feature_agent 退化为自由抽取，结果的 features_source 记成 unfilled_config",
        }


class SchemaAgent(BaseAgent):
    """取单句的 schema：先按关注词收窄、再查表；都未命中时由 LLM 从该类的候选里挑一个最接近的

    schema 名是「组名 + 编号」（如「上12」「下1」），所以选图式的两步顺序是固定的：

    1. 按上游传下来的关注词排除别的**词条**的图式（见 st.schema_candidates）。词条可以用
       words 声明自己覆盖哪些词形（外面/外边/外头 都归「外」），关注词落在
       {组名} ∪ words 里就取这一组，名字仍用组名——判定表因此只需按类写一份；
    2. 在剩下的图式里按语义特征匹配（见 st.lookup_schema），命中即用，不再调模型。

    两步都没命中才动用 LLM，且**只让它在候选里挑**（source="llm"）：名字落在两套编号之间
    的话，判定环节按名字查表就再也对不上，schema 也就白抽了。

    唯一例外是该关注词所属的词条下一条图式都没有（表还没填、或这个词还没开始收，只有 words
    占位）：没有候选可挑，退化为自由命名（source="llm_free"，旧口径），名字仍要求符合
    「组名 + 编号」的格式——自由命名与候选名单共用 class_word 这一个前缀（见 run），两处不同源
    的话，别名那一路（关注词「外面」归「外」）的自由命名会全被拒成 llm_invalid；
    合不上格式、或挑了候选名单外的名字，记 source="llm_invalid" 并让 name 为 None——
    宁可如实留空，也不把一个下游对不上的名字写进结果。空表曾长期是本仓库的常态，
    这条退路保证它照旧跑得出可看的 schema，而不是整栏皆空。
    """

    NAME = "schema"
    INPUTS = ("sentence", "features", "focus_word")
    OUTPUTS = ("name", "reason", "source")
    CONFIG_NOTE = "config_files/<语料类型>/schema_table.json5（match 的键取自同目录的 feature_classes.json5）"
    PROMPT_FILES = ("schema",)
    CACHEABLE_SOURCES = ("llm", "llm_free")
    """成功选出的名字按 user message 缓存：a 侧输入只由语料决定，原样复用
    （查表命中在缓存之前就返回、不进缓存；llm_invalid 与 error 不入缓存，留待各候选重试）"""

    def __init__(
        self,
        *,
        table: dict | None = None,
        feature_keys: Iterable[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if table is not None:
            self.table = table
            self.table_path = "调用方注入的 schema 表，无路径"
        else:
            self.table_path = str(st.config_path("schema_table", self.corpus_type))
            self.table = st.load_schema_table(self.table_path)
        # 特征键用来校验 match 里没有写错的键：写错一个键不会报错，只会让那条图式永远匹配不上。
        # 类别表为空（feature_agent 自由抽特征、键名不定）时给空集，该项校验自动跳过。
        # 取类别表必须带上同一个 corpus_type：否则趋向动词的 schema 表会拿方位词的键去校验，
        # 轻则误报「键不存在」、重则把 feature_classes 为空当理由静默跳过校验。
        if feature_keys is not None:
            self.feature_keys = set(feature_keys)
            self.feature_table_path = "调用方给的特征键，无路径"
        else:
            self.feature_table_path = str(st.config_path("feature_classes", self.corpus_type))
            self.feature_keys = {c["key"] for c in st.load_feature_classes(self.feature_table_path)}
        st.validate_schema_table(
            self.table,
            self.feature_keys,
            source=self.table_path,
            feature_source=self.feature_table_path,
        )

    @property
    def schema_names(self) -> list[str]:
        """表里已有的全部 schema 名（平铺口径，describe/--list 用）"""
        return list(st.schema_candidates(self.table))

    def _build(self):
        return agent_factory.get_agent(
            self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(SchemaVerdict)
        )

    def run(self, sentence: str, features: dict, focus_word: dict) -> dict:
        # 适配器取不到状态里的值时会给 None，这里兜一下，避免查表时 TypeError
        features = features or {}
        word = (focus_word or {}).get("word") or ""
        # 关注词归属的词条（组名）。收窄、候选名单、自由命名的格式校验三处必须同源：
        # 别名那一路（关注词「外面」归「外」）要是各算各的，名字就会对不上。
        class_word = st.schema_group_key(self.table, focus_word)
        name, detail = st.lookup_schema(features, self.table, focus_word)
        if name:
            return {
                "name": name,
                "source": "table",
                "reason": f"查表命中 {name}（{st.format_match(detail.get('match'))}）",
            }

        candidates = st.schema_candidates(self.table, focus_word)
        miss = self._miss_note(word, class_word, candidates)
        if not self.llm:
            return {"name": None, "source": "no_llm", "reason": f"{miss}，且未调用 LLM（--no-llm）"}

        content = self._content(sentence, features, focus_word, word, class_word, miss, candidates)
        return self._cached_or_ask(content, lambda c: self._ask(c, word, class_word, candidates))

    def _ask(self, content: str, word: str, class_word: str | None, candidates: dict) -> dict:
        """发一次请求并把名字校验成 run() 的同形结果（成功与失败两种形状）

        校验挡掉的是「名字下游对不上」：候选名单外的名字、自由命名时不合格式的名字都记成
        llm_invalid 并不入缓存（见 CACHEABLE_SOURCES）——同一句话换个候选重试时还能再问一次。
        """
        try:
            verdict: SchemaVerdict = self._invoke(content)
        except Exception as e:
            return {"name": None, "source": "error", "reason": f"调用失败：{e!r}"}

        chosen = (verdict.schema_name or "").strip()
        if candidates:
            ok = chosen in candidates
            why = f"不在本次的候选名单里（{'、'.join(candidates)}）"
        else:
            prefix = class_word or word
            ok = st.is_schema_name(chosen, prefix)
            why = f"不符合「{prefix or '关注词'} + 编号」的格式"
        if not ok:
            return {
                "name": None,
                "source": "llm_invalid",
                "reason": f"LLM 给出的名字 {verdict.schema_name!r} {why}；{verdict.reason}",
            }
        # 有候选可挑时是「从表里挑的」，没有候选时是「这一类下还没收录、自行命名的」，两者分开记
        return {"name": chosen, "source": "llm" if candidates else "llm_free", "reason": verdict.reason}

    def _miss_note(self, word: str, class_word: str | None, candidates: dict) -> str:
        """没走成查表这一步的原因，写进 reason 与提示语（三种情形的下一步动作完全不同）

        class_word 是关注词归属的词条名：关注词是别名（「里面」归「里」）时，两者措辞不能混用
        ——说成「表里还没有关注词「里面」的图式」是错的，表里有「里面」，它是「里」的词形。
        """
        if candidates:
            owner = f"（属「{class_word}」这一类）" if class_word and class_word != word else ""
            return f"关注词「{word}」{owner}的 {len(candidates)} 条图式都没命中该句特征"
        if class_word and class_word != word:
            return f"关注词「{word}」归入「{class_word}」这一类，但这一类下还没有编号图式"
        if word:
            return f"表里还没有关注词「{word}」的图式"
        return "表为空（且本句没有关注词可用以收窄）"

    def _content(
        self,
        sentence: str,
        features: dict,
        focus_word: dict,
        word: str,
        class_word: str | None,
        miss: str,
        candidates: dict,
    ) -> str:
        """LLM 兜底时的 user message：关注词行 + 句子 + 句法证据 + 特征 + 候选名单（或自由命名要求）

        带类名的那句必须拼在 head_block 整块**之后**：head_block 刻意把语料类型行钉在第 3 行，
        插在中间会把类型行顶到第 4 行（verify_focus_wiring._check_focus_head 就在断言这个位置）。
        句法证据同样拼在首块之后（取不到时整两行省略），选图式要看的正是关注词在句法上的角色。
        """
        block = syntax_block(sentence)
        syntax_lines = f"\n句法证据：\n{block}" if block else ""
        head = (
            f"{head_block(focus_word, corpus_type=self.corpus_type)}\n"
            f"句子：{sentence}"
            f"{syntax_lines}\n"
            f"语义特征：{json.dumps(features, ensure_ascii=False)}\n"
            f"{miss}。"
        )
        if not candidates:
            # 前缀与候选名单同源（class_word），否则「外面」这一类自由命名出的名字与查表路径对不上
            prefix = class_word or word
            style = f"，格式为「{prefix} + 编号」（如 {prefix}1）" if prefix else ""
            return head + f"请自行给出一个 schema 名{style}。"
        note = f"这些候选属于关注词所属的「{class_word}」这一类。" if class_word and class_word != word else ""
        return (
            head
            + note
            + "请从下面这些候选 schema 里挑一个最接近的，名字原样照抄（不要自造名字、不要改编号）：\n"
            f"{st.render_schema_candidates(candidates)}\n"
            "一个候选的命中条件有多组（用「／」分开）时，命中任意一组即可；都不完全吻合时挑差别最小的那个，"
            "并在 reason 里说明差在哪。"
        )

    def _group_summary(self) -> str:
        """「M 个词条（覆盖 K 个词形…）」这段摘要：词条数 + 它们声明的词形数

        顺带核对一次词表：words 里写了本语料类型词表外的词形（多半是错字）时，那个词形永远
        当不成关注词、这条配置静默失效，这里把它点出来（不在词表里只作提示，不改动其他部分，
        也不给 config_report 加键——那边逐个 agent 断言了键集合）。
        """
        schemas = self.table.get("schemas") or {}
        forms = st.schema_forms(self.table)
        vocab = st.load_focus_vocab().get(self.corpus_type or st.DEFAULT_CORPUS_TYPE, set())
        unknown = [form for form in forms if form not in vocab]
        tail = f"，其中 {len(unknown)} 个不在本语料类型词表里：{'、'.join(unknown)}" if unknown else ""
        return f"{len(schemas)} 个词条（覆盖 {len(forms)} 个词形{tail}）"

    def describe(self) -> str:
        count = len(self.schema_names)
        groups = len(self.table.get("schemas") or {})
        if not self.llm:
            return f"schema：查表 {count} 条（{self.table_path}；未调用 LLM）"
        tail = f"＋LLM 在候选里挑：{self._group_summary()}" if groups else "＋LLM 自由命名（表为空）"
        return f"schema：查表 {count} 条{tail}（{self.table_path}）"

    def config_report(self) -> dict:
        groups = len(self.table.get("schemas") or {})
        count = len(self.schema_names)
        return {
            "table": "schema_table",
            "path": self.table_path,
            "summary": f"{count} 条图式／{self._group_summary()}",
            "empty": not groups,
            "missing": _table_missing(self.table_path),
            "warnings": [],
            "degrade": "查表必然未命中，schema 交由 LLM 自由命名（source=llm_free）",
        }


class SynonymAgent(BaseAgent):
    """直接比较两句是否同义，不依赖特征抽取、图式或判定表。

    提示语只从 system_prompt/synonym.txt 读取，正文留给用户填写。读取空文件沿用
    utils.read_prompt_path 的规则（返回空字符串）；空提示语下 run() 不调用模型，
    warm() 不预热，返回 same_meaning=None、source=unfilled_prompt。

    两侧各带自己的关注词，消息开头复用 pair_head_block 的定位信息；后面只附两句文本。
    成功的判定按完整 user message 缓存，失败不缓存，沿用 BaseAgent 的并发保护与调用统计。
    """

    NAME = "synonym"
    INPUTS = ("sentence_a", "sentence_b", "focus_word_a", "focus_word_b")
    OUTPUTS = ("same_meaning", "reason", "source")
    CONFIG_NOTE = "无配置表；system_prompt/synonym.txt 由用户填写，空提示语不调模型"
    PROMPT_FILES = ("synonym",)
    CACHEABLE_SOURCES = ("llm",)

    def _build(self):
        if not self.system_prompt.strip():
            raise ValueError(f"同义判断提示语尚未填写：{self.prompt_paths['synonym']}")
        return agent_factory.get_agent(
            self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(SynonymVerdict)
        )

    def warm(self) -> None:
        if self.system_prompt.strip():
            super().warm()

    def _content(
        self, sentence_a: str, sentence_b: str, focus_word_a: dict, focus_word_b: dict
    ) -> str:
        head = pair_head_block(focus_word_a, focus_word_b, corpus_type=self.corpus_type)
        # 保留两侧首行的定位信息，去掉旧 agent 的任务说明，判断标准全部留给用户提示语。
        for role in (FOCUS_ROLE_ORIGINAL, FOCUS_ROLE_REPLACEMENT):
            head = head.replace(_FOCUS_TAIL[role], "")
        head = head.replace(FOCUS_A_ABSENT, "原句关注词：未提供。")
        head = head.replace(FOCUS_B_ABSENT, "改写句关注词：未提供。")
        head = "\n".join(line for line in head.splitlines() if line)
        return (
            f"{head}\n"
            f"原句：{sentence_a}\n"
            f"改写句：{sentence_b}"
        )

    def _ask(self, content: str) -> dict:
        try:
            verdict: SynonymVerdict = self._invoke(content)
            return {"same_meaning": verdict.same_meaning, "reason": verdict.reason, "source": "llm"}
        except Exception as e:
            return {"same_meaning": None, "reason": f"调用失败：{e!r}", "source": "error"}

    def run(
        self, sentence_a: str, sentence_b: str, focus_word_a: dict, focus_word_b: dict
    ) -> dict:
        if not self.llm:
            return {"same_meaning": None, "reason": "未调用 LLM（--no-llm）", "source": "no_llm"}
        if not self.system_prompt.strip():
            return {
                "same_meaning": None,
                "reason": f"同义判断提示语尚未填写：{self.prompt_paths['synonym']}，未调用 LLM",
                "source": "unfilled_prompt",
            }
        return self._cached_or_ask(
            self._content(sentence_a, sentence_b, focus_word_a, focus_word_b), self._ask
        )

    def describe(self) -> str:
        if not self.llm:
            status = "未调用 LLM（--no-llm）"
        elif not self.system_prompt.strip():
            status = "提示语尚未填写，未调用 LLM"
        else:
            status = f"独立同义判断，模型 {self.model_key}"
        return f"synonym：{status}（{self.prompt_paths['synonym']}）"


class JudgeAgent(BaseAgent):
    """判断两句意义是否相同：先查判定表，表里没给出结论的再由 LLM 判断

    两侧的句子、schema、
    特征、关注词由 graph 合并后一起传进来，判的就是「把 a 侧那个词换成 b 侧那个词之后，
    两句意义是否还相同」。b 侧的 schema 是拿替换词重新收窄、重新定的（见 st.replacement_focus_word），
    所以判定表里 上12 × 里1 这样的格子才对得上——两侧若共用一个关注词，b 侧会退回收窄回原词
    那一组，判定就只在同一组内自比了。

    两句的句法证据各带一份拼进消息（见 run），不绑工具。刻意不进按 message 的缓存：b 侧句子
    逐候选不同，消息也就条条不同，去重收益为零（CACHEABLE_SOURCES 留空）。
    """

    NAME = "judge"
    INPUTS = (
        "sentence_a",
        "sentence_b",
        "schema_a",
        "schema_b",
        "features_a",
        "features_b",
        "focus_word",
        "focus_word_b",
    )
    OUTPUTS = ("same_meaning", "reason", "source", "lookup")
    CONFIG_NOTE = "config_files/<语料类型>/judge_table.csv（类型见 --corpus-type，默认 localization）"
    PROMPT_FILES = ("judge",)

    def __init__(self, *, table: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        if table is not None:
            self.table = table
            self.table_path = "调用方注入的判定表，无路径"
        else:
            self.table_path = str(st.config_path("judge_table", self.corpus_type))
            self.table = st.load_judge_table(self.table_path)

    def _build(self):
        return agent_factory.get_agent(
            self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(JudgeVerdict)
        )

    def run(
        self,
        sentence_a: str,
        sentence_b: str,
        schema_a: dict,
        schema_b: dict,
        features_a: dict,
        features_b: dict,
        focus_word: dict,
        focus_word_b: dict | None = None,
    ) -> dict:
        schema_a, schema_b = schema_a or {}, schema_b or {}
        same, detail = st.lookup_judgement(schema_a.get("name"), schema_b.get("name"), self.table)
        pair = f"{detail['label_a']} × {detail['label_b']}"
        if same is not None:
            return {
                "same_meaning": same,
                "source": "table",
                "lookup": "same" if same else "different",
                "reason": f"查表命中：{pair} = {detail['cell']}（{'意义相同' if same else '意义不同'}）",
            }
        # 查表没给出直接结论：○（交 LLM）、未命中、非法空格三种，都走 LLM 兜底。
        # 结果里的 lookup 记「落在哪一类」：○ 那一档最终确实由 LLM 判，故记成 llm；
        # miss 与 illegal 分开记，前者是表没管这一对（正常），后者是表该填没填（配置问题）。
        lookup = "llm" if detail["source"] == "defer" else detail["source"]
        situation = {
            "llm": f"判定表标了 {st.JUDGE_CELL_LLM}（交大模型做进一步判断）",
            "miss": "判定表未收录这两个 schema",
            "illegal": f"判定表里 {pair} 这一格留空或符号认不出（非法）",
        }[lookup]
        if not self.llm:
            return {
                "same_meaning": None,
                "source": "no_llm",
                "lookup": lookup,
                "reason": f"查表没有直接结论（{situation}），且未调用 LLM（--no-llm）",
            }

        block_a, block_b = syntax_block(sentence_a), syntax_block(sentence_b)
        syntax_a = f"\n原句句法证据：\n{block_a}" if block_a else ""
        syntax_b = f"\n改写句句法证据：\n{block_b}" if block_b else ""
        content = (
            f"{pair_head_block(focus_word, focus_word_b, corpus_type=self.corpus_type)}\n"
            f"原句：{sentence_a}"
            f"{syntax_a}\n"
            f"原句 schema：{schema_a.get('name')}（来源 {schema_a.get('source')}）\n"
            f"原句语义特征：{json.dumps(features_a or {}, ensure_ascii=False)}\n"
            f"改写句：{sentence_b}"
            f"{syntax_b}\n"
            f"改写句 schema：{schema_b.get('name')}（来源 {schema_b.get('source')}）\n"
            f"改写句语义特征：{json.dumps(features_b or {}, ensure_ascii=False)}\n"
            f"判定表情况：{situation}。\n"
            "请判断这两句话的意义是否相同。"
        )
        try:
            verdict: JudgeVerdict = self._invoke(content)
        except Exception as e:
            return {"same_meaning": None, "source": "error", "lookup": lookup, "reason": f"调用失败：{e!r}"}
        return {
            "same_meaning": verdict.same_meaning,
            "source": "llm",
            "lookup": lookup,
            "reason": verdict.reason,
        }

    def describe(self) -> str:
        table = self.table or {}
        tail = "＋LLM 兜底" if self.llm else "（未调用 LLM）"
        return (
            f"judge：查表 {len(table.get('labels') or [])} 个 schema／已填 {table.get('filled') or 0} 格"
            f"{tail}（{self.table_path}）"
        )

    def config_report(self) -> dict:
        table = self.table or {}
        labels = table.get("labels") or []
        filled = table.get("filled") or 0
        counts: dict[str, int] = {}
        seen_pairs: set[tuple[str, str]] = set()
        for (label_a, label_b), symbol in (table.get("cells") or {}).items():
            # 对称的镜像格只数一遍。不能按 a>=b 跳过：读表时两把键都会注册，注入的表未必，
            # 只写了 (外1, 里1) 那一格时按 a>=b 会被跳过、进而被算进下面的「认不出」。
            if label_a == label_b:
                continue
            pair = (label_a, label_b) if label_a <= label_b else (label_b, label_a)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            counts[symbol] = counts.get(symbol, 0) + 1
        parts = [f"{s} {counts[s]}" for s in st.JUDGE_CELLS if s in counts]
        # filled 含「写了但符号认不出」的对（那几对不在 cells 里），差额要露出来，
        # 否则「已填 5 格（√ 1、× 2、○ 1）」这种括号加起来不等于前面那个数。
        # 只在差值 > 0 时说：注入的裸表没有 filled 键（按 0 算），差值是负的，
        # 写成「（○ 1、认不出 -1）」比不写还费解。
        unreadable = filled - sum(counts.values())
        if unreadable > 0:
            parts.append(f"认不出 {unreadable}")
        counted = "、".join(parts)
        problems = table.get("problems") or []
        warnings = []
        if problems:
            body = "；".join(problems[:5]) + (f"；还有 {len(problems) - 5} 处" if len(problems) > 5 else "")
            # 只有一条就不再套「1 处：」（问题正文自己带计数，套上去会读成「1 处：149 格」）
            warnings.append(f"{len(problems)} 处：{body}" if len(problems) > 1 else body)
        return {
            "table": "judge_table",
            "path": self.table_path,
            "summary": f"{len(labels)} 个标签／已填 {filled} 格" + (f"（{counted}）" if counted else ""),
            "empty": not filled,
            "missing": _table_missing(self.table_path),
            "warnings": warnings,
            # 没有标签时（空文件/文件不存在）根本不存在对角线，别宣称自动判定过
            "degrade": "除对角线自动判定（同一个 schema 判 √、未判出的对角格交 LLM）外，"
                       "其余每对都走 LLM 兜底（judgement.source=llm）"
            if labels
            else "表里没有任何 schema 名，每一对都走 LLM 兜底（judgement.source=llm）",
        }


# --------------------------------------------------------------------------
# 五、注册表与工厂
# --------------------------------------------------------------------------

AGENTS: dict[str, type[BaseAgent]] = {
    # 顺序即 --list 的打印顺序，也即流水线的先后（reality 是第一步的闸门）
    cls.NAME: cls
    for cls in (RealityAgent, AlterAgent, SemanticAgent, FeatureAgent, SchemaAgent, JudgeAgent, SynonymAgent)
}


def build_agent(name: str, **config) -> BaseAgent:
    """按名字建一个 agent，额外配置用关键字参数透传（如 feature_classes=...）"""
    try:
        cls = AGENTS[name]
    except KeyError:
        raise ValueError(f"没有名为 {name!r} 的 agent，可用：{'、'.join(AGENTS)}") from None
    return cls(**config)


class AgentSet(dict):
    """一套按名字取用的 agent。图只声明它需要的那几个（见 spatial_graphs.GraphSpec.needs）"""

    def need(self, name: str) -> BaseAgent:
        if name not in self:
            raise KeyError(f"这套 agent 里没有 {name!r}：请在所属 GraphSpec.needs 里声明它")
        return self[name]

    @contextmanager
    def budget_scope(self):
        """整张图跑一条候选状态期间，所有 agent 都处在各自的候选预算作用域里

        跑批入口按状态套这一层（见 spatial_runner）：一个 graph.invoke 里可能依次走
        reality → semantic → feature → …，每个 agent 的候选预算各算各的；退出时全部清零。
        """
        with ExitStack() as stack:
            for agent in self.values():
                stack.enter_context(agent.budget_scope())
            yield


def build_agents(
    needs: Iterable[str],
    *,
    model_key: str = DEFAULT_MODEL,
    llm: bool = True,
    overrides: dict[str, dict] | None = None,
    corpus_type: str | None = None,
    max_requests: int | None = None,
    batch_max_requests: int | None = None,
) -> AgentSet:
    """按需建 agent —— 只建 needs 里声明的那些，用不到的不建（deep agent 尤其贵）

    Args:
        needs (Iterable[str]): 需要的 agent 名
        model_key (str): 模型配置名
        llm (bool): 是否允许调模型。纯规则的 AlterAgent 不受影响，照建不误
        overrides (dict, optional): 逐个 agent 的额外构造参数，形如
            {"feature": {"feature_classes": [...]}}，供实验注入非默认的配置表。
            不要写 model_key / llm / corpus_type（与这里的同名参数撞车，会 TypeError）
        corpus_type (str, optional): 语料类型（中文名或别名，见 st.normalize_corpus_type）。
            构造期注入给每个 agent：agent 据此选四张配置表的目录，也据此在 user message 里
            交代本次跑批的类型。None = 调用方没交代类型：表仍按默认类型（方位词）解析，
            但提示语里不加那一行（保持与旧版逐字节一致，离线接线测试靠这条）
        max_requests (int, optional): 每个 agent **每条候选**的请求预算（A3）。None = 不限；
            给了就由 BaseAgent._invoke 在 budget_scope 内检查，超预算抛 RequestBudgetExceeded。
            一条候选的失败重试不该被候选数放大，这条预算就是那个上限。
        batch_max_requests (int, optional): 整批共用的请求预算（A3）。None = 不限；
            所有 agent 共用一份 BatchRequestBudget，不受候选作用域影响，超了整批余下的调用
            都不再发（记录照常落盘、转入人工复核）。

    Returns:
        AgentSet: {名字: agent 实例}
    """
    overrides = overrides or {}
    batch_budget = BatchRequestBudget(batch_max_requests)
    request_policy = getattr(agent_factory, "request_policy", None)
    retry_policy = (
        request_policy(model_key)
        if llm and request_policy is not None
        else {"max_retries": 0}
    )
    return AgentSet(
        (
            name,
            build_agent(
                name,
                model_key=model_key,
                llm=llm,
                corpus_type=corpus_type,
                max_requests=max_requests,
                batch_budget=batch_budget,
                retry_policy=retry_policy,
                **overrides.get(name, {}),
            ),
        )
        for name in needs
    )


# --------------------------------------------------------------------------
# 六、命令行入口
# --------------------------------------------------------------------------


def _describe_prompt_files(cls: type[BaseAgent]) -> str:
    """--list 用：把类上声明的提示语文件列成实际路径（两种来源见 PROMPT_SOURCE）

    顺带标出缺失的：--list 只读类属性、不构造 agent，不这样查一遍就跟「构造期读文件」脱节了。
    只查文件是否存在，不渲染——占位符写错由构造期兜住（任何真正建 agent 的入口都会报）。
    """
    if not cls.PROMPT_FILES:
        return "无"
    if cls.PROMPT_SOURCE == PROMPT_SOURCE_CONFIG:
        # 按语料类型分两份，两份都列：跑哪一类由 --corpus-type 定，缺哪一份就在那一份后面标出来。
        # 这里只读类属性、不构造 agent，所以路径得自己拼（与 _prompt_path 同一处真源：
        # 目录名取自 st.CORPUS_TYPE_DIRS，文件名取自 st.CONFIG_PROMPT_FILES）
        return "、".join(
            f"config_files/{st.CORPUS_TYPE_DIRS[kind]}/{st.CONFIG_PROMPT_FILES[name]}"
            + ("" if st.config_prompt_path(name, kind).exists() else "（缺失！）")
            for name in cls.PROMPT_FILES
            for kind in st.CORPUS_TYPE_DIRS
        )
    return "、".join(
        f"system_prompt/{name}.txt" + ("" if utils.prompt_path(name).exists() else "（缺失！）")
        for name in cls.PROMPT_FILES
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="七个独立的空间语义 agent：列出契约，或跑一次纯规则冒烟测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument("--list", action="store_true", help="打印七个 agent 的输入/输出契约，不调模型")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list:
        print("七个 agent（* = 会调模型；输入是 run() 的形参名）：")
        for name, cls in AGENTS.items():
            mark = "*" if cls.NEEDS_LLM else " "
            print(f"  [{mark}] {name:<9} 输入({', '.join(cls.INPUTS)})")
            print(f"{'':<16}输出({', '.join(cls.OUTPUTS)})  配置：{cls.CONFIG_NOTE}")
            print(f"{'':<16}提示语：{_describe_prompt_files(cls)}")
        print("\n组图见 spatial_graphs.py，跑批见 spatial_runner.py --list-graphs")
        return

    # 冒烟测试：全部不调模型、不需要 GPU，验证纯规则改写、关注词渲染与两张查表
    print("纯规则冒烟测试（不调模型）")
    demo_slice = ["在", "硬件", "上", "说", ",", "它", "将", "涉及", "到", "参数", "设置", "。"]
    # 图里传进来的关注词就是这个形状（spatial_tools.pick_focus_word 的返回值，不含 children）
    demo_focus = {"index": 2, "word": "上", "kind": "方位词", "source": "vocab"}
    print(f"  无类型时的开头块：{head_block(demo_focus)}")
    print(f"  带语料类型：{head_block(demo_focus, corpus_type='方位词')}")
    # 改写句那一侧的关注词：把原词换成替换词，index 不变、kind 按替换词重算（来源记 alter）
    demo_focus_b = st.replacement_focus_word(demo_focus, "中", ["root", "locality_phrase"])
    print(f"  改写句那侧的开头块：{head_block(demo_focus_b)}")
    print(f"  judge 的开头块（两份关注词 + 类型）：{pair_head_block(demo_focus, demo_focus_b, corpus_type='方位词')}")
    print(f"  关注词行（没有关注词）：{focus_line(None)}")
    # reality（第一道闸门）：这里只跑不调模型的那条路，真判「是不是现实空间」要调模型。
    # --no-llm 下它一条都不拦（source=no_llm，图据此放行），那个模式的用途正是绕开模型验纯规则
    reality = RealityAgent(llm=False)
    print(f"  reality（--no-llm，如实记 source）：{reality.run(st.join_slice(demo_slice), demo_focus)}")
    print(f"  reality（趋向动词那一组提示语）：{RealityAgent(llm=False, corpus_type='direction').describe()}")
    # 用注入的规则表演示多待选句：默认那张表读的是 config_files/localization/alter_rules.json5，
    # 本地已填内容，跑它验不出空表行为——要验空表看下面 injects 空 dict 那次
    alter = AlterAgent(
        rules={"groups": [{"words": ["上", "中", "里"], "replacements": ["上", "中", "里"]}]}
    )
    candidates, _ = alter.expand(demo_slice, demo_focus)  # expand 返回 (待选句列表, detail)
    for candidate in candidates:
        print(f"  alter：{candidate['sentence']}（{candidate['replacement']}，来自 {candidate['source']}）")
    result = alter.run(demo_slice, {**demo_focus, "replacement": "里"})
    print(f"  alter（显式指定替换词）：{result['sentences']}（applied={result['applied']}）")
    alter = AlterAgent(rules={})
    print(f"  alter（空表）：{alter.run(demo_slice, demo_focus)['detail']['reason']}")
    alter = AlterAgent()
    result = alter.run(demo_slice, demo_focus)
    print(f"  alter（默认规则表）：{len(result['sentences'])} 个待选句——{alter.describe()}")
    print(f"  alter（趋向动词那张表）：{AlterAgent(corpus_type='direction').describe()}")

    # schema：先按关注词收窄、再按特征查表（注入一张小表，免得依赖实际配置里填了什么）
    schema_table = {
        "schemas": {
            "上": {"1": {"description": "方位短语表处所", "match": [{"area": ["[+边界内]"]}]}},
            "外": {"1": {"description": "外部贴近", "match": [{"area": ["[-容器内]"], "distance": ["[+紧连]"]}]}},
        }
    }
    schema = SchemaAgent(llm=False, table=schema_table, feature_keys={"area", "distance"})
    print(f"  schema（查表命中）：{schema.run('在硬件上说话。', {'area': ['[+边界内]']}, demo_focus)}")
    print(f"  schema（收窄后未命中，--no-llm 不去猜）：{schema.run('在硬件外说话。', {'area': ['[-容器内]']}, demo_focus)}")
    print(f"  schema（表里没有这个词）：{schema.run('他走进屋里。', {}, {**demo_focus, 'word': '进'})}")
    print(f"  schema（不注入表、读实际配置）：{SchemaAgent(llm=False).describe()}")
    # judge：判定表是张 CSV 矩阵，四类符号（√/×/○/留空）各走各的路，外加「表里没这一对」的未命中。
    # 注入一张小表，不依赖实际配置填了什么；里2 那一列故意不写，(里1, 里2) 落到的就是「留空 = 非法」。
    judge_cells = {}
    for label_a, label_b, symbol in (
        ("里1", "里1", "√"),
        ("里1", "外1", "×"),
        ("里1", "未判出", "○"),
        ("外1", "外1", "√"),
        ("外1", "未判出", "○"),
        ("未判出", "未判出", "○"),
    ):
        judge_cells[(label_a, label_b)] = judge_cells[(label_b, label_a)] = symbol
    judge_table = {
        "cells": judge_cells,
        "labels": ["里1", "里2", "外1", "未判出"],
        "filled": 3,
        "problems": ["1 格留空或符号认不出（非法）：里1×里2；这些格查表时如实降级交 LLM"],
    }
    judge = JudgeAgent(llm=False, table=judge_table)
    for name_a, name_b in (("里1", "里1"), ("里1", "外1"), ("里1", "里2"), ("里1", "未判出"), ("里1", "旁1")):
        out = judge.run("甲。", "乙。", {"name": name_a}, {"name": name_b}, {}, {}, demo_focus)
        print(
            f"  judge（{name_a} × {name_b}）：source={out['source']} lookup={out['lookup']} "
            f"same={out['same_meaning']}——{out['reason']}"
        )
    print(f"  judge（不注入表、读实际配置）：{JudgeAgent(llm=False).describe()}")
    print("\n各 agent 状态：" + "；".join(a.describe() for a in build_agents(AGENTS).values()))


if __name__ == "__main__":
    main()
