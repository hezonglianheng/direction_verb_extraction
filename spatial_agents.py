# encoding: utf8

"""空间语义改写-判定用到的六个 agent，每个都独立构造、独立调用。

    reality_agent   输入一个句子与它的关注词，判断这个词是不是用于表达现实世界中的空间场景
                    （is_spatial_scene）。全流程的第一道闸门：判为 False 的记录就此收尾，
                    引申用法（会上/理论上）、时间用法（三天内）、时体与结果用法
                    （笑起来/看出来）、比喻用法（走出困境）都不再往下跑。
    alter_agent     输入切片（list[str]）与关注词（dict，含 index/word/kind），按 index 定位
                    该词，用规则表里包含它的同义词组中其他词逐一替换，产出一批待选句
                    （一个词能换几个词就有几个）。纯规则、不调 LLM（实现在 spatial_tools.alter_sentence）。
    semantic_agent  输入一个句子，判断其语义是否通顺（is_fluent）。
    feature_agent   输入一个句子，输出语义特征。用 deep agent：每类特征一个 subagent，
                    主 agent 逐类派发后汇总成一个特征对象。
    schema_agent    输入一个句子、它的语义特征与关注词：先按关注词排除别的词条的图式（一个词条
                    可用 words 覆盖一类词形，外面/外边 都归「外」）、再按特征查表；都未命中时
                    才调 LLM，且只让它在该类的候选图式里挑一个最接近的（这一类下一条图式都没有
                    时才退化为自由命名，名字也按组名起）。
    judge_agent     输入两个句子及其 schema，先查判定表（CSV 矩阵）；表里直接给了结论（√/×）
                    就用它，标了 ○、格子留空非法、或这一对没收录进表（未命中）才交 LLM 判断。

六个 agent 的输入里都有 focus_word：这一句是**为哪个方位词/趋向动词**抽出来的（index、词本身、
属哪一类），由 spatial_tools.pick_focus_word 从语料记录里取，图从第一个 agent 开始逐级往下传。
它是 user message 的第一行（措辞见 focus_line），五个 LLM agent 都按它展开判断——一句里有多个
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
    .venv/bin/python spatial_agents.py --list        # 打印六个 agent 的输入输出契约，不调模型
    .venv/bin/python spatial_agents.py               # 纯规则冒烟测试（不调模型，不需要 GPU）
"""

import argparse
import json
import threading
from pathlib import Path
from typing import Any, Iterable

from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field, create_model

import agents as agent_factory
import spatial_tools as st
import utils

DEFAULT_MODEL = "deepseek-flash"
"""默认模型配置名（model_config/api_keys.json 里的键）"""


class StructuredOutputMissing(RuntimeError):
    """模型这一轮没给出结构化输出，拿不到可用的判定

    这不是「调用失败」，但在本框架里必须当成失败抛出来：langchain 的 ToolStrategy 只在模型
    这一轮**带了 tool call** 时才解析结构化输出，模型只回一段自然语言时它既不报错也不重试，
    而是让子图正常结束、把 structured_response 显式写成 None（见 factory 里 _build_commands
    的 has_structured_output 分支）。也就是说「拿不到判定」是一条**正常返回**的路径。

    而各 agent 的 run() 只兜异常（try 包住 _invoke，返回值却在 try 之外解引用），真让
    _invoke 返回 None 的话 except 永远等不到：AttributeError 会穿透图节点、经 pool.map
    冒回跑批入口，一条记录出事带走整批（见 BaseAgent._invoke）。所以在 _invoke 这一处把它
    转成异常，五个 agent 既有的兜底路径就都生效了——reality 记 error 后放行、semantic 记
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
    """semantic_agent 的输出"""

    is_fluent: bool = Field(description="该文本是否存在空间语义方面的异常")
    reason: str = Field(description="判断依据")


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
"""提示语在 system_prompt/ 下（默认，六个 agent 里五个都在那儿，见 BaseAgent.PROMPT_SOURCE）"""
PROMPT_SOURCE_CONFIG = "config"
"""提示语与配置表同目录：config_files/<语料类型>/ 下（唯一一份 reality_agent 的提示语）

它按语料类型分两份、判据随词类不同（方位词挡的是引申与时间用法，趋向动词挡的是时体、结果与
比喻用法），所以与那四张表同一个目录、同一套分组口径，见 spatial_tools.CONFIG_PROMPT_FILES。
"""

REALITY_PROMPT_FILE = "reality"
"""reality_agent 的提示语（放 config_files/<语料类型>/ 下，见 PROMPT_SOURCE_CONFIG）"""
FEATURE_PROMPT_FILE = "feature"
"""feature_agent 表已填时的主提示语（含 {{n}} 与 {{roster}}）"""
FEATURE_UNFILLED_PROMPT_FILE = "feature_unfilled"
"""feature_agent 类别表为空时的兜底提示语"""
FEATURE_CLASS_PROMPT_FILE = "feature_class"
"""每类 subagent 的提示语模板（含 {{name}}/{{description}}/{{values}}/{{examples}}）"""


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
    # parse_sentence(原句) 与语料切片逐 token 相同（实测 2000/2000）。
    FOCUS_ROLE_RECORD: "在分词结果里是第 {index} 个词",
    FOCUS_ROLE_ORIGINAL: "在原句的分词结果里是第 {index} 个词",
    FOCUS_ROLE_REPLACEMENT: "占的是原句分词结果里第 {index} 个词的位置",
}

_FOCUS_TAIL = {
    # 角色 -> 第二行（以哪个词为准、要判什么）。提示语里那些「紧接着的下一行」就指着它
    FOCUS_ROLE_RECORD: (
        "它是这条语料按筛选规则提取出来的词；本句若经过同义词替换，"
        "该位置上的词已换成同义词组里的另一个，一律以句子中实际的用词为准。"
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

FOCUS_INDEX_NOTE = "（下标从 0 起，与 parse_sentence 分词行里 token 前的序号一致）"
"""下标那句注（record / original 共用：这两处的下标就是本句分词的序号）"""

FOCUS_INDEX_NOTE_ALTER = "（下标从 0 起，是原句分词结果里的序号，不要在改写句的分词行上数它）"
"""改写句那一侧的下标注：同一个数，但说的是**原句**的分词

与 FOCUS_INDEX_NOTE 只差半句，差别却是要紧的：b 侧的下标是从 a 侧搬过去的（见
st.replacement_focus_word），模型手上只有改写句，照 FOCUS_INDEX_NOTE 去 parse_sentence 改写句、
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

    角色只决定措辞（见 focus_line）。由数据自己推、而不是让五个 agent 各接一个形参：同一份
    关注词在五个 agent 的 user message 里必须说成同一回事，而 agent 手上就只有这个 dict。
    手写的关注词（没有 source）一律按 record 算。
    """
    if (focus_word or {}).get("source") == st.FOCUS_SOURCE_ALTER:
        return FOCUS_ROLE_REPLACEMENT
    return FOCUS_ROLE_RECORD


def focus_line(focus_word: dict | None, *, role: str = FOCUS_ROLE_RECORD) -> str:
    """把关注词渲染成 user message 的首行（五个 LLM agent 共用这一处措辞）

    focus_word 是关注词（{"index", "word", "kind", "source"}，见 spatial_tools.pick_focus_word
    与 replacement_focus_word）；没有关注词（None 或取不到词）时给一句中性说明，让 agent 按整句判断。
    role 决定后面两句怎么说（默认按 focus_role 由 source 推）：

    - record：这条语料提取出来的词——single 图，以及 pair / alter_check 图的 a 侧（原句）；
    - replacement：改写句里换上去的那个词——pair / alter_check 图的 b 侧，下标是它**顶替掉的那个词
      在原句**里的位置（不是它在改写句里的位置：改写句要重新分词，下标不保证搬得过去，见 _FOCUS_TAIL）；
    - original：一对句子里原句那一侧的词——judge 专用（它同时拿着原句与改写句，另外那句由
      pair_head_block 的下一行交代），下标是它在**原句**里的位置。

    措辞为什么这么绕，两条都要照顾到：

    - 五个 agent 的 system_prompt 在几张图之间共用，所以措辞里不能出现只对某一张图成立的说法，
      只能用条件句交代，并给出 index 这个可核对锚点（parse_sentence 的分词行里每个 token 前都
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


def head_block(focus_word: dict | None, *, corpus_type: str | None = None) -> str:
    """user message 的开头块：关注词那段（1~2 行）+ 语料类型行（有类型时）

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
        for part in (focus_line(focus_word, role=focus_role(focus_word)), corpus_type_line(corpus_type))
        if part
    )


def pair_head_block(
    focus_word: dict | None, focus_word_b: dict | None, *, corpus_type: str | None = None
) -> str:
    """judge 的 user message 开头块：原句那份关注词 + 改写句那份 + 语料类型行（有类型时）

    比其余三个 agent 多一段（各 2 行 + 类型行）：judge 同时拿着原句与改写句，而两侧的关注词
    是两个词——原句里被替换掉的那个，与改写句里换上去的那个（见 st.replacement_focus_word）。
    第 1 行仍是「原句里被替换掉的那个词」：四份提示语的条件句都按「第一行点名了本次要关注的词」
    写，judge.txt 也照这个说。

    两侧各自没有关注词时用 FOCUS_A_ABSENT / FOCUS_B_ABSENT 占位（原句取不到词、没走到改写、
    状态是手工拼的），**行数不变**：focus_line 对空关注词只回一行中性说明，让它顶上去整体就少一行，
    提示语里「第一行 / 紧接着的下一行」这些锚点全错位。两份占位都写成两行，四种组合都是 5 行。
    """
    return "\n".join(
        part
        for part in (
            focus_line(focus_word, role=FOCUS_ROLE_ORIGINAL)
            if isinstance(focus_word, dict) and focus_word.get("word")
            else FOCUS_A_ABSENT,
            focus_line(focus_word_b, role=FOCUS_ROLE_REPLACEMENT)
            if isinstance(focus_word_b, dict) and focus_word_b.get("word")
            else FOCUS_B_ABSENT,
            corpus_type_line(corpus_type),
        )
        if part
    )


def _class_prompt(template: str, feature_class: dict) -> str:
    """按模板给某一类语义特征的 subagent 写系统提示语（模板见 system_prompt/feature_class.txt）"""
    values = "、".join(str(v) for v in (feature_class.get("values") or [])) or "（未限定，按判据自行概括）"
    example_lines = "\n".join(
        f"    {json.dumps(e, ensure_ascii=False)}"
        for e in (feature_class.get("examples") or [])[:5]
        if isinstance(e, dict)
    )
    return utils.render_prompt(
        template,
        {
            "name": feature_class.get("name") or feature_class["key"],
            "description": feature_class.get("description") or "（未填写）",
            "values": values,
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


def _render_feature_prompts(
    feature_classes: list[dict], head_template: str, class_template: str
) -> tuple[str, list[str]]:
    """渲染 feature_agent 的提示语，返回 (主提示语, [各类 subagent 的提示语])

    两个模板由调用方读好传进来（FeatureAgent 在构造期用 system_prompt/ 下的文件渲染）。
    """
    roster = "\n".join(
        f"    {spec['name']}：负责「{spec['label']}」，对应特征字段 {spec['key']}"
        for spec in _subagent_specs(feature_classes)
    )
    head = utils.render_prompt(
        head_template, {"n": len(feature_classes), "roster": roster}, source=f"{FEATURE_PROMPT_FILE}.txt"
    )
    return head, [_class_prompt(class_template, feature_class) for feature_class in feature_classes]


def _build_deep_feature_agent(
    model_key: str,
    feature_classes: list[dict],
    tools: list,
    head_prompt: str,
    class_prompts: list[str],
):
    """建特征抽取用的 deep agent：每类特征一个 subagent，主 agent 汇总成结构化特征对象

    head_prompt 与 class_prompts 由 FeatureAgent 构造期按 system_prompt/ 下的模板渲染好传进来
    （本函数是模块级函数，拿不到 agent 实例），class_prompts 与 feature_classes 一一对应。

    key 是汉语也是合法的 Python 标识符，可以直接当 pydantic 字段名（合法性由调用方先校验）；
    含空格、连字符之类的键不行。
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
    feature_model = create_model("SemanticFeatures", **fields)

    subagents = [
        {
            "name": spec["name"],
            "description": spec["description"],
            "system_prompt": prompt,
            "tools": tools,
        }
        for spec, prompt in zip(_subagent_specs(feature_classes), class_prompts, strict=True)
    ]

    return agent_factory.get_deep_agent(
        model_key,
        tools=tools,
        system_prompt=head_prompt,
        subagents=subagents,
        response_format=ToolStrategy(feature_model),
    )


# --------------------------------------------------------------------------
# 三、agent 基类
# --------------------------------------------------------------------------


class BaseAgent:
    """六个 agent 的公共骨架：输入契约、llm 开关、惰性建模型。

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

    def __init__(
        self,
        *,
        model_key: str = DEFAULT_MODEL,
        llm: bool = True,
        tools: list | None = None,
        corpus_type: str | None = None,
    ):
        self.model_key = model_key
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
        """本 agent 的工具列表，默认给一个 LTP 句法分析工具"""
        return list(self._tools) if self._tools is not None else [st.parse_sentence]

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

    def _invoke(self, content: str):
        """跑一次底层 agent 并取结构化输出。异常不吞，由 run() 决定怎么如实记 source

        取不到结构化输出时抛 StructuredOutputMissing 而不是返回 None——这一处是五个 agent
        共用的关口，理由见该异常类。顺带把模型的原始回复带进异常：它平时落不了盘，出岔子时
        是事后唯一能查的东西（见 _reply_digest）。

        用 .get 而不是 []：键缺失同样该走这条兜底，而不是让 run() 的兜底把 KeyError 记成
        一句「调用失败」——那样就看不出到底是模型没给判定、还是返回形状变了。
        """
        result = self._model().invoke({"messages": [{"role": "user", "content": content}]})
        verdict = result.get("structured_response")
        if verdict is None:
            raise StructuredOutputMissing(_reply_digest(result.get("messages") or []))
        return verdict

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
# 四、六个 agent
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

    不给工具：判「这个词说的是不是物理空间」靠读句子与常识，句法结构帮不上什么忙（关注词的下标
    已在消息首行给出，不劳模型自己去数），所以 tools() 默认给空表（显式注入 tools 仍生效）。
    提示语按语料类型分两份、与四张表同目录（见 PROMPT_SOURCE_CONFIG）。
    """

    NAME = "reality"
    INPUTS = ("sentence", "focus_word")
    OUTPUTS = ("is_spatial_scene", "reason", "source")
    CONFIG_NOTE = "config_files/<语料类型>/reality.txt（提示语，与该组四张表同目录）"
    PROMPT_FILES = (REALITY_PROMPT_FILE,)
    PROMPT_SOURCE = PROMPT_SOURCE_CONFIG

    def tools(self) -> list:
        """本 agent 默认不调工具（见类文档）；显式注入 tools 时以注入的为准"""
        return list(self._tools) if self._tools is not None else []

    def _build(self):
        return agent_factory.get_agent(
            self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(RealityVerdict)
        )

    def run(self, sentence: str, focus_word: dict) -> dict:
        """判这一句里的关注词是不是在说现实空间；判不了就给 None 并记下是哪种判不了

        四种 source 分开记而不是都回 None：汇总里要能看出「这一批有多少条压根没判」
        （见 spatial_graphs._tally_reality），混成一种就分不清「没问模型」与「模型说判不了」了。
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
        try:
            verdict: RealityVerdict = self._invoke(
                f"{head_block(focus_word, corpus_type=self.corpus_type)}\n"
                f"请判断这个句子里的关注词是否用于表达现实世界中的空间场景：{sentence}"
            )
        except Exception as e:  # 单条记录失败不该带走整批，如实记下原因（图会放行，见 reality_gate）
            return {"is_spatial_scene": None, "source": "error", "reason": f"调用失败：{e!r}"}
        return {"is_spatial_scene": verdict.is_spatial_scene, "reason": verdict.reason, "source": "llm"}

    def describe(self) -> str:
        tail = f"模型 {self.model_key}" if self.llm else "未调用 LLM"
        return f"reality：{tail}（提示语 {self.prompt_paths[REALITY_PROMPT_FILE]}）"


class AlterAgent(BaseAgent):
    """按 index 定位切片中的关注词，用同义词组里的其他词逐一替换，得到一批待选句。

    纯规则，不调 LLM。rules 可由构造参数注入，于是同一个类能按不同规则表建多份供对照实验
    （spatial_runner.py --alter-rules 走的就是这条路）。

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

    @property
    def rules_path(self) -> str:
        """兼容旧名（spatial_runner 的告警、文档里都这么叫）：与 table_path 同一个值"""
        return self.table_path

    def run(self, sentence_slice: list[str], focus_word: dict) -> dict:
        """改写出全部待选句：slices 是"词的列表的列表"，sentences 是对应的句子字符串"""
        slices, detail = st.alter_sentence(sentence_slice, focus_word or {}, self.rules)
        return {
            "slices": slices,
            "sentences": [st.join_slice(s) for s in slices],
            "applied": bool(detail["applied"]),
            "detail": detail,
        }

    def expand(self, sentence_slice: list[str], focus_word: dict) -> tuple[list[dict], dict]:
        """把改写结果摊成一条待选句一项，连同改写详情一起返回 (待选句列表, detail)

        每项形如：
            {"candidate_index": 0, "candidate_total": 2, "word": "上", "span": [1, 2],
             "slice": [...], "sentence": "在硬件中说…", "replacement": "中",
             "source": "groups[0]", "group": ["上", "中", "里"]}
        跑批前的展开（spatial_graphs.alter_init_states）、节点里取自己那一条、预填充清点句子
        三处都走本方法，保证"这条语料展开成几个待选句"与"记录里选中的是哪个候选"不会对不上。

        没有待选句时列表为空，原因在 detail["reason"]（applied=False）。
        """
        result = self.run(sentence_slice, focus_word)
        detail = result["detail"]
        candidates = [
            {
                "candidate_index": index,
                "candidate_total": len(result["slices"]),
                "word": detail["word"],
                "span": detail["span"],
                "slice": slice_,
                "sentence": sentence,
                **candidate,
            }
            for index, (candidate, slice_, sentence) in enumerate(
                zip(detail["candidates"], result["slices"], result["sentences"])
            )
        ]
        return candidates, detail

    def plan_all(self, sentence_slice: list[str], focus_word: dict) -> list[str]:
        """只算出全部待选句、无任何副作用

        供图在跑批之前统计"整批会出现哪些句子"，以便一次性预填充 LTP（见 spatial_graphs.*_plan）。
        与 run()/expand() 同源，保证预填充的句子和图里真正会用到的一致：漏掉任何一个待选句，
        它就会在线程池里第一次遇到时才做句法分析，而推理在锁外，多线程下就是并发调同一份 LTP。
        """
        candidates, _ = self.expand(sentence_slice, focus_word)
        return [candidate["sentence"] for candidate in candidates]

    def describe(self) -> str:
        groups = self.rules.get("groups") or []
        return f"alter：替换规则 {len(groups)} 组（{self.rules_path}）"

    def config_report(self) -> dict:
        groups = self.rules.get("groups") or []
        return {
            "table": "alter_rules",
            "path": self.rules_path,
            "summary": f"{len(groups)} 组同义词",
            "empty": not groups,
            "missing": _table_missing(self.table_path),
            "warnings": [],
            "degrade": "所有记录都会在改写环节停下（alter_missed），包括本闸门在内的五个 LLM agent 一次都不调",
        }


class SemanticAgent(BaseAgent):
    """判断单句的语义是否通顺"""

    NAME = "semantic"
    INPUTS = ("sentence", "focus_word")
    OUTPUTS = ("is_fluent", "reason", "source")
    CONFIG_NOTE = "无"
    PROMPT_FILES = ("semantic",)

    def _build(self):
        return agent_factory.get_agent(
            self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(SemanticVerdict)
        )

    def run(self, sentence: str, focus_word: dict) -> dict:
        if not self.llm:
            return {"is_fluent": None, "reason": "未调用 LLM（--no-llm）", "source": "no_llm"}
        try:
            verdict: SemanticVerdict = self._invoke(
                f"{head_block(focus_word, corpus_type=self.corpus_type)}\n"
                f"该文本是否存在空间语义方面的异常：{sentence}"
            )
        except Exception as e:  # 单条记录失败不该带走整批，如实记下原因
            return {"is_fluent": None, "reason": f"调用失败：{e!r}", "source": "error"}
        return {"is_fluent": verdict.is_fluent, "reason": verdict.reason, "source": "llm"}


class FeatureAgent(BaseAgent):
    """抽取单句的语义特征

    类别表填了就用 deep agent（每类一个 subagent 各自抽取、主 agent 汇总）；
    表为空则退化成单个 agent 自由抽取，并把 source 标成 unfilled_config，
    提醒这不是按既定类别抽的。

    deep agent 建起来贵，所以 __init__ 只读表与校验、不建模型；--no-llm 下一次都不建。
    """

    NAME = "feature"
    INPUTS = ("sentence", "focus_word")
    OUTPUTS = ("features", "source", "reason")
    CONFIG_NOTE = "config_files/<语料类型>/feature_classes.json5（类型见 --corpus-type，默认 localization）"
    PROMPT_FILES = (FEATURE_PROMPT_FILE, FEATURE_UNFILLED_PROMPT_FILE, FEATURE_CLASS_PROMPT_FILE)
    """三份都在构造期读入，表为空时用不上的那两份也顺带校验（配置错早暴露）"""

    def __init__(self, *, feature_classes: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)
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
            self.system_prompt, self.class_prompts = _render_feature_prompts(
                self.feature_classes,
                self.prompts[FEATURE_PROMPT_FILE],
                self.prompts[FEATURE_CLASS_PROMPT_FILE],
            )
        else:
            self.system_prompt = self.prompts[FEATURE_UNFILLED_PROMPT_FILE]
            self.class_prompts = []

    def _build(self):
        if not self.feature_classes:
            return agent_factory.get_agent(
                self.model_key, self.tools(), self.system_prompt, response_format=ToolStrategy(FreeFeatures)
            )
        return _build_deep_feature_agent(
            self.model_key, self.feature_classes, self.tools(), self.system_prompt, self.class_prompts
        )

    def run(self, sentence: str, focus_word: dict) -> dict:
        if not self.llm:
            return {"features": {}, "source": "no_llm", "reason": "未调用 LLM（--no-llm）"}
        try:
            # 关注词写在首行：主 agent 只有把它一起写进 task 的任务文本，subagent 才看得到
            # （subagent 的 system_prompt 是构造期固定的，见 system_prompt/feature.txt）
            data = self._invoke(
                f"{head_block(focus_word, corpus_type=self.corpus_type)}\n请抽取这个句子的语义特征：{sentence}"
            ).model_dump()
        except Exception as e:
            return {"features": {}, "source": "error", "reason": f"调用失败：{e!r}"}

        # 类别表已填时 data 直接就是特征字段；未填（FreeFeatures）时特征装在 features 键下
        features = data.get("features") if set(data) == {"features"} else data
        return {"features": features or {}, "source": self.table_source, "reason": ""}

    def describe(self) -> str:
        n = len(self.feature_classes)
        tail = "" if n else "（未填写，自由抽取）"
        if not self.llm:
            return f"feature：未调用 LLM；特征类别 {n} 类{tail}（{self.table_path}）"
        return f"feature：模型 {self.model_key}；特征类别 {n} 类{tail}（{self.table_path}）"

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

        try:
            verdict: SchemaVerdict = self._invoke(
                self._content(sentence, features, focus_word, word, class_word, miss, candidates)
            )
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
        """LLM 兜底时的 user message：关注词行 + 句子 + 特征 + 候选名单（或自由命名要求）

        带类名的那句必须拼在 head_block 整块**之后**：head_block 刻意把语料类型行钉在第 3 行，
        插在中间会把类型行顶到第 4 行（verify_focus_wiring._check_focus_head 就在断言这个位置）。
        """
        head = (
            f"{head_block(focus_word, corpus_type=self.corpus_type)}\n"
            f"句子：{sentence}\n"
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


class JudgeAgent(BaseAgent):
    """判断两句意义是否相同：先查判定表，表里没给出结论的再由 LLM 判断

    这是唯一一个同时拿到两句的 agent（也是唯一一个要两份关注词的）：两侧的句子、schema、
    特征、关注词由 graph 合并后一起传进来，判的就是「把 a 侧那个词换成 b 侧那个词之后，
    两句意义是否还相同」。b 侧的 schema 是拿替换词重新收窄、重新定的（见 st.replacement_focus_word），
    所以判定表里 上12 × 里1 这样的格子才对得上——两侧若共用一个关注词，b 侧会退回收窄回原词
    那一组，判定就只在同一组内自比了。
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

        content = (
            f"{pair_head_block(focus_word, focus_word_b, corpus_type=self.corpus_type)}\n"
            f"原句：{sentence_a}\n"
            f"原句 schema：{schema_a.get('name')}（来源 {schema_a.get('source')}）\n"
            f"原句语义特征：{json.dumps(features_a or {}, ensure_ascii=False)}\n"
            f"改写句：{sentence_b}\n"
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
    for cls in (RealityAgent, AlterAgent, SemanticAgent, FeatureAgent, SchemaAgent, JudgeAgent)
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


def build_agents(
    needs: Iterable[str],
    *,
    model_key: str = DEFAULT_MODEL,
    llm: bool = True,
    overrides: dict[str, dict] | None = None,
    corpus_type: str | None = None,
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

    Returns:
        AgentSet: {名字: agent 实例}
    """
    overrides = overrides or {}
    return AgentSet(
        (
            name,
            build_agent(
                name,
                model_key=model_key,
                llm=llm,
                corpus_type=corpus_type,
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
        description="六个独立的空间语义 agent：列出契约，或跑一次纯规则冒烟测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument("--list", action="store_true", help="打印六个 agent 的输入/输出契约，不调模型")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list:
        print("六个 agent（* = 会调模型；输入是 run() 的形参名）：")
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
    alter = AlterAgent(rules={"groups": [["上", "中", "里"]]})
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
