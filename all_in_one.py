# encoding: utf8

"""把 pair 图六步链路压成一次 LLM 调用的对照实现。

spatial_runner.py --graph pair 跑一条记录要八次模型调用（reality 一次 → semantic 两次 →
feature 两次 → schema 两次 → judge 一次），而且**后一步只能看前一步的
结论**：改写词到底有没有换掉空间关系，是等 schema 定了才在 judge 里被比较的。本脚本是它的
单次调用对照版：一条记录只发**一次**请求，把五个 LLM 判断一起答出来，纯规则的改写与两张
查表仍在本地做，最后落一条**与 pair 图逐字段同形**的记录（同形靠直接调 sg.serialize_pair，
不是照着抄一遍字段名）。于是两件事同时成立：用一次调用换到原来八次调用的结论；两份 jsonl
可以直接逐字段对账，看单次调用掉了多少精度、省了多少次往返。这份同形也包括顶层 id 与 csv
首列——两者分别由那同一份 sg.serialize_pair（见 process_state）与 pair 图的列序真源
st.PAIR_CSV_COLUMNS 给出，同一段语料跑下来两侧的 id 必然逐字相同。

其中 reality 那一问在 pair 侧是**按原句共享**的：闸门判的只是原句，而一条语料会展开成多个
候选，所以那边在跑批前把要判的原句一次判完、候选之间共用一份判定（见 sr._prefill_reality），
不再是「每个候选各问一遍」。收尾打印的对照数正是按每状态累加、不计这边的复用（见 PAIR_CALLS_PER_STEP）。

一次调用到底指什么：一次 agent.invoke、一次消息往返。两道不需要模型的闸门在最前面短路，
连这一次也不发——

    bad_record     记录的 sentence 字段缺失或不是分词切片
    alter_missed   规则表里没有包含关注词的词组（含 --candidate-index 越界）

与流水线同一口径：没命中改写规则就不该问模型。默认 --limit 1、每条记录只跑一个待选句，
所以默认运行正好一次调用（收尾会打印实际次数）。

这一次也是**硬上限**，不是「通常一次」：结构化输出不合法时框架默认会补发一次请求去纠正
（ToolStrategy(handle_errors=True)），那是第二次 API 调用，与「只调一次」冲突，所以默认关掉
（想放宽用 --allow-retry），解析不出来就如实记 all_in_one.error。实测这一次 invoke 对应
**一次**真实 chat 请求（回调计数器量的，不是推断的：强制工具调用一返回就结束，不再有第二次往返）。
这个回调计数现在挂在每条记录上（all_in_one.requests，见 CallCounter / _RequestProbe）：
补发若真的发生会如实多算一次请求，补发次数按每次调用中首个请求之后的请求数累计；
未发出请求就失败的调用不计补发，也不会抵消其他调用的补发次数。

仍有一类失败是躲不掉的：deepseek 偶发地回 400（assistant 消息里的 tool_calls 没有被逐条回应，
像是模型偶尔一次吐出多个并发工具调用），strict 下同样会出现。它不带走整批，记进 error、
收尾连语料行号一起打出来，好按行重跑。

收尾还会打印一个对照数：同样这些状态，pair 图按每状态走到哪一步累加要发几次请求（不计它按句/
按消息的复用，故是上界；精确数看 pair 产物 summary 里的 agent_stats.roundtrips）。

为什么不给 parse_sentence 工具：工具调用就是又一次 API 请求，八次往返正是要消掉的东西。
句法证据由本地 st.parse_sentence_text 算好塞进消息（--no-syntax 可以省掉，但那样就与
六 agent 那条路不可比了——那边同样把句法证据内联进消息，只是没有这个开关）。

记录形状、status 词表、闸门顺序、查表优先级全部复用 spatial_graphs.py 与 spatial_agents.py
里的纯函数，本脚本不另立一套：模型把五步都答了，**采信与否**在装配记录时按流水线口径施加
（sg.reality_gate / sg.pair_semantic_gate），被挡下的答案留在 all_in_one.raw 里、
并在 all_in_one.suppressed 里逐项标出来。这是单次调用的固有代价：前面一步判了否定，
后面几步的答案就已经付过钱了。

复用到的几处私有名（sg._step、sr._config_tables、sr._print_config_tables、sr._corpus_type_arg、
sg.alter_node 返回的 history 口径）是刻意的：它们是那套口径的唯一真源，照抄一份才是分叉的开始。

用法：
    .venv/bin/python all_in_one.py --print-prompt                     # 看消息长什么样，0 次调用
    .venv/bin/python all_in_one.py --limit 1                          # 默认：一条语料，一次调用
    .venv/bin/python all_in_one.py --limit 5 --all-candidates         # 跑遍每条语料的全部待选句
    .venv/bin/python all_in_one.py --limit 5 --no-llm                 # 纯规则+查表，0 次调用
    .venv/bin/python all_in_one.py --corpus-type 趋向动词 --limit 3    # 换语料类型
    .venv/bin/python all_in_one.py --limit 5 --allow-retry            # 允许补发一次请求（默认不补）
    .venv/bin/python all_in_one.py --limit 200 --all-candidates --workers 8   # 全量：与 pair 图逐条对账

--workers 只改**同时有多少条记录在跑**，不改每条记录的语义：一条记录仍至多一次调用，
落盘顺序也仍是输入顺序（pool.map 按输入顺序产出、由主线程逐条写）。与 pair 图对账按
(line_no, candidate.index) 对齐——两侧的 ResultWriter.finish() 会做同一遍排序（见
spatial_runner._sort_records），所以跑完之后两个文件顺序完全一致，本侧的有序是天然就有的
（记录里的顶层 id 两侧同值，逐条对账时可以直接拿它当标识用）。
默认 1 = 逐条跑；
只有全量对照时才需要它：--all-candidates 会把 200 条语料展开成上千个状态，逐条发请求
按 4 秒/条算接近两小时。并发有个前提——预填充必须真兜住全部句子（推理在锁外做，见
sentenceTable.get_task_value），装不下缓存时启动会像 runner 那样告警，见 main 里那段自检。
"""

import argparse
import threading
import time
import warnings
from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import Field, create_model

import agents as agent_factory
import config
import spatial_agents as sa
import spatial_graphs as sg
import spatial_runner as sr
import spatial_tools as st
import utils

_PROJECT_ROOT = Path(__file__).resolve().parent

PROMPT_FILE = "all_in_one"
"""本脚本的 system prompt（system_prompt/all_in_one.txt）

它是 semantic / feature / feature_class / schema / judge / reality 六份提示语的单次调用改写版，
**改那六份时要同步核对这一份**。
"""

DEFAULT_OUT_PREFIX = "all_in_one"
"""输出目录 results/all_in_one/ 与文件名前缀；与 pair 图的 spatial_pairs_* 不会混"""
DEFAULT_LIMIT = 1
"""默认只处理 1 条：本脚本是单次调用的诊断入口，不是跑批工具（runner 那边默认 100）"""

PAIR_CALLS_PER_STEP = {"reality": 1, "semantic": 2, "feature": 2, "schema": 2, "judgement": 1}
"""pair 图走到某一步要发几次请求（semantic / feature / schema 三个节点都按原句、改写句两侧各跑一次，所以各是 2）

整条链路在无缓存、每步都问模型时是 1+2+2+2+1 = 8 次。**不计缓存的上界**：按「每个走到这一步的
状态各算一份」累加。pair 侧实际的按句/按消息复用一律不计——reality 按原句去重（见
sr._prefill_reality），semantic 的 a 侧、feature 与 schema 的 a 侧输入逐字相同也只问一次，
schema 查表命中（含兜底条目）一问都不发——所以真实往返数只会更低。它的用途是给「一次调用省了
多少」一个可核对的**朴素口径**，既不是精确值也不是下界；要精确数就去读 pair 产物 summary 里的
agent_stats.roundtrips。alter 不在表里：它是纯规则，0 次。
"""


# --------------------------------------------------------------------------
# 一、结构化输出模型
# --------------------------------------------------------------------------


def build_verdict_model(feature_classes: list[dict]) -> type:
    """本次一次作答要回的五个块，收成一个 pydantic 模型

    叶子 verdict 直接复用 spatial_agents 里的四个（RealityVerdict / SemanticVerdict /
    SchemaVerdict / JudgeVerdict）：它们的 Field(description=...) 已经把规则写进去了，而且
    verify_focus_wiring.check_schema_selection 在断言「提示语与字段描述不许打架」——
    自己复刻一份等于让那套断言作废。

    特征那一块分两档，与 FeatureAgent 同源：

    - 类别表已填 -> 按 key 建模型、每字段 list[str]（同 _build_deep_feature_agent）；
    - 类别表为空 -> dict[str, Any] 自由抽取（同 FreeFeatures）。未填那一路不另包一层
      features 键：字段名已经分了 a/b 两侧，类别表空不空又是构造期已知的。

    顶层字段名与记录的五个块刻意同名（reality / semantic / features / schema / judgement），
    装配时少一层翻译；叶子模型的描述是分侧的，所以外层 Field 要写明 a 是原句、b 是改写句。
    """
    # 顶层的 schema 字段会遮住 BaseModel.schema() 那个已废弃的类方法，pydantic 于是每次建类都
    # 告警一句。这里**只**关掉这一条（按模型名与字段名逐字匹配）：遮住的是没人再调的旧方法，
    # 而 model_dump / model_json_schema / ToolStrategy 都照常（已实测），字段名又是刻意同名的。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r'Field name "schema" in "AllInOneVerdict" shadows an attribute in parent',
        )
        return _build_verdict_model(feature_classes)


def _build_verdict_model(feature_classes: list[dict]) -> type:
    """build_verdict_model 的正身：建类那一步会告警，所以外面套了一层就地消音"""

    if feature_classes:
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
        feature_type: Any = create_model("SemanticFeatures", **fields)
    else:
        feature_type = dict[str, Any]

    side_features = create_model(
        "SideFeatures",
        a=(feature_type, Field(description="原句读出的语义特征")),
        b=(feature_type, Field(description="改写句读出的语义特征")),
    )
    side_fluency = create_model(
        "SideFluency",
        a=(sa.SemanticVerdict, Field(description="原句的语法是否通顺、空间语义是否异常")),
        b=(sa.SemanticVerdict, Field(description="改写句的语法是否通顺、空间语义是否异常")),
    )
    side_schema = create_model(
        "SideSchema",
        a=(sa.SchemaVerdict, Field(description="原句的 schema")),
        b=(sa.SchemaVerdict, Field(description="改写句的 schema")),
    )
    return create_model(
        "AllInOneVerdict",
        reality=(sa.RealityVerdict, Field(description="第 1 步：原句那个关注词是否表达现实世界中的空间场景")),
        semantic=(side_fluency, Field(description="第 2 步：两句各自的语法是否通顺、空间语义是否异常")),
        features=(side_features, Field(description="第 3 步：两侧的语义特征")),
        schema=(side_schema, Field(description="第 4 步：两侧各自的 schema")),
        judgement=(sa.JudgeVerdict, Field(description="第 5 步：两句意义是否相同")),
    )


# --------------------------------------------------------------------------
# 二、提示语
# --------------------------------------------------------------------------


def _feature_roster(feature_agent) -> tuple[str, str]:
    """渲染 {{n}} 与 {{roster}} 两个占位符的值：类别表的规模说明 + 逐类那一段

    类别段直接复用 sa.render_class_blocks（与 pair flat 的 {{classes}} 同一份渲染：取值按「、」
    拼接、逐字照抄规则与 flat 用法段随块带入），于是两处的类别段逐字一致，不会各自演化——这正是
    不在本脚本里另写一套的原因。模板从已建好的 feature agent 取（它构造期就把这几份读进来了）。
    """
    feature_classes = feature_agent.feature_classes
    if not feature_classes:
        return "尚未填写（还是空表），没有既定类别可依", ""
    n = f"已填写，共 {len(feature_classes)} 类；features.a 与 features.b 各有这几个字段，字段名用键名"
    blocks = sa.render_class_blocks(
        feature_classes,
        feature_agent.prompts[sa.FEATURE_CLASS_PROMPT_FILE],
        usage=feature_agent.prompts[sa.FEATURE_CLASS_USAGE_FLAT_FILE],
    )
    return n, "\n\n".join(blocks)


def build_system_prompt(feature_agent, *, reality_criteria: str) -> str:
    """渲染 system_prompt/all_in_one.txt

    渲染放**启动期**（--no-llm、--print-prompt 也渲染）：占位符写错、文件缺失属于配置错，
    该启动就报，与 BaseAgent.__init__ 同一口径。

    reality_criteria 由调用方从 reality agent 的 system_prompt 取、类别段从 feature agent 取
    （见 build_context）：那两份判据分别按语料类型分档，读法与路径的单一真源在 spatial_agents 那边，
    此处不再自己读一遍文件。

    注意渲染是单次扫描（utils.render_prompt）：reality_criteria 里若含 {{...}} 不会被再替换一遍，
    注入的顺序因此无关紧要。
    """
    template = utils.read_prompt(PROMPT_FILE)
    n, roster = _feature_roster(feature_agent)
    return utils.render_prompt(
        template,
        {"reality_criteria": reality_criteria, "n": n, "roster": roster},
        source=f"{PROMPT_FILE}.txt",
    )


def print_numbered(title: str, text: str) -> None:
    """带行号打印一段文本（--print-prompt 用：提示语里按行号指位置，不打行号没法核对）"""
    print(f"\n===== {title}（共 {len(text.splitlines())} 行）=====")
    for index, line in enumerate(text.splitlines(), start=1):
        print(f"{index:>4}| {line}")


# --------------------------------------------------------------------------
# 三、user message
# --------------------------------------------------------------------------


class _Prepared(NamedTuple):
    """一条记录这次调用要发的东西（消息 + 各项诊断），准备好但还没发"""

    message: str
    syntax: dict
    """两侧的句法证据是否真的放进了消息（{a, b}；--no-syntax 或 LTP 起不来时为 False）"""
    syntax_errors: list
    """省掉句法证据的原因（LTP 抛异常时记原文，不吞掉）"""


def _syntax_block(sentence: str) -> tuple:
    """一段句法证据（分词/词性 + 依存句法），取不到时返回原因

    LTP 起不来（没装模型、显存不够）不该带走整批：省掉这一段，模型仍能凭句子作答，
    只是退化成「只凭语感」。事实记进 all_in_one.syntax，并在启动时打一条警告。
    """
    try:
        return st.parse_sentence_text(sentence), None
    except Exception as e:  # 句法分析失败只降级，不让它决定这一条记录跑不跑
        return None, f"{e!r}"


def _candidates_block(label: str, focus_word: dict | None, table: dict, candidates: dict) -> str:
    """一侧的 schema 候选那一段（行首固定是 system prompt 里列出的那六个标签之一）

    两个分支，措辞上有个刻意的禁区：**不能说「候选都没命中」**——命中与否要拿语义特征去比，
    而这一轮特征也是模型自己抽的、调用前拿不到。所以有候选就只说「只能从名单里挑」，
    没有候选（这一类下一条编号图式都没有）才让模型自由命名。

    自由命名的前缀与查表那一路同源（st.schema_group_key，别名归组时名字才不会分家）：
    关注词是「外面」而归入「里」这一类时，名字必须是「里 + 编号」，命名成「外面1」
    与判定表里按类写的那套对不上，这一次抽取就白抽了。
    """
    head = f"{label}的 schema 候选："
    if candidates:
        word = (focus_word or {}).get("word") or ""
        class_word = st.schema_group_key(table, focus_word)
        note = f"（关注词「{word}」归入「{class_word}」这一类）" if class_word and class_word != word else ""
        return (
            f"{head}只从下面这些里挑一个最接近的，名字原样照抄（不要自造名字、不要改编号）{note}：\n"
            f"{st.render_schema_candidates(candidates)}"
        )
    prefix = st.schema_group_key(table, focus_word) or ((focus_word or {}).get("word") or "")
    style = f"「{prefix} + 编号」（如 {prefix}1）" if prefix else "「组名 + 编号」"
    return f"{head}这一类下还没有编号图式，请按 {style} 的格式自行命名。"


def build_user_message(
    head: str,
    sentence_a: str,
    sentence_b: str,
    *,
    syntax_a: str | None,
    syntax_b: str | None,
    block_a: str,
    block_b: str,
) -> str:
    """拼这次的 user message

    前 5 行用 sa.pair_head_block（a 侧关注词 2 行 + b 侧 2 行 + 语料类型行）。a 侧按
    FOCUS_ROLE_RECORD 说——与 reality/semantic/feature/schema 四个 agent 一致，注入的 reality
    判据也说「第二行交代了这个词是怎么被提取出来的」；b 侧仍是 FOCUS_ROLE_REPLACEMENT：它的 schema
    要收窄到**替换词**所属的词条，判定表里 上12 × 里1 这类跨组格子才对得上（两侧共用一个词会退化
    成同组自比）。corpus_type 恒不为 None（同 runner：先规范化再传），所以第 5 行恒在、行数恒为 5。

    第 6 行起一律走行首标签（句法证据本身是多行，不能按行号数）：

        1-5  pair_head_block(a, b, role_a=RECORD)（含末行「本次跑批的语料类型：方位词。」）
        6    （空行）
        7    原句：<sentence_a>
        8    原句句法证据：<st.parse_sentence_text(sentence_a)>   ← 多行，--no-syntax 时整段省掉
        …    改写句：<sentence_b>
        …    改写句句法证据：<...>
        …    原句的 schema 候选：<候选名单 或 自由命名要求>
        …    改写句的 schema 候选：<...>
        …    请按顺序作答：① … ② … ③ … ④ … ⑤ …

    为什么句法证据与两个句子在前、候选名单在后：候选名单是第 4 步才要用的东西，放在特征抽取
    之前会让模型按名字反推特征（先入为主）；放到最后则紧挨着第一次用它的地方。两个句子挨着放，
    第 5 步的比较也不必来回跨段。
    """
    parts = [head, "", f"原句：{sentence_a}"]
    if syntax_a:
        parts += ["原句句法证据：", syntax_a]
    parts.append(f"改写句：{sentence_b}")
    if syntax_b:
        parts += ["改写句句法证据：", syntax_b]
    parts += [
        block_a,
        block_b,
        "请按顺序作答：① 第 1 行那个词是不是在说现实空间（reality）"
        "② 两个句子各自的语法是否通顺、空间语义是否异常（semantic.a / semantic.b）"
        "③ 两侧各自的语义特征（features.a / features.b）"
        "④ 两侧各自的 schema（schema.a / schema.b）"
        "⑤ 两句意义是否相同（judgement）。",
    ]
    return "\n".join(parts)


def prepare(state: dict, ctx: "_Context", args: argparse.Namespace) -> _Prepared:
    """把这次调用要发的消息拼好——纯本地计算，不花一次请求

    --no-llm 时调用方根本不会调到这里（消息只为模型而拼），于是不碰 LTP、也不加载模型。
    """
    table = ctx.agents.need("schema").table
    focus_a, focus_b = state.get("focus_word"), state.get("focus_word_b")
    sentence_a, sentence_b = state.get("sentence_a") or "", state.get("sentence_b") or ""

    syntax_a = syntax_b = None
    errors: list[str] = []
    if not args.no_syntax:
        if ctx.syntax_error:
            errors = [ctx.syntax_error]
        else:
            syntax_a, error_a = _syntax_block(sentence_a)
            syntax_b, error_b = _syntax_block(sentence_b)
            errors = [error for error in (error_a, error_b) if error]

    return _Prepared(
        message=build_user_message(
            sa.pair_head_block(focus_a, focus_b, corpus_type=ctx.corpus_type, role_a=sa.FOCUS_ROLE_RECORD),
            sentence_a,
            sentence_b,
            syntax_a=syntax_a,
            syntax_b=syntax_b,
            block_a=_candidates_block("原句", focus_a, table, st.schema_candidates(table, focus_a)),
            block_b=_candidates_block("改写句", focus_b, table, st.schema_candidates(table, focus_b)),
        ),
        syntax={"a": syntax_a is not None, "b": syntax_b is not None},
        syntax_errors=errors,
    )


# --------------------------------------------------------------------------
# 四、一次调用
# --------------------------------------------------------------------------


class _RequestProbe(BaseCallbackHandler):
    """一次 invoke 里真发出几次 chat 请求的回调计数器

    由 call_once 挂在**本次 invoke** 的 config 上，而不是挂在 agent 上：worker 并发下
    每条记录各带各的探针，请求数才按记录归账。on_chat_model_start 在模型每次真正开始
    一次请求时触发——ToolStrategy(handle_errors=True) 的补发是同一个 invoke 里的第二次
    模型请求，会再触发一次（「模型轮次」数不到、这里数得到）。底层 HTTP 连接层的自动
    重试（langchain-openai 的 max_retries）发生在一次模型请求内部，不会再触发本回调，
    不在本口径里。
    """

    def __init__(self) -> None:
        self.count = 0
        self.lock = threading.Lock()

    def on_chat_model_start(self, *args, **kwargs) -> None:
        with self.lock:
            self.count += 1


class CallCounter:
    """本次跑批的调用账：模型轮次、真实请求数与补发次数

    rounds   外层调用数（模型轮次）：只在 call_once 里 +1——「每条记录至多一次调用」
             这句承诺唯一可核对的地方；agent 内部的补发不经过 call_once，数不到；
    requests 实际请求数（含内部补发）：call_once 按每条记录的回调探针（_RequestProbe）累加；
    retried  补发次数：每次调用按 max(0, 请求数 - 1) 累加，未发请求的失败调用不计补发。

    --workers 下 call_once 会从各个 worker 并发调到这来，故三个计数都用锁串起来
    （与 ResultWriter 同一口径）；收尾是等池子跑完再读，读到的是终值。
    """

    def __init__(self) -> None:
        self.rounds = 0
        self.requests = 0
        self.retried = 0
        self.lock = threading.Lock()

    def bump(self) -> None:
        with self.lock:
            self.rounds += 1

    def add_requests(self, count: int) -> None:
        with self.lock:
            self.requests += count
            self.retried += max(0, count - 1)


def call_once(agent, content: str, counter: CallCounter) -> tuple:
    """发一次请求，返回 (结构化输出, 出错原因, 本次真实请求数)

    本函数自己不重试：重试就是第二次请求，与「只调一次」直接冲突。单条记录失败不带走整批，
    原因如实记进 all_in_one.error，下游按 source="error" 诚实记录（闸门由此落到 semantic_error）。

    注意 agent 内部还藏着一条补发路径（ToolStrategy 的 handle_errors，见 build_call_agent）——
    这里管不到它，所以那边默认也是关的。两处一起才构成「这一次就是唯一一次」。补发若真的
    发生，会如实落进返回的请求数里（见 _RequestProbe），并由 CallCounter 按本次请求数累计补发。
    """

    counter.bump()
    probe = _RequestProbe()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": content}]},
            config={"callbacks": [probe]},
        )
    except Exception as e:
        counter.add_requests(probe.count)
        return None, f"调用失败：{e!r}", probe.count
    counter.add_requests(probe.count)
    response = result.get("structured_response") if isinstance(result, dict) else None
    if response is None:
        return None, "模型这次没有给出结构化输出（structured_response 为空）", probe.count
    return response.model_dump(), None, probe.count


def build_call_agent(model_key: str, system_prompt: str, verdict_model: type, *, allow_retry: bool):
    """建这一次调用用的 agent：不给任何工具（给工具就等于允许第二次请求）

    response_format 走 agent_factory.get_agent 的现成通道，它顺带把思考模式关掉——
    deepseek 的思考模式不支持被强制的 tool_choice，而 ToolStrategy 正是靠强制调用
    结构化输出工具实现的（见 agents.get_model）。

    handle_errors 默认关（allow_retry=False）：结构化输出不合法时框架会**补发一次请求**去纠正，
    那就是第二次 API 调用，与「只调一次」这句话直接冲突。所以默认不补发：异常照原样抛出，
    记进 all_in_one.error，闸门落到 semantic_error。想要「宁可多问一次也要拿到结构化输出」
    时才开 --allow-retry。

    注意别指望补发能救回那一类 400（见模块 docstring）：那是模型偶发吐出并发工具调用、
    deepseek 直接拒掉整个请求，重试同样发不出去。
    """
    return agent_factory.get_agent(
        model_key, [], system_prompt, response_format=ToolStrategy(verdict_model, handle_errors=allow_retry)
    )


# --------------------------------------------------------------------------
# 五、先查表、后信模型：逐个对齐对应的 agent
#
# 口径与 spatial_agents 里各个 run() 一一对应。凡是「表给了结论」「没调模型」这两种不需要模型
# 答案的分支，都把活直接交给那个 agent 自己（用它自己的措辞与判据），本脚本只在**表没给结论
# 且这次确实调了模型**时读本次那一次作答。这样 --no-llm 的输出与 pair 图在结构上完全一致。
# --------------------------------------------------------------------------


def resolve_reality(
    agent, sentence: str, focus_word: dict | None, dumps: dict | None, *, llm_used: bool, error: str | None
) -> dict:
    """对齐 RealityAgent.run：只有「调了模型、也确实答了」这一种情形才读本次的答案"""
    if not llm_used:
        return agent.run(sentence, focus_word)  # 用 llm=False 建的那份 ⇒ 恒回 no_llm
    if error or dumps is None:
        return {"is_spatial_scene": None, "source": "error", "reason": error}
    verdict = dumps.get("reality") or {}
    if verdict.get("is_spatial_scene") is None:
        return {"is_spatial_scene": None, "source": "error", "reason": "模型没有给出 is_spatial_scene"}
    return {
        "is_spatial_scene": bool(verdict["is_spatial_scene"]),
        "reason": verdict.get("reason") or "",
        "source": "llm",
    }


def resolve_fluency(
    agent, sentence: str, focus_word: dict | None, side_dump: dict | None, *, llm_used: bool, error: str | None
) -> dict:
    """对齐 SemanticAgent.run：返回形状由 sa.semantic_verdict_result / sa.semantic_blank 合成

    合成的两处（合 is_fluent 的算式、失败时的键集）都从 spatial_agents 取，这里不再自己拼 dict
    ——两次链路各拼一遍正是「对不上账」的来源。
    """
    if not llm_used:
        return agent.run(sentence, focus_word)
    if error or side_dump is None:
        return sa.semantic_blank("error", error)
    if side_dump.get("grammar_ok") is None or side_dump.get("spatial_ok") is None:
        return sa.semantic_blank("error", "模型没有给出 grammar_ok / spatial_ok")
    return sa.semantic_verdict_result(
        side_dump["grammar_ok"],
        side_dump.get("grammar_reason") or "",
        side_dump["spatial_ok"],
        side_dump.get("spatial_reason") or "",
    )


def resolve_schema(
    agent,
    sentence: str,
    features: dict,
    focus_word: dict | None,
    side_dump: dict | None,
    *,
    llm_used: bool,
    error: str | None,
) -> dict:
    """对齐 SchemaAgent.run：查表命中（含兜底条目）优先于模型给的名字

    agent 是用 llm=False 建的，所以它只回 table（命中，含兜底条目）或 no_llm（没命中）两种
    source；命中那一种直接采信，模型挑的那个名字只作诊断留在 all_in_one.raw 里——查表优先正是
    SchemaAgent 的口径，单次调用不该把它换掉。
    """
    local = agent.run(sentence, features, focus_word)
    if local["source"] == "table" or not llm_used:
        return local
    if error or side_dump is None:
        return {"name": None, "source": "error", "reason": error}

    chosen = (side_dump.get("schema_name") or "").strip()
    reason = side_dump.get("reason") or ""
    word = (focus_word or {}).get("word") or ""
    class_word = st.schema_group_key(agent.table, focus_word)
    candidates = st.schema_candidates(agent.table, focus_word)
    if candidates:
        ok = chosen in candidates
        why = f"不在本次的候选名单里（{'、'.join(candidates)}）"
    else:
        prefix = class_word or word
        ok = st.is_schema_name(chosen, prefix)
        why = f"不符合「{prefix or '关注词'} + 编号」的格式"
    if not ok:
        # 宁可如实留空，也不把一个下游对不上的名字写进结果（判定按名字查表，见 JudgeAgent）
        return {"name": None, "source": "llm_invalid", "reason": f"LLM 给出的名字 {chosen!r} {why}；{reason}"}
    return {"name": chosen, "source": "llm" if candidates else "llm_free", "reason": reason}


def resolve_judgement(
    agent,
    sentences: tuple,
    schema: dict,
    features: dict,
    focus: tuple,
    judgement_dump: dict | None,
    *,
    llm_used: bool,
    error: str | None,
) -> dict:
    """对齐 JudgeAgent.run：表里 √/× 优先于模型的 same_meaning，○/miss/illegal 才用模型的

    两侧的 schema 名用**上一步的最终名字**（含 llm_free；非法时已是 None，按「未判出」标签走）
    ——查表要的就是这两侧名字；用模型原始输出会让「表说不同、模型改了名字」这类分歧算不清。
    """
    local = agent.run(
        sentences[0],
        sentences[1],
        schema.get("a") or {},
        schema.get("b") or {},
        features.get("a") or {},
        features.get("b") or {},
        focus[0],
        focus[1],
    )
    if local["source"] == "table" or not llm_used:
        return local
    if error or judgement_dump is None:
        return {"same_meaning": None, "source": "error", "lookup": local.get("lookup"), "reason": error}
    same = judgement_dump.get("same_meaning")
    if same is None:
        return {
            "same_meaning": None,
            "source": "error",
            "lookup": local.get("lookup"),
            "reason": "模型没有给出 same_meaning",
        }
    return {
        "same_meaning": bool(same),
        "source": "llm",
        "lookup": local.get("lookup"),
        "reason": judgement_dump.get("reason") or "",
    }


# --------------------------------------------------------------------------
# 六、装配：闸门决定采信哪些块
# --------------------------------------------------------------------------


_LlmFlag = namedtuple("_LlmFlag", "llm")
"""sg.pair_semantic_gate 只读 agent.llm 这一个属性

本脚本没有 semantic agent 实例可以传（那份 agent 是用 llm=False 建的，只为拿措辞与表），
所以垫一个同形状的替身，把「这次到底调没调模型」如实告诉那个纯函数。不复制一份闸门逻辑：
闸门取值（semantic_ok / semantic_skipped / semantic_failed_a|b|both / semantic_error）是
判读结果的第一现场，两处各写必然分叉。
"""


def _apply(state: dict, updates: dict) -> dict:
    """把节点返回的 updates 并进状态：history 累加，其余覆盖

    本脚本不跑图，但改写那一步刻意直接调 sg.alter_node。它返回的 history 在 langgraph 里是
    「追加」（状态上挂的是 operator.add），这里就得按同一口径自己累加，否则逐节点的上下文
    只剩最后一条——那正是 PairState 里警告过的退化。
    """
    history = [*(state.get("history") or []), *(updates.get("history") or [])]
    return {**state, **updates, "history": history}


class _Outcome(NamedTuple):
    """装配的中间结果：五个块、最终的 stage、走到过哪几步"""

    stage: str
    status_reason: str
    reality: dict
    semantic: dict
    features: dict
    features_source: str | None
    features_detail: dict
    schema: dict
    judgement: dict
    reached: dict


def assemble(state: dict, ctx: "_Context", *, dumps: dict | None, error: str | None, llm_used: bool) -> _Outcome:
    """按 pair 图的闸门顺序决定采信哪些块

    闸门链与 spatial_graphs 里那条路径逐字对齐：

        stage == "altered"        -> reality_gate       （False 才拦，判不出的三档一律放行）
        stage in REALITY_CONTINUE -> pair_semantic_gate （两句都通顺才继续）
        stage in (ok, skipped)    -> feature / schema / judgement 依次采信

    bad_record / alter_missed 既不是 "altered"、也不在 REALITY_CONTINUE 里，于是自然停在原地
    ——它们的记录里 reality 之后的块全是空的，与 pair 图落到 save 时一致。

    模型把五步都答了（system prompt 明确要求各步互不牵连），**采信与否在这里施加**：没走到的
    步骤留空，答案本身留在 all_in_one.raw 里，suppressed 逐项标明。
    """
    focus_a, focus_b = state.get("focus_word"), state.get("focus_word_b")
    focus = (focus_a, focus_b)
    sentences = (state.get("sentence_a") or "", state.get("sentence_b") or "")
    reached = {
        "reality": state.get("stage") == "altered",
        "semantic": False,
        "feature": False,
        "schema": False,
        "judgement": False,
    }
    stage, reason = state.get("stage", "unknown"), state.get("status_reason", "")
    reality: dict = {}
    if reached["reality"]:
        reality = resolve_reality(
            ctx.agents.need("reality"), sentences[0], focus_a, dumps, llm_used=llm_used, error=error
        )
        stage, reason = sg.reality_gate(reality)

    semantic: dict = {}
    features: dict = {}
    features_source: str | None = None
    features_detail: dict = {}
    schema: dict = {}
    judgement: dict = {}
    if stage in sg.REALITY_CONTINUE:
        reached["semantic"] = True
        semantic = {
            side: resolve_fluency(
                ctx.agents.need("semantic"),
                sentences[index],
                focus[index],
                (dumps or {}).get("semantic", {}).get(side) if dumps else None,
                llm_used=llm_used,
                error=error,
            )
            for index, side in enumerate(("a", "b"))
        }
        # 闸门只读 agent.llm 与两侧的 is_fluent，它那个 state 形参用不到
        stage, reason = sg.pair_semantic_gate(_LlmFlag(llm_used), {}, semantic)

    if stage in ("semantic_ok", "semantic_skipped"):
        reached["feature"] = reached["schema"] = reached["judgement"] = True
        feature_agent = ctx.agents.need("feature")
        if not llm_used:
            local = feature_agent.run(sentences[0], focus_a)  # llm=False ⇒ 恒回 no_llm
            feature_results = {side: {**local, "features": dict(local["features"])} for side in ("a", "b")}
        elif error or dumps is None:
            # 调用失败 / 没给出结构化输出：与 FeatureAgent.run 的 error 分支同形
            feature_results = {
                side: {"features": {}, "source": "error", "reason": error or "未返回结构化输出"}
                for side in ("a", "b")
            }
        else:
            feature_results = {
                side: {
                    "features": (dumps.get("features") or {}).get(side) or {},
                    "source": feature_agent.table_source,
                    "reason": "",
                }
                for side in ("a", "b")
            }
        feature_fields = sg.pair_feature_update({"status_reason": reason}, feature_results)
        features = feature_fields["features"]
        features_source = feature_fields["features_source"]
        features_detail = feature_fields["features_detail"]
        reason = feature_fields.get("status_reason", reason)

        schema_dumps = (dumps or {}).get("schema", {}) if dumps else {}
        schema = {
            side: resolve_schema(
                ctx.agents.need("schema"),
                sentences[index],
                features.get(side) or {},
                focus[index],
                schema_dumps.get(side),
                llm_used=llm_used,
                error=error,
            )
            for index, side in enumerate(("a", "b"))
        }
        judgement = resolve_judgement(
            ctx.agents.need("judge"),
            sentences,
            schema,
            features,
            focus,
            (dumps or {}).get("judgement") if dumps else None,
            llm_used=llm_used,
            error=error,
        )

    return _Outcome(stage, reason, reality, semantic, features, features_source, features_detail, schema, judgement, reached)


def _lookups(outcome: _Outcome) -> dict:
    """两张查表的旁证：走到过那一步就记下最终落在哪儿，没走到就 null

    它解释的是「表会怎么判、模型这一答与表差在哪」，与记录里采信的那一份互为对照。
    没走到时给 null，而不是拿空特征去查一次假装有结论。
    """
    return {
        "schema_names": {side: (outcome.schema.get(side) or {}).get("name") for side in ("a", "b")}
        if outcome.reached["schema"]
        else None,
        "judgement": {"same_meaning": outcome.judgement.get("same_meaning"), "lookup": outcome.judgement.get("lookup")}
        if outcome.reached["judgement"]
        else None,
    }


def process_state(state: dict, *, ctx: "_Context", args: argparse.Namespace, call_agent, counter: CallCounter) -> dict:
    """跑一条记录：纯规则改写 →（改写成功才）一次调用 → 按闸门装配 → 落盘用的记录"""
    started = time.perf_counter()
    state = _apply(state, sg.alter_node(ctx.agents.need("alter"))(state))

    prepared, dumps, error, requests = None, None, None, 0
    if state.get("stage") == "altered" and (ctx.llm or args.print_prompt):
        prepared = prepare(state, ctx, args)
        if args.print_prompt:
            print_numbered(
                f"user message（第 {state['record'].get('line_no')} 行语料，待选句 {state.get('candidate_index')}）",
                prepared.message,
            )
        elif ctx.llm:
            dumps, error, requests = call_once(call_agent, prepared.message, counter)

    outcome = assemble(state, ctx, dumps=dumps, error=error, llm_used=ctx.llm)
    state = _apply(state, {"stage": outcome.stage, "status_reason": outcome.status_reason})
    # 只写**走到过**的那几个块，与图上的路由同形：alter 没成功就去 save，reality 节点根本不跑，
    # 记录里是 "reality": {}（serialize_pair 对缺席的键取 {}）。若这里把五步答案无条件写进去，
    # alter_missed 的记录会凭空多出 reality / semantic 块，而且那些块的 source 会记成
    # no_llm/error——明明没人问过、也没出错，逐字段对账时一眼就是它。
    blocks = {
        key: value
        for key, reached, value in (
            ("reality", "reality", outcome.reality),
            ("semantic", "semantic", outcome.semantic),
            ("features", "feature", outcome.features),
            ("schema", "schema", outcome.schema),
            ("judge", "judgement", outcome.judgement),
        )
        if outcome.reached[reached]
    }
    if outcome.reached["feature"]:
        blocks["features_source"] = outcome.features_source
        blocks["features_detail"] = outcome.features_detail
    state = _apply(state, blocks)
    answered = dumps is not None
    state = _apply(
        state,
        {
            "history": [
                sg._step(
                    "all_in_one",
                    stage=outcome.stage,
                    status_reason=outcome.status_reason,
                    llm_used=ctx.llm,
                    suppressed=[name for name, ok in outcome.reached.items() if answered and not ok],
                    elapsed=round(time.perf_counter() - started, 2),
                ),
                sg._step(
                    "llm",
                    called=answered or error is not None,
                    error=error,
                    syntax=prepared.syntax if prepared else None,
                    syntax_errors=prepared.syntax_errors if prepared else None,
                ),
                sg._step("gate", reached=dict(outcome.reached), stage=outcome.stage),
            ]
        },
    )

    table = ctx.agents.need("schema").table
    record = sg.serialize_pair(state)
    record["all_in_one"] = {
        "model": ctx.model_key,
        # 三个调用口径：pair_llm_calls 是理论调用数（每状态按走到哪一步累加、不计缓存与复用），
        # llm_calls 是模型轮次（外层调用，每条记录至多 1 次），requests 是实际请求数
        # （回调实测，含框架内部补发，见 CallCounter）
        "llm_calls": 1 if (answered or error is not None) else 0,
        "requests": requests,
        "corpus_type": ctx.corpus_type,
        "feature_classes": [feature_class["key"] for feature_class in ctx.feature_classes],
        "prompt": {"name": PROMPT_FILE, "path": ctx.prompt_path, "reality_path": ctx.reality_path},
        "syntax": prepared.syntax if prepared else None,
        "candidates": {
            side: list(st.schema_candidates(table, state.get("focus_word" if side == "a" else "focus_word_b")))
            for side in ("a", "b")
        },
        "reached": dict(outcome.reached),
        "pair_llm_calls": PAIR_CALLS_PER_STEP["reality"] * outcome.reached["reality"]
        + PAIR_CALLS_PER_STEP["semantic"] * outcome.reached["semantic"]
        + PAIR_CALLS_PER_STEP["feature"] * outcome.reached["feature"]
        + PAIR_CALLS_PER_STEP["schema"] * outcome.reached["schema"]
        + PAIR_CALLS_PER_STEP["judgement"] * outcome.reached["judgement"],
        "suppressed": {name: bool(answered and not ok) for name, ok in outcome.reached.items()},
        "lookups": _lookups(outcome),
        "raw": dumps,
        "error": error,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    return record


# --------------------------------------------------------------------------
# 七、跑批上下文
# --------------------------------------------------------------------------


class _Context(NamedTuple):
    """本次运行一建到底的东西：agent 套件、提示语、结构化输出模型、输出路径"""

    agents: sa.AgentSet
    corpus_type: str
    model_key: str
    llm: bool
    out_path: Path
    system_prompt: str
    verdict_model: type
    feature_classes: list
    prompt_path: str
    reality_path: str
    syntax_error: str | None = None
    """预填充失败的原因；本轮省略句法证据，避免每条记录再次尝试不可用的 LTP。"""


def build_context(args: argparse.Namespace, corpus_type: str, out_path: Path) -> _Context:
    """建本次运行要用的六个 agent、两份提示语与结构化输出模型

    六个 agent 一律 **llm=False**：它们在这里只提供配置表、提示语与 --no-llm 的措辞，
    一次模型都不建（「不调 LLM」于是成为构造期事实，而不是运行期小心避开的一串分支）。
    结构照 sg.GRAPHS["pair"].needs 来——本脚本要对照的就是 pair 图那一条链路。
    """
    overrides = (
        {"alter": {"rules": st.load_alter_rules(args.alter_rules), "rules_path": args.alter_rules}}
        if args.alter_rules
        else {}
    )
    agents = sa.build_agents(
        sg.GRAPHS["pair"].needs,
        model_key=args.model,
        llm=False,
        overrides=overrides,
        corpus_type=corpus_type,
    )
    reality_agent = agents.need("reality")
    feature_agent = agents.need("feature")
    feature_classes = list(feature_agent.feature_classes)
    return _Context(
        agents=agents,
        corpus_type=corpus_type,
        model_key=args.model,
        llm=not args.no_llm,
        out_path=out_path,
        system_prompt=build_system_prompt(feature_agent, reality_criteria=reality_agent.system_prompt),
        verdict_model=build_verdict_model(feature_classes),
        feature_classes=feature_classes,
        prompt_path=str(utils.prompt_path(PROMPT_FILE)),
        reality_path=reality_agent.prompt_paths[sa.REALITY_PROMPT_FILE],
    )


def plan_sentences(states: list) -> list:
    """本次真正会喂给 LTP 的句子：每个状态的原句与它那一个待选句

    比 sg.pair_plan 精确——那个把全部待选句都算上（图会跑遍它们，本脚本默认只跑一个）。
    少喂几句只影响速度；一个都不能漏才是要紧的（漏掉的那句会在拼消息时首次前向）。
    """
    sentences: list[str] = []
    for state in states:
        sentence_slice = state["record"].get("sentence") or []
        if not isinstance(sentence_slice, list) or not sentence_slice:
            continue
        index = state.get("candidate_index") or 0
        expanded = state.get("candidates") or []
        chosen = expanded[index] if 0 <= index < len(expanded) else None
        if chosen is None:
            continue
        sentences += [st.join_slice(sentence_slice), chosen["sentence"]]
    return sentences


def expand_states(record: dict, ctx: _Context, args: argparse.Namespace) -> list:
    """一条语料展开成本次要跑的初始状态（同 sg.alter_init_states 的口径）

    默认只取 --candidate-index 那一个待选句（本脚本是单次调用的诊断入口，一条语料一次调用）。
    请求的下标不在展开结果里时**照样建一个同下标的初始状态**，让 alter 节点如实记 alter_missed：
    悄悄换成第 0 个，等于把人点名要看的那一条换掉了还不说。
    """
    states = sg.alter_init_states([record], ctx.agents, str(ctx.out_path))
    if args.all_candidates:
        return states
    for state in states:
        if state.get("candidate_index") == args.candidate_index:
            return [state]
    state = sg.init_state(record, str(ctx.out_path))
    state.update(candidates=states[0].get("candidates", []) if states else [], candidate_index=args.candidate_index)
    return [state]


# --------------------------------------------------------------------------
# 八、命令行入口
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单次调用的对照实现：一条语料只发一次请求，把 pair 图五个 LLM 判断一次答完",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument("--corpus", default=None, help="输入语料 jsonl 路径（默认按语料类型取 results/ 下对应那份）")
    parser.add_argument(
        "--corpus-type",
        default=None,
        metavar="类型",
        type=sr._corpus_type_arg,
        help="语料类型：方位词（别名 localization/location/loc）或 趋向动词（别名 direction/dir/direction_verb）。"
        "决定读 config_files/<类型目录>/ 下那组表与 reality 判据，并写进消息第 5 行；默认方位词",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"最多处理多少条语料（默认 {DEFAULT_LIMIT}）——本脚本是单次调用的诊断入口，不是跑批工具",
    )
    parser.add_argument("--offset", type=int, default=0, help="跳过前多少条语料")
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
        help="该语料展开出的第几个待选句（默认第 0 个，即句内首个位置的首个替换词）；"
        "序号在**全位置展开**上数，所以 0 之后的候选可能落在句内另一个位置；"
        "越界时如实记 alter_missed，不悄悄换成别的。"
        "注意记录里的 id 第二、三段是另一套从 1 起的数（该候选是句内第几个目标词、该词第几个"
        "替换句，见 sg._record_id），别拿这里的 0 基下标去对",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="跑遍每条语料展开出的全部待选句，含句内每个目标词位置各自那批"
        "（有几条待选句就几次调用；收尾打印总次数）",
    )
    parser.add_argument("--model", default=sa.DEFAULT_MODEL, help="model_config/api_keys.json 里的配置名")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发处理的记录数（LLM 调用是 IO 密集）；默认 1 = 逐条跑。只改同时跑几条，"
        "不改每条记录的语义（仍至多一次调用），输出顺序也仍是输入顺序",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不建模型、不调用：纯规则改写 + 两张查表 + 闸门，0 次调用（输出与 pair 图同口径，可逐字段对账）",
    )
    parser.add_argument(
        "--alter-rules",
        default=None,
        help="替换规则表路径（默认读本语料类型目录下的 alter_rules.json5）；走 build_agents 的注入通道",
    )
    parser.add_argument("--out-dir", default=None, help=f"输出目录（默认 results/{DEFAULT_OUT_PREFIX}）")
    parser.add_argument("--out-file", default=None, help="直接指定输出文件（默认按时间戳生成）")
    parser.add_argument(
        "--no-syntax",
        action="store_true",
        help="不把句法证据放进消息：与六 agent 那条路不可比（那边把句法证据内联进消息，且没有这个开关）",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="只带行号打印 system prompt 与 user message，0 次调用、不落盘",
    )
    parser.add_argument(
        "--allow-retry",
        action="store_true",
        help="结构化输出不合法时允许框架补发一次请求（handle_errors=True）。默认**不允许**：补发就是"
        "第二次调用，与「只调一次」冲突，而且那条补发请求实测会被 API 拒掉",
    )
    parser.add_argument(
        "--no-pairs-csv",
        action="store_true",
        help="不导出成对信息 csv（默认导出，列与行序跟 pair 图一致；与 spatial_runner 同名同义）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # args.corpus_type 已被 argparse 规范化（None 仍是 None），这里才落默认值：
    # 落到 None 上就分不清「没指定」与「显式指定方位词」了（同 spatial_runner）
    corpus_type = st.normalize_corpus_type(args.corpus_type)
    corpus_path = (
        Path(args.corpus) if args.corpus else _PROJECT_ROOT / "results" / st.CORPUS_TYPE_CORPORA[corpus_type]
    )
    records = st.read_corpus(corpus_path, limit=args.limit, offset=args.offset)

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "results" / DEFAULT_OUT_PREFIX
    out_path = Path(args.out_file) if args.out_file else st.build_output_path(out_dir, prefix=DEFAULT_OUT_PREFIX)

    print("单次调用版（all_in_one）：一次请求答完 pair 图的五个 LLM 判断")
    print(f"语料类型：{corpus_type}（{'--corpus-type' if args.corpus_type else '默认'}）")
    print(f"配置目录：config_files/{st.CORPUS_TYPE_DIRS[corpus_type]}/")
    # 与 runner 同一道提醒：语料文件名像另一类语料时出声（类型只由 --corpus-type 决定，不按文件名推断）
    sr._warn_corpus_type_mismatch(corpus_path, corpus_type)
    ctx = build_context(args, corpus_type, out_path)
    print(f"模型：{args.model}（{'--no-llm：不建、不调' if not ctx.llm else '每条记录至多一次调用'}）")
    print(f"输入语料：{corpus_path}（{len(records)} 条）")
    if not records:
        print(f"没有从 {corpus_path} 读到记录")
        return
    # 启动时逐张打印本次读的配置表（含 --alter-rules 换过的那张）：缺表/空表当场说清楚，
    # 与 pair 图同一处实现、同一套告警口径（alter_rules 空是警告，另三张空是提示）
    sr._print_config_tables(ctx.agents)
    print("关注词表：" + "；".join(f"{kind} {len(words)} 个" for kind, words in st.load_focus_vocab().items()))
    if args.no_syntax:
        print("提示：--no-syntax 已指定，消息里不含句法证据（与六 agent 那条路不可比）")

    states = [state for record in records for state in expand_states(record, ctx, args)]
    print(f"展开待选句：{len(records)} 条语料 → {len(states)} 个状态（每个状态至多一次调用）")

    if (ctx.llm or args.print_prompt) and not args.no_syntax:
        # 不调模型、也不打印消息时根本不碰 LTP：那些消息只为模型而拼。
        # --no-syntax 时消息里根本不放句法证据，预填充就是白算一遍（prepare 那边同样跳过）
        unique = [s for s in dict.fromkeys(plan_sentences(states)) if s]
        if unique:
            try:
                filled = st.prefill_ltp(unique)
            except Exception as e:
                ctx = ctx._replace(syntax_error=f"LTP 预填充失败：{e!r}")
                print(f"警告：{ctx.syntax_error}；本轮省略句法证据，继续处理。")
            else:
                print(f"LTP 预填充：本次需 {len(unique)} 句，实际送入模型 {filled} 句（其余命中缓存）")
                # 与 spatial_runner._prefill 同一套自检：推理是在锁外做的（见 sentenceTable.get_task_value），
                # 预填充没兜住的句子会由 worker 线程现场补算，那正是「多线程并发调同一份模型」的窗口
                missing = [s for s in unique if not st.cache_ready(s)]
                if missing:
                    print(f"警告：预填充后仍有 {len(missing)} 句不在缓存里，--workers 大于 1 时可能并发推理")
                if len(unique) > config.CACHE_MAX_SIZE:
                    print(
                        f"警告：本次需 {len(unique)} 句，超过缓存容量 {config.CACHE_MAX_SIZE}，"
                        "多线程下可能边填边淘汰；建议减小 --limit 或调大 config.CACHE_MAX_SIZE"
                    )

    if args.print_prompt:
        print_numbered("system prompt（all_in_one.txt 渲染后）", ctx.system_prompt)
        for state in states:
            process_state(state, ctx=ctx, args=args, call_agent=None, counter=CallCounter())
        print("\n（--print-prompt：0 次调用、不落盘）")
        return

    call_agent = (
        build_call_agent(args.model, ctx.system_prompt, ctx.verdict_model, allow_retry=args.allow_retry)
        if ctx.llm
        else None
    )
    writer = sr.ResultWriter(
        out_path,
        counter_names=sg.GRAPHS["pair"].counter_names,
        tally=sg.pair_tally,
        corpus_type=corpus_type,
        config_tables=sr._config_tables(ctx.agents),
        # 成对信息 csv：默认导出（--no-pairs-csv 关），列与行序跟 pair 图一致（用 pair 图那套列）
        csv_columns=sg.GRAPHS["pair"].csv_columns if not args.no_pairs_csv else (),
    )
    counter = CallCounter()
    suppressed: Counter = Counter()
    failed = 0
    failed_lines: list[int] = []
    pair_calls = 0
    """按每状态走到哪一步累加的 pair 侧朴素调用数（不计其按句/按消息的复用，见收尾打印）"""
    started = time.time()
    print(f"输出文件：{out_path}")
    if args.no_pairs_csv:
        print("成对信息 csv：本次不导出（--no-pairs-csv）")
    else:
        print(f"成对信息 csv：收尾导出到 {out_path.with_suffix('.csv')}（--no-pairs-csv 可关）")
    # 落盘与统计都留在主线程：pool.map 按输入顺序产出，于是输出文件与串行跑逐字节同序
    # （line_no 升序、同句再按 candidate.index）。pair 图那边是各 worker 在 save 节点并发落盘、
    # 顺序是完成顺序，但两侧收尾都由 ResultWriter.finish 按同一个键重排，所以两份文件最终同序、
    # 可逐行对账（本侧的重排是兜底，顺序本来就对）。worker 里只有 CallCounter 在自加，
    # 它自带锁；异常照旧在此处抛出（与 runner 的 pool.map 同一行为），不吞进记录里。
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for record in pool.map(
            lambda state: process_state(state, ctx=ctx, args=args, call_agent=call_agent, counter=counter),
            states,
        ):
            writer.write(record)
            for name, blocked in record["all_in_one"]["suppressed"].items():
                if blocked:
                    suppressed[name] += 1
            if record["all_in_one"]["error"]:
                failed += 1
                failed_lines.append(record["line_no"])
            pair_calls += record["all_in_one"]["pair_llm_calls"]

    # 各 agent 的调用账：本脚本的 agent 一律 llm=False，数值恒为全零，但 summary 的形状与 pair 图一致
    writer.agent_stats = {name: agent.stats() for name, agent in ctx.agents.items()}
    summary = writer.finish()
    print(
        f"\n完成 {summary['count']} 条记录（{len(records)} 条语料，{len(states)} 个状态）；"
        f"模型轮次 {counter.rounds} 次（每条记录至多 1 次外层调用）"
    )
    print(
        f"实际请求 {counter.requests} 次（回调实测，含框架内部补发 {counter.retried} 次；"
        "底层 HTTP 连接层重试不在此列）"
    )
    # 对照数是按**逐条实际走到哪一步**累加的，不是「记录数 × 换算系数」：停在 reality 闸门的记录
    # 本来只要一次调用，一律按整条链路算会把这批语料说得比实际贵。
    #
    # 但它**不计 pair 侧的任何复用**：reality 按原句去重、semantic/feature/schema 的 a 侧消息逐字
    # 相同也只问一次（见 PAIR_CALLS_PER_STEP），所以这只是个上界——只扣 reality 那一半反而不成立。
    # 要精确数就去读 pair 产物 summary 里的 agent_stats.roundtrips。
    print(
        f"对照（理论调用数，不计缓存）：这些状态按每状态走到哪一步累加是 {pair_calls} 次调用；"
        "pair 侧还会按句/按消息复用，实际往返数更低——以 pair 产物 summary 里的 "
        "agent_stats.roundtrips 为准"
    )
    if failed:
        # 给出语料行号与重跑办法：这类失败多半是模型偶发地吐出并发工具调用，deepseek 那边直接回
        # 400（assistant 消息里的 tool_calls 没有被逐条回应），重跑一次通常会好
        print(f"调用失败 {failed} 条（原因见记录里的 all_in_one.error）：语料第 {failed_lines} 行")
        for line_no in failed_lines:
            print(f"    重跑：--offset {line_no - 1} --limit 1")
    if suppressed:
        print(f"被闸门挡下、未采信的模型答案：{dict(suppressed)}（原答案见 all_in_one.raw）")
    for name in sg.GRAPHS["pair"].counter_names:
        print(f"{name}：{summary[name]}")
    print(f"用时 {summary['elapsed_seconds']} 秒（总耗时 {round(time.time() - started, 1)} 秒）")
    print(f"汇总文件：{summary['summary_file']}")


if __name__ == "__main__":
    main()
