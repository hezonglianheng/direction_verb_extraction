# encoding: utf8

"""把 spatial_agents.py 里的 agent 接成图：状态、节点适配器、各图 builder、图注册表。

本模块只做"谁接谁"：agent 做什么在 spatial_agents.py，跑批与落盘在 spatial_runner.py。
三个模块的依赖方向是单向的：spatial_runner → spatial_graphs → spatial_agents → spatial_tools。

现有的三张图（GRAPHS）：

    pair         六个 agent 全用：alter → reality → semantic → feature → schema → judge → save
                 改写未命中、或 reality 判为「不是现实空间用法」则直接 save 收尾。产出前缀 spatial_pairs。
    single       单句：read → reality → 语义 → 特征 → schema。用来看单个句子抽出来的东西是否合理。
                 前缀 spatial_single。
    alter_check  改写检查：alter → reality → semantic（两句各判一次通顺），语义那一步不做闸门路由
                 （两侧都判、不通顺也照样落盘），目的就是看改写句与原句各自是否通顺。
                 前缀 spatial_alter_check。

三张图的第一道闸门都是 reality（见 spatial_agents.RealityAgent）：判这一句里的关注词是不是在说
现实世界中的空间场景，判为 False 的记录就此收尾，后面几步的模型调用一概不发生。它排在各自的
入口节点（alter / read）**之后**、五个 LLM agent 之前：那两个入口都是纯规则（不调模型、不花钱），
先跑它们，停下来的记录里才带着句子与切片，事后能看到被滤掉的是哪一句。判不出来时一律放行
（见 reality_gate）：闸门只拦明确判为 False 的，不拿「没问过模型」当「不是空间用法」。

图内一律是"一对句子"（a 原句 / b 改写句），可一个原句能改出好几个待选句（见 spatial_tools.alter_sentence）：
**有几个待选句就展开成几条记录**，每条记录仍是完整的一对句子，用 candidate 块标明是第几个待选句。
展开放在跑批前一次算好（见 GraphSpec.init_states），所以图本身、serialize 与统计口径的形状都不用变。

一对句子的两侧**各看各的关注词**：a 侧看记录里那个词（focus_word），b 侧看规则表换上去的那个词
（focus_word_b，由 alter_node 用 spatial_tools.replacement_focus_word 算出）。所以 semantic / feature /
schema 三个节点传下去的关注词是**按侧取**的（inputs 里写 {"a": ..., "b": ...}，见 _resolve），
而不是两侧共用一个字段。这样 b 侧的 schema 才会收窄到替换词所属的词组，判定表里
上12 × 里1 这种跨组的格子才对得上；两侧共用原词的话 b 侧会收窄回原词组，判定退化成同组自比。
judge 是唯一要同时看到两份关注词的节点——它的 user message 开头两份、各占 2 行，就是这两份
（见 pair_head_block）。

b 侧那份的 index 是照搬 a 侧的（改写只动了切片里那一个 token）：改出来的切片（slice_b）上它指的就是
替换词，但模型拿到的是改写句字符串、会自己重新分词，替换点并词时同一下标会指到别的 token，
所以 b 侧的措辞只说这个下标是原句上的位置（见 spatial_agents.FOCUS_INDEX_NOTE_ALTER）。

新建一张图的步骤：

    1. 在本模块写一个 TypedDict 状态，继承 BaseState，只加这张图自己的字段；
    2. 写 build_xxx_graph(agents, writer)，用 agent_node 把需要的 agent 包成节点，
       用 StateGraph 接起来，最后 compile()；
    3. 写 init_states(records, agents, output_file)、plan(records, agents) 与 serialize(state)，
       以及 tally/counter_names；
    4. 在 GRAPHS 里登记一个 GraphSpec。跑批无需改动：spatial_runner.py --graph <名字>。

    不需要的 agent 别写进 needs：needs 里没有的名字根本不会被构造，
    feature 那种带 subagent 的 deep agent 构造一次很贵，能省就省。
"""

import operator
import time
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, Callable, NamedTuple, TypedDict, get_type_hints

from langgraph.graph import END, START, StateGraph

import spatial_tools as st

if TYPE_CHECKING:  # 只为类型标注，避免与 spatial_runner 相互 import
    from spatial_runner import ResultWriter


# --------------------------------------------------------------------------
# 一、图的状态
# --------------------------------------------------------------------------


class BaseState(TypedDict, total=False):
    """各图状态共用的部分

    history 用 operator.add 累加，各节点只返回自己新增的条目。

    警告：子类**不要**重新声明 history。langgraph 走 MRO 取注解，子类里再写一遍
    history: Annotated[list[dict], operator.add] 会把 Annotated 里的 reducer 覆盖掉，
    history 静默退化成整体替换（只剩最后一条）。已实测：子类只新增字段时累加正常。
    """

    record: dict
    """原始语料记录"""
    output_file: str
    """结果落盘路径，随记录一起存档"""
    focus_word: dict | None
    """本次需要重点关注的词 {"index", "word", "kind", "source"}，见 spatial_tools.pick_focus_word

    它是 record 的纯函数，三张图都要用（single 图也拿它标注这一句是为哪个词抽的），
    所以由 init_state 统一写入；alter_node / single_entry_node 再按 record 复算一次写回，
    图被单独 invoke（状态里没这个字段）时也不会退化成 None。
    """

    reality: dict
    """第一道闸门的判定 {"is_spatial_scene", "reason", "source"}，见 spatial_agents.RealityAgent

    三张图都有它，位置也一致：各自的入口节点之后、五个 LLM agent 之前。
    没走到这一步的记录（bad_record、改写未命中）里是空的。
    """

    stage: str
    """走到哪一步停下，供条件边路由"""
    status_reason: str
    """停下或判定异常的原因"""
    history: Annotated[list[dict], operator.add]
    """逐节点的上下文记录，按发生顺序追加"""

    # 以下三个字段由跑批前的展开写入（见 alter_init_states）：一段语料能改出好几个待选句，
    # 就展开成几个状态，每个状态负责其中一个。不改写的图（single）不会写它们。
    candidate: dict
    """本状态负责的待选句（AlterAgent.expand 的一项）；没展开出待选句时为 None"""
    candidates: list[dict]
    """该语料展开出的全部待选句，供留档对照"""
    candidate_index: int
    """本状态是第几个待选句，与 candidate 里的 candidate_index 一致"""


def _step(node: str, **fields) -> dict:
    """构造一条 history 记录"""
    return {"node": node, "at": datetime.now().isoformat(timespec="seconds"), **fields}


def init_state(record: dict, output_file: str) -> BaseState:
    """把一条语料记录转成图的初始状态（三张图通用，各自的字段由节点自己写）"""
    return {
        "record": record,
        "output_file": str(output_file),
        "focus_word": st.pick_focus_word(record),
        "stage": "unknown",
        "status_reason": "",
        "history": [],
    }


# --------------------------------------------------------------------------
# 二、节点适配器：把任意 agent 包成 langgraph 节点
# --------------------------------------------------------------------------


def _state_keys(state_type) -> set[str]:
    """状态类型接受的全部字段名（含从 BaseState 继承来的）"""
    try:
        return set(get_type_hints(state_type))
    except Exception:  # 注解里有解析不了的名字时退回到本类自己的注解
        return set(getattr(state_type, "__annotations__", {}))


def _resolve(state: dict, path, side: str | None):
    """按状态路径取值，取不到给 None

    路径支持点分与 {side} 占位："sentence_{side}"、"features.{side}"、"schema.a"。

    也可以写成按侧取的字典 {"a": 路径, "b": 路径}：同一个 agent 形参在两侧喂不同的字段，
    改写句那侧单看一个关注词就是这么接的（见 pair_semantic_node）。
    形态在 agent_node 建图时就校验过（sides 与字典的键必须对得上），这里只管按 side 取。

    取不到就给 None 而不是抛异常：图里提前收尾是常态（比如没走到 schema 就落盘），
    agent 侧对 None 做兜底（见 SchemaAgent.run / JudgeAgent.run）。
    """
    if isinstance(path, dict):
        path = path.get(side)
        if path is None:
            return None
    parts = (path.replace("{side}", side) if side is not None else path).split(".")
    value: Any = state
    for part in parts:
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def agent_node(
    agent,
    *,
    node: str,
    inputs: dict,
    output: str | None = None,
    sides: tuple[str, ...] | None = None,
    update: Callable[[dict, Any], dict] | None = None,
    detail: Callable[[dict, Any, dict | None], dict] | None = None,
    state_type=None,
) -> Callable[[dict], dict]:
    """把 agent 包成节点：取状态里的输入 → 跑 agent → 写回状态并记一条 history

    Args:
        agent: spatial_agents 里的 agent 实例
        node (str): 节点名，同时也是 history 里的 node 名
        inputs (dict): agent 的形参名 -> 状态路径。路径可以是字符串（支持 {side} 占位与点分路径），
            也可以是按侧取的字典 {"a": 路径, "b": 路径}——键必须与 sides 一致。声明了 sides 时
            字符串路径必须含 {side}（两侧同源的值也要走字典形式写两遍），否则建图即报错
        output (str, optional): 把 agent 的返回值写进哪个状态字段。给了 sides 时
            写成 {"a": ..., "b": ...}；不给则由 update 自己写
        sides (tuple, optional): 给了就按 ("a", "b") 各跑一次
        update (callable, optional): (state, value) -> 额外写回的状态字段。
            value 是 agent 的返回值（两侧是 {side: 返回值}）
        detail (callable, optional): (state, value, update 结果) -> 这条 history 记什么
        state_type (optional): 用于在建图时校验 output 字段确实存在

    建图时就对账 agent 的输入契约与接线，少接一条线立即报错，
    而不是等跑到那条记录才 TypeError（见 spatial_agents.BaseAgent.INPUTS）。
    """
    expected, given = set(agent.INPUTS), set(inputs)
    if expected != given:
        diff = []
        if expected - given:
            diff.append(f"缺少 {sorted(expected - given)}")
        if given - expected:
            diff.append(f"多出 {sorted(given - expected)}")
        raise ValueError(f"{node} 节点的接线与 {agent.NAME}_agent 的输入不匹配：" + "，".join(diff))
    if sides:
        sides = tuple(sides)
    elif any("{side}" in path for path in inputs.values() if isinstance(path, str)):
        raise ValueError(f"{node} 节点的状态路径用了 {{side}} 占位，但没声明 sides")
    # 声明了 sides 就得两侧真的分开取值：一个不含 {side} 的字符串路径会让 a/b 两侧拿到同一个值，
    # 那正是「b 侧悄悄用回原词」这类错误的样子（改了关注词却漏改一条线），建图时就拦下来。
    # 确实是两侧同源的值（跑批级常量之类）就走按侧字典写两遍，明确写下「两侧取同一处」。
    if sides:
        for name, path in inputs.items():
            if isinstance(path, str) and "{side}" not in path:
                raise ValueError(
                    f"{node} 节点的 {name} 路径 {path!r} 不含 {{side}}，"
                    f"但节点声明了 sides {list(sides)}——两侧会取到同一个值；"
                    f'两侧确实同源就写成 {{"a": {path!r}, "b": {path!r}}}'
                )
    # 按侧取的路径（{"a": ..., "b": ...}）与 sides 必须一一对上：少一侧就取到 None，
    # 那正是「b 侧悄悄用回原词」这类错误的样子，建图时就拦下来，别等跑到记录才发现
    for name, path in inputs.items():
        if not isinstance(path, dict):
            continue
        if not sides:
            raise ValueError(f"{node} 节点的 {name} 按侧取值（{sorted(path)}），但没声明 sides")
        if set(path) != set(sides):
            raise ValueError(
                f"{node} 节点的 {name} 按侧取值的键 {sorted(path)} 与 sides {list(sides)} 对不上"
            )
    if output is not None and state_type is not None and output not in _state_keys(state_type):
        raise ValueError(f"{node} 节点的输出字段 {output!r} 不在 {state_type.__name__} 里")
    # 输入路径同样要对得上状态字段：写错一个字段名（sentence_{side} 写成 sentences_{side}）
    # 既不报错也不影响落盘，_resolve 取到 None 就喂进 agent，模型那边只看到「…：None」，
    # 直到语义层结果被污染才看得出来。这里只查路径的**首段**是不是状态字段，
    # 往下的形状（dict 里有没有那个键）交给运行期——状态是逐节点长出来的，建图时看不全。
    if state_type is not None:
        for name, path in inputs.items():
            for one in (path.values() if isinstance(path, dict) else [path]):
                first = one.split(".")[0]
                for side in sides or (None,):
                    field = first.replace("{side}", side) if side is not None else first
                    if "{" in field:
                        continue  # 还剩别的占位符：形态本身已经在上面校验过，这里不猜
                    if field not in _state_keys(state_type):
                        raise ValueError(
                            f"{node} 节点的输入 {name} 路径 {one!r} 的首段 {field!r} "
                            f"不是 {state_type.__name__} 的字段"
                        )

    def node_fn(state: dict) -> dict:
        started = time.perf_counter()
        if sides:
            value = {
                side: agent.run(**{name: _resolve(state, path, side) for name, path in inputs.items()})
                for side in sides
            }
        else:
            value = agent.run(**{name: _resolve(state, path, None) for name, path in inputs.items()})

        wrote = update(state, value) if update else None
        fields: dict = {"elapsed": round(time.perf_counter() - started, 2)}
        if detail:
            fields.update(detail(state, value, wrote))

        changes = dict(wrote or {})
        if output is not None:
            changes[output] = value
        changes["history"] = [_step(node, **fields)]
        return changes

    return node_fn


# --------------------------------------------------------------------------
# 三、跑批前的展开：一条语料 -> 若干初始状态
# --------------------------------------------------------------------------


def alter_init_states(records: list[dict], agents, output_file: str) -> list[dict]:
    """pair / alter_check 两张图的展开：一条语料按它的待选句展开成多个初始状态

    改写在跑批前先算一遍（纯规则、无副作用，见 AlterAgent.expand）：一个原句有几个待选句，
    就展开成几个状态，每个状态跑一条完整链路、落一条记录。没有待选句（规则表没命中）或
    sentence 字段本身不合法时仍展开一个状态，好让 alter 节点照旧如实记下 alter_missed / bad_record。

    节点里还会再调一次同一个纯函数（微秒级），换来的是图能脱离本函数单独 invoke。
    """
    alter = agents.need("alter")
    states: list[dict] = []
    for record in records:
        sentence_slice = record.get("sentence") or []
        candidates = []
        if isinstance(sentence_slice, list) and sentence_slice:
            candidates, _ = alter.expand(sentence_slice, st.pick_focus_word(record) or {})

        if not candidates:
            # 展开不出待选句也得跑一个状态，否则这条语料不会留下任何记录
            state = init_state(record, output_file)
            state.update(candidates=[], candidate_index=0)
            states.append(state)
            continue
        for candidate in candidates:
            state = init_state(record, output_file)
            state.update(candidates=candidates, candidate_index=candidate["candidate_index"])
            states.append(state)
    return states


def single_init_states(records: list[dict], agents, output_file: str) -> list[dict]:
    """单句图的展开：这张图不改写，一条语料就是一个状态"""
    return [init_state(record, output_file) for record in records]


# --------------------------------------------------------------------------
# 四、reality 闸门：三张图共用的第一道关卡
# --------------------------------------------------------------------------


REALITY_OK = "spatial_ok"
"""闸门放行的 stage：关注词确实在说现实世界里的空间"""
REALITY_NOT_SPATIAL = "not_spatial"
"""闸门停下（记录 status 也是它）的 stage：关注词在这一句里不是现实空间用法"""
REALITY_SKIPPED = "spatial_skipped"
"""闸门放行的 stage：没调 LLM（--no-llm），无从判断"""
REALITY_UNKNOWN = "spatial_unknown"
"""闸门放行的 stage：判不了（没有关注词、调用失败），不拿它当「不是空间用法」"""

REALITY_CONTINUE = (REALITY_OK, REALITY_SKIPPED, REALITY_UNKNOWN)
"""放行的那三个 stage（三档判不出来的一律在里面，理由见 reality_gate）"""


def reality_gate(value: dict) -> tuple[str, str]:
    """由 reality_agent 的判定推出 (stage, reason)

    True -> 放行；False -> 停下（唯一会拦的值）；判不出来 -> 放行，但三档各记一个 stage
    （汇总里的 reality_counts 据此分桶，见 _tally_reality）。

    「判不出来」绝不能当成「不是空间用法」：没问过模型（--no-llm）、这条记录没有关注词可判、
    模型调用失败，三件事都不是「这一句不在说现实空间」的证据。按后者处理的话，一次接口抖动
    就会把整批记录拦在闸门上，而 --no-llm 的用途恰恰是绕开模型去验纯规则环节。代价是这些
    记录会照旧往下走一遍（白跑几步），比静默丢掉一批语料划算得多。
    """
    if value.get("is_spatial_scene") is True:
        return REALITY_OK, ""
    if value.get("is_spatial_scene") is False:
        return REALITY_NOT_SPATIAL, f"关注词在句中没有表达现实世界中的空间场景：{value.get('reason')}"
    if value.get("source") == "no_llm":
        return REALITY_SKIPPED, "未调用 LLM（--no-llm），跳过空间场景闸门"
    return REALITY_UNKNOWN, value.get("reason") or "没有判出是否表达现实空间场景"


def reality_node(agent, *, sentence: str, state_type) -> Callable[[dict], dict]:
    """闸门节点：判这一句里的关注词是不是在说现实空间，并按下闸门

    sentence 是「这一句」在哪：pair / alter_check 读 sentence_a（语料记录里那个句子，
    alter 成功与否都写下了它），single 读 sentence（read 节点拼好的）。三张图的位置一致
    ——各自的入口节点之后、五个 LLM agent 之前（理由见模块 docstring）。

    只看 a 侧：闸门判的是这条语料自己的那个关注词，与改写句那侧换上去的词无关，
    所以它接的 focus_word 是原词，不按侧取（见 pair_head_block 那一路只管 judge）。
    """

    def update(state: dict, value: dict) -> dict:
        stage, reason = reality_gate(value)
        return {"stage": stage, "status_reason": reason}

    def detail(state: dict, value: dict, wrote: dict) -> dict:
        return {
            "stage": wrote["stage"],
            "is_spatial_scene": value.get("is_spatial_scene"),
            "reason": wrote["status_reason"],
        }

    return agent_node(
        agent,
        node="reality",
        inputs={"sentence": sentence, "focus_word": "focus_word"},
        output="reality",
        update=update,
        detail=detail,
        state_type=state_type,
    )


def route_after_reality(state: dict) -> str:
    """判为「不是现实空间用法」才停下（经 save 落盘后结束）；判不出来的照旧往下走"""
    return "semantic" if state.get("stage") in REALITY_CONTINUE else "save"


# --------------------------------------------------------------------------
# 五、pair 图：六个 agent 全用，一对句子走到底
# --------------------------------------------------------------------------


class PairState(BaseState):
    """一条记录走完全流程的状态，也是结果记录的主体"""

    slice_a: list[str]
    sentence_a: str
    slice_b: list[str]
    sentence_b: str
    alter: dict
    """改写详情：index、原词、选中的替换词、它的来源词组，以及全部待选词（candidates）"""
    focus_word_b: dict | None
    """改写句那一侧的关注词 {"index", "word", "kind", "source"}

    由 alter_node 用 spatial_tools.replacement_focus_word 从 focus_word 与选中的替换词算出
    （index 不变、kind 按替换词重算、source 记 "alter"）。改写没命中时为 None。

    focus_word（a 侧，来自语料记录）仍在 BaseState 里，两侧分开存是为了让 b 侧的 schema
    收窄到替换词所属的词组——合用一个词的话收窄回去的仍是原词那一组，判定表就白写了。
    """

    semantic: dict
    """{"a": {...}, "b": {...}}"""
    features: dict
    """{"a": {特征对象}, "b": {特征对象}}"""
    features_source: str
    schema: dict
    """{"a": {...}, "b": {...}}"""
    judge: dict
    """{"same_meaning": bool, "source": ..., "lookup": ..., "reason": ...}

    lookup 记的是判定表那一查落在哪一类：same/different（表里写了 √/×，直接出结论）、
    llm（表里写了 ○，交 LLM）、miss（表里没这一对）、illegal（这一格非法留空）。
    """


def alter_node(agent) -> Callable[[dict], dict]:
    """alter 节点：原句切片 + 关注词 -> 待选句里由本状态负责的那一条。纯规则

    这一步不走适配器：它的入口是 record 而不是某个 sentence_* 字段，还要判 bad_record、
    把结果拆成 a/b 两侧、并从展开出的多个待选句里取出自己那一个——这些是"一对句子"
    这个概念固有的逻辑，不是 agent 的输入契约。pair 图与 alter_check 图共用这一个节点。

    candidate_index 由跑批前的展开写进状态（见 alter_init_states），本节点只按它取一条。

    改写成功时顺手把 b 侧的关注词换成规则表换上去的那个词（focus_word_b）：这件事只有这里做得成，
    选中的替换词是这里才知道的。三个分支都会写 focus_word_b（改写没命中则是 None），
    好让下游按侧取词时总能取到"确定没有"而不是"忘了写"。
    """

    def node(state: dict) -> dict:
        started = time.perf_counter()
        record = state["record"]
        sentence_slice = record.get("sentence") or []
        if not isinstance(sentence_slice, list) or not sentence_slice:
            return {
                "stage": "bad_record",
                # 句子用不了，但关注词仍按 record 记上：落盘口径与正常记录一致（见 _save_node）
                "focus_word": state.get("focus_word") or st.pick_focus_word(record),
                # 没改写过，b 侧自然没有替换词可看
                "focus_word_b": None,
                "status_reason": "记录的 sentence 字段缺失或不是分词切片",
                "history": [
                    _step("alter", stage="bad_record", elapsed=round(time.perf_counter() - started, 2))
                ],
            }

        # 状态里没有（图被单独 invoke）就按 record 复算一次。这里不是「唯一入口」：
        # 这一行与 init_state、alter_init_states、pair_plan、single_entry_node 是**五处并列**的
        # 调用点（grep st.pick_focus_word 就能数全），口径之所以不会分叉，全靠它们都走同一个
        # 纯函数 st.pick_focus_word(record)。改取词口径要改那个函数，别只改 init_state。
        focus_word = state.get("focus_word") or st.pick_focus_word(record)
        sentence_a = st.join_slice(sentence_slice)
        elapsed = round(time.perf_counter() - started, 2)

        # 与展开同源（同一个纯函数），只是这里只挑一条
        candidates, detail = agent.expand(sentence_slice, focus_word or {})
        index = state.get("candidate_index") or 0
        chosen = candidates[index] if 0 <= index < len(candidates) else None
        if chosen is None:
            # 没有待选句（规则表没命中）或 index 越界（图被单独 invoke 时才会出现）：
            # 产不出一对句子，如实记下原因后直接收尾，不往下走
            detail["reason"] = detail["reason"] or f"待选句下标 {index} 越界（共 {len(candidates)} 个）"
            # 与改写成功那条分支同形状：alter 块里都带 kind，免得下游按 alter["kind"] 读时
            # 在 alter_missed 的记录上取到缺失值
            detail["kind"] = (focus_word or {}).get("kind")
            return {
                "stage": "alter_missed",
                "status_reason": detail["reason"],
                "slice_a": sentence_slice,
                "sentence_a": sentence_a,
                "slice_b": sentence_slice,
                "sentence_b": sentence_a,
                "alter": detail,
                "focus_word": focus_word,
                # 没有替换词，b 侧不猜：下游按侧取词时取到 None，agent 会如实说"没有可用的关注词"
                "focus_word_b": None,
                "candidate": None,
                "candidates": candidates,
                "history": [
                    _step(
                        "alter",
                        stage="alter_missed",
                        elapsed=elapsed,
                        focus_word=focus_word,
                        detail=detail,
                    )
                ],
            }

        # alter 里既有选中的那个替换词（顶层，方便直接读），也留着全部候选词供对照
        alter = {
            **detail,
            "replacement": chosen["replacement"],
            "source": chosen["source"],
            "group": chosen["group"],
            "kind": (focus_word or {}).get("kind"),
            "candidate_index": chosen["candidate_index"],
            "candidate_total": chosen["candidate_total"],
        }
        # b 侧的关注词就是换上去的那个词：index 沿用 a 侧的（改写只动了 index 处的 token，
        # 见 st.alter_sentence 的 new_slice[index] = replacement），kind 按替换词重算——
        # 替换词若不在本语料的词表里，kind 就是空串（消息里因此没有类别标注，汇总里的
        # replacement_kinds 记 "none" 数它）。收窄看的是**词形**落在哪一组的 组名 ∪ words 里
        # （见 st._narrowed_groups），与 kind 空不空不是一回事：词形也不在 schema 表里时
        # b 侧才真的一条候选都收不到，那属于配置该补的内容。
        focus_word_b = st.replacement_focus_word(
            focus_word, chosen["replacement"], record.get("filter_methods") or []
        )
        return {
            "stage": "altered",
            "status_reason": "",
            "slice_a": sentence_slice,
            "sentence_a": sentence_a,
            "slice_b": chosen["slice"],
            "sentence_b": chosen["sentence"],
            "alter": alter,
            "focus_word": focus_word,
            "focus_word_b": focus_word_b,
            "candidate": chosen,
            "candidates": candidates,
            "history": [
                _step(
                    "alter",
                    stage="ok",
                    elapsed=elapsed,
                    focus_word=focus_word,
                    candidate={
                        "index": chosen["candidate_index"],
                        "total": chosen["candidate_total"],
                        "replacement": chosen["replacement"],
                        "source": chosen["source"],
                    },
                )
            ],
        }

    return node


def pair_semantic_gate(agent, state: dict, value: dict) -> tuple[str, str]:
    """语义闸门：由两句的通顺性判断推出 (stage, reason)

    取值与拆分前逐字一致：semantic_ok / semantic_skipped / semantic_failed_a|b|both / semantic_error。
    --no-llm 下闸门无从判断，记为 semantic_skipped 放行——这一模式的用途正是在不调模型的前提下
    验证改写规则与两张查表，若在此处停下就什么也验证不了。
    """
    verdicts = {side: result["is_fluent"] for side, result in value.items()}
    failed = [side for side, ok in verdicts.items() if ok is not True]
    if not agent.llm:
        return "semantic_skipped", "未调用 LLM（--no-llm），跳过语义闸门"
    if not failed:
        return "semantic_ok", ""
    if verdicts["a"] is None or verdicts["b"] is None:
        return "semantic_error", "；".join(f"{side}: {value[side]['reason']}" for side in failed)
    return (
        "semantic_failed_" + ("both" if len(failed) == 2 else failed[0]),
        "；".join(
            f"{'原句' if side == 'a' else '改写句'}不通顺：{value[side]['reason']}" for side in failed
        ),
    )


def pair_semantic_node(agent, state_type=PairState) -> Callable[[dict], dict]:
    """semantic 节点：对原句与改写句各判一次语义是否通顺，并按下闸门"""

    def update(state: dict, value: dict) -> dict:
        stage, reason = pair_semantic_gate(agent, state, value)
        return {"stage": stage, "status_reason": reason}

    def detail(state: dict, value: dict, wrote: dict) -> dict:
        return {
            "stage": wrote["stage"],
            "verdicts": {side: result["is_fluent"] for side, result in value.items()},
            "reason": wrote["status_reason"],
        }

    return agent_node(
        agent,
        node="semantic",
        # 两侧各看各的词：原句看记录里的词，改写句看换上去的那个词
        inputs={"sentence": "sentence_{side}", "focus_word": {"a": "focus_word", "b": "focus_word_b"}},
        output="semantic",
        sides=("a", "b"),
        update=update,
        detail=detail,
        state_type=state_type,
    )


def pair_feature_node(agent, state_type=PairState) -> Callable[[dict], dict]:
    """feature 节点：对原句与改写句各抽一次语义特征

    写进状态的是特征本身（不是 agent 的整个返回值），所以不用 output，由 update 写两个字段。
    """

    def update(state: dict, value: dict) -> dict:
        return {
            "features": {side: result["features"] for side, result in value.items()},
            "features_source": value["a"]["source"],
        }

    def detail(state: dict, value: dict, wrote: dict) -> dict:
        return {
            "status": "ok",
            "features": {side: result["features"] for side, result in value.items()},
            "source": value["a"]["source"],
        }

    return agent_node(
        agent,
        node="feature",
        # 同 semantic：两侧各看各的词，改写句那侧看的是换上去的那个词
        inputs={"sentence": "sentence_{side}", "focus_word": {"a": "focus_word", "b": "focus_word_b"}},
        sides=("a", "b"),
        update=update,
        detail=detail,
        state_type=state_type,
    )


def pair_schema_node(agent, state_type=PairState) -> Callable[[dict], dict]:
    """schema 节点：定两句各自的 schema——先按关注词收窄、再查表，未命中才由 LLM 从候选里挑

    两句各跑一次，关注词**按侧取**：a 侧用记录里的原词，b 侧用规则表换上去的那个词。
    这正是这张判定表要的——表里的格子按两组名字写（如 上12 × 里1），两侧都得收窄到
    自己那一组才有意义；两侧共用一个词的话，b 侧只会收窄回原词组，判定退化成同组自比
    （见 spatial_tools.replacement_focus_word 与 _narrowed_groups）。
    """

    def detail(state: dict, value: dict, wrote: dict) -> dict:
        return {
            "result": {
                side: {"name": item["name"], "source": item["source"]} for side, item in value.items()
            }
        }

    return agent_node(
        agent,
        node="schema",
        inputs={
            "sentence": "sentence_{side}",
            "features": "features.{side}",
            "focus_word": {"a": "focus_word", "b": "focus_word_b"},
        },
        output="schema",
        sides=("a", "b"),
        detail=detail,
        state_type=state_type,
    )


def pair_judge_node(agent, state_type=PairState) -> Callable[[dict], dict]:
    """judge 节点：先查表，未命中再由 LLM 判断两句意义是否相同

    这是唯一**同时**看到两份关注词的节点，也是两侧合并的地方：查表要的是两侧的 schema 名，
    而 schema 是各自按自己那侧的关注词收窄出来的，所以 LLM 分支也得知道两侧各在看哪个词
    （user message 开头两份、各占 2 行，见 spatial_agents.pair_head_block）。这里不按侧跑，
    两份关注词各接一条线、各是一个形参。
    """

    def detail(state: dict, value: dict, wrote: dict) -> dict:
        return {
            "source": value["source"],
            "lookup": value.get("lookup"),
            "same_meaning": value["same_meaning"],
            "reason": value["reason"],
        }

    return agent_node(
        agent,
        node="judge",
        inputs={
            "sentence_a": "sentence_a",
            "sentence_b": "sentence_b",
            "schema_a": "schema.a",
            "schema_b": "schema.b",
            "features_a": "features.a",
            "features_b": "features.b",
            "focus_word": "focus_word",
            "focus_word_b": "focus_word_b",
        },
        output="judge",
        detail=detail,
        state_type=state_type,
    )


def pair_final_status(state: PairState) -> str:
    """由最终状态推出记录里的 status

    stage 记的是"走到哪一步停下"，status 记的是这条记录最终成不成立：
        ok                       语义闸门通过、judge 也给出了判定，一条完整链路
        ok_no_semantic_gate      --no-llm 下跳过语义闸门但走完了后续查表
        judgement_unknown        走到了 judge 却没能判定（无 LLM 或调用失败）
        alter_missed 等其余取值   等于停下时的 stage，原因见 status_reason
    """
    stage = state.get("stage", "unknown")
    if stage not in ("semantic_ok", "semantic_skipped"):
        return stage
    if (state.get("judge") or {}).get("same_meaning") is None:
        return "judgement_unknown"
    return "ok" if stage == "semantic_ok" else "ok_no_semantic_gate"


def _candidate_block(state: dict) -> dict | None:
    """结果记录里的 candidate 块：这条记录是哪个待选句、共几个、换成哪个词、来自哪个词组

    一个原句有几个待选句就落几条记录，靠这个块区分它们；没展开出待选句（alter_missed）时为 None。
    """
    chosen = state.get("candidate")
    if not chosen:
        return None
    return {
        "index": chosen.get("candidate_index"),
        "total": chosen.get("candidate_total"),
        "word": chosen.get("word"),
        "replacement": chosen.get("replacement"),
        "source": chosen.get("source"),
        "group": chosen.get("group"),
    }


def serialize_pair(state: PairState) -> dict:
    """把最终状态整理成一条结果记录

    一条记录包含：两个句子、各自的语义特征、各自对应的 schema、判定信息，
    以及逐节点的上下文历史（context.history）。candidate 块标明这是一条语料展开出的第几个待选句。

    focus_word（a 侧）与 focus_word_b（b 侧）都落盘：两侧的 schema 是各按自己那侧的关注词收窄出来的，
    事后要核对"这一对落在判定表哪一格、为什么"就得两份都在。
    """
    record = state["record"]
    return {
        "status": pair_final_status(state),
        "stage": state.get("stage", "unknown"),
        "status_reason": state.get("status_reason", ""),
        "line_no": record.get("line_no"),
        "output_file": state.get("output_file"),
        "candidate": _candidate_block(state),
        "focus_word": state.get("focus_word"),
        "focus_word_b": state.get("focus_word_b"),
        "source": {
            "source_file": record.get("source_file"),
            "filter_methods": record.get("filter_methods"),
            "match_tree": record.get("match_tree"),
        },
        "reality": state.get("reality", {}),
        "pair": {
            "a": {
                "sentence": state.get("sentence_a"),
                "slice": state.get("slice_a"),
                "alter": None,
                "role": "原句",
            },
            "b": {
                "sentence": state.get("sentence_b"),
                "slice": state.get("slice_b"),
                "alter": state.get("alter"),
                "role": "改写句",
            },
        },
        "semantic": state.get("semantic", {}),
        "features": state.get("features", {}),
        "features_source": state.get("features_source"),
        "schema": state.get("schema", {}),
        "judgement": state.get("judge", {}),
        "context": {
            "history": state.get("history", []),
        },
    }


def pair_plan(records: list[dict], agents) -> list[str]:
    """这张图会让模型看到的全部句子：原句 + 规则算出的全部待选句

    待选句由 alter_agent.plan_all 算（纯规则、无副作用），与节点里真正用到的句子同源。
    没有它，待选句就会在节点里第一次遇到时才做句法分析——那是在线程池里，而推理在锁外，
    多线程下就是并发调同一份 LTP 模型（见 sentenceTable.SentenceTaskCache.get_task_value）。
    一个都不能漏：这条记录展开出几个待选句，这里就得算出几个。
    """
    alter = agents.need("alter")
    sentences = []
    for record in records:
        sentence_slice = record.get("sentence") or []
        if not sentence_slice:
            continue
        sentences.append(st.join_slice(sentence_slice))
        sentences.extend(alter.plan_all(sentence_slice, st.pick_focus_word(record) or {}))
    return sentences


def _tally_focus(record: dict, counters: dict[str, Counter]) -> None:
    """汇总里记一笔关注词的口径：取到的是哪一级（sampled_word/vocab/match_tree）、属哪一类

    这两组计数是排查「这一轮跑的口径对不对」的第一现场：source 全是 match_tree 就说明词表没推出来；
    kind 里趋向动词占比高时，alter_missed 多多半是替换规则表里还没有趋向动词的同义词组
    （规则表缺内容是配置问题，代码不兜底，见 spatial_tools.alter_sentence）。

    这里记的是 **a 侧**（原句）的关注词；改写句那侧（替换词）另有一组，见 _tally_replacement。
    """
    focus = record.get("focus_word") or {}
    counters["focus_sources"][focus.get("source") or "none"] += 1
    counters["focus_kinds"][focus.get("kind") or "none"] += 1


def _tally_replacement(record: dict, counters: dict[str, Counter]) -> None:
    """汇总里记一笔改写句那侧关注词（替换词）的口径：属哪一类，或压根没有

    它是 focus_kinds 的对照面：focus_kinds 记的是原词属哪一类，这一组记的是换上去的词属哪一类。
    替换词不在本语料的词表里时 kind 是空串（记 "none"）——这类记录值得单独看：消息里没有类别标注，
    而且多半意味着替换规则表里放进了本类型词表之外的词形（词形也不在 schema 表的 组名 ∪ words 里时，
    b 侧会一条候选都收不到，直接落到 judge 的 miss）。"absent" 是还没改写过（改写未命中、
    记录本身不可用），与"换上去的词不在表里"是两回事。
    """
    focus_b = record.get("focus_word_b") or {}
    if not focus_b.get("word"):
        counters["replacement_kinds"]["absent"] += 1
        return
    counters["replacement_kinds"][focus_b.get("kind") or "none"] += 1


def _reality_label(item: dict) -> str:
    """一条记录闸门判定的桶名：真判过的按结论分两桶，判不出来的按 source 分（no_llm / error …）

    桶名就是「为什么没判」，与 RealityAgent.OUTPUTS 里的 source 一一对应，外加没走到闸门
    （bad_record、改写未命中）时的 "absent"。
    """
    if item.get("is_spatial_scene") is True:
        return "spatial"
    if item.get("is_spatial_scene") is False:
        return "not_spatial"
    return item.get("source") or "absent"


def _tally_reality(record: dict, counters: dict[str, Counter]) -> None:
    """汇总里记一笔闸门的判定：现实空间用法 / 引申用法 / 各类「判不了」

    这组计数是「闸门开得对不对」的第一现场：status_counts 只能看到停在闸门上的那些，
    看不到放行的那部分判的到底是什么。它一条都不拦（not_spatial 为 0）或几乎全拦（not_spatial
    接近总数）都说明提示语或语料要调，而 no_llm / error 那几桶涨起来就说明这批压根没判
    （见 reality_gate：它们一律放行，不会以 not_spatial 的名义把记录吞掉）。
    """
    counters["reality_counts"][_reality_label(record.get("reality") or {})] += 1


def pair_tally(record: dict, counters: dict[str, Counter]) -> None:
    """pair 图的汇总口径：状态分布、闸门判定、判定分布、判定表的查表分布、两侧 schema 的来源、两侧关注词口径"""
    _tally_focus(record, counters)
    _tally_reality(record, counters)
    _tally_replacement(record, counters)
    counters["status_counts"][record.get("status", "unknown")] += 1
    for side in ("a", "b"):
        counters["schema_sources"][(record.get("schema") or {}).get(side, {}).get("source", "none")] += 1
    judge = record.get("judgement") or {}
    if judge.get("same_meaning") is not None:
        counters["judge_counts"]["same" if judge["same_meaning"] else "different"] += 1
    # 查表分布：same+different 是表里写了 √/× 直接判掉的，llm/miss/illegal 是走 LLM 的三种由头。
    # 跟 judge_counts 一起看才知道「这一轮有多少是手工裁定、多少是模型兜的」——把 miss（表没管）
    # 与 illegal（表该填没填）分开记，是为了让后者的数量成为补表的依据。
    lookup = judge.get("lookup")
    if lookup:
        counters["judge_lookups"][lookup] += 1


def build_pair_graph(agents, writer: "ResultWriter"):
    """按组织逻辑接好六个 agent，返回可 invoke 的图"""
    graph = StateGraph(PairState)
    graph.add_node("alter", alter_node(agents.need("alter")))
    graph.add_node("reality", reality_node(agents.need("reality"), sentence="sentence_a", state_type=PairState))
    graph.add_node("semantic", pair_semantic_node(agents.need("semantic")))
    graph.add_node("feature", pair_feature_node(agents.need("feature")))
    graph.add_node("schema", pair_schema_node(agents.need("schema")))
    graph.add_node("judge", pair_judge_node(agents.need("judge")))
    graph.add_node("save", _save_node(serialize_pair, writer))

    graph.add_edge(START, "alter")
    graph.add_conditional_edges("alter", route_after_alter, {"reality": "reality", "save": "save"})
    graph.add_conditional_edges("reality", route_after_reality, {"semantic": "semantic", "save": "save"})
    graph.add_conditional_edges("semantic", route_after_semantic, {"feature": "feature", "save": "save"})
    graph.add_edge("feature", "schema")
    graph.add_edge("schema", "judge")
    graph.add_edge("judge", "save")
    graph.add_edge("save", END)
    return graph.compile()


def route_after_alter(state: dict) -> str:
    """改写成功才往下走（进 reality 闸门）；没命中规则直接去落盘"""
    return "reality" if state.get("stage") == "altered" else "save"


def route_after_semantic(state: dict) -> str:
    """两句都通顺才继续抽特征、定 schema、做判定；否则指向 END（经 save 落盘后结束）"""
    return "feature" if state.get("stage") in ("semantic_ok", "semantic_skipped") else "save"


def _save_node(serialize, writer: "ResultWriter") -> Callable[[dict], dict]:
    """save 节点：把结果记录写进 jsonl。这是走完全程的终点，也是提前收尾时的落点"""

    def node(state: dict) -> dict:
        record = serialize(state)
        writer.write(record)
        return {"history": [_step("save", status=record["status"], output_file=str(writer.path))]}

    return node


# --------------------------------------------------------------------------
# 六、single 图：read → reality → 语义 → 特征 → schema
# --------------------------------------------------------------------------


class SingleSentenceState(BaseState):
    """单句走一遍语义闸门与抽取，看单个句子抽出来的东西是否合理"""

    slice: list[str]
    sentence: str
    semantic: dict
    """{is_fluent, reason, source}"""
    features: dict
    features_source: str
    schema: dict
    """{name, source, reason}"""


def single_entry_node(state: dict) -> dict:
    """入口节点：从语料记录里取出原句（这张图不改写，只读原句）"""
    started = time.perf_counter()
    record = state["record"]
    sentence_slice = record.get("sentence") or []
    if not isinstance(sentence_slice, list) or not sentence_slice:
        return {
            "stage": "bad_record",
            # 同 alter_node：句子用不了也照样记下关注词，落盘口径与正常记录一致
            "focus_word": state.get("focus_word") or st.pick_focus_word(record),
            "status_reason": "记录的 sentence 字段缺失或不是分词切片",
            "history": [_step("read", stage="bad_record", elapsed=round(time.perf_counter() - started, 2))],
        }
    return {
        "stage": "read",
        "status_reason": "",
        "slice": sentence_slice,
        "sentence": st.join_slice(sentence_slice),
        # 这张图不改写，但语义/特征/schema 仍要知道这一句是为哪个词抽的，理由同 alter_node
        "focus_word": state.get("focus_word") or st.pick_focus_word(record),
        "history": [_step("read", stage="read", elapsed=round(time.perf_counter() - started, 2))],
    }


def route_after_read(state: dict) -> str:
    """读到句子才往下走（进 reality 闸门），否则直接落盘"""
    return "reality" if state.get("stage") == "read" else "save"


def single_semantic_node(agent, state_type=SingleSentenceState) -> Callable[[dict], dict]:
    """semantic 节点：判这一句是否通顺。只有一句，所以不写 {a, b} 而是直接写结果"""

    def update(state: dict, value: dict) -> dict:
        stage, reason = _single_gate(agent, value)
        return {"stage": stage, "status_reason": reason}

    def detail(state: dict, value: dict, wrote: dict) -> dict:
        return {"stage": wrote["stage"], "verdict": value["is_fluent"], "reason": wrote["status_reason"]}

    return agent_node(
        agent,
        node="semantic",
        inputs={"sentence": "sentence", "focus_word": "focus_word"},
        output="semantic",
        update=update,
        detail=detail,
        state_type=state_type,
    )


def _single_gate(agent, value: dict) -> tuple[str, str]:
    """单句版的语义闸门，与 pair 图同一套取值（failed 时不分 a/b）"""
    if not agent.llm:
        return "semantic_skipped", "未调用 LLM（--no-llm），跳过语义闸门"
    if value["is_fluent"] is True:
        return "semantic_ok", ""
    if value["is_fluent"] is None:
        return "semantic_error", value["reason"]
    return "semantic_failed", f"不通顺：{value['reason']}"


def single_feature_node(agent, state_type=SingleSentenceState) -> Callable[[dict], dict]:
    """feature 节点：抽这一句的语义特征"""

    def update(state: dict, value: dict) -> dict:
        return {"features": value["features"], "features_source": value["source"]}

    def detail(state: dict, value: dict, wrote: dict) -> dict:
        return {"status": "ok", "features": value["features"], "source": value["source"]}

    return agent_node(
        agent,
        node="feature",
        inputs={"sentence": "sentence", "focus_word": "focus_word"},
        update=update,
        detail=detail,
        state_type=state_type,
    )


def single_schema_node(agent, state_type=SingleSentenceState) -> Callable[[dict], dict]:
    """schema 节点：定这一句的 schema——先按关注词收窄、再查表，未命中才由 LLM 从候选里挑"""

    def detail(state: dict, value: dict, wrote: dict) -> dict:
        return {"result": {"name": value["name"], "source": value["source"]}}

    return agent_node(
        agent,
        node="schema",
        inputs={"sentence": "sentence", "features": "features", "focus_word": "focus_word"},
        output="schema",
        detail=detail,
        state_type=state_type,
    )


def single_final_status(state: SingleSentenceState) -> str:
    """单句图的 status：这张图没有判定环节，schema 取不到就不能算跑完"""
    stage = state.get("stage", "unknown")
    if stage not in ("semantic_ok", "semantic_skipped"):
        return stage
    if (state.get("schema") or {}).get("name") is None:
        return "schema_unknown"
    return "ok" if stage == "semantic_ok" else "ok_no_semantic_gate"


def serialize_single(state: SingleSentenceState) -> dict:
    """单句图的结果记录：一个句子、它的语义特征、对应的 schema 与逐节点历史"""
    record = state["record"]
    return {
        "status": single_final_status(state),
        "stage": state.get("stage", "unknown"),
        "status_reason": state.get("status_reason", ""),
        "line_no": record.get("line_no"),
        "output_file": state.get("output_file"),
        "focus_word": state.get("focus_word"),
        "source": {
            "source_file": record.get("source_file"),
            "filter_methods": record.get("filter_methods"),
            "match_tree": record.get("match_tree"),
        },
        "reality": state.get("reality", {}),
        "sentence": state.get("sentence"),
        "slice": state.get("slice"),
        "semantic": state.get("semantic", {}),
        "features": state.get("features", {}),
        "features_source": state.get("features_source"),
        "schema": state.get("schema", {}),
        "context": {
            "history": state.get("history", []),
        },
    }


def single_plan(records: list[dict], agents) -> list[str]:
    """这张图只把原句喂给模型"""
    return [st.join_slice(r.get("sentence") or []) for r in records if r.get("sentence")]


def single_tally(record: dict, counters: dict[str, Counter]) -> None:
    """单句图的汇总口径：状态分布 + 闸门判定 + schema 来源（只有一句，不分两侧）+ 关注词口径"""
    _tally_focus(record, counters)
    _tally_reality(record, counters)
    counters["status_counts"][record.get("status", "unknown")] += 1
    counters["schema_sources"][(record.get("schema") or {}).get("source", "none")] += 1


def build_single_graph(agents, writer: "ResultWriter"):
    """接单句图：read →（读到句子）reality →（现实空间用法）semantic →（通顺）feature → schema → save"""
    graph = StateGraph(SingleSentenceState)
    graph.add_node("read", single_entry_node)
    graph.add_node("reality", reality_node(agents.need("reality"), sentence="sentence", state_type=SingleSentenceState))
    graph.add_node("semantic", single_semantic_node(agents.need("semantic")))
    graph.add_node("feature", single_feature_node(agents.need("feature")))
    graph.add_node("schema", single_schema_node(agents.need("schema")))
    graph.add_node("save", _save_node(serialize_single, writer))

    graph.add_edge(START, "read")
    graph.add_conditional_edges("read", route_after_read, {"reality": "reality", "save": "save"})
    graph.add_conditional_edges("reality", route_after_reality, {"semantic": "semantic", "save": "save"})
    graph.add_conditional_edges(
        "semantic", route_after_semantic, {"feature": "feature", "save": "save"}
    )
    graph.add_edge("feature", "schema")
    graph.add_edge("schema", "save")
    graph.add_edge("save", END)
    return graph.compile()


# --------------------------------------------------------------------------
# 七、alter_check 图：改写 → 闸门 → 两句各判一次通顺
# --------------------------------------------------------------------------


class AlterCheckState(BaseState):
    """只关心改写本身：改写句与原句各自是否通顺

    状态里的字段与 PairState 的前半段相同（有 a/b 两句），后面的特征/schema/判定都不需要。
    """

    slice_a: list[str]
    sentence_a: str
    slice_b: list[str]
    sentence_b: str
    alter: dict
    focus_word_b: dict | None
    """改写句那一侧的关注词，口径同 PairState：这张图的语义节点也按侧取词（见 alter_node）"""
    semantic: dict
    """{"a": {...}, "b": {...}}"""


def alter_check_final_status(state: AlterCheckState) -> str:
    """这张图的 status 就等于 stage

    它没有下游产物可以为跳过的闸门背书，所以不把 semantic_skipped 谎称成 ok：
    --no-llm 下就是 semantic_skipped，一眼能看出这次跑没跑模型。
    """
    return state.get("stage", "unknown")


def serialize_alter_check(state: AlterCheckState) -> dict:
    """改写检查图的结果记录：一对句子 + 各自的通顺性判断（一条语料有几个待选句就几条记录）

    两张关注词都落盘，与 pair 图同口径：这张图看的正是"改写句通不通顺"，
    而改写句那侧的通顺判断是按换上去的那个词问的（见 pair_semantic_node）。
    """
    record = state["record"]
    return {
        "status": alter_check_final_status(state),
        "stage": state.get("stage", "unknown"),
        "status_reason": state.get("status_reason", ""),
        "line_no": record.get("line_no"),
        "output_file": state.get("output_file"),
        "candidate": _candidate_block(state),
        "focus_word": state.get("focus_word"),
        "focus_word_b": state.get("focus_word_b"),
        "source": {
            "source_file": record.get("source_file"),
            "filter_methods": record.get("filter_methods"),
            "match_tree": record.get("match_tree"),
        },
        "reality": state.get("reality", {}),
        "pair": {
            "a": {
                "sentence": state.get("sentence_a"),
                "slice": state.get("slice_a"),
                "alter": None,
                "role": "原句",
            },
            "b": {
                "sentence": state.get("sentence_b"),
                "slice": state.get("slice_b"),
                "alter": state.get("alter"),
                "role": "改写句",
            },
        },
        "semantic": state.get("semantic", {}),
        "context": {
            "history": state.get("history", []),
        },
    }


def alter_check_plan(records: list[dict], agents) -> list[str]:
    """与 pair 图一样：原句与改写句都要看语义，两句都预填充"""
    return pair_plan(records, agents)


def alter_check_tally(record: dict, counters: dict[str, Counter]) -> None:
    """改写检查图的汇总口径：状态分布 + 闸门判定 + 两侧通顺性分布 + 两侧关注词口径"""
    _tally_focus(record, counters)
    _tally_reality(record, counters)
    _tally_replacement(record, counters)
    counters["status_counts"][record.get("status", "unknown")] += 1
    for side in ("a", "b"):
        item = (record.get("semantic") or {}).get(side) or {}
        if not item:
            # 没走到 semantic 节点（比如改写未命中）时该侧没有判断，与"判了但失败"是两回事
            label = "absent"
        elif item.get("source") == "no_llm":
            label = "skipped"
        elif item.get("is_fluent") is True:
            label = "fluent"
        elif item.get("is_fluent") is False:
            label = "not_fluent"
        else:
            label = "error"
        counters["semantic_counts"][label] += 1


def build_alter_check_graph(agents, writer: "ResultWriter"):
    """接改写检查图：alter →（改写成功）reality →（现实空间用法）semantic → save

    这里的 semantic 不做闸门路由：这张图的目的就是看两侧是否通顺，
    无论结果如何都要落盘。reality 那一关照旧拦（它判的是这条语料值不值得判语义，
    与这张图看什么无关），拦下的记录落盘时 stage 是 not_spatial。
    """
    graph = StateGraph(AlterCheckState)
    graph.add_node("alter", alter_node(agents.need("alter")))
    graph.add_node("reality", reality_node(agents.need("reality"), sentence="sentence_a", state_type=AlterCheckState))
    graph.add_node("semantic", pair_semantic_node(agents.need("semantic"), state_type=AlterCheckState))
    graph.add_node("save", _save_node(serialize_alter_check, writer))

    graph.add_edge(START, "alter")
    graph.add_conditional_edges("alter", route_after_alter, {"reality": "reality", "save": "save"})
    graph.add_conditional_edges("reality", route_after_reality, {"semantic": "semantic", "save": "save"})
    graph.add_edge("semantic", "save")
    graph.add_edge("save", END)
    return graph.compile()


# --------------------------------------------------------------------------
# 八、图注册表
# --------------------------------------------------------------------------


class GraphSpec(NamedTuple):
    """一张图的全部信息：跑批侧只认这个接口，新增图不必动驱动"""

    name: str
    """--graph 的名字"""
    prefix: str
    """默认输出目录 results/<prefix> 与输出文件名前缀"""
    doc: str
    """"这张图是干什么的"，--list-graphs 时打印"""
    state: type
    """状态类型，用于节点建的校验"""
    needs: tuple[str, ...]
    """需要哪些 agent（名字取自 spatial_agents.AGENTS）"""
    init_states: Callable[[list[dict], Any, str], list[dict]]
    """把语料展开成初始状态列表：改写有几个待选句，一条语料就展开成几个状态"""
    build: Callable[..., Any]
    serialize: Callable[[dict], dict]
    plan: Callable[[list[dict], Any], list[str]]
    """这张图会让模型看到的全部句子，用于跑批前的 LTP 预填充"""
    counter_names: tuple[str, ...]
    """汇总里输出哪几组计数"""
    tally: Callable[[dict, dict], None]
    """每写一条记录，往 counters 里累加"""


GRAPHS: dict[str, GraphSpec] = {
    "pair": GraphSpec(
        name="pair",
        prefix="spatial_pairs",
        doc="改写并判定：六个 agent 全用，一对句子走完闸门-语义-特征-schema-判定",
        state=PairState,
        needs=("alter", "reality", "semantic", "feature", "schema", "judge"),
        init_states=alter_init_states,
        build=build_pair_graph,
        serialize=serialize_pair,
        plan=pair_plan,
        counter_names=(
            "status_counts",
            "reality_counts",
            "judge_counts",
            "judge_lookups",
            "schema_sources",
            "focus_sources",
            "focus_kinds",
            "replacement_kinds",
        ),
        tally=pair_tally,
    ),
    "single": GraphSpec(
        name="single",
        prefix="spatial_single",
        doc="单句抽取：reality → 语义 → 特征 → schema，只读原句、不改写、不判定",
        state=SingleSentenceState,
        needs=("reality", "semantic", "feature", "schema"),
        init_states=single_init_states,
        build=build_single_graph,
        serialize=serialize_single,
        plan=single_plan,
        counter_names=("status_counts", "reality_counts", "schema_sources", "focus_sources", "focus_kinds"),
        tally=single_tally,
    ),
    "alter_check": GraphSpec(
        name="alter_check",
        prefix="spatial_alter_check",
        doc="改写检查：alter → reality → semantic，只看原句与改写句各自是否通顺，不做语义闸门路由",
        state=AlterCheckState,
        needs=("alter", "reality", "semantic"),
        init_states=alter_init_states,
        build=build_alter_check_graph,
        serialize=serialize_alter_check,
        plan=alter_check_plan,
        counter_names=(
            "status_counts",
            "reality_counts",
            "semantic_counts",
            "focus_sources",
            "focus_kinds",
            "replacement_kinds",
        ),
        tally=alter_check_tally,
    ),
}


def get_graph(name: str) -> GraphSpec:
    try:
        return GRAPHS[name]
    except KeyError:
        raise ValueError(f"没有名为 {name!r} 的图，可用：{'、'.join(GRAPHS)}") from None
