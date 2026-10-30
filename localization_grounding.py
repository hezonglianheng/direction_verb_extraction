# encoding: utf8

"""方位词结构整理与参照物语义闸门。

规则筛选只回答「这个方位词像不像方位短语」，并不保证句子里真的出现了参照物、
被定位物体（目标物）和谓词。本模块在规则命中之后、LLM 筛选与落盘之前补两步：

1. **结构整理**（纯规则，不调模型）
   参照物      方位词在 DEP 里的名词性 ATT 定语（「桌子＋上」的「桌子」），
               并要求它在 SDP 里确实承担 LOC/DATV/FEAT（时间成分是 TIME/rTIME，不是空间）。
   目标物      与参照物共享同一个 SDP 中心语的 EXP/PAT/AGT 子节点。
               「书在桌子上」是 EXP 书；「他把书放到桌子上」是 PAT 书；
               「小鸟站在树上」只有 AGT 小鸟时才取 AGT。绝不固定取主语：
               「他把书放到桌子上」的主语是「他」，被定位物体是「书」。
               普通活动谓词（批评/讲课/查/做…）的 PAT 只是动作对象，不能证明它位于该处，
               因此只认 EXP/AGT；只有放置/位移/姿势类谓词才允许 PAT（见 PLACEMENT_PREDICATES）。
               放置类里的 CONT 也只在「张贴/悬挂/陈列…」等以内容为附着物的谓词下才认。
               「X 上的 Y」里参照物的 LOC 直接指向 Y（桌子 LOC→书），
               且 Y 在目标方位词的 DEP 祖先链上时，Y 本身就是被定位物体，记作 SELF；
               「他放在桌子上的书」这类关系从句的中心语按 rPAT/rAGT/rEXP 记作 REL。
   动词        SDP 中心语或其 SDP 祖先中的谓词；SDP 给不出时回退到目标词的 DEP 动词祖先，
               仍找不到时置 null（动词不是结构必需项）。
               **目标物是结构必需项**：参照物语义合格但找不到目标物时，该方位词位置仍会被剔除。

2. **参照物 / 目标物语义闸门**（OpenHowNet）
   参照物和目标物都必须至少有一个义项（sense），且该义项中至少一个义原沿 hypernym
   到达各自允许的根义原。
   - 参照物：`REFERENCE_SEMEME_ROOTS`（`physical|物质` + 地点类放宽）。
   - 目标物：`TARGET_SEMEME_ROOTS`，只认 `physical|物质` 子树。
   HowNet 没有该词条、或词条没有任何允许的义原时，该候选被剔除。

为什么参照物还含 ``place|地方`` 与 ``InstitutePlace|场所``：
   学校/图书馆/公司这类地标在 HowNet 里归入 InstitutePlace（organization 分支），
   并不在 physical 子树下；但它们是现实空间里的可指称参照物，按「实际物体」处理。
   目标物不额外接受 `InstitutePlace|场所`：学校、公司这类只有该分支义项的目标物会被
   `target_not_physical` 剔除；`place|地方` 本身就在 physical 子树下，因此城市、国家仍可通过。
   抽象组织（organization）、时间（time）、精神（mental）、事情（fact）等均不通过。

异常口径：
   没取到目标词、DEP/SDP 长度不一致、下标越界等**工程问题**一律 fail-open（保留记录、
   记 error），与 sentence_screening 的失败放行政策一致；只有结构明确缺失、参照物
   或目标物语义明确不合格才 fail-closed。
"""

from collections import defaultdict
from typing import Any

from pipeline_utils import prune_tree

# --------------------------------------------------------------------------
# 一、词性与角色常量
# --------------------------------------------------------------------------

REFERENCE_POS = frozenset({"n", "r", "nh", "ni", "nl", "ns", "nz"})
"""可当参照物的词性。"""

LOCATED_POS = frozenset({"n", "r", "nh", "ni", "nl", "ns", "nz"})
"""可当目标物（被定位物体）的词性。"""

REFERENCE_LINK_LABELS = frozenset({"mDEPD", "mRang"})
"""SDP 里方位词依附参照物时使用的标记。"""

REFERENCE_SDP_LABELS = frozenset({"LOC", "DATV", "FEAT"})
"""参照物必须承担的 SDP 空间角色；时间/范围成分是 TIME/rTIME/SCO，不在此列。"""

LOCATED_ROLES = ("EXP", "PAT", "AGT")
"""目标物可承担的基础 SDP 角色（顺序即优先级）。"""

COORDINATED_LINK_LABELS = frozenset({"eCOO"})
"""并列目标物的 SDP 连接标记（「书和证据」里的并列项）。

刻意不收 `eSUCC`：它在 LTP 里更多表示「后续子句/事件」（如「…景象」），
不是同一谓词下的并列目标物；把它当并列会跨子句错选目标物。
"""

CONTENT_OBJECT_PREDICATES = frozenset({"陈列", "悬挂", "展示", "挂", "布置", "张贴", "贴"})
"""这些谓词的 CONT 可以视为被附着物（「海报贴在墙上」的海报）。"""

PLACEMENT_PREDICATES = frozenset({
    "丢", "举", "住", "倒", "停", "关", "写", "列", "刻", "包", "升", "印", "取", "困",
    "坐", "垫", "堆", "塞", "填", "存", "带", "戴", "扔", "扛", "抓", "投", "抛", "抬",
    "抱", "拉", "拖", "拿", "捆", "捡", "掉", "排", "推", "提", "插", "搁", "搬", "摆",
    "放", "晒", "晾", "栽", "沉", "泡", "泼", "洒", "流", "浮", "浸", "游", "滚", "漂",
    "灌", "爬", "画", "盖", "种", "移", "穿", "站", "系", "绑", "背", "落", "藏", "装",
    "裹", "走", "趴", "跑", "跪", "蹲", "躺", "转", "运", "送", "铺", "锁", "降", "飞",
    "位于", "停放", "储存", "坐落", "堆放", "填充", "存放", "安放", "安装", "排列",
    "搁置", "搬运", "摆放", "收藏", "放置", "种植", "移动", "覆盖", "隐藏",
    "上", "下", "出", "到", "回", "挂", "来", "贴", "起", "过", "进", "布置", "张贴",
    "悬挂", "陈列",
})
"""放置/位移/姿势类谓词；只有这类谓词才允许把 PAT/CONT 当被定位物体。"""

PLACEMENT_SUFFIXES = ("到", "在", "进", "出", "上", "下", "回", "过", "起", "来", "去", "着", "了", "给")
"""谓词后的趋向/体标记；判断前逐层剥掉。"""

NON_PLACEMENT_COMPOUNDS = frozenset({"存在", "提出", "提到", "提起"})
"""含放置字样但实际是言语/存在类谓词的词，必须先排除。"""

NON_PHYSICAL_NOUNS = frozenset({
    "世界", "义务", "事件", "事实", "事情", "价值", "价格", "任务", "作用", "信息",
    "做法", "关系", "内容", "冲突", "决定", "利益", "制度", "办法", "动作", "区别",
    "历史", "压力", "原因", "反应", "发展", "变化", "基础", "实际", "局面", "层面",
    "工作", "年代", "建议", "形势", "影响", "态度", "思想", "情况", "想法", "意义",
    "意见", "感情", "成本", "成果", "成绩", "手段", "挑战", "支出", "收入", "政策",
    "效果", "教训", "数据", "数量", "方式", "方案", "方法", "方面", "时期", "机会",
    "权利", "条件", "标准", "根据", "概念", "水平", "活动", "渠道", "状态", "现象",
    "理由", "理论", "目标", "目的", "看法", "矛盾", "知识", "研究", "社会", "程序",
    "精神", "组织", "经验", "结果", "联系", "能力", "范围", "行为", "要求", "观点",
    "规则", "角度", "计划", "语言", "说法", "责任", "质量", "趋势", "过程", "进步",
    "选择", "途径", "问题", "阶段", "需求", "领域", "风险",
})
"""常用事件/抽象名词，只用于 REL/SELF 中心语，不用于普通 EXP/PAT/AGT。

参照物本身是否「实际物体」由 OpenHowNet 判定，不再依赖这份词表——同一个词可能
既有抽象义项又有实体义项（如「组织」的 tissue 义项）。
"""

REFERENCE_SEMEME_ROOTS = frozenset({"physical|物质", "place|地方", "InstitutePlace|场所"})
"""参照物义原向上追溯到这些根之一，即视为「实际物体」。

`place|地方` 与 `InstitutePlace|场所` 是为学校、图书馆、公司这类现实地标放宽的。
"""

TARGET_SEMEME_ROOTS = frozenset({"physical|物质"})
"""目标物义原只认 `physical|物质` 子树，不沿用参照物的地点类放宽。"""

PHYSICAL_SEMEME_ROOTS = REFERENCE_SEMEME_ROOTS
"""旧名保留：等同于参照物根义原集合；目标物请用 TARGET_SEMEME_ROOTS。"""

# --------------------------------------------------------------------------
# 二、统计块
# --------------------------------------------------------------------------


def _is_reference_nominal(index: int, context: dict[str, Any]) -> bool:
    """可当参照物的名词：词性合适；抽象/事件名词交给 OpenHowNet 判，不按词形一刀切。"""
    return _is_nominal(index, context, REFERENCE_POS)


def _is_located_head(index: int, context: dict[str, Any]) -> bool:
    """可当 REL/SELF 中心语的名词：词性合适，且不是常用事件/抽象名词。

    普通 EXP/PAT/CONT/AGT 角色不走这个过滤器，见 NON_PHYSICAL_NOUNS 的说明。
    """
    return _is_nominal(index, context, LOCATED_POS) and context["words"][index] not in NON_PHYSICAL_NOUNS


def _allows_content_object(word: str) -> bool:
    """该谓词的 CONT 是否可当被附着物（张贴/悬挂…，不含写/画）。"""
    base = str(word)
    for _ in range(3):
        if base in CONTENT_OBJECT_PREDICATES:
            return True
        for suffix in PLACEMENT_SUFFIXES:
            if base.endswith(suffix) and len(base) > len(suffix):
                base = base[: -len(suffix)]
                break
        else:
            return False
    return False


STAT_KEYS = (
    "records",
    "kept_records",
    "dropped_records",
    "candidates",
    "kept_candidates",
    "dropped_candidates",
    "no_target",
    "errors",
)
"""可 JSON 序列化、可跨批/跨进程相加的统计键。"""


def empty_stats() -> dict[str, Any]:
    """一份可 JSON 序列化、可跨进程相加的统计块"""
    return {key: 0 for key in STAT_KEYS} | {"reasons": {}}


def merge_stats(total: dict[str, Any], delta: dict[str, Any] | None) -> dict[str, Any]:
    """把一批/一个进程的统计累加进 ``total``，返回 ``total`` 便于链式调用"""
    if not delta:
        return total
    for key in STAT_KEYS:
        total[key] = int(total.get(key, 0)) + int(delta.get(key, 0))
    reasons = total.setdefault("reasons", {})
    for reason, count in (delta.get("reasons") or {}).items():
        reasons[reason] = int(reasons.get(reason, 0)) + int(count)
    return total


# --------------------------------------------------------------------------
# 三、DEP/SDP 上下文
# --------------------------------------------------------------------------


def _head_index(head: Any, size: int) -> int | None:
    """DEP/SDP 的 1-based head -> 0-based 下标；0（根）与越界值返回 None"""
    if not isinstance(head, int) or isinstance(head, bool):
        return None
    if head <= 0 or head > size:
        return None
    return head - 1


def _children_map(heads: list[int], labels: list[str], size: int) -> dict[int, list[tuple[int, str]]]:
    mapping: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for index, (head, label) in enumerate(zip(heads, labels)):
        parent = _head_index(head, size)
        if parent is not None:
            mapping[parent].append((index, label))
    return mapping


def _validate_heads(heads: list[int], name: str, size: int) -> None:
    """head 必须是 0（根）或 1..size；非法值按工程异常处理，交给调用方 fail-open"""
    for index, head in enumerate(heads):
        if (
            not isinstance(head, int)
            or isinstance(head, bool)
            or head < 0
            or head > size
        ):
            raise ValueError(f"{name}[{index}] 非法：{head!r}（只允许 0 或 1..{size}）")


def build_context(tokens: list[str], pos: list[str], dep: dict, sdp: dict) -> dict[str, Any]:
    """把一句的 CWS/POS/DEP/SDP 收成校验用的只读上下文

    Raises:
        ValueError: 四项长度不一致、dep/sdp 缺 head/label，或 head 越界/非整数。
            调用方应 fail-open。
    """
    words = list(tokens or [])
    if not isinstance(dep, dict) or not isinstance(sdp, dict):
        raise ValueError("DEP/SDP 结果不是字典")
    heads = dep.get("head")
    dep_labels = dep.get("label")
    sdp_heads = sdp.get("head")
    sdp_labels = sdp.get("label")
    for name, value in (
        ("DEP.head", heads),
        ("DEP.label", dep_labels),
        ("SDP.head", sdp_heads),
        ("SDP.label", sdp_labels),
    ):
        if not isinstance(value, list):
            raise ValueError(f"{name} 不是列表")
    size = len(words)
    lengths = {len(pos or []), len(heads), len(dep_labels), len(sdp_heads), len(sdp_labels)}
    if lengths != {size}:
        raise ValueError(f"CWS/POS/DEP/SDP 长度不一致：words={size}, lengths={sorted(lengths)}")
    _validate_heads(heads, "DEP.head", size)
    _validate_heads(sdp_heads, "SDP.head", size)
    return {
        "words": words,
        "pos": list(pos),
        "dep_children": _children_map(heads, dep_labels, size),
        "dep_heads": [_head_index(head, size) for head in heads],
        "dep_labels": list(dep_labels),
        "sdp_children": _children_map(sdp_heads, sdp_labels, size),
        "sdp_heads": [_head_index(head, size) for head in sdp_heads],
        "sdp_labels": list(sdp_labels),
    }


def _is_nominal(index: int, context: dict[str, Any], allowed: frozenset[str]) -> bool:
    pos = context["pos"]
    return 0 <= index < len(pos) and pos[index] in allowed


def _is_dep_ancestor(target_index: int, ancestor_index: int, context: dict[str, Any]) -> bool:
    """``ancestor_index`` 是否在目标词的 DEP 祖先链上（含直接父节点）

    「桌子上的书」是直接父节点；「坐在椅子上的学生」里目标「上」的 DEP 链是
    上→在→坐→学生，中心语「学生」是更上层的祖先，也要认。
    """
    current = context["dep_heads"][target_index]
    seen: set[int] = set()
    while current is not None:
        if current in seen:
            raise ValueError(f"DEP 祖先链成环：{sorted(seen)}")
        if current == ancestor_index:
            return True
        seen.add(current)
        current = context["dep_heads"][current]
    return False


def _is_placement_predicate(word: str) -> bool:
    """谓词是否属于放置/位移/姿势类（逐层剥趋向/体标记，每层都先查排除表与词表）"""
    base = str(word)
    for _ in range(5):
        if base in NON_PLACEMENT_COMPOUNDS:
            return False
        if base in PLACEMENT_PREDICATES:
            return True
        for suffix in PLACEMENT_SUFFIXES:
            if base.endswith(suffix) and len(base) > len(suffix):
                base = base[: -len(suffix)]
                break
        else:
            return False
    return False


# --------------------------------------------------------------------------
# 四、参照物 / 目标物 / 动词
# --------------------------------------------------------------------------


def _reference_candidates(target_index: int, context: dict[str, Any]) -> list[tuple[int, int]]:
    """参照物候选：[(下标, 优先级)]，优先级 0 = DEP 定语，1 = SDP 依附标记

    DEP 侧「名词+方位词」是 ATT 子节点（桌子 ATT→上）；SDP 侧方位词反过来依附名词
    （上 mDEPD→桌子）。两边都收，取并集后再要求 SDP 角色是空间角色，避免只靠一棵树漏掉。
    """
    by_index: dict[int, int] = {}
    for child, label in context["dep_children"].get(target_index, []):
        if label == "ATT" and _is_reference_nominal(child, context):
            by_index[child] = min(by_index.get(child, 0), 0)
    sdp_parent = context["sdp_heads"][target_index]
    if (
        sdp_parent is not None
        and context["sdp_labels"][target_index] in REFERENCE_LINK_LABELS
        and _is_reference_nominal(sdp_parent, context)
    ):
        by_index[sdp_parent] = min(by_index.get(sdp_parent, 1), 1)
    by_index.pop(target_index, None)
    return sorted(by_index.items(), key=lambda item: (item[1], abs(item[0] - target_index), item[0]))


def _is_spatial_reference(reference_index: int, context: dict[str, Any]) -> bool:
    """参照物候选是否承担空间角色；FEAT 只认非代词参照物"""
    label = context["sdp_labels"][reference_index]
    if label not in REFERENCE_SDP_LABELS:
        return False
    if label == "FEAT" and context["pos"][reference_index] == "r":
        return False
    return True


def _resolve_predicate(reference_index: int, target_index: int, context: dict[str, Any]) -> int | None:
    """参照物的 SDP 中心语；若它正好是方位词自身/另一个方位词，沿 LOC 链再往上走一层

    实测「学校东边有一条河」：学校 LOC→东边，东边 LOC→有。参照物的中心语因此先是目标词
    「东边」，再往上一跳才是真正的谓语「有」。不跟随这一步会误判成 no_predicate。
    成环/指回自身属于工程异常，抛 ValueError 让调用方 fail-open（不是结构缺失）。
    """
    predicate = context["sdp_heads"][reference_index]
    seen: set[int] = set()
    while predicate is not None:
        if predicate in seen:
            raise ValueError(f"SDP 中心语成环：{sorted(seen)}")
        seen.add(predicate)
        if predicate == reference_index:
            raise ValueError(f"参照物 {reference_index} 的 SDP 中心语指回自身")
        if predicate == target_index or context["pos"][predicate] == "nd":
            predicate = context["sdp_heads"][predicate]
            continue
        return predicate
    return None


def _located_candidates(
    predicate_index: int,
    target_index: int,
    reference_index: int,
    context: dict[str, Any],
    exclude: set[int],
) -> list[dict[str, Any]]:
    """在一个共同 SDP 中心语下找被定位物体，返回按优先级排列的候选。

    同一角色可能有多个并列目标物（「书和证据」里一个标 PAT、一个标 eCOO，
    或两个都标 EXP）。这里把**同一角色**下的候选都返回，而不是只取 index 最小的那个：
    后续目标物语义检查可以逐个尝试，避免“先选到抽象词就误杀整条记录”。
    `SELF/REL` 分支也会带上其 eCOO 并列成员；`eSUCC` 是后续子句连接，不当并列。
    不跨优先级角色回退——例如 PAT 存在时不会退而取 AGT（主语不是被放置物）。
    """
    predicate_word = context["words"][predicate_index]
    placement = _is_placement_predicate(predicate_word)
    predicate_label = context["sdp_labels"][predicate_index]
    predicate_head = context["sdp_heads"][predicate_index]

    def make(index: int, role: str) -> dict[str, Any]:
        return {"index": index, "word": context["words"][index], "role": role}

    all_children = context["sdp_children"]

    def coordinated_related(start_indices: list[int]) -> list[int]:
        """收集 eCOO 并列组里的其他名词（双向）。

        LTP 可能把并列中心语标成 `eCOO`、也可能把它作为别的成员的 `eCOO` 子节点：
          - 「证据(EXP) + 书(eCOO)」：从中心语向下找子节点；
          - 「证据(eCOO) + 书」：从 eCOO 成员沿 SDP head 向上找中心语。
        两个方向都走，避免「先拿到抽象成员就误杀」。
        """
        found: set[int] = set()
        stack = list(start_indices)
        seen: set[int] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node not in (target_index, reference_index) and _is_nominal(node, context, LOCATED_POS):
                if context["sdp_labels"][node] in COORDINATED_LINK_LABELS:
                    found.add(node)
                    parent = context["sdp_heads"][node]
                    if (
                        parent is not None
                        and parent not in exclude
                        and parent not in (target_index, reference_index)
                        and _is_nominal(parent, context, LOCATED_POS)
                    ):
                        # eCOO 成员向上找并列中心语（LTP 可能把中心语标成 EXP/PAT）。
                        found.add(parent)
                        stack.append(parent)
            for child, label in all_children.get(node, []):
                if (
                    label in COORDINATED_LINK_LABELS
                    and child not in exclude
                    and child not in (target_index, reference_index)
                    and _is_nominal(child, context, LOCATED_POS)
                ):
                    found.add(child)
                    stack.append(child)
        return sorted(found)

    def same_role_siblings(candidate: int) -> list[int]:
        """同一 SDP 中心语下、与 candidate 同角色的其他名词。

        LTP 对三个以上并列项可能把首尾都标成直接角色（EXP），中间项标 eCOO：
        「证据(EXP) + 资料(eCOO) + 书(EXP)」。只沿 eCOO 走不到末尾的「书」，
        因此 SELF/REL 还要在共同 SDP 中心语下补同角色兄弟。
        """
        anchor = context["sdp_heads"][candidate]
        if anchor is None:
            return []
        label = context["sdp_labels"][candidate]
        role = context["sdp_labels"][anchor] if label in COORDINATED_LINK_LABELS else label
        return sorted(
            child
            for child, child_label in all_children.get(anchor, [])
            if child_label == role
            and child != candidate
            and child not in exclude
            and child not in (target_index, reference_index)
            and _is_nominal(child, context, LOCATED_POS)
        )

    if predicate_label.startswith("r"):
        allowed = {"rEXP", "rPAT", "rCONT", "rAGT"} if placement else {"rAGT", "rEXP"}
        if predicate_label in allowed:
            if (
                predicate_head is not None
                and predicate_head not in exclude
                and predicate_head not in (target_index, reference_index)
                and _is_located_head(predicate_head, context)
            ):
                related = {
                    child for child in coordinated_related([predicate_head]) if child != predicate_head
                } | set(same_role_siblings(predicate_head))
                return [make(predicate_head, "REL")] + [
                    make(child, context["sdp_labels"][child]) for child in sorted(related)
                ]
            if placement:
                return []
        elif placement:
            return []

    if _is_nominal(predicate_index, context, LOCATED_POS):
        if (
            _is_located_head(predicate_index, context)
            and _is_dep_ancestor(target_index, predicate_index, context)
            and predicate_index not in exclude
            and predicate_index not in (target_index, reference_index)
        ):
            related = {
                child for child in coordinated_related([predicate_index]) if child != predicate_index
            } | set(same_role_siblings(predicate_index))
            return [make(predicate_index, "SELF")] + [
                make(child, context["sdp_labels"][child]) for child in sorted(related)
            ]
        return []

    if placement:
        roles = list(LOCATED_ROLES[:2])
        if _allows_content_object(predicate_word):
            roles += ["CONT"]
        roles += ["AGT"]
    else:
        roles = ["EXP", "AGT"]

    children = context["sdp_children"].get(predicate_index, [])

    def allowed_child(child: int, label: str, role: str) -> bool:
        return (
            label == role
            and child not in exclude
            and child not in (target_index, reference_index)
            and _is_nominal(child, context, LOCATED_POS)
        )

    for role in roles:
        found = sorted(child for child, label in children if allowed_child(child, label, role))
        if found:
            # 同一角色下的并列项（书和证据）也参与目标物检查；
            # 放在直接角色候选之后，优先保留直接标注为目标物的那个。
            coordinated = [
                child
                for child in coordinated_related([predicate_index] + found)
                if child not in found
            ]
            return [make(child, role) for child in found] + [
                make(child, context["sdp_labels"][child]) for child in coordinated
            ]

    # 没有直接角色标注、只剩并列连接时，仍把它们作为候选交给语义检查。
    coordinated = coordinated_related([predicate_index])
    return [make(child, context["sdp_labels"][child]) for child in coordinated]


def _located_for(
    predicate_index: int,
    target_index: int,
    reference_index: int,
    context: dict[str, Any],
    exclude: set[int],
) -> dict[str, Any] | None:
    """兼容旧调用：返回第一个目标物候选；找不到返回 None。"""
    candidates = _located_candidates(predicate_index, target_index, reference_index, context, exclude)
    return candidates[0] if candidates else None


def _verb_for(
    predicate_index: int | None,
    target_index: int,
    reference_index: int,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    """谓词：优先取 SDP 中心语里的动词；SDP 给不出时沿目标词的 DEP 祖先链找动词。

    「桌子上的书」这类没有谓词的定中结构返回 None，而不是把中心名词当动词。
    """
    if predicate_index is not None and context["pos"][predicate_index].startswith("v"):
        return {"index": predicate_index, "word": context["words"][predicate_index]}

    current = context["dep_heads"][target_index]
    seen: set[int] = set()
    while current is not None and current not in seen:
        seen.add(current)
        if current not in (target_index, reference_index) and context["pos"][current].startswith("v"):
            return {"index": current, "word": context["words"][current]}
        current = context["dep_heads"][current]
    return None


# --------------------------------------------------------------------------
# 五、OpenHowNet 参照物语义闸门
# --------------------------------------------------------------------------


class HowNetUnavailable(RuntimeError):
    """OpenHowNet 数据不可用；属于工程故障，必须中止本轮而不是静默丢弃语料。"""


class PhysicalGate:
    """判断一个词是否至少有一个义项的义原表示「实际物体」。

    初始化与查询都只读本地 ~/.openhownet；没有网络请求。构造时显式校验数据已加载，
    因为 OpenHowNet 2.0 在数据缺失时只打印 FileNotFoundError 而不抛异常——不校验就会
    把整批参照物静默判成「没有义项」。
    """

    def __init__(self, hownet=None):
        if hownet is None:
            try:
                import OpenHowNet
            except Exception as exc:  # noqa: BLE001 —— 缺依赖是环境问题，必须让调用方看到
                raise HowNetUnavailable(f"无法导入 OpenHowNet：{exc!r}") from exc
            try:
                hownet = OpenHowNet.HowNetDict()
            except Exception as exc:  # noqa: BLE001
                raise HowNetUnavailable(f"OpenHowNet 初始化失败：{exc!r}") from exc
        if not getattr(hownet, "sense_dic", None) or not getattr(hownet, "zh_map", None):
            raise HowNetUnavailable(
                "OpenHowNet 数据未加载（~/.openhownet/resources 不完整）；"
                "请先运行 OpenHowNet.download() 或补齐数据"
            )
        self.hownet = hownet
        self._word_cache: dict[tuple[str, frozenset[str]], dict[str, Any]] = {}
        self._ancestor_cache: dict[tuple[str, frozenset[str]], list[str] | None] = {}

    def _path_to_root(self, sememe, roots: frozenset[str]) -> list[str] | None:
        """从一个义原沿 hypernym 向上找 ``roots`` 之一，返回路径（含起点与根）"""
        start = sememe.en_zh
        cache_key = (start, roots)
        cached = self._ancestor_cache.get(cache_key, ...)
        if cached is not ...:
            return cached
        queue: list[tuple[Any, list[str]]] = [(sememe, [])]
        seen: set[str] = set()
        while queue:
            current, path = queue.pop(0)
            key = current.en_zh
            if key in seen:
                continue
            seen.add(key)
            current_path = path + [key]
            if key in roots:
                self._ancestor_cache[cache_key] = current_path
                return current_path
            for parent in current.related_sememes.get("hypernym", []):
                queue.append((parent, current_path))
        self._ancestor_cache[cache_key] = None
        return None

    def _check(self, word: Any, roots: frozenset[str], prefix: str) -> dict[str, Any]:
        """按给定根义原检查一个词；失败原因用 ``<prefix>_no_sense`` / ``<prefix>_not_physical``。"""
        text = str(word or "").strip()
        cache_key = (text, roots)
        cached = self._word_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        if not text:
            result = {
                "ok": False,
                "reason": f"{prefix}_no_sense",
                "sense_id": None,
                "sememe": None,
                "path": [],
            }
            self._word_cache[cache_key] = result
            return dict(result)

        try:
            senses = sorted(self.hownet.get_sense(text, language="zh", strict=True), key=lambda s: s.No)
            if not senses:
                result = {
                    "ok": False,
                    "reason": f"{prefix}_no_sense",
                    "sense_id": None,
                    "sememe": None,
                    "path": [],
                }
                self._word_cache[cache_key] = result
                return dict(result)

            for sense in senses:
                for sememe in sense.sememes:
                    path = self._path_to_root(sememe, roots)
                    if path:
                        result = {
                            "ok": True,
                            "reason": "ok",
                            "sense_id": sense.No,
                            "sememe": sememe.en_zh,
                            "path": path,
                        }
                        self._word_cache[cache_key] = result
                        return dict(result)
        except HowNetUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 —— 词典数据异常是工程故障，不能当作「没有义项」
            raise HowNetUnavailable(f"检查 {prefix} 词 {text!r} 时 OpenHowNet 数据异常：{exc!r}") from exc

        result = {
            "ok": False,
            "reason": f"{prefix}_not_physical",
            "sense_id": None,
            "sememe": None,
            "path": [],
        }
        self._word_cache[cache_key] = result
        return dict(result)

    def check(self, word: Any) -> dict[str, Any]:
        """参照物检查：`physical|物质` 或 `place|地方` / `InstitutePlace|场所` 均通过。"""
        return self._check(word, REFERENCE_SEMEME_ROOTS, "reference")

    def check_located(self, word: Any) -> dict[str, Any]:
        """目标物检查：只认 `physical|物质` 子树，不沿用参照物的地点类放宽。"""
        return self._check(word, TARGET_SEMEME_ROOTS, "target")


# --------------------------------------------------------------------------
# 六、单候选校验与批量落盘
# --------------------------------------------------------------------------


def check_target(
    target: dict[str, Any],
    context: dict[str, Any],
    exclude: set[int] | None = None,
    *,
    gate: PhysicalGate | None = None,
) -> dict[str, Any]:
    """整理一个候选方位词，并用 OpenHowNet 检查参照物语义。

    keep:
        True  参照物和目标物都通过 OpenHowNet 的 physical 检查
        False 结构缺失（没有参照物 / 参照物不是空间角色 / 找不到目标物）、参照物不合格，
              或目标物没有 `physical|物质` 义原
        None  工程异常，调用方应保留记录待复核

    目标物是结构必需项：参照物即使通过 OpenHowNet 检查，只要同一 SDP 中心语下找不到
    可用的目标物，该方位词位置仍会被剔除。目标物还必须至少有一个义项的义原沿 hypernym
    到达 `physical|物质`；参照物的 `place|地方` / `InstitutePlace|场所` 放宽不适用于目标物。
    动词不是必需项，找不到时允许为 null。

    Returns:
        dict: 可 JSON 序列化的判定明细，含
            index/word/reference/target/verb/hownet/target_hownet/keep/reason/detail。
    """
    exclude = set(exclude or ())
    index = target.get("index")
    word = target.get("word")
    size = len(context["words"])
    base = {
        "index": index,
        "word": word,
        "keep": None,
        "reference": None,
        "target": None,
        "verb": None,
        "hownet": None,
        "target_hownet": None,
    }
    if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < size):
        return base | {"reason": "error", "detail": f"目标词下标非法：{index!r}"}

    candidates = _reference_candidates(index, context)
    spatial = [(candidate, priority) for candidate, priority in candidates if _is_spatial_reference(candidate, context)]
    if not spatial:
        reason = "reference_not_loc" if candidates else "no_reference"
        if reason == "reference_not_loc":
            detail = "参照物候选没有 SDP LOC/DATV/FEAT（多为时间/范围成分）：" + "、".join(
                f"{context['words'][candidate]}/{context['sdp_labels'][candidate]}"
                for candidate, _ in candidates
            )
        else:
            detail = "方位词没有名词性定语/依附名词"
        return base | {"keep": False, "reason": reason, "detail": detail}

    fallback: dict[str, Any] | None = None
    target_fallback: dict[str, Any] | None = None
    failures: list[tuple[dict[str, Any], Any, str]] = []
    for reference_index, _priority in spatial:
        reference = {
            "index": reference_index,
            "word": context["words"][reference_index],
            "sdp": context["sdp_labels"][reference_index],
        }
        semantic = gate.check(reference["word"]) if gate is not None else {"ok": True, "reason": "ok"}
        if not semantic.get("ok"):
            failures.append((reference, semantic, str(semantic.get("reason") or "reference_not_physical")))
            continue

        predicate_index = _resolve_predicate(reference_index, index, context)
        located_candidates = (
            _located_candidates(predicate_index, index, reference_index, context, exclude)
            if predicate_index is not None
            else []
        )
        verb = _verb_for(predicate_index, index, reference_index, context)
        if located_candidates:
            # 同一角色可能有并列目标物；逐个检查，取第一个通过 physical 的。
            for located in located_candidates:
                target_semantic = (
                    gate.check_located(located["word"])
                    if gate is not None
                    else {"ok": True, "reason": "ok", "sense_id": None, "sememe": None, "path": []}
                )
                if target_semantic.get("ok"):
                    return base | {
                        "keep": True,
                        "reference": reference,
                        "target": located,
                        "verb": verb,
                        "hownet": semantic,
                        "target_hownet": target_semantic,
                        "reason": "ok",
                        "detail": (
                            f"参照物「{reference['word']}」的义原 {semantic.get('sememe')} 属于实际物体；"
                            f"目标物「{located['word']}」以 {located['role']} 角色出现，"
                            f"其义原 {target_semantic.get('sememe')} 属于 physical|物质"
                        ),
                    }
                # 目标物找到了但语义不合格：先记下，继续试并列目标物/其他参照物候选。
                if target_fallback is None:
                    target_reason = str(target_semantic.get("reason") or "target_not_physical")
                    target_fallback = base | {
                        "keep": False,
                        "reference": reference,
                        "target": located,
                        "verb": verb,
                        "hownet": semantic,
                        "target_hownet": target_semantic,
                        "reason": target_reason,
                        "detail": (
                            f"目标物「{located['word']}」在 HowNet 中没有可用义项"
                            if target_reason == "target_no_sense"
                            else f"目标物「{located['word']}」没有义项的义原沿 hypernym 到达 "
                            "physical|物质（参照物的地点类放宽不适用于目标物）"
                        ),
                    }
            continue

        # 参照物合格但找不到目标物：目标物是结构必需项，该候选位置要剔除。
        # 先记下这个更具体的失败原因，继续试其他参照物候选；若另有候选两端齐全，仍优先保留。
        if fallback is None:
            fallback = base | {
                "keep": False,
                "reference": reference,
                "target": None,
                "verb": verb,
                "hownet": semantic,
                "reason": "no_located",
                "detail": (
                    f"参照物「{reference['word']}」属于实际物体，但同一 SDP 中心语下"
                    "找不到可用的目标物（EXP/PAT/AGT，或该谓词允许的 CONT/SELF/REL）"
                ),
            }

    # 目标物“找到了但语义不合格”比“根本没找到”更具体，优先作为失败原因。
    if target_fallback is not None:
        return target_fallback
    if fallback is not None:
        return fallback
    failures.sort(key=lambda item: 0 if item[2] == "reference_not_physical" else 1)
    reference, semantic, reason = failures[0]
    detail = (
        "参照物在 HowNet 中没有义项的义原属于实际物体"
        if reason == "reference_not_physical"
        else "参照物在 HowNet 中没有可用义项"
    )
    return base | {
        "keep": False,
        "reference": reference,
        "hownet": semantic,
        "reason": reason,
        "detail": detail,
    }


def apply_grounding(
    record: dict[str, Any],
    vocab: set[str],
    tokens: list[str] | None = None,
    pos: list[str] | None = None,
    dep: dict | None = None,
    sdp: dict | None = None,
    *,
    gate: PhysicalGate | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """对一条规则命中记录做逐候选词整理与参照物语义闸门。

    Args:
        record: 已带 sentence/filter_methods/match_tree 的记录（match_tree 是 simplify_tree 的字典形状）
        vocab: 本规则族的方位词词表（word_sampler.derive_vocab）
        tokens/pos/dep/sdp: 该句已预填充的 LTP 结果
        gate: OpenHowNet 闸门；None 时只做结构整理（测试/降级用）

    Returns:
        (保留的记录或 None, 统计块)。返回 None 表示所有候选词都不合格，整条记录不进入下一步。
        ``record["localization_grounding"]`` 会写入每个候选词的判定明细，供复核。
    """
    import word_sampler

    stats = empty_stats()
    stats["records"] = 1
    try:
        targets = word_sampler.target_nodes(record, vocab)
    except Exception as exc:  # noqa: BLE001 —— 单条记录形状异常只降级，不带走整批
        stats["errors"] = 1
        stats["kept_records"] = 1
        record["localization_grounding"] = {
            "source": "error",
            "valid": None,
            "error": repr(exc),
            "candidates": [],
        }
        return record, stats

    if not targets:
        stats["no_target"] = 1
        stats["kept_records"] = 1
        record["localization_grounding"] = {"source": "no_target", "valid": None, "candidates": []}
        return record, stats

    if tokens is None:
        tokens = record.get("sentence") if isinstance(record.get("sentence"), list) else None
    try:
        context = build_context(
            list(tokens or []),
            list(pos or []),
            dep or {},
            sdp or {},
        )
    except Exception as exc:  # noqa: BLE001
        stats["errors"] = 1
        stats["kept_records"] = 1
        record["localization_grounding"] = {
            "source": "error",
            "valid": None,
            "error": repr(exc),
            "candidates": [],
        }
        return record, stats

    target_indices = {
        target["index"]
        for target in targets
        if isinstance(target, dict) and isinstance(target.get("index"), int)
    }
    verdicts: list[dict[str, Any]] = []
    for target in targets:
        try:
            verdict = check_target(
                target,
                context,
                target_indices - {target.get("index")},
                gate=gate,
            )
        except HowNetUnavailable:
            # 词典/查询故障是全局工程问题：中止本轮，不能把它降级成「这一条判不出来」。
            raise
        except Exception as exc:  # noqa: BLE001 —— 工程异常 fail-open
            verdict = {
                "index": target.get("index"),
                "word": target.get("word"),
                "keep": None,
                "reference": None,
                "target": None,
                "verb": None,
                "hownet": None,
                "target_hownet": None,
                "reason": "error",
                "detail": repr(exc),
            }
        verdicts.append(verdict)

    valid = {v["index"] for v in verdicts if v.get("keep") is True}
    invalid = {v["index"] for v in verdicts if v.get("keep") is False}
    unknown = {v["index"] for v in verdicts if v.get("keep") is None}
    stats["candidates"] = len(verdicts)
    stats["kept_candidates"] = len(valid)
    stats["dropped_candidates"] = len(invalid)
    stats["errors"] = len(unknown)
    for verdict in verdicts:
        reason = str(verdict.get("reason") or "error")
        stats["reasons"][reason] = int(stats["reasons"].get(reason, 0)) + 1

    if invalid and not valid and not unknown:
        stats["dropped_records"] = 1
        return None, stats

    if invalid:
        # unknown（工程异常）按 fail-open 保留，裁剪时也要把它当作要保留的下标，
        # 否则删掉一个 false 父节点会把它下面的 unknown 子节点一起带走。
        record["match_tree"] = prune_tree(record.get("match_tree") or [], invalid, valid | unknown)
    stats["kept_records"] = 1
    record["localization_grounding"] = {
        "source": "hownet",
        "valid": True if valid else None if unknown else False,
        "candidates": verdicts,
    }
    return record, stats


def attach_analysis(
    record: dict[str, Any],
    sentence: str,
    cache: Any,
    *,
    kind: str | None,
) -> dict[str, Any]:
    """把该句的 POS/DEP/SDP 附到记录上，供主进程做整理与语义闸门。

    只在方位词语料上附加。这里不导入 OpenHowNet，因此可以安全地在多进程 worker 中调用。
    """
    if kind != "方位词" or not isinstance(sentence, str) or not sentence:
        return record
    try:
        import config

        record["_locative_analysis"] = {
            "pos": list(cache.get_task_value(sentence, config.POS)),
            "dep": cache.get_task_value(sentence, config.DEP),
            "sdp": cache.get_task_value(sentence, config.SDP),
        }
    except Exception as exc:  # noqa: BLE001 —— 主进程按 error 记录并 fail-open
        record["_locative_analysis"] = {"error": repr(exc)}
    return record


def prepare_families(
    filter_config_paths: list[str],
    classify_family,
    *,
    kind: str | None = None,
) -> tuple[dict[str, dict[str, Any]], "PhysicalGate | None"]:
    """为每个规则族准备词类型/方位词表，并在需要时初始化一次 OpenHowNet。

    只给方位词族建闸门：趋向动词语料完全绕过，既不读词表也不加载 HowNet。
    ``kind`` 是命令行 ``--screen-kind`` 的显式覆盖，优先级高于按文件名推断。
    """
    from pathlib import Path

    metas: dict[str, dict[str, Any]] = {}
    need_gate = False
    for path in filter_config_paths:
        family = Path(path).stem
        resolved = kind or classify_family(family)
        vocab = None
        if resolved == "方位词":
            import word_sampler

            vocab = set(word_sampler.derive_vocab(str(path), None)[0])
            need_gate = True
        metas[family] = {"kind": resolved, "vocab": vocab}
    return metas, (PhysicalGate() if need_gate else None)


def gate_rows(
    rows: list[dict[str, Any]],
    vocab: set[str],
    *,
    gate: PhysicalGate | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对一批记录做语义闸门；没有结构分析块的记录原样保留（fail-open）。"""
    stats = empty_stats()
    kept: list[dict[str, Any]] = []
    for row in rows:
        analysis = row.pop("_locative_analysis", None)
        if not isinstance(analysis, dict):
            kept.append(row)
            continue
        if analysis.get("error"):
            stats["records"] = int(stats.get("records", 0)) + 1
            stats["errors"] = int(stats.get("errors", 0)) + 1
            stats["kept_records"] = int(stats.get("kept_records", 0)) + 1
            row["localization_grounding"] = {
                "source": "error",
                "valid": None,
                "error": analysis["error"],
                "candidates": [],
            }
            kept.append(row)
            continue
        result, delta = apply_grounding(
            row,
            vocab,
            pos=analysis.get("pos"),
            dep=analysis.get("dep"),
            sdp=analysis.get("sdp"),
            gate=gate,
        )
        merge_stats(stats, delta)
        if result is not None:
            kept.append(result)
    return kept, stats


def prune_candidates(record: dict[str, Any], false_indices: set[int], true_indices: set[int]) -> None:
    """LLM 筛选裁掉某个词后，同步裁掉它的定位整理结果。"""
    block = record.get("localization_grounding")
    if not isinstance(block, dict):
        return
    candidates = block.get("candidates")
    if not isinstance(candidates, list):
        return
    kept = [
        candidate
        for candidate in candidates
        if not (candidate.get("index") in false_indices and candidate.get("index") not in true_indices)
    ]
    block["candidates"] = kept
    keeps = [candidate.get("keep") for candidate in kept]
    block["valid"] = True if True in keeps else None if None in keeps else False


def format_stats(stats: dict[str, Any]) -> str:
    """一行日志：记录/候选词/各剔除原因，便于跑批后对账"""
    reasons = "、".join(
        f"{reason} {count}" for reason, count in sorted((stats.get("reasons") or {}).items())
    )
    text = (
        f"记录 {stats.get('records', 0)}"
        f"（保留 {stats.get('kept_records', 0)} / 剔除 {stats.get('dropped_records', 0)}），"
        f"候选词 {stats.get('candidates', 0)}"
        f"（保留 {stats.get('kept_candidates', 0)} / 剔除 {stats.get('dropped_candidates', 0)}），"
        f"no_target {stats.get('no_target', 0)}，error {stats.get('errors', 0)}"
    )
    return text + (f"；{reasons}" if reasons else "")
