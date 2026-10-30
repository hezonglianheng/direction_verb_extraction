# encoding: utf8

"""趋向动词结构整理、目标物/参照物/动词三元组与 HowNet 语义闸门。

规则筛选只回答「这个句子像不像趋向动词语料」，并不保证句子里真的出现了移动物体、
位移参照，也不保证依附动词是表示移动的动词。本模块在规则命中之后、LLM 筛选与落盘之前
补三步：

1. **三元组整理**（纯规则，不调模型）
   目标物    发生移动的物体。沿趋向动词依附的事件中心找 SDP 的 PAT/EXP/AGT/CONT 子节点。
              角色优先顺序只看命中的规则名：cause_/accompany_ 优先受事
              （他把书拿进教室 → 书），manner_ 及其他优先施事（他跑进教室 → 他）。
              本版不再用动词义原改选目标物。
              连动/兼语里事件中心找不到参与者时，沿 eSUCC 等事件链回退到前一个动词的参与者。
   动词      趋向动词所依附的动词；趋向动词独立作谓语时置 null（这是合法值，不是识别失败）。
              只沿 SDP/DEP 向上找动词，跳过词表里的其他趋向动词（走进来 → 动词是「走」）。
   参照物    位移的参照。优先取趋向动词的 DEP 宾语（进教室 → 教室），其次取趋向动词/事件中心
              SDP 的 LOC/DATV/FEAT 名词（从屋里跑出来 → 屋里）。找不到时置 null：本任务没有把
              参照物列为必需项。

2. **HowNet 语义闸门（只筛目标物）**
   目标物必须至少有一个义项的义原沿 hypernym 到达 ``config.DIRECTION_TARGET_SEMEME_ROOTS``
   （默认只有 physical|物质）。
   动词和参照物仍会抽取并写入三元组，但**不参与语义筛选**：动词为空合法，参照物缺失也合法。
   根义原清单、含义与修改方式见 config.py 的「趋向动词三元组的目标物义原闸门」一节。
   HowNet 没有目标物词条或没有允许义原时该候选被剔除；词典加载/查询故障按工程问题中止本轮。

3. **判定明细落盘**
   ``record["direction_grounding"]`` 的 candidates 每项含 index/word/keep/reason/detail、
   target/reference/verb、target_hownet。与方位词链路一样，keep=False 也保留证据；
   keep=None 表示工程异常（fail-open，待复核），不能当成「已通过硬筛」。

异常口径与 localization_grounding 一致：单条记录形状异常 fail-open；HowNet 全局不可用抛
``HowNetUnavailable`` 中止本轮，不能把整批静默判成「没有义项」。
"""

from typing import Any

import config
import localization_grounding as lg
from pipeline_utils import prune_tree

# --------------------------------------------------------------------------
# 一、结构与角色常量
# --------------------------------------------------------------------------

NOMINAL_POS = lg.LOCATED_POS
"""可当目标物/参照物的词性（与方位词链路同一套）。"""

TARGET_ROLES = ("PAT", "EXP", "AGT", "CONT")
"""目标物可承担的基础 SDP 角色。具体优先顺序见 _role_order。"""

REFERENCE_SDP_LABELS = frozenset({"LOC", "DATV", "FEAT"})
"""参照物必须承担的 SDP 空间角色；TIME/rTIME/SCO 不算位移参照。"""

EVENT_LINK_LABELS = frozenset({"eSUCC", "eCOO", "ePREP", "ePROG"})
"""连动/顺承等事件链标记：事件中心没有参与者时，沿这些标记回退到前一个动词。"""

DEP_OBJECT_LABELS = frozenset({"POB"})
"""趋向动词的 DEP 宾语标签（「进教室」的教室）。"""

DEP_TARGET_ROLES = {"SBV": "AGT", "VOB": "PAT", "FOB": "PAT", "DBL": "PAT"}
"""SDP 没给出参与者时的 DEP 兜底角色。"""

SDP_COMPLEMENT_LABELS = frozenset({"mDEPD", "mRang", "dCONT"})
"""趋向动词依附前面的动词时使用的 SDP 标记。

mDEPD/mRang 是趋向补语；dCONT 是内容从句（「他喜欢进教室」的「进」依附「喜欢」）。
eSUCC/eCOO 是事件顺承/并列，不能当成依附关系，否则「他吃完饭进屋」的「进屋」会被
错判成依附「吃」。dCONT 必须收：否则「喜欢/想/决定 + 进教室」无法记录真实依附动词；
本版不据此删语料，但三元组应如实写 verb=喜欢。
"""

DEP_COMPLEMENT_LABELS = frozenset({"CMP"})
"""趋向补语在 DEP 里的标记；SDP 给不出时用它兜底。"""

STAT_KEYS = lg.STAT_KEYS
"""沿用方位词链路的统计键，便于日志与入口共用一套渲染。"""


def empty_stats() -> dict[str, Any]:
    """一份可 JSON 序列化、可跨批/跨进程相加的统计块"""
    return lg.empty_stats()


def merge_stats(total: dict[str, Any], delta: dict[str, Any] | None) -> dict[str, Any]:
    """把一批/一个进程的统计累加进 ``total``"""
    return lg.merge_stats(total, delta)


def format_stats(stats: dict[str, Any]) -> str:
    """一行日志：记录/候选词/各剔除原因"""
    return lg.format_stats(stats)


def policy_snapshot() -> dict:
    """当前义原口径快照（写入语料与筛选日志）"""
    return config.direction_sememe_policy()


# --------------------------------------------------------------------------
# 二、OpenHowNet 义原闸门
# --------------------------------------------------------------------------


class DirectionGate(lg.PhysicalGate):
    """趋向动词的语义闸门：只检查目标物的 physical 义原。

    复用 PhysicalGate 的 OpenHowNet 加载、hypernym 追溯与缓存；构造时也可以传入
    已经加载好的 HowNetDict，避免与方位词闸门各加载一份词典。动词与参照物不参与
    语义筛选，因此本类不提供 check_verb。
    """

    def check_target(self, word: Any) -> dict[str, Any]:
        """目标物检查：只认 config.DIRECTION_TARGET_SEMEME_ROOTS（默认 physical|物质）"""
        return self._check(str(word or ""), config.DIRECTION_TARGET_SEMEME_ROOTS, "target")


# --------------------------------------------------------------------------
# 三、上下文与三元组整理
# --------------------------------------------------------------------------


def _is_verb(index: int, context: dict[str, Any]) -> bool:
    return 0 <= index < len(context["pos"]) and context["pos"][index].startswith("v")


def _is_nominal(index: int, context: dict[str, Any]) -> bool:
    return lg._is_nominal(index, context, NOMINAL_POS)


def _rule_kind(filter_methods: Any) -> str | None:
    """从命中的规则名判断这条语料属于哪一类（只用于目标物角色优先顺序）"""
    names = " ".join(str(item) for item in (filter_methods or ()))
    if "cause_" in names:
        return "cause"
    if "accompany_" in names:
        return "accompany"
    if "manner_" in names:
        return "manner"
    if "direction_verb_as_center" in names:
        return "direction_center"
    return None


def _role_order(rule_kind: str | None) -> tuple[str, ...]:
    """目标物角色优先顺序（只依据命中的规则名，不查动词义原）

    他移/伴随类优先 PAT（书、孩子），自移类优先 AGT（他、小鸟）；没有规则信息时
    按施事优先。CONT 只作为最后的兜底，避免把「墙上贴的海报」这类内容物错当移动主体。

    注意：动词义原筛选已移除。若连动句只命中 manner 规则（如「他抱着孩子走进屋」的
    LOC 挂在「走」上），这里会按 manner 优先 AGT，目标物取「他」；不会再借「抱」的
    HowNet 类别改选「孩子」。这是本版明确的行为口径。
    """
    if rule_kind in ("cause", "accompany"):
        return ("PAT", "EXP", "AGT", "CONT")
    return ("AGT", "EXP", "PAT", "CONT")


def _resolve_verb_and_center(
    direction_index: int,
    context: dict[str, Any],
    direction_vocab: set[str],
) -> tuple[int | None, int]:
    """返回 (趋向动词依附的动词下标, 事件中心下标)。

    只有明确的补语/从句关系（SDP 的 mDEPD/mRang/dCONT、DEP 的 CMP）才沿父节点找动词；
    eSUCC/eCOO 是事件顺承/并列，不能当成依附关系：
    「他吃完饭进屋」的「进」是独立谓语，verb 应为 None，而不是「吃」。

    复合趋向「他／进／屋／来」里「来」依附「进」、最终没有非趋向动词时，
    返回 (None, 最上层趋向谓语「进」)，让「来」共享「进」的目标物与参照物。

    Returns:
        (verb_index, event_center_index): verb_index 为 None 表示趋向动词独立作谓语；
            event_center_index 是找参与者/参照物时的事件中心，可能是上层趋向动词。
    """
    center = direction_index
    # SDP 语义依存优先；若它明确标成事件顺承/并列，直接认定独立谓语，不再用 DEP 反推。
    if context["sdp_labels"][direction_index] in EVENT_LINK_LABELS:
        return None, center
    for heads_key, labels_key, allowed in (
        ("sdp_heads", "sdp_labels", SDP_COMPLEMENT_LABELS),
        ("dep_heads", "dep_labels", DEP_COMPLEMENT_LABELS),
    ):
        current = direction_index
        seen = {direction_index}
        while context[labels_key][current] in allowed:
            parent = context[heads_key][current]
            if parent is None or parent in seen:
                break
            seen.add(parent)
            if not _is_verb(parent, context):
                break
            if context["words"][parent] in direction_vocab:
                center = parent
                current = parent
                continue
            return parent, parent
    return None, center


def _event_chain(
    start_index: int,
    context: dict[str, Any],
    direction_vocab: set[str],
) -> list[int]:
    """事件中心链：当前动词 + 沿 eSUCC/eCOO 等回退到的前一个动词。

    「他抱着孩子走进屋」里「走」没有自己的 AGT/PAT，参与者挂在「抱」上；
    回退一层才能拿到「孩子」。只沿动词事件链走，不跨普通修饰/从句。
    """
    chain: list[int] = []
    current: int | None = start_index
    seen: set[int] = set()
    while current is not None and current not in seen:
        seen.add(current)
        chain.append(current)
        label = context["sdp_labels"][current]
        parent = context["sdp_heads"][current]
        if (
            parent is not None
            and label in EVENT_LINK_LABELS
            and _is_verb(parent, context)
            and context["words"][parent] not in direction_vocab
        ):
            current = parent
            continue
        break
    return chain


def _target_candidates(
    chain: list[int],
    direction_index: int,
    context: dict[str, Any],
    role_order: tuple[str, ...],
    exclude: set[int],
) -> list[dict[str, Any]]:
    """沿事件链找目标物候选，按 (事件中心优先, 角色优先, 词序) 排列"""
    found: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(index: int, role: str, center: int) -> None:
        if index in seen or index in exclude or index == direction_index:
            return
        if not _is_nominal(index, context):
            return
        seen.add(index)
        found.append(
            {
                "index": index,
                "word": context["words"][index],
                "role": role,
                "center": center,
            }
        )

    for center in chain:
        children = context["sdp_children"].get(center, [])
        for role in role_order:
            for child, label in children:
                if label == role:
                    add(child, role, center)

    if found:
        return found

    # SDP 没给角色时用 DEP 兜底：主语当 AGT，宾语当 PAT。
    # 兜底也必须遵守 role_order，否则「他拿书进」会先收 SBV 他、把他移类目标物抽成施事。
    for center in chain:
        children = context["dep_children"].get(center, [])
        for role in role_order:
            for child, label in children:
                if DEP_TARGET_ROLES.get(label) == role:
                    add(child, role, center)
    return found


def _reference_candidates(
    direction_index: int,
    chain: list[int],
    context: dict[str, Any],
    exclude: set[int],
) -> list[dict[str, Any]]:
    """参照物候选：趋向动词的 DEP 宾语优先，其次 SDP 空间角色，最后事件链上的 LOC。

    优先 DEP 宾语是为了在「从屋里跑进教室」里取目标参照「教室」，而不是来源「屋里」：
    教室是「进」的 POB，屋里是介词「从」的 POB、只挂在「跑」的 LOC 上。
    """
    found: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(index: int) -> None:
        if index in seen or index in exclude or index == direction_index:
            return
        if not _is_nominal(index, context):
            return
        seen.add(index)
        found.append(
            {
                "index": index,
                "word": context["words"][index],
                "sdp": context["sdp_labels"][index],
            }
        )

    for child, label in context["dep_children"].get(direction_index, []):
        if label in DEP_OBJECT_LABELS:
            add(child)

    for child, label in context["sdp_children"].get(direction_index, []):
        if label in REFERENCE_SDP_LABELS:
            add(child)

    for center in chain:
        for child, label in context["sdp_children"].get(center, []):
            if label in REFERENCE_SDP_LABELS:
                add(child)

    for center in chain:
        for child, label in context["dep_children"].get(center, []):
            if label in DEP_OBJECT_LABELS:
                add(child)

    return found


def _verb_payload(index: int | None, context: dict[str, Any]) -> dict[str, Any] | None:
    if index is None:
        return None
    return {"index": index, "word": context["words"][index]}


def _target_roots_text() -> str:
    """把目标物允许的根义原渲染成日志文本，避免改配置后 detail 仍写死 physical|物质"""
    return "、".join(sorted(config.DIRECTION_TARGET_SEMEME_ROOTS))


def _target_failure_detail(word: str, semantic: dict[str, Any]) -> str:
    if semantic.get("reason") == "target_no_sense":
        return f"目标物「{word}」在 HowNet 中没有可用义项"
    return (
        f"目标物「{word}」没有义项的义原沿 hypernym 到达 {_target_roots_text()}"
        "（目标物必须是实际物体）"
    )


def check_target(
    target: dict[str, Any],
    context: dict[str, Any],
    exclude: set[int] | None = None,
    *,
    gate: DirectionGate | None = None,
    direction_vocab: set[str] | None = None,
    filter_methods: Any = (),
) -> dict[str, Any]:
    """整理一个候选趋向动词，并用 HowNet **只检查目标物**。

    keep:
        True  目标物通过 physical 检查
        False 找不到目标物，或目标物不是实际物体
        None  工程异常（调用方保留记录待复核）

    动词和参照物仍会抽取并写入三元组，但不参与语义筛选：动词为空合法，参照物缺失也合法。
    """
    exclude = set(exclude or ())
    index = target.get("index")
    word = target.get("word")
    size = len(context["words"])
    base = {
        "index": index,
        "word": word,
        "keep": None,
        "target": None,
        "reference": None,
        "verb": None,
        "target_hownet": None,
        "verb_optional": True,
        "rule_kind": _rule_kind(filter_methods),
    }
    if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < size):
        return base | {"reason": "error", "detail": f"目标词下标非法：{index!r}"}

    vocabulary = set(direction_vocab or ())
    verb_index, event_center = _resolve_verb_and_center(index, context, vocabulary)
    verb = _verb_payload(verb_index, context)
    chain = _event_chain(event_center, context, vocabulary)
    if not chain:
        chain = [event_center]

    role_order = _role_order(base["rule_kind"])
    candidates = _target_candidates(
        chain,
        index,
        context,
        role_order,
        exclude | {index},
    )
    if not candidates:
        return base | {
            "verb": verb,
            "keep": False,
            "reason": "no_target",
            "detail": (
                "事件中心（" + "、".join(context["words"][item] for item in chain)
                + "）下找不到 PAT/EXP/AGT/CONT 名词性目标物"
            ),
        }

    first_target_failure: dict[str, Any] | None = None
    for candidate in candidates:
        if gate is None:
            target_semantic = {"ok": True, "reason": "ok", "sememe": None, "path": []}
        else:
            target_semantic = gate.check_target(candidate["word"])
        if not target_semantic.get("ok"):
            if first_target_failure is None:
                first_target_failure = base | {
                    "target": candidate,
                    "verb": verb,
                    "target_hownet": target_semantic,
                    "keep": False,
                    "reason": str(target_semantic.get("reason") or "target_not_physical"),
                    "detail": _target_failure_detail(candidate["word"], target_semantic),
                }
            continue

        references = _reference_candidates(
            index,
            chain,
            context,
            exclude | {index, candidate["index"]},
        )
        reference = references[0] if references else None
        if verb is None:
            verb_text = "趋向动词独立作谓语，未依附其他动词（允许为空）"
        else:
            verb_text = f"动词「{verb['word']}」已抽取（本版不校验动词义原）"
        reference_text = f"；参照物「{reference['word']}」" if reference else "；句中未识别到参照物"
        return base | {
            "keep": True,
            "target": candidate,
            "reference": reference,
            "verb": verb,
            "target_hownet": target_semantic,
            "reason": "ok" if verb is not None else "ok_no_verb",
            "detail": (
                f"目标物「{candidate['word']}」以 {candidate['role']} 角色出现，"
                f"义原 {target_semantic.get('sememe')} 沿 hypernym 到达 {_target_roots_text()}；"
                f"{verb_text}{reference_text}"
            ),
        }

    if first_target_failure is not None:
        return first_target_failure
    return base | {
        "verb": verb,
        "keep": False,
        "reason": "no_target",
        "detail": "事件链上找不到可用的名词性目标物",
    }


# --------------------------------------------------------------------------
# 四、单条记录与批量落盘
# --------------------------------------------------------------------------


def apply_grounding(
    record: dict[str, Any],
    vocab: set[str],
    tokens: list[str] | None = None,
    pos: list[str] | None = None,
    dep: dict | None = None,
    sdp: dict | None = None,
    *,
    gate: DirectionGate | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """对一条规则命中记录做逐候选词三元组整理与语义闸门。

    Args:
        record: 已带 sentence/filter_methods/match_tree 的记录
        vocab: 趋向动词词表（word_sampler.derive_vocab(..., ("direction_verb_",))）
        tokens/pos/dep/sdp: 该句已预填充的 LTP 结果
        gate: OpenHowNet 闸门；None 时只做结构整理（测试/降级用）

    Returns:
        (保留的记录或 None, 统计块)。所有候选都不合格时返回 None；
        ``record["direction_grounding"]`` 写入每个候选词的判定明细。
    """
    import word_sampler

    stats = empty_stats()
    stats["records"] = 1
    try:
        targets = word_sampler.target_nodes(record, vocab)
    except Exception as exc:  # noqa: BLE001 —— 单条记录形状异常只降级
        stats["errors"] = 1
        stats["kept_records"] = 1
        record["direction_grounding"] = {
            "source": "error",
            "valid": None,
            "error": repr(exc),
            "policy": policy_snapshot(),
            "candidates": [],
        }
        return record, stats

    if not targets:
        stats["no_target"] = 1
        stats["kept_records"] = 1
        record["direction_grounding"] = {
            "source": "no_target",
            "valid": None,
            "policy": policy_snapshot(),
            "candidates": [],
        }
        return record, stats

    if tokens is None:
        tokens = record.get("sentence") if isinstance(record.get("sentence"), list) else None
    try:
        context = lg.build_context(
            list(tokens or []),
            list(pos or []),
            dep or {},
            sdp or {},
        )
    except Exception as exc:  # noqa: BLE001
        stats["errors"] = 1
        stats["kept_records"] = 1
        record["direction_grounding"] = {
            "source": "error",
            "valid": None,
            "error": repr(exc),
            "policy": policy_snapshot(),
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
                direction_vocab=vocab,
                filter_methods=record.get("filter_methods") or (),
            )
        except lg.HowNetUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 —— 工程异常 fail-open
            verdict = {
                "index": target.get("index"),
                "word": target.get("word"),
                "keep": None,
                "target": None,
                "reference": None,
                "verb": None,
                "target_hownet": None,
                "verb_optional": True,
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
        # unknown（工程异常）按 fail-open 保留，裁剪时也要当作要保留的下标。
        record["match_tree"] = prune_tree(record.get("match_tree") or [], invalid, valid | unknown)
    stats["kept_records"] = 1
    record["direction_grounding"] = {
        "source": "hownet",
        "valid": True if valid else None if unknown else False,
        "policy": policy_snapshot(),
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
    """把该句的 POS/DEP/SDP 附到记录上，供主进程做三元组整理与语义闸门。

    只在趋向动词语料上附加；这里不导入 OpenHowNet，可以安全地在多进程 worker 中调用。
    """
    if kind != "趋向动词" or not isinstance(sentence, str) or not sentence:
        return record
    try:
        record["_direction_analysis"] = {
            "pos": list(cache.get_task_value(sentence, config.POS)),
            "dep": cache.get_task_value(sentence, config.DEP),
            "sdp": cache.get_task_value(sentence, config.SDP),
        }
    except Exception as exc:  # noqa: BLE001 —— 主进程按 error 记录并 fail-open
        record["_direction_analysis"] = {"error": repr(exc)}
    return record


def gate_rows(
    rows: list[dict[str, Any]],
    vocab: set[str],
    *,
    gate: DirectionGate | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对一批记录做语义闸门；没有结构分析块的记录原样保留（fail-open）。"""
    stats = empty_stats()
    kept: list[dict[str, Any]] = []
    for row in rows:
        analysis = row.pop("_direction_analysis", None)
        if not isinstance(analysis, dict):
            kept.append(row)
            continue
        if analysis.get("error"):
            stats["records"] = int(stats.get("records", 0)) + 1
            stats["errors"] = int(stats.get("errors", 0)) + 1
            stats["kept_records"] = int(stats.get("kept_records", 0)) + 1
            row["direction_grounding"] = {
                "source": "error",
                "valid": None,
                "error": analysis["error"],
                "policy": policy_snapshot(),
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
    """LLM 筛选裁掉某个词后，同步裁掉它的三元组整理结果。"""
    block = record.get("direction_grounding")
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
