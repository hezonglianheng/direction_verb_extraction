# encoding: utf8
"""空间标注的纯本地解析：不读文件、不建 agent、不调用模型，也不改写输入。

pair 与 all_in_one 共用这层；表由调用方在构造期加载，查表结论始终先于模型结果。
"""

from annotate import spatial_tools as st


def schema_from_table(table: dict, features: dict, focus: dict | None) -> dict | None:
    """按本侧关注词收窄后查表，含原有的同词兜底条目；未命中返回 None。"""
    name, detail = st.lookup_schema(features or {}, table, focus)
    if not name:
        return None
    return {"name": name, "source": "table",
            "reason": f"查表命中 {name}（{st.format_match(detail.get('match'))}）"}


def schema_miss_note(word: str, class_word: str | None, candidates: dict) -> str:
    """原 pair 的未命中措辞；同时用于消息和 no_llm 结果，不改变命名空间。"""
    if candidates:
        owner = f"（属「{class_word}」这一类）" if class_word and class_word != word else ""
        return f"关注词「{word}」{owner}的 {len(candidates)} 条图式都没命中该句特征"
    if class_word and class_word != word:
        return f"关注词「{word}」归入「{class_word}」这一类，但这一类下还没有编号图式"
    if word:
        return f"表里还没有关注词「{word}」的图式"
    return "表为空（且本句没有关注词可用以收窄）"


def schema_without_model(table: dict, focus: dict | None) -> dict:
    """查表未命中且不调用模型时的标准未知结果。"""
    miss = schema_miss_note((focus or {}).get("word") or "", st.schema_group_key(table, focus),
                            st.schema_candidates(table, focus))
    return {"name": None, "source": "no_llm", "reason": f"{miss}，且未调用 LLM（--no-llm）"}


def schema_from_model(table: dict, focus: dict | None, verdict: dict) -> dict:
    """验证本侧候选；无候选时按所属组名（不是别名词形）验证自由命名。"""
    raw = verdict.get("schema_name")
    chosen = raw.strip() if isinstance(raw, str) else ""
    candidates = st.schema_candidates(table, focus)
    if candidates:
        valid = chosen in candidates
        why = f"不在本次的候选名单里（{'、'.join(candidates)}）"
    else:
        prefix = st.schema_group_key(table, focus) or ((focus or {}).get("word") or "")
        valid = st.is_schema_name(chosen, prefix)
        why = f"不符合「{prefix or '关注词'} + 编号」的格式"
    reason = verdict.get("reason") or ""
    if not valid:
        return {"name": None, "source": "llm_invalid", "reason": f"LLM 给出的名字 {raw!r} {why}；{reason}"}
    return {"name": chosen, "source": "llm" if candidates else "llm_free", "reason": reason}


def _judgement_context(table: dict, schema_a: dict, schema_b: dict) -> tuple:
    same, detail = st.lookup_judgement((schema_a or {}).get("name"), (schema_b or {}).get("name"), table)
    pair = f"{detail['label_a']} × {detail['label_b']}"
    lookup = "llm" if detail["source"] == "defer" else detail["source"]
    return same, detail, pair, lookup


def judgement_situation(table: dict, schema_a: dict, schema_b: dict) -> str:
    """pair user message 中原有的判定表情况；只用于没有直接结论的格子。"""
    _, _, pair, lookup = _judgement_context(table, schema_a, schema_b)
    return {
        "llm": f"判定表标了 {st.JUDGE_CELL_LLM}（交大模型做进一步判断）",
        "miss": "判定表未收录这两个 schema",
        "illegal": f"判定表里 {pair} 这一格留空或符号认不出（非法）",
    }[lookup]


def judgement_from_table(table: dict, schema_a: dict, schema_b: dict) -> dict:
    """√/× 直接给结论；○、miss、illegal 返回带 lookup 的 no_llm 未知结果。"""
    same, detail, pair, lookup = _judgement_context(table, schema_a, schema_b)
    if same is not None:
        return {"same_meaning": same, "source": "table", "lookup": "same" if same else "different",
                "reason": f"查表命中：{pair} = {detail['cell']}（{'意义相同' if same else '意义不同'}）"}
    situation = judgement_situation(table, schema_a, schema_b)
    return {"same_meaning": None, "source": "no_llm", "lookup": lookup,
            "reason": f"查表没有直接结论（{situation}），且未调用 LLM（--no-llm）"}


def judgement_from_model(local: dict, verdict: dict | None, *, error: str | None = None) -> dict:
    """本地结论永远优先；没有调用时保持未知，失败时保留查表分类。"""
    if local.get("source") == "table" or (verdict is None and error is None):
        return dict(local)
    if error is not None or not isinstance((verdict or {}).get("same_meaning"), bool):
        return {"same_meaning": None, "source": "error", "lookup": local.get("lookup"),
                "reason": error or "模型没有给出 same_meaning"}
    return {"same_meaning": verdict["same_meaning"], "source": "llm", "lookup": local.get("lookup"),
            "reason": verdict.get("reason") or ""}
