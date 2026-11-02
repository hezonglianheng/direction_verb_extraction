# encoding: utf8
"""提取/替换与标注之间的句对契约，不依赖模型。"""

import sentence_screening as screening

PAIR_SCHEMA_VERSION = "1.0"
PAIR_CORPORA = {"方位词": "location_pairs.jsonl", "趋向动词": "direction_pairs.jsonl"}


def pair_status(a: dict, b: dict) -> str:
    statuses = {screening.verdict_status(a), screening.verdict_status(b)}
    for status in ("error", "limit", "no_llm", "false"):
        if status in statuses:
            return status
    return "true"


def is_prepared_pair(record: dict) -> bool:
    """不信任行级 true 标记：复核两侧逐位置判定以及切片/关注词的对应关系。"""
    if not isinstance(record, dict) or record.get("pair_schema_version") != PAIR_SCHEMA_VERSION:
        return False
    if record.get("corpus_type") not in PAIR_CORPORA or not isinstance(record.get("candidate"), dict):
        return False
    checks = record.get("pair_screening")
    pair = record.get("pair")
    if not isinstance(checks, dict) or checks.get("status") != "true" or not isinstance(pair, dict):
        return False
    for side, focus_key in (("a", "focus_word"), ("b", "focus_word_b")):
        check, data, focus = checks.get(side), pair.get(side), record.get(focus_key)
        if not all(isinstance(value, dict) for value in (check, data, focus)):
            return False
        if not screening.verdict_passes(check):
            return False
        tokens, index = data.get("slice"), focus.get("index")
        if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
            return False
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(tokens):
            return False
        if tokens[index] != focus.get("word") or check.get("index") != index or check.get("word") != focus.get("word"):
            return False
        if data.get("sentence") != "".join(tokens):
            return False
    a, b = pair["a"]["slice"], pair["b"]["slice"]
    index = record["focus_word"]["index"]
    if record["focus_word_b"]["index"] != index or len(a) != len(b):
        return False
    candidate = record["candidate"]
    return (record.get("sentence") == a and candidate.get("focus_index") == index
            and candidate.get("word") == a[index] and candidate.get("replacement") == b[index]
            and a[index] != b[index] and a[:index] == b[:index] and a[index + 1:] == b[index + 1:])


def read_prepared_pairs(path, limit=None, offset: int = 0) -> list[dict]:
    import spatial_tools

    return spatial_tools.read_corpus(path, limit=limit, offset=offset, prepared_only=True)
