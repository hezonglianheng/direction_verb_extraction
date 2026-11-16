# encoding: utf8
"""annotate 阶段的句对限额筛选：顺序 / 随机 / 按关注词对比例 / 平行对齐。

X 的口径是**送入标注的合格句对记录数**（`pair_schema_version` 存在、`replace` 的
`is_prepared_pair` 通过），不是原句数、句子位置或候选数。一次标注一行就是一个合格句对。

四种模式（``--limit-mode``）：

1. ``sequential``：按文件顺序取前 X 条合格句对，读够即停，单遍流式读取。
2. ``random``：从全部合格句对中等概率抽取 X 条；首遍只统计条数，次遍用
   Algorithm S（无放回、均匀）按 ``--seed`` 抽取。额外内存 O(1)，与语料大小无关。
3. ``proportional``：按每条句对的**无向关注词对**分层，先统计各词对条数，再用最大
   余数法把 min(X, 可分层句对数) 个配额按比例分给各词对，次遍在每层内用由 ``--seed``
   派生的独立随机种子抽取。两种顺序相反的词对视为同一层（"上-下" 与 "下-上" 同层）。
   两侧关注词任一缺失的合格句对计入 ``skipped_no_word``，不占配额，因此这类记录存在时
   实际选中可能少于 X。额外内存只与词对种数有关，不随语料大小增长。
4. ``aligned``：不按 X 随机抽，而是读一份 **annotate 结果文件**（reference），取其中
   全部句对，再按当前 pairs 文件顺序逐条对回。身份优先用结果里的 ``candidate_key``；
   缺失时用 ``sentence_id`` + ``candidate.focus_index`` + ``candidate.replacement`` 重算。
   reference 中任何一条句对在源文件里找不到、或一条对到多条源句对，都会报错而不是静默
   跳过。结果文件里的非句对记录（``bad_record`` / ``alter_missed``，候选为空）忽略并计数。
   ``--limit`` 可省略（表示 reference 中全部句对）；给了则只取源文件顺序中的前 X 条。

“可用记录”沿用 annotate 的口径：坏 JSON 直接报错；不是句对的记录也会报错并提示先运行
``python replace``；句对但两侧检查未通过或字段损坏的计入 ``skipped_ineligible``，不占 X。
``--offset`` 在三种抽样模式下先跳过前 N 条合格候选再抽样；aligned 先对齐全部 reference，
再按源文件顺序跳过前 N 条匹配记录。

内存：三种抽样模式都不把语料读进内存；random/proportional 需要重复读一遍文件（第一遍
只计数），用 I/O 换内存。aligned 只需在内存里保留 reference 的句对身份集合；源语料仍是
流式读取。最终只保留选中的记录。

复现：random 的随机序列只由 ``seed`` 与文件行序决定；proportional 的每个词对用
``Random(f"{seed}:proportional:{词对}")``，与词对遍历顺序及其他层的规模无关。
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterator

from annotate import spatial_tools as st
from filter_funcs import (
    DEFAULT_MODE,
    DEFAULT_SEED,
    LIMIT_MODES,
    SELECTION_VERSION,
    Selection,
    allocate_quotas,
    fingerprint,
    record_content_hash,
    remember_selected,
)
from replace import corpus_pairs as cp


def _base_stats(*, limit: int | None, mode: str, seed: int, corpus_type: str, offset: int) -> dict:
    return {
        "version": SELECTION_VERSION,
        "mode": mode,
        "limit": limit,
        "seed": seed,
        "corpus_type": corpus_type,
        "offset": offset,
        "scanned": 0,
        "blank": 0,
        "skipped_ineligible": 0,
        "skipped_bad_record": 0,
        "skipped_no_word": 0,
        "offset_skipped": 0,
        "candidates": 0,
        "available": None,
        "selected": 0,
        "stopped_early": False,
        "selected_line_nos": [],
        # proportional
        "pair_totals": {},
        "pair_quotas": {},
        "pair_selected": {},
        # aligned
        "reference_file": None,
        "reference_rows": 0,
        "reference_blank": 0,
        "reference_non_pairs": 0,
        "reference_duplicates": 0,
        "reference_records": 0,
        "matched_reference_records": 0,
        "matched_source_records": 0,
        "unmatched_reference_records": 0,
        "limit_excluded": 0,
    }


def _parse_pair_row(line: str, line_no: int, path: Path) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{path} 第 {line_no} 行不是 JSON 对象，无法作为句对记录")
    if "pair_schema_version" not in row:
        raise ValueError(
            f"{path} 第 {line_no} 行不是替换阶段产出的句子对；"
            "请先运行 python replace，再将其输出传给标注入口"
        )
    return row


def _iter_candidates(
    path: Path, stats: dict | None, offset: int, corpus_type: str
) -> Iterator[tuple[int, dict]]:
    """流式产出合格句对及物理行号；先跳过 offset 条合格候选。"""
    skip = offset
    with open(path, "r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            if not line.strip():
                if stats is not None:
                    stats["blank"] += 1
                continue
            row = _parse_pair_row(line, line_no, path)
            if stats is not None:
                stats["scanned"] += 1
            if not cp.is_prepared_pair(row):
                if stats is not None:
                    stats["skipped_ineligible"] += 1
                continue
            if row.get("corpus_type") != corpus_type:
                raise ValueError(
                    f"{path} 第 {line_no} 行句对的 corpus_type={row.get('corpus_type')!r} "
                    f"与 --corpus-type={corpus_type!r} 不一致，不能使用另一词性的标注配置"
                )
            if skip > 0:
                skip -= 1
                if stats is not None:
                    stats["offset_skipped"] += 1
                continue
            if stats is not None:
                stats["candidates"] += 1
            yield line_no, row


def _iter_sequential(path: Path, limit: int, stats: dict, offset: int, corpus_type: str):
    selected = 0
    for line_no, row in _iter_candidates(path, stats, offset, corpus_type):
        selected += 1
        remember_selected(stats, line_no)
        yield line_no, row
        if selected >= limit:
            # 读够即停：不再探测尾部，否则 GB 级文件后面若全是坏行/不合格句对，
            # 这个“探测一条”会一直扫到 EOF。stopped_early 只表示“因达到限额停止”，
            # 不保证后面还有候选。
            stats["stopped_early"] = True
            return


def _iter_random(
    path: Path, limit: int, seed: int, stats: dict, offset: int, corpus_type: str
):
    # 第一遍只数合格句对，不保留行；这样 GB 级语料也不会进内存。
    snapshot = fingerprint(path)
    for _ in _iter_candidates(path, stats, offset, corpus_type):
        pass
    if fingerprint(path) != snapshot:
        raise RuntimeError(f"{path} 在随机抽样的第一遍读取过程中发生变化；请固定语料后重跑")
    total = stats["candidates"]
    stats["available"] = total
    quota = min(limit, total)
    if quota <= 0:
        return

    rng = random.Random(seed)
    remaining = total
    need = quota
    # 即使已经选满也继续读到 EOF：既能核对两遍读取看到的是同一份语料，
    # 也不会因为第二遍读到追加的记录而突破 X 的硬上限。
    for line_no, row in _iter_candidates(path, None, offset, corpus_type):
        if remaining <= 0:
            raise RuntimeError(f"{path} 在随机抽样的两遍读取之间变长了；请固定语料后重跑")
        take = need > 0 and rng.randrange(remaining) < need
        remaining -= 1
        if not take:
            continue
        need -= 1
        remember_selected(stats, line_no)
        yield line_no, row
    if fingerprint(path) != snapshot or remaining != 0 or need != 0:
        raise RuntimeError(f"{path} 在随机抽样的两遍读取之间发生变化；请固定语料后重跑")


def _pair_layer(record: dict) -> tuple[str, str] | None:
    """一条句对用于比例分层的无向关注词对（顺序相反视为同层）。

    两侧关注词任一不是非空字符串时返回 None：不猜词，也不把缺失当成分层。
    """
    focus_a = record.get("focus_word") or {}
    focus_b = record.get("focus_word_b") or {}
    word_a, word_b = focus_a.get("word"), focus_b.get("word")
    if not (isinstance(word_a, str) and word_a and isinstance(word_b, str) and word_b):
        return None
    return tuple(sorted((word_a, word_b)))


def _layer_label(layer: tuple[str, str]) -> str:
    return "|".join(layer)


def _iter_proportional(
    path: Path, limit: int, seed: int, stats: dict, offset: int, corpus_type: str
):
    snapshot = fingerprint(path)
    counts: Counter[tuple[str, str]] = Counter()
    for _, row in _iter_candidates(path, stats, offset, corpus_type):
        layer = _pair_layer(row)
        if layer is None:
            stats["skipped_no_word"] += 1
            continue
        counts[layer] += 1
    if fingerprint(path) != snapshot:
        raise RuntimeError(f"{path} 在比例抽样的第一遍读取过程中发生变化；请固定语料后重跑")

    total = sum(counts.values())
    stats["available"] = total
    quotas = allocate_quotas({_layer_label(layer): count for layer, count in counts.items()}, min(limit, total))
    stats["pair_totals"] = {_layer_label(layer): count for layer, count in sorted(counts.items())}
    stats["pair_quotas"] = dict(sorted(quotas.items()))

    # 每个词对一份独立随机状态：加词对/改别的层的规模不会重洗已有样本。
    states = {
        _layer_label(layer): [counts[layer], quotas[_layer_label(layer)],
                              random.Random(f"{seed}:proportional:{_layer_label(layer)}")]
        for layer in sorted(counts)
        if quotas[_layer_label(layer)] > 0
    }
    stats["pair_selected"] = {label: 0 for label in quotas}
    if not states:
        return
    pending = sum(state[1] for state in states.values())
    seen: Counter[str] = Counter()
    # 同样读到 EOF：核对逐层计数，避免“选满就 break”漏掉文件追加/删减。
    for line_no, row in _iter_candidates(path, None, offset, corpus_type):
        layer = _pair_layer(row)
        if layer is None:
            continue
        label = _layer_label(layer)
        seen[label] += 1
        state = states.get(label)
        if state is None:
            continue
        remaining, need, rng = state
        if remaining <= 0:
            raise RuntimeError(
                f"{path} 的「{label}」在比例抽样的两遍读取之间变多了；请固定语料后重跑"
            )
        take = rng.randrange(remaining) < need
        state[0] -= 1
        if not take:
            continue
        state[1] -= 1
        pending -= 1
        stats["pair_selected"][label] += 1
        remember_selected(stats, line_no)
        yield line_no, row
    if fingerprint(path) != snapshot or dict(seen) != {_layer_label(layer): count for layer, count in counts.items()} or pending:
        raise RuntimeError(f"{path} 在比例抽样的两遍读取之间发生变化；请固定语料后重跑")


# --------------------------------------------------------------------------
# aligned：用 annotate 结果文件逐条对回当前 pairs 输入
# --------------------------------------------------------------------------


def _pair_identity(record: dict) -> str | None:
    """源句对的稳定身份：candidate_key 优先，缺失时按同一公式重算。"""
    key = record.get("candidate_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    candidate = record.get("candidate")
    sentence_id = record.get("sentence_id")
    if not isinstance(sentence_id, str) or not sentence_id.strip():
        sentence_id = st.sentence_id(record)
    if not isinstance(candidate, dict):
        return None
    return st.candidate_key(sentence_id, candidate.get("focus_index"), candidate.get("replacement"))


def _reference_identity(record: dict) -> str | None:
    """annotate 结果记录的句对身份；非句对（没有候选）返回 None。"""
    return _pair_identity(record)


def _parse_reference_row(line: str, row_no: int, path: Path) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reference {path} 第 {row_no} 行不是合法 JSON：{exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"reference {path} 第 {row_no} 行不是 JSON 对象")
    return row


def _load_reference(path: Path, stats: dict) -> dict[str, dict]:
    """读 reference 的句对身份；返回 {candidate_key: {"row_no", "hash"}}。"""
    entries: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as reader:
        for row_no, line in enumerate(reader, start=1):
            if not line.strip():
                stats["reference_blank"] += 1
                continue
            row = _parse_reference_row(line, row_no, path)
            stats["reference_rows"] += 1
            key = _reference_identity(row)
            if key is None:
                if isinstance(row.get("candidate"), dict):
                    raise ValueError(
                        f"reference {path} 第 {row_no} 行有 candidate 却算不出句对身份"
                        "（缺 sentence_id / focus_index / replacement）；无法与源句对严格对齐"
                    )
                # annotate 结果也会落 bad_record / alter_missed；它们不是句对，不能强行对齐。
                stats["reference_non_pairs"] += 1
                continue
            content = record_content_hash(row)
            existing = entries.get(key)
            if existing is not None:
                if existing["hash"] and content and existing["hash"] != content:
                    raise ValueError(
                        f"reference {path} 第 {row_no} 行与第 {existing['row_no']} 行 candidate_key 相同"
                        "但句子不同；请检查 reference 是否混入了不同来源的记录"
                    )
                existing["hash"] = existing["hash"] or content
                stats["reference_duplicates"] += 1
                continue
            entries[key] = {"row_no": row_no, "hash": content}
    if not entries:
        raise ValueError(f"reference {path} 没有可用句对（空文件、只有空行或都不是 annotate 句对结果）")
    stats["reference_records"] = len(entries)
    return entries


def _iter_aligned(
    path: Path,
    reference_path: Path,
    limit: int | None,
    stats: dict,
    offset: int,
    corpus_type: str,
):
    entries = _load_reference(reference_path, stats)
    stats["reference_file"] = str(reference_path)
    snapshot = fingerprint(path)
    matched: set[str] = set()
    skipped = 0
    for line_no, row in _iter_candidates(path, stats, 0, corpus_type):
        key = _pair_identity(row)
        if key is None:
            # is_prepared_pair 已保证 candidate 完整；走到这里说明记录形状异常，不能静默丢。
            raise ValueError(f"{path} 第 {line_no} 行是合格句对却算不出 candidate_key，无法与 reference 对齐")
        entry = entries.get(key)
        if entry is None:
            continue
        if key in matched:
            raise ValueError(
                f"{path} 第 {line_no} 行与前面的源记录 candidate_key 重复（{key}）；"
                "无法与 reference 逐条一一对应"
            )
        if entry["hash"]:
            source_hash = record_content_hash(row)
            if source_hash and source_hash != entry["hash"]:
                raise ValueError(
                    f"{path} 第 {line_no} 行的 candidate_key 在 reference 中对应另一句"
                    f"（reference 第 {entry['row_no']} 行）；请固定输入后重跑"
                )
        matched.add(key)
        stats["matched_source_records"] += 1
        if skipped < offset:
            skipped += 1
            stats["offset_skipped"] += 1
            continue
        if limit is None or stats["selected"] < limit:
            remember_selected(stats, line_no)
            yield line_no, row
        else:
            stats["limit_excluded"] += 1
    if fingerprint(path) != snapshot:
        raise RuntimeError(f"{path} 在按 reference 对齐的读取过程中发生变化；请固定语料后重跑")

    stats["matched_reference_records"] = len(matched)
    stats["available"] = stats["matched_source_records"]
    unmatched = [key for key in entries if key not in matched]
    stats["unmatched_reference_records"] = len(unmatched)
    if unmatched:
        examples = "；".join(
            f"reference 第 {entries[key]['row_no']} 行（{key}）" for key in unmatched[:5]
        )
        raise ValueError(
            f"reference {reference_path} 有 {len(unmatched)} 条句对在源文件 {path} 中找不到对应记录："
            f"{examples}"
        )


def select_pairs(
    path: str | Path,
    *,
    limit: int | None = None,
    mode: str | None = None,
    seed: int = DEFAULT_SEED,
    corpus_type: str = st.DEFAULT_CORPUS_TYPE,
    offset: int = 0,
    reference_file: str | Path | None = None,
) -> Selection:
    """构造一次 annotate 限额筛选；返回惰性 Selection，消费 records 后才完成统计。

    Args:
        path: 当前 pairs 输入 JSONL（``python replace`` 的产物）
        limit: 正整数 X；aligned 模式可为 None（表示 reference 中全部句对）
        mode: ``sequential`` / ``random`` / ``proportional`` / ``aligned``；
            省略时为 sequential，给出 reference_file 时自动为 aligned
        seed: random 与 proportional 的随机种子；sequential/aligned 忽略但仍记录
        corpus_type: 已规范化的语料类型，用于逐行校验与比例分层前的类型检查
        offset: 先跳过多少条合格候选（aligned 按匹配到的源记录顺序跳过）
        reference_file: annotate 结果 JSONL；给出时自动使用 aligned 模式

    Raises:
        ValueError: limit/mode/seed/offset 非法，语料中出现坏 JSON/非句对/词性不符，
            reference 缺失、为空或无法与源句对一一对应
    """
    if reference_file is not None:
        if mode not in (None, "aligned"):
            raise ValueError("给出 reference_file 时必须使用 aligned 模式")
        mode = "aligned"
    elif mode is None:
        mode = DEFAULT_MODE
    if mode not in LIMIT_MODES:
        raise ValueError(f"limit_mode 必须是 {LIMIT_MODES} 之一，收到 {mode!r}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError(f"offset 必须是大于等于 0 的整数，收到 {offset!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed 必须是大于等于 0 的整数，收到 {seed!r}")
    if mode == "aligned":
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError(f"limit 必须是大于 0 的整数或 None，收到 {limit!r}")
    elif isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit 必须是大于 0 的整数，收到 {limit!r}")

    corpus = Path(path)
    stats = _base_stats(limit=limit, mode=mode, seed=seed, corpus_type=corpus_type, offset=offset)
    if mode == "sequential":
        records = _iter_sequential(corpus, limit, stats, offset, corpus_type)
    elif mode == "random":
        records = _iter_random(corpus, limit, seed, stats, offset, corpus_type)
    elif mode == "proportional":
        records = _iter_proportional(corpus, limit, seed, stats, offset, corpus_type)
    else:
        if reference_file is None:
            raise ValueError("aligned 模式必须给出 reference_file")
        reference = Path(reference_file)
        if not reference.is_file():
            raise FileNotFoundError(f"找不到 reference 文件 {reference}")
        records = _iter_aligned(corpus, reference, limit, stats, offset, corpus_type)
    return Selection(records=records, stats=stats)
