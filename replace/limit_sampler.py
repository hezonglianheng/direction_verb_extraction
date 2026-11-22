# encoding: utf8
"""replace 阶段的输入限额抽样：顺序 / 随机 / 按关注词比例。

X 的口径是**送入 PairPreparer 之前的原句记录数**，不是句子位置、候选项或最终句对数：
replace 仍会把一条原句展开成多个位置、多个替换句，本模块只决定哪些原句进入这一轮。

三种模式（``--limit-mode``）：

1. ``sequential``：按文件顺序取前 X 条可用记录，读够即停，单遍流式读取。
2. ``random``：从全部可用记录中等概率抽取 X 条；首遍只统计条数，次遍用
   Algorithm S（无放回、均匀）按 ``--seed`` 抽取。额外内存 O(1)，与语料大小无关。
3. ``proportional``：按每条记录的**主关注词**（方位词/趋向动词）分层，先统计各词条数，
   再用最大余数法把 min(X, 可分层记录数) 个配额按比例分给各词，次遍在每层内用由
   ``--seed`` 派生的独立随机种子抽取。额外内存只与词种数有关，不随语料大小增长。
   只有取得到本类关注词的记录进入比例池；取不到词的可用记录计入 ``skipped_no_word``，
   不占配额，因此这类记录存在时实际选中可能少于 X（真实语料实测为 0 条）。
   配额描述的是**选中记录按主关注词的分层**；replace 后续仍会展开句内其他合格位置，
   所以最终句对的逐词分布不直接等于这里的配额。
4. ``aligned``：不按 X 随机抽，而是读一份“已筛选文件”（reference），把其中每条记录
   按行号/内容对回同一份源语料，选出对应的源记录。``--limit`` 可省略（表示全部），
   给了则只取源文件顺序中的前 X 条匹配记录。reference 在源语料里找不到、源行状态不可用
   （false/error/limit 等）或目标 verdict 缺失/不可用时，会跳过并在消费完时汇总到
   ``unmatched_details``；真正的歧义（一条 reference 对到多条可用源记录、内容兜底多条
   可用同文记录、一条源记录对多条不同 reference 等）仍报错。
   reference 可以是 word_sampler 产物、replace 句对、annotate 结果或同源子集；行号字段
   （source_line_no / input_line_no / line_no，含 source 块）和句子内容都会参与校验。

“可用记录”沿用 replace 的口径：非空行、合法 JSON 对象、不是句对、``sentence`` 为非空
字符串列表，且 ``sentence_screening.screening_status`` 为 ``true``/``no_llm``（即
``is_judgement_eligible``）。不可判定或坏记录不占 X，但会计入统计，避免静默吞掉。

内存：三种模式都不把语料读入内存；random/proportional 需要重复读一遍文件（第一遍
只计数），用 I/O 换内存。语料再大也只是逐行处理 + 与词种数/批次大小相关的少量对象。

复现：random 的随机序列只由 ``seed`` 与文件行序决定；proportional 的每个词用
``Random(f"{seed}:proportional:{word}")``，与词的遍历顺序和其他词的规模无关。
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from extract import sentence_screening as ss
from annotate import spatial_tools as st
from filter_funcs import (
    DEFAULT_MODE,
    DEFAULT_SEED,
    LIMIT_MODES,
    REFERENCE_LEVELS,
    SAMPLE_MODES,
    VERDICT_FOCUS_ORDER_KEY,
    SELECTION_VERSION,
    Selection,
    allocate_quotas,
    explicit_sentence_id,
    fingerprint,
    positive_line,
    record_content_hash,
    record_line_keys,
    remember_selected,
)

# 旧调用名保持不变：本模块（及外部脚本）仍可用私有名，行为与 filter_funcs 的实现一致。
_fingerprint = fingerprint
_positive_line = positive_line
_record_line_keys = record_line_keys
_record_content_hash = record_content_hash
_explicit_sentence_id = explicit_sentence_id
_remember_selected = remember_selected


def _base_stats(*, limit: int | None, mode: str, seed: int, corpus_type: str) -> dict:
    return {
        "version": SELECTION_VERSION,
        "mode": mode,
        "limit": limit,
        "seed": seed,
        "corpus_type": corpus_type,
        "scanned": 0,
        "blank": 0,
        "skipped_ineligible": 0,
        "skipped_bad_record": 0,
        "skipped_no_word": 0,
        "candidates": 0,
        "available": None,
        "selected": 0,
        "stopped_early": False,
        "selected_line_nos": [],
    }


def _usable_sentence(row: dict) -> bool:
    """当前 replace 能处理的句子形状：非空、每个 token 都是字符串。"""
    tokens = row.get("sentence")
    return isinstance(tokens, list) and bool(tokens) and all(isinstance(t, str) for t in tokens)


def _parse_row(line: str, line_no: int, path: Path) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"{path} 第 {line_no} 行不是 JSON 对象，无法作为语料记录")
    if "pair_schema_version" in row:
        raise ValueError(
            f"{path} 第 {line_no} 行已是句对，请勿再次替换；"
            "将它交给 python annotate 或其 --engine all-in-one 模式"
        )
    return row


def _iter_candidates(path: Path, stats: dict | None) -> Iterator[tuple[int, dict]]:
    """流式产出可用记录及其物理行号；stats 为 None 时只读不重复计数（第二遍用）。"""
    with open(path, "r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            if not line.strip():
                if stats is not None:
                    stats["blank"] += 1
                continue
            row = _parse_row(line, line_no, path)
            if stats is not None:
                stats["scanned"] += 1
            if not ss.is_judgement_eligible(row):
                if stats is not None:
                    stats["skipped_ineligible"] += 1
                continue
            if not _usable_sentence(row):
                if stats is not None:
                    stats["skipped_bad_record"] += 1
                continue
            if stats is not None:
                stats["candidates"] += 1
            yield line_no, row


_CANDIDATE_STAT_KEYS = (
    "scanned",
    "blank",
    "skipped_ineligible",
    "skipped_bad_record",
    "candidates",
)
"""逐个候选扫描时会累加的统计键；用于判断是否还有候选时临时回滚这次探测。"""


def _iter_sequential(path: Path, limit: int, stats: dict) -> Iterator[tuple[int, dict]]:
    selected = 0
    candidates = _iter_candidates(path, stats)
    while True:
        try:
            line_no, row = next(candidates)
        except StopIteration:
            return
        selected += 1
        _remember_selected(stats, line_no)
        yield line_no, row
        if selected >= limit:
            break

    # 再探测一个候选，仅用于区分「达到上限且还有剩余」与「恰好读到文件末尾」。
    # 探测本身不应改变本轮统计：候选数/扫描数按实际取走的前 limit 条计。
    snapshot = {key: stats.get(key, 0) for key in _CANDIDATE_STAT_KEYS}
    try:
        next(candidates)
    except StopIteration:
        return
    except Exception:  # noqa: BLE001 —— 限额之后的行不应再影响本轮结果
        return
    else:
        stats["stopped_early"] = True
    finally:
        for key, value in snapshot.items():
            stats[key] = value


def _iter_random(path: Path, limit: int, seed: int, stats: dict) -> Iterator[tuple[int, dict]]:
    # 第一遍只数可用记录，不保留行；这样 GB 级语料也不会进内存。
    snapshot = _fingerprint(path)
    for _ in _iter_candidates(path, stats):
        pass
    if _fingerprint(path) != snapshot:
        raise RuntimeError(f"{path} 在随机抽样的第一遍读取过程中发生变化；请固定语料后重跑")
    total = stats["candidates"]
    stats["available"] = total
    quota = min(limit, total)
    if quota <= 0:
        return

    rng = random.Random(seed)
    remaining = total
    need = quota
    # 即使已经选满也继续读到 EOF：这样既能核对两遍读取看到的是同一份语料，
    # 也不会因为第二遍读到追加的记录而突破 X 的硬上限。
    for line_no, row in _iter_candidates(path, None):
        if remaining <= 0:
            raise RuntimeError(f"{path} 在随机抽样的两遍读取之间变长了；请固定语料后重跑")
        take = need > 0 and rng.randrange(remaining) < need
        remaining -= 1
        if not take:
            continue
        need -= 1
        _remember_selected(stats, line_no)
        yield line_no, row
    if _fingerprint(path) != snapshot or remaining != 0 or need != 0:
        raise RuntimeError(f"{path} 在随机抽样的两遍读取之间发生变化；请固定语料后重跑")


def _layer_word(row: dict, corpus_type: str, vocab: dict[str, set[str]] | None = None) -> str | None:
    """一条记录用于比例分层的“主关注词”。

    取 ``pick_focus_words`` 中类别等于本次语料类型的第一个位置：它就是这个位置被改写的
    那个词，也代表这条记录属于哪一层。一条记录含多个词时只计一层，避免同一句被多层
    重复分配配额；replace 后续仍按原规则展开句内全部合格位置。

    找不到本类关注词（如纯防御的回退记录）时返回 None，不猜词、不把搭配动词当方位词。
    """
    entries = st.pick_focus_words(row, vocab)
    for entry in entries:
        if entry.get("kind") == corpus_type and isinstance(entry.get("word"), str):
            return entry["word"]
    return None


def _iter_proportional(
    path: Path, limit: int, seed: int, corpus_type: str, stats: dict
) -> Iterator[tuple[int, dict]]:
    vocab = st.load_focus_vocab()
    snapshot = _fingerprint(path)
    counts: Counter[str] = Counter()
    for _, row in _iter_candidates(path, stats):
        word = _layer_word(row, corpus_type, vocab)
        if word is None:
            stats["skipped_no_word"] += 1
            continue
        counts[word] += 1
    if _fingerprint(path) != snapshot:
        raise RuntimeError(f"{path} 在比例抽样的第一遍读取过程中发生变化；请固定语料后重跑")

    total = sum(counts.values())
    stats["available"] = total
    quotas = allocate_quotas(dict(counts), min(limit, total))
    stats["word_totals"] = dict(sorted(counts.items()))
    stats["word_quotas"] = dict(sorted(quotas.items()))

    # 每个词一份独立随机状态：加词/改别的词的规模不会重洗已有样本。
    states = {
        word: [counts[word], quotas[word], random.Random(f"{seed}:proportional:{word}")]
        for word in sorted(quotas)
        if quotas[word] > 0
    }
    stats["word_selected"] = {word: 0 for word in quotas}
    if not states:
        return
    pending = sum(state[1] for state in states.values())
    seen: Counter[str] = Counter()
    # 同样读到 EOF：核对逐词计数，避免“选满就 break”漏掉文件追加/删减。
    for line_no, row in _iter_candidates(path, None):
        word = _layer_word(row, corpus_type, vocab)
        if word is None:
            continue
        seen[word] += 1
        state = states.get(word)
        if state is None:
            continue
        remaining, need, rng = state
        if remaining <= 0:
            raise RuntimeError(
                f"{path} 的「{word}」在比例抽样的两遍读取之间变多了；请固定语料后重跑"
            )
        take = rng.randrange(remaining) < need
        state[0] -= 1
        if not take:
            continue
        state[1] -= 1
        pending -= 1
        stats["word_selected"][word] += 1
        _remember_selected(stats, line_no)
        yield line_no, row
    if _fingerprint(path) != snapshot or seen != counts or pending:
        raise RuntimeError(f"{path} 在比例抽样的两遍读取之间发生变化；请固定语料后重跑")


# --------------------------------------------------------------------------
# aligned：用已筛选文件逐条对回同一份源语料
# --------------------------------------------------------------------------

@dataclass
class _ReferenceEntry:
    """一条 reference 记录（已按身份去重）以及与源记录对齐所需的信息。"""

    line_keys: dict[str, set[int]]
    content_hash: str | None
    document: str | None
    sentence_id: str | None
    row_no: int
    allow_content_fallback: bool = False
    matched: bool = False
    match_count: int = 0
    matched_by: str | None = None
    target_index: int | None = None
    target_word: str | None = None

    @property
    def target_key(self) -> tuple[int, str] | None:
        if self.target_index is None or not self.target_word:
            return None
        return (self.target_index, self.target_word)


def _record_document(record: dict) -> str | None:
    """来源文档：先看 source 块，再看顶层；与 spatial_tools.source_document_id 同源。"""
    source = record.get("source")
    if isinstance(source, dict):
        document = st.source_document_id(source)
        if document:
            return document
    return st.source_document_id(record)


def _looks_like_current_file_reference(record: dict) -> bool:
    """这条 reference 的行号是否可能指向“当前 pairs 文件”而不是源语料。

    annotate 结果（含 voting 派生）顶层 sentence 常为空，原句在 pair.a.slice；它的
    line_no/input_line_no 是 pairs 文件行号。word_sampler 产物没有 pair/sentence_id，
    顶层 sentence 非空，它的 line_no 就是源语料行号。两类记录都只有 line_no 时，
    必须靠这个形状区分，否则 annotate 记录会被误当成源行号。
    """
    if "pair_schema_version" in record:
        # 旧版 replace 句对可能没有 source_line_no；此时 line_no 仍是源语料行号，
        # 不能把整份句对文件当成 annotate 的“当前文件行号”而放弃按行匹配。
        return False
    if isinstance(record.get("pair"), dict):
        return True
    if _explicit_sentence_id(record):
        return True
    tokens = record.get("sentence")
    return not (isinstance(tokens, list) and tokens)


def _parse_reference_row(line: str, row_no: int, path: Path) -> dict:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reference {path} 第 {row_no} 行不是合法 JSON：{exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"reference {path} 第 {row_no} 行不是 JSON 对象")
    return row


def _reference_identity(entry: _ReferenceEntry, *, verdict_level: bool = False) -> tuple | None:
    """去重身份：显式 sentence_id 最稳，其次行号集合+内容，最后仅内容。

    verdict 级在记录身份后再加目标词 (index, word)：同一句的不同目标位置必须
    保留成不同选择项；同一目标位置的重复 pair 行仍会合并。
    """
    if entry.sentence_id:
        base: tuple | None = ("sentence_id", entry.sentence_id)
    else:
        line_pairs = frozenset(
            (label, value) for label, values in entry.line_keys.items() for value in values
        )
        if line_pairs:
            base = ("lines", entry.document, entry.content_hash, line_pairs)
        elif entry.content_hash:
            base = ("content", entry.document, entry.content_hash)
        else:
            base = None
    if not verdict_level:
        return base
    if base is None or entry.target_key is None:
        return None
    return ("verdict", base, entry.target_index, entry.target_word)


def _reference_mismatches(
    entry: _ReferenceEntry, source_hash: str | None, source_document: str | None, source_sid: str | None
) -> list[str]:
    """reference 与某条源记录在内容/来源/ID 上的差异，用于对齐和跳过诊断。"""
    mismatches: list[str] = []
    if entry.content_hash and entry.content_hash != source_hash:
        mismatches.append("content_hash")
    if entry.document and entry.document != source_document:
        mismatches.append("document")
    if entry.sentence_id and source_sid and entry.sentence_id != source_sid:
        mismatches.append("sentence_id")
    return mismatches


def _reference_entry_matches(
    entry: _ReferenceEntry, source_hash: str | None, source_document: str | None, source_sid: str | None
) -> bool:
    """行号命中后还要过内容/来源/ID 校验；有内容的记录不能用同号不同句冒充。"""
    return not _reference_mismatches(entry, source_hash, source_document, source_sid)


def _describe_reference_entry(entry: _ReferenceEntry) -> str:
    parts = []
    for label, values in sorted(entry.line_keys.items()):
        parts.append(f"{label}={sorted(values)}")
    if entry.content_hash:
        parts.append(f"hash={entry.content_hash}")
    if entry.sentence_id:
        parts.append(f"sentence_id={entry.sentence_id}")
    if entry.target_key is not None:
        parts.append(f"target={entry.target_index}:{entry.target_word}")
    return f"第 {entry.row_no} 行（{'，'.join(parts) or '无可用键'}）"


_MISMATCH_LABELS = {
    "content_hash": "句子内容",
    "document": "来源文档",
    "sentence_id": "sentence_id",
}


def _observation_priority(observation: dict) -> tuple:
    """同一类观测只留最能解释跳过原因的一条：先无 mismatch，再不可用。"""
    return (
        0 if not observation.get("mismatches") else 1,
        0 if not observation.get("usable") else 1,
        0 if not observation.get("eligible") else 1,
        observation.get("line_no") or 0,
    )


def _remember_observation(
    observations: dict[int, dict[str, dict]],
    index: int,
    *,
    kind: str,
    line_no: int,
    source_hash: str | None,
    source_document: str | None,
    source_sid: str | None,
    status: str,
    eligible: bool,
    usable: bool,
    mismatches: list[str],
) -> None:
    """按“行号/内容 × 是否匹配”保留最多 4 条观测。

    行号观测比内容观测更权威；同一类观测中优先保留没有 mismatch、且状态不可用
    的那条，这样“同文但来源不同”的前一条不会把真正命中的后一条挤掉。
    """
    bucket = observations.setdefault(index, {})
    key = f"{kind}_{'match' if not mismatches else 'mismatch'}"
    observation = {
        "kind": kind,
        "line_no": line_no,
        "source_hash": source_hash,
        "source_document": source_document,
        "source_sid": source_sid,
        "status": status,
        "eligible": eligible,
        "usable": usable,
        "mismatches": list(mismatches),
    }
    existing = bucket.get(key)
    if existing is None or _observation_priority(observation) < _observation_priority(existing):
        bucket[key] = observation


def _describe_line_keys(entry: _ReferenceEntry) -> str:
    return (
        "，".join(
            f"{label}={sorted(values)}" for label, values in sorted(entry.line_keys.items())
        )
        or "无行号"
    )


def _detail_text(
    entry: _ReferenceEntry,
    detail: dict,
    observation: dict | None,
    extra: str = "",
) -> str:
    base = _describe_reference_entry(entry)
    reason = detail["reason"]
    status = detail.get("status")
    source_line = detail.get("source_line_no")
    if reason == "source_line_missing":
        tail = ""
        if observation is not None:
            labels = "、".join(
                _MISMATCH_LABELS.get(item, item) for item in observation.get("mismatches", [])
            )
            detail_text = f"；内容在源语料第 {source_line} 行存在"
            if labels:
                detail_text += f"，但{labels}不匹配"
            if status:
                detail_text += f"（status={status}）"
            tail = detail_text + "；reference 有权威行号，不做内容兜底"
        return f"{base}：源语料中找不到对应的行号 {_describe_line_keys(entry)}{tail}"
    if reason == "line_collision":
        labels = "、".join(
            _MISMATCH_LABELS.get(item, item) for item in detail.get("mismatches", [])
        ) or "内容"
        state = f"；源记录 status={status}" if status else ""
        return (
            f"{base}：行号命中源语料第 {source_line} 行，但{labels}不匹配"
            f"（源 hash={detail.get('source_content_hash')}，"
            f"reference hash={entry.content_hash}）{state}"
        )
    if reason == "record_unavailable":
        hit = "身份匹配" if entry.line_keys else "内容命中"
        return (
            f"{base}：源语料第 {source_line} 行{hit}，"
            f"但源记录 status={status}，不进入 replace"
        )
    if reason == "source_record_invalid":
        return f"{base}：源语料第 {source_line} 行句子形状不可用"
    if reason == "content_missing":
        return f"{base}：源语料中没有匹配的句子内容"
    if reason == "content_mismatch":
        labels = "、".join(
            _MISMATCH_LABELS.get(item, item) for item in detail.get("mismatches", [])
        ) or "内容"
        return f"{base}：句子内容命中源语料第 {source_line} 行，但{labels}不匹配"
    if reason == "content_unmatched":
        return f"{base}：句子内容命中源语料第 {source_line} 行，但无法唯一对齐"
    if reason == "verdict_missing":
        return f"{base}：源语料第 {source_line} 行找不到目标 verdict{extra}"
    if reason == "verdict_unavailable":
        return f"{base}：源语料第 {source_line} 行目标 verdict status={status}，不进入 replace"
    if reason == "screening_invalid":
        return f"{base}：源语料第 {source_line} 行 screening 块不是对象，无法读取目标 verdict"
    return f"{base}：{extra or reason}"


def _detail_from_entry(
    entry: _ReferenceEntry,
    *,
    reason: str,
    status: str | None,
    status_scope: str | None,
    observation: dict | None,
    extra: str = "",
) -> dict:
    detail = {
        "reference_row_no": entry.row_no,
        "line_keys": {
            label: sorted(values) for label, values in sorted(entry.line_keys.items())
        },
        "content_hash": entry.content_hash,
        "document": entry.document,
        "sentence_id": entry.sentence_id,
        "target_index": entry.target_index,
        "target_word": entry.target_word,
        "reason": reason,
        "status": status,
        "status_scope": status_scope,
        "source_line_no": observation.get("line_no") if observation else None,
        "source_content_hash": observation.get("source_hash") if observation else None,
        "source_document": observation.get("source_document") if observation else None,
        "mismatches": list(observation.get("mismatches", [])) if observation else [],
    }
    detail["text"] = _detail_text(entry, detail, observation, extra)
    return detail


def _classify_unmatched(
    entry: _ReferenceEntry,
    observations: dict[str, dict],
    skip_info: dict | None,
) -> tuple[str, str | None, str | None, dict | None, str]:
    """把未对齐项归到可报告的原因；优先使用 verdict 跳过信息。"""
    if skip_info is not None:
        return (
            skip_info["reason"],
            skip_info.get("status"),
            skip_info.get("status_scope"),
            skip_info.get("observation"),
            skip_info.get("extra", ""),
        )

    line_match = observations.get("line_match")
    line_mismatch = observations.get("line_mismatch")
    content_match = observations.get("content_match")
    content_mismatch = observations.get("content_mismatch")

    if entry.line_keys:
        if line_match is not None:
            if not line_match.get("usable"):
                return "source_record_invalid", line_match.get("status"), "record", line_match, ""
            if not line_match.get("eligible"):
                return "record_unavailable", line_match.get("status"), "record", line_match, ""
            # annotate 的 line_no 可能只是 pairs 文件行号；此时内容兜底才是真正的
            # 匹配依据，行号观测可用但没进入 matched 时优先看内容观测。
            if entry.allow_content_fallback and content_match is not None:
                if not content_match.get("usable"):
                    return "source_record_invalid", content_match.get("status"), "record", content_match, ""
                if not content_match.get("eligible"):
                    return "record_unavailable", content_match.get("status"), "record", content_match, ""
                return "content_unmatched", content_match.get("status"), "record", content_match, ""
            # 身份已经匹配且可用却没进入 matched，说明是结构/歧义问题；
            # 这类问题在扫描过程中会直接报错，正常不会走到这里。
            return "line_collision", line_match.get("status"), "record", line_match, ""
        if line_mismatch is not None:
            if entry.allow_content_fallback and content_match is not None:
                if not content_match.get("usable"):
                    return "source_record_invalid", content_match.get("status"), "record", content_match, ""
                if not content_match.get("eligible"):
                    return "record_unavailable", content_match.get("status"), "record", content_match, ""
                return "content_unmatched", content_match.get("status"), "record", content_match, ""
            return "line_collision", line_mismatch.get("status"), "record", line_mismatch, ""
        # 有权威行号但行号没出现；内容即使存在也不做兜底，但把内容命中状态一并报告。
        observation = content_match or content_mismatch
        status = content_match.get("status") if content_match is not None else None
        scope = "record" if status else None
        return "source_line_missing", status, scope, observation, ""

    if content_match is not None:
        if not content_match.get("usable"):
            return "source_record_invalid", content_match.get("status"), "record", content_match, ""
        if not content_match.get("eligible"):
            return "record_unavailable", content_match.get("status"), "record", content_match, ""
        return "content_unmatched", content_match.get("status"), "record", content_match, ""
    if content_mismatch is not None:
        return "content_mismatch", content_mismatch.get("status"), "record", content_mismatch, ""
    return "content_missing", None, None, None, ""


def _build_unmatched_details(
    entries: list[_ReferenceEntry],
    unmatched_indices: list[int],
    observations: dict[int, dict[str, dict]],
    skips: dict[int, dict],
) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    details: list[dict] = []
    reasons: Counter = Counter()
    statuses: Counter = Counter()
    for index in sorted(unmatched_indices, key=lambda item: entries[item].row_no):
        entry = entries[index]
        reason, status, status_scope, observation, extra = _classify_unmatched(
            entry, observations.get(index, {}), skips.get(index)
        )
        detail = _detail_from_entry(
            entry,
            reason=reason,
            status=status,
            status_scope=status_scope,
            observation=observation,
            extra=extra,
        )
        details.append(detail)
        reasons[reason] += 1
        if status:
            statuses[f"{status_scope}:{status}"] += 1
    return details, dict(sorted(reasons.items())), dict(sorted(statuses.items()))


def _valid_target(index, word) -> bool:
    return (
        isinstance(index, int)
        and not isinstance(index, bool)
        and isinstance(word, str)
        and bool(word)
    )


def _reference_target_keys(
    row: dict, reference_level: str, row_no: int, path: Path
) -> list[tuple[int, str] | None]:
    """reference 行在 verdict 模式下代表哪些目标词位置。

    pair 行优先用 a 侧 ``focus_word``；普通记录则把 ``screening.verdicts[]``
    展开成多项。record 模式始终只有一项 ``None``。
    """
    if reference_level == "record":
        return [None]
    pair_like = (
        "focus_word" in row
        or "pair_schema_version" in row
        or isinstance(row.get("pair"), dict)
    )
    if pair_like:
        focus = row.get("focus_word")
        if isinstance(focus, dict):
            index, word = focus.get("index"), focus.get("word")
            if _valid_target(index, word):
                return [(index, word)]
        raise ValueError(
            f"reference {path} 第 {row_no} 行是 pair/含 focus_word 的 reference，"
            "但缺少合法的 focus_word.index/word；verdict 模式不能改用整句 verdict 展开"
        )
    screening = row.get("screening")
    if "screening" in row and not isinstance(screening, dict):
        raise ValueError(
            f"reference {path} 第 {row_no} 行 screening 块不是对象；"
            "verdict 模式无法读取目标词位置"
        )
    verdicts = screening.get("verdicts") if isinstance(screening, dict) else None
    keys: list[tuple[int, str] | None] = []
    if isinstance(verdicts, list):
        for position, verdict in enumerate(verdicts):
            if not isinstance(verdict, dict):
                raise ValueError(
                    f"reference {path} 第 {row_no} 行 screening.verdicts[{position}] "
                    "不是 JSON 对象；verdict 模式无法确定目标词位置"
                )
            index, word = verdict.get("index"), verdict.get("word")
            if not _valid_target(index, word):
                raise ValueError(
                    f"reference {path} 第 {row_no} 行 screening.verdicts[{position}] "
                    f"缺少合法的 index/word（index={index!r}, word={word!r}）"
                )
            keys.append((index, word))
    if keys:
        # 同一行里重复的 verdict 目标只算一次；真正相同的重复行交给下方按身份去重。
        return list(dict.fromkeys(keys))
    target = row.get("target")
    if isinstance(target, dict):
        index, word = target.get("index"), target.get("word")
        if _valid_target(index, word):
            return [(index, word)]
    raise ValueError(
        f"reference {path} 第 {row_no} 行在 verdict 模式下没有可用的目标词 index/word；"
        "需要 focus_word、screening.verdicts[] 或 target 中至少一项"
    )


def _load_reference(
    path: Path, reference_level: str = "record"
) -> tuple[list[_ReferenceEntry], dict]:
    """读 reference，按身份去重；行号/内容/ID 都没有的记录直接报错。"""
    if reference_level not in REFERENCE_LEVELS:
        raise ValueError(f"reference_level 必须是 {REFERENCE_LEVELS} 之一，收到 {reference_level!r}")
    verdict_level = reference_level == "verdict"
    entries: list[_ReferenceEntry] = []
    identities: dict[tuple, int] = {}
    stats = {"rows": 0, "blank": 0, "duplicates": 0, "targets": 0}
    with open(path, "r", encoding="utf-8") as reader:
        for row_no, line in enumerate(reader, start=1):
            if not line.strip():
                stats["blank"] += 1
                continue
            row = _parse_reference_row(line, row_no, path)
            stats["rows"] += 1
            line_keys = _record_line_keys(row)
            content_hash = _record_content_hash(row)
            document = _record_document(row)
            sentence_id = _explicit_sentence_id(row)
            allow_content_fallback = (
                not line_keys
                or (
                    not line_keys.get("source_line_no")
                    and _looks_like_current_file_reference(row)
                )
            )
            for target in _reference_target_keys(row, reference_level, row_no, path):
                stats["targets"] += 1
                target_index, target_word = (None, None) if target is None else target
                entry = _ReferenceEntry(
                    line_keys=dict(line_keys),
                    content_hash=content_hash,
                    document=document,
                    sentence_id=sentence_id,
                    row_no=row_no,
                    allow_content_fallback=allow_content_fallback,
                    target_index=target_index,
                    target_word=target_word,
                )
                identity = _reference_identity(entry, verdict_level=verdict_level)
                if identity is None:
                    suffix = (
                        f"的目标词 {target_index}:{target_word} 没有可用于对齐的行号、句子内容或 sentence_id"
                        if verdict_level
                        else "没有任何可用于对齐的行号、句子内容或 sentence_id"
                    )
                    raise ValueError(
                        f"reference {path} 第 {row_no} 行{suffix}；"
                        "无法确定它对应源语料的哪一条记录"
                    )
                if identity in identities:
                    existing = entries[identities[identity]]
                    if existing.content_hash and entry.content_hash and existing.content_hash != entry.content_hash:
                        raise ValueError(
                            f"reference {path} 第 {row_no} 行与第 {existing.row_no} 行身份相同但句子不同；"
                            "请检查 reference 是否混入了不同来源的记录"
                        )
                    for label, values in entry.line_keys.items():
                        existing.line_keys.setdefault(label, set()).update(values)
                    if existing.content_hash is None:
                        existing.content_hash = entry.content_hash
                    if existing.document is None:
                        existing.document = entry.document
                    existing.allow_content_fallback = (
                        (existing.allow_content_fallback or entry.allow_content_fallback)
                        and not existing.line_keys.get("source_line_no")
                    )
                    stats["duplicates"] += 1
                    continue
                identities[identity] = len(entries)
                entries.append(entry)
    if not entries:
        raise ValueError(f"reference {path} 没有可用记录（空文件或只有空行），无法做平行对齐")
    return entries, stats


def _reference_indexes(entries: list[_ReferenceEntry]) -> tuple[dict, dict, dict, dict]:
    """行号按命名空间分开建索引，绝不把源行号和当前文件行号当成同一个整数。

    - source_line_no：源语料（或祖先语料）行号，权威。
    - input_line_no / line_no：当前输入文件行号；只有 reference 没有 source_line_no 时才用。
    """
    by_source_line: dict[int, set[int]] = {}
    by_current_line: dict[int, set[int]] = {}
    by_hash: dict[str, set[int]] = {}
    by_sid: dict[str, set[int]] = {}
    for index, entry in enumerate(entries):
        for value in entry.line_keys.get("source_line_no", ()):
            by_source_line.setdefault(value, set()).add(index)
        # annotate/内容型 reference 的 line_no 是 pairs 文件行号，不是源行号；它只允许
        # 走内容兜底，绝不能拿这个数字去源语料里“按行命中”，否则重复句会被静默误配。
        if not entry.line_keys.get("source_line_no") and not entry.allow_content_fallback:
            for label in ("input_line_no", "line_no"):
                for value in entry.line_keys.get(label, ()):
                    by_current_line.setdefault(value, set()).add(index)
        if entry.content_hash:
            by_hash.setdefault(entry.content_hash, set()).add(index)
        if entry.sentence_id:
            by_sid.setdefault(entry.sentence_id, set()).add(index)
    return by_source_line, by_current_line, by_hash, by_sid


def _iter_aligned_source(
    path: Path, stats: dict, reference_level: str
) -> Iterator[tuple[int, dict, dict]]:
    """aligned 的源语料读法：产出源行及其状态，供对齐和跳过诊断共同使用。

    record 模式仍按 ``is_judgement_eligible`` 决定能否进入匹配，但不再在这里
    静默丢弃不可用行；调用方需要这些行来报告“为什么 reference 找不到”。
    verdict 模式读到记录内部再按 verdict 判断，所以记录级状态只作诊断。
    """
    with open(path, "r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            if not line.strip():
                stats["blank"] += 1
                continue
            row = _parse_row(line, line_no, path)
            stats["scanned"] += 1
            usable = _usable_sentence(row)
            status = ss.screening_status(row)
            if reference_level == "record":
                eligible = ss.is_judgement_eligible(row)
                if not eligible:
                    stats["skipped_ineligible"] += 1
                elif not usable:
                    stats["skipped_bad_record"] += 1
                else:
                    stats["candidates"] += 1
            else:
                eligible = True
                if not usable:
                    stats["skipped_bad_record"] += 1
                else:
                    stats["candidates"] += 1
                    stats["source_candidates"] = stats.get("source_candidates", 0) + 1
            yield line_no, row, {
                "status": status,
                "eligible": eligible,
                "usable": usable,
            }


def _prune_match_tree(nodes, index: int, word: str) -> list[dict]:
    """只保留目标词节点；若目标嵌在子节点里，也保留通向它的祖先。"""
    kept: list[dict] = []
    for node in nodes or ():
        if not isinstance(node, dict):
            continue
        if node.get("index") == index and node.get("word") == word:
            kept.append(copy.deepcopy(node))
            continue
        children = _prune_match_tree(node.get("children") or (), index, word)
        if children:
            copied = copy.deepcopy(node)
            copied["children"] = children
            kept.append(copied)
    return kept


def _prune_grounding_block(block, index: int, word: str) -> None:
    """把 grounding 候选裁到当前目标位置，绝不借用其他位置的三元组。"""
    if not isinstance(block, dict):
        return
    candidates = block.get("candidates")
    if not isinstance(candidates, list):
        return
    kept = [
        copy.deepcopy(candidate)
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("index") == index
        and candidate.get("word") == word
    ]
    block["candidates"] = kept
    keeps = [candidate.get("keep") for candidate in kept]
    block["valid"] = True if True in keeps else None if None in keeps else False


def _find_source_verdict(row: dict, index: int, word: str) -> dict | None:
    screening = row.get("screening")
    if not isinstance(screening, dict):
        return None
    verdicts = screening.get("verdicts")
    if not isinstance(verdicts, list):
        return None
    for verdict in verdicts:
        if (
            isinstance(verdict, dict)
            and verdict.get("index") == index
            and verdict.get("word") == word
        ):
            return verdict
    return None


class _SkipVerdict(Exception):
    """目标 verdict 不可用：可跳过并记入报告，不视为数据损坏。"""

    def __init__(self, reason: str, *, status: str | None = None, detail: str = ""):
        super().__init__(detail)
        self.reason = reason
        self.status = status
        self.detail = detail


def _prune_source_to_verdict(
    row: dict,
    line_no: int,
    entry: _ReferenceEntry,
    reference_path: Path,
    path: Path,
) -> dict:
    """生成只含一个 verdict 的源记录副本，供 PairPreparer 精确展开该位置。"""
    index, word = entry.target_key  # type: ignore[misc]
    screening = row.get("screening")
    if "screening" in row and not isinstance(screening, dict):
        raise _SkipVerdict(
            "screening_invalid",
            status=ss.screening_status(row),
            detail="源记录 screening 块不是对象",
        )
    verdict = _find_source_verdict(row, index, word)
    if verdict is None:
        raise _SkipVerdict(
            "verdict_missing",
            detail=f"找不到目标 verdict (index={index}, word={word!r})",
        )
    status = ss.verdict_status(verdict)
    if status not in ("true", "no_llm"):
        raise _SkipVerdict(
            "verdict_unavailable",
            status=status,
            detail=f"目标 verdict status={status}",
        )

    pruned = copy.deepcopy(row)
    pruned.pop("sampled_words", None)
    pruned.pop("sampled_word", None)
    pruned["match_tree"] = _prune_match_tree(row.get("match_tree") or (), index, word)
    if not pruned["match_tree"]:
        raise ValueError(
            f"reference {reference_path} 的{_describe_reference_entry(entry)}"
            f"在源语料 {path} 第 {line_no} 行的 match_tree 里找不到目标节点 "
            f"(index={index}, word={word!r})"
        )

    screening = pruned.get("screening")
    if not isinstance(screening, dict):
        raise ValueError(
            f"reference {reference_path} 的{_describe_reference_entry(entry)}"
            f"对应源记录没有 screening 块，无法裁剪 verdict"
        )
    screening["verdicts"] = [copy.deepcopy(verdict)]
    screening["status"] = status
    screening["extractable"] = (
        True if status == "true" else False if status == "false" else None
    )
    # 单 verdict 的 source 不能继承整句聚合的 mixed/error。
    screening["source"] = verdict.get("source") or "screening"

    for block_key in ("localization_grounding", "direction_grounding"):
        _prune_grounding_block(pruned.get(block_key), index, word)

    source_line_no = st.record_line_no(row)
    pruned["source_line_no"] = source_line_no if source_line_no is not None else line_no

    # 原始序号要按“不受抽样字段限制”的完整关注词序列算：裁剪副本已经去掉
    # sampled_words/sampled_word，reference 明确选中的位置不应再被旧抽样集合挡住。
    full_row = {
        key: value
        for key, value in row.items()
        if key not in ("sampled_words", "sampled_word")
    }
    full_focuses = st.pick_focus_words(full_row)
    original_orders = [
        position
        for position, focus in enumerate(full_focuses, start=1)
        if focus.get("index") == index and focus.get("word") == word
    ]
    if not original_orders:
        raise ValueError(
            f"reference {reference_path} 的{_describe_reference_entry(entry)}"
            f"在源语料 {path} 第 {line_no} 行的完整关注词序列里找不到目标 "
            f"(index={index}, word={word!r})"
        )
    pruned[VERDICT_FOCUS_ORDER_KEY] = original_orders[0]

    focuses = st.pick_focus_words(pruned)
    if (
        len(focuses) != 1
        or focuses[0].get("index") != index
        or focuses[0].get("word") != word
    ):
        raise ValueError(
            f"reference {reference_path} 的{_describe_reference_entry(entry)}"
            f"裁剪后关注词不是唯一目标 (index={index}, word={word!r})；"
            "请检查 match_tree 与 sampled_words/词表"
        )
    return pruned


def _iter_aligned(
    path: Path,
    reference_path: Path,
    limit: int | None,
    stats: dict,
    reference_level: str = "record",
) -> Iterator[tuple[int, dict]]:
    if reference_level not in REFERENCE_LEVELS:
        raise ValueError(f"reference_level 必须是 {REFERENCE_LEVELS} 之一，收到 {reference_level!r}")
    verdict_level = reference_level == "verdict"
    entries, reference_stats = _load_reference(reference_path, reference_level)
    stats["reference_file"] = str(reference_path)
    stats["reference_level"] = reference_level
    stats["reference_rows"] = reference_stats["rows"]
    stats["reference_blank"] = reference_stats["blank"]
    stats["reference_duplicates"] = reference_stats["duplicates"]
    reference_record_keys = {
        key
        for entry in entries
        if (key := _reference_identity(entry)) is not None
    }
    stats["reference_records"] = len(reference_record_keys)
    stats["matched_reference_records"] = 0
    stats["unmatched_reference_records"] = len(reference_record_keys)
    stats["matched_source_records"] = 0
    stats["content_matches"] = 0
    stats["line_collisions"] = 0
    stats["limit_excluded"] = 0
    stats["key_counts"] = {}
    if verdict_level:
        stats["reference_verdicts"] = len(entries)
        stats["matched_reference_verdicts"] = 0
        stats["unmatched_reference_verdicts"] = len(entries)
        stats["expanded_records"] = 0
        stats["duplicate_matched_verdicts"] = 0

    by_source_line, by_current_line, by_hash, by_sid = _reference_indexes(entries)
    snapshot = _fingerprint(path)
    selected = 0
    matched_record_keys: set[tuple] = set()
    matched_verdict_keys: set[tuple[int, tuple[int, str]]] = set()
    observations: dict[int, dict[str, dict]] = {}
    skips: dict[int, dict] = {}

    for line_no, row, source_info in _iter_aligned_source(path, stats, reference_level):
        source_lines = _record_line_keys(row)
        source_hash = _record_content_hash(row)
        source_document = _record_document(row)
        source_sid = _explicit_sentence_id(row)

        source_namespace = set(source_lines.get("source_line_no", ()))
        if not source_namespace:
            # 源语料本身没有 source_line_no 时，物理行号就是它的源行号。
            source_namespace = {line_no}
        current_namespace = {line_no}
        for label in ("input_line_no", "line_no"):
            current_namespace.update(source_lines.get(label, ()))

        source_hits: set[int] = set()
        for value in source_namespace:
            source_hits.update(by_source_line.get(value, set()))
        current_hits: set[int] = set()
        for value in current_namespace:
            current_hits.update(by_current_line.get(value, set()))
        sid_hits = by_sid.get(source_sid, set()) if source_sid else set()
        hash_hits = by_hash.get(source_hash, set()) if source_hash else set()

        # 先记录所有命中，再决定能否进入匹配：被状态过滤掉的行也要能解释“为什么没对齐”。
        for index in source_hits | current_hits | sid_hits:
            _remember_observation(
                observations, index,
                kind="line",
                line_no=line_no,
                source_hash=source_hash,
                source_document=source_document,
                source_sid=source_sid,
                status=source_info["status"],
                eligible=source_info["eligible"],
                usable=source_info["usable"],
                mismatches=_reference_mismatches(
                    entries[index], source_hash, source_document, source_sid
                ),
            )
        for index in hash_hits:
            _remember_observation(
                observations, index,
                kind="content",
                line_no=line_no,
                source_hash=source_hash,
                source_document=source_document,
                source_sid=source_sid,
                status=source_info["status"],
                eligible=source_info["eligible"],
                usable=source_info["usable"],
                mismatches=_reference_mismatches(
                    entries[index], source_hash, source_document, source_sid
                ),
            )

        if not source_info["eligible"] or not source_info["usable"]:
            continue

        matched: list[int] = []
        matched_by: str | None = None
        source_good: list[int] = []
        current_good: list[int] = []
        for index in source_hits:
            entry = entries[index]
            if _reference_entry_matches(entry, source_hash, source_document, source_sid):
                source_good.append(index)
            else:
                stats["line_collisions"] += 1
        for index in current_hits | sid_hits:
            entry = entries[index]
            if _reference_entry_matches(entry, source_hash, source_document, source_sid):
                current_good.append(index)
            else:
                stats["line_collisions"] += 1

        if source_good and current_good and set(source_good) != set(current_good):
            examples = "；".join(
                _describe_reference_entry(entries[index]) for index in sorted(set(source_good) | set(current_good))[:5]
            )
            raise ValueError(
                f"源语料 {path} 第 {line_no} 行被 reference 的源行号与当前文件行号指向不同记录：{examples}"
            )
        if source_good:
            matched = source_good
            matched_by = "source_line"
        elif current_good:
            matched = current_good
            matched_by = "sentence_id" if any(index in sid_hits for index in current_good) else "line"

        # 内容兜底只用于 reference 自己标了“当前文件行号”或本来就没有行号的记录；
        # 有 source_line_no 的权威记录行号对不上就不做内容兜底。
        # verdict 模式下，同一句可能一部分 verdict 走行号、另一部分只能走内容；
        # 此时不能在已有行号命中后跳过内容兜底，否则合法的内容型 verdict 会被漏配。
        content_matched: set[int] = set()
        if source_hash and (not matched or verdict_level):
            before = len(matched)
            for index in by_hash.get(source_hash, set()):
                if index in matched:
                    continue
                entry = entries[index]
                if not entry.allow_content_fallback:
                    continue
                if entry.matched and not verdict_level:
                    if entry.matched_by == "content":
                        raise ValueError(
                            f"reference {reference_path} 的{_describe_reference_entry(entry)}"
                            f"只能靠句子内容对齐，但源语料 {path} 里有多条同文记录；无法逐条对应"
                        )
                    continue
                if _reference_entry_matches(entry, source_hash, source_document, source_sid):
                    matched.append(index)
                    content_matched.add(index)
            if len(matched) > before:
                stats["content_matches"] += 1
                if matched_by is None:
                    matched_by = "content"

        if not matched:
            continue

        matched = list(dict.fromkeys(matched))
        if verdict_level:
            matched.sort(
                key=lambda index: (
                    entries[index].target_index if entries[index].target_index is not None else -1,
                    entries[index].target_word or "",
                    entries[index].row_no,
                )
            )
        matched_entries = [entries[index] for index in matched]
        if not verdict_level and len(matched_entries) > 1:
            examples = "；".join(_describe_reference_entry(entry) for entry in matched_entries[:5])
            raise ValueError(
                f"源语料 {path} 第 {line_no} 行同时对上多条 reference 记录，无法逐条一一对应：{examples}"
            )

        if verdict_level:
            missing_targets = [entry for entry in matched_entries if entry.target_key is None]
            if missing_targets:
                examples = "；".join(_describe_reference_entry(entry) for entry in missing_targets[:5])
                raise ValueError(
                    f"reference {reference_path} 在 verdict 模式下有条目没有目标词位置：{examples}"
                )
            # 同一目标可能同时以“带行号”和“仅句子内容”两种身份出现；它们指向
            # 同一个源 verdict，只能展开一次，否则 candidate_key/id 会重复。
            groups: dict[tuple[int, str], list[int]] = {}
            for index in matched:
                groups.setdefault(entries[index].target_key, []).append(index)
            ordered_groups: list[tuple[tuple[int, str], list[int], int]] = []
            for target_key, indices in groups.items():
                indices.sort(
                    key=lambda index: (
                        index in content_matched,
                        entries[index].row_no,
                    )
                )
                ordered_groups.append((target_key, indices, indices[0]))
            ordered_groups.sort(
                key=lambda item: (
                    entries[item[2]].target_index if entries[item[2]].target_index is not None else -1,
                    entries[item[2]].target_word or "",
                    entries[item[2]].row_no,
                )
            )

            any_success = False
            for target_key, group_indices, primary_index in ordered_groups:
                primary = entries[primary_index]
                try:
                    output_row = _prune_source_to_verdict(
                        row, line_no, primary, reference_path, path
                    )
                except _SkipVerdict as exc:
                    extra = f" (index={primary.target_index}, word={primary.target_word!r})"
                    current_observation = {
                        "kind": "line",
                        "line_no": line_no,
                        "source_hash": source_hash,
                        "source_document": source_document,
                        "source_sid": source_sid,
                        "status": source_info["status"],
                        "eligible": source_info["eligible"],
                        "usable": source_info["usable"],
                        "mismatches": _reference_mismatches(
                            primary, source_hash, source_document, source_sid
                        ),
                    }
                    for index in group_indices:
                        skips[index] = {
                            "reason": exc.reason,
                            "status": exc.status,
                            "status_scope": "record" if exc.reason == "screening_invalid" else "verdict",
                            "observation": current_observation,
                            "extra": extra,
                        }
                    continue

                any_success = True
                for index in group_indices:
                    entry = entries[index]
                    if entry.match_count:
                        raise ValueError(
                            f"reference {reference_path} 的{_describe_reference_entry(entry)}"
                            f"对到了多条可用源记录（又遇到 {path} 第 {line_no} 行）；"
                            "请给 reference 补上唯一行号，或确认源语料是否有重复行号"
                        )
                    entry.match_count += 1
                    if not entry.matched:
                        reason = "content" if index in content_matched else (matched_by or "content")
                        entry.matched = True
                        entry.matched_by = reason
                        record_key = _reference_identity(entry)
                        if record_key is not None:
                            matched_record_keys.add(record_key)
                        stats["key_counts"][reason] = stats["key_counts"].get(reason, 0) + 1
                matched_verdict_keys.add((line_no, target_key))
                stats["duplicate_matched_verdicts"] += len(group_indices) - 1
                if limit is not None and selected >= limit:
                    stats["limit_excluded"] += 1
                    continue
                selected += 1
                _remember_selected(stats, line_no)
                stats["expanded_records"] += 1
                yield line_no, output_row

            if any_success:
                stats["matched_source_records"] += 1
            continue

        for index, entry in zip(matched, matched_entries):
            if entry.match_count:
                raise ValueError(
                    f"reference {reference_path} 的{_describe_reference_entry(entry)}"
                    f"对到了多条源记录（又遇到 {path} 第 {line_no} 行）；"
                    "请给 reference 补上唯一行号，或确认源语料是否有重复行号"
                )
            entry.match_count += 1
            if not entry.matched:
                reason = "content" if index in content_matched else (matched_by or "content")
                entry.matched = True
                entry.matched_by = reason
                record_key = _reference_identity(entry)
                if record_key is not None:
                    matched_record_keys.add(record_key)
                stats["key_counts"][reason] = stats["key_counts"].get(reason, 0) + 1

        stats["matched_source_records"] += 1
        for _entry in matched_entries:
            if limit is not None and selected >= limit:
                stats["limit_excluded"] += 1
                continue
            selected += 1
            _remember_selected(stats, line_no)
            yield line_no, row

    if _fingerprint(path) != snapshot:
        raise RuntimeError(f"{path} 在按 reference 对齐的读取过程中发生变化；请固定语料后重跑")

    stats["matched_reference_records"] = len(matched_record_keys)
    if verdict_level:
        stats["matched_reference_verdicts"] = len(matched_verdict_keys)
        stats["available"] = stats["matched_reference_verdicts"]
        stats["candidates"] = stats["matched_reference_verdicts"]
        stats["expanded_records"] = selected
    else:
        stats["available"] = stats["matched_source_records"]
        stats["candidates"] = stats["matched_source_records"]

    unmatched_indices = [
        index for index, entry in enumerate(entries) if not entry.matched
    ]
    unmatched_record_keys = reference_record_keys - matched_record_keys
    stats["unmatched_reference_records"] = len(unmatched_record_keys)
    if verdict_level:
        stats["unmatched_reference_verdicts"] = len(unmatched_indices)
    details, reason_counts, status_counts = _build_unmatched_details(
        entries, unmatched_indices, observations, skips
    )
    stats["unmatched_details"] = details
    stats["unmatched_reason_counts"] = reason_counts
    stats["unmatched_status_counts"] = status_counts


def select_aligned_records(
    path: str | Path,
    *,
    reference_file: str | Path,
    limit: int | None = None,
    corpus_type: str = st.DEFAULT_CORPUS_TYPE,
    reference_level: str = "record",
) -> Selection:
    """按已筛选文件从源语料中逐条取回对应记录。

    Args:
        path: 未筛选的源语料 JSONL
        reference_file: 已筛选文件 JSONL；支持 word_sampler 产物、replace 句对、annotate 结果或同源子集
        limit: 可选上限；None 表示 reference 能对上的记录全部取出，否则只取源文件顺序中的前 X 条。
            verdict 级表示最多展开多少个 verdict。
        corpus_type: 仅用于统计字段，不参与对齐
        reference_level: ``record``（默认）按整条源记录对齐；``verdict`` 按单个目标词
            verdict 对齐，并把源记录裁剪成只含该目标位置的副本。

    Returns:
        Selection: 按源文件顺序产出匹配记录；找不到或状态不可用的 reference 项会跳过，
        其明细、原因和状态分布写入 ``stats["unmatched_details"]`` 等字段

    Raises:
        ValueError: reference 为空/有坏行/无身份/非法 target，或存在真正的对齐歧义
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError(f"limit 必须是大于 0 的整数或 None，收到 {limit!r}")
    if reference_level not in REFERENCE_LEVELS:
        raise ValueError(f"reference_level 必须是 {REFERENCE_LEVELS} 之一，收到 {reference_level!r}")
    reference = Path(reference_file)
    if not reference.is_file():
        raise FileNotFoundError(f"找不到 reference 文件 {reference}")
    stats = _base_stats(limit=limit, mode="aligned", seed=DEFAULT_SEED, corpus_type=corpus_type)
    records = _iter_aligned(Path(path), reference, limit, stats, reference_level)
    return Selection(records=records, stats=stats)


def select_records(
    path: str | Path,
    *,
    limit: int,
    mode: str = DEFAULT_MODE,
    seed: int = DEFAULT_SEED,
    corpus_type: str = st.DEFAULT_CORPUS_TYPE,
) -> Selection:
    """构造一次限额抽样；返回惰性 Selection，消费 records 后才完成统计。

    Args:
        path: 输入语料 JSONL
        limit: 正整数 X，表示本轮最多送入 replace 的可用原句数
        mode: ``sequential`` / ``random`` / ``proportional``
        seed: random 与 proportional 的随机种子；sequential 忽略但仍记录
        corpus_type: 比例分层用的关注词类别（方位词/趋向动词），已由入口规范化

    Raises:
        ValueError: limit/mode/seed 非法，或语料中出现坏 JSON/句对记录
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"limit 必须是大于 0 的整数，收到 {limit!r}")
    if mode not in SAMPLE_MODES:
        raise ValueError(
            f"limit_mode 必须是 {SAMPLE_MODES} 之一，收到 {mode!r}；"
            "aligned 模式请用 select_aligned_records 并传入 reference_file"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed 必须是大于等于 0 的整数，收到 {seed!r}")

    corpus = Path(path)
    stats = _base_stats(limit=limit, mode=mode, seed=seed, corpus_type=corpus_type)
    if mode == "sequential":
        records = _iter_sequential(corpus, limit, stats)
    elif mode == "random":
        records = _iter_random(corpus, limit, seed, stats)
    else:
        records = _iter_proportional(corpus, limit, seed, corpus_type, stats)
    return Selection(records=records, stats=stats)
