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
   给了则只取源文件顺序中的前 X 条匹配记录。这是严格对齐：reference 中任何一条在源
   语料里找不到、或一条 reference 对到多条内容不同的源记录，都会报错而不是静默跳过。
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

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from extract import sentence_screening as ss
from annotate import spatial_tools as st

SELECTION_VERSION = 1
"""抽样统计口径版本；算法变化时递增，便于日志里的旧结果对账。"""

SAMPLE_MODES = ("sequential", "random", "proportional")
LIMIT_MODES = (*SAMPLE_MODES, "aligned")
DEFAULT_MODE = "sequential"
DEFAULT_SEED = 0

PREVIEW_SIZE = 20
"""日志里最多保留多少个选中行号，避免 X 很大时日志本身变成大文件。"""


@dataclass
class Selection:
    """一次限额抽样的结果。

    ``records`` 是惰性迭代器：真正读取发生在调用方消费时（random/proportional 会先做
    第一遍计数）。``stats`` 也在消费过程中补齐，全部消费完后再写日志。
    """

    records: Iterator[tuple[int, dict]]
    stats: dict


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


def _remember_selected(stats: dict, line_no: int) -> None:
    stats["selected"] += 1
    if len(stats["selected_line_nos"]) < PREVIEW_SIZE:
        stats["selected_line_nos"].append(line_no)


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


def _fingerprint(path: Path) -> tuple[int, int, int]:
    """语料快照的大小/mtime/inode；两遍读取前后一致才允许采用抽样结果。"""
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


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


def allocate_quotas(counts: dict[str, int], total_quota: int) -> dict[str, int]:
    """按最大余数法把 total_quota 个名额分给各词，返回每个词的名额（含 0）。

    约束：0 <= quota[word] <= counts[word]，sum(quota.values()) == min(total_quota, sum(counts))。
    余数并列时按词形字典序破平局，保证不同机器/不同进程下配额完全一致。
    """
    if not counts:
        return {}
    available = sum(counts.values())
    quota = {word: 0 for word in counts}
    if available <= 0 or total_quota <= 0:
        return quota
    if total_quota >= available:
        return {word: int(count) for word, count in counts.items()}

    remainders: list[tuple[int, str]] = []
    for word, count in counts.items():
        # 整数除法取商与余数：余数越大越该补 1，全程精确、无浮点误差。
        base, remainder = divmod(count * total_quota, available)
        quota[word] = base
        remainders.append((remainder, word))

    left = total_quota - sum(quota.values())
    for _, word in sorted(remainders, key=lambda item: (-item[0], item[1]))[:left]:
        quota[word] += 1
    return quota


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

_ALIGNED_LINE_KEYS = ("source_line_no", "input_line_no", "line_no")


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


def _positive_line(value) -> int | None:
    """把行号字段收成正整数；bool、0、负数、非数字一律不认。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
        return number if number > 0 else None
    return None


def _record_line_keys(record: dict) -> dict[str, set[int]]:
    """一条记录里所有可用行号，按字段名分组（顶层与 source 块都看）。

    不在这里决定“source_line_no 优先还是 line_no 优先”：两者可能分别指向祖先语料和
    当前输入，究竟哪个能对回源文件取决于用户拿的是哪一份。全部保留，后续用句子内容
    校验；只有内容一致的命中才算数，避免同号不同句的误对齐。
    """
    found: dict[str, set[int]] = {}
    source = record.get("source")
    containers = [("source", source), ("top", record)] if isinstance(source, dict) else [("top", record)]
    for _, container in containers:
        if not isinstance(container, dict):
            continue
        for key in _ALIGNED_LINE_KEYS:
            value = _positive_line(container.get(key))
            if value is not None:
                found.setdefault(key, set()).add(value)
    return found


def _record_content_hash(record: dict) -> str | None:
    """reference/source 的句子指纹：顶层 sentence，其次句对的 a 侧切片/句子。"""
    tokens = record.get("sentence")
    if isinstance(tokens, list) and tokens and all(isinstance(token, str) for token in tokens):
        return st.content_hash(tokens)
    if isinstance(tokens, str) and tokens.strip():
        return st.content_hash(tokens)
    pair = record.get("pair")
    if isinstance(pair, dict):
        side_a = pair.get("a")
        if isinstance(side_a, dict):
            tokens = side_a.get("slice")
            if isinstance(tokens, list) and tokens and all(isinstance(token, str) for token in tokens):
                return st.content_hash(tokens)
            sentence = side_a.get("sentence")
            if isinstance(sentence, str) and sentence.strip():
                return st.content_hash(sentence)
    explicit = record.get("content_hash")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return None


def _record_document(record: dict) -> str | None:
    """来源文档：先看 source 块，再看顶层；与 spatial_tools.source_document_id 同源。"""
    source = record.get("source")
    if isinstance(source, dict):
        document = st.source_document_id(source)
        if document:
            return document
    return st.source_document_id(record)


def _explicit_sentence_id(record: dict) -> str | None:
    value = record.get("sentence_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


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


def _reference_identity(entry: _ReferenceEntry) -> tuple | None:
    """去重身份：显式 sentence_id 最稳，其次行号集合+内容，最后仅内容。"""
    if entry.sentence_id:
        return ("sentence_id", entry.sentence_id)
    line_pairs = frozenset(
        (label, value) for label, values in entry.line_keys.items() for value in values
    )
    if line_pairs:
        return ("lines", entry.document, entry.content_hash, line_pairs)
    if entry.content_hash:
        return ("content", entry.document, entry.content_hash)
    return None


def _reference_entry_matches(
    entry: _ReferenceEntry, source_hash: str | None, source_document: str | None, source_sid: str | None
) -> bool:
    """行号命中后还要过内容/来源/ID 校验；有内容的记录不能用同号不同句冒充。"""
    if entry.content_hash and entry.content_hash != source_hash:
        return False
    if entry.document and entry.document != source_document:
        return False
    if entry.sentence_id and source_sid and entry.sentence_id != source_sid:
        return False
    return True


def _describe_reference_entry(entry: _ReferenceEntry) -> str:
    parts = []
    for label, values in sorted(entry.line_keys.items()):
        parts.append(f"{label}={sorted(values)}")
    if entry.content_hash:
        parts.append(f"hash={entry.content_hash}")
    if entry.sentence_id:
        parts.append(f"sentence_id={entry.sentence_id}")
    return f"第 {entry.row_no} 行（{'，'.join(parts) or '无可用键'}）"


def _load_reference(path: Path) -> tuple[list[_ReferenceEntry], dict]:
    """读 reference，按身份去重；行号/内容/ID 都没有的记录直接报错。"""
    entries: list[_ReferenceEntry] = []
    identities: dict[tuple, int] = {}
    stats = {"rows": 0, "blank": 0, "duplicates": 0}
    with open(path, "r", encoding="utf-8") as reader:
        for row_no, line in enumerate(reader, start=1):
            if not line.strip():
                stats["blank"] += 1
                continue
            row = _parse_reference_row(line, row_no, path)
            stats["rows"] += 1
            line_keys = _record_line_keys(row)
            entry = _ReferenceEntry(
                line_keys=line_keys,
                content_hash=_record_content_hash(row),
                document=_record_document(row),
                sentence_id=_explicit_sentence_id(row),
                row_no=row_no,
                # source_line_no 是源语料的权威行号，不允许再靠内容猜；
                # 只有“没有源行号”的 annotate/内容型 reference 才允许唯一内容兜底。
                allow_content_fallback=(
                    not line_keys
                    or (
                        not line_keys.get("source_line_no")
                        and _looks_like_current_file_reference(row)
                    )
                ),
            )
            identity = _reference_identity(entry)
            if identity is None:
                raise ValueError(
                    f"reference {path} 第 {row_no} 行没有任何可用于对齐的行号、句子内容或 sentence_id；"
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


def _iter_aligned(
    path: Path, reference_path: Path, limit: int | None, stats: dict
) -> Iterator[tuple[int, dict]]:
    entries, reference_stats = _load_reference(reference_path)
    stats["reference_file"] = str(reference_path)
    stats["reference_rows"] = reference_stats["rows"]
    stats["reference_blank"] = reference_stats["blank"]
    stats["reference_duplicates"] = reference_stats["duplicates"]
    stats["reference_records"] = len(entries)
    stats["matched_reference_records"] = 0
    stats["unmatched_reference_records"] = len(entries)
    stats["matched_source_records"] = 0
    stats["content_matches"] = 0
    stats["line_collisions"] = 0
    stats["limit_excluded"] = 0
    stats["key_counts"] = {}

    by_source_line, by_current_line, by_hash, by_sid = _reference_indexes(entries)
    snapshot = _fingerprint(path)
    selected = 0
    for line_no, row in _iter_candidates(path, stats):
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
        # 有 source_line_no 的权威记录行号对不上就保持未匹配，最终报错。
        if not matched and source_hash:
            for index in by_hash.get(source_hash, set()):
                entry = entries[index]
                if not entry.allow_content_fallback:
                    continue
                if entry.matched:
                    if entry.matched_by == "content":
                        raise ValueError(
                            f"reference {reference_path} 的{_describe_reference_entry(entry)}"
                            f"只能靠句子内容对齐，但源语料 {path} 里有多条同文记录；无法逐条对应"
                        )
                    continue
                if _reference_entry_matches(entry, source_hash, source_document, source_sid):
                    matched.append(index)
            if matched:
                matched_by = "content"
                stats["content_matches"] += 1

        if not matched:
            continue

        matched_entries = [entries[index] for index in matched]
        if len(matched_entries) > 1:
            examples = "；".join(_describe_reference_entry(entry) for entry in matched_entries[:5])
            raise ValueError(
                f"源语料 {path} 第 {line_no} 行同时对上多条 reference 记录，无法逐条一一对应：{examples}"
            )

        for entry in matched_entries:
            if entry.match_count:
                raise ValueError(
                    f"reference {reference_path} 的{_describe_reference_entry(entry)}"
                    f"对到了多条源记录（又遇到 {path} 第 {line_no} 行）；"
                    "请给 reference 补上唯一行号，或确认源语料是否有重复行号"
                )
            entry.match_count += 1
            if not entry.matched:
                entry.matched = True
                entry.matched_by = matched_by
                stats["matched_reference_records"] += 1
                stats["key_counts"][matched_by] = stats["key_counts"].get(matched_by, 0) + 1

        stats["matched_source_records"] += 1
        if limit is None or selected < limit:
            selected += 1
            _remember_selected(stats, line_no)
            yield line_no, row
        else:
            stats["limit_excluded"] += 1

    if _fingerprint(path) != snapshot:
        raise RuntimeError(f"{path} 在按 reference 对齐的读取过程中发生变化；请固定语料后重跑")

    stats["available"] = stats["matched_source_records"]
    stats["candidates"] = stats["matched_source_records"]
    unmatched = [entry for entry in entries if not entry.matched]
    stats["unmatched_reference_records"] = len(unmatched)
    if unmatched:
        examples = "；".join(_describe_reference_entry(entry) for entry in unmatched[:5])
        raise ValueError(
            f"reference {reference_path} 有 {len(unmatched)} 条记录在源语料 {path} 中找不到唯一对应记录：{examples}"
        )


def select_aligned_records(
    path: str | Path,
    *,
    reference_file: str | Path,
    limit: int | None = None,
    corpus_type: str = st.DEFAULT_CORPUS_TYPE,
) -> Selection:
    """按已筛选文件从源语料中逐条取回对应记录。

    Args:
        path: 未筛选的源语料 JSONL
        reference_file: 已筛选文件 JSONL；支持 word_sampler 产物、replace 句对、annotate 结果或同源子集
        limit: 可选上限；None 表示 reference 能对上的记录全部取出，否则只取源文件顺序中的前 X 条
        corpus_type: 仅用于统计字段，不参与对齐

    Returns:
        Selection: 按源文件顺序产出匹配记录；reference 中任何一条对不上都会在消费完时报错

    Raises:
        ValueError: reference 为空/有坏行/无身份/与源语料无法一一对应
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError(f"limit 必须是大于 0 的整数或 None，收到 {limit!r}")
    reference = Path(reference_file)
    if not reference.is_file():
        raise FileNotFoundError(f"找不到 reference 文件 {reference}")
    stats = _base_stats(limit=limit, mode="aligned", seed=DEFAULT_SEED, corpus_type=corpus_type)
    records = _iter_aligned(Path(path), reference, limit, stats)
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
