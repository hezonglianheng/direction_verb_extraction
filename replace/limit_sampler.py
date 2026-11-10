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

LIMIT_MODES = ("sequential", "random", "proportional")
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


def _base_stats(*, limit: int, mode: str, seed: int, corpus_type: str) -> dict:
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
    if mode not in LIMIT_MODES:
        raise ValueError(f"limit_mode 必须是 {LIMIT_MODES} 之一，收到 {mode!r}")
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
