# encoding: utf8
"""限额筛选/抽样共用的通用函数（replace 与 annotate 两个入口共用）。

这里只放与具体阶段无关、只依赖标准库的件：JSON 行号/内容身份、文件快照指纹、
最大余数配额、随机抽样的统一统计壳。**不导入** extract/replace/annotate 的业务模块，
避免 import 环，也让 replace 的既有抽样链路保持轻量。

各阶段的“哪些记录算可用”“按什么分层”仍留在自己的 sampler 里：
replace 用原句筛选状态与主关注词，annotate 用合格句对（两侧检查通过）与无向词对。
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

SELECTION_VERSION = 1
"""抽样统计口径版本；算法变化时递增，便于日志里的旧结果对账。"""

SAMPLE_MODES = ("sequential", "random", "proportional")
LIMIT_MODES = (*SAMPLE_MODES, "aligned")
DEFAULT_MODE = "sequential"
DEFAULT_SEED = 0

PREVIEW_SIZE = 20
"""统计里最多保留多少个选中行号，避免 X 很大时统计本身变成大文件。"""

ALIGNED_LINE_KEYS = ("source_line_no", "input_line_no", "line_no")
"""reference/源记录里可参与对齐的行号字段；source_line_no 最权威。"""


@dataclass
class Selection:
    """一次限额抽样的结果。

    ``records`` 是惰性迭代器：真正读取发生在调用方消费时（random/proportional 会先做
    第一遍计数）。``stats`` 也在消费过程中补齐，全部消费完后再写日志。
    """

    records: Iterator[tuple[int, dict]]
    stats: dict


def fingerprint(path: str | Path) -> tuple[int, int, int]:
    """语料快照的大小/mtime/inode；两遍读取前后一致才允许采用抽样结果。"""
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def normalize(text: str) -> str:
    """规范化句子文本，用于查表、去重和比较

    统一全角/半角、去掉所有空白，这样 "软件上 说" 与 "软件上说" 会落到同一个键上。
    """
    return "".join(unicodedata.normalize("NFKC", text).split())


def content_hash(text: str | Iterable[str]) -> str:
    """句子文本指纹（查重用）：先规范化，再取 sha1 的前 12 位

    参数是分词切片或拼好的句子都行，只回答「这两句文本是不是一样」，不含来源信息。
    """
    body = text if isinstance(text, str) else "".join(text)
    return hashlib.sha1(normalize(body).encode("utf-8")).hexdigest()[:12]


def positive_line(value) -> int | None:
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


def record_line_keys(record: dict) -> dict[str, set[int]]:
    """一条记录里所有可用行号，按字段名分组（顶层与 source 块都看）。

    不在这里决定哪个行号优先：两者可能分别指向祖先语料和当前输入，究竟哪个能对回
    源文件取决于用户拿的是哪一份。全部保留，后续再用句子内容校验；只有内容一致的
    命中才算数，避免同号不同句的误对齐。
    """
    found: dict[str, set[int]] = {}
    source = record.get("source")
    containers = [("source", source), ("top", record)] if isinstance(source, dict) else [("top", record)]
    for _, container in containers:
        if not isinstance(container, dict):
            continue
        for key in ALIGNED_LINE_KEYS:
            value = positive_line(container.get(key))
            if value is not None:
                found.setdefault(key, set()).add(value)
    return found


def record_content_hash(record: dict) -> str | None:
    """reference/source 的句子指纹：顶层 sentence，其次句对的 a 侧切片/句子。"""
    tokens = record.get("sentence")
    if isinstance(tokens, list) and tokens and all(isinstance(token, str) for token in tokens):
        return content_hash(tokens)
    if isinstance(tokens, str) and tokens.strip():
        return content_hash(tokens)
    pair = record.get("pair")
    if isinstance(pair, dict):
        side_a = pair.get("a")
        if isinstance(side_a, dict):
            tokens = side_a.get("slice")
            if isinstance(tokens, list) and tokens and all(isinstance(token, str) for token in tokens):
                return content_hash(tokens)
            sentence = side_a.get("sentence")
            if isinstance(sentence, str) and sentence.strip():
                return content_hash(sentence)
    explicit = record.get("content_hash")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    return None


def explicit_sentence_id(record: dict) -> str | None:
    value = record.get("sentence_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def allocate_quotas(counts: dict[str, int], total_quota: int) -> dict[str, int]:
    """按最大余数法把 total_quota 个名额分给各层，返回每层的名额（含 0）。

    约束：0 <= quota[word] <= counts[word]，sum(quota.values()) == min(total_quota, sum(counts))。
    余数并列时按键的字典序破平局，保证不同机器/不同进程下配额完全一致。
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


def remember_selected(stats: dict, line_no: int) -> None:
    stats["selected"] += 1
    if len(stats["selected_line_nos"]) < PREVIEW_SIZE:
        stats["selected_line_nos"].append(line_no)
