# encoding: utf8

"""文本 → 句子：在 StnSplit 的句子切分之外，再按各种空格/不可见分隔符切分。

为什么要有这一层（而不是在两个入口各写一份 ``StnSplit().split()``）：
    entry_in_linux / entry_in_win 原本各有一份 ``get_sentences_list``，两份只差平台，
    切句口径却必须一致——否则同一份原始语料在 Windows 与 Linux 上会产出不同的句子集合，
    后续筛选与语料 id 都对不上。切分规则（尤其是新增的空白符集合）收在这里一处真源。

切分顺序：
    1. ``ltp.StnSplit.split`` 按句末标点切；
    2. 每一段再按 ``WHITESPACE_RE`` 切开。``\\s`` 已覆盖普通空格、制表、换行、回车、
       垂直制表、换页；下面再把 Unicode 里各类"看起来是空白但不被 ``\\s`` 认"的字符补上：
       不换行空格、窄不换行空格、表意空格、零宽空格/连接符、BOM、蒙古文元音分隔符等。
    3. 逐段 strip；空段丢弃。

长度过滤放在 strip 之后：旧实现先按含首尾空白的长度过滤再 strip，一段刚好卡在长度下界
的句子会被首尾空白"撑"过去或"撑"出界，口径不稳定。先 strip 再判长度，结果与空白无关。

这一层不 import sentenceTable / torch，只 import ltp（StnSplit 是原生扩展，本身很轻）。
"""

import re
import threading
from typing import Iterable

from ltp import StnSplit

import config

WHITESPACE_RE = re.compile(
    "["
    r"\s"  # 普通空白：空格、\t、\n、\r、\v、\f
    "\u00a0"  # NO-BREAK SPACE
    "\u1680"  # OGHAM SPACE MARK
    "\u180e"  # MONGOLIAN VOWEL SEPARATOR
    "\u2000-\u200a"  # EN QUAD ... HAIR SPACE
    "\u2028\u2029"  # LINE / PARAGRAPH SEPARATOR
    "\u202f"  # NARROW NO-BREAK SPACE
    "\u205f"  # MEDIUM MATHEMATICAL SPACE
    "\u2060"  # WORD JOINER（常被当成零宽空格）
    "\u3000"  # IDEOGRAPHIC SPACE（中文全角空格）
    "\u200b-\u200d"  # ZERO WIDTH SPACE / NON-JOINER / JOINER
    "\ufeff"  # BOM（常出现在文本拼接处）
    "]+"
)
"""切分用的空白/不可见分隔符集合。"""

_THREAD_LOCAL = threading.local()
"""每个线程一份 StnSplit：切句可能在抽取线程池里并发调用

StnSplit 是原生扩展对象；共享一份要赌它无内部状态。每线程一份的代价可以忽略（构造很轻），
换来的是并发切句不依赖对第三方实现的线程安全假设。
"""


def _splitter() -> StnSplit:
    splitter = getattr(_THREAD_LOCAL, "splitter", None)
    if splitter is None:
        splitter = StnSplit()
        _THREAD_LOCAL.splitter = splitter
    return splitter


def filter_by_vocab(sentences: Iterable[str], vocabs: Iterable[str] | None) -> list[str]:
    """按初筛词表做子串筛选；词表为空/None 时原样返回。

    入口把词表初筛从切句工作线程挪到消费线程（为了统计「初筛前读入总数」）后，
    必须和 ``split_sentences`` 共用同一份匹配口径，避免两处子串判断慢慢漂移。
    """
    vocab_list = list(vocabs or ())
    if not vocab_list:
        return list(sentences)
    return [sentence for sentence in sentences if any(vocab in sentence for vocab in vocab_list)]


def split_sentences(
    article: str,
    vocabs: Iterable[str] | None = None,
    *,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
) -> list[str]:
    """把一篇文本切成句子：先按句末标点，再按各种空格/不可见分隔符。

    Args:
        article: 原始文本（语料 jsonl 的 ``content`` 字段）。
        vocabs: 初筛词表；给了就只保留至少包含其中一个词（子串）的片段。
        lower_bound: 句子长度下界（字符数），默认 ``config.SENTENCE_LEN_LOWER_BOUND``。
        upper_bound: 句子长度上界（字符数），默认 ``config.SENTENCE_LEN_UPPER_BOUND``。

    Returns:
        list[str]: 已 strip、已过滤空串、已按长度/中文内容/占位符/词表过滤的句子。
    """
    lower = config.SENTENCE_LEN_LOWER_BOUND if lower_bound is None else lower_bound
    upper = config.SENTENCE_LEN_UPPER_BOUND if upper_bound is None else upper_bound

    pieces: list[str] = []
    for chunk in _splitter().split(article or ""):
        # 句末标点切出来的段落里仍可能夹着换行、全角空格、制表等：每一段都要再切
        pieces.extend(WHITESPACE_RE.split(chunk))

    sentences: list[str] = []
    for piece in pieces:
        sentence = piece.strip()
        if not sentence:
            continue
        if not (lower <= len(sentence) <= upper):
            continue
        # 与旧口径一致：纯 ASCII 或含汉字才留下
        if not (sentence.isascii() or re.search(r"[\u4e00-\u9fff]", sentence)):
            continue
        if "【【【缺文】】】" in sentence:
            continue
        sentences.append(sentence)
    return filter_by_vocab(sentences, vocabs)
