# encoding: utf8

"""方位词后缀切分

LTP 偶尔把「名词+方位词」并成一个 token 并标成 n/nl（实测如 树上/墙上/门外/门前/门旁），
这会让 sentenceFilters 里按 CWS 精确匹配的方位词规则静默漏召——词表里的「上」对不上
「树上」这个 token，整条规则不会命中。

本模块只负责「把这种 token 按后缀切开」这一件事：不依赖 LTP，也不依赖
sentenceTable / sentenceFilters（后者会与 sentenceTable 形成循环导入），因此可以离线单测。

后缀词表与筛选规则共用同一份 config_files/localization_vocab.json5，并按
config.LOCATIVE_SPLIT_SUFFIX_EXCLUDE 剔除高风险单字后缀——这样「筛选用的方位词」
与「切分用的后缀」不会各读一份文件而悄悄漂移。
"""

import threading
from pathlib import Path
from typing import Iterable, Sequence

import json5

import config

_suffix_vocab: tuple[str, ...] | None = None
_lock = threading.Lock()


def _resolve_vocab_path(path: str | Path) -> Path:
    """解析词表路径，候选顺序与 sentenceFilters.load_vocab_file 保持一致

    两处必须落到同一个文件上，否则筛选与切分看的不是同一份方位词表。
    """
    candidates: list[Path] = []
    if Path(path).is_absolute():
        candidates.append(Path(path))
    else:
        candidates.append(Path(config._PROJECT_ROOT) / path)
        candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"找不到方位词词表 {path}，已尝试：{[str(c) for c in candidates]}")


def get_suffix_vocab() -> tuple[str, ...]:
    """可切分的后缀词表，按长度降序排列（供最长后缀匹配用）

    进程内惰性加载，线程安全；剔除 config.LOCATIVE_SPLIT_SUFFIX_EXCLUDE 里的条目。
    排序键取 (-长度, 词形) 而非仅按长度，是为了让同长度的词有稳定次序——
    否则 set 的迭代顺序受 PYTHONHASHSEED 影响，同一份输入在不同进程会得到不同的中间态。
    """
    global _suffix_vocab
    if _suffix_vocab is None:
        with _lock:
            if _suffix_vocab is None:
                path = _resolve_vocab_path(config.LOCATIVE_SPLIT_VOCAB_FILE)
                with open(path, "r", encoding="utf-8") as f:
                    words = json5.load(f)
                if not isinstance(words, list):
                    raise ValueError(
                        f"方位词词表 {path} 的内容必须是词条列表（JSON5 数组），"
                        f"实际是 {type(words).__name__}"
                    )
                exclude = set(config.LOCATIVE_SPLIT_SUFFIX_EXCLUDE)
                _suffix_vocab = tuple(
                    sorted((w for w in set(words) if w not in exclude), key=lambda w: (-len(w), w))
                )
    return _suffix_vocab


def reset_cache() -> None:
    """清空后缀索引缓存

    仅供测试：改了 config 里的词表路径或排除集之后，用它强制重新加载。
    """
    global _suffix_vocab
    with _lock:
        _suffix_vocab = None


def split_tokens_by_suffix(
    words: Sequence[str],
    pos_tags: Sequence[str],
    suffix_vocab: Sequence[str],
    allowed_pos: Iterable[str],
    token_blacklist: Iterable[str] = frozenset(),
) -> list[str] | None:
    """把「名词+方位词」合并成的 token 按后缀切开

    切分不改变字符序列（只插入切点），所以 "".join(结果) == "".join(words)。

    以下情况一律不切：
      - 词性不在 allowed_pos（例如 下午/nt、东边/nd、这里/r）；
      - token 长度 < 2（「上」本身就是一个词）；
      - token 整体命中 token_blacklist（词典化词，如 东西/海外）；
      - 没有词表项能作为它的真后缀。

    suffix_vocab 必须是按长度降序的序列，这样第一个命中的就是最长后缀——
    「桌子上面」切成 ["桌子", "上面"]，而不是 ["桌子", "上", "面"]。
    判定用 len(suffix) < len(word)，它同时保证前缀非空、且 token 恰等于词表项时不切。

    Args:
        words (Sequence[str]): 原 token 序列
        pos_tags (Sequence[str]): 与 words 对齐的词性
        suffix_vocab (Sequence[str]): 按长度降序的候选后缀
        allowed_pos (Iterable[str]): 只有这些词性的 token 才切
        token_blacklist (Iterable[str]): 整词命中即跳过

    Returns:
        list[str] | None: 切分后的新 token 序列；**没有任何变化时返回 None**，
            调用方据此零成本跳过（不必比对两个列表）。
    """
    allowed = set(allowed_pos)
    blacklist = set(token_blacklist)
    changed = False
    result: list[str] = []

    for word, pos in zip(words, pos_tags):
        if pos not in allowed or len(word) < 2 or word in blacklist:
            result.append(word)
            continue

        suffix = next(
            (v for v in suffix_vocab if len(v) < len(word) and word.endswith(v)),
            None,
        )
        if suffix is None:
            result.append(word)
            continue

        result.append(word[: len(word) - len(suffix)])
        result.append(suffix)
        changed = True

    return result if changed else None
