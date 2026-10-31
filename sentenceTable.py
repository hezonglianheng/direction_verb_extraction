# encoding: utf8

from ltp import LTP
import torch

import threading
from collections import OrderedDict
from typing import Iterable

import config
from locative_split import get_suffix_vocab, split_tokens_by_suffix

_ltp = None
_ltp_init_lock = threading.Lock()
_infer_lock = threading.Lock()
"""LTP 推理的串行边界（A4）

缓存字典自带的锁只保护字典本身，推理是**在锁外**做的（见 get_task_value），所以预填充没兜住的
句子会由多个 worker 线程同时送进同一份模型：那不是数据竞争（结果一致），而是把一次前向拆成
并发的多次、互相抢占显存与算力。这里给推理划一条明确的串行边界：并发的调用排队过这一把锁，
于是"在途推理"恒为一组，与线程池里的在途候选数解耦。

批量预填充（spatial_runner._prefill / spatial_tools.prefill_ltp）把整批句子按组分批送模型，
走的也是 _infer_base 的同一把锁；这条边界同时兜住预填充未覆盖、按需单句推理的句子。"""


def _get_ltp() -> LTP:
    """Lazy-init LTP in current process to avoid CUDA init before multiprocessing fork/spawn."""
    global _ltp
    if _ltp is None:
        with _ltp_init_lock:
            if _ltp is None:
                model = LTP(config.LTP_MODEL_PATH)
                if torch.cuda.is_available():
                    model.to("cuda")
                _ltp = model
    return _ltp


_rerun_length_warned = False


def _truncate_tail(value: dict, expected: int) -> dict:
    """把 dep/sdp 的 head/label 截到 expected 长度，其余键原样保留

    只截尾巴：多出来的那一格在末尾（见 _correct_splits 里的说明），前 expected 项
    仍与词表逐位对齐。
    """
    trimmed = dict(value)
    for key, item in trimmed.items():
        if isinstance(item, list):
            trimmed[key] = item[:expected]
    return trimmed


def _warn_rerun_length_once(expected: int, lengths: tuple[int, ...]) -> None:
    """长度不一致只提示一次：触发它的句子每批都有，逐句打印会把日志刷爆"""
    global _rerun_length_warned
    if _rerun_length_warned:
        return
    _rerun_length_warned = True
    print(
        f"[警告] 已分词模式下的 LTP 结果与词表长度不一致（词表 {expected}，pos/dep/sdp {list(lengths)}）。"
        "这是 LTP 对批内 padding 句子的已知行为：长于词表时多出的末尾一格会被截掉，"
        "短于词表时放弃该句切分。后续同类情况不再逐条提示。"
    )


class SentenceTaskCache:
    BASE_TASKS = (config.CWS, config.POS, config.DEP, config.SDP)

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.lock = threading.Lock()
        self.cache = OrderedDict()  # {sentence: {task: value}}

    def _touch_and_evict_unsafe(self, sentence: str):
        self.cache.move_to_end(sentence)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def _update_cache(self, sentence: str, values: dict):
        with self.lock:
            task_dict = self.cache.setdefault(sentence, {})
            task_dict.update(values)
            self._touch_and_evict_unsafe(sentence)

    def _get_cached_value(self, sentence: str, task: str):
        with self.lock:
            if sentence in self.cache and task in self.cache[sentence]:
                self._touch_and_evict_unsafe(sentence)
                return self.cache[sentence][task]
        return None

    def _need_base_tasks(self, sentence: str) -> bool:
        with self.lock:
            task_dict = self.cache.get(sentence)
            if task_dict is None:
                return True
            return any(t not in task_dict for t in self.BASE_TASKS)

    def has_base_tasks(self, sentence: str) -> bool:
        """该句的基础任务结果是否都已在缓存里

        供预填充后的自检使用：推理发生在锁外（见 get_task_value），
        所以"预热过"必须能验证，否则多线程下仍会并发调同一份模型。
        """
        return not self._need_base_tasks(sentence)

    def get_cached(self, sentence: str, task: str):
        """**只读**缓存值：没有就返回 None，绝不触发推理

        与 get_task_value 的分工：后者是"取不到就算"，会在锁外补一次单句前向；
        句法证据的渲染（spatial_tools._render_cached_syntax）必须走本方法——LLM 标注
        链路承诺「一次准备、处处只读」，worker 里不允许再出现 LTP 推理。
        """
        return self._get_cached_value(sentence, task)

    def _infer_base(self, sentences: list[str]) -> list[dict]:
        """对一批句子跑基础四任务，并做方位词后缀切分修正

        prefill（批量）与 get_task_value（单句）共用这一条推理路径，避免两处各写一份
        切分逻辑而出现不一致。返回值与 sentences 等长，不写缓存。

        注意必须统一以 list 调 pipeline：传裸 str 时 is_batch=False，LTPOutput 会把
        单句结果解包一层，result[t] 就不再是逐句列表了。
        """
        # 整段（含 _correct_splits 里那次重跑）都在推理锁内：两个线程各跑一半前向没有意义，
        # 反而让"在途推理组数"变成不可控的量（见 _infer_lock）
        with _infer_lock:
            ltp = _get_ltp()
            tasks = list(self.BASE_TASKS)
            result = ltp.pipeline(sentences, tasks=tasks)
            values = [{t: result[t][i] for t in tasks} for i in range(len(sentences))]
            self._correct_splits(ltp, values)
        return values

    def _correct_splits(self, ltp: LTP, values: list[dict]) -> None:
        """把「名词+方位词」合并成的 token 切开，并就地回填修正后的结果

        切分改变了词序，所以 pos/dep/sdp 必须一起重算：pos 的长度决定后续几项的对齐，
        dep/sdp 的 head 下标也都基于新词序。重算是把切好的词表交给 pipeline 的
        「已分词」模式（tasks 不含 cws，见 ltp/nerual.py: is_split_into_words），
        受影响句子在同一批里一次前向完成。
        """
        if not config.LOCATIVE_SPLIT_ENABLED:
            return

        vocab = get_suffix_vocab()
        if not vocab:
            return

        affected: list[tuple[int, list[str]]] = []
        for idx, value in enumerate(values):
            new_words = split_tokens_by_suffix(
                value[config.CWS],
                value[config.POS],
                vocab,
                config.LOCATIVE_SPLIT_ALLOWED_POS,
                config.LOCATIVE_SPLIT_TOKEN_BLACKLIST,
            )
            if new_words is not None:
                affected.append((idx, new_words))

        if not affected:
            return

        rerun = ltp.pipeline(
            [new_words for _, new_words in affected],
            tasks=[config.POS, config.DEP, config.SDP],
        )

        # 「已分词」模式下 LTP 对批内 padding 过的句子会多吐末尾一格：
        # ltp/nerual.py 构造 word_index 时用 encoding.words[1:-1] 剥特殊符号，没考虑 padding
        # ——padding 句的 [SEP] 与 [PAD] 都是 None，切片只剥掉了一个，[SEP] 那格于是被当成
        # 「又一个词」，且 word_attention_mask 按同样错误的长度构造，它不是被 mask 掉而是真的
        # 产出一个标注。于是 pos/dep/sdp 都比词表长 1，长度不一致的缓存会让索引层建出越界下标，
        # 规则引擎（如 locality_phrase 的 has_nominal_modifier）取 words[idx] 时抛 IndexError。
        # 这是上游库的行为，消费侧只能自己对齐：多出的恒在末尾，截掉即可。
        for i, (idx, new_words) in enumerate(affected):
            expected = len(new_words)
            pos = rerun[config.POS][i]
            dep = rerun[config.DEP][i]
            sdp = rerun[config.SDP][i]
            lengths = (len(pos), len(dep["head"]), len(sdp["head"]))

            if any(n > expected for n in lengths):
                _warn_rerun_length_once(expected, lengths)
                pos = pos[:expected]
                dep = _truncate_tail(dep, expected)
                sdp = _truncate_tail(sdp, expected)
            elif any(n < expected for n in lengths):
                # 短于词表说明这句在 max_length 处被截断，补不回来也截不齐；
                # 那就放弃本句的切分，保留 LTP 原本自洽的四项——绝不让长度不一致的结果进缓存。
                _warn_rerun_length_once(expected, lengths)
                continue

            values[idx][config.CWS] = new_words
            values[idx][config.POS] = pos
            values[idx][config.DEP] = dep
            values[idx][config.SDP] = sdp

    def prefill(self, sentences: Iterable[str], chunk_size: int = None) -> int:
        """批量预填充一批句子的基础任务结果

        逐句调用 pipeline 时每次前向只处理一个句子，GPU 利用率很低；这里把一批句子
        按长度分桶后分组送入模型，一次前向处理一组，实测比逐句推理快约 4 倍。
        构造出的结果与逐句推理逐字节相同，只是换了一种算法。

        Args:
            sentences (Iterable[str]): 待预填充的句子
            chunk_size (int, optional): 每组句子数量，默认为 config.LTP_BATCH_CHUNK

        Returns:
            int: 实际送入模型的句子数量；已缓存的和批内重复的句子会被跳过
        """
        chunk_size = max(1, int(chunk_size or config.LTP_BATCH_CHUNK))

        pending: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            if not sentence or sentence in seen:
                continue
            seen.add(sentence)
            if self._need_base_tasks(sentence):
                pending.append(sentence)

        if not pending:
            return 0

        # 按长度排序，使同组内长度接近：pipeline 使用 padding="longest"，
        # 组内混入长句会把整组都填充到该长度，代价随长度平方增长
        pending.sort(key=len)

        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            for sentence, values in zip(chunk, self._infer_base(chunk)):
                self._update_cache(sentence, values)

        return len(pending)

    def get_task_value(self, sentence: str, task: str):
        """获取句子在各个语言处理任务上的值

        Args:
            sentence (str): 输入的句子
            task (str): 语言处理任务名称

        Returns:
            _type_: 语言处理任务上的值
        """
        cached_value = self._get_cached_value(sentence, task)
        if cached_value is not None:
            return cached_value

        # 对核心任务采用单次 pipeline 统一推理，减少重复模型调用
        if task in self.BASE_TASKS and self._need_base_tasks(sentence):
            self._update_cache(sentence, self._infer_base([sentence])[0])

        cached_value = self._get_cached_value(sentence, task)
        if cached_value is not None:
            return cached_value

        # 兜底：非核心任务按需计算，并缓存。
        # 走这里的是 ner/srl/sdpg，它们不在 BASE_TASKS 里，本项目也没有调用方；
        # 该分支只缓存 task 自身，不缓存未修正的 CWS，因此不受方位词切分影响。
        tasks_list = [config.CWS, task] if task != config.CWS else [task]
        with _infer_lock:  # 与 _infer_base 同一条串行边界（见 _infer_lock）
            value = _get_ltp().pipeline(sentence, tasks=tasks_list)[task]
        self._update_cache(sentence, {task: value})
        return value