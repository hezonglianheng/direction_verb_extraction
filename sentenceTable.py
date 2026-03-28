# encoding: utf8

from ltp import LTP
import torch

import threading
from collections import OrderedDict

import config

_ltp = None
_ltp_init_lock = threading.Lock()


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
            result = _get_ltp().pipeline(sentence, tasks=list(self.BASE_TASKS))
            self._update_cache(sentence, {t: result[t] for t in self.BASE_TASKS})

        cached_value = self._get_cached_value(sentence, task)
        if cached_value is not None:
            return cached_value

        # 兜底：非核心任务按需计算，并缓存
        tasks_list = [config.CWS, task] if task != config.CWS else [task]
        value = _get_ltp().pipeline(sentence, tasks=tasks_list)[task]
        self._update_cache(sentence, {task: value})
        return value