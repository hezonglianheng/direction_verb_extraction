# encoding: utf8

"""流式流水线的小工具：把「抽取」放到后台线程，与「筛选」重叠进行。

语料抽取（读 JSONL + StnSplit + 空白切分）原本完全串行地夹在筛选循环里：主线程 next()
一条、筛一批、再 next() 一条。抽取与筛选因此严格交替，磁盘等待与切句的 CPU 时间都浪费在
筛选的空档里。``prefetch_iter`` 用一个有界队列把抽取变成后台生产者，主线程只消费：

    producer thread ──put──► bounded Queue ──get──► filter loop

有界队列是硬约束：生产者最多领先消费者 ``maxsize`` 条句子，大语料下内存占用仍只随队列大小
增长，不会把整份语料读进内存。队列满时生产者阻塞、消费者异常退出时生产者被 ``stop`` 唤醒，
不会留下一个永远卡在 put 上的后台线程。

``tag_rows`` 是配套的小工具：给每条记录补一个整轮唯一的临时键，供多规则/多进程筛选后
准确统计「至少命中一套规则」的记录数；这个键在落盘前必须摘掉（见两个入口的 ``_consume``）。
"""

import queue
import threading
from typing import Any, Iterable, Iterator

_SENTINEL = object()


def prune_tree(tree: Iterable[dict], false_indices: set[int], keep_indices: set[int]) -> list[dict]:
    """把被明确判为不可提取的目标词从 match_tree 里摘掉

    一个 false 节点若还带着要保留的下级目标词（明确有效或**判不出来、按 fail-open 保留**），
    不能把该节点整个删掉；把它提到根层又会留下一个词表内的 false 父节点，而下游取词口径是
    「父节点在词表里时不再认子节点」——被保留的子节点反而会被当成结构上下文忽略。
    所以这里**删掉 false 节点、把含保留目标词的子节点提上一层**。
    """

    def contains_index(node: dict, indices: set[int]) -> bool:
        if node.get("index") in indices:
            return True
        return any(
            contains_index(child, indices)
            for child in node.get("children") or []
            if isinstance(child, dict)
        )

    def walk(node: Any) -> list[dict]:
        if not isinstance(node, dict):
            return []
        index = node.get("index")
        children = [kept for child in (node.get("children") or []) for kept in walk(child)]
        if index in false_indices and index not in keep_indices:
            return [child for child in children if contains_index(child, keep_indices)]
        return [{**node, "children": children}]

    return [kept for node in tree or [] for kept in walk(node)]

ROW_ID_FIELD = "_corpus_row_id"
"""批内记录的临时唯一键：只用于「至少命中一套规则」的去重统计，落盘前必须摘掉

抽取阶段每条句子都是独立 dict，这里按流顺序编号即可保证整轮唯一；筛选的多进程分支
按批次返回、各规则族各自一份记录副本，唯独这个字段会原样带回来，于是能在父进程里
准确算出「这一批有多少条记录至少被一套规则留下」，不会被批内重复句折叠、也不会把
同一记录命中多套规则重复计数。
"""


def tag_rows(iterable: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """给每条记录补一个整轮唯一的临时键（复制一份，不改调用方手里的对象）"""
    for index, item in enumerate(iterable):
        yield {**item, ROW_ID_FIELD: index}


def row_key(row: dict[str, Any]) -> Any:
    """记录的去重键：优先用整轮唯一临时键，缺失时退回 (来源文件, 分词切片)

    LLM 前后的「送筛句子数」与「落盘句子数」必须用同一把尺子去重，否则同一条记录
    命中多套规则时会被重复计数；旧调用形状（测试桩/外部脚本）不带 ``ROW_ID_FIELD``，
    用来源文件 + 分词结果兜底，与原先 filter_sentences_streaming 里的内联写法一致。
    """
    row_id = row.get(ROW_ID_FIELD)
    if row_id is not None:
        return row_id
    return (row.get("source_file"), tuple(row.get("sentence") or ()))


def prefetch_iter(iterable: Iterable[Any], maxsize: int) -> Iterator[Any]:
    """在有界队列后面跑一个后台生产者，按原顺序产出元素

    Args:
        iterable: 原始可迭代对象（通常是逐文件逐句的生成器）
        maxsize: 队列最多囤多少条；<=0 按 1 处理

    Yields:
        Any: 原始元素，顺序与 ``iterable`` 一致

    Raises:
        生产者线程里的异常会在消费者下一次取值时原样抛出（不会静默丢句）。
    """
    maxsize = max(1, int(maxsize))
    buffer: queue.Queue = queue.Queue(maxsize=maxsize)
    stop = threading.Event()

    def produce() -> None:
        try:
            for item in iterable:
                while not stop.is_set():
                    try:
                        buffer.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:  # noqa: BLE001 —— 交给消费者线程原样抛出
            while not stop.is_set():
                try:
                    buffer.put((_SENTINEL, exc), timeout=0.2)
                    return
                except queue.Full:
                    continue
            return
        while not stop.is_set():
            try:
                buffer.put((_SENTINEL, None), timeout=0.2)
                return
            except queue.Full:
                continue

    thread = threading.Thread(target=produce, name="corpus-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = buffer.get()
            if isinstance(item, tuple) and len(item) == 2 and item[0] is _SENTINEL:
                error = item[1]
                if error is not None:
                    raise error
                return
            yield item
    finally:
        stop.set()
        # 生产者可能正卡在 put 上：清空队列让它有机会看见 stop（daemon 线程不阻塞退出）
        try:
            while True:
                buffer.get_nowait()
        except queue.Empty:
            pass
