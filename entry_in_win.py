import jsonlines
from tqdm import tqdm

from contextlib import ExitStack
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable
import json
import os
import threading

import config
import sentenceFilters
import sentence_split
import file_paths
from file_paths import atomic_write_text


_WORKER_FILTERS: list[tuple[str, Any]] = []
"""[(规则族名, 规则过滤器)]，一个进程内同时持有多套规则"""

_READ_COUNT_LOCK = threading.Lock()
"""切句线程池会并发累加「词表初筛前读入数」，用一把模块级锁保护这一次自增"""

ANY_HIT = "_any"
"""进度信息里表示“至少命中一套规则的句子数”的保留键"""


def _as_list(value: Any) -> list[Any]:
	"""兼容单个值与列表两种传参"""
	if isinstance(value, (list, tuple)):
		return list(value)
	return [value]


def _init_filter_worker(filter_config_paths: Any):
	global _WORKER_FILTERS
	_WORKER_FILTERS = [
		(Path(path).stem, sentenceFilters.FilterGenerator(path).generate_filter())
		for path in _as_list(filter_config_paths)
	]


def _filter_sentence(item: dict[str, Any]) -> list[dict[str, Any] | None]:
	"""对单个句子跑全部规则

	Returns:
		list[dict[str, Any] | None]: 与 _WORKER_FILTERS 同序的结果，未命中为 None
	"""
	if not _WORKER_FILTERS:
		raise RuntimeError("过滤进程尚未初始化过滤器")

	curr_sentence: str = item.get("sentence", "")
	results: list[dict[str, Any] | None] = []
	cws_result = None
	for _, curr_filter in _WORKER_FILTERS:
		judge_result, explanation = curr_filter.explain(curr_sentence)
		if not judge_result:
			results.append(None)
			continue

		if cws_result is None:
			# 获取CWS结果并替换原本的sentence（延迟到确有命中时才取）
			cws_result = sentenceFilters.cache.get_task_value(curr_sentence, config.CWS)

		filter_methods = explanation.get("methods", [])
		match_tree: list[sentenceFilters.MatchNode] = explanation.get("match_tree", [])
		# 带上分词结果：simplify_tree 据此记原文字符跨度（D4）。cws_result 在上面已取到
		simplified_tree = sentenceFilters.simplify_tree(match_tree, tokens=cws_result)

		results.append(item | {"sentence": cws_result, "filter_methods": filter_methods, "match_tree": simplified_tree})
	return results


def _build_progress_postfix(candidate_count: int, hit_counts: dict[str, int], *, batch_size: int | None = None) -> dict[str, str]:
	"""构造进度信息

	Args:
		candidate_count (int): 已处理的备选句数
		hit_counts (dict[str, int]): 键为规则族名（外加保留键 ANY_HIT）的命中数
	"""
	def _with_rate(count: int) -> str:
		rate = (count / candidate_count * 100) if candidate_count else 0.0
		return f"{count} ({rate:.2f}%)"

	any_hit = hit_counts.get(ANY_HIT, 0)
	any_rate = (any_hit / candidate_count * 100) if candidate_count else 0.0
	postfix: dict[str, str] = {"cand": str(candidate_count), "hit": str(any_hit), "rate": f"{any_rate:.2f}%"}
	postfix.update({family: _with_rate(count) for family, count in hit_counts.items() if family != ANY_HIT})
	if batch_size is not None:
		postfix["batch"] = str(batch_size)
	return postfix


def get_sentences_list(article: str, vocabs: list[str] = None) -> list[str]:
	"""把一篇文本切成句子：句末标点 + 各种空格/不可见分隔符（规则见 sentence_split）"""
	return sentence_split.split_sentences(article, vocabs)


def _sentences_after_vocab(sentence_list: list[str], vocabs: list[str] | None, stats: dict[str, Any] | None) -> list[str]:
	"""统计「词表初筛前读入数」，再按初筛词表过滤（口径同 Linux 入口）

	可能在线程池里并发调用：统计带锁自增，过滤仍在工作线程完成。
	"""
	if stats is not None:
		with _READ_COUNT_LOCK:
			stats["read_sentences"] = int(stats.get("read_sentences", 0)) + len(sentence_list)
	return sentence_split.filter_by_vocab(sentence_list, vocabs)


def _iter_article_sentences(
	articles: Iterable[dict[str, Any]],
	vocabs: list[str] | None,
	workers: int,
	stats: dict[str, Any] | None = None,
):
	"""按批把文章交给线程池切句，限制在途批大小（理由同 entry_in_linux 的同名函数）

	仍以两个位置参数调用 get_sentences_list（第二个为 None：先不按词表过滤），
	统计「初筛前」句子数后再在同一个工作线程里过滤词表。
	"""
	batch_size = max(1, workers * 4)
	batch: list[dict[str, Any]] = []
	with ThreadPoolExecutor(max_workers=workers) as pool:
		for obj in articles:
			batch.append(obj)
			if len(batch) >= batch_size:
				for sentence_list in pool.map(
					lambda item: _sentences_after_vocab(
						get_sentences_list(item.get("content", ""), None), vocabs, stats
					),
					batch,
				):
					yield from sentence_list
				batch = []
		if batch:
			for sentence_list in pool.map(
				lambda item: _sentences_after_vocab(
					get_sentences_list(item.get("content", ""), None), vocabs, stats
				),
				batch,
			):
				yield from sentence_list


def iter_jsonl_file_sentences(
	input_file: str,
	vocabs: list[str] = None,
	*,
	workers: int | None = None,
	stats: dict[str, Any] | None = None,
):
	"""逐条读入、逐条切分并产出句子

	与整文件读入版（历史实现）的区别：jsonlines 本身是流式读的，这里不再把整份文件
	一次性 list() 进内存——大语料下内存占用只随单条文章（或一批文章）的大小增长。
	workers > 1 时批内用线程池并行切句。
	stats 用于累加「词表初筛前读入总数」（见 extraction_log）。
	"""
	extracted_count = 0
	configured = int(config.EXTRACT_WORKERS if workers is None else workers)
	workers = max(1, configured if configured > 0 else min(8, os.cpu_count() or 1))
	with jsonlines.open(input_file) as reader:
		if workers == 1:
			sentence_iter = (
				sentence
				for obj in reader
				for sentence in _sentences_after_vocab(
					get_sentences_list(obj.get("content", ""), None), vocabs, stats
				)
			)
		else:
			sentence_iter = _iter_article_sentences(reader, vocabs, workers, stats)
		for sentence in sentence_iter:
			extracted_count += 1
			yield {"sentence": sentence, "source_file": input_file}
	print(f"从文件 {input_file} 中提取了 {extracted_count} 个句子")


def iter_path_sentences(
	src_path: str,
	vocabs: list[str] = None,
	*,
	output_paths=(),
	workers: int | None = None,
	stats: dict[str, Any] | None = None,
):
	"""目录（或单文件）下的 jsonl 逐文件流式产出句子

	收集与遍历走 file_paths.iter_path_sentences（与 entry_in_linux 同一份）：
	路径检查和目录扫描必须在打开输出文件前完成，那条约束两入口一致，不该各写一遍。
	这里只带本平台自己的切句方式进来；workers 透传给切句线程池。
	"""
	return file_paths.iter_path_sentences(
		src_path,
		# index 是文件在清单里的序号：Windows 入口不写"第几个文件"，忽略它
		lambda file, index: iter_jsonl_file_sentences(
			str(file), vocabs=vocabs, workers=workers, stats=stats
		),
		output_paths=output_paths,
	)


def filter_sentences_streaming(
	sentences: Iterable[dict[str, Any]],
	filter_config_paths: Any,
	out_files: list[Any],
):
	"""一次遍历、多套规则，命中即按批落盘

	与 filter_sentences_windows_in_memory 同一套规则与 LTP 预填充口径，差别只在
	不把全量候选句与全部命中结果留在内存里：句子边读边筛，每批（FILTER_BATCH_SIZE）
	筛完立即把命中行写进打开好的输出文件——内存占用不随语料规模增长。

	Args:
		sentences (Iterable[dict[str, Any]]): 备选句子流
		filter_config_paths (Any): 一套或多套筛选规则配置文件路径
		out_files (list[Any]): 与 filter_config_paths 同序的可写文件对象
	"""
	families = [Path(path).stem for path in _as_list(filter_config_paths)]
	if len(families) != len(out_files):
		raise ValueError(f"规则套数（{len(families)}）与输出文件数（{len(out_files)}）不一致")

	_init_filter_worker(filter_config_paths)
	candidate_count = 0
	hit_counts: dict[str, int] = {ANY_HIT: 0}
	hit_counts.update({family: 0 for family in families})
	batch_size = max(1, int(config.FILTER_BATCH_SIZE))

	def _filter_batch(batch: list[dict[str, Any]], pbar: tqdm) -> None:
		"""整批预填充 LTP → 逐句走规则引擎 → 本批命中立即写盘"""
		nonlocal candidate_count
		# 分块「先整批预填充 LTP 结果，再逐句走规则引擎」：逐句调用 pipeline 实测慢约 4 倍。
		# 必须分块而不能一次性预填充，否则整批会被 LRU 在读取前淘汰。
		# 多套规则共用这份缓存，故 prefill 每批只做一次，不会重复推理。
		sentenceFilters.cache.prefill(item.get("sentence", "") for item in batch)

		for sentence_item in batch:
			candidate_count += 1
			item_results = _filter_sentence(sentence_item)
			if any(result is not None for result in item_results):
				hit_counts[ANY_HIT] += 1
			for idx, family in enumerate(families):
				if item_results[idx] is not None:
					hit_counts[family] += 1
					out_files[idx].write(json.dumps(item_results[idx], ensure_ascii=False) + "\n")
			pbar.update(1)
			pbar.set_postfix(
				_build_progress_postfix(
					candidate_count,
					hit_counts,
				),
				refresh=False,
			)

	with tqdm(
		desc="Filtering sentences (Windows streaming)",
		leave=True,
		dynamic_ncols=True,
		unit="sentence",
	) as pbar:
		batch: list[dict[str, Any]] = []
		for sentence_item in sentences:
			batch.append(sentence_item)
			if len(batch) >= batch_size:
				_filter_batch(batch, pbar)
				batch = []
		if batch:
			_filter_batch(batch, pbar)

	print(f"从 {candidate_count} 个备选句子中筛选出了 {hit_counts[ANY_HIT]} 个符合条件的句子")
	for family in families:
		print(f"    {family}：{hit_counts[family]} 个句子")


def run_windows_entry(src_path: str, filter_config_paths: Any, output_paths: Any, vocabs: list[str] = None):
	print("[运行模式] Windows: 流式处理（分批读取、筛选、落盘）")
	filter_config_paths = _as_list(filter_config_paths)
	paths = [Path(p) for p in _as_list(output_paths)]
	if len(filter_config_paths) != len(paths):
		raise ValueError(f"规则套数（{len(filter_config_paths)}）与输出文件数（{len(paths)}）不一致")

	# 输入清单与路径校验在打开任何输出之前完成（见 iter_path_sentences），
	# 随后输出文件保持打开，命中按批直接写入，收尾不必再统一写盘。
	sentences = iter_path_sentences(src_path, vocabs, output_paths=paths)
	# 分词口径打印一次（D4）：切分配置改了下标会位移，事后要能判断一份产物是哪个口径跑出来的
	print(f"[分词口径] {config.segmentation_signature()}")
	with ExitStack() as stack:
		# 先写临时文件、全部成功后才原子替换（E1）。旧写法用 "w" 直接在目标上打开：那在真正
		# 消费输入之前就把输出截断了，之后初始化过滤器、预填充 LTP 或筛选任一步失败，都会留下
		# 一个已清空的（或半截的）输出，上一轮的完整结果同时也没了。
		out_files = [stack.enter_context(atomic_write_text(path)) for path in paths]
		filter_sentences_streaming(sentences, filter_config_paths, out_files)
