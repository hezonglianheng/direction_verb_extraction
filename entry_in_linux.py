# encoding: utf8

import jsonlines

from contextlib import ExitStack
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Any, Iterable
import json
import os
import threading
import time

import config
import localization_grounding
import sentenceFilters
import sentence_split
import sentence_screening
import file_paths
from file_paths import atomic_write_text
from pipeline_utils import ROW_ID_FIELD, prefetch_iter, row_key, tag_rows


_WORKER_FILTERS: list[tuple[str, Any]] = []
"""[(规则族名, 规则过滤器)]，一个进程内同时持有多套规则"""

_WORKER_KIND_OVERRIDE: str | None = None
"""`--screen-kind` 显式指定的词类型；None = 按规则配置文件名推断。"""

_START_TIME = None

_READ_COUNT_LOCK = threading.Lock()
"""切句线程池会并发累加「词表初筛前读入数」，用一把模块级锁保护这一次自增"""

ANY_HIT = "_any"
"""进度信息里表示“至少命中一套规则的句子数”的保留键"""


def _as_list(value: Any) -> list[Any]:
	"""兼容单个值与列表两种传参"""
	if isinstance(value, (list, tuple)):
		return list(value)
	return [value]


def _kind_for_family(family: str) -> str | None:
	"""词类型：命令行显式指定优先，否则按规则配置文件名推断。"""
	return _WORKER_KIND_OVERRIDE or sentence_screening.classify_family(family)


def _init_filter_worker(filter_config_paths: Any, kind: str | None = None):
	global _WORKER_FILTERS, _WORKER_KIND_OVERRIDE
	_WORKER_KIND_OVERRIDE = kind
	_WORKER_FILTERS = [
		(Path(path).stem, sentenceFilters.FilterGenerator(path).generate_filter())
		for path in _as_list(filter_config_paths)
	]


def _filter_sentence_batch(batch: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], int]:
	"""对一批句子跑全部规则

	Returns:
		tuple[list[list[dict[str, Any]]], int]:
			[每套规则的命中结果]（与 _WORKER_FILTERS 同序）以及至少命中一套规则的句子数。
	"""
	if not _WORKER_FILTERS:
		raise RuntimeError("过滤进程尚未初始化过滤器")

	# 先把整批句子的 LTP 结果批量算出来写进缓存，再逐句走规则引擎。
	# 否则规则引擎会按需逐句触发单句推理，实测慢约 4 倍。
	# 多套规则共用这份缓存，故 prefill 每批只做一次，不会重复推理。
	sentenceFilters.cache.prefill(s.get("sentence", "") for s in batch)

	filtered_batches: list[list[dict[str, Any]]] = [[] for _ in _WORKER_FILTERS]
	any_hit_count = 0
	for s in batch:
		curr_sentence: str = s.get("sentence", "")
		cws_result = None
		hit_any = False
		for idx, (family, curr_filter) in enumerate(_WORKER_FILTERS):
			# explain 的真值判定与 judge 严格等价（同一套 context + 集合运算），单次调用即可
			judge_result, explanation = curr_filter.explain(curr_sentence)
			if not judge_result:
				continue

			hit_any = True
			if cws_result is None:
				# 获取CWS结果并替换原本的sentence（延迟到确有命中时才取）
				cws_result = sentenceFilters.cache.get_task_value(curr_sentence, config.CWS)

			filter_methods = explanation.get("methods", [])
			match_tree: list[sentenceFilters.MatchNode] = explanation.get("match_tree", [])
			# 带上分词结果：simplify_tree 据此记原文字符跨度（D4）。cws_result 在上面已取到
			simplified_tree = sentenceFilters.simplify_tree(match_tree, tokens=cws_result)
			record = s | {"sentence": cws_result, "filter_methods": filter_methods, "match_tree": simplified_tree}
			# 方位词记录把 POS/DEP/SDP 一并带回主进程；OpenHowNet 只在主进程初始化一次。
			localization_grounding.attach_analysis(
				record,
				curr_sentence,
				sentenceFilters.cache,
				kind=_kind_for_family(family),
			)
			filtered_batches[idx].append(record)

		if hit_any:
			any_hit_count += 1
	return filtered_batches, any_hit_count


def _iter_batches(items: Iterable[dict[str, Any]], batch_size: int):
	batch: list[dict[str, Any]] = []
	for item in items:
		batch.append(item)
		if len(batch) >= batch_size:
			yield batch
			batch = []
	if batch:
		yield batch


def _build_progress_postfix(
	candidate_count: int,
	hit_counts: dict[str, int],
	*,
	batch_size: int | None = None,
	pending_size: int | None = None,
	max_pending: int | None = None,
) -> list[tuple[str, str | int]]:
	"""构造一行进度信息

	Args:
		candidate_count (int): 已处理的备选句数
		hit_counts (dict[str, int]): 键为规则族名（外加保留键 ANY_HIT）的命中数

	Returns:
		list[tuple[str, str | int]]: [(标签, 值)]，由 _print_batch_progress 拼接输出
	"""
	def _with_rate(count: int) -> str:
		rate = (count / candidate_count * 100) if candidate_count else 0.0
		return f"{count} ({rate:.2f}%)"

	any_hit = hit_counts.get(ANY_HIT, 0)
	any_rate = (any_hit / candidate_count * 100) if candidate_count else 0.0
	items: list[tuple[str, str | int]] = [("cand", candidate_count), ("hit", any_hit), ("rate", f"{any_rate:.2f}%")]
	items.extend((family, _with_rate(count)) for family, count in hit_counts.items() if family != ANY_HIT)
	if batch_size is not None:
		items.append(("batch", batch_size))
	if pending_size is not None and max_pending is not None:
		items.append(("pending", f"{pending_size}/{max_pending}"))
	return items


def _print_batch_progress(items: list[tuple[str, str | int]]) -> None:
	parts = [f"{label}：{value}" for label, value in items]
	if _START_TIME is not None:
		elapsed = time.time() - _START_TIME
		parts.append(f"elapsed：{elapsed:.2f}s")
	print("\t".join(parts))


def get_sentences_list(article: str, vocabs: list[str] = None) -> list[str]:
	"""把一篇文本切成句子：句末标点 + 各种空格/不可见分隔符（规则见 sentence_split）

	保留这个函数名是为了两个平台入口与既有测试继续有同一个接缝；真正的切分口径只有
	sentence_split.split_sentences 一处真源。
	"""
	return sentence_split.split_sentences(article, vocabs)


def _sentences_after_vocab(sentence_list: list[str], vocabs: list[str] | None, stats: dict[str, Any] | None) -> list[str]:
	"""统计「词表初筛前读入数」，再按初筛词表过滤

	可能在线程池的多个切句线程里并发调用：统计只做一次带锁自增，过滤仍在工作线程内
	完成，跨线程返回的只是过滤后的句子（与改动前的内存上界一致）。
	词表过滤本身走 sentence_split 的同一份子串口径。
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
	"""按批把文章交给线程池切句，限制在途批大小，避免一次性读完整个文件

	ThreadPoolExecutor.map 会先把所有输入提交出去；直接 map 整个 reader 等于把整份 JSONL
	读进内存。这里按 ``workers * 4`` 一批提交，批间产出，内存只随这一批增长。

	调用 ``get_sentences_list`` 时仍传两个位置参数（第二个为 None：先不按词表过滤），
	既保留既有测试/外部替换的调用形状，也拿到「初筛前」句子数；词表过滤随后在同一
	工作线程里完成。
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
	position: int = 0,
	vocabs: list[str] = None,
	*,
	workers: int | None = None,
	stats: dict[str, Any] | None = None,
):
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

	收集与遍历走 file_paths.iter_path_sentences（与 entry_in_win 同一份，理由见那里）。
	本入口多带一个 position=文件序号，产物里因此知道这句出自第几个输入文件；
	workers 透传给单文件切句的线程池（0/None 取 config.EXTRACT_WORKERS）。
	stats 是本轮漏斗统计：在这里透传给切句层累加「读入总数」。
	"""
	return file_paths.iter_path_sentences(
		src_path,
		lambda file, index: iter_jsonl_file_sentences(
			str(file), position=index, vocabs=vocabs, workers=workers, stats=stats
		),
		output_paths=output_paths,
	)


def filter_sentences_streaming(
	sentences: Iterable[dict[str, Any]],
	filter_config_paths: Any,
	out_files: list[Any],
	*,
	screeners: dict[str, sentence_screening.SentenceScreener] | None = None,
	filter_workers: int | None = None,
	screen_kind: str | None = None,
):
	"""一次遍历、多套规则、各自落盘

	Args:
		sentences (Iterable[dict[str, Any]]): 备选句子流
		filter_config_paths (Any): 一套或多套筛选规则配置文件路径
		out_files (list[Any]): 与 filter_config_paths 同序的可写文件对象
		screeners (dict[str, SentenceScreener] | None): 按规则族取用的 LLM 筛选器；
			缺省或提示语为空时不做 LLM 筛选，记录形状与旧版一致
		filter_workers (int | None): 规则筛选进程数；None 取 config.FILTER_WORKERS
		screen_kind (str | None): 显式指定词类型（方位词/趋向动词）；None = 按配置文件名推断。
			只能用于单套规则，且会同时决定是否启用方位词参照物语义闸门
	"""
	families = [Path(path).stem for path in _as_list(filter_config_paths)]
	if len(families) != len(out_files):
		raise ValueError(f"规则套数（{len(families)}）与输出文件数（{len(out_files)}）不一致")
	if screen_kind and len(families) > 1:
		raise ValueError("screen_kind 只能用于单套规则；多套规则请让程序按各自的配置文件名推断")

	candidate_count = 0
	hit_counts: dict[str, int] = {ANY_HIT: 0}
	hit_counts.update({family: 0 for family in families})
	batch_size = max(1, int(config.FILTER_BATCH_SIZE))
	configured_workers = int(config.FILTER_WORKERS if filter_workers is None else filter_workers)
	max_workers = max(1, configured_workers or (os.cpu_count() or 1))
	# 给每条记录一个整轮唯一键，供「至少命中一套规则」去重；落盘前会摘掉（见 _consume）
	sentences = tag_rows(sentences)
	print(f"[启动参数] platform=linux, configured_workers={configured_workers}, active_workers={max_workers}, rules={families}")
	for family in families:
		if screeners and family in screeners:
			print(f"[LLM 筛选] {screeners[family].describe()}")

	# 方位词族先备好词表与 OpenHowNet 闸门；趋向动词族完全不加载 HowNet。
	family_meta, physical_gate = localization_grounding.prepare_families(
		_as_list(filter_config_paths), sentence_screening.classify_family, kind=screen_kind
	)
	grounding_total = localization_grounding.empty_stats()
	if physical_gate is not None:
		print("[方位词语义闸门] OpenHowNet 已就绪：逐词检查参照物是否至少有一个义项的义原表示实际物体")
	for family, meta in family_meta.items():
		if meta.get("kind") is None:
			print(
				f"[方位词语义闸门] 警告：无法从规则配置名 {family!r} 判断词类型，"
				"已跳过参照物语义检查；如确属方位词，请用 --screen-kind 方位词 显式指定"
			)

	def _screen(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
		"""命中记录在落盘前过一遍 LLM 筛选；没有筛选器时补 no_llm 标注"""
		screener = screeners.get(family) if screeners else None
		if screener is None:
			return sentence_screening.annotate_no_llm(rows)
		return screener.screen_rows(rows)

	def _consume(
		filtered_batches: list[list[dict[str, Any]]],
		any_hit_count: int,
		batch_len: int,
		*,
		pending_size: int | None = None,
		max_pending: int | None = None,
	) -> None:
		gated: list[list[dict[str, Any]]] = []
		for idx, family in enumerate(families):
			rows = filtered_batches[idx]
			meta = family_meta.get(family)
			if meta and meta.get("kind") == "方位词":
				rows, delta = localization_grounding.gate_rows(
					rows, meta["vocab"], gate=physical_gate
				)
				localization_grounding.merge_stats(grounding_total, delta)
			gated.append(rows)

		# screen_rows 现在返回与输入等长的记录：false/error/limit 也落盘，只是不进判定工作流。
		screened = [_screen(rows, family) for family, rows in zip(families, gated)]
		status_by_key: dict[Any, set[str]] = {}
		for rows in screened:
			for row in rows:
				key = row_key(row)
				status = sentence_screening.screening_status(row)
				status_by_key.setdefault(key, set()).add(status)

		hit_counts[ANY_HIT] += len(status_by_key)
		for idx, family in enumerate(families):
			rows = screened[idx]
			hit_counts[family] += len(rows)
			for row in rows:
				row.pop(ROW_ID_FIELD, None)
				out_files[idx].write(json.dumps(row, ensure_ascii=False) + "\n")
		_print_batch_progress(
			_build_progress_postfix(
				candidate_count,
				hit_counts,
				batch_size=batch_len,
				pending_size=pending_size,
				max_pending=max_pending,
			)
		)

	if max_workers == 1:
		# 旧调用形状（测试桩只接受一个参数）保持兼容；只有显式指定 kind 时才多传一个。
		if screen_kind is None:
			_init_filter_worker(filter_config_paths)
		else:
			_init_filter_worker(filter_config_paths, screen_kind)
		for batch in _iter_batches(sentences, batch_size):
			candidate_count += len(batch)
			filtered_batches, any_hit_count = _filter_sentence_batch(batch)
			_consume(filtered_batches, any_hit_count, len(batch))
	else:
		max_pending = max_workers * 2
		spawn_ctx = mp.get_context("spawn")
		with ProcessPoolExecutor(
			max_workers=max_workers,
			mp_context=spawn_ctx,
			initializer=_init_filter_worker,
			initargs=(_as_list(filter_config_paths), screen_kind),
		) as executor:
			batch_iter = iter(_iter_batches(sentences, batch_size))
			pending: dict[Any, tuple[int, int]] = {}
			buffered: dict[int, tuple[list[list[dict[str, Any]]], int, int]] = {}
			next_seq = 0
			submit_seq = 0
			exhausted = False

			while True:
				# 在途批次 + 已完成但尚未按序落盘的批次一起受限：某个慢批次不能把缓冲撑大
				while not exhausted and len(pending) + len(buffered) < max_pending:
					try:
						batch = next(batch_iter)
					except StopIteration:
						exhausted = True
						break
					candidate_count += len(batch)
					future = executor.submit(_filter_sentence_batch, batch)
					pending[future] = (submit_seq, len(batch))
					submit_seq += 1

				if not pending and not buffered:
					break

				if pending:
					done_future = next(as_completed(pending))
					done_seq, done_batch_size = pending.pop(done_future)
					filtered_batches, any_hit_count = done_future.result()
					buffered[done_seq] = (filtered_batches, any_hit_count, done_batch_size)

				# 严格按提交顺序消费：多进程只改变算的顺序，不改变落盘顺序与下游行号
				while next_seq in buffered:
					filtered_batches, any_hit_count, done_batch_size = buffered.pop(next_seq)
					_consume(filtered_batches, any_hit_count, done_batch_size, pending_size=len(pending), max_pending=max_pending)
					next_seq += 1

	print(f"从 {candidate_count} 个备选句子中筛选出了 {hit_counts[ANY_HIT]} 个符合条件的句子")
	for family in families:
		print(f"    {family}：{hit_counts[family]} 个句子")
	if grounding_total.get("records"):
		print(f"[方位词参照物语义闸门] {localization_grounding.format_stats(grounding_total)}")


def run(
	src_path: str,
	filter_config_paths: Any,
	output_paths: Any,
	vocabs: list[str] = None,
	*,
	screeners: dict[str, sentence_screening.SentenceScreener] | None = None,
	extract_workers: int | None = None,
	filter_workers: int | None = None,
	screen_kind: str | None = None,
):
	global _START_TIME
	_START_TIME = time.time()
	print("[运行模式] Linux/Other: 流式处理")
	filter_config_paths = _as_list(filter_config_paths)
	paths = [Path(p) for p in _as_list(output_paths)]
	if len(filter_config_paths) != len(paths):
		raise ValueError(f"规则套数（{len(filter_config_paths)}）与输出文件数（{len(paths)}）不一致")

	sentences = iter_path_sentences(
		src_path, vocabs=vocabs, output_paths=paths, workers=extract_workers
	)
	# 抽取放到后台线程、与筛选重叠；队列有界（见 pipeline_utils.prefetch_iter），不会整份读入内存
	sentences = prefetch_iter(sentences, config.EXTRACT_PREFETCH_ITEMS)
	# 分词口径打印一次（D4）：切分配置改了下标会位移，事后要能判断一份产物是哪个口径跑出来的
	print(f"[分词口径] {config.segmentation_signature()}")

	with ExitStack() as stack:
		# 先写临时文件、全部成功后才原子替换（E1）：理由与 entry_in_win.run_windows_entry 一致
		out_files = [stack.enter_context(atomic_write_text(path)) for path in paths]
		if screeners is None and filter_workers is None and screen_kind is None:
			# 旧调用形状（测试桩与外部脚本只接受三个参数）保持兼容
			filter_sentences_streaming(sentences, filter_config_paths, out_files)
		else:
			filter_sentences_streaming(
				sentences,
				filter_config_paths,
				out_files,
				screeners=screeners,
				filter_workers=filter_workers,
				screen_kind=screen_kind,
			)

	if screeners:
		for family, screener in screeners.items():
			print(f"[LLM 筛选统计] {family}：{screener.stats()}")


def run_linux_entry(
	src_path: str,
	filter_config_paths: Any,
	output_paths: Any,
	vocabs: list[str] = None,
	*,
	screeners: dict[str, sentence_screening.SentenceScreener] | None = None,
	extract_workers: int | None = None,
	filter_workers: int | None = None,
	screen_kind: str | None = None,
):
	return run(
		src_path,
		filter_config_paths,
		output_paths,
		vocabs,
		screeners=screeners,
		extract_workers=extract_workers,
		filter_workers=filter_workers,
		screen_kind=screen_kind,
	)
