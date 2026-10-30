import jsonlines
from tqdm import tqdm

from contextlib import ExitStack
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import Any, Iterable
import json
import os
import threading

import config
import extraction_log
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

_READ_COUNT_LOCK = threading.Lock()
"""切句线程池会并发累加「词表初筛前读入数」，用一把模块级锁保护这一次自增"""

ANY_HIT = "_any"
"""进度信息里表示“至少命中一套规则的句子数”的保留键"""


def _as_list(value: Any) -> list[Any]:
	"""兼容单个值与列表两种传参"""
	if isinstance(value, (list, tuple)):
		return list(value)
	return [value]


def _filter_sentence_batch(batch: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], int]:
	"""对一批句子跑全部规则（Windows 多进程分支的入口）

	与 Linux 入口同形：先在**当前进程**里把整批句子的 LTP 结果预填充，再逐句走规则引擎。
	返回值是 [每套规则的命中结果]（与 _WORKER_FILTERS 同序）与至少命中一套规则的句子数。
	"""
	if not _WORKER_FILTERS:
		raise RuntimeError("过滤进程尚未初始化过滤器")

	sentenceFilters.cache.prefill(entry.get("sentence", "") for entry in batch)

	filtered_batches: list[list[dict[str, Any]]] = [[] for _ in _WORKER_FILTERS]
	any_hit_count = 0
	for item in batch:
		cws_result = None
		hit_any = False
		for idx, (family, curr_filter) in enumerate(_WORKER_FILTERS):
			judge_result, explanation = curr_filter.explain(item.get("sentence", ""))
			if not judge_result:
				continue
			hit_any = True
			if cws_result is None:
				cws_result = sentenceFilters.cache.get_task_value(item.get("sentence", ""), config.CWS)
			filter_methods = explanation.get("methods", [])
			match_tree: list[sentenceFilters.MatchNode] = explanation.get("match_tree", [])
			simplified_tree = sentenceFilters.simplify_tree(match_tree, tokens=cws_result)
			record = item | {"sentence": cws_result, "filter_methods": filter_methods, "match_tree": simplified_tree}
			localization_grounding.attach_analysis(
				record,
				item.get("sentence", ""),
				sentenceFilters.cache,
				kind=_kind_for_family(family),
			)
			filtered_batches[idx].append(record)
		if hit_any:
			any_hit_count += 1
	return filtered_batches, any_hit_count


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
	for family, curr_filter in _WORKER_FILTERS:
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
		record = item | {"sentence": cws_result, "filter_methods": filter_methods, "match_tree": simplified_tree}
		localization_grounding.attach_analysis(
			record,
			curr_sentence,
			sentenceFilters.cache,
			kind=_kind_for_family(family),
		)
		results.append(record)
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
	*,
	screeners: dict[str, sentence_screening.SentenceScreener] | None = None,
	filter_workers: int | None = None,
	screen_kind: str | None = None,
	stats: dict[str, Any] | None = None,
):
	"""一次遍历、多套规则，命中即按批落盘

	与 filter_sentences_windows_in_memory 同一套规则与 LTP 预填充口径，差别只在
	不把全量候选句与全部命中结果留在内存里：句子边读边筛，每批（FILTER_BATCH_SIZE）
	筛完立即把命中行写进打开好的输出文件——内存占用不随语料规模增长。

	Args:
		sentences (Iterable[dict[str, Any]]): 备选句子流
		filter_config_paths (Any): 一套或多套筛选规则配置文件路径
		out_files (list[Any]): 与 filter_config_paths 同序的可写文件对象
		screeners (dict[str, SentenceScreener] | None): 按规则族取用的 LLM 筛选器；
			缺省或提示语为空时不做 LLM 筛选
		filter_workers (int | None): 规则筛选进程数；None 取 config.FILTER_WORKERS
		stats (dict | None): 本轮漏斗统计（见 extraction_log）；抽取阶段已累加的
			read_sentences 会原样保留

	Returns:
		dict: 与传入的 stats 同一份对象（便于成功收尾时写 extract-log）。
	"""
	stats = extraction_log.ensure_stats(stats)
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
		screener = screeners.get(family) if screeners else None
		if screener is None:
			return sentence_screening.annotate_no_llm(rows)
		return screener.screen_rows(rows)

	def _consume(
		filtered_batches: list[list[dict[str, Any]]],
		any_hit_count: int,
		batch_len: int,
		pbar: tqdm,
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

		# LLM 前：规则命中 + 语义闸门之后、真正送筛的句子数（按批内唯一键去重）
		pre_keys = {row_key(row) for rows in gated for row in rows}
		stats["program_selected"] += len(pre_keys)
		for family, rows in zip(families, gated):
			extraction_log.family_stats(stats, family)["program_selected"] += len(rows)

		# screen_rows 返回与输入等长：false/error/limit 也落盘，只是不进判定工作流。
		screened = [_screen(rows, family) for family, rows in zip(families, gated)]
		status_by_key: dict[Any, set[str]] = {}
		for family, rows in zip(families, screened):
			family_counts = extraction_log.family_stats(stats, family)
			for row in rows:
				key = row_key(row)
				status = sentence_screening.screening_status(row)
				status_by_key.setdefault(key, set()).add(status)
				family_counts["written"] += 1
				if status == sentence_screening.SCREENING_TRUE:
					family_counts["llm_kept"] += 1
				elif status == sentence_screening.SCREENING_FALSE:
					family_counts["llm_excluded"] += 1
				elif status == sentence_screening.SCREENING_ERROR:
					family_counts["llm_error"] += 1
				elif status == sentence_screening.SCREENING_LIMIT:
					family_counts["llm_limit"] += 1
				elif status == sentence_screening.SCREENING_NO_LLM:
					family_counts["llm_kept"] += 1
					family_counts["no_llm"] += 1

		allowed_statuses = sentence_screening.JUDGEMENT_SCREENING_STATUSES
		allowed_keys = {key for key, values in status_by_key.items() if values & allowed_statuses}
		false_keys = {
			key for key, values in status_by_key.items()
			if sentence_screening.SCREENING_FALSE in values and key not in allowed_keys
		}
		error_keys = {
			key for key, values in status_by_key.items()
			if sentence_screening.SCREENING_ERROR in values and key not in allowed_keys
		}
		limit_keys = {
			key for key, values in status_by_key.items()
			if sentence_screening.SCREENING_LIMIT in values and key not in allowed_keys
		}
		no_llm_keys = {
			key for key, values in status_by_key.items()
			if sentence_screening.SCREENING_NO_LLM in values
		}
		stats["corpus_written"] += len(status_by_key)
		stats["llm_kept"] += len(allowed_keys)
		stats["llm_excluded"] += len(false_keys)
		stats["llm_error"] += len(error_keys)
		stats["llm_limit"] += len(limit_keys)
		stats["no_llm"] += len(no_llm_keys)
		for status in sentence_screening.SCREENING_STATUSES:
			count = sum(1 for values in status_by_key.values() if status in values)
			if count:
				status_counts = stats.setdefault("screening_status", {})
				status_counts[status] = status_counts.get(status, 0) + count
		hit_counts[ANY_HIT] += len(status_by_key)
		for idx, family in enumerate(families):
			rows = screened[idx]
			hit_counts[family] += len(rows)
			for row in rows:
				row.pop(ROW_ID_FIELD, None)
				out_files[idx].write(json.dumps(row, ensure_ascii=False) + "\n")
		pbar.update(batch_len)
		pbar.set_postfix(
			_build_progress_postfix(candidate_count, hit_counts),
			refresh=False,
		)

	with tqdm(
		desc="Filtering sentences (Windows streaming)",
		leave=True,
		dynamic_ncols=True,
		unit="sentence",
	) as pbar:
		if max_workers == 1:
			# 旧调用形状（测试桩只接受一个参数）保持兼容；只有显式指定 kind 时才多传一个。
			if screen_kind is None:
				_init_filter_worker(filter_config_paths)
			else:
				_init_filter_worker(filter_config_paths, screen_kind)
			batch: list[dict[str, Any]] = []
			for sentence_item in sentences:
				batch.append(sentence_item)
				if len(batch) >= batch_size:
					candidate_count += len(batch)
					filtered_batches, any_hit_count = _filter_sentence_batch(batch)
					_consume(filtered_batches, any_hit_count, len(batch), pbar)
					batch = []
			if batch:
				candidate_count += len(batch)
				filtered_batches, any_hit_count = _filter_sentence_batch(batch)
				_consume(filtered_batches, any_hit_count, len(batch), pbar)
		else:
			max_pending = max_workers * 2
			spawn_ctx = mp.get_context("spawn")
			with ProcessPoolExecutor(
				max_workers=max_workers,
				mp_context=spawn_ctx,
				initializer=_init_filter_worker,
				initargs=(_as_list(filter_config_paths), screen_kind),
			) as executor:
				batch_iter = iter(_batch_sentences(sentences, batch_size))
				pending: dict[Any, tuple[int, int]] = {}
				buffered: dict[int, tuple[list[list[dict[str, Any]]], int, int]] = {}
				next_seq = 0
				submit_seq = 0
				exhausted = False
				while True:
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
					while next_seq in buffered:
						filtered_batches, any_hit_count, done_batch_size = buffered.pop(next_seq)
						_consume(filtered_batches, any_hit_count, done_batch_size, pbar)
						next_seq += 1

	print(f"从 {candidate_count} 个备选句子中筛选出了 {hit_counts[ANY_HIT]} 个符合条件的句子")
	for family in families:
		print(f"    {family}：{hit_counts[family]} 个句子")
	if grounding_total.get("records"):
		print(f"[方位词参照物语义闸门] {localization_grounding.format_stats(grounding_total)}")
	stats["vocab_candidates"] = candidate_count
	stats["grounding"] = dict(grounding_total)
	print(
		f"[筛选漏斗] 读入 {stats['read_sentences']}，初筛后 {candidate_count}，"
		f"程序筛选 {stats['program_selected']}，落盘 {stats['corpus_written']}，"
		f"可判定 {stats['llm_kept']}（true/no_llm），LLM 判假 {stats['llm_excluded']}，"
		f"LLM 错误 {stats['llm_error']}，超限 {stats['llm_limit']}，未用 LLM {stats['no_llm']}"
	)
	return stats


def _batch_sentences(items: Iterable[dict[str, Any]], batch_size: int):
	"""把句子流切成固定大小的批次（Windows 多进程分支用）"""
	batch: list[dict[str, Any]] = []
	for item in items:
		batch.append(item)
		if len(batch) >= batch_size:
			yield batch
			batch = []
	if batch:
		yield batch


def run_windows_entry(
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
	print("[运行模式] Windows: 流式处理（分批读取、筛选、落盘）")
	filter_config_paths = _as_list(filter_config_paths)
	paths = [Path(p) for p in _as_list(output_paths)]
	if len(filter_config_paths) != len(paths):
		raise ValueError(f"规则套数（{len(filter_config_paths)}）与输出文件数（{len(paths)}）不一致")

	stats = extraction_log.new_stats()
	# 输入清单与路径校验在打开任何输出之前完成（见 iter_path_sentences），
	# 随后输出文件保持打开，命中按批直接写入，收尾不必再统一写盘。
	sentences = iter_path_sentences(
		src_path, vocabs, output_paths=paths, workers=extract_workers, stats=stats
	)
	# 抽取放到后台线程、与筛选重叠；队列有界（见 pipeline_utils.prefetch_iter）
	sentences = prefetch_iter(sentences, config.EXTRACT_PREFETCH_ITEMS)
	# 分词口径打印一次（D4）：切分配置改了下标会位移，事后要能判断一份产物是哪个口径跑出来的
	print(f"[分词口径] {config.segmentation_signature()}")
	with ExitStack() as stack:
		# 先写临时文件、全部成功后才原子替换（E1）。旧写法用 "w" 直接在目标上打开：那在真正
		# 消费输入之前就把输出截断了，之后初始化过滤器、预填充 LTP 或筛选任一步失败，都会留下
		# 一个已清空的（或半截的）输出，上一轮的完整结果同时也没了。
		out_files = [stack.enter_context(atomic_write_text(path)) for path in paths]
		if screeners is None and filter_workers is None and screen_kind is None:
			# 旧调用形状（测试桩与外部脚本只接受三个参数）保持兼容
			returned = filter_sentences_streaming(sentences, filter_config_paths, out_files)
			if isinstance(returned, dict):
				read_sentences = int(stats.get("read_sentences", 0))
				stats.update(returned)
				# 旧形状下 streaming 自己新建统计，不知道抽取阶段的读入数；保留较大的那个
				stats["read_sentences"] = max(read_sentences, int(returned.get("read_sentences", 0)))
		else:
			filter_sentences_streaming(
				sentences,
				filter_config_paths,
				out_files,
				screeners=screeners,
				filter_workers=filter_workers,
				screen_kind=screen_kind,
				stats=stats,
			)

	if screeners:
		for family, screener in screeners.items():
			print(f"[LLM 筛选统计] {family}：{screener.stats()}")
	return stats
