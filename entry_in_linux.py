# encoding: utf8

from ltp import StnSplit
import jsonlines

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import re
from typing import Any, Iterable
import csv
import os
import time

import config
import sentenceFilters


_WORKER_FILTER = None
_START_TIME = None


def _init_filter_worker(filter_config_path: str):
	global _WORKER_FILTER
	generator = sentenceFilters.FilterGenerator(filter_config_path)
	_WORKER_FILTER = generator.generate_filter()


def _filter_sentence_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
	if _WORKER_FILTER is None:
		raise RuntimeError("过滤进程尚未初始化过滤器")

	filtered_batch: list[dict[str, Any]] = []
	for s in batch:
		curr_sentence: str = s.get("sentence", "")
		if not _WORKER_FILTER.judge(curr_sentence):
			continue

		# fast path: 仅对命中句子构建 explain 与匹配树
		judge_result, explanation = _WORKER_FILTER.explain(curr_sentence)
		if not judge_result:
			continue

		filter_methods = explanation.get("methods", [])
		match_tree: list[sentenceFilters.MatchNode] = explanation.get("match_tree", [])
		simplified_tree = sentenceFilters.simplify_tree(match_tree)
		filtered_batch.append(s | {"filter_methods": filter_methods, "match_tree": simplified_tree})
	return filtered_batch


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
	filtered_count: int,
	*,
	batch_size: int | None = None,
	pending_size: int | None = None,
	max_pending: int | None = None,
) -> dict[str, str | int]:
	hit_rate = (filtered_count / candidate_count * 100) if candidate_count else 0.0
	postfix: dict[str, str | int] = {
		"cand": candidate_count,
		"hit": filtered_count,
		"rate": f"{hit_rate:.2f}%",
	}
	if batch_size is not None:
		postfix["batch"] = batch_size
	if pending_size is not None and max_pending is not None:
		postfix["pending"] = f"{pending_size}/{max_pending}"
	return postfix


def _print_batch_progress(postfix: dict[str, str | int]) -> None:
	lines = [
		f"cand：{postfix['cand']}",
		f"hit：{postfix['hit']}",
		f"rate：{postfix['rate']}",
		f"batch：{postfix['batch']}",
	]
	if "pending" in postfix:
		lines.append(f"pending：{postfix['pending']}")
	if _START_TIME is not None:
		elapsed = time.time() - _START_TIME
		lines.append(f"elapsed：{elapsed:.2f}s")
	print("\t".join(lines))


def get_sentences_list(article: str, vocabs: list[str] = None) -> list[str]:
	sentences = StnSplit().split(article)
	sentences = [s for s in sentences if config.SENTENCE_LEN_LOWER_BOUND <= len(s) <= config.SENTENCE_LEN_UPPER_BOUND]
	sentences = [s for s in sentences if s.isascii() or re.search(r"[\u4e00-\u9fff]", s)]
	sentences = [s for s in sentences if "【【【缺文】】】" not in s]
	sentences = [s.strip() for s in sentences]
	if vocabs:
		sentences = [s for s in sentences if any(vocab in s for vocab in vocabs)]
	return [s.split()[-1] for s in sentences]


def iter_jsonl_file_sentences(input_file: str, position: int = 0, vocabs: list[str] = None):
	extracted_count = 0
	with jsonlines.open(input_file) as reader:
		for obj in reader:
			article: str = obj["content"]
			sentence_list = get_sentences_list(article, vocabs)
			for sentence in sentence_list:
				extracted_count += 1
				yield {"sentence": sentence, "source_file": input_file}
	print(f"从文件 {input_file} 中提取了 {extracted_count} 个句子")


def iter_path_sentences(src_path: str, vocabs: list[str] = None):
	input_path = Path(src_path)
	if input_path.is_file():
		yield from iter_jsonl_file_sentences(src_path, position=0, vocabs=vocabs)
		return

	if input_path.is_dir():
		files = list(input_path.glob("*.jsonl"))
		for i, file in enumerate(files):
			yield from iter_jsonl_file_sentences(str(file), position=i, vocabs=vocabs)
		return

	raise ValueError(f"输入路径 {src_path} 既不是文件也不是目录")


def filter_sentences_streaming(
	sentences: Iterable[dict[str, Any]],
	filter_config_path: str,
	writer: csv.DictWriter,
):
	candidate_count = 0
	filtered_count = 0
	batch_size = max(1, int(config.FILTER_BATCH_SIZE))
	configured_workers = int(config.FILTER_WORKERS) if config.FILTER_WORKERS else (os.cpu_count() or 1)
	max_workers = max(1, configured_workers)
	print(f"[启动参数] platform=linux, configured_workers={configured_workers}, active_workers={max_workers}")

	if max_workers == 1:
		_init_filter_worker(filter_config_path)
		for batch in _iter_batches(sentences, batch_size):
			candidate_count += len(batch)
			filtered_batch = _filter_sentence_batch(batch)
			filtered_count += len(filtered_batch)
			for row in filtered_batch:
				writer.writerow(row)
			_print_batch_progress(
				_build_progress_postfix(
					candidate_count,
					filtered_count,
					batch_size=len(batch),
				)
			)
	else:
		max_pending = max_workers * 2
		spawn_ctx = mp.get_context("spawn")
		with ProcessPoolExecutor(
			max_workers=max_workers,
			mp_context=spawn_ctx,
			initializer=_init_filter_worker,
			initargs=(filter_config_path,),
		) as executor:
			batch_iter = iter(_iter_batches(sentences, batch_size))
			pending: dict[Any, int] = {}

			while True:
				while len(pending) < max_pending:
					try:
						batch = next(batch_iter)
					except StopIteration:
						break
					candidate_count += len(batch)
					future = executor.submit(_filter_sentence_batch, batch)
					pending[future] = len(batch)

				if not pending:
					break

				done_future = next(as_completed(pending))
				done_batch_size = pending.pop(done_future)
				filtered_batch = done_future.result()
				filtered_count += len(filtered_batch)
				for row in filtered_batch:
					writer.writerow(row)
				_print_batch_progress(
					_build_progress_postfix(
						candidate_count,
						filtered_count,
						batch_size=done_batch_size,
						pending_size=len(pending),
						max_pending=max_pending,
					)
				)

	print(f"从 {candidate_count} 个备选句子中筛选出了 {filtered_count} 个符合条件的句子")


def run(src_path: str, tgt_path: str = ".", filter_config_path: str = "filter_config.json5", vocabs: list[str] = None):
	global _START_TIME
	_START_TIME = time.time()
	print("[运行模式] Linux/Other: 流式处理")
	output_file = Path(tgt_path) / "filtered_sentences.csv"
	sentences = iter_path_sentences(src_path, vocabs=vocabs)

	with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=["sentence", "source_file", "filter_methods", "match_tree", ])
		writer.writeheader()
		filter_sentences_streaming(sentences, filter_config_path, writer)


def run_linux_entry(src_path: str, tgt_path: str, filter_config_path: str, vocabs: list[str] = None):
	run(src_path, tgt_path, filter_config_path, vocabs)
