from ltp import StnSplit
import jsonlines
from tqdm import tqdm

from pathlib import Path
import re
from typing import Any
import json

import config
import sentenceFilters


_WORKER_FILTERS: list[tuple[str, Any]] = []
"""[(规则族名, 规则过滤器)]，一个进程内同时持有多套规则"""

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
		simplified_tree = sentenceFilters.simplify_tree(match_tree)

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
	sentences = StnSplit().split(article)
	sentences = [s for s in sentences if config.SENTENCE_LEN_LOWER_BOUND <= len(s) <= config.SENTENCE_LEN_UPPER_BOUND]
	sentences = [s for s in sentences if s.isascii() or re.search(r"[\u4e00-\u9fff]", s)]
	sentences = [s for s in sentences if "【【【缺文】】】" not in s]
	sentences = [s.strip() for s in sentences]
	if vocabs:
		sentences = [s for s in sentences if any(vocab in s for vocab in vocabs)]
	# 纯空白片段在 strip 后为空串，必须剔除（旧写法 s.split()[-1] 会在此抛 IndexError）
	return [s for s in sentences if s]


def load_jsonl_file_sentences_in_memory(input_file: str, position: int = 0, vocabs: list[str] = None) -> list[dict[str, Any]]:
	extracted_count = 0
	extracted_sentences: list[dict[str, Any]] = []
	with jsonlines.open(input_file) as reader:
		records = list(reader)
		for obj in tqdm(
			records,
			total=len(records),
			desc=f"Processing {Path(input_file).name}",
			position=position,
			leave=True,
			dynamic_ncols=True,
		):
			article: str = obj["content"]
			sentence_list = get_sentences_list(article, vocabs)
			for sentence in sentence_list:
				extracted_count += 1
				extracted_sentences.append({"sentence": sentence, "source_file": input_file})
	print(f"从文件 {input_file} 中提取了 {extracted_count} 个句子")
	return extracted_sentences


def load_path_sentences_in_memory(src_path: str, vocabs: list[str] = None) -> list[dict[str, Any]]:
	input_path = Path(src_path)
	all_sentences: list[dict[str, Any]] = []

	if input_path.is_file():
		return load_jsonl_file_sentences_in_memory(src_path, position=0, vocabs=vocabs)

	if input_path.is_dir():
		files = list(input_path.glob("*.jsonl"))
		for i, file in enumerate(files):
			all_sentences.extend(load_jsonl_file_sentences_in_memory(str(file), position=i, vocabs=vocabs))
		return all_sentences

	raise ValueError(f"输入路径 {src_path} 既不是文件也不是目录")


def filter_sentences_windows_in_memory(sentences: list[dict[str, Any]], filter_config_paths: Any) -> list[list[dict[str, Any]]]:
	"""一次遍历、多套规则，各自把命中句累积在一个列表里

	Returns:
		list[list[dict[str, Any]]]: 与 filter_config_paths 同序的命中结果
	"""
	families = [Path(path).stem for path in _as_list(filter_config_paths)]
	filtered_sentences: list[list[dict[str, Any]]] = [[] for _ in families]
	candidate_count = 0
	hit_counts: dict[str, int] = {ANY_HIT: 0}
	hit_counts.update({family: 0 for family in families})

	_init_filter_worker(filter_config_paths)
	batch_size = max(1, int(config.FILTER_BATCH_SIZE))
	with tqdm(
		total=len(sentences),
		desc="Filtering sentences (Windows memory)",
		position=1,
		leave=True,
		dynamic_ncols=True,
		unit="sentence",
	) as pbar:
		# 分块「先整批预填充 LTP 结果，再逐句走规则引擎」：逐句调用 pipeline 实测慢约 4 倍。
		# 必须分块而不能一次性预填充，否则整批会被 LRU 在读取前淘汰。
		# 多套规则共用这份缓存，故 prefill 每批只做一次，不会重复推理。
		for start in range(0, len(sentences), batch_size):
			batch = sentences[start : start + batch_size]
			sentenceFilters.cache.prefill(item.get("sentence", "") for item in batch)

			for sentence_item in batch:
				candidate_count += 1
				item_results = _filter_sentence(sentence_item)
				if any(result is not None for result in item_results):
					hit_counts[ANY_HIT] += 1
				for idx, family in enumerate(families):
					if item_results[idx] is not None:
						hit_counts[family] += 1
						filtered_sentences[idx].append(item_results[idx])
				pbar.update(1)
				pbar.set_postfix(
					_build_progress_postfix(
						candidate_count,
						hit_counts,
					),
					refresh=False,
				)

	print(f"从 {candidate_count} 个备选句子中筛选出了 {hit_counts[ANY_HIT]} 个符合条件的句子")
	for family in families:
		print(f"    {family}：{hit_counts[family]} 个句子")
	return filtered_sentences


def run_windows_entry(src_path: str, filter_config_paths: Any, output_paths: Any, vocabs: list[str] = None):
	print("[运行模式] Windows: 整文件读入内存处理")
	filter_config_paths = _as_list(filter_config_paths)
	paths = [Path(p) for p in _as_list(output_paths)]
	if len(filter_config_paths) != len(paths):
		raise ValueError(f"规则套数（{len(filter_config_paths)}）与输出文件数（{len(paths)}）不一致")

	sentences = load_path_sentences_in_memory(src_path, vocabs)
	filtered_sentences = filter_sentences_windows_in_memory(sentences, filter_config_paths)

	for path, rows in zip(paths, filtered_sentences):
		with open(path, "w", encoding="utf-8") as f:
			for row in rows:
				f.write(json.dumps(row, ensure_ascii=False) + "\n")
