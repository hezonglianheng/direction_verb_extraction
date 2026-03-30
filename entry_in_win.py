from ltp import StnSplit
import jsonlines
from tqdm import tqdm

from pathlib import Path
import re
from typing import Any
import json

import config
import sentenceFilters


_WORKER_FILTER = None


def _init_filter_worker(filter_config_path: str):
	global _WORKER_FILTER
	generator = sentenceFilters.FilterGenerator(filter_config_path)
	_WORKER_FILTER = generator.generate_filter()


def _filter_sentence(item: dict[str, Any]) -> dict[str, Any] | None:
	if _WORKER_FILTER is None:
		raise RuntimeError("过滤进程尚未初始化过滤器")

	curr_sentence: str = item.get("sentence", "")
	judge_result, explanation = _WORKER_FILTER.explain(curr_sentence)
	if not judge_result:
		return None

	filter_methods = explanation.get("methods", [])
	match_tree: list[sentenceFilters.MatchNode] = explanation.get("match_tree", [])
	simplified_tree = sentenceFilters.simplify_tree(match_tree)
	
	# 获取CWS结果并替换原本的sentence
	cws_result = sentenceFilters.cache.get_task_value(curr_sentence, config.CWS)
	
	return item | {"sentence": cws_result, "filter_methods": filter_methods, "match_tree": simplified_tree}


def _build_progress_postfix(candidate_count: int, filtered_count: int, *, batch_size: int | None = None) -> dict[str, str | int]:
	hit_rate = (filtered_count / candidate_count * 100) if candidate_count else 0.0
	postfix: dict[str, str | int] = {
		"cand": candidate_count,
		"hit": filtered_count,
		"rate": f"{hit_rate:.2f}%",
	}
	if batch_size is not None:
		postfix["batch"] = batch_size
	return postfix


def get_sentences_list(article: str, vocabs: list[str] = None) -> list[str]:
	sentences = StnSplit().split(article)
	sentences = [s for s in sentences if config.SENTENCE_LEN_LOWER_BOUND <= len(s) <= config.SENTENCE_LEN_UPPER_BOUND]
	sentences = [s for s in sentences if s.isascii() or re.search(r"[\u4e00-\u9fff]", s)]
	sentences = [s for s in sentences if "【【【缺文】】】" not in s]
	sentences = [s.strip() for s in sentences]
	if vocabs:
		sentences = [s for s in sentences if any(vocab in s for vocab in vocabs)]
	return [s.split()[-1] for s in sentences]


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


def filter_sentences_windows_in_memory(sentences: list[dict[str, Any]], filter_config_path: str) -> list[dict[str, Any]]:
	filtered_sentences: list[dict[str, Any]] = []
	candidate_count = 0
	filtered_count = 0

	_init_filter_worker(filter_config_path)
	with tqdm(
		sentences,
		total=len(sentences),
		desc="Filtering sentences (Windows memory)",
		position=1,
		leave=True,
		dynamic_ncols=True,
		unit="sentence",
	) as pbar:
		for sentence_item in pbar:
			candidate_count += 1
			filtered_item = _filter_sentence(sentence_item)
			if filtered_item is not None:
				filtered_count += 1
				filtered_sentences.append(filtered_item)
			pbar.set_postfix(
				_build_progress_postfix(
					candidate_count,
					filtered_count,
				),
				refresh=False,
			)

	print(f"从 {candidate_count} 个备选句子中筛选出了 {filtered_count} 个符合条件的句子")
	return filtered_sentences


def run_windows_entry(src_path: str, tgt_path: str, filter_config_path: str, vocabs: list[str] = None):
	output_file = Path(tgt_path) / "filtered_sentences.jsonl"
	print("[运行模式] Windows: 整文件读入内存处理")
	sentences = load_path_sentences_in_memory(src_path, vocabs)
	filtered_sentences = filter_sentences_windows_in_memory(sentences, filter_config_path)

	with open(output_file, "w", encoding="utf-8") as f:
		for row in filtered_sentences:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")
