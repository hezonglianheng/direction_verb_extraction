# encoding: utf8
"""语料替换阶段：原句及三元组 → 机械替换 → 检查替换句 → 一行一个句对。

示例：python replace --corpus results/location_corpus.jsonl
                    --corpus-type 方位词 --out-file results/location_pairs.jsonl
旧语料缺少完整原句判定时先补筛；标注阶段不再负责替换或语法/空间义检查。

限额抽样（X 为送入替换的可用原句数）：
    python replace --corpus ... --limit 500                          # 按顺序取前 500 条
    python replace --corpus ... --limit 500 --limit-mode random --seed 0
    python replace --corpus ... --limit 500 --limit-mode proportional --seed 0
    python replace --corpus ... --reference-file filtered.jsonl      # 按已筛选文件逐条对回
random/proportional 只额外读一遍语料做计数，不把 GB 级语料读进内存；同一 seed 可复现。
aligned（--reference-file）按行号/句子内容严格对回同一份源语料，找不到或一对多会报错。
"""

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

import config
from replace import corpus_pairs as cp
from replace import limit_sampler
from replace import replacement_log
import file_paths
from extract import local_model_service
from extract import sentence_screening as ss
from annotate import spatial_agents as sa
from annotate import spatial_tools as st


def replacement_grounding(grounding: dict | None, index: int, replacement: str) -> dict | None:
    """保持原分词切片坐标，更新机械替换后所有落在该位置的角色词；不重跑句法分析。"""
    if not isinstance(grounding, dict):
        return None
    result = copy.deepcopy(grounding)
    result.update(word=replacement, index=index, source="mechanical_replacement")
    for role in ("target", "reference", "verb"):
        if isinstance(result.get(role), dict) and result[role].get("index") == index:
            result[role]["word"] = replacement
    return result


class PairPreparer:
    def __init__(self, *, corpus_type: str, rules: dict | None = None,
                 screener: ss.SentenceScreener | None = None, model_key: str = config.SCREEN_MODEL,
                 llm: bool = True, workers: int | None = None, batch_size: int | None = None):
        self.corpus_type = st.normalize_corpus_type(corpus_type)
        self.alter = sa.AlterAgent(corpus_type=self.corpus_type, rules=rules, llm=False)
        path = st.FOCUS_VOCAB_SPECS[self.corpus_type][0]
        self.screener = screener if screener is not None else ss.SentenceScreener(
            family=Path(path).stem, config_path=str(path), kind=self.corpus_type,
            model_key=model_key, llm=llm, workers=workers, batch_size=batch_size,
        )
        self.counts = Counter()

    def _item(self, sentence: str, focus: dict, row: dict) -> dict:
        grounding = focus.get("grounding")
        return {
            "sentence": sentence, "target": focus,
            "tree": [{"index": focus["index"], "word": focus["word"], "children": []}],
            "filter_methods": row.get("filter_methods") or [],
            "grounding": {"candidates": [grounding]} if grounding else None,
        }

    def prepare_rows(self, rows: list[dict]) -> list[dict]:
        originals, pending, slots = [], [], []
        for row in rows:
            self.counts["originals"] += 1
            # 在生成稳定键之前规范化，而不是等读句对时再补行号。不能修改调用方原记录。
            source_line_no = st.record_line_no(row)
            if source_line_no is not None:
                row = {**row, "source_line_no": source_line_no}
            tokens = row.get("sentence")
            if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
                self.counts["bad_record"] += 1
                continue
            focuses = st.pick_focus_words(row)
            for focus_order, focus in enumerate(focuses, 1):
                focus = copy.deepcopy(focus)
                focus["kind"] = self.corpus_type
                known = focus.get("screening")
                original = {"row": row, "focus": focus, "verdict": known, "focus_order": focus_order}
                originals.append(original)
                if ss.verdict_status(known) not in ("true", "false"):
                    pending.append(self._item(st.join_slice(tokens), focus, row))
                    slots.append(original)
        for original, check in zip(slots, self.screener.screen_items(pending)):
            original["verdict"] = check

        pairs, replacement_items = [], []
        # 一条原句按全部合格位置一次展开，保留同句候选的全局序号。
        by_row: dict[int, list[dict]] = {}
        for original in originals:
            if ss.verdict_passes(original["verdict"]):
                original["focus"]["screening"] = original["verdict"]
                by_row.setdefault(id(original["row"]), []).append(original)
            else:
                self.counts["original_" + ss.verdict_status(original["verdict"])] += 1
        for group in by_row.values():
            row = group[0]["row"]
            focuses = [item["focus"] for item in group]
            focus_orders = {item["focus"]["index"]: item["focus_order"] for item in group}
            candidates, _ = self.alter.expand_positions(row["sentence"], focuses)
            if not candidates:
                self.counts["alter_missed"] += 1
            for candidate in candidates:
                # 被筛掉的位置仍占原先的序号，避免 ID 的目标位置段指向另一个词。
                candidate["focus_order"] = focus_orders[candidate["focus_index"]]
                focus_a = copy.deepcopy(candidate["focus_word"])
                focus_b = st.replacement_focus_word(focus_a, candidate["replacement"], row.get("filter_methods") or [])
                focus_b["kind"] = self.corpus_type
                grounding_b = replacement_grounding(focus_a.get("grounding"), candidate["focus_index"], candidate["replacement"])
                if grounding_b is not None:
                    focus_b["grounding"] = grounding_b
                check_a = copy.deepcopy(focus_a["screening"])
                pair = {
                    **copy.deepcopy(row), "pair_schema_version": cp.PAIR_SCHEMA_VERSION,
                    "corpus_type": self.corpus_type, "focus_word": focus_a, "focus_word_b": focus_b,
                    "candidate": {**copy.deepcopy(candidate), "index": candidate["candidate_index"],
                                  "total": candidate["candidate_total"]},
                    "candidate_key": st.candidate_key(st.sentence_id(row), candidate["focus_index"], candidate["replacement"]),
                    "pair": {
                        "a": {"sentence": st.join_slice(row["sentence"]), "slice": list(row["sentence"]),
                              "grounding": copy.deepcopy(focus_a.get("grounding")), "role": "原句", "alter": None},
                        "b": {"sentence": candidate["sentence"], "slice": list(candidate["slice"]),
                              "grounding": copy.deepcopy(grounding_b), "role": "改写句",
                              "alter": {key: copy.deepcopy(candidate[key]) for key in (
                                  "word", "replacement", "source", "group", "focus_index", "candidate_index", "candidate_total")}},
                    },
                    "pair_screening": {"a": check_a},
                }
                pairs.append(pair)
                replacement_items.append(self._item(candidate["sentence"], focus_b, row))
        for pair, check_b in zip(pairs, self.screener.screen_items(replacement_items)):
            pair["focus_word_b"]["screening"] = check_b
            checks = pair["pair_screening"]
            checks.update(b=check_b, status=cp.pair_status(checks["a"], check_b))
            self.counts[checks["status"]] += 1
        return pairs


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是大于 0 的整数，收到 {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"必须是大于 0 的整数，收到 {value!r}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"必须是大于等于 0 的整数，收到 {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"必须是大于等于 0 的整数，收到 {value!r}")
    return number


def _iter_source_rows(path: Path):
    """不限额时的读法：保持原行为，只跳过空行、拒绝句对输入。"""
    with path.open(encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "pair_schema_version" in row:
                raise ValueError("输入已是句对，请勿再次替换；将它交给 python annotate 或其 --engine all-in-one 模式")
            yield line_no, row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="提取阶段的原句 JSONL")
    parser.add_argument("--corpus-type", default=st.DEFAULT_CORPUS_TYPE, type=st.normalize_corpus_type)
    parser.add_argument("--out-file", default=None, help="句对 JSONL，默认 results/<类型>_pairs.jsonl")
    parser.add_argument("--alter-rules", default=None, help="覆盖本词性的机械替换规则表")
    parser.add_argument("--model", default=config.SCREEN_MODEL, help="语法、空间义检查的模型配置名")
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"并发检查数；不传时用模型配置 screen_workers，再回退到全局默认 {config.SCREEN_WORKERS}",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"每次 LLM 请求放入多少条判定项；不传时用模型配置 screen_batch_size，"
        f"再回退到全局默认 {config.SCREEN_BATCH_SIZE}",
    )
    parser.add_argument("--chunk-size", type=int, default=100, help="每批读入多少条原句")
    parser.add_argument(
        "--limit", type=_positive_int, default=None,
        help="只把最多 X 条可用原句送入替换（X>0）；不传则处理全部语料。"
             "可用原句指 true/no_llm 且句子形状合法的记录，坏记录与不可判定记录不占名额",
    )
    parser.add_argument(
        "--limit-mode", choices=limit_sampler.LIMIT_MODES, default=None,
        help="--limit 的抽取方式：sequential=按文件顺序取前 X 条（默认）；"
             "random=从全部可用原句中随机抽 X 条；"
             "proportional=在带本类关注词的可用原句中按主关注词比例分配 X 条"
             "（配额描述选中记录的分层，句内其他位置仍会展开）；"
             "aligned=按 --reference-file 逐条对回源语料",
    )
    parser.add_argument(
        "--seed", type=_non_negative_int, default=None,
        help="random/proportional 的随机种子，默认 0；同一份语料与种子可复现同一批记录。"
             "仅与 --limit 一起使用；aligned 不使用随机种子",
    )
    parser.add_argument(
        "--reference-file", default=None,
        help="已筛选文件（JSONL）：按行号/句子内容把它逐条对回 --corpus。"
             "给出后自动进入 aligned 模式；--limit 可省略（全部对齐），给了则只取源文件顺序的前 X 条。"
             "reference 中任何一条找不到或一对多都会报错",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="替换日志目录；默认项目根目录下的 log/（不存在会自动创建）",
    )
    parser.add_argument("--no-llm", action="store_true", help="不调用模型；未检查的句对不可进入标注")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.reference_file is not None:
        if args.limit_mode not in (None, "aligned"):
            parser.error("--reference-file 不能与 --limit-mode sequential/random/proportional 同时使用")
        if args.seed is not None:
            parser.error("--reference-file 不对齐随机过程，不能与 --seed 同时使用")
        args.limit_mode = "aligned"
        args.seed = limit_sampler.DEFAULT_SEED
    elif args.limit_mode == "aligned":
        parser.error("--limit-mode aligned 必须同时给出 --reference-file")
    elif args.limit is None:
        if args.limit_mode is not None or args.seed is not None:
            parser.error("--limit-mode 与 --seed 只能在给出 --limit 时使用")
        args.limit_mode = limit_sampler.DEFAULT_MODE
        args.seed = limit_sampler.DEFAULT_SEED
    else:
        if args.limit_mode is None:
            args.limit_mode = limit_sampler.DEFAULT_MODE
        if args.seed is None:
            args.seed = limit_sampler.DEFAULT_SEED
    return args


def main(argv=None):
    args = parse_args(argv)
    started_at = time.time()
    source = Path(args.corpus)
    reference = Path(args.reference_file) if args.reference_file else None
    output = Path(args.out_file) if args.out_file else Path(config._PROJECT_ROOT) / "results" / cp.PAIR_CORPORA[args.corpus_type]
    file_paths.validate_output_paths([source, *([reference] if reference is not None else [])], [output])
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件 {output} 已存在；请换路径或显式加 --overwrite")
    preparer = PairPreparer(corpus_type=args.corpus_type,
                            rules=st.load_alter_rules(args.alter_rules) if args.alter_rules else None,
                            model_key=args.model, llm=not args.no_llm, workers=args.workers,
                            batch_size=args.batch_size)
    # 抽样在消费 records 时才真正读文件；random/proportional 的第一遍计数也发生在
    # 这里，不改变 --chunk-size 的分批边界，因此同一 seed 的选中结果与批次大小无关。
    if args.limit_mode == "aligned":
        selection = limit_sampler.select_aligned_records(
            source,
            reference_file=args.reference_file,
            limit=args.limit,
            corpus_type=args.corpus_type,
        )
    else:
        selection = limit_sampler.select_records(
            source,
            limit=args.limit,
            mode=args.limit_mode,
            seed=args.seed,
            corpus_type=args.corpus_type,
        ) if args.limit is not None else None
    # 本地小模型（api_keys.json 里该模型带 local_service 块）在启动时拉起服务，
    # 退出（含异常/Ctrl-C）时关闭；远程模型或 --no-llm 时是空上下文。
    with local_model_service.for_model(
        args.model, enabled=not args.no_llm and preparer.screener.enabled
    ):
        preparer.screener.warm()
        output.parent.mkdir(parents=True, exist_ok=True)
        with file_paths.atomic_write_text(output) as writer:
            batch = []

            def emit():
                for pair in preparer.prepare_rows(batch):
                    writer.write(json.dumps(pair, ensure_ascii=False) + "\n")

            rows = selection.records if selection is not None else _iter_source_rows(source)
            for line_no, row in rows:
                source_line_no = st.record_line_no(row)
                row["source_line_no"] = source_line_no if source_line_no is not None else line_no
                row["line_no"] = line_no
                batch.append(row)
                if len(batch) >= max(1, args.chunk_size):
                    emit()
                    batch = []
            if batch:
                emit()
    print(f"句对输出：{output}；统计：{dict(preparer.counts)}")
    limit_stats = selection.stats if selection is not None else None
    if limit_stats is not None:
        limit_text = "全部" if limit_stats["limit"] is None else limit_stats["limit"]
        print(
            f"限制抽样：模式={limit_stats['mode']} 上限={limit_text} "
            f"候选={limit_stats['candidates']} 选中={limit_stats['selected']} "
            f"种子={limit_stats['seed']}"
        )
        if limit_stats.get("mode") == "aligned":
            print(
                f"参考文件：{limit_stats.get('reference_file')} "
                f"reference 记录={limit_stats.get('reference_records')} "
                f"已对齐={limit_stats.get('matched_reference_records')} "
                f"源记录={limit_stats.get('matched_source_records')}"
            )
        if limit_stats["skipped_no_word"]:
            print(
                f"注意：{limit_stats['skipped_no_word']} 条可用记录取不到本类关注词，"
                "未进入比例抽样池"
            )
        if limit_stats["limit"] is not None and limit_stats["selected"] < limit_stats["limit"]:
            print(
                f"注意：可用量不足，实际只选中 {limit_stats['selected']} 条"
                f"（上限 {limit_stats['limit']} 条）"
            )

    stats_fn = getattr(preparer.screener, "stats", None)
    screening_stats = {preparer.corpus_type: stats_fn()} if callable(stats_fn) else {}
    replacement_log.write_log(
        corpus=str(source),
        output=str(output),
        corpus_type=args.corpus_type,
        model_key=args.model,
        counts=dict(preparer.counts),
        screening_stats=screening_stats,
        limit_stats=limit_stats,
        log_dir=args.log_dir,
        started_at=started_at,
        finished_at=time.time(),
    )


if __name__ == "__main__":
    main()
