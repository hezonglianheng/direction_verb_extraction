# encoding: utf8
"""语料替换阶段的实现：机械替换 → 检查替换句 → 一行一个句对。

命令行参数、帮助文案与抽样说明见 replace.cli；本模块不再重复维护一份。
"""

if __name__ == "__main__":
    # 直接运行实现模块时也先走轻量 CLI，--help 不会触发下面的模型依赖导入。
    from replace import cli
    raise SystemExit(cli.main())

import copy
import json
import time
from collections import Counter
from pathlib import Path

import config
from replace import corpus_pairs as cp
from replace import limit_sampler
from replace import replacement_log
from replace.cli import parse_args
from filter_funcs import VERDICT_FOCUS_ORDER_KEY
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
            forced_focus_order = row.get(VERDICT_FOCUS_ORDER_KEY)
            if (
                len(focuses) != 1
                or isinstance(forced_focus_order, bool)
                or not isinstance(forced_focus_order, int)
                or forced_focus_order < 1
            ):
                forced_focus_order = None
            for focus_order, focus in enumerate(focuses, 1):
                focus = copy.deepcopy(focus)
                focus["kind"] = self.corpus_type
                known = focus.get("screening")
                original = {
                    "row": row,
                    "focus": focus,
                    "verdict": known,
                    "focus_order": forced_focus_order or focus_order,
                }
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
                pair_row = {
                    key: value for key, value in row.items()
                    if key != VERDICT_FOCUS_ORDER_KEY
                }
                pair = {
                    **copy.deepcopy(pair_row), "pair_schema_version": cp.PAIR_SCHEMA_VERSION,
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


def run(args):
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
            reference_level=getattr(args, "reference_level", "record"),
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
    pending_unmatched_details: list = []
    if limit_stats is not None:
        limit_text = "全部" if limit_stats["limit"] is None else limit_stats["limit"]
        print(
            f"限制抽样：模式={limit_stats['mode']} 上限={limit_text} "
            f"候选={limit_stats['candidates']} 选中={limit_stats['selected']} "
            f"种子={limit_stats['seed']}"
        )
        if limit_stats.get("mode") == "aligned":
            level = limit_stats.get("reference_level", "record")
            extra = ""
            if level == "verdict":
                extra = (
                    f" verdict={limit_stats.get('reference_verdicts')}"
                    f" 已对齐verdict={limit_stats.get('matched_reference_verdicts')}"
                    f" 展开记录={limit_stats.get('expanded_records')}"
                )
            print(
                f"参考文件：{limit_stats.get('reference_file')} "
                f"reference 记录={limit_stats.get('reference_records')} "
                f"已对齐={limit_stats.get('matched_reference_records')} "
                f"源记录={limit_stats.get('matched_source_records')} "
                f"级别={level}{extra}"
            )
            unmatched_details = limit_stats.get("unmatched_details") or []
            if unmatched_details:
                print(f"注意：{len(unmatched_details)} 条 reference 未能对齐，已跳过；明细如下：")
                reasons = limit_stats.get("unmatched_reason_counts") or {}
                if reasons:
                    print(
                        "  原因："
                        + "，".join(f"{key}={value}" for key, value in sorted(reasons.items()))
                    )
                statuses = limit_stats.get("unmatched_status_counts") or {}
                if statuses:
                    print(
                        "  状态分布："
                        + "，".join(f"{key}={value}" for key, value in sorted(statuses.items()))
                    )
                for detail in unmatched_details[:10]:
                    text = detail.get("text") if isinstance(detail, dict) else str(detail)
                    print(f"  {text}")
                if len(unmatched_details) > 10:
                    pending_unmatched_details = unmatched_details[10:]
                    print(f"  ...（其余 {len(pending_unmatched_details)} 条详见替换日志）")
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
    log_path = replacement_log.write_log(
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
    if log_path is None and pending_unmatched_details:
        print(
            f"替换日志写入失败，补打其余 {len(pending_unmatched_details)} 条未对齐明细："
        )
        for detail in pending_unmatched_details:
            text = detail.get("text") if isinstance(detail, dict) else str(detail)
            print(f"  {text}")


def main(argv=None):
    """兼容旧调用：解析参数后执行替换；实现主体见 run(args)。"""
    return run(parse_args(argv))

