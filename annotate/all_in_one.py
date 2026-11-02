# encoding: utf8
"""已筛选句对的单次标注：空间特征 → 空间图式 → 同义判断。

替换、语法与空间义检查都在 replace/sentence_replace.py 完成，不属于 all-in-one。
与 spatial_runner.py --graph pair 读取同一份句对、复用同一序列化与查表实现。

用法：
    python annotate --engine all-in-one --corpus results/location_pairs.jsonl --limit 5
    python annotate --engine all-in-one --corpus results/location_pairs.jsonl --print-prompt
    python annotate --engine all-in-one --corpus results/direction_pairs.jsonl --corpus-type 趋向动词
    python annotate --engine all-in-one --corpus results/location_pairs.jsonl --no-llm --batch off

每个句对一次逻辑调用，不提供句法工具或句法证据。传输层重试及可选的结构化输出
补发沿用现有配置，真实请求数由回调计数。同步、原生 Batch、同步回退使用同一份消息。
"""

import argparse
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import Field, create_model

import agents as agent_factory
import batch_inference as bi
import config
from replace import corpus_pairs as cp
import file_paths
from annotate import prompt_examples as pe
import retry
from annotate import spatial_agents as sa
from annotate import spatial_graphs as sg
from annotate import spatial_runner as sr
from annotate import spatial_tools as st
import utils

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE = "all_in_one"
DEFAULT_OUT_PREFIX = "all_in_one"
DEFAULT_LIMIT = 1
PAIR_CALLS_PER_STEP = {"feature": 2, "schema": 2, "judgement": 1}


def build_verdict_model(feature_classes: list[dict]) -> type:
    """一次调用只输出三项标注，不再让模型重判两侧语法和空间义。"""
    if feature_classes:
        feature_type: Any = create_model("SemanticFeatures", **{
            c["key"]: (list[str], Field(default_factory=list, description=f"{c.get('name') or c['key']}：{c.get('description') or ''}"))
            for c in feature_classes
        })
    else:
        feature_type = dict[str, Any]
    features = create_model("SideFeatures", a=(feature_type, Field(description="原句空间特征")),
                            b=(feature_type, Field(description="替换句空间特征")))
    schemas = create_model("SideSchema", a=(sa.SchemaVerdict, Field(description="原句图式")),
                           b=(sa.SchemaVerdict, Field(description="替换句图式")))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r'Field name "schema" in "AllInOneVerdict" shadows an attribute in parent')
        return create_model("AllInOneVerdict",
                            features=(features, Field(description="第 1 步：两侧空间特征")),
                            schema=(schemas, Field(description="第 2 步：两侧图式")),
                            judgement=(sa.JudgeVerdict, Field(description="第 3 步：句对同义关系")))


def build_system_prompt(feature_agent) -> str:
    classes = feature_agent.feature_classes
    roster = sa.render_class_blocks(classes, feature_agent.prompts[sa.FEATURE_CLASS_PROMPT_FILE],
                                     usage=feature_agent.prompts[sa.FEATURE_CLASS_USAGE_FLAT_FILE]) if classes else []
    return utils.render_prompt(utils.read_prompt(PROMPT_FILE), {
        "n": f"已填写，共 {len(classes)} 类" if classes else "尚未填写，请自行归纳空间特征维度",
        "roster": "\n\n".join(roster),
    }, source=f"{PROMPT_FILE}.txt")


def print_numbered(title: str, text: str) -> None:
    print(f"\n===== {title}（共 {len(text.splitlines())} 行）=====")
    for index, line in enumerate(text.splitlines(), 1):
        print(f"{index:>4}| {line}")


def _candidates_block(label: str, focus_word: dict | None, table: dict, candidates: dict) -> str:
    head = f"{label}的 schema 候选："
    prefix = st.schema_group_key(table, focus_word) or ((focus_word or {}).get("word") or "")
    if candidates:
        return f"{head}只从下面这些里挑一个最接近的，名字原样照抄：\n{st.render_schema_candidates(candidates)}"
    return f"{head}这一类下还没有编号图式，请按「{prefix or '组名'} + 编号」的格式自行命名。"


class _Prepared(NamedTuple):
    message: str


def prepare(state: dict, ctx: "_Context", args: argparse.Namespace) -> _Prepared:
    focus_a, focus_b = state.get("focus_word"), state.get("focus_word_b")
    feature, schema = ctx.agents.need("feature"), ctx.agents.need("schema")
    parts = [sa.pair_head_block(focus_a, focus_b, corpus_type=ctx.corpus_type),
             f"原句：{state.get('sentence_a') or ''}", f"改写句：{state.get('sentence_b') or ''}",
             "两侧均已通过语法、空间自洽和空间义检查；本次只标注，不重复筛选。"]
    for stage, agent in (("feature", feature), ("schema", schema)):
        title = "空间特征" if stage == "feature" else "空间图式"
        for label, focus in (("原句", focus_a), ("改写句", focus_b)):
            block = pe.example_block(agent.example_groups, focus, stage)
            if block:
                parts.append(f"{label}的{title}示例（只供本侧本步骤参考）：\n{block}")
    for label, focus in (("原句", focus_a), ("改写句", focus_b)):
        parts.append(_candidates_block(label, focus, schema.table, st.schema_candidates(schema.table, focus)))
    parts.append("请依次给出两侧空间特征（features）、两侧图式（schema）及同义关系（judgement）。")
    return _Prepared("\n\n".join(part for part in parts if part))


class _RequestProbe(BaseCallbackHandler):
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def on_chat_model_start(self, *args, **kwargs):
        with self.lock:
            self.count += 1


class CallCounter:
    def __init__(self):
        self.rounds = self.requests = self.retried = 0
        self.lock = threading.Lock()

    def bump(self):
        with self.lock:
            self.rounds += 1

    def add_requests(self, count: int):
        with self.lock:
            self.requests += count
            self.retried += max(0, count - 1)


def call_once(agent, content: str, counter: CallCounter, retry_policy: dict | None = None) -> tuple:
    """一次逻辑调用；回调计数包含传输重试及框架结构化输出补发。"""
    counter.bump()
    total_requests = 0

    def invoke_once():
        nonlocal total_requests
        probe = _RequestProbe()
        try:
            return agent.invoke({"messages": [{"role": "user", "content": content}]},
                                config={"callbacks": [probe]})
        finally:
            total_requests += probe.count

    try:
        result = retry.call_with_retry(invoke_once, policy=retry_policy or {"max_retries": 0})
    except Exception as exc:
        counter.add_requests(total_requests)
        return None, f"调用失败：{exc!r}", total_requests
    counter.add_requests(total_requests)
    response = result.get("structured_response") if isinstance(result, dict) else None
    if response is None:
        return None, "模型没有给出结构化输出（structured_response 为空）", total_requests
    return response.model_dump(), None, total_requests


def build_call_agent(model_key: str, system_prompt: str, verdict_model: type, *, allow_retry: bool):
    return agent_factory.get_agent(model_key, [], system_prompt,
                                   response_format=ToolStrategy(verdict_model, handle_errors=allow_retry))


def resolve_schema(agent, sentence: str, features: dict, focus_word: dict | None,
                   side_dump: dict | None, *, llm_used: bool, error: str | None) -> dict:
    """本地查表优先，只有表没有结论时才采信单次调用给出的图式。"""
    local = agent.run(sentence, features, focus_word)
    if local["source"] == "table" or not llm_used:
        return local
    if error or side_dump is None:
        return {"name": None, "source": "error", "reason": error or "模型未给出图式"}
    chosen = (side_dump.get("schema_name") or "").strip()
    candidates = st.schema_candidates(agent.table, focus_word)
    prefix = st.schema_group_key(agent.table, focus_word) or ((focus_word or {}).get("word") or "")
    valid = chosen in candidates if candidates else st.is_schema_name(chosen, prefix)
    if not valid:
        return {"name": None, "source": "llm_invalid", "reason": f"LLM 给出的图式 {chosen!r} 不符合本侧候选或命名格式"}
    return {"name": chosen, "source": "llm" if candidates else "llm_free", "reason": side_dump.get("reason") or ""}


def resolve_judgement(agent, sentences: tuple, schema: dict, features: dict, focus: tuple,
                      judgement_dump: dict | None, *, llm_used: bool, error: str | None) -> dict:
    local = agent.run(sentences[0], sentences[1], schema.get("a") or {}, schema.get("b") or {},
                      features.get("a") or {}, features.get("b") or {}, focus[0], focus[1])
    if local["source"] == "table" or not llm_used:
        return local
    if error or not judgement_dump or judgement_dump.get("same_meaning") is None:
        return {"same_meaning": None, "source": "error", "lookup": local.get("lookup"),
                "reason": error or "模型没有给出 same_meaning"}
    return {"same_meaning": bool(judgement_dump["same_meaning"]), "source": "llm",
            "lookup": local.get("lookup"), "reason": judgement_dump.get("reason") or ""}


def _apply(state: dict, updates: dict) -> dict:
    return {**state, **updates, "history": [*(state.get("history") or []), *(updates.get("history") or [])]}


def process_state(state: dict, *, ctx: "_Context", args: argparse.Namespace, call_agent,
                  counter: CallCounter, precomputed: tuple | None = None) -> dict:
    started = time.perf_counter()
    state = _apply(state, sg.prepared_pair_node(state))
    ready = state.get("stage") == "semantic_ok"
    prepared, dumps, error, requests = None, None, None, 0
    if ready and (ctx.llm or args.print_prompt):
        prepared = prepare(state, ctx, args)
        if args.print_prompt:
            print_numbered(f"user message（输入第 {state['record'].get('line_no')} 行句对）", prepared.message)
        elif precomputed is not None:
            dumps, error, requests = precomputed
            if requests:
                counter.bump()
                counter.add_requests(requests)
        else:
            dumps, error, requests = call_once(call_agent, prepared.message, counter, ctx.retry_policy)
    reached = {name: ready for name in PAIR_CALLS_PER_STEP}
    if ready:
        feature_agent = ctx.agents.need("feature")
        if not ctx.llm:
            results = {side: feature_agent.run(state[f"sentence_{side}"], state["focus_word" if side == "a" else "focus_word_b"])
                       for side in ("a", "b")}
        elif error or dumps is None:
            results = {side: {"features": {}, "source": "error", "reason": error or "未返回结构化输出"} for side in ("a", "b")}
        else:
            results = {side: {"features": (dumps.get("features") or {}).get(side) or {},
                              "source": feature_agent.table_source, "reason": "",
                              "validation": sa.validate_feature_values((dumps.get("features") or {}).get(side) or {}, ctx.feature_classes)}
                       for side in ("a", "b")}
        state = _apply(state, sg.pair_feature_update(state, results))
        schemas = {side: resolve_schema(ctx.agents.need("schema"), state[f"sentence_{side}"], state["features"][side],
                                       state["focus_word" if side == "a" else "focus_word_b"],
                                       ((dumps or {}).get("schema") or {}).get(side), llm_used=ctx.llm, error=error)
                   for side in ("a", "b")}
        judgement = resolve_judgement(ctx.agents.need("judge"), (state["sentence_a"], state["sentence_b"]),
                                      schemas, state["features"], (state["focus_word"], state["focus_word_b"]),
                                      (dumps or {}).get("judgement"), llm_used=ctx.llm, error=error)
        state = _apply(state, {"schema": schemas, "judge": judgement})
    state = _apply(state, {"history": [sg._step("all_in_one", llm_used=ctx.llm and ready, error=error,
                                                elapsed=round(time.perf_counter() - started, 2))]})
    record = sg.serialize_pair(state)
    record["all_in_one"] = {
        "model": ctx.model_key, "corpus_type": ctx.corpus_type,
        "llm_calls": 1 if dumps is not None or error is not None else 0, "requests": requests,
        "feature_classes": [c["key"] for c in ctx.feature_classes],
        "prompt": {"name": PROMPT_FILE, "path": ctx.prompt_path},
        "reached": reached, "pair_llm_calls": sum(PAIR_CALLS_PER_STEP[n] for n, ok in reached.items() if ok),
        "raw": dumps, "error": error, "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    return record


class _Context(NamedTuple):
    agents: sa.AgentSet
    corpus_type: str
    model_key: str
    llm: bool
    out_path: Path
    system_prompt: str
    verdict_model: type
    feature_classes: list
    prompt_path: str
    retry_policy: dict | None = None


def build_context(args: argparse.Namespace, corpus_type: str, out_path: Path) -> _Context:
    agents = sa.build_agents(sg.GRAPHS["pair"].needs, model_key=args.model, llm=False, corpus_type=corpus_type)
    feature = agents.need("feature")
    return _Context(agents, corpus_type, args.model, not args.no_llm, out_path,
                    build_system_prompt(feature), build_verdict_model(feature.feature_classes),
                    list(feature.feature_classes), str(utils.prompt_path(PROMPT_FILE)),
                    agent_factory.request_policy(args.model) if not args.no_llm and not args.print_prompt else {"max_retries": 0})


def expand_states(record: dict, ctx: _Context, args: argparse.Namespace) -> list:
    """兼容函数名；不再展开，一条输入句对严格对应一个状态。"""
    return sg.prepared_init_states([record], ctx.agents, str(ctx.out_path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=None, help="python replace 产出的句对 JSONL")
    parser.add_argument("--corpus-type", default=None, type=sr._corpus_type_arg)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="最多标注多少条合格句对，默认 1")
    parser.add_argument("--offset", type=int, default=0, help="跳过多少条合格句对")
    parser.add_argument("--model", default=sa.DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    sr.add_batch_args(parser)
    parser.add_argument("--no-llm", action="store_true", help="不调用模型，只读句对并查表")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-prompt", action="store_true", help="打印渲染后的消息；不调用、不落盘")
    parser.add_argument("--allow-retry", action="store_true", help="允许框架为结构化输出错误补发请求")
    parser.add_argument("--no-pairs-csv", action="store_true")
    parser.add_argument("--all-candidates", action="store_true", help="兼容旧命令；现在一行一个句对，不再展开")
    parser.add_argument("--no-syntax", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-inference", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    kind = st.normalize_corpus_type(args.corpus_type)
    corpus = Path(args.corpus) if args.corpus else _PROJECT_ROOT / "results" / cp.PAIR_CORPORA[kind]
    records = cp.read_prepared_pairs(corpus, limit=args.limit, offset=args.offset)
    if any(record.get("corpus_type") != kind for record in records):
        raise ValueError("句对的 corpus_type 与 --corpus-type 不一致")
    if not records:
        print(f"没有可标注的合格句对：{corpus}；请先运行 python replace")
        return
    if args.limit is not None and len(records) < args.limit:
        print(f"提示：仅有 {len(records)} 条合格句对，少于 --limit {args.limit}；按实际数量运行")
    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "results" / DEFAULT_OUT_PREFIX
    output = Path(args.out_file) if args.out_file else st.build_output_path(out_dir, prefix=DEFAULT_OUT_PREFIX)
    cache_path = bi.resolve_cache_path(args.batch_cache, output)
    sidecars = bi.sidecar_paths(cache_path) if not args.no_llm and args.batch != "off" else ()
    file_paths.validate_run_outputs(corpus, output, st.pairs_csv_path(output) if not args.no_pairs_csv else None,
                                    overwrite=args.overwrite, extra_outputs=sidecars)
    ctx = build_context(args, kind, output)
    states = [s for record in records for s in expand_states(record, ctx, args)]
    print(f"单次标注：{len(records)} 条合格句对；仅特征、图式与同义判断；无句法证据")
    sr._print_config_tables(ctx.agents)
    if args.print_prompt:
        print_numbered("system prompt", ctx.system_prompt)
        for state in states:
            applied = _apply(state, sg.prepared_pair_node(state))
            print_numbered(f"user message（第 {state['record'].get('line_no')} 行）", prepare(applied, ctx, args).message)
        return
    call_agent = build_call_agent(args.model, ctx.system_prompt, ctx.verdict_model, allow_retry=args.allow_retry) if ctx.llm else None
    spec = sg.GRAPHS["pair"]
    writer = sr.ResultWriter(output, counter_names=spec.counter_names, tally=spec.tally, corpus_type=kind,
                             config_tables=sr._config_tables(ctx.agents),
                             csv_columns=spec.csv_columns if not args.no_pairs_csv else (),
                             model_name=args.model if ctx.llm else None, overwrite=args.overwrite,
                             input_records=len(records), expanded_states=len(states))
    counter = CallCounter()
    try:
        writer.extra_summary["request_policy"] = ctx.retry_policy
        session, keys = None, {}
        if ctx.llm and args.batch != "off":
            profile = bi.batch_profile(args.model)
            if profile.enabled:
                session = bi.BatchSession(profile, cache_path=cache_path, poll_seconds=args.batch_poll_seconds,
                                          timeout_seconds=args.batch_timeout_seconds, chunk_size=args.batch_chunk_size,
                                          retry_policy=ctx.retry_policy, resume=args.batch_resume, strict_native=args.batch == "native")
            elif args.batch == "native":
                raise SystemExit(f"错误：--batch native 不可用 —— {profile.describe()}")
            else:
                print(f"批处理：{profile.describe()}（回退同步执行）")
        if session is not None:
            items = {}
            for i, state in enumerate(states):
                applied = _apply(state, sg.prepared_pair_node(state))
                if applied.get("stage") != "semantic_ok":
                    continue
                prepared = prepare(applied, ctx, args)
                key = bi.item_key("all_in_one", prepared.message, ctx.system_prompt)
                keys[i] = key

                def sync(message=prepared.message):
                    dumps, error, _ = call_once(call_agent, message, counter, ctx.retry_policy)
                    return dumps, error

                items[key] = bi.BatchItem(key=key, agent_name="all_in_one", system_prompt=ctx.system_prompt,
                                          content=prepared.message, response_model=ctx.verdict_model, sync_call=sync)
            if items:
                session.submit_direct(list(items.values()), fallback_workers=max(1, args.workers))
            writer.extra_summary["batch"] = session.summary()
            for i, state in enumerate(states):
                hit = session.results.get(keys.get(i))
                precomputed = (hit.get("dumps"), hit.get("error"), 0) if hit is not None else (None, "批处理缺失结果", 0)
                writer.write(process_state(state, ctx=ctx, args=args, call_agent=call_agent,
                                            counter=counter, precomputed=precomputed))
        else:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                for result in pool.map(lambda s: process_state(s, ctx=ctx, args=args, call_agent=call_agent, counter=counter), states):
                    writer.write(result)
        writer.agent_stats = {name: agent.stats() for name, agent in ctx.agents.items()}
        writer.extra_summary["all_in_one_calls"] = {"rounds": counter.rounds, "requests": counter.requests, "retried": counter.retried}
        summary = writer.finish()
    finally:
        writer._close()
    print(f"完成 {summary['count']} 条；同步逻辑调用 {counter.rounds} 次，实际请求 {counter.requests} 次（含补发 {counter.retried} 次）")
    print(f"输出：{output}；汇总：{summary['summary_file']}")


if __name__ == "__main__":
    main()
