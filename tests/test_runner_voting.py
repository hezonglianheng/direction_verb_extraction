"""spatial_runner 的多模型投票接线测试。

这里不跑真实模型/LTP，只把外部引擎换成最小桩，验证：
    - --models 真正并行跑多个模型，原始结果分文件；
    - 加权多数、平票舍弃、不完整投票在落盘结果与 summary 里的形状；
    - 单模型路径仍然写原来的文件、不产生 .models 目录；
    - 最终结果仍是 ResultWriter 的排序/汇总产物。
"""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIGS = {
    "a": {"voting_weight": 1},
    "b": {"voting_weight": 1},
}
LOAD_CALLS = []


def jsonl_append(record, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def jsonl_read(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def jsonl_write(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class FakeAgent:
    def __init__(self, model_key):
        self.model_key = model_key

    def warm(self):
        pass

    def describe(self):
        return f"fake:{self.model_key}"

    def stats(self):
        return {
            "roundtrips": 1,
            "failed": 0,
            "cache_hits": 0,
            "retries": 0,
            "requests": 1,
            "elapsed_seconds": 0.0,
            "budget_exceeded": 0,
        }

    def config_report(self):
        return None


class FakeAgentSet(dict):
    def __init__(self, model_key):
        super().__init__({"fake": FakeAgent(model_key)})
        self.model_key = model_key

    def need(self, name):
        return self[name]

    @contextlib.contextmanager
    def budget_scope(self):
        yield


class FakeGraph:
    """按模型名产出不同判定；顺带记录并发峰值"""

    active = 0
    max_active = 0
    lock = threading.Lock()

    def __init__(self, agents, writer):
        self.model_key = agents.model_key
        self.writer = writer

    def invoke(self, state, config):
        with FakeGraph.lock:
            FakeGraph.active += 1
            FakeGraph.max_active = max(FakeGraph.max_active, FakeGraph.active)
        try:
            time.sleep(0.05)
            same = self.model_key != "b"
            record = {
                "line_no": state["record"]["line_no"],
                "candidate": {
                    "index": 0,
                    "focus_index": 1,
                    "replacement": "上",
                },
                "reality": {"is_spatial_scene": True},
                "semantic": {"a": {"is_fluent": True}, "b": {"is_fluent": True}},
                "judgement": {
                    "same_meaning": same,
                    "source": "llm",
                    "reason": "",
                },
                "status": "ok",
                "admission": {"is_admitted": True, "review_required": False, "reasons": []},
            }
            self.writer.write(record)
        finally:
            with FakeGraph.lock:
                FakeGraph.active -= 1
        return {}

    def get_graph(self):
        pass


class FakeSpec:
    name = "pair"
    prefix = "fake_pairs"
    doc = "fake"
    needs = ()
    counter_names = ()
    csv_columns = ()
    tally = staticmethod(lambda record, counters: None)

    @staticmethod
    def plan(records, agents):
        return []

    @staticmethod
    def gate_queries(records, agents):
        return []

    @staticmethod
    def init_states(records, agents, output_file):
        return [
            {
                "record": record,
                "output_file": output_file,
                "focus_words": [],
                "candidate_index": 0,
            }
            for record in records
        ]

    @staticmethod
    def build(agents, writer):
        return FakeGraph(agents, writer)


def load_runner():
    tools = ModuleType("spatial_tools")
    tools.CORPUS_TYPE_CORPORA = {"方位词": "corpus.jsonl"}
    tools.CORPUS_TYPE_DIRS = {"方位词": "localization"}
    tools.DEFAULT_CORPUS_TYPE = "方位词"
    tools.RESULT_SCHEMA_VERSION = "test"
    tools.append_jsonl = jsonl_append
    tools.read_jsonl = jsonl_read
    tools.write_jsonl = jsonl_write
    tools.pairs_csv_path = lambda path: Path(path).with_suffix(".csv")
    tools.write_pairs_csv = lambda records, path, columns: (0, 0)
    tools.build_output_path = lambda out_dir, prefix: Path(out_dir) / f"{prefix}.jsonl"
    tools.normalize_corpus_type = lambda value: value or "方位词"
    tools.read_corpus = lambda path, limit=100, offset=0, judge_only=False: [
        {"line_no": index + 1, "sentence": ["上", "山"]}
        for index in range(2)
    ]
    tools.load_focus_vocab = lambda: {"方位词": {"上"}}
    tools.prefill_ltp = lambda sentences: 0
    tools.cache_ready = lambda sentence: True
    tools.load_alter_rules = lambda path: {}

    agents_mod = ModuleType("agents")
    agents_mod.DEFAULT_MODEL = "a"

    def load_model_config(name):
        LOAD_CALLS.append(name)
        return dict(MODEL_CONFIGS[name])

    agents_mod.load_model_config = load_model_config
    agents_mod.request_policy = lambda name: {"max_retries": 0}

    sa = ModuleType("spatial_agents")
    sa.DEFAULT_MODEL = "a"
    sa.build_agents = lambda needs, **kwargs: FakeAgentSet(kwargs["model_key"])

    sg = ModuleType("spatial_graphs")
    sg.GRAPHS = {"pair": FakeSpec}
    sg.GraphSpec = object
    sg.get_graph = lambda name: FakeSpec
    sg.record_order_key = lambda record: (
        record.get("line_no", -1),
        (record.get("candidate") or {}).get("index", 0),
    )

    spec = importlib.util.spec_from_file_location("_test_spatial_runner", ROOT / "spatial_runner.py")
    module = importlib.util.module_from_spec(spec)
    replacements = {
        "spatial_tools": tools,
        "agents": agents_mod,
        "spatial_agents": sa,
        "spatial_graphs": sg,
        "_test_spatial_runner": module,
    }
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    module._warn_corpus_type_mismatch = lambda path, kind: None
    return module


class RunnerVotingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="runner-voting-"))
        cls.runner = load_runner()

    def setUp(self):
        MODEL_CONFIGS["a"]["voting_weight"] = 1
        MODEL_CONFIGS["b"]["voting_weight"] = 1

    def run_main(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.runner.main(
                [
                    "--corpus",
                    str(self.tmp / "corpus.jsonl"),
                    "--limit",
                    "2",
                    "--no-pairs-csv",
                    *extra,
                ]
            )
        return out.getvalue()

    def test_two_models_tie_is_discarded_and_raw_files_are_separate(self):
        out_path = self.tmp / "tie.jsonl"
        self.run_main("--models", "a", "b", "--out-file", str(out_path), "--model-workers", "2")
        final = jsonl_read(out_path)
        self.assertEqual(len(final), 2)
        self.assertTrue(all(record["status"] == "discarded" for record in final))
        self.assertTrue(all(record["voting"]["status"] == "tie" for record in final))
        self.assertTrue(all(record["admission"]["review_required"] for record in final))

        raw_dir = self.tmp / "tie.models"
        self.assertTrue((raw_dir / "00_a.jsonl").exists())
        self.assertTrue((raw_dir / "01_b.jsonl").exists())
        a_records = jsonl_read(raw_dir / "00_a.jsonl")
        b_records = jsonl_read(raw_dir / "01_b.jsonl")
        self.assertEqual(len(a_records), 2)
        self.assertEqual(len(b_records), 2)
        self.assertEqual({record["model_name"] for record in a_records}, {"a"})
        self.assertEqual({record["model_name"] for record in b_records}, {"b"})

        summary = json.loads((self.tmp / "tie.summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["model_name"], "voting")
        self.assertEqual(summary["voting"]["discarded"], 2)
        self.assertEqual(summary["voting"]["tie"], 2)
        self.assertEqual(summary["model_names"], ["a", "b"])

    def test_weighted_majority_wins(self):
        MODEL_CONFIGS["a"]["voting_weight"] = 2
        out_path = self.tmp / "weighted.jsonl"
        self.run_main("--models", "a", "b", "--out-file", str(out_path))
        final = jsonl_read(out_path)
        self.assertEqual(len(final), 2)
        self.assertTrue(all(record["status"] == "ok" for record in final))
        self.assertTrue(all(record["voting_label"] == "same" for record in final))
        self.assertTrue(all(record["voting"]["winner_model"] == "a" for record in final))
        self.assertTrue(all(record["voting"]["scores"] == {"same": 2, "different": 1} for record in final))
        summary = json.loads((self.tmp / "weighted.summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["voting"]["accepted"], 2)
        self.assertEqual(summary["voting_weights"], {"a": 2, "b": 1})

    def test_models_run_in_parallel(self):
        FakeGraph.max_active = 0
        out_path = self.tmp / "parallel.jsonl"
        self.run_main("--models", "a", "b", "--out-file", str(out_path), "--model-workers", "2")
        self.assertGreaterEqual(FakeGraph.max_active, 2)

    def test_single_model_keeps_original_paths(self):
        out_path = self.tmp / "single.jsonl"
        self.run_main("--model", "a", "--out-file", str(out_path))
        self.assertTrue(out_path.exists())
        self.assertFalse((self.tmp / "single.models").exists())
        summary = json.loads((self.tmp / "single.summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["model_name"], "a")
        self.assertNotIn("voting", summary)

    def test_models_with_one_name_is_single_mode(self):
        out_path = self.tmp / "one.jsonl"
        self.run_main("--models", "a", "--out-file", str(out_path))
        self.assertTrue(out_path.exists())
        self.assertFalse((self.tmp / "one.models").exists())

    def test_single_no_llm_does_not_need_model_config(self):
        LOAD_CALLS.clear()
        out_path = self.tmp / "no_llm.jsonl"
        self.run_main("--model", "a", "--no-llm", "--out-file", str(out_path))
        self.assertTrue(out_path.exists())
        self.assertEqual(LOAD_CALLS, [])

    def test_sort_temp_file_never_overwrites_input(self):
        # 旧实现的临时文件是确定的 `<输出>.sorting`；若它正好是输入文件，收尾排序会覆盖原始语料
        input_path = self.tmp / "sortcase.jsonl.sorting"
        input_path.write_text("原始语料\n", encoding="utf-8")
        out_path = self.tmp / "sortcase.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            self.runner.main(
                [
                    "--corpus",
                    str(input_path),
                    "--limit",
                    "2",
                    "--no-pairs-csv",
                    "--model",
                    "a",
                    "--out-file",
                    str(out_path),
                ]
            )
        self.assertEqual(input_path.read_text(encoding="utf-8"), "原始语料\n")
        self.assertTrue(out_path.exists())

    def test_no_llm_rejects_multi_model(self):
        with self.assertRaises(SystemExit):
            self.run_main("--models", "a", "b", "--no-llm", "--out-file", str(self.tmp / "no.jsonl"))

    def test_duplicate_model_names_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_main("--models", "a", "a", "--out-file", str(self.tmp / "dup.jsonl"))

    def test_invalid_weight_rejected_before_writing(self):
        MODEL_CONFIGS["a"]["voting_weight"] = 0
        with self.assertRaises(ValueError):
            self.run_main("--models", "a", "b", "--out-file", str(self.tmp / "bad.jsonl"))
        self.assertFalse((self.tmp / "bad.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
