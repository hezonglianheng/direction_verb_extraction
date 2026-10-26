"""多模型投票的纯函数测试（不依赖 langchain / LTP / 网络）。"""

import importlib.util
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import voting

ROOT = Path(__file__).resolve().parents[1]


def load_graphs():
    """只加载 spatial_graphs 的统计函数，用于验证 alter_check 平票不计入正式结论"""
    graph_api = ModuleType("langgraph.graph")
    graph_api.START, graph_api.END = "START", "END"
    graph_api.StateGraph = object
    tools = ModuleType("spatial_tools")
    tools.PAIR_CSV_COLUMNS = ()
    spec = importlib.util.spec_from_file_location("_test_voting_graphs", ROOT / "spatial_graphs.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "langgraph.graph": graph_api,
            "spatial_tools": tools,
            "_test_voting_graphs": module,
        },
    ):
        spec.loader.exec_module(module)
    return module


def pair_record(
    label="same",
    *,
    line_no=1,
    candidate_index=0,
    focus_index=1,
    replacement="上",
    status=None,
    admission=None,
):
    """构造一条 pair 图结果记录；label 决定 judgement/reality/semantic 的形状"""
    record = {
        "line_no": line_no,
        "id": f"1-{line_no}-1-{candidate_index + 1}",
        "candidate": {
            "index": candidate_index,
            "focus_index": focus_index,
            "replacement": replacement,
        },
        "reality": {"is_spatial_scene": True, "reason": ""},
        "semantic": {
            "a": {"is_fluent": True, "reason": ""},
            "b": {"is_fluent": True, "reason": ""},
        },
        "judgement": {},
        "status": status or "ok",
        "admission": admission
        or {"is_admitted": True, "review_required": False, "reasons": []},
    }
    if label == "same":
        record["judgement"] = {"same_meaning": True, "source": "llm", "reason": "同义"}
    elif label == "different":
        record["judgement"] = {"same_meaning": False, "source": "llm", "reason": "不同义"}
    elif label == "not_spatial":
        record["reality"] = {"is_spatial_scene": False, "reason": "引申用法"}
        record["semantic"] = {}
        record["status"] = status or "spatial_not_spatial"
    elif label == "not_fluent":
        record["semantic"] = {
            "a": {"is_fluent": True, "reason": ""},
            "b": {"is_fluent": False, "reason": "改写句不通顺"},
        }
        record["status"] = status or "semantic_failed_b"
    elif label == "invalid":
        record["judgement"] = {"same_meaning": None, "source": "error", "reason": "调用失败"}
        record["status"] = status or "judgement_unknown"
    else:
        raise ValueError(label)
    return record


class WeightTests(unittest.TestCase):
    def test_default_is_one(self):
        self.assertEqual(voting.normalize_weight(None), 1)

    def test_integer_weight(self):
        self.assertEqual(voting.normalize_weight(3, model_name="a"), 3)

    def test_rejects_bool_string_zero_and_negative(self):
        for value in (True, False, "2", 0, -1, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    voting.normalize_weight(value, model_name="a")


class BallotTests(unittest.TestCase):
    def test_pair_labels(self):
        cases = {
            "same": "same",
            "different": "different",
            "not_spatial": "not_spatial",
            "not_fluent": "not_fluent_b",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                got, _ = voting.ballot_label("pair", pair_record(label))
                self.assertEqual(got, expected)

    def test_missing_judgement_is_abstention(self):
        label, reason = voting.ballot_label("pair", pair_record("invalid"))
        self.assertIsNone(label)
        self.assertIn("未给出同义判定", reason)

    def test_feature_error_is_abstention_even_if_judgement_present(self):
        record = pair_record("same", status="feature_error_b")
        label, _ = voting.ballot_label("pair", record)
        self.assertIsNone(label)

    def test_single_feature_error_is_abstention_even_if_schema_present(self):
        record = {
            "reality": {"is_spatial_scene": True},
            "semantic": {"is_fluent": True},
            "schema": {"name": "上1", "reason": ""},
            "status": "feature_error",
        }
        label, reason = voting.ballot_label("single", record)
        self.assertIsNone(label)
        self.assertIn("特征提取失败", reason)

    def test_single_graph_uses_schema_name(self):
        record = {
            "reality": {"is_spatial_scene": True},
            "semantic": {"is_fluent": True},
            "schema": {"name": "上1", "reason": ""},
            "status": "ok",
        }
        self.assertEqual(voting.ballot_label("single", record)[0], "schema:上1")

    def test_alter_check_uses_both_fluency_verdicts(self):
        record = {
            "reality": {"is_spatial_scene": True},
            "semantic": {
                "a": {"is_fluent": True},
                "b": {"is_fluent": False},
            },
        }
        self.assertEqual(voting.ballot_label("alter_check", record)[0], "fluent:a=1,b=0")

    def test_unknown_graph_raises(self):
        with self.assertRaises(ValueError):
            voting.ballot_label("nope", {})


class AggregateTests(unittest.TestCase):
    def test_weighted_majority(self):
        result = voting.aggregate_votes(
            {
                "a": {"label": "same", "reason": ""},
                "b": {"label": "different", "reason": ""},
            },
            {"a": 2, "b": 1},
            ["a", "b"],
        )
        self.assertEqual(result["status"], voting.VOTING_ACCEPTED)
        self.assertEqual(result["winner_label"], "same")
        self.assertEqual(result["winner_model"], "a")
        self.assertEqual(result["scores"], {"same": 2, "different": 1})

    def test_tie(self):
        result = voting.aggregate_votes(
            {
                "a": {"label": "same", "reason": ""},
                "b": {"label": "different", "reason": ""},
            },
            {"a": 1, "b": 1},
            ["a", "b"],
        )
        self.assertEqual(result["status"], voting.VOTING_TIE)
        self.assertIsNone(result["winner_label"])
        self.assertEqual(result["tie_labels"], ["different", "same"])

    def test_plurality_with_three_models(self):
        result = voting.aggregate_votes(
            {
                "a": {"label": "same", "reason": ""},
                "b": {"label": "different", "reason": ""},
                "c": {"label": "different", "reason": ""},
            },
            {"a": 1, "b": 1, "c": 1},
            ["a", "b", "c"],
        )
        self.assertEqual(result["status"], voting.VOTING_ACCEPTED)
        self.assertEqual(result["winner_label"], "different")
        self.assertEqual(result["winner_model"], "b")

    def test_no_valid_votes(self):
        result = voting.aggregate_votes(
            {
                "a": {"label": None, "reason": "失败"},
                "b": {"label": None, "reason": "失败"},
            },
            {"a": 1, "b": 1},
            ["a", "b"],
        )
        self.assertEqual(result["status"], voting.VOTING_NO_VALID_VOTES)
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["invalid_models"], ["a", "b"])


class CombineTests(unittest.TestCase):
    def test_accepted_record_keeps_winner_and_marks_voting(self):
        records = {
            "a": [pair_record("same")],
            "b": [pair_record("different")],
        }
        result = voting.combine_model_records(
            "pair", records, {"a": 2, "b": 1}, ["a", "b"], output_file="final.jsonl"
        )
        self.assertEqual(len(result), 1)
        record = result[0]
        self.assertEqual(record["model_name"], "voting")
        self.assertEqual(record["output_file"], "final.jsonl")
        self.assertEqual(record["voting_label"], "same")
        self.assertTrue(record["judgement"]["same_meaning"])
        self.assertEqual(record["voting"]["winner_model"], "a")
        self.assertFalse(record["voting"]["incomplete"])

    def test_tie_is_discarded_not_written_as_false(self):
        records = {
            "a": [pair_record("same")],
            "b": [pair_record("different")],
        }
        result = voting.combine_model_records("pair", records, {"a": 1, "b": 1}, ["a", "b"])
        record = result[0]
        self.assertEqual(record["status"], "discarded")
        self.assertIn("平票", record["status_reason"])
        self.assertIsNone(record["judgement"]["same_meaning"])
        self.assertEqual(record["judgement"]["source"], "voting_tie")
        self.assertFalse(record["admission"]["is_admitted"])
        self.assertTrue(record["admission"]["review_required"])
        self.assertIn("voting_tie", record["admission"]["reasons"])
        self.assertIsNone(record["voting_label"])

    def test_missing_model_is_incomplete_and_not_admitted(self):
        records = {"a": [pair_record("same")]}
        result = voting.combine_model_records("pair", records, {"a": 1, "b": 1}, ["a", "b"])
        record = result[0]
        self.assertEqual(record["voting"]["status"], voting.VOTING_ACCEPTED)
        self.assertTrue(record["voting"]["incomplete"])
        self.assertEqual(record["voting"]["missing_models"], ["b"])
        self.assertFalse(record["admission"]["is_admitted"])
        self.assertIn("voting_incomplete", record["admission"]["reasons"])
        self.assertEqual(record["voting"]["valid_weight"], 1)
        self.assertEqual(record["voting"]["configured_weight"], 2)

    def test_all_invalid_is_discarded(self):
        records = {"a": [pair_record("invalid")], "b": [pair_record("invalid")]}
        result = voting.combine_model_records("pair", records, {"a": 1, "b": 1}, ["a", "b"])
        record = result[0]
        self.assertEqual(record["status"], "discarded")
        self.assertEqual(record["voting"]["status"], voting.VOTING_NO_VALID_VOTES)
        self.assertIn("没有有效票", record["status_reason"])
        self.assertIsNone(record["judgement"]["same_meaning"])

    def test_records_are_aligned_by_line_and_candidate(self):
        a = [pair_record("same", line_no=1), pair_record("same", line_no=2)]
        b = [pair_record("different", line_no=2), pair_record("same", line_no=1)]
        result = voting.combine_model_records("pair", {"a": a, "b": b}, {"a": 1, "b": 1}, ["a", "b"])
        self.assertEqual([r["line_no"] for r in result], [1, 2])
        # 第 2 行 a=same、b=different 且权重相同 -> 平票
        self.assertEqual([r["voting"]["status"] for r in result], ["accepted", "tie"])

    def test_missing_record_for_one_line_marks_that_line_incomplete(self):
        records = {
            "a": [pair_record("same", line_no=1), pair_record("same", line_no=2)],
            "b": [pair_record("same", line_no=1)],
        }
        result = voting.combine_model_records("pair", records, {"a": 1, "b": 1}, ["a", "b"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["voting"]["missing_models"], ["b"])
        self.assertTrue(result[1]["voting"]["incomplete"])

    def test_single_tie_drops_schema_name(self):
        base = {
            "line_no": 1,
            "candidate": None,
            "reality": {"is_spatial_scene": True},
            "semantic": {"is_fluent": True},
            "schema": {"name": "上1", "source": "table", "reason": ""},
            "status": "ok",
            "admission": {"is_admitted": True, "review_required": False, "reasons": []},
        }
        other = json.loads(json.dumps(base))
        other["schema"]["name"] = "上2"
        result = voting.combine_model_records("single", {"a": [base], "b": [other]}, {"a": 1, "b": 1}, ["a", "b"])
        self.assertEqual(result[0]["status"], "discarded")
        self.assertIsNone(result[0]["schema"]["name"])
        self.assertEqual(result[0]["schema"]["source"], "voting_tie")

    def test_duplicate_key_invalidates_only_that_model(self):
        records = {
            "a": [pair_record("same"), pair_record("different")],
            "b": [pair_record("same")],
        }
        result = voting.combine_model_records("pair", records, {"a": 1, "b": 1}, ["a", "b"])
        self.assertEqual(len(result), 1)
        record = result[0]
        self.assertEqual(record["voting"]["duplicate_models"], ["a"])
        self.assertEqual(record["voting"]["invalid_models"], ["a"])
        self.assertTrue(record["voting"]["incomplete"])
        self.assertEqual(record["voting"]["winner_model"], "b")
        self.assertFalse(record["admission"]["is_admitted"])
        self.assertIn("voting_incomplete", record["admission"]["reasons"])

    def test_alter_check_tie_clears_final_semantic_verdict(self):
        def alter_record(a_fluent, b_fluent):
            return {
                "line_no": 1,
                "candidate": {"index": 0, "focus_index": 1, "replacement": "上"},
                "reality": {"is_spatial_scene": True, "reason": ""},
                "semantic": {
                    "a": {"is_fluent": a_fluent, "reason": "a"},
                    "b": {"is_fluent": b_fluent, "reason": "b"},
                },
                "status": "semantic_ok" if a_fluent and b_fluent else "semantic_failed_b",
                "admission": {"is_admitted": True, "review_required": False, "reasons": []},
            }

        result = voting.combine_model_records(
            "alter_check",
            {"a": [alter_record(True, False)], "b": [alter_record(True, True)]},
            {"a": 1, "b": 1},
            ["a", "b"],
        )
        record = result[0]
        self.assertEqual(record["status"], "discarded")
        self.assertEqual(record["voting"]["status"], voting.VOTING_TIE)
        for side in ("a", "b"):
            self.assertIsNone(record["semantic"][side]["is_fluent"])
            self.assertEqual(record["semantic"][side]["source"], "voting_tie")


class AlterCheckTallyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = load_graphs()

    def test_discarded_record_is_not_counted_as_a_fluency_verdict(self):
        counters = {
            name: Counter()
            for name in (
                "status_counts",
                "reality_counts",
                "semantic_counts",
                "focus_sources",
                "focus_kinds",
                "replacement_kinds",
            )
        }
        record = {
            "status": "discarded",
            "voting": {"status": "tie"},
            "semantic": {
                "a": {"is_fluent": True},
                "b": {"is_fluent": False},
            },
            "focus_word": {},
            "focus_word_b": {},
        }
        self.graphs.alter_check_tally(record, counters)
        self.assertEqual(counters["semantic_counts"]["discarded"], 2)
        self.assertEqual(counters["semantic_counts"]["fluent"], 0)
        self.assertEqual(counters["semantic_counts"]["not_fluent"], 0)

    def test_normal_record_still_counts_fluent_and_not_fluent(self):
        counters = {
            name: Counter()
            for name in (
                "status_counts",
                "reality_counts",
                "semantic_counts",
                "focus_sources",
                "focus_kinds",
                "replacement_kinds",
            )
        }
        record = {
            "status": "semantic_failed_b",
            "semantic": {
                "a": {"is_fluent": True},
                "b": {"is_fluent": False},
            },
            "focus_word": {},
            "focus_word_b": {},
        }
        self.graphs.alter_check_tally(record, counters)
        self.assertEqual(counters["semantic_counts"]["fluent"], 1)
        self.assertEqual(counters["semantic_counts"]["not_fluent"], 1)


class SummaryAndPathTests(unittest.TestCase):
    def test_summary_counts(self):
        records = [
            voting.combine_model_records(
                "pair",
                {"a": [pair_record("same", line_no=1)], "b": [pair_record("same", line_no=1)]},
                {"a": 1, "b": 1},
                ["a", "b"],
            )[0],
            voting.combine_model_records(
                "pair",
                {"a": [pair_record("same", line_no=2)], "b": [pair_record("different", line_no=2)]},
                {"a": 1, "b": 1},
                ["a", "b"],
            )[0],
        ]
        summary = voting.summarize_voting(records)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["tie"], 1)
        self.assertEqual(summary["discarded"], 1)
        self.assertEqual(summary["labels"], {"same": 1})

    def test_model_output_paths_are_sanitized_and_ordered(self):
        raw_dir, paths = voting.model_output_paths("out/final.jsonl", ["a/b", "a_b", "模型"])
        self.assertEqual(raw_dir, Path("out/final.models"))
        self.assertEqual(list(paths), ["a/b", "a_b", "模型"])
        self.assertEqual(len(set(paths.values())), 3)
        self.assertNotIn("/", paths["a/b"].name[:-6])
        for path in paths.values():
            self.assertEqual(path.parent, raw_dir)


if __name__ == "__main__":
    unittest.main()
