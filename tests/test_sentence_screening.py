"""LLM 筛选：true/false 落盘、明确 false 才丢弃、失败与空提示语一律放行。"""

import unittest
from pathlib import Path
from unittest.mock import patch

from sentence_screening import (
    BatchExtractionItem,
    BatchExtractionVerdict,
    ScreenItemBudget,
    SentenceScreener,
    classify_family,
    is_judgement_eligible,
    screening_status,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config_files" / "localization_expressions.json5"


def sample_record(word: str = "上") -> dict:
    return {
        "sentence": ["在", "桌子", "上", "。"],
        "source_file": "demo.jsonl",
        "filter_methods": ["root", "locality_phrase"],
        "match_tree": [
            {"index": 2, "word": word, "rule": "locality_phrase", "children": [{"index": 1, "word": "桌子", "children": []}]}
        ],
    }


def make_bare_screener(*, vocab=None, llm=True) -> SentenceScreener:
    """只在内存里拼出可用状态，不读提示语、不建模型"""
    screener = SentenceScreener(
        family="localization_expressions",
        config_path=str(CONFIG),
        model_key="unused",
        llm=False,
    )
    screener.llm = llm
    screener.kind = "方位词"
    screener.system_prompt = "测试提示语"
    screener._vocab = set(vocab or {"上"})
    screener.workers = 1
    return screener


def make_screener(verdicts, *, vocab=None, llm=True) -> SentenceScreener:
    """在 bare screener 上装一个按任务顺序返回 verdict 的批量桩"""
    screener = make_bare_screener(vocab=vocab, llm=llm)
    results = iter(verdicts)

    def ask_batch(tasks):
        return [next(results) for _ in tasks]

    screener._ask_batch = ask_batch  # type: ignore[method-assign]
    return screener


class FamilyClassificationTests(unittest.TestCase):
    def test_known_families(self):
        self.assertEqual(classify_family("localization_expressions"), "方位词")
        self.assertEqual(classify_family("direction_expressions"), "趋向动词")
        self.assertEqual(classify_family("my_location_rules"), "方位词")
        self.assertIsNone(classify_family("custom_rules"))


class ScreenItemBudgetTests(unittest.TestCase):
    def test_limit_caps_reservations(self):
        budget = ScreenItemBudget(3)
        self.assertEqual(budget.reserve(2), 2)
        self.assertEqual(budget.reserve(5), 1)
        self.assertEqual(budget.reserve(1), 0)
        self.assertEqual(budget.stats()["items_sent"], 3)

    def test_non_positive_limit_is_unlimited(self):
        for value in (0, -1):
            budget = ScreenItemBudget(value)
            self.assertEqual(budget.reserve(100), 100)
            self.assertIsNone(budget.max_items)
            self.assertEqual(budget.stats()["items_sent"], 100)


class ScreeningStatusTests(unittest.TestCase):
    def test_new_statuses_and_eligibility(self):
        for status, eligible in (
            ("true", True),
            ("no_llm", True),
            ("false", False),
            ("error", False),
            ("limit", False),
        ):
            record = {"screening": {"status": status}}
            self.assertEqual(screening_status(record), status)
            self.assertIs(is_judgement_eligible(record), eligible)

    def test_legacy_screening_and_missing_block(self):
        self.assertEqual(screening_status({}), "no_llm")
        self.assertTrue(is_judgement_eligible({}))
        self.assertEqual(screening_status({"screening": {"extractable": True, "source": "llm"}}), "true")
        self.assertEqual(screening_status({"screening": {"extractable": False, "source": "llm"}}), "false")
        self.assertEqual(screening_status({"screening": {"source": "limit"}}), "limit")
        self.assertEqual(screening_status({"screening": {"source": "error"}}), "error")
        self.assertEqual(screening_status({"screening": {"source": "unfilled_prompt"}}), "no_llm")
        self.assertEqual(screening_status({"screening": None}), "error")
        self.assertEqual(screening_status({"screening": {"status": "bogus"}}), "error")
        self.assertEqual(screening_status({"screening": {"status": "bogus", "extractable": True}}), "error")
        self.assertEqual(screening_status({"screening": {"status": None, "source": "no_llm"}}), "error")
        self.assertEqual(screening_status({"screening": {"status": []}}), "error")
        self.assertEqual(screening_status({"screening": []}), "error")


class ScreenRowsTests(unittest.TestCase):
    def test_true_keeps_record_and_formalizes_boolean(self):
        screener = make_screener([{"extractable": True, "reason": "是处所", "source": "llm"}])
        rows = screener.screen_rows([sample_record()])
        self.assertEqual(len(rows), 1)
        screening = rows[0]["screening"]
        self.assertIs(screening["extractable"], True)
        self.assertEqual(screening["status"], "true")
        self.assertEqual(screening["kind"], "方位词")
        verdict = screening["verdicts"][0]
        self.assertIs(verdict["extractable"], True)
        self.assertEqual(verdict["word"], "上")
        self.assertEqual(screener.stats()["extractable"], 1)

    def test_false_is_kept_and_annotated_without_pruning(self):
        screener = make_screener([{"extractable": False, "reason": "不是处所", "source": "llm"}])
        row = sample_record()
        original_tree = [dict(node) for node in row["match_tree"]]
        rows = screener.screen_rows([row])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["screening"]["status"], "false")
        self.assertIs(rows[0]["screening"]["extractable"], False)
        self.assertEqual(rows[0]["match_tree"], original_tree)
        self.assertEqual(screener.stats()["not_extractable"], 1)
        self.assertEqual(screener.stats()["false_records"], 1)

    def test_mixed_verdicts_keep_true_target_and_prune_false(self):
        record = {
            "sentence": ["上", "面", "和", "下", "面", "都", "有", "东", "西", "。"],
            "source_file": "demo.jsonl",
            "filter_methods": ["root"],
            "match_tree": [
                {"index": 0, "word": "上", "children": [{"index": 1, "word": "面", "children": []}]},
                {"index": 3, "word": "下", "children": [{"index": 4, "word": "面", "children": []}]},
            ],
        }
        screener = make_screener(
            [
                {"extractable": False, "reason": "不是处所", "source": "llm"},
                {"extractable": True, "reason": "是方向", "source": "llm"},
            ],
            vocab={"上", "下"},
        )
        rows = screener.screen_rows([record])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["screening"]["status"], "true")
        self.assertIs(rows[0]["screening"]["extractable"], True)
        self.assertEqual([node["word"] for node in rows[0]["match_tree"]], ["下"])

    def test_error_is_kept_with_none(self):
        screener = make_screener([{"extractable": None, "reason": "超时", "source": "error"}])
        rows = screener.screen_rows([sample_record()])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["screening"]["extractable"])
        self.assertEqual(rows[0]["screening"]["status"], "error")
        self.assertEqual(rows[0]["screening"]["source"], "error")
        self.assertEqual(screener.stats()["error_records"], 1)

    def test_true_plus_error_is_error_and_kept(self):
        record = {
            "sentence": ["上", "面", "和", "下", "面", "。"],
            "source_file": "demo.jsonl",
            "filter_methods": ["root"],
            "match_tree": [
                {"index": 0, "word": "上", "children": []},
                {"index": 3, "word": "下", "children": []},
            ],
        }
        screener = make_screener(
            [
                {"extractable": True, "reason": "是处所", "source": "llm"},
                {"extractable": None, "reason": "超时", "source": "error"},
            ],
            vocab={"上", "下"},
        )
        rows = screener.screen_rows([record])
        self.assertEqual(rows[0]["screening"]["status"], "error")
        self.assertIsNone(rows[0]["screening"]["extractable"])

    def test_no_target_is_kept_without_calling_model(self):
        screener = make_screener([], vocab={"下"})
        rows = screener.screen_rows([sample_record("上")])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["screening"]["source"], "no_target")
        self.assertEqual(rows[0]["screening"]["status"], "error")
        self.assertIsNone(rows[0]["screening"]["extractable"])

    def test_disabled_screener_annotates_no_llm(self):
        screener = SentenceScreener(
            family="localization_expressions",
            config_path=str(CONFIG),
            model_key="unused",
            llm=False,
        )
        row = sample_record()
        result = screener.screen_rows([row])
        self.assertIs(result[0], row)
        self.assertEqual(row["screening"]["status"], "no_llm")
        self.assertEqual(row["screening"]["source"], "no_llm")
        self.assertIsNone(row["screening"]["extractable"])
        self.assertFalse(screener.enabled)

    def test_empty_prompt_disables_without_model_config(self):
        with patch("sentence_screening.utils.read_prompt_path", return_value=""):
            screener = SentenceScreener(
                family="localization_expressions",
                config_path=str(CONFIG),
                model_key="模型配置不存在也不该被读取",
                llm=True,
            )
        self.assertFalse(screener.enabled)
        self.assertIn("提示语尚未填写", screener.disabled_reason)

    def test_tasks_are_grouped_by_batch_size(self):
        words = ["上", "下", "前", "后", "里"]
        call_sizes = []
        verdicts = [{"extractable": True, "reason": "ok", "source": "llm"} for _ in words]
        screener = make_screener(verdicts, vocab=set(words))
        screener.batch_size = 2

        original = screener._ask_batch

        def spy(tasks):
            call_sizes.append(len(tasks))
            return original(tasks)

        screener._ask_batch = spy  # type: ignore[method-assign]
        rows = screener.screen_rows([sample_record(word) for word in words])
        self.assertEqual(len(rows), 5)
        self.assertEqual(call_sizes, [2, 2, 1])
        self.assertEqual(screener.stats()["requests"], 3)
        self.assertEqual(screener.stats()["items_sent"], 5)

    def test_send_limit_truncates_last_batch_and_fails_open(self):
        words = ["上", "下", "前", "后", "里"]
        calls = []
        verdicts = [{"extractable": True, "reason": "ok", "source": "llm"} for _ in range(3)]
        screener = make_screener(verdicts, vocab=set(words))
        screener.batch_size = 2
        screener.budget = ScreenItemBudget(3)

        original = screener._ask_batch

        def spy(tasks):
            calls.append(len(tasks))
            return original(tasks)

        screener._ask_batch = spy  # type: ignore[method-assign]
        rows = screener.screen_rows([sample_record(word) for word in words])
        self.assertEqual(len(rows), 5)
        self.assertEqual(calls, [2, 1])
        stats = screener.stats()
        self.assertEqual(stats["items_sent"], 3)
        self.assertEqual(stats["limit_skipped"], 2)
        self.assertEqual(stats["budget"]["items_sent"], 3)
        limited = [row for row in rows if row["screening"]["source"] == "limit"]
        self.assertEqual(len(limited), 2)
        for row in limited:
            self.assertIsNone(row["screening"]["extractable"])
            self.assertEqual(row["screening"]["status"], "limit")
            self.assertIsNone(row["screening"]["verdicts"][0]["extractable"])
        self.assertEqual(stats["limit_records"], 2)

    def test_batch_output_is_mapped_by_item_id_not_position(self):
        screener = make_bare_screener()
        screener.workers = 1
        tasks = [
            (0, "句子甲", {"index": 0, "word": "上"}, [], ["root"], None),
            (1, "句子乙", {"index": 0, "word": "下"}, [], ["root"], None),
        ]
        screener._invoke = lambda _content: BatchExtractionVerdict(  # type: ignore[method-assign]
            results=[
                BatchExtractionItem(item_id=1, extractable=True, reason="乙可提取"),
                BatchExtractionItem(item_id=0, extractable=False, reason="甲不可提取"),
            ]
        )
        results = screener._ask_batch(tasks)
        self.assertIs(results[0]["extractable"], False)
        self.assertIs(results[1]["extractable"], True)

    def test_missing_batch_item_invalidates_whole_batch_and_fails_open(self):
        screener = make_bare_screener()
        tasks = [
            (0, "句子甲", {"index": 0, "word": "上"}, [], ["root"], None),
            (1, "句子乙", {"index": 0, "word": "下"}, [], ["root"], None),
        ]
        screener._invoke = lambda _content: BatchExtractionVerdict(  # type: ignore[method-assign]
            results=[BatchExtractionItem(item_id=0, extractable=True, reason="甲可提取")]
        )
        results = screener._ask_batch(tasks)
        self.assertEqual([item["source"] for item in results], ["error", "error"])
        self.assertTrue(all(item["extractable"] is None for item in results))
        self.assertEqual(screener.stats()["invalid_items"], 1)

    def test_duplicate_item_id_invalidates_whole_batch(self):
        screener = make_bare_screener()
        tasks = [
            (0, "句子甲", {"index": 0, "word": "上"}, [], ["root"], None),
            (1, "句子乙", {"index": 0, "word": "下"}, [], ["root"], None),
        ]
        screener._invoke = lambda _content: BatchExtractionVerdict(  # type: ignore[method-assign]
            results=[
                BatchExtractionItem(item_id=0, extractable=False, reason="甲不可提取"),
                BatchExtractionItem(item_id=0, extractable=True, reason="重复"),
                BatchExtractionItem(item_id=1, extractable=True, reason="乙可提取"),
            ]
        )
        results = screener._ask_batch(tasks)
        self.assertEqual([item["source"] for item in results], ["error", "error"])
        self.assertTrue(all(item["extractable"] is None for item in results))
        self.assertEqual(screener.stats()["invalid_items"], 1)


if __name__ == "__main__":
    unittest.main()
