"""迁移脚本 migrate_result_ids.py：旧 id -> 源行号-位置-替换句，并产出新 jsonl / csv。"""

import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def _stub_engines():
    langchain = ModuleType("langchain")
    tools = ModuleType("langchain.tools")
    tools.tool = lambda func: func
    langchain.tools = tools
    ltp = ModuleType("ltp")
    ltp.LTP = object
    torch = ModuleType("torch")
    torch.no_grad = lambda: None
    return {"langchain": langchain, "langchain.tools": tools, "ltp": ltp, "torch": torch}


def _load(filename, replacements, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**replacements, name: module}):
        spec.loader.exec_module(module)
    return module


def load_stack():
    """真实的 spatial_tools / spatial_graphs + 待测的迁移脚本。"""
    engines = _stub_engines()
    st = _load("spatial_tools.py", engines, "_migrate_st")
    sys.modules["spatial_tools"] = st
    graph_api = ModuleType("langgraph.graph")
    graph_api.START, graph_api.END = "START", "END"
    graph_api.StateGraph = Mock()
    sg = _load("spatial_graphs.py", {**engines, "langgraph.graph": graph_api}, "_migrate_sg")
    sys.modules["spatial_graphs"] = sg
    migrate = _load("migrate_result_ids.py", {}, "_migrate_ids")
    return st, sg, migrate


TREE = [
    {"index": 2, "word": "上", "children": []},
    {"index": 7, "word": "下", "children": []},
]


def pair_record(
    line_no: int,
    *,
    source_file: str = "/tmp/source.jsonl",
    candidate: dict | None = None,
    old_id="__absent__",
    source_line_no=None,
):
    record = {
        "status": "ok",
        "stage": "semantic_ok",
        "line_no": line_no,
        "sentence_id": f"{source_file}#{line_no}#fingerprint",
        "content_hash": "fingerprint",
        "candidate_key": f"{source_file}#{line_no}#fingerprint|2|到",
        "candidate": candidate,
        "source": {
            "source_file": source_file,
            "source_document_id": source_file,
            "source_line_no": source_line_no,
            "input_line_no": line_no,
            "filter_methods": [],
            "match_tree": TREE,
        },
        "pair": {"a": {"sentence": "他爬上山，又走下来。"}, "b": {"sentence": "他爬到山，又走下来。"}},
        "schema_version": "1.0",
    }
    if old_id != "__absent__":
        record["id"] = old_id
    return record


def candidate(focus_index=2, index=0, order=None):
    value = {
        "index": index,
        "total": 1,
        "focus_index": focus_index,
        "word": "上",
        "replacement": "到",
        "source": "table",
        "group": [],
    }
    if order is not None:
        value["candidate_order"] = order
    return value


class MutateRecordsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st, cls.sg, cls.migrate = load_stack()

    def test_long_id_with_hyphenated_path_is_reencoded(self):
        # 旧 id 首段是 sentence_id，路径里带 '-'；仍只看末尾两段
        record = pair_record(1, candidate=candidate(focus_index=2), old_id="/tmp/data-set/x.jsonl#1#abc-1-7")
        migrated, stats = self.migrate.migrate_records([record])
        self.assertEqual(migrated[0]["id"], "1-1-1-7")
        self.assertEqual(stats["migrated_ok"], 1)

    def test_short_id_recomputes_focus_ordinal_over_all_positions(self):
        # 句内上@2、下@7 两个位置；改的是第二个位置 -> 位置段应是 2，不是旧的 1
        record = pair_record(2, candidate=candidate(focus_index=7), old_id="2-1-3")
        migrated, _ = self.migrate.migrate_records([record])
        self.assertEqual(migrated[0]["id"], "1-2-2-3")

    def test_missing_id_is_rebuilt_and_candidate_order_grouped(self):
        first = pair_record(3, candidate=candidate(focus_index=7, index=3))
        second = pair_record(3, candidate=candidate(focus_index=7, index=4))
        migrated, stats = self.migrate.migrate_records([first, second])
        self.assertEqual([record["id"] for record in migrated], ["1-3-2-1", "1-3-2-2"])
        self.assertEqual(stats["grouped_fallback"], 2)

    def test_no_candidate_uses_zero_zero(self):
        record = pair_record(4, candidate=None)
        migrated, _ = self.migrate.migrate_records([record])
        self.assertEqual(migrated[0]["id"], "1-4-0-0")

    def test_focus_word_index_is_used_when_candidate_lacks_it(self):
        record = pair_record(5, candidate=candidate(focus_index=None))
        record["focus_word"] = {"index": 7, "word": "下", "kind": "方位词", "source": "vocab"}
        migrated, _ = self.migrate.migrate_records([record])
        self.assertEqual(migrated[0]["id"], "1-5-2-1")

    def test_documents_are_numbered_by_first_appearance(self):
        # 行序 1(a) < 3(b) < 9(b)：文档号按行序首次出现给，a=1、b=2
        records = [
            pair_record(3, source_file="/tmp/b.jsonl", candidate=candidate(focus_index=2), old_id="3-1-1"),
            pair_record(1, source_file="/tmp/a.jsonl", candidate=candidate(focus_index=2), old_id="1-1-1"),
            pair_record(9, source_file="/tmp/b.jsonl", candidate=candidate(focus_index=2), old_id="9-1-1"),
        ]
        migrated, _ = self.migrate.migrate_records(records)
        self.assertEqual(
            [record["id"] for record in migrated], ["2-3-1-1", "1-1-1-1", "2-9-1-1"]
        )

    def test_document_order_uses_physical_line_not_source_line(self):
        # 源行号在每篇文档里都从头数：a 的源行号 100 比 b 的 1 大，但物理行序 a 在前
        a = pair_record(
            5,
            source_file="/tmp/a.jsonl",
            source_line_no=100,
            candidate=candidate(focus_index=2),
            old_id="5-1-1",
        )
        b = pair_record(
            6,
            source_file="/tmp/b.jsonl",
            source_line_no=1,
            candidate=candidate(focus_index=2),
            old_id="6-1-1",
        )
        migrated, _ = self.migrate.migrate_records([a, b])
        # 文档号按物理行序：a=1、b=2；id 里的行号段仍按源行号（100 / 1）
        self.assertEqual(
            [record["id"] for record in migrated], ["1-100-1-1", "2-1-1-1"]
        )

    def test_corpus_document_numbers_override_recomputation(self):
        record = pair_record(1, candidate=candidate(focus_index=2), old_id="1-1-1")
        migrated, _ = self.migrate.migrate_records(
            [record], None, {"/tmp/source.jsonl": 4}
        )
        self.assertEqual(migrated[0]["id"], "4-1-1-1")
        self.assertEqual(migrated[0]["source"]["source_document_no"], 4)

    def test_document_number_is_written_beside_document_id(self):
        record = pair_record(2, candidate=candidate(focus_index=2), old_id="2-1-1")
        migrated, _ = self.migrate.migrate_records([record])
        source = migrated[0]["source"]
        self.assertEqual(source["source_document_no"], 1)
        keys = list(source)
        self.assertEqual(keys.index("source_document_no"), keys.index("source_document_id") + 1)

    def test_missing_source_document_leaves_id_empty(self):
        record = pair_record(1, candidate=candidate(focus_index=2), old_id="1-1-1")
        record["source"].pop("source_file")
        record["source"].pop("source_document_id")
        migrated, stats = self.migrate.migrate_records([record])
        self.assertIsNone(migrated[0]["id"])
        self.assertEqual(stats["no_document"], 1)

    def test_line_is_read_from_four_segment_id(self):
        self.assertEqual(self.migrate._line_from_old_id("3-42-1-2"), 42)

    def test_other_identity_fields_and_schema_are_preserved(self):
        record = pair_record(6, candidate=candidate(focus_index=2), old_id="6-1-1")
        migrated, _ = self.migrate.migrate_records([record])
        self.assertEqual(migrated[0]["sentence_id"], record["sentence_id"])
        self.assertEqual(migrated[0]["candidate_key"], record["candidate_key"])
        self.assertEqual(migrated[0]["content_hash"], record["content_hash"])
        self.assertEqual(migrated[0]["schema_version"], self.st.RESULT_SCHEMA_VERSION)
        self.assertEqual(next(iter(migrated[0])), "id")  # id 仍在第一个键


class CommandLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st, cls.sg, cls.migrate = load_stack()

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="migrate-")))
        self.input = self.tmp / "old.jsonl"
        records = [
            pair_record(10, candidate=candidate(focus_index=2, index=1), old_id="10-1-1"),
            pair_record(2, candidate=candidate(focus_index=7, index=0), old_id="2-1-1"),
            pair_record(4, candidate=None),
        ]
        self.input.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
        )

    def run_main(self, *extra):
        out = self.tmp / "new.jsonl"
        csv_path = self.tmp / "new.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            code = self.migrate.main([str(self.input), "-o", str(out), "--csv-out", str(csv_path), *extra])
        return code, out, csv_path

    def test_writes_sorted_jsonl_and_program_csv(self):
        code, out, csv_path = self.run_main()
        self.assertEqual(code, 0)
        records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(
            [record["id"] for record in records], ["1-2-2-1", "1-4-0-0", "1-10-1-1"]
        )
        self.assertEqual(records, sorted(records, key=self.sg.record_order_key))
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], [header for header, _ in self.st.PAIR_CSV_COLUMNS])
        self.assertEqual(
            [row[0] for row in rows[1:]], ["1-2-2-1", "1-4-0-0", "1-10-1-1"]
        )

    def test_corpus_supplies_document_numbers(self):
        corpus = self.tmp / "corpus.jsonl"
        corpus.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in [
                    {"sentence": ["甲"], "source_file": "/tmp/other.jsonl"},
                    {"sentence": ["乙"], "source_file": "/tmp/source.jsonl"},
                ]
            ),
            encoding="utf-8",
        )
        code, out, _ = self.run_main("--corpus", str(corpus))
        self.assertEqual(code, 0)
        records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        # 语料里 /tmp/source.jsonl 是第二篇 -> 文档号 2
        self.assertTrue(all(record["id"].startswith("2-") for record in records))

    def test_refuses_existing_output_without_overwrite(self):
        self.assertEqual(self.run_main()[0], 0)
        before = (self.tmp / "new.jsonl").read_bytes(), (self.tmp / "new.csv").read_bytes()
        self.assertEqual(self.run_main()[0], 2)
        after = (self.tmp / "new.jsonl").read_bytes(), (self.tmp / "new.csv").read_bytes()
        self.assertEqual(before, after)
        self.assertEqual(self.run_main("--overwrite")[0], 0)

    def test_strict_fails_on_duplicate_ids(self):
        duplicate = pair_record(7, candidate=candidate(focus_index=2, index=0), old_id="7-1-1")
        self.input.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in [duplicate, duplicate]),
            encoding="utf-8",
        )
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(self.run_main("--overwrite", "--strict")[0], 1)


if __name__ == "__main__":
    unittest.main()
