"""在现有函数级桩测试之外，补一层「小样本 + 真实模块」的链路检查（E3）。

现有测试覆盖到的是**单个函数**在被隔离时的行为，挡不住几个模块接起来之后的口径漂移：
ID 换了来源、候选唯一键重复、抽样口径与展开口径分叉、输出文件在真正写之前就被截断。
本文件用固定小样本跑真实的 spatial_tools / spatial_graphs / file_paths，
逐条盯住跨模块的契约：

    D1  抽样单元：带 sampled_words 的记录只展开被抽中的词；候选 (句子, 位置, 替换词) 唯一
    D2  稳定 ID：源行号不被输入行号覆盖；sentence_id 与文件顺序无关
    D3  schema：正式产物带 schema_version，且能通过统一的字段检查
    A1  数据准入：流程跑完（status=ok）不等于数据准入（is_admitted），判不出来的进复核
    E1  输出政策：撞输入拒绝、已存在默认拒绝、写入失败保留旧内容

依赖说明：本环境没装 langchain，且 ltp_extension 是 PyO3 原生模块、**一个进程只能初始化一次**
（别的用例已经用桩顶过 ltp，再真导入就会 ImportError）。所以这里只把三个外部引擎换成最小桩：
`langchain.tools.tool`（恒等装饰器）、`ltp`、`torch`。被替掉的是**模型推理与装饰器**，
被测的 spatial_tools / spatial_graphs / file_paths 全是真实模块，因此这些用例在无网络、
无模型的环境里也能跑，属于"完整链路的小型检查"而非桩测试。
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def _stub_engines():
    """把三个外部引擎换成最小桩：只为 import 通过，被测逻辑不在这里

        langchain.tools.tool  spatial_tools 的 @tool 装饰器 -> 恒等装饰器
        ltp / torch           sentenceTable 在 import 期就要它们；跑批用不到推理，
                              真导入反而会因 PyO3 模块只能初始化一次而炸
    """
    langchain = ModuleType("langchain")
    tools = ModuleType("langchain.tools")
    tools.tool = lambda func: func
    langchain.tools = tools
    ltp = ModuleType("ltp")
    ltp.LTP = object
    torch = ModuleType("torch")
    torch.no_grad = lambda: None
    return {
        "langchain": langchain,
        "langchain.tools": tools,
        "ltp": ltp,
        "torch": torch,
    }


def load_module(filename, replacements, name=None):
    name = name or f"_test_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**replacements, name: module}):
        spec.loader.exec_module(module)
    return module


def load_real_stack():
    """加载真实的 spatial_tools 与 spatial_graphs（只替掉外部引擎与 langgraph）

    spatial_tools 用 importlib 单独加载（而不是 import）：它连带 import 的 word_sampler /
    sentenceFilters / sentenceTable 在别的用例里被桩顶过，走正常 import 会拿到别人那份。
    加载完把它挂到 sys.modules["spatial_tools"] 上，spatial_graphs 里的
    `import spatial_tools as st` 才取到这一个真实模块。
    """
    engines = _stub_engines()
    st = load_module("spatial_tools.py", engines)
    sys.modules["spatial_tools"] = st
    graph_api = ModuleType("langgraph.graph")
    graph_api.START, graph_api.END = "START", "END"
    graph_api.StateGraph = Mock()
    graphs = load_module("spatial_graphs.py", {**engines, "langgraph.graph": graph_api})
    return st, graphs


VOCAB = {"方位词": {"里", "上边"}, "趋向动词": {"上", "下"}}
"""固定词表：pick_focus_words 的 vocab 参数直接收，测试因此不必读 config_files"""

# 「他爬上山，又走下来。」——上@2 与 下@7 是两个不同的目标词位置
SENTENCE = ["他", "爬", "上", "山", "，", "又", "走", "下", "来", "。"]


def sample_record(**extra):
    """一条固定样本：目标词 上@2 与 下@7（match_tree 取筛选产物简化后的形状）"""
    record = {
        "source_file": str(ROOT / "corpus" / "origin" / "001.jsonl"),
        "sentence": list(SENTENCE),
        "match_tree": [
            {"index": 2, "word": "上", "rule": "direction_verb_as_center", "vocab_matched": True, "children": []},
            {"index": 7, "word": "下", "rule": "direction_verb_as_center", "vocab_matched": True, "children": []},
        ],
        "filter_methods": ["direction_verb_as_center"],
        "source_line_no": 42,
        # read_corpus 写的：本次输入里第几篇来源文档（id 首段）
        "source_document_no": 1,
    }
    record.update(extra)
    return record


class SamplingUnitTests(unittest.TestCase):
    """D1：采样口径与展开口径必须在同一条路上"""

    @classmethod
    def setUpClass(cls):
        cls.st, _ = load_real_stack()

    def words(self, record):
        return [entry["word"] for entry in self.st.pick_focus_words(record, vocab=dict(VOCAB))]

    def test_only_sampled_words_are_expanded(self):
        # 一句里 上、下 都在，但只抽中了「上」：不能再顺手把「下」也展开一遍
        self.assertEqual(self.words(sample_record(sampled_words=["上"])), ["上"])

    def test_sampled_words_all_expanded(self):
        # 抽中两个词就展开两个词的位置——配额与展开的词层分布因此可对账
        self.assertEqual(self.words(sample_record(sampled_words=["上", "下"])), ["上", "下"])

    def test_legacy_sampled_word_single_field(self):
        # 旧抽样产物只带 sampled_word（一份记录一个词），同一条口径
        self.assertEqual(self.words(sample_record(sampled_word="下")), ["下"])

    def test_unsampled_record_still_expands_all_positions(self):
        # 原始语料没有抽样字段：行为不变，句内目标词位置全部展开
        self.assertEqual(self.words(sample_record()), ["上", "下"])

    def test_sampled_but_absent_word_does_not_fall_back_to_everything(self):
        # 抽中的词在本配置的词表里没有：不静默回退成"全展开"，那等于替没抽过的词造样本
        self.assertEqual(self.words(sample_record(sampled_words=["里"])), [])


class StableIdentityTests(unittest.TestCase):
    """D2：源行号与输入行号各记各的；sentence_id 与文件顺序无关"""

    @classmethod
    def setUpClass(cls):
        cls.st, _ = load_real_stack()

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="contract-")))

    def write(self, records, name="input.jsonl"):
        path = self.tmp / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        return path

    def test_source_documents_are_numbered_by_first_appearance(self):
        # 两篇文档交替出现：编号按第一次遇到的顺序给，同一篇后面复用同一个号
        path = self.write(
            [
                sample_record(source_file="b.jsonl", source_line_no=1),
                sample_record(source_file="a.jsonl", source_line_no=2),
                sample_record(source_file="b.jsonl", source_line_no=3),
            ]
        )
        records = self.st.read_corpus(path)
        self.assertEqual([record["source_document_no"] for record in records], [1, 2, 1])
        self.assertEqual([record["source_line_no"] for record in records], [1, 2, 3])

    def test_read_corpus_recomputes_stale_document_numbers(self):
        # 上游残留的编号不沿用：否则两份语料拼接后同一个号会指两篇文档
        path = self.write([sample_record(source_file="a.jsonl", source_document_no=9)])
        self.assertEqual(self.st.read_corpus(path)[0]["source_document_no"], 1)

    def test_source_line_no_survives_reordering(self):
        # 抽样产物写的 source_line_no 是源语料行号，读输入文件时不能被当前物理行号顶掉
        path = self.write([{"sentence": SENTENCE}, sample_record(source_line_no=42)])
        first, second = self.st.read_corpus(path)
        self.assertEqual(second["source_line_no"], 42)   # 源行号留着
        self.assertEqual(second["input_line_no"], 2)     # 当前物理行号另记一份
        self.assertEqual(second["line_no"], 2)           # 历史字段仍指当前行号（排序键/回查旧产物靠它）
        self.assertNotEqual(second["source_line_no"], second["input_line_no"])

    def test_sentence_id_independent_of_input_order(self):
        record = sample_record(source_line_no=42)
        head = self.write([record], "a.jsonl")
        tail = self.write([{"sentence": ["无关的一句"]}, record], "b.jsonl")
        same = self.st.read_corpus(head)[0]
        moved = self.st.read_corpus(tail)[1]
        self.assertNotEqual(same["line_no"], moved["line_no"])          # 行号确实变了
        self.assertEqual(self.st.sentence_id(same), self.st.sentence_id(moved))  # 稳定 ID 不变

    def test_sentence_id_distinguishes_same_text_from_other_document(self):
        # 同一句文本来自两篇不同文章：内容哈希相同，稳定 ID 不同（查重与溯源是两件事）
        a = sample_record()
        b = sample_record(source_file=str(ROOT / "corpus" / "origin" / "002.jsonl"))
        self.assertEqual(self.st.content_hash(a["sentence"]), self.st.content_hash(b["sentence"]))
        self.assertNotEqual(self.st.sentence_id(a), self.st.sentence_id(b))

    def test_candidate_key_is_unique_and_stable(self):
        # 候选唯一键 (sentence_id, 位置, 替换词)：换输入顺序不变，换位置或替换词就变
        record = sample_record(source_line_no=42)
        moved = sample_record(source_line_no=42)
        moved["line_no"] = 99
        sid = self.st.sentence_id(record)
        self.assertEqual(self.st.sentence_id(moved), sid)
        self.assertEqual(self.st.candidate_key(sid, 2, "到"), self.st.candidate_key(sid, 2, "到"))
        self.assertNotEqual(self.st.candidate_key(sid, 2, "到"), self.st.candidate_key(sid, 2, "进"))
        self.assertNotEqual(self.st.candidate_key(sid, 2, "到"), self.st.candidate_key(sid, 7, "到"))
        self.assertIsNone(self.st.candidate_key(sid, 2, ""))   # 缺替换词时不拼半个键

    def test_judge_only_filters_status_before_offset_and_limit(self):
        def with_status(status):
            return sample_record() | {"screening": {"status": status}}

        path = self.write(
            [
                with_status("false"),
                with_status("true"),
                with_status("error"),
                sample_record(),  # 旧语料无 screening：按 no_llm 放行
                with_status("limit"),
                sample_record() | {"screening": None},  # 形状损坏：必须排除
                sample_record() | {"screening": {"status": "bogus", "extractable": True}},  # 显式未知：排除
            ]
        )
        records = self.st.read_corpus(path, judge_only=True)
        self.assertEqual([record["input_line_no"] for record in records], [2, 4])

        # offset/limit 按“过滤后的可用记录”计数，而不是物理行号
        page = self.st.read_corpus(path, judge_only=True, offset=1, limit=5)
        self.assertEqual([record["input_line_no"] for record in page], [4])
        first = self.st.read_corpus(path, judge_only=True, limit=1)
        self.assertEqual([record["input_line_no"] for record in first], [2])

    def test_judge_only_accepts_legacy_screening_without_status(self):
        path = self.write(
            [
                sample_record()
                | {"screening": {"extractable": True, "source": "llm"}},
                sample_record()
                | {"screening": {"extractable": False, "source": "llm"}},
                sample_record()
                | {"screening": {"extractable": None, "source": "limit"}},
                sample_record()
                | {"screening": {"extractable": None, "source": "no_llm"}},
            ]
        )
        records = self.st.read_corpus(path, judge_only=True)
        self.assertEqual([record["input_line_no"] for record in records], [1, 4])

    def test_non_positive_limit_reads_nothing(self):
        path = self.write([sample_record()])
        self.assertEqual(self.st.read_corpus(path, limit=0), [])
        self.assertEqual(self.st.read_corpus(path, limit=-1), [])
        self.assertEqual(self.st.read_corpus(path, limit=0, judge_only=True), [])


class RecordIdTests(unittest.TestCase):
    """记录 id：来源文档编号-源行号-第几个目标词位置-第几个替换句（各段从 1 起）"""

    @classmethod
    def setUpClass(cls):
        cls.st, cls.graphs = load_real_stack()

    def state(self, candidate=None, **record_overrides):
        return {
            "record": sample_record(**record_overrides),
            "candidate": {"focus_order": 3, "candidate_order": 2} if candidate is None else candidate,
        }

    def test_prefers_source_line_no(self):
        state = self.state(source_line_no=42, input_line_no=7, line_no=9)
        self.assertEqual(self.st.record_line_no(state["record"]), 42)
        self.assertEqual(self.graphs._record_id(state), "1-42-3-2")

    def test_reads_document_number_from_source_block(self):
        # 落盘后的结果记录把编号收在 source 块里（serialize_pair），迁移脚本也要能认
        record = {"source": {"source_document_no": 3}, "source_line_no": 42}
        self.assertEqual(self.st.record_document_no(record), 3)

    def test_document_number_types(self):
        self.assertEqual(self.st.record_document_no({"source_document_no": "2"}), 2)
        self.assertEqual(self.st.record_document_no({"source_document_no": 2.0}), 2)
        self.assertIsNone(self.st.record_document_no({"source_document_no": True}))
        self.assertIsNone(self.st.record_document_no({"source_document_no": 0}))
        self.assertIsNone(self.st.record_document_no({}))

    def test_missing_document_number_returns_none(self):
        record = sample_record(source_line_no=42)
        record.pop("source_document_no")
        self.assertIsNone(self.st.record_document_no(record))
        self.assertIsNone(
            self.graphs._record_id(
                {"record": record, "candidate": {"focus_order": 1, "candidate_order": 1}}
            )
        )

    def test_falls_back_to_input_line_no_then_line_no(self):
        record = sample_record(source_line_no=None, input_line_no=7, line_no=9)
        self.assertEqual(self.st.record_line_no(record), 7)
        record = sample_record(source_line_no=None, input_line_no=None, line_no=9)
        self.assertEqual(self.st.record_line_no(record), 9)

    def test_reads_line_numbers_from_source_block(self):
        # 落盘后的结果记录把行号收在 source 块里（serialize_pair），迁移脚本也要能认
        record = {"source": {"source_line_no": None, "input_line_no": 5, "line_no": 9}, "line_no": 9}
        self.assertEqual(self.st.record_line_no(record), 5)

    def test_line_number_types(self):
        self.assertEqual(self.st.record_line_no({"source_line_no": "42"}), 42)
        self.assertEqual(self.st.record_line_no({"source_line_no": 42.0}), 42)
        self.assertIsNone(self.st.record_line_no({"source_line_no": True}))
        self.assertIsNone(self.st.record_line_no({"source_line_no": "abc"}))
        self.assertIsNone(self.st.record_line_no({}))

    def test_no_candidate_uses_zero_zero(self):
        state = self.state(source_line_no=42)
        state["candidate"] = None
        self.assertEqual(self.graphs._record_id(state), "1-42-0-0")

    def test_missing_or_invalid_orders_return_none(self):
        record = sample_record(source_line_no=42)
        self.assertIsNone(self.graphs._record_id({"record": record, "candidate": {"focus_order": 1}}))
        self.assertIsNone(
            self.graphs._record_id(
                {"record": record, "candidate": {"focus_order": 0, "candidate_order": 1}}
            )
        )

    def test_no_usable_line_returns_none(self):
        record = sample_record(source_line_no=None)
        self.assertIsNone(self.st.record_line_no(record))
        self.assertIsNone(self.graphs._record_id({"record": record, "candidate": {"focus_order": 1, "candidate_order": 1}}))


class RecordContractTests(unittest.TestCase):
    """D3 / A1：正式产物的 schema 与数据准入"""

    @classmethod
    def setUpClass(cls):
        cls.st, cls.graphs = load_real_stack()

    def pair_state(self, **overrides):
        candidate = {
            "focus_word": {"index": 2, "word": "上", "kind": "趋向动词", "source": "sampled_word"},
            "focus_index": 2,
            "focus_order": 1,
            "word": "上",
            "replacement": "到",
            "candidate_order": 1,
            "candidate_index": 0,
            "candidate_total": 1,
            "source": "table",
            "slice": list(SENTENCE),
            "sentence": "他爬到山，又走下来。",
        }
        state = {
            "record": sample_record(sampled_words=["上"]),
            "stage": "semantic_ok",
            "status_reason": "",
            "candidate": candidate,
            "focus_word": candidate["focus_word"],
            "focus_word_b": {"index": 7, "word": "下", "kind": "趋向动词", "source": "vocab"},
            "sentence_a": "他爬上山，又走下来。",
            "sentence_b": candidate["sentence"],
            "reality": {"is_spatial_scene": True, "source": "llm", "reason": "space"},
            "fluency": {"is_fluent": True, "source": "llm", "reason": "fluent"},
            "features_detail": {"a": {"status": "ok"}, "b": {"status": "ok"}},
            "schema": {"a": {"name": "上1", "source": "table"}, "b": {"name": "下1", "source": "table"}},
            "judge": {"same_meaning": True, "source": "llm", "reason": "same"},
            "output_file": "mock.jsonl",
        }
        state.update(overrides)
        return state

    def test_serialized_record_passes_schema_check(self):
        record = self.graphs.serialize_pair(self.pair_state())
        self.assertEqual(record["schema_version"], self.st.RESULT_SCHEMA_VERSION)
        self.assertEqual(self.st.validate_result_record(record), [])
        self.assertTrue(record["sentence_id"])
        self.assertEqual(record["candidate_key"], self.st.candidate_key(record["sentence_id"], 2, "到"))
        self.assertEqual(record["source"]["source_line_no"], 42)  # 源行号留在产物里
        self.assertEqual(record["source"]["source_document_no"], 1)  # 文档编号也留在产物里
        # 文档号-源行号-位置-替换句，不再嵌 sentence_id
        self.assertEqual(record["id"], "1-42-1-1")

    def test_flow_success_is_not_data_admission(self):
        # 闸门判不出来、后续都成功：status 仍是 ok，但空间性根本没确认——两者必须分开
        state = self.pair_state(reality={"is_spatial_scene": None, "source": "no_llm", "reason": "未调用"})
        record = self.graphs.serialize_pair(state)
        self.assertTrue(record["status"].startswith("ok"))
        self.assertFalse(record["admission"]["is_admitted"])
        self.assertTrue(record["admission"]["review_required"])
        self.assertEqual(record["admission"]["reality"], "unknown")
        self.assertEqual(record["admission"]["reasons"], ["reality"])

    def test_admitted_when_every_dimension_has_verdict(self):
        record = self.graphs.serialize_pair(self.pair_state())
        self.assertTrue(record["admission"]["is_admitted"])
        self.assertFalse(record["admission"]["review_required"])

    def test_tally_reports_admission_alongside_status(self):
        counters = {name: __import__("collections").Counter() for name in self.graphs.GRAPHS["pair"].counter_names}
        self.graphs.pair_tally(self.graphs.serialize_pair(self.pair_state()), counters)
        unknown = self.graphs.serialize_pair(
            self.pair_state(reality={"is_spatial_scene": None, "source": "no_llm", "reason": ""})
        )
        self.graphs.pair_tally(unknown, counters)
        self.assertEqual(counters["admission_counts"]["admitted"], 1)
        self.assertEqual(counters["admission_counts"]["review_required"], 1)
        self.assertEqual(counters["reality_admission"]["confirmed"], 1)
        self.assertEqual(counters["reality_admission"]["unknown"], 1)
        self.assertEqual(sum(counters["status_counts"].values()), 2)  # 流程侧两条都算跑完


class OutputPolicyTests(unittest.TestCase):
    """E1：输出路径校验、覆盖政策与写入的事务性"""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="output-policy-")))
        sys.path.insert(0, str(ROOT))
        self.file_paths = importlib.import_module("file_paths")

    def test_output_colliding_with_input_is_rejected(self):
        source = self.tmp / "source.jsonl"
        source.write_text('{"content": "原始语料"}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.file_paths.validate_run_outputs(source, source)

    def test_existing_output_is_refused_without_overwrite(self):
        out = self.tmp / "result.jsonl"
        out.write_text('{"old": true}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.file_paths.validate_run_outputs(self.tmp / "source.jsonl", out)
        # 给了 overwrite 就放行
        self.file_paths.validate_run_outputs(self.tmp / "source.jsonl", out, overwrite=True)

    def test_existing_companion_output_is_refused_without_overwrite(self):
        source = self.tmp / "source.jsonl"
        for index, suffix in enumerate((".csv", ".summary.json")):
            with self.subTest(suffix=suffix):
                out = self.tmp / f"result_{index}.jsonl"
                csv = out.with_suffix(".csv")
                companion = out.with_suffix(suffix)
                companion.write_text("previous run\n", encoding="utf-8")
                with self.assertRaises(ValueError) as error:
                    self.file_paths.validate_run_outputs(source, out, csv)
                self.assertIn(str(companion), str(error.exception))
                self.assertFalse(out.exists())
                self.assertEqual(companion.read_text(encoding="utf-8"), "previous run\n")
                self.file_paths.validate_run_outputs(source, out, csv, overwrite=True)

    def test_open_exclusive_never_truncates_before_refusing(self):
        out = self.tmp / "result.jsonl"
        out.write_text("旧输出\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            with self.file_paths.open_exclusive(out):
                pass
        self.assertEqual(out.read_text(encoding="utf-8"), "旧输出\n")  # 拒绝发生在写之前

    def test_open_exclusive_rejects_file_created_after_precheck(self):
        out = self.tmp / "result.jsonl"
        out.write_text("another process\n", encoding="utf-8")
        # Simulate another process creating the file after a stale exists() check.
        with patch.object(Path, "exists", return_value=False):
            with self.assertRaises(FileExistsError):
                with self.file_paths.open_exclusive(out):
                    pass
        self.assertEqual(out.read_text(encoding="utf-8"), "another process\n")

    def test_atomic_write_keeps_previous_output_on_failure(self):
        out = self.tmp / "result.jsonl"
        out.write_text("上一轮的完整结果\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            with self.file_paths.atomic_write_text(out) as handle:
                handle.write("本轮写到一半")
                raise RuntimeError("筛选中途崩了")
        # 旧结果原样保留，也不会留下半截文件
        self.assertEqual(out.read_text(encoding="utf-8"), "上一轮的完整结果\n")
        self.assertEqual(list(self.tmp.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
