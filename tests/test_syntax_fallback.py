"""不加载模型、不写语料，验证 LTP 预填充失败后的真实入口和记录诊断。"""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_all_in_one():
    structured = ModuleType("langchain.agents.structured_output")
    structured.ToolStrategy = Mock()
    pydantic = ModuleType("pydantic")
    pydantic.Field = pydantic.create_model = Mock()
    callbacks = ModuleType("langchain_core.callbacks")
    callbacks.BaseCallbackHandler = object
    agents = ModuleType("spatial_agents")
    agents.AgentSet = object
    agents.DEFAULT_MODEL = "test"
    agents.FOCUS_ROLE_RECORD = "record"
    agents.pair_head_block = Mock(return_value="关注词：上")
    graphs = ModuleType("spatial_graphs")
    tools = ModuleType("spatial_tools")
    runner = ModuleType("spatial_runner")
    runner._corpus_type_arg = lambda value: value
    replacements = {
        "langchain.agents.structured_output": structured,
        "langchain_core.callbacks": callbacks,
        "pydantic": pydantic,
        "agents": ModuleType("agents"),
        "spatial_agents": agents,
        "spatial_graphs": graphs,
        "spatial_runner": runner,
        "spatial_tools": tools,
    }
    name = "_test_syntax_all_in_one"
    spec = importlib.util.spec_from_file_location(name, ROOT / "all_in_one.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**replacements, name: module}):
        spec.loader.exec_module(module)
    return module


class SyntaxFallbackTests(unittest.TestCase):
    def setUp(self):
        self.aio = load_all_in_one()
        self.output = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.output))
        tools, graphs, runner = self.aio.st, self.aio.sg, self.aio.sr
        tools.normalize_corpus_type = lambda value: value or "方位词"
        tools.CORPUS_TYPE_DIRS = {"方位词": "localization"}
        tools.read_corpus = Mock(return_value=[
            {"line_no": 1, "sentence": ["原句"]},
            {"line_no": 2, "sentence": ["原句"]},
        ])
        tools.load_focus_vocab = Mock(return_value={"方位词": {"上"}})
        tools.join_slice = "".join
        tools.schema_candidates = Mock(return_value={})
        tools.schema_group_key = Mock(return_value=None)
        tools.prefill_ltp = Mock(return_value=3)
        tools.cache_ready = Mock(return_value=True)
        tools.parse_sentence_text = Mock(side_effect=lambda sentence: f"句法分析：{sentence}")
        # 成对 csv 路径（E1）：与 jsonl 本身错开，别让收尾导 csv 覆盖刚写好的 jsonl
        tools.pairs_csv_path = Mock(side_effect=lambda path: Path(path).with_suffix(".csv"))
        # 特征值业务校验（A2）：桩里也给同一份口径，形状与本体一致
        self.aio.sa.validate_feature_values = lambda features, classes: {"ok": True}
        runner._warn_corpus_type_mismatch = Mock()
        runner._print_config_tables = Mock()
        runner._config_tables = Mock(return_value={})
        self.writer = SimpleNamespace(
            write=Mock(),
            finish=Mock(return_value={"count": 2, "elapsed_seconds": 0, "summary_file": "mock.summary.json"}),
        )
        runner.ResultWriter = Mock(return_value=self.writer)
        graphs.GRAPHS = {"pair": SimpleNamespace(counter_names=(), csv_columns=())}
        graphs.pair_tally = Mock()
        graphs.alter_node = Mock(return_value=lambda state: {})
        graphs._step = lambda node, **detail: {"node": node, **detail}
        graphs.serialize_pair = lambda state: {
            "line_no": state["record"]["line_no"], "context": {"history": state["history"]}
        }
        self.ctx = self.aio._Context(
            agents=SimpleNamespace(need=lambda name: SimpleNamespace(table={}), items=lambda: []),
            corpus_type="方位词", model_key="test", llm=True, out_path=Path("mock.jsonl"),
            system_prompt="测试提示语", verdict_model=dict, feature_classes=[],
            prompt_path="mock.txt", reality_path="reality.txt",
        )
        self.aio.build_context = Mock(return_value=self.ctx)
        self.aio.expand_states = Mock(side_effect=lambda record, ctx, args: [self.state(record)])
        reached = {name: False for name in self.aio.PAIR_CALLS_PER_STEP}
        self.aio.assemble = Mock(return_value=self.aio._Outcome(
            "not_spatial", "", {}, {}, {}, None, {}, {}, {}, reached,
        ))
        self.call_agent = SimpleNamespace(invoke=Mock(return_value={
            "structured_response": SimpleNamespace(model_dump=lambda: {}),
        }))
        self.aio.build_call_agent = Mock(return_value=self.call_agent)

    def state(self, record):
        sentence_b = f"改写句{record['line_no']}"
        return {
            "record": record, "stage": "altered", "sentence_a": "原句", "sentence_b": sentence_b,
            "focus_word": {"word": "上"}, "focus_word_b": {"word": "下"},
            "candidate_index": 0, "candidates": [{"sentence": sentence_b}],
        }

    def run_main(self, *options, llm=True):
        self.aio.build_context.return_value = self.ctx._replace(llm=llm)
        self.aio.main([
            "--corpus", "mock-corpus.jsonl", "--out-file", "mock.jsonl", "--limit", "2",
            "--workers", "2", *options,
        ])
        return [call.args[0] for call in self.writer.write.call_args_list]

    def syntax_history(self, record):
        return next(step for step in record["context"]["history"] if step["node"] == "llm")

    def test_prefill_failure_continues_calls_and_preserves_diagnostics(self):
        self.aio.st.prefill_ltp.side_effect = RuntimeError("CUDA out of memory")
        records = self.run_main()
        self.assertEqual(len(records), 2)
        self.assertEqual(self.call_agent.invoke.call_count, 2)
        self.aio.st.prefill_ltp.assert_called_once_with(["原句", "改写句1", "改写句2"])
        self.aio.st.parse_sentence_text.assert_not_called()
        self.aio.st.cache_ready.assert_not_called()
        for record in records:
            self.assertEqual(record["all_in_one"]["syntax"], {"a": False, "b": False})
            self.assertIsNone(record["all_in_one"]["error"])
            self.assertIn("RuntimeError('CUDA out of memory')", self.syntax_history(record)["syntax_errors"][0])
        for call in self.call_agent.invoke.call_args_list:
            message = call.args[0]["messages"][0]["content"]
            self.assertIn("原句：原句", message)
            self.assertNotIn("句法证据", message)
        self.assertEqual(self.output.getvalue().count("警告：LTP 预填充失败"), 1)

    def test_partial_prefill_failure_does_not_trigger_worker_inference(self):
        cached = {}

        def partial_prefill(sentences):
            cached[sentences[0]] = "已经缓存的句法证据"
            raise RuntimeError("后续批次失败")

        self.aio.st.prefill_ltp.side_effect = partial_prefill
        self.aio.st.parse_sentence_text.side_effect = lambda sentence: cached[sentence]
        records = self.run_main()
        self.assertEqual(len(records), 2)
        self.aio.st.parse_sentence_text.assert_not_called()
        self.assertEqual(cached, {"原句": "已经缓存的句法证据"})

    def test_successful_prefill_keeps_cached_evidence(self):
        cached = {}

        def prefill(sentences):
            cached.update({sentence: f"缓存证据：{sentence}" for sentence in sentences})
            return len(sentences)

        self.aio.st.prefill_ltp.side_effect = prefill
        self.aio.st.parse_sentence_text.side_effect = lambda sentence: cached[sentence]
        records = self.run_main()
        self.aio.st.prefill_ltp.assert_called_once()
        self.assertEqual(self.aio.st.parse_sentence_text.call_count, 4)
        self.assertEqual(self.aio.st.cache_ready.call_count, 3)
        for record in records:
            self.assertEqual(record["all_in_one"]["syntax"], {"a": True, "b": True})
            self.assertEqual(self.syntax_history(record)["syntax_errors"], [])
        self.assertIn("缓存证据：原句", self.call_agent.invoke.call_args.args[0]["messages"][0]["content"])

    def test_no_syntax_skips_ltp_without_failure_diagnostics(self):
        records = self.run_main("--no-syntax")
        self.aio.st.prefill_ltp.assert_not_called()
        self.aio.st.parse_sentence_text.assert_not_called()
        for record in records:
            self.assertEqual(record["all_in_one"]["syntax"], {"a": False, "b": False})
            self.assertEqual(self.syntax_history(record)["syntax_errors"], [])

    def test_no_llm_skips_ltp(self):
        records = self.run_main("--no-llm", llm=False)
        self.assertEqual(len(records), 2)
        self.aio.st.prefill_ltp.assert_not_called()
        self.aio.st.parse_sentence_text.assert_not_called()
        self.call_agent.invoke.assert_not_called()
        self.assertTrue(all(record["all_in_one"]["syntax"] is None for record in records))

    def test_call_failure_before_model_callback_reports_no_retries(self):
        self.call_agent.invoke.side_effect = RuntimeError("模型启动前失败")
        records = self.run_main()
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["all_in_one"]["requests"] == 0 for record in records))
        self.assertTrue(all("模型启动前失败" in record["all_in_one"]["error"] for record in records))
        self.assertIn("模型轮次 2 次", self.output.getvalue())
        self.assertIn("实际请求 0 次（回调实测，含框架内部补发 0 次；", self.output.getvalue())

    def test_print_prompt_continues_after_prefill_failure_without_calls_or_writes(self):
        self.aio.st.prefill_ltp.side_effect = FileNotFoundError("模型不存在")
        self.run_main("--print-prompt", "--no-llm", llm=False)
        self.assertIn("原句：原句", self.output.getvalue())
        self.assertNotIn("原句句法证据：", self.output.getvalue())
        self.assertIn("FileNotFoundError('模型不存在')", self.output.getvalue())
        self.aio.st.parse_sentence_text.assert_not_called()
        self.call_agent.invoke.assert_not_called()
        self.aio.sr.ResultWriter.assert_not_called()

    def test_failure_does_not_disable_syntax_in_later_run(self):
        self.aio.st.prefill_ltp.side_effect = [RuntimeError("暂时失败"), 3]
        self.run_main()
        self.assertIsNone(self.ctx.syntax_error)
        self.writer.write.reset_mock()
        records = self.run_main()
        self.assertTrue(all(record["all_in_one"]["syntax"] == {"a": True, "b": True} for record in records))

    def test_single_sentence_analysis_failure_keeps_existing_fallback(self):
        self.aio.st.parse_sentence_text.side_effect = ["原句证据", ValueError("单句分析失败")]
        prepared = self.aio.prepare(
            self.state({"line_no": 1}), self.ctx, SimpleNamespace(no_syntax=False),
        )
        self.assertEqual(prepared.syntax, {"a": True, "b": False})
        self.assertEqual(prepared.syntax_errors, ["ValueError('单句分析失败')"])

    def test_user_interrupt_is_not_swallowed(self):
        self.aio.st.prefill_ltp.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_main()
        self.call_agent.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
