"""替换外部模型依赖，验证真实节点和序列化逻辑不会把特征失败记为成功。"""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename, replacements):
    name = f"_test_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**replacements, name: module}):
        spec.loader.exec_module(module)
    return module


def load_graphs():
    graph_api = ModuleType("langgraph.graph")
    graph_api.START, graph_api.END = "START", "END"
    graph_api.StateGraph = Mock()
    tools = ModuleType("spatial_tools")
    tools.PAIR_CSV_COLUMNS = ()
    return load_module("spatial_graphs.py", {"langgraph.graph": graph_api, "spatial_tools": tools})


def result(source="llm", reason="", features=None):
    return {"features": features or {}, "source": source, "reason": reason}


class FeatureStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = load_graphs()

    def pair_state(self, stage="semantic_ok"):
        return {
            "record": {"line_no": 1},
            "stage": stage,
            "status_reason": "已有说明",
            "sentence_a": "原句",
            "sentence_b": "改写句",
            "focus_word": {"word": "上"},
            "focus_word_b": {"word": "下"},
            "judge": {"same_meaning": True},
        }

    def run_pair(self, a, b, stage="semantic_ok"):
        agent = SimpleNamespace(
            NAME="feature",
            INPUTS=("sentence", "focus_word"),
            run=Mock(side_effect=[a, b]),
        )
        state = self.pair_state(stage)
        original = deepcopy(state)
        changes = self.graphs.pair_feature_node(agent)(state)
        self.assertEqual(state, original)
        state.update(changes)
        return state, self.graphs.serialize_pair(state), agent

    def test_b_side_failure_retains_reason_and_cannot_become_ok(self):
        state, record, agent = self.run_pair(result(features={"方向": "上"}), result("error", "TimeoutError"))
        self.assertEqual(record["status"], "feature_error_b")
        self.assertEqual(record["stage"], "semantic_ok")
        self.assertEqual(record["features_source"], "error")
        self.assertEqual(record["features_detail"]["b"], {
            "status": "error", "source": "error", "reason": "TimeoutError"
        })
        self.assertEqual(record["features"]["a"], {"方向": "上"})
        self.assertEqual(record["context"]["history"][0]["status"], "partial_error")
        self.assertIn("已有说明", record["status_reason"])
        self.assertIn("TimeoutError", record["status_reason"])
        self.assertEqual(agent.run.call_args_list[1].kwargs["focus_word"], state["focus_word_b"])

    def test_a_side_failure_is_distinguished(self):
        _, record, _ = self.run_pair(result("error", "原句调用失败"), result())
        self.assertEqual(record["status"], "feature_error_a")
        self.assertIn("原句调用失败", record["status_reason"])
        self.assertEqual(record["features_detail"]["b"]["status"], "ok")

    def test_both_sides_fail_with_both_reasons(self):
        _, record, _ = self.run_pair(result("error", "错误 A"), result("error", "错误 B"))
        self.assertEqual(record["status"], "feature_error_both")
        self.assertEqual(record["context"]["history"][0]["status"], "error")
        self.assertIn("错误 A", record["status_reason"])
        self.assertIn("错误 B", record["status_reason"])

    def test_empty_features_are_valid_success(self):
        _, record, _ = self.run_pair(result(), result())
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["features_source"], "llm")
        self.assertEqual(record["context"]["history"][0]["status"], "ok")

    def test_mixed_sources_are_preserved(self):
        _, record, _ = self.run_pair(result("llm"), result("unfilled_config"))
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["features_source"], "mixed")
        self.assertEqual(record["features_detail"]["b"]["source"], "unfilled_config")

    def test_no_llm_is_skipped_without_becoming_failure(self):
        _, record, _ = self.run_pair(
            result("no_llm", "未调用 LLM"), result("no_llm", "未调用 LLM"), "semantic_skipped"
        )
        self.assertEqual(record["status"], "ok_no_semantic_gate")
        self.assertEqual(record["context"]["history"][0]["status"], "skipped")
        self.assertTrue(all(item["status"] == "skipped" for item in record["features_detail"].values()))

    def test_gate_status_takes_precedence(self):
        for stage in ("not_spatial", "alter_missed", "semantic_failed_a", "semantic_error"):
            with self.subTest(stage=stage):
                state = {**self.pair_state(stage), "features_source": "error"}
                self.assertEqual(self.graphs.pair_final_status(state), stage)

    def test_legacy_state_remains_serializable(self):
        state = self.pair_state()
        self.assertEqual(self.graphs.serialize_pair(state)["status"], "ok")
        self.assertEqual(self.graphs.serialize_pair(state)["features_detail"], {})
        state["features_source"] = "error"
        self.assertEqual(self.graphs.serialize_pair(state)["status"], "feature_error")

    def test_unknown_judgement_remains_unknown_after_successful_features(self):
        state, _, _ = self.run_pair(result(), result())
        state["judge"] = {"same_meaning": None}
        self.assertEqual(self.graphs.pair_final_status(state), "judgement_unknown")

    def test_single_graph_retains_failure_in_record_and_history(self):
        agent = SimpleNamespace(NAME="feature", INPUTS=("sentence", "focus_word"), run=Mock(return_value=result("error", "单句超时")))
        state = {
            "record": {"line_no": 1}, "stage": "semantic_ok", "sentence": "原句",
            "focus_word": {"word": "上"}, "schema": {"name": "上1"}, "status_reason": "",
        }
        state.update(self.graphs.single_feature_node(agent)(state))
        record = self.graphs.serialize_single(state)
        self.assertEqual(record["status"], "feature_error")
        self.assertEqual(record["features_detail"]["reason"], "单句超时")
        self.assertEqual(record["context"]["history"][0]["status"], "error")
        self.assertIn("单句超时", record["status_reason"])

    def test_single_no_llm_and_early_gates_keep_existing_status(self):
        agent = SimpleNamespace(NAME="feature", INPUTS=("sentence", "focus_word"), run=Mock(return_value=result("no_llm")))
        state = {"stage": "semantic_skipped", "sentence": "原句", "focus_word": None, "schema": {"name": "上1"}}
        state.update(self.graphs.single_feature_node(agent)(state))
        self.assertEqual(self.graphs.single_final_status(state), "ok_no_semantic_gate")
        self.assertEqual(state["history"][0]["status"], "skipped")
        state.update(stage="not_spatial", features_source="error")
        self.assertEqual(self.graphs.single_final_status(state), "not_spatial")


class AllInOneFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = load_graphs()
        structured = ModuleType("langchain.agents.structured_output")
        structured.ToolStrategy = Mock()
        callbacks = ModuleType("langchain_core.callbacks")
        callbacks.BaseCallbackHandler = type("BaseCallbackHandler", (), {})
        pydantic = ModuleType("pydantic")
        pydantic.Field = Mock()
        pydantic.create_model = Mock()
        spatial_agents = ModuleType("spatial_agents")
        spatial_agents.AgentSet = object
        cls.aio = load_module("all_in_one.py", {
            "langchain.agents.structured_output": structured,
            "langchain_core.callbacks": callbacks,
            "pydantic": pydantic,
            "agents": ModuleType("agents"),
            "spatial_agents": spatial_agents,
            "spatial_graphs": cls.graphs,
            "spatial_runner": ModuleType("spatial_runner"),
            "spatial_tools": cls.graphs.st,
        })

    def test_process_state_persists_matching_feature_metadata(self):
        for llm in (True, False):
            with self.subTest(llm=llm):
                feature = SimpleNamespace(table_source="llm", run=Mock(return_value=result("no_llm", "未调用 LLM")))
                schema = SimpleNamespace(table={})
                registry = {"feature": feature, "schema": schema}
                agents = SimpleNamespace(need=lambda name: registry.get(name, object()))
                ctx = SimpleNamespace(
                    llm=llm, agents=agents, model_key="test", corpus_type="方位词",
                    feature_classes=[], prompt_path="test", reality_path="test",
                )
                dumps = {"features": {"a": {"方向": "上"}, "b": {"方向": "下"}}}
                state = {
                    "record": {"line_no": 1}, "stage": "altered", "status_reason": "",
                    "sentence_a": "原句", "sentence_b": "改写句", "focus_word": None, "focus_word_b": None,
                }
                prepared = SimpleNamespace(message="测试消息", syntax=None, syntax_errors=None)
                with (
                    patch.object(self.graphs, "alter_node", return_value=lambda state: {}),
                    patch.object(self.aio, "prepare", return_value=prepared),
                    patch.object(self.aio, "call_once", return_value=(dumps, None, 1)),
                    patch.object(self.aio, "resolve_reality", return_value={"is_spatial_scene": True, "source": "llm", "reason": ""}),
                    patch.object(self.aio, "resolve_fluency", return_value={"is_fluent": True, "reason": ""}),
                    patch.object(self.aio, "resolve_schema", return_value={"name": "上1", "source": "table"}),
                    patch.object(self.aio, "resolve_judgement", return_value={"same_meaning": True}),
                    patch.object(self.graphs.st, "schema_candidates", create=True, return_value={}),
                ):
                    record = self.aio.process_state(state, ctx=ctx, args=SimpleNamespace(print_prompt=False), call_agent=None, counter=None)
                self.assertEqual(set(record["features_detail"]), {"a", "b"})
                for detail in record["features_detail"].values():
                    self.assertEqual(detail["status"], "ok" if llm else "skipped")
                    self.assertEqual(detail["source"], "llm" if llm else "no_llm")
                self.assertEqual(record["status"], "ok" if llm else "ok_no_semantic_gate")
                if llm:
                    self.assertEqual(record["features"], dumps["features"])

    def test_call_once_counts_real_requests_via_probe(self):
        class FakeAgent:
            def invoke(self, input, config=None):
                for handler in (config or {}).get("callbacks") or []:
                    handler.on_chat_model_start()
                return {"structured_response": SimpleNamespace(model_dump=lambda: {"x": 1})}

        counter = self.aio.CallCounter()
        dumps, error, requests = self.aio.call_once(FakeAgent(), "消息", counter)
        self.assertIsNone(error)
        self.assertEqual(requests, 1)
        self.assertEqual(counter.rounds, 1)
        self.assertEqual(counter.requests, 1)
        self.assertEqual(counter.retried, 0)

    def test_call_once_failure_still_counts_rounds_and_requests(self):
        class FailingAgent:
            def invoke(self, input, config=None):
                for handler in (config or {}).get("callbacks") or []:
                    handler.on_chat_model_start()
                raise RuntimeError("boom")

        counter = self.aio.CallCounter()
        dumps, error, requests = self.aio.call_once(FailingAgent(), "消息", counter)
        self.assertIsNone(dumps)
        self.assertIn("boom", error)
        self.assertEqual(requests, 1)
        self.assertEqual(counter.rounds, 1)
        self.assertEqual(counter.requests, 1)
        self.assertEqual(counter.retried, 0)

    @staticmethod
    def agent_with_requests(count, *, fail=False):
        def invoke(input, config=None):
            for _ in range(count):
                for handler in (config or {}).get("callbacks") or []:
                    handler.on_chat_model_start()
            if fail:
                raise RuntimeError("request failed")
            return {"structured_response": SimpleNamespace(model_dump=lambda: {"x": 1})}

        return SimpleNamespace(invoke=invoke)

    def test_failure_before_model_callback_does_not_count_retries(self):
        counter = self.aio.CallCounter()
        dumps, error, requests = self.aio.call_once(
            self.agent_with_requests(0, fail=True), "消息", counter
        )
        self.assertIsNone(dumps)
        self.assertIn("request failed", error)
        self.assertEqual(requests, 0)
        self.assertEqual((counter.rounds, counter.requests, counter.retried), (1, 0, 0))

    def test_multiple_model_callbacks_count_retries_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                counter = self.aio.CallCounter()
                dumps, error, requests = self.aio.call_once(
                    self.agent_with_requests(3, fail=fail), "消息", counter
                )
                if fail:
                    self.assertIsNone(dumps)
                    self.assertIn("request failed", error)
                else:
                    self.assertEqual(dumps, {"x": 1})
                    self.assertIsNone(error)
                self.assertEqual(requests, 3)
                self.assertEqual((counter.rounds, counter.requests, counter.retried), (1, 3, 2))

    def test_early_failure_does_not_cancel_retries_from_other_rounds(self):
        counter = self.aio.CallCounter()
        for count, fail in ((0, True), (3, False), (1, False)):
            self.aio.call_once(self.agent_with_requests(count, fail=fail), "消息", counter)
        self.assertEqual((counter.rounds, counter.requests, counter.retried), (3, 4, 2))


if __name__ == "__main__":
    unittest.main()
