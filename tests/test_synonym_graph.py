"""Validate the independent synonym graph without loading models or calling APIs."""

from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename, replacements):
    name = f"_synonym_test_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**replacements, name: module}):
        spec.loader.exec_module(module)
    return module


class GraphHarness:
    """Run registered nodes and routes; external LangGraph is replaced only here."""

    def __init__(self, state_type):
        self.state_type = state_type
        self.nodes, self.edges, self.routes = {}, {}, {}

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, source, target):
        self.edges[source] = target

    def add_conditional_edges(self, source, route, targets):
        self.routes[source] = (route, targets)

    def compile(self):
        return self

    def invoke(self, initial):
        state = deepcopy(initial)
        node = self.edges["START"]
        while node != "END":
            changes = self.nodes[node](state)
            history = state.get("history", []) + changes.pop("history", [])
            state.update(changes)
            state["history"] = history
            if node in self.routes:
                route, targets = self.routes[node]
                node = targets[route(state)]
            else:
                node = self.edges[node]
        return state


def load_graphs():
    graph_api = ModuleType("langgraph.graph")
    graph_api.START, graph_api.END = "START", "END"
    graph_api.StateGraph = GraphHarness
    tools = ModuleType("spatial_tools")
    tools.PAIR_CSV_COLUMNS = ()
    tools.join_slice = "".join
    tools.pick_focus_words = lambda record: [record["focus_word"]]
    tools.pick_focus_word = lambda record: record["focus_word"]
    tools.replacement_focus_word = lambda focus, word, methods: {
        **focus, "word": word, "source": "alter"
    }
    return load_module("spatial_graphs.py", {
        "langgraph.graph": graph_api, "spatial_tools": tools,
    })


def load_agents():
    structured = ModuleType("langchain.agents.structured_output")
    structured.ToolStrategy = Mock()
    pydantic = ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = pydantic.create_model = Mock()
    factory = ModuleType("agents")
    factory.get_agent = Mock()
    return load_module("spatial_agents.py", {
        "langchain.agents.structured_output": structured,
        "pydantic": pydantic,
        "agents": factory,
        "spatial_tools": ModuleType("spatial_tools"),
    })


class SynonymGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graphs = load_graphs()

    def setUp(self):
        self.focus_a = {"index": 1, "word": "进", "kind": "趋向动词", "source": "vocab"}
        candidate = {
            "focus_word": self.focus_a, "focus_index": 1, "word": "进",
            "replacement": "入", "source": "table", "group": "进",
            "candidate_index": 0, "candidate_total": 1,
            "slice": ["他", "入", "屋。"], "sentence": "他入屋。",
        }
        alter = SimpleNamespace(expand_positions=Mock(return_value=(
            [candidate], [{"focus_word": self.focus_a, "detail": {"reason": "rule"}}]
        )))
        reality = SimpleNamespace(
            NAME="reality", INPUTS=("sentence", "focus_word"),
            run=Mock(return_value={"is_spatial_scene": True, "source": "llm", "reason": "space"}),
        )
        semantic = SimpleNamespace(
            NAME="semantic", INPUTS=("sentence", "focus_word"), llm=True,
            run=Mock(return_value={"is_fluent": True, "source": "llm", "reason": "fluent"}),
        )
        synonym = SimpleNamespace(
            NAME="synonym", INPUTS=("sentence_a", "sentence_b", "focus_word_a", "focus_word_b"),
            run=Mock(return_value={"same_meaning": True, "source": "llm", "reason": "same"}),
        )
        self.registry = {"alter": alter, "reality": reality, "semantic": semantic, "synonym": synonym}
        self.agents = SimpleNamespace(need=Mock(side_effect=self.registry.__getitem__))
        self.writer = SimpleNamespace(path=Path("mock.jsonl"), write=Mock())
        self.graph = self.graphs.build_synonym_graph(self.agents, self.writer)
        self.initial = {"record": {
            "line_no": 1, "sentence": ["他", "进", "屋。"], "focus_word": self.focus_a,
        }}

    def run_graph(self):
        state = self.graph.invoke(self.initial)
        self.writer.write.assert_called_once()
        return state, self.writer.write.call_args.args[0]

    def test_independent_agent_receives_both_sentences_and_each_focus_word(self):
        state, record = self.run_graph()
        self.assertEqual(set(self.graph.nodes), {"alter", "reality", "semantic", "synonym", "save"})
        self.assertEqual([call.args[0] for call in self.agents.need.call_args_list],
                         ["alter", "reality", "semantic", "synonym"])
        self.registry["synonym"].run.assert_called_once_with(
            sentence_a="他进屋。", sentence_b="他入屋。",
            focus_word_a=self.focus_a, focus_word_b=state["focus_word_b"],
        )
        self.assertEqual(state["focus_word_b"]["word"], "入")
        self.registry["reality"].run.assert_called_once_with(sentence="他进屋。", focus_word=self.focus_a)
        calls = self.registry["semantic"].run.call_args_list
        self.assertEqual(calls[0].kwargs, {"sentence": "他进屋。", "focus_word": self.focus_a})
        self.assertEqual(calls[1].kwargs, {"sentence": "他入屋。", "focus_word": state["focus_word_b"]})
        self.assertEqual(record["status"], "ok")
        self.assertEqual([item["node"] for item in state["history"]],
                         ["alter", "reality", "semantic", "synonym", "save"])

    def test_alter_missed_saves_without_running_later_agents(self):
        self.registry["alter"].expand_positions.return_value = ([], [])
        _, record = self.run_graph()
        self.assertEqual(record["status"], "alter_missed")
        self.assertEqual(record["judgement"], {})
        for name in ("reality", "semantic", "synonym"):
            self.registry[name].run.assert_not_called()

    def test_nonspatial_saves_without_semantic_or_synonym(self):
        self.registry["reality"].run.return_value["is_spatial_scene"] = False
        _, record = self.run_graph()
        self.assertEqual(record["status"], "not_spatial")
        self.registry["semantic"].run.assert_not_called()
        self.registry["synonym"].run.assert_not_called()

    def test_semantic_failures_save_without_synonym(self):
        for verdicts, status in (
            ((False, True), "semantic_failed_a"),
            ((True, False), "semantic_failed_b"),
            ((False, False), "semantic_failed_both"),
            ((True, None), "semantic_error"),
        ):
            with self.subTest(verdicts=verdicts):
                self.writer.write.reset_mock()
                self.registry["semantic"].run.side_effect = [
                    {"is_fluent": value, "source": "llm", "reason": "check"} for value in verdicts
                ]
                _, record = self.run_graph()
                self.assertEqual(record["status"], status)
                self.registry["synonym"].run.assert_not_called()

    def test_skipped_checks_continue_but_unknown_synonym_stays_unknown(self):
        self.registry["reality"].run.return_value = {
            "is_spatial_scene": None, "source": "no_llm", "reason": "skipped",
        }
        self.registry["semantic"].llm = False
        self.registry["semantic"].run.return_value = {
            "is_fluent": None, "source": "no_llm", "reason": "skipped",
        }
        self.registry["synonym"].run.return_value = {
            "same_meaning": None, "source": "no_llm", "reason": "not called",
        }
        _, record = self.run_graph()
        self.assertEqual(record["stage"], "semantic_skipped")
        self.assertEqual(record["status"], "judgement_unknown")
        self.assertEqual(record["judgement"]["source"], "no_llm")
        self.registry["synonym"].run.assert_called_once()

    def test_serialized_verdict_preserves_true_false_and_unknown(self):
        for verdict, expected in ((True, "ok"), (False, "ok"), (None, "judgement_unknown")):
            with self.subTest(verdict=verdict):
                self.writer.write.reset_mock()
                result = {"same_meaning": verdict, "source": "llm", "reason": "verdict"}
                self.registry["synonym"].run.return_value = result
                _, record = self.run_graph()
                self.assertEqual(record["status"], expected)
                self.assertEqual(record["judgement"], result)
                self.assertEqual(record["pair"]["a"]["sentence"], "他进屋。")
                self.assertEqual(record["pair"]["b"]["sentence"], "他入屋。")
                self.assertEqual(set(record["semantic"]), {"a", "b"})
                if verdict is None:
                    self.assertIn("verdict", record["status_reason"])

    def test_tally_counts_all_verdicts_and_keeps_existing_checks(self):
        counters = {name: Counter() for name in self.graphs.GRAPHS["synonym"].counter_names}
        for verdict in (True, False, None):
            self.graphs.synonym_tally({
                "status": "ok" if verdict is not None else "judgement_unknown",
                "judgement": {"same_meaning": verdict, "source": "llm"},
                "semantic": {"a": {"is_fluent": True}, "b": {"is_fluent": True}},
            }, counters)
        self.assertEqual(counters["synonym_counts"], {"same": 1, "different": 1, "unknown": 1})
        self.assertEqual(counters["synonym_sources"], {"llm": 3})
        self.assertEqual(counters["semantic_counts"], {"fluent": 6})
        self.assertEqual(counters["status_counts"], {"ok": 2, "judgement_unknown": 1})

    def test_registration_reuses_existing_input_and_planning(self):
        spec = self.graphs.get_graph("synonym")
        self.assertEqual(spec.needs, ("alter", "reality", "semantic", "synonym"))
        self.assertEqual(spec.prefix, "spatial_synonym")
        self.assertIs(spec.init_states, self.graphs.alter_init_states)
        self.assertIs(spec.plan, self.graphs.pair_plan)
        self.assertIs(spec.gate_queries, self.graphs.reality_queries_alter)


class SynonymAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agents = load_agents()

    def test_empty_prompt_is_read_and_warm_or_run_never_calls_model(self):
        with patch.object(self.agents.utils, "read_prompt_path", return_value="") as read_prompt:
            agent = self.agents.SynonymAgent()
        read_prompt.assert_called_once_with(agent.prompt_paths["synonym"])
        self.assertEqual(agent.system_prompt, "")
        self.assertEqual(agent.PROMPT_FILES, ("synonym",))
        with patch.object(agent, "_build") as build, patch.object(agent, "_invoke") as invoke:
            agent.warm()
            result = agent.run("original", "rewritten", {}, {})
        build.assert_not_called()
        invoke.assert_not_called()
        self.assertIsNone(result["same_meaning"])
        self.assertEqual(result["source"], "unfilled_prompt")
        self.assertIn("synonym.txt", result["reason"])

    def test_no_llm_is_distinct_from_unfilled_prompt(self):
        with patch.object(self.agents.utils, "read_prompt_path", return_value=""):
            agent = self.agents.SynonymAgent(llm=False)
        with patch.object(agent, "_build") as build, patch.object(agent, "_invoke") as invoke:
            agent.warm()
            result = agent.run("original", "rewritten", {}, {})
        build.assert_not_called()
        invoke.assert_not_called()
        self.assertIsNone(result["same_meaning"])
        self.assertEqual(result["source"], "no_llm")

    def test_filled_prompt_passes_user_text_and_returns_independent_verdict(self):
        user_prompt = "user supplied prompt"
        with patch.object(self.agents.utils, "read_prompt_path", return_value=user_prompt):
            agent = self.agents.SynonymAgent()
        with patch.object(self.agents.agent_factory, "get_agent") as get_agent:
            agent._build()
        get_agent.assert_called_once_with(
            agent.model_key, [], user_prompt,
            response_format=self.agents.ToolStrategy.return_value,
        )
        focus_a, focus_b = {"word": "进"}, {"word": "入"}
        with (
            patch.object(self.agents, "pair_head_block", return_value="focus context") as head,
            patch.object(agent, "_invoke", return_value=SimpleNamespace(same_meaning=False, reason="different")) as invoke,
        ):
            result = agent.run("他进屋。", "他入屋。", focus_a, focus_b)
        head.assert_called_once_with(focus_a, focus_b, corpus_type=None)
        self.assertIn("原句：他进屋。", invoke.call_args.args[0])
        self.assertIn("改写句：他入屋。", invoke.call_args.args[0])
        self.assertEqual(result, {"same_meaning": False, "reason": "different", "source": "llm"})

    def test_call_error_returns_unknown_and_is_retried_on_next_run(self):
        with patch.object(self.agents.utils, "read_prompt_path", return_value="user supplied prompt"):
            agent = self.agents.SynonymAgent()
        with (
            patch.object(self.agents, "pair_head_block", return_value="focus context"),
            patch.object(agent, "_invoke", side_effect=RuntimeError("model unavailable")) as invoke,
        ):
            results = [agent.run("original", "rewritten", {}, {}) for _ in range(2)]
        self.assertEqual(invoke.call_count, 2)
        for result in results:
            self.assertIsNone(result["same_meaning"])
            self.assertEqual(result["source"], "error")
            self.assertIn("model unavailable", result["reason"])


if __name__ == "__main__":
    unittest.main()
