"""重试与请求预算的联动测试：重试必须继续扣预算，不能绕过费用上限。"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import retry

ROOT = Path(__file__).resolve().parents[1]


class TimeoutModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        raise TimeoutError("timed out")


def load_agents():
    structured = ModuleType("langchain.agents.structured_output")
    structured.ToolStrategy = Mock()
    pydantic = ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = Mock()
    pydantic.create_model = Mock()

    factory = ModuleType("agents")
    factory.call_with_retry = retry.call_with_retry
    factory.is_retryable_error = retry.is_retryable_error
    factory.get_agent = Mock()
    factory.get_deep_agent = Mock()
    factory.request_policy = Mock(return_value={"max_retries": 6})

    spec = importlib.util.spec_from_file_location("_test_spatial_agents_budget", ROOT / "spatial_agents.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "langchain.agents.structured_output": structured,
            "pydantic": pydantic,
            "agents": factory,
            "spatial_tools": ModuleType("spatial_tools"),
            "utils": ModuleType("utils"),
            "_test_spatial_agents_budget": module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class BudgetRetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sa = load_agents()

    def make_agent(self, *, max_requests=None, batch_budget=None):
        model = TimeoutModel()

        class ProbeAgent(self.sa.BaseAgent):
            NAME = "probe"
            INPUTS = ()
            OUTPUTS = ()
            NEEDS_LLM = True
            PROMPT_FILES = ()

            def _build(self):
                return model

        agent = ProbeAgent(
            llm=True,
            max_requests=max_requests,
            batch_budget=batch_budget,
            retry_policy={"max_retries": 6, "backoff_seconds": 0, "jitter_seconds": 0},
        )
        return agent, model

    def test_batch_budget_stops_retries(self):
        budget = self.sa.BatchRequestBudget(1)
        agent, model = self.make_agent(batch_budget=budget)
        with self.assertRaises(self.sa.RequestBudgetExceeded):
            agent._invoke("消息")
        stats = agent.stats()
        self.assertEqual(model.calls, 1)  # 第一次超时后，第二次尝试在发请求前被预算拦下
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.exceeded, 1)
        self.assertEqual(stats["budget_exceeded"], 1)
        self.assertEqual(stats["retries"], 1)
        # 首发确实发出去并超时了：这一次逻辑调用要记失败与耗时，不能整笔漏记
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["requests"], 1)
        self.assertIn("Timeout", stats.get("first_error", ""))

    def test_candidate_budget_stops_retries(self):
        agent, model = self.make_agent(max_requests=1)
        with agent.budget_scope():
            with self.assertRaises(self.sa.RequestBudgetExceeded):
                agent._invoke("消息")
        stats = agent.stats()
        self.assertEqual(model.calls, 1)
        self.assertEqual(stats["budget_exceeded"], 1)
        self.assertEqual(stats["retries"], 1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["requests"], 1)

    def test_candidate_exhaustion_does_not_consume_batch_budget(self):
        budget = self.sa.BatchRequestBudget(2)
        agent, model = self.make_agent(max_requests=1, batch_budget=budget)
        with agent.budget_scope():
            with self.assertRaises(self.sa.RequestBudgetExceeded):
                agent._invoke("消息")
        self.assertEqual(model.calls, 1)
        # 候选预算先拒绝重试，整批预算不能再被扣一次
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.exceeded, 0)
        self.assertEqual(agent.stats()["failed"], 1)

    def test_first_attempt_blocked_by_budget_is_not_counted_as_failure(self):
        budget = self.sa.BatchRequestBudget(1)
        budget.reserve("prefill")
        agent, model = self.make_agent(batch_budget=budget)
        with self.assertRaises(self.sa.RequestBudgetExceeded):
            agent._invoke("消息")
        stats = agent.stats()
        self.assertEqual(model.calls, 0)
        self.assertEqual(stats["budget_exceeded"], 1)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["requests"], 0)
        self.assertEqual(stats["retries"], 0)

    def test_batch_refusal_releases_reserved_candidate_slot(self):
        budget = self.sa.BatchRequestBudget(1)
        agent, model = self.make_agent(max_requests=2, batch_budget=budget)
        with agent.budget_scope():
            with self.assertRaises(self.sa.RequestBudgetExceeded):
                agent._invoke("消息")
            scope = self.sa._CURRENT_BUDGET.get()
            # 首发占了 1 个候选名额；重试先占第 2 个、整批拒绝后要退回，最终仍只占 1 个
            self.assertEqual(scope.requests.get(agent), 1)
        stats = agent.stats()
        self.assertEqual(model.calls, 1)
        self.assertEqual(budget.requests, 1)
        self.assertEqual(budget.exceeded, 1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["requests"], 1)

    def test_zero_retries_counts_one_failed_attempt(self):
        agent, model = self.make_agent()
        agent.retry_policy = {"max_retries": 0}
        with self.assertRaises(TimeoutError):
            agent._invoke("消息")
        stats = agent.stats()
        self.assertEqual(model.calls, 1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["requests"], 1)
        self.assertEqual(stats["retries"], 0)
        self.assertEqual(stats["retry_exhausted"], 0)

    def test_retry_exhaustion_is_counted(self):
        agent, model = self.make_agent()
        agent.retry_policy = {"max_retries": 2, "backoff_seconds": 0, "jitter_seconds": 0}
        with self.assertRaises(TimeoutError):
            agent._invoke("消息")
        stats = agent.stats()
        self.assertEqual(model.calls, 3)  # 首发 + 2 次重试
        self.assertEqual(stats["retries"], 2)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["retry_exhausted"], 1)


if __name__ == "__main__":
    unittest.main()
