"""记录 id 第二段的编号口径：句内每个目标词位置都占号，空位置也不例外。

不加载 LTP / LLM：只把 spatial_agents 导入所需的外部包换成桩，然后用
`AlterAgent.expand_positions` + 打桩的 `run` 直接检查候选上的 focus_order。
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_alter_agent():
    """导入真实的 spatial_agents，只替掉 langchain/pydantic/agents 这几个外部依赖。"""
    structured = ModuleType("langchain.agents.structured_output")
    structured.ToolStrategy = Mock()
    pydantic = ModuleType("pydantic")
    pydantic.BaseModel = type("BaseModel", (), {})
    pydantic.Field = pydantic.create_model = Mock()
    factory = ModuleType("agents")
    factory.get_agent = Mock()

    name = "_test_spatial_agents_id_order"
    spec = importlib.util.spec_from_file_location(name, ROOT / "spatial_agents.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            name: module,
            "langchain.agents.structured_output": structured,
            "pydantic": pydantic,
            "agents": factory,
            "spatial_tools": ModuleType("spatial_tools"),
        },
    ):
        spec.loader.exec_module(module)
    return module.AlterAgent


def run_result(replacements, word="上"):
    """AlterAgent.run 的返回形状：detail 里有词、span 与候选表。"""
    candidates = [{"replacement": replacement, "source": "table", "group": []} for replacement in replacements]
    return {
        "detail": {"word": word, "span": [0], "candidates": candidates},
        "slices": [["原", replacement] for replacement in replacements],
        "sentences": [f"原{replacement}" for replacement in replacements],
    }


FOCUS_WORDS = [
    {"index": 2, "word": "上", "kind": "方位词", "source": "vocab"},
    {"index": 5, "word": "里", "kind": "方位词", "source": "vocab"},
    {"index": 7, "word": "下", "kind": "方位词", "source": "vocab"},
]


class PositionOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.alter_class = load_alter_agent()

    def expand(self, focus_words, results):
        alter = self.alter_class.__new__(self.alter_class)
        with patch.object(self.alter_class, "run", side_effect=results):
            return alter.expand_positions(["他", "上", "山"], focus_words)

    def test_empty_position_still_consumes_a_number(self):
        # 位置 1 有候选、位置 2 无候选、位置 3 有候选：第三段的编号应是 1、3
        candidates, details = self.expand(
            FOCUS_WORDS,
            [
                run_result(["中", "里"]),
                run_result([], word="里"),
                run_result(["下"]),
            ],
        )
        self.assertEqual(len(details), 3)  # 空位置仍留在 details 里
        self.assertEqual([item["focus_order"] for item in candidates], [1, 1, 3])
        self.assertEqual([item["candidate_order"] for item in candidates], [1, 2, 1])
        self.assertEqual([item["candidate_index"] for item in candidates], [0, 1, 2])
        self.assertEqual([item["candidate_total"] for item in candidates], [3, 3, 3])

    def test_leading_empty_positions_shift_the_first_emitted_number(self):
        # 前两个位置都展不开候选：第一条真正落盘的记录从第 3 个位置开始
        candidates, details = self.expand(
            FOCUS_WORDS,
            [run_result([]), run_result([], word="里"), run_result(["下"])],
        )
        self.assertEqual(len(details), 3)
        self.assertEqual([item["focus_order"] for item in candidates], [3])
        self.assertEqual([item["candidate_order"] for item in candidates], [1])


if __name__ == "__main__":
    unittest.main()
