import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotate import spatial_agents as sa


GROUPS = [
    {"words": ["上", "上面"], "examples": [{"sentence": "海报贴在墙上。", "instruction": "上例：表面附着。"}]},
    {"words": ["下"], "examples": [{"sentence": "书在桌下。", "instruction": "下例：低于参照物。"}]},
    {"words": ["上"], "examples": [{"sentence": "杯子在桌上。", "instruction": "上例：高度较高。\n结合语境判断。"}]},
]


class ExampleTests(unittest.TestCase):
    def test_example_loader_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec("annotate.prompt_examples"))

    def test_exact_word_matches_all_groups_in_file_order(self):
        from annotate import prompt_examples as pe
        rendered = pe.render_examples(GROUPS, "上")
        self.assertIn("海报贴在墙上。", rendered)
        self.assertIn("表面附着", rendered)
        self.assertIn("结合语境判断。", rendered)
        self.assertNotIn("书在桌下", rendered)
        self.assertLess(rendered.index("海报"), rendered.index("杯子"))
        self.assertEqual(pe.render_examples(GROUPS, "上方"), "")
        self.assertEqual(pe.render_examples(GROUPS, ""), "")

    def test_files_separate_stage_and_part_of_speech_and_reject_malformed_groups(self):
        from annotate import prompt_examples as pe
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ("feature", "schema"):
                for suffix, kind in (("localization", "方位词"), ("direction", "趋向动词")):
                    path = root / f"{stage}_{suffix}.json5"
                    marker = f"{stage}/{kind}"
                    path.write_text(json.dumps([{"words": ["上"], "examples": [
                        {"sentence": marker, "instruction": "说明"}]}], ensure_ascii=False), encoding="utf-8")
                    groups = pe.load_examples(stage, kind, root=root)
                    self.assertIn(marker, pe.render_examples(groups, "上"))
            (root / "feature_localization.json5").write_text('[{words: "上", examples: []}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "words"):
                pe.load_examples("feature", "方位词", root=root)


class AnnotationPromptTests(unittest.TestCase):
    def focus(self, word="上", source="vocab"):
        return {
            "word": word, "index": 3, "kind": "方位词", "source": source,
            "grounding": {"word": word, "index": 3, "keep": True,
                          "target": {"word": "杯子", "role": "SYNTAX_ROLE"},
                          "reference": {"word": "桌子", "sdp": "SYNTAX_SDP"},
                          "verb": {"word": "放"}, "source": "mechanical_replacement" if source == "alter" else "hownet"},
        }

    def test_feature_and_schema_never_read_syntax_and_choose_examples_by_own_word(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ("feature", "schema"):
                (root / f"{stage}_localization.json5").write_text(json.dumps(GROUPS, ensure_ascii=False), encoding="utf-8")
            feature = sa.FeatureAgent(llm=False, corpus_type="方位词", feature_classes=[], examples_root=root)
            schema = sa.SchemaAgent(llm=False, corpus_type="方位词", table={"schemas": {}}, examples_root=root)
            with patch.object(sa.st, "syntax_text", side_effect=AssertionError("标注不应查询句法")):
                for word in ("上", "下"):
                    focus = self.focus(word, "alter" if word == "下" else "vocab")
                    for content in (feature._content("杯子放桌子" + word + "。", focus),
                                    schema._content("杯子放桌子" + word + "。", {}, focus, word, word, "空表", {})):
                        self.assertIn("上例" if word == "上" else "下例", content)
                        self.assertNotIn("下例" if word == "上" else "上例", content)
                        self.assertNotIn("SYNTAX_", content)
                        self.assertIn("杯子", content)

    def test_deep_has_no_syntax_tool_and_legacy_class_examples_do_not_leak(self):
        feature = sa.FeatureAgent(llm=False, corpus_type="方位词", deep=True, feature_classes=[{
            "key": "area", "name": "范围", "description": "范围", "values": ["内"],
            "examples": [{"sentence": "LEGACY_UNFILTERED", "area": "内"}],
        }])
        self.assertEqual(feature._subagent_tools(), [])
        self.assertNotIn("LEGACY_UNFILTERED", feature.system_prompt)
        self.assertIn("按关注词选出的", feature.system_prompt)

    def test_all_in_one_selects_four_independent_example_blocks(self):
        from annotate import all_in_one as aio
        from annotate import spatial_graphs as sg
        from test_corpus_stages import AnnotationGraphTests
        args = aio.parse_args(["--no-llm"])
        ctx = aio.build_context(args, "方位词", Path("unused.jsonl"))
        ctx.agents.need("feature").example_groups = GROUPS
        ctx.agents.need("schema").example_groups = [
            {"words": ["上"], "examples": [{"sentence": "图式上例", "instruction": "说明上"}]},
            {"words": ["下"], "examples": [{"sentence": "图式下例", "instruction": "说明下"}]},
        ]
        record = AnnotationGraphTests().prepared()[0]
        state = {"record": record, **sg.prepared_pair_node({"record": record})}
        with patch.object(sa.st, "syntax_text", side_effect=AssertionError("不得读句法")):
            message = aio.prepare(state, ctx, args).message
        self.assertIn("原句的空间特征示例", message)
        self.assertIn("改写句的空间特征示例", message)
        self.assertIn("图式上例", message)
        self.assertIn("图式下例", message)
        self.assertNotIn("句法证据：", message)

    def test_batch_items_include_selected_examples_without_syntax(self):
        import batch_inference as bi
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ("feature", "schema"):
                (root / f"{stage}_localization.json5").write_text(json.dumps(GROUPS, ensure_ascii=False), encoding="utf-8")
            feature = sa.FeatureAgent(llm=False, corpus_type="方位词", feature_classes=[], examples_root=root)
            schema = sa.SchemaAgent(llm=False, corpus_type="方位词", table={"schemas": {}}, examples_root=root)
            feature.llm = schema.llm = True
            profile = bi.BatchProfile("unused", "unused", "unused", "https://unused.invalid", "unused")
            session = bi.BatchSession(profile, cache_path=root / "cache.jsonl")
            with session.active(), patch.object(sa.st, "syntax_text", side_effect=AssertionError("不得查询句法")):
                for agent, args in ((feature, ("杯子在桌上。", self.focus())),
                                    (schema, ("书在桌下。", {}, self.focus("下", "alter")))):
                    with self.assertRaises(bi.BatchPending):
                        agent.run(*args)
            items = list(session.pending.values())
            self.assertEqual(len(items), 2)
            self.assertIn("上例", items[0].content)
            self.assertNotIn("下例", items[0].content)
            self.assertIn("下例", items[1].content)
            self.assertNotIn("上例", items[1].content)

    def test_judgement_does_not_reintroduce_syntax_into_annotation(self):
        agent = sa.JudgeAgent(llm=False, corpus_type="方位词", table={"cells": {}, "labels": [], "filled": 0})
        agent.llm = True
        messages = []

        def invoke(content):
            messages.append(content)
            return sa.JudgeVerdict(same_meaning=False, reason="方向不同")

        with patch.object(agent, "_invoke", side_effect=invoke), \
             patch.object(sa.st, "syntax_text", return_value="FORBIDDEN_SYNTAX"):
            agent.run("杯子放桌子上。", "杯子放桌子下。", {}, {}, {}, {}, self.focus(), self.focus("下", "alter"))
        self.assertNotIn("FORBIDDEN_SYNTAX", messages[0])

    def test_emitted_instructions_request_spatial_meaning_not_syntactic_roles(self):
        feature = sa.FeatureAgent(llm=False, corpus_type="趋向动词", feature_classes=[])
        schema = sa.SchemaAgent(llm=False, corpus_type="趋向动词", table={"schemas": {}})
        reason = schema.response_model().model_json_schema()["properties"]["reason"]["description"]
        self.assertNotIn("句法", reason)
        self.assertIn("空间", reason)
        self.assertNotIn("句法角色", feature.system_prompt)


if __name__ == "__main__":
    unittest.main()
