import copy
import json
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from extract import sentence_screening as ss
from annotate import spatial_agents as sa
from annotate import spatial_tools as st
from annotate import spatial_graphs as sg


def verdict(word="上", index=3, **changes):
    return {
        "word": word, "index": index, "extractable": True,
        "grammatical": True, "spatial_coherent": True,
        "reason": "实际空间关系", "grammar_reason": "通顺", "spatial_reason": "自洽",
        "source": "llm", **changes,
    }


def original_row():
    return {
        "sentence": ["杯子", "放", "桌子", "上", "。"],
        "source_file": "article.jsonl", "line_no": 7, "source_line_no": 7,
        "source_document_no": 1, "filter_methods": ["locality_phrase"],
        "match_tree": [{"index": 3, "word": "上", "children": []}],
        "localization_grounding": {
            "candidates": [{"index": 3, "word": "上", "keep": True,
                            "target": {"index": 0, "word": "杯子"},
                            "reference": {"index": 2, "word": "桌子", "sdp": "Loc"},
                            "verb": {"index": 1, "word": "放"}}],
        },
        "screening": {"status": "true", "extractable": True, "verdicts": [verdict()]},
    }


RULES = {"groups": [{"words": ["上"], "replacements": ["下", "里"]}]}


class ScreeningTests(unittest.TestCase):
    def test_ungrammatical_or_unknown_verdict_does_not_pass(self):
        self.assertTrue(hasattr(ss, "verdict_passes"), "需要三项明确通过的统一准入")
        self.assertTrue(ss.verdict_passes(verdict()))
        for field in ("extractable", "grammatical", "spatial_coherent"):
            for value in (False, None):
                with self.subTest(field=field, value=value):
                    self.assertFalse(ss.verdict_passes(verdict(**{field: value})))
        self.assertFalse(ss.verdict_passes(verdict(source="error")))

    def test_screener_rejects_extractable_but_ungrammatical(self):
        screener = ss.SentenceScreener(
            family="localization_expressions", config_path=str(st.FOCUS_VOCAB_SPECS["方位词"][0]),
            model_key="unused", workers=1,
        )
        response = ss.BatchExtractionVerdict(results=[ss.BatchExtractionItem(
            item_id=0, **{k: v for k, v in verdict(grammatical=False).items()
                          if k not in ("word", "index", "source")}
        )])
        with patch.object(screener, "_invoke", return_value=response):
            result = screener.screen_rows([original_row()])[0]
        self.assertEqual(result["screening"]["status"], "false")
        self.assertFalse(result["screening"]["extractable"])


class PairPreparationTests(unittest.TestCase):
    def make_preparer(self, resolve):
        from replace import sentence_replace as sr
        screener = ss.SentenceScreener(
            family="localization_expressions", config_path=str(st.FOCUS_VOCAB_SPECS["方位词"][0]),
            model_key="unused", workers=1,
        )
        screener._resolve_tasks = resolve
        return sr.PairPreparer(corpus_type="方位词", rules=RULES, screener=screener)

    def test_pair_preparer_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec("replace.sentence_replace"), "替换应有独立入口")

    def test_replacement_checked_before_annotation_and_triple_is_independent(self):
        from replace import corpus_pairs as cp
        source = original_row()
        saved = copy.deepcopy(source)

        def resolve(tasks):
            self.assertEqual([task[2]["word"] for task in tasks], ["下", "里"])
            self.assertEqual(tasks[0][5]["candidates"][0]["word"], "下")
            return [verdict("下", grammatical=True), verdict("里", grammatical=False)]

        pairs = self.make_preparer(resolve).prepare_rows([source])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["pair"]["b"]["sentence"], "杯子放桌子下。")
        self.assertEqual(pairs[0]["focus_word_b"]["grounding"]["word"], "下")
        self.assertTrue(cp.is_prepared_pair(pairs[0]))
        self.assertFalse(cp.is_prepared_pair(pairs[1]))
        self.assertEqual(pairs[1]["pair_screening"]["status"], "false")
        pairs[0]["focus_word_b"]["grounding"]["target"]["word"] = "修改"
        self.assertEqual(source, saved)

    def test_legacy_original_is_checked_before_generating_replacements(self):
        source = original_row()
        source.pop("screening")
        seen = []

        def resolve(tasks):
            seen.append([task[2]["word"] for task in tasks])
            return [verdict(task[2]["word"], task[2]["index"], grammatical=False) for task in tasks]

        pairs = self.make_preparer(resolve).prepare_rows([source])
        self.assertEqual(seen, [["上"]])
        self.assertEqual(pairs, [])

    def test_only_valid_target_position_is_replaced(self):
        source = original_row()
        source["sentence"] += ["架子", "上"]
        source["match_tree"].append({"index": 6, "word": "上", "children": []})
        source["screening"]["verdicts"].append(verdict(index=6, spatial_coherent=False))

        def resolve(tasks):
            return [verdict(task[2]["word"], task[2]["index"]) for task in tasks]

        pairs = self.make_preparer(resolve).prepare_rows([source])
        self.assertEqual([p["candidate"]["focus_index"] for p in pairs], [3, 3])
        self.assertEqual(pairs[0]["pair"]["b"]["slice"][-1], "上")

    def test_unchecked_and_failed_replacements_are_not_eligible(self):
        from replace import corpus_pairs as cp
        for source in ("no_llm", "error", "limit"):
            with self.subTest(source=source):
                def resolve(tasks):
                    return [verdict(task[2]["word"], extractable=None, source=source) for task in tasks]
                pairs = self.make_preparer(resolve).prepare_rows([original_row()])
                self.assertTrue(pairs)
                self.assertFalse(any(cp.is_prepared_pair(p) for p in pairs))

    def test_pair_reader_rejects_old_input_and_counts_only_accepted_pairs(self):
        from replace import corpus_pairs as cp
        pairs = self.make_preparer(lambda tasks: [verdict(t[2]["word"]) for t in tasks]).prepare_rows([original_row()])
        bad = copy.deepcopy(pairs[0])
        bad["pair_screening"]["b"]["grammatical"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            st.write_jsonl([bad, *pairs], path)
            selected = cp.read_prepared_pairs(path, limit=1, offset=1)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["focus_word_b"]["word"], "里")
            st.write_jsonl([original_row()], path)
            with self.assertRaisesRegex(ValueError, "python replace"):
                cp.read_prepared_pairs(path)

    def test_damaged_pair_metadata_cannot_enter_annotation(self):
        from replace import corpus_pairs as cp
        pairs = self.make_preparer(lambda tasks: [verdict(t[2]["word"]) for t in tasks]).prepare_rows([original_row()])
        for mutate in (
            lambda p: p.pop("candidate"),
            lambda p: p["candidate"].update(focus_index=0),
            lambda p: p["sentence"].__setitem__(0, "另一句"),
            lambda p: p.update(corpus_type="unknown"),
        ):
            damaged = copy.deepcopy(pairs[0])
            mutate(damaged)
            self.assertFalse(cp.is_prepared_pair(damaged))

    def test_replacement_cli_preserves_inputs_and_existing_outputs(self):
        from replace import sentence_replace as replace
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source, output = Path(directory) / "original.jsonl", Path(directory) / "pairs.jsonl"
            rules = Path(directory) / "rules.json5"
            rules.write_text(json.dumps(RULES), encoding="utf-8")
            st.write_jsonl([original_row()], source)
            saved = source.read_bytes()
            with self.assertRaises(ValueError):
                replace.main(["--corpus", str(source), "--out-file", str(source), "--no-llm", "--overwrite"])
            output.write_text("old result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                replace.main(["--corpus", str(source), "--out-file", str(output), "--no-llm"])
            self.assertEqual(output.read_text(encoding="utf-8"), "old result")
            replace.main(["--corpus", str(source), "--out-file", str(output), "--no-llm", "--overwrite", "--alter-rules", str(rules)])
            records = st.read_jsonl(output)
            self.assertEqual(len(records), 2)
            self.assertEqual({r["pair_screening"]["status"] for r in records}, {"no_llm"})
            self.assertEqual(source.read_bytes(), saved)

    def test_null_source_line_keeps_candidate_key_across_preparation_and_annotation(self):
        from replace import corpus_pairs as cp
        source = original_row()
        source["source_line_no"] = None
        preparer = self.make_preparer(lambda tasks: [verdict(t[2]["word"]) for t in tasks])
        pairs = preparer.prepare_rows([source])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            st.write_jsonl(pairs, path)
            loaded = cp.read_prepared_pairs(path)
            for before, after in zip(pairs, loaded):
                state = {"record": after, **sg.prepared_pair_node({"record": after})}
                result = sg.serialize_pair(state)
                self.assertEqual(before["candidate_key"], result["candidate_key"])
                self.assertIn("#7#", before["candidate_key"])
        self.assertIsNone(source["source_line_no"])

    def test_cli_fills_null_source_and_old_line_with_physical_input_line(self):
        from replace import sentence_replace as replace
        from replace import corpus_pairs as cp
        source = original_row()
        source.update(source_line_no=None, line_no=None)
        preparer = self.make_preparer(lambda tasks: [verdict(t[2]["word"]) for t in tasks])
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path, output = Path(directory) / "input.jsonl", Path(directory) / "pairs.jsonl"
            path.write_text("\n" + json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
            with patch.object(replace, "PairPreparer", return_value=preparer), patch.object(preparer.screener, "warm"):
                replace.main(["--corpus", str(path), "--out-file", str(output)])
            before = st.read_jsonl(output)[0]
            after = cp.read_prepared_pairs(output)[0]
            self.assertEqual(before["source_line_no"], 2)
            self.assertEqual(before["candidate_key"], st.candidate_key(st.sentence_id(after), 3, "下"))

    def test_rejected_first_position_still_occupies_its_original_id_slot(self):
        source = original_row()
        source["sentence"] += ["架子", "上"]
        source["match_tree"].append({"index": 6, "word": "上", "children": []})
        source["screening"]["verdicts"] = [verdict(spatial_coherent=False), verdict(index=6)]
        pairs = self.make_preparer(lambda tasks: [verdict(t[2]["word"], t[2]["index"]) for t in tasks]).prepare_rows([source])
        self.assertEqual([p["candidate"]["focus_index"] for p in pairs], [6, 6])
        self.assertEqual([p["candidate"]["focus_order"] for p in pairs], [2, 2])
        state = {"record": pairs[0], **sg.prepared_pair_node({"record": pairs[0]})}
        self.assertEqual(sg.serialize_pair(state)["id"], "1-7-2-1")


class AnnotationGraphTests(unittest.TestCase):
    def prepared(self):
        return PairPreparationTests().make_preparer(
            lambda tasks: [verdict(t[2]["word"], t[2]["index"]) for t in tasks]
        ).prepare_rows([original_row()])

    def test_pair_graph_reads_prepared_pair_without_alter_or_validation_agents(self):
        agents = sa.build_agents(
            ["feature", "schema", "judge"], llm=False, corpus_type="方位词",
            overrides={"schema": {"table": {"schemas": {
                "上": {"1": {"description": "上方", "match": []}},
                "下": {"1": {"description": "下方", "match": []}},
            }}}},
        )

        class Writer:
            path = Path("unused.jsonl")
            records = []

            def write(self, record):
                self.records.append(record)

        writer = Writer()
        spec = sg.GRAPHS["pair"]
        pair = self.prepared()[0]
        states = spec.init_states([pair], agents, str(writer.path))
        self.assertEqual(len(states), 1)
        graph = spec.build(agents, writer)
        graph.invoke(states[0])
        result = writer.records[0]
        self.assertEqual(result["pair"]["b"]["sentence"], "杯子放桌子下。")
        self.assertEqual(result["focus_word_b"]["word"], "下")
        self.assertEqual(result["candidate_key"], pair["candidate_key"])
        self.assertEqual(result["pair_screening"], pair["pair_screening"])
        self.assertEqual(result["semantic"]["b"]["source"], "preparation")
        self.assertEqual([h["node"] for h in result["context"]["history"]], ["read_pair", "feature", "schema", "judge"])

    def test_pair_and_synonym_specs_do_not_request_preparation_agents(self):
        for name in ("pair", "synonym"):
            for agent in ("alter", "reality", "semantic"):
                self.assertNotIn(agent, sg.GRAPHS[name].needs)

    def test_invalid_pair_never_reaches_annotation(self):
        pair = self.prepared()[0]
        pair["pair_screening"]["b"]["spatial_coherent"] = False
        self.assertTrue(hasattr(sg, "prepared_pair_node"))
        updates = sg.prepared_pair_node({"record": pair})
        self.assertEqual(updates["stage"], "bad_pair")

    def test_formal_runner_does_not_prepare_syntax(self):
        from annotate import spatial_runner as runner
        agents = sa.build_agents(sg.GRAPHS["pair"].needs, llm=False, corpus_type="方位词")
        with patch.object(st, "prepare_syntax", side_effect=AssertionError("不应预填句法")):
            runner._prefill(self.prepared(), agents, sg.GRAPHS["pair"])

    def test_synonym_graph_reads_same_pair_without_revalidation(self):
        agents = sa.build_agents(["synonym"], llm=False, corpus_type="方位词")
        records = []
        writer = type("Writer", (), {"path": Path("unused"), "write": lambda self, r: records.append(r)})()
        spec = sg.GRAPHS["synonym"]
        spec.build(agents, writer).invoke(spec.init_states([self.prepared()[0]], agents, "unused")[0])
        self.assertEqual(records[0]["pair"]["b"]["sentence"], "杯子放桌子下。")
        self.assertEqual(records[0]["pair_screening"]["status"], "true")


class AllInOneTests(unittest.TestCase):
    def context(self, *, llm=False):
        from annotate import all_in_one as aio
        args = aio.parse_args(["--no-llm", "--all-candidates"])
        ctx = aio.build_context(args, "方位词", Path("unused.jsonl"))
        return aio, args, ctx._replace(llm=llm)

    def test_structured_output_only_contains_annotation(self):
        from annotate import all_in_one as aio
        model = aio.build_verdict_model([])
        self.assertEqual(set(model.model_fields), {"features", "schema", "judgement"})

    def test_prepare_preserves_pair_checks_and_only_asks_annotation_once(self):
        aio, args, ctx = self.context(llm=True)
        pair = AnnotationGraphTests().prepared()[0]
        states = aio.expand_states(pair, ctx, args)
        self.assertEqual(len(states), 1)
        dumps = {
            "features": {"a": {"vertical_direction": ["[+上方]"]}, "b": {"vertical_direction": ["[+下方]"]}},
            "schema": {"a": {"schema_name": "上11", "reason": "上方"}, "b": {"schema_name": "下1", "reason": "下方"}},
            "judgement": {"same_meaning": False, "reason": "竖直方向不同"},
        }
        with patch.object(st, "syntax_text", side_effect=AssertionError("标注不得查询句法")):
            result = aio.process_state(states[0], ctx=ctx, args=args, call_agent=None,
                                       counter=aio.CallCounter(), precomputed=(dumps, None, 1))
        self.assertEqual(result["pair_screening"], pair["pair_screening"])
        self.assertEqual(result["semantic"]["b"]["source"], "preparation")
        self.assertFalse(result["judgement"]["same_meaning"])
        self.assertEqual(result["all_in_one"]["llm_calls"], 1)
        self.assertEqual(set(result["all_in_one"]["reached"]), {"feature", "schema", "judgement"})

    def test_failed_all_in_one_still_keeps_preparation_and_reports_feature_error(self):
        aio, args, ctx = self.context(llm=True)
        pair = AnnotationGraphTests().prepared()[0]
        state = aio.expand_states(pair, ctx, args)[0]
        result = aio.process_state(state, ctx=ctx, args=args, call_agent=None,
                                   counter=aio.CallCounter(), precomputed=(None, "网络错误", 1))
        self.assertEqual(result["status"], "feature_error_both")
        self.assertEqual(result["pair_screening"]["status"], "true")
        self.assertEqual(result["all_in_one"]["error"], "网络错误")

    def test_table_judgement_wins_over_all_in_one_model_answer(self):
        aio, args, ctx = self.context(llm=True)
        schema = ctx.agents.need("schema")
        schema.table = {"schemas": {"上": {"1": {"match": []}}, "下": {"1": {"match": []}}}}
        ctx.agents.need("judge").table = {"cells": {("上1", "下1"): "×"}, "labels": ["上1", "下1"], "filled": 1}
        state = aio.expand_states(AnnotationGraphTests().prepared()[0], ctx, args)[0]
        dumps = {"features": {"a": {}, "b": {}}, "schema": {}, "judgement": {"same_meaning": True, "reason": "模型认为同义"}}
        result = aio.process_state(state, ctx=ctx, args=args, call_agent=None, counter=aio.CallCounter(),
                                   precomputed=(dumps, None, 1))
        self.assertFalse(result["judgement"]["same_meaning"])
        self.assertEqual(result["judgement"]["source"], "table")

    def test_annotation_clis_read_pairs_and_never_call_ltp(self):
        from annotate import all_in_one as aio
        from annotate import spatial_runner as runner
        import agent_log
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            corpus = root / "pairs.jsonl"
            st.write_jsonl(AnnotationGraphTests().prepared(), corpus)
            with patch.object(st, "prepare_syntax", side_effect=AssertionError("不应调用 LTP")), \
                 patch.object(agent_log, "DEFAULT_LOG_DIR", root / "logs"):
                for entry, filename in ((aio.main, "aio.jsonl"), (runner.main, "pair.jsonl")):
                    output = root / filename
                    entry(["--corpus", str(corpus), "--limit", "2", "--out-file", str(output), "--no-llm", "--batch", "off"])
                    results = st.read_jsonl(output)
                    self.assertEqual(len(results), 2)
                    self.assertEqual([r["pair"]["b"]["sentence"] for r in results], ["杯子放桌子下。", "杯子放桌子里。"])


if __name__ == "__main__":
    unittest.main()
