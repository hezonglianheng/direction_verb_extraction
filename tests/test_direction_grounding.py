# encoding: utf8
"""离线验证：趋向动词三元组抽取与“只筛目标物”的 HowNet 义原闸门（不联网、不建真实模型）"""

import unittest

import config
import direction_grounding as dg
import spatial_agents as sa
import spatial_tools as st


def _context(words, pos, dep_heads, dep_labels, sdp_heads, sdp_labels):
    return dg.lg.build_context(
        words,
        pos,
        {"head": dep_heads, "label": dep_labels},
        {"head": sdp_heads, "label": sdp_labels},
    )


class FakeGate:
    """只提供目标物判定；刻意不实现 check_verb，用来证明生产代码不再查询动词义原"""

    def __init__(self, *, bad_targets=()):
        self.bad_targets = set(bad_targets)

    def check_target(self, word):
        if word in self.bad_targets:
            return {"ok": False, "reason": "target_not_physical", "sememe": None, "path": []}
        return {"ok": True, "reason": "ok", "sememe": "physical|物质", "path": ["physical|物质"]}


class BrokenGate:
    """模拟 HowNet 数据故障：目标物查询抛 HowNetUnavailable"""

    def check_target(self, word):
        raise dg.lg.HowNetUnavailable("boom")


class DirectionTripleTests(unittest.TestCase):
    def test_intransitive_manner(self):
        context = _context(
            ["他", "跑", "进", "教室", "。"],
            ["r", "v", "v", "n", "wp"],
            [2, 0, 2, 3, 2],
            ["SBV", "HED", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2],
            ["AGT", "Root", "mDEPD", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 2, "word": "进"},
            context,
            {2},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "manner_verb_one_complement"],
        )
        self.assertTrue(verdict["keep"])
        self.assertEqual(verdict["target"]["word"], "他")
        self.assertEqual(verdict["reference"]["word"], "教室")
        self.assertEqual(verdict["verb"]["word"], "跑")
        self.assertEqual(verdict["reason"], "ok")

    def test_cause_prefers_patient(self):
        context = _context(
            ["他", "把", "书", "拿", "进", "教室", "。"],
            ["r", "p", "n", "v", "v", "n", "wp"],
            [4, 4, 2, 0, 4, 5, 4],
            ["SBV", "ADV", "POB", "HED", "CMP", "POB", "WP"],
            [4, 3, 4, 0, 4, 4, 4],
            ["AGT", "mRELA", "PAT", "Root", "mDEPD", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 4, "word": "进"},
            context,
            {4},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "cause_verb_one_complement"],
        )
        self.assertTrue(verdict["keep"])
        self.assertEqual(verdict["target"]["word"], "书")
        self.assertEqual(verdict["target"]["role"], "PAT")
        self.assertEqual(verdict["verb"]["word"], "拿")
        self.assertEqual(verdict["reference"]["word"], "教室")

    def test_accompany_serial_walks_back_to_patient(self):
        context = _context(
            ["他", "抱", "着", "孩子", "走", "进", "屋", "。"],
            ["r", "v", "u", "n", "v", "v", "n", "wp"],
            [2, 0, 2, 2, 2, 5, 6, 2],
            ["SBV", "HED", "RAD", "VOB", "COO", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2, 5, 5, 5],
            ["AGT", "Root", "mDEPD", "PAT", "eSUCC", "mDEPD", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 5, "word": "进"},
            context,
            {5},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "accompany_verb_one_complement", "manner_verb_one_complement"],
        )
        self.assertTrue(verdict["keep"])
        self.assertEqual(verdict["target"]["word"], "孩子")
        self.assertEqual(verdict["verb"]["word"], "走")
        self.assertEqual(verdict["reference"]["word"], "屋")

    def test_manner_only_serial_prefers_agent(self):
        # 只命中 manner 规则时，目标物按 AGT 优先取「他」；不再借事件链上「抱」的义原改选孩子。
        context = _context(
            ["他", "抱", "着", "孩子", "走", "进", "屋", "。"],
            ["r", "v", "u", "n", "v", "v", "n", "wp"],
            [2, 0, 2, 2, 2, 5, 6, 2],
            ["SBV", "HED", "RAD", "VOB", "COO", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2, 5, 5, 5],
            ["AGT", "Root", "mDEPD", "PAT", "eSUCC", "mDEPD", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 5, "word": "进"},
            context,
            {5},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "manner_verb_one_complement"],
        )
        self.assertTrue(verdict["keep"])
        self.assertEqual(verdict["target"]["word"], "他")
        self.assertEqual(verdict["verb"]["word"], "走")

    def test_direction_as_center_allows_null_verb(self):
        context = _context(
            ["他", "进去", "了", "。"],
            ["r", "v", "u", "wp"],
            [2, 0, 2, 2],
            ["SBV", "HED", "RAD", "WP"],
            [2, 0, 2, 2],
            ["AGT", "Root", "mDEPD", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 1, "word": "进去"},
            context,
            {1},
            gate=FakeGate(),
            direction_vocab={"进去"},
            filter_methods=["root", "direction_verb_as_center"],
        )
        self.assertTrue(verdict["keep"])
        self.assertIsNone(verdict["verb"])
        self.assertEqual(verdict["target"]["word"], "他")
        self.assertIsNone(verdict["reference"])
        self.assertEqual(verdict["reason"], "ok_no_verb")
        self.assertTrue(verdict["verb_optional"])

    def test_sequential_event_is_not_an_attached_verb(self):
        # 「他吃完饭进屋」：进 eSUCC 依附吃，但这是事件顺承，不是趋向补语；
        # 进 应独立作谓语（verb=None），目标物仍可从事件链上的吃找到。
        context = _context(
            ["他", "吃", "完", "饭", "进", "屋", "。"],
            ["r", "v", "v", "n", "v", "n", "wp"],
            [2, 0, 2, 2, 2, 5, 5],
            ["SBV", "HED", "CMP", "VOB", "COO", "POB", "WP"],
            [2, 0, 2, 2, 2, 5, 5],
            ["AGT", "Root", "mDEPD", "PAT", "eSUCC", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 4, "word": "进"},
            context,
            {4},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "direction_verb_as_center"],
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertIsNone(verdict["verb"], verdict)
        self.assertEqual(verdict["target"]["word"], "他")
        self.assertEqual(verdict["reference"]["word"], "屋")

    def test_psych_verb_clause_still_attaches_but_is_not_filtered(self):
        # 「他喜欢进教室」的「进」是 dCONT 内容从句，仍应抽取 verb=喜欢；
        # 但本版不再用动词义原筛选，所以只要目标物「他」是 physical 就保留。
        context = _context(
            ["他", "喜欢", "进", "教室", "。"],
            ["r", "v", "v", "n", "wp"],
            [2, 0, 2, 3, 2],
            ["SBV", "HED", "VOB", "VOB", "WP"],
            [2, 0, 2, 3, 2],
            ["AGT", "Root", "dCONT", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 2, "word": "进"},
            context,
            {2},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "direction_verb_as_center"],
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["target"]["word"], "他")
        self.assertEqual(verdict["verb"]["word"], "喜欢")
        self.assertEqual(verdict["reason"], "ok")

    def test_compound_direction_shares_event_center(self):
        # 他／进／屋／来：来 依附 进；没有非趋向动词时，来 共享 进 的目标物和参照物。
        context = _context(
            ["他", "进", "屋", "来", "。"],
            ["r", "v", "n", "v", "wp"],
            [2, 0, 2, 2, 2],
            ["SBV", "HED", "POB", "CMP", "WP"],
            [2, 0, 2, 2, 2],
            ["AGT", "Root", "LOC", "mDEPD", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 3, "word": "来"},
            context,
            {3},
            gate=FakeGate(),
            direction_vocab={"进", "来"},
            filter_methods=["root", "direction_verb_as_second_complement"],
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertIsNone(verdict["verb"], verdict)
        self.assertEqual(verdict["target"]["word"], "他")
        self.assertEqual(verdict["reference"]["word"], "屋")

    def test_dep_fallback_respects_role_order(self):
        # SDP 没给 AGT/PAT，只剩 DEP 的 SBV/VOB；他移类必须先取 VOB「书」而不是 SBV「他」。
        context = _context(
            ["他", "拿", "书", "进", "教室", "。"],
            ["r", "v", "n", "v", "n", "wp"],
            [2, 0, 2, 2, 2, 2],
            ["SBV", "HED", "VOB", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2, 2],
            ["mDEPD", "Root", "mDEPD", "mDEPD", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 3, "word": "进"},
            context,
            {3},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "cause_verb_one_complement"],
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["target"]["word"], "书")
        self.assertEqual(verdict["target"]["role"], "PAT")
        self.assertEqual(verdict["verb"]["word"], "拿")
        self.assertEqual(verdict["reference"]["word"], "教室")

    def test_non_physical_target_is_rejected(self):
        context = _context(
            ["事情", "跑", "进", "教室", "。"],
            ["n", "v", "v", "n", "wp"],
            [2, 0, 2, 3, 2],
            ["SBV", "HED", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2],
            ["AGT", "Root", "mDEPD", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 2, "word": "进"},
            context,
            {2},
            gate=FakeGate(bad_targets={"事情"}),
            direction_vocab={"进"},
            filter_methods=["root", "manner_verb_one_complement"],
        )
        self.assertFalse(verdict["keep"])
        self.assertEqual(verdict["reason"], "target_not_physical")

    def test_abstract_reference_is_not_filtered(self):
        # 参照物不参与语义筛选：即使「事情」不是 physical，只要目标物合格就保留。
        context = _context(
            ["他", "跑", "进", "事情", "。"],
            ["r", "v", "v", "n", "wp"],
            [2, 0, 2, 3, 2],
            ["SBV", "HED", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2],
            ["AGT", "Root", "mDEPD", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 2, "word": "进"},
            context,
            {2},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "manner_verb_one_complement"],
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["reference"]["word"], "事情")

    def test_non_motion_verb_is_not_filtered(self):
        # 动词不是运动类也照常抽取；只要目标物是 physical，本版就保留。
        context = _context(
            ["他", "喜欢", "进", "教室", "。"],
            ["r", "v", "v", "n", "wp"],
            [2, 0, 2, 3, 2],
            ["SBV", "HED", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2],
            ["AGT", "Root", "mDEPD", "LOC", "mPUNC"],
        )
        verdict = dg.check_target(
            {"index": 2, "word": "进"},
            context,
            {2},
            gate=FakeGate(),
            direction_vocab={"进"},
            filter_methods=["root", "manner_verb_one_complement"],
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["reason"], "ok")
        self.assertEqual(verdict["verb"]["word"], "喜欢")

    def test_apply_grounding_writes_block(self):
        record = {
            "sentence": ["他", "跑", "进", "教室", "。"],
            "filter_methods": ["root", "manner_verb_one_complement"],
            "match_tree": [{"index": 2, "word": "进", "children": []}],
        }
        context = _context(
            record["sentence"],
            ["r", "v", "v", "n", "wp"],
            [2, 0, 2, 3, 2],
            ["SBV", "HED", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2],
            ["AGT", "Root", "mDEPD", "LOC", "mPUNC"],
        )
        kept, stats = dg.apply_grounding(
            record,
            {"进"},
            pos=context["pos"],
            dep={"head": [2, 0, 2, 3, 2], "label": ["SBV", "HED", "CMP", "POB", "WP"]},
            sdp={"head": [2, 0, 2, 2, 2], "label": ["AGT", "Root", "mDEPD", "LOC", "mPUNC"]},
            gate=FakeGate(),
        )
        self.assertIsNotNone(kept)
        block = kept["direction_grounding"]
        self.assertEqual(block["source"], "hownet")
        self.assertTrue(block["valid"])
        self.assertEqual(block["candidates"][0]["target"]["word"], "他")
        self.assertNotIn("verb_hownet", block["candidates"][0])
        self.assertEqual(block["policy"]["version"], 2)
        self.assertNotIn("verb_roots", block["policy"])
        self.assertFalse(block["policy"]["verb_filtering"])
        self.assertEqual(stats["kept_candidates"], 1)
        self.assertNotIn("verb_not_motion", stats["reasons"])


class DirectionBatchTests(unittest.TestCase):
    def test_gate_rows_fail_open_on_analysis_error(self):
        rows, stats = dg.gate_rows(
            [{"sentence": [], "_direction_analysis": {"error": "boom"}}],
            {"进"},
            gate=None,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["direction_grounding"]["source"], "error")
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["kept_records"], 1)

    def test_apply_grounding_without_direction_target_keeps_record(self):
        record = {"sentence": ["他", "跑", "。"], "match_tree": []}
        kept, stats = dg.apply_grounding(record, {"进"}, gate=None)
        self.assertIsNotNone(kept)
        self.assertEqual(kept["direction_grounding"]["source"], "no_target")
        self.assertEqual(stats["no_target"], 1)

    def test_hownet_unavailable_is_not_swallowed(self):
        context = _context(
            ["他", "跑", "进", "教室", "。"],
            ["r", "v", "v", "n", "wp"],
            [2, 0, 2, 3, 2],
            ["SBV", "HED", "CMP", "POB", "WP"],
            [2, 0, 2, 2, 2],
            ["AGT", "Root", "mDEPD", "LOC", "mPUNC"],
        )
        with self.assertRaises(dg.lg.HowNetUnavailable):
            dg.check_target(
                {"index": 2, "word": "进"},
                context,
                {2},
                gate=BrokenGate(),
                direction_vocab={"进"},
                filter_methods=["root", "manner_verb_one_complement"],
            )

    def test_apply_grounding_propagates_hownet_unavailable(self):
        record = {
            "sentence": ["他", "跑", "进", "教室", "。"],
            "filter_methods": ["root", "manner_verb_one_complement"],
            "match_tree": [{"index": 2, "word": "进", "children": []}],
        }
        with self.assertRaises(dg.lg.HowNetUnavailable):
            dg.apply_grounding(
                record,
                {"进"},
                pos=["r", "v", "v", "n", "wp"],
                dep={"head": [2, 0, 2, 3, 2], "label": ["SBV", "HED", "CMP", "POB", "WP"]},
                sdp={"head": [2, 0, 2, 2, 2], "label": ["AGT", "Root", "mDEPD", "LOC", "mPUNC"]},
                gate=BrokenGate(),
            )

    def test_gate_rows_propagates_hownet_unavailable(self):
        rows = [
            {
                "sentence": ["他", "跑", "进", "教室", "。"],
                "filter_methods": ["root", "manner_verb_one_complement"],
                "match_tree": [{"index": 2, "word": "进", "children": []}],
                "_direction_analysis": {
                    "pos": ["r", "v", "v", "n", "wp"],
                    "dep": {"head": [2, 0, 2, 3, 2], "label": ["SBV", "HED", "CMP", "POB", "WP"]},
                    "sdp": {"head": [2, 0, 2, 2, 2], "label": ["AGT", "Root", "mDEPD", "LOC", "mPUNC"]},
                },
            }
        ]
        with self.assertRaises(dg.lg.HowNetUnavailable):
            dg.gate_rows(rows, {"进"}, gate=BrokenGate())


class DownstreamGroundingTests(unittest.TestCase):
    def _record(self):
        return {
            "sentence": ["他", "进去", "了", "。"],
            "filter_methods": ["direction_verb_as_center"],
            "match_tree": [{"index": 1, "word": "进去", "children": []}],
            "direction_grounding": {
                "candidates": [
                    {
                        "index": 1,
                        "word": "进去",
                        "keep": True,
                        "target": {"index": 0, "word": "他", "role": "AGT"},
                        "reference": None,
                        "verb": None,
                        "verb_optional": True,
                    }
                ]
            },
        }

    def test_select_grounding_reads_direction_block(self):
        grounding = st.select_grounding(self._record(), 1, "进去")
        self.assertIsNotNone(grounding)
        self.assertEqual(grounding["target"]["word"], "他")
        self.assertIsNone(grounding["verb"])
        self.assertTrue(grounding["verb_optional"])

    def test_head_block_marks_legal_empty_verb(self):
        focus = st.pick_focus_word(self._record())
        text = sa.head_block(focus, corpus_type="趋向动词")
        self.assertIn("目标物「他」", text)
        self.assertIn("动词「无（趋向动词独立作谓语，允许为空）」", text)


class DirectionHowNetTests(unittest.TestCase):
    """真实 OpenHowNet 词条覆盖；本机没有词典数据时跳过"""

    @classmethod
    def setUpClass(cls):
        try:
            cls.gate = dg.DirectionGate()
        except dg.lg.HowNetUnavailable as exc:
            raise unittest.SkipTest(str(exc))

    def test_policy_snapshot_only_filters_target(self):
        policy = config.direction_sememe_policy()
        self.assertEqual(policy["version"], 2)
        self.assertIn("physical|物质", policy["target_roots"])
        self.assertNotIn("verb_roots", policy)
        self.assertFalse(policy["verb_filtering"])
        self.assertFalse(policy["reference_filtering"])

    def test_gate_has_no_verb_check(self):
        # 生产闸门不再提供动词义原查询；防止后续误把动词筛选加回来。
        self.assertFalse(hasattr(self.gate, "check_verb"))

    def test_target_physical_and_abstract(self):
        self.assertTrue(self.gate.check_target("他")["ok"])
        self.assertTrue(self.gate.check_target("孩子")["ok"])
        self.assertFalse(self.gate.check_target("事情")["ok"])
        self.assertFalse(self.gate.check_target("价值")["ok"])


class LogRenderTests(unittest.TestCase):
    def test_direction_target_gate_is_rendered(self):
        import extraction_log

        stats = extraction_log.new_stats()
        stats["direction_grounding"] = {
            "records": 1,
            "kept_records": 1,
            "dropped_records": 0,
            "candidates": 1,
            "kept_candidates": 1,
            "dropped_candidates": 0,
            "no_target": 0,
            "errors": 0,
            "reasons": {"ok": 1},
        }
        stats["direction_sememe_policy"] = config.direction_sememe_policy()
        text = extraction_log.render_log(
            stats,
            src_path="probe",
            filter_configs=[],
            output_paths=[],
        )
        self.assertIn("趋向动词目标物语义闸门", text)
        self.assertIn("义原口径", text)
        self.assertIn("'version': 2", text)
        self.assertNotIn("verb_roots", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
