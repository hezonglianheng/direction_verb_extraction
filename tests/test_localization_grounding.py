"""方位词结构整理 + OpenHowNet 参照物语义闸门。

用例里的 DEP/SDP 数组来自 LTP base2 的实测输出（head 均为 1-based，0 表示根），
覆盖「书在桌子上」「把书放到桌子上」「桌子上的书」「学校东边有河」等主句式，
以及时间/抽象参照物等负例。结构测试不加载 LTP；只有 OpenHowNet 用例读本地词典。
"""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import config
import localization_grounding as lg
import sentenceFilters


def ltp(words, pos, dep_head, dep_label, sdp_head, sdp_label):
    return (
        list(words),
        list(pos),
        {"head": list(dep_head), "label": list(dep_label)},
        {"head": list(sdp_head), "label": list(sdp_label)},
    )


def record(words, targets, methods=("root", "locality_phrase")):
    return {
        "sentence": list(words),
        "source_file": "demo.jsonl",
        "filter_methods": list(methods),
        "match_tree": [
            {"index": index, "word": words[index], "rule": "locality_phrase", "children": []}
            for index in targets
        ],
    }


BOOK_ON_TABLE = ltp(
    ("书", "在", "桌子", "上", "。"),
    ("n", "v", "n", "nd", "wp"),
    (2, 0, 4, 2, 2),
    ("SBV", "HED", "ATT", "VOB", "WP"),
    (2, 0, 2, 3, 2),
    ("EXP", "Root", "LOC", "mDEPD", "mPUNC"),
)
PUT_BOOK_ON_TABLE = ltp(
    ("他", "把", "书", "放到", "桌子", "上", "。"),
    ("r", "p", "n", "v", "n", "nd", "wp"),
    (4, 4, 2, 0, 6, 4, 4),
    ("SBV", "ADV", "POB", "HED", "ATT", "VOB", "WP"),
    (4, 3, 4, 0, 4, 5, 4),
    ("AGT", "mRELA", "PAT", "Root", "LOC", "mDEPD", "mPUNC"),
)
TABLE_BOOK = ltp(
    ("桌子", "上", "的", "书", "很", "旧", "。"),
    ("n", "nd", "u", "n", "d", "a", "wp"),
    (2, 4, 2, 6, 6, 0, 6),
    ("ATT", "ATT", "RAD", "SBV", "ADV", "HED", "WP"),
    (4, 1, 1, 6, 6, 0, 6),
    ("LOC", "mDEPD", "mDEPD", "EXP", "mDEPD", "Root", "mPUNC"),
)
BIRD_ON_TREE = ltp(
    ("小鸟", "站", "在", "树", "上", "。"),
    ("n", "v", "p", "nl", "nd", "wp"),
    (2, 0, 2, 5, 3, 2),
    ("SBV", "HED", "CMP", "ATT", "POB", "WP"),
    (2, 0, 4, 2, 4, 2),
    ("AGT", "Root", "mRELA", "LOC", "mDEPD", "mPUNC"),
)
EAST_OF_SCHOOL = ltp(
    ("学校", "东边", "有", "一", "条", "河", "。"),
    ("n", "nd", "v", "m", "q", "n", "wp"),
    (2, 3, 0, 5, 6, 3, 3),
    ("ATT", "SBV", "HED", "ATT", "ATT", "VOB", "WP"),
    (3, 1, 0, 3, 4, 3, 3),
    ("LOC", "mDEPD", "Root", "MEAS", "MEAS", "EXP", "mPUNC"),
)
LIBRARY = ltp(
    ("她", "在", "图书馆", "里", "查", "资料", "。"),
    ("r", "p", "n", "nd", "v", "n", "wp"),
    (5, 5, 4, 2, 0, 5, 5),
    ("SBV", "ADV", "ATT", "POB", "HED", "VOB", "WP"),
    (5, 3, 5, 3, 0, 5, 5),
    ("AGT", "mRELA", "LOC", "mDEPD", "Root", "CONT", "mPUNC"),
)
MEETING_BEFORE = ltp(
    ("会议", "前", "三十", "分钟", "，", "他", "到", "了", "。"),
    ("n", "nd", "m", "q", "wp", "r", "v", "u", "wp"),
    (2, 4, 4, 7, 4, 7, 0, 7, 7),
    ("ATT", "ATT", "ATT", "ADV", "WP", "SBV", "HED", "RAD", "WP"),
    (4, 4, 4, 7, 4, 7, 0, 7, 7),
    ("rTIME", "FEAT", "MEAS", "TIME", "mPUNC", "AGT", "Root", "mDEPD", "mPUNC"),
)
THREE_DAYS_AFTER = ltp(
    ("三", "天", "后", "我们", "再", "见面", "。"),
    ("m", "q", "nd", "r", "d", "v", "wp"),
    (3, 3, 6, 6, 6, 0, 6),
    ("ATT", "ATT", "ADV", "SBV", "ADV", "HED", "WP"),
    (3, 6, 3, 6, 6, 0, 6),
    ("MEAS", "TIME", "mDEPD", "EXP", "mDEPD", "Root", "mPUNC"),
)
IN_THEORY = ltp(
    ("理论", "上", "这", "是", "可行", "的", "。"),
    ("n", "nd", "r", "v", "a", "u", "wp"),
    (2, 4, 4, 0, 4, 5, 4),
    ("ATT", "ADV", "SBV", "HED", "VOB", "RAD", "WP"),
    (5, 1, 5, 5, 0, 5, 5),
    ("LOC", "mDEPD", "EXP", "mDEPD", "Root", "mDEPD", "mPUNC"),
)
DRAW_NO_AGT = ltp(
    ("在", "教室", "里", "画", "窗", "外", "的", "风景", "。"),
    ("p", "n", "nd", "v", "nl", "nd", "u", "n", "wp"),
    (4, 3, 1, 0, 6, 8, 6, 4, 4),
    ("ADV", "ATT", "POB", "HED", "ATT", "ATT", "RAD", "VOB", "WP"),
    (2, 4, 2, 0, 8, 5, 5, 4, 4),
    ("mRELA", "LOC", "mDEPD", "Root", "LOC", "mDEPD", "mDEPD", "CONT", "mPUNC"),
)
TABLE_EVIDENCE = ltp(
    ("桌子", "上", "的", "证据", "很", "重要", "。"),
    ("n", "nd", "u", "n", "d", "a", "wp"),
    (2, 4, 2, 6, 6, 0, 6),
    ("ATT", "ATT", "RAD", "SBV", "ADV", "HED", "WP"),
    (4, 1, 1, 6, 6, 0, 6),
    ("LOC", "mDEPD", "mDEPD", "EXP", "mDEPD", "Root", "mPUNC"),
)
MIXED_TARGET = ltp(
    ("书", "在", "桌子", "上", "，", "证据", "在", "桌子", "上", "。"),
    ("n", "v", "n", "nd", "wp", "n", "v", "n", "nd", "wp"),
    (2, 0, 4, 2, 2, 7, 2, 9, 7, 2),
    ("SBV", "HED", "ATT", "VOB", "WP", "SBV", "COO", "ATT", "VOB", "WP"),
    (2, 0, 2, 3, 2, 7, 2, 7, 8, 7),
    ("EXP", "Root", "LOC", "mDEPD", "mPUNC", "EXP", "eSUCC", "LOC", "mDEPD", "mPUNC"),
)
TARGET_NO_SENSE = ltp(
    ("摊点", "在", "桌子", "上", "。"),
    ("n", "p", "n", "nd", "wp"),
    (2, 0, 4, 2, 2),
    ("SBV", "HED", "ATT", "VOB", "WP"),
    (2, 0, 2, 3, 2),
    ("EXP", "Root", "LOC", "mDEPD", "mPUNC"),
)
SELF_COORDINATED = ltp(
    ("桌子", "上", "的", "证据", "和", "书", "很", "旧", "。"),
    ("n", "nd", "u", "n", "c", "n", "d", "a", "wp"),
    (2, 4, 2, 8, 6, 4, 8, 0, 8),
    ("ATT", "ATT", "RAD", "SBV", "LAD", "COO", "ADV", "HED", "WP"),
    (4, 1, 1, 6, 6, 8, 8, 0, 8),
    ("LOC", "mDEPD", "mDEPD", "eCOO", "mRELA", "EXP", "mDEPD", "Root", "mPUNC"),
)
REL_COORDINATED = ltp(
    ("他", "放", "在", "桌子", "上", "的", "证据", "和", "书", "很", "旧", "。"),
    ("r", "v", "p", "n", "nd", "u", "n", "c", "n", "d", "a", "wp"),
    (2, 7, 2, 5, 3, 2, 11, 9, 7, 11, 0, 11),
    ("SBV", "ATT", "CMP", "ATT", "POB", "RAD", "SBV", "LAD", "COO", "ADV", "HED", "WP"),
    (2, 7, 4, 2, 4, 2, 11, 9, 7, 11, 0, 11),
    ("AGT", "rPAT", "mRELA", "LOC", "mDEPD", "mDEPD", "EXP", "mRELA", "eCOO", "mDEPD", "Root", "mPUNC"),
)
SELF_COORDINATED3 = ltp(
    ("桌子", "上", "的", "证据", "和", "资料", "和", "书", "很", "旧", "。"),
    ("n", "nd", "u", "n", "c", "n", "c", "n", "d", "a", "wp"),
    (2, 4, 2, 10, 6, 4, 8, 4, 10, 0, 10),
    ("ATT", "ATT", "RAD", "SBV", "LAD", "COO", "LAD", "COO", "ADV", "HED", "WP"),
    (4, 1, 1, 10, 6, 8, 8, 10, 10, 0, 10),
    ("LOC", "mDEPD", "mDEPD", "EXP", "mRELA", "eCOO", "mRELA", "EXP", "mDEPD", "Root", "mPUNC"),
)
REL_COORDINATED3 = ltp(
    ("他", "放", "在", "桌子", "上", "的", "证据", "、", "资料", "和", "书", "很", "旧", "。"),
    ("r", "v", "p", "n", "nd", "u", "n", "wp", "n", "c", "n", "d", "a", "wp"),
    (2, 7, 2, 5, 3, 2, 13, 9, 7, 11, 7, 13, 0, 13),
    ("SBV", "ATT", "CMP", "ATT", "POB", "RAD", "SBV", "WP", "COO", "LAD", "COO", "ADV", "HED", "WP"),
    (2, 7, 4, 2, 4, 2, 13, 7, 11, 11, 13, 13, 0, 13),
    ("AGT", "rPAT", "mRELA", "LOC", "mDEPD", "mDEPD", "EXP", "mPUNC", "eCOO", "mRELA", "EXP", "mDEPD", "Root", "mPUNC"),
)
COORDINATED_TARGET = ltp(
    ("他", "把", "书", "和", "证据", "放在", "桌子", "上", "。"),
    ("r", "p", "n", "c", "n", "v", "n", "nd", "wp"),
    (6, 6, 2, 5, 3, 0, 8, 6, 6),
    ("SBV", "ADV", "POB", "LAD", "COO", "HED", "ATT", "VOB", "WP"),
    (6, 5, 5, 5, 6, 0, 6, 7, 6),
    ("AGT", "mRELA", "eCOO", "mRELA", "PAT", "Root", "LOC", "mDEPD", "mPUNC"),
)
MIXED_NO_LOCATED = ltp(
    ("书", "在", "桌子", "上", "，", "教室", "里", "很", "安静", "。"),
    ("n", "v", "n", "nd", "wp", "n", "nd", "d", "a", "wp"),
    (2, 0, 4, 2, 2, 7, 9, 9, 2, 2),
    ("SBV", "HED", "ATT", "VOB", "WP", "ATT", "SBV", "ADV", "COO", "WP"),
    (2, 0, 2, 3, 2, 9, 6, 9, 2, 9),
    ("EXP", "Root", "LOC", "mDEPD", "mPUNC", "LOC", "mDEPD", "mDEPD", "eSUCC", "mPUNC"),
)
MIXED = (
    ["会议", "前", "书", "在", "桌子", "上", "。"],
    ["n", "nd", "n", "v", "n", "nd", "wp"],
    {"head": [2, 4, 4, 0, 6, 4, 4], "label": ["ATT", "ADV", "SBV", "HED", "ATT", "VOB", "WP"]},
    {"head": [2, 4, 4, 0, 4, 5, 4], "label": ["rTIME", "FEAT", "EXP", "Root", "LOC", "mDEPD", "mPUNC"]},
)


class FakeGate:
    """按词形返回固定判定；不读 OpenHowNet，供结构与入口测试使用。"""

    def __init__(self, non_physical=(), non_physical_target=()):
        self.non_physical = set(non_physical)
        self.non_physical_target = set(non_physical_target)
        self.words = []

    @staticmethod
    def _physical():
        return {"ok": True, "reason": "ok", "sense_id": "No.test", "sememe": "physical|物质", "path": ["physical|物质"]}

    def check(self, word):
        self.words.append(word)
        if word in self.non_physical:
            return {
                "ok": False,
                "reason": "reference_not_physical",
                "sense_id": None,
                "sememe": None,
                "path": [],
            }
        return self._physical()

    def check_located(self, word):
        self.words.append(word)
        if word in self.non_physical_target:
            return {
                "ok": False,
                "reason": "target_not_physical",
                "sense_id": None,
                "sememe": None,
                "path": [],
            }
        return self._physical()


class ContextTests(unittest.TestCase):
    def test_length_mismatch_is_error(self):
        words, pos, dep, sdp = BOOK_ON_TABLE
        with self.assertRaisesRegex(ValueError, "长度不一致"):
            lg.build_context(words, pos, dep, {"head": [], "label": []})

    def test_invalid_head_is_error(self):
        words, pos, dep, sdp = BOOK_ON_TABLE
        bad = {"head": list(dep["head"]), "label": list(dep["label"])}
        bad["head"][0] = 99
        with self.assertRaisesRegex(ValueError, "DEP.head"):
            lg.build_context(words, pos, bad, sdp)


class CheckTargetTests(unittest.TestCase):
    def check(self, sample, target_index, reference, target, role, verb=None):
        words, pos, dep, sdp = sample
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target({"index": target_index, "word": words[target_index]}, context, gate=FakeGate())
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["reference"]["word"], reference)
        self.assertEqual(verdict["target"]["word"], target)
        self.assertEqual(verdict["target"]["role"], role)
        if verb is not None:
            self.assertEqual(verdict["verb"]["word"], verb)
        self.assertEqual(verdict["reason"], "ok")

    def test_book_on_table(self):
        self.check(BOOK_ON_TABLE, 3, "桌子", "书", "EXP", "在")

    def test_put_book_uses_patient_not_subject(self):
        self.check(PUT_BOOK_ON_TABLE, 5, "桌子", "书", "PAT", "放到")

    def test_table_book_uses_locative_head_as_located(self):
        self.check(TABLE_BOOK, 1, "桌子", "书", "SELF")

    def test_bird_on_tree(self):
        self.check(BIRD_ON_TREE, 4, "树", "小鸟", "AGT", "站")

    def test_follow_locative_chain(self):
        self.check(EAST_OF_SCHOOL, 1, "学校", "河", "EXP", "有")

    def test_activity_patient_is_not_mistaken_for_located(self):
        self.check(LIBRARY, 3, "图书馆", "她", "AGT", "查")

    def test_self_coordinated_target_tries_physical_member(self):
        words, pos, dep, sdp = SELF_COORDINATED
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target(
            {"index": 1, "word": "上"},
            context,
            gate=FakeGate(non_physical_target={"证据"}),
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["target"]["word"], "书")
        self.assertEqual(verdict["target"]["role"], "EXP")

    def test_rel_coordinated_target_tries_physical_member(self):
        words, pos, dep, sdp = REL_COORDINATED
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target(
            {"index": 4, "word": "上"},
            context,
            gate=FakeGate(non_physical_target={"证据"}),
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["target"]["word"], "书")
        self.assertEqual(verdict["target"]["role"], "eCOO")

    def test_self_three_member_coordination_tries_last_physical(self):
        words, pos, dep, sdp = SELF_COORDINATED3
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target(
            {"index": 1, "word": "上"},
            context,
            gate=FakeGate(non_physical_target={"证据", "资料"}),
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["target"]["word"], "书")

    def test_rel_three_member_coordination_tries_last_physical(self):
        words, pos, dep, sdp = REL_COORDINATED3
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target(
            {"index": 4, "word": "上"},
            context,
            gate=FakeGate(non_physical_target={"证据", "资料"}),
        )
        self.assertTrue(verdict["keep"], verdict)
        self.assertEqual(verdict["target"]["word"], "书")

    def test_missing_target_is_rejected(self):
        words, pos, dep, sdp = DRAW_NO_AGT
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target({"index": 2, "word": "里"}, context, gate=FakeGate())
        self.assertFalse(verdict["keep"], verdict)
        self.assertEqual(verdict["reason"], "no_located")
        self.assertEqual(verdict["reference"]["word"], "教室")
        self.assertIsNone(verdict["target"])
        # 动词仍作为证据保留；剔除的原因是目标物缺失，不是动词缺失。
        self.assertEqual(verdict["verb"]["word"], "画")

    def test_temporal_reference_is_not_loc(self):
        words, pos, dep, sdp = MEETING_BEFORE
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target({"index": 1, "word": "前"}, context, gate=FakeGate())
        self.assertFalse(verdict["keep"], verdict)
        self.assertEqual(verdict["reason"], "reference_not_loc")

    def test_quantity_reference_is_not_nominal(self):
        words, pos, dep, sdp = THREE_DAYS_AFTER
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target({"index": 2, "word": "后"}, context, gate=FakeGate())
        self.assertFalse(verdict["keep"], verdict)
        self.assertEqual(verdict["reason"], "no_reference")


class HowNetGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = lg.PhysicalGate()

    def test_physical_nouns_pass(self):
        for word in ("桌子", "学校", "图书馆", "病房", "水"):
            verdict = self.gate.check(word)
            self.assertTrue(verdict["ok"], (word, verdict))
            self.assertTrue(verdict["path"])

    def test_abstract_and_temporal_nouns_fail(self):
        for word in ("会议", "理论", "时间", "上面"):
            verdict = self.gate.check(word)
            self.assertFalse(verdict["ok"], (word, verdict))
            self.assertEqual(verdict["reason"], "reference_not_physical")

    def test_missing_word_has_no_sense(self):
        verdict = self.gate.check("三天")
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "reference_no_sense")

    def test_check_target_rejects_abstract_reference(self):
        words, pos, dep, sdp = IN_THEORY
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target({"index": 1, "word": "上"}, context, gate=self.gate)
        self.assertFalse(verdict["keep"], verdict)
        self.assertEqual(verdict["reason"], "reference_not_physical")
        self.assertEqual(verdict["reference"]["word"], "理论")

    def test_target_gate_only_accepts_physical_subtree(self):
        self.assertTrue(self.gate.check_located("桌子")["ok"])
        self.assertTrue(self.gate.check_located("书")["ok"])
        self.assertTrue(self.gate.check_located("她")["ok"])
        # place|地方 本身在 physical 子树下，所以「城市」仍能通过；
        # 只有 InstitutePlace|场所 这类 organization 分支的目标物会失败。
        self.assertTrue(self.gate.check_located("城市")["ok"])
        self.assertFalse(self.gate.check_located("学校")["ok"])
        self.assertEqual(self.gate.check_located("学校")["reason"], "target_not_physical")
        self.assertFalse(self.gate.check_located("公司")["ok"])
        self.assertEqual(self.gate.check_located("公司")["reason"], "target_not_physical")

    def test_target_without_sense_uses_target_reason(self):
        verdict = self.gate.check_located("摊点")
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["reason"], "target_no_sense")

    def test_reference_and_target_cache_are_isolated(self):
        # 学校作为参照物可通过 InstitutePlace；作为目标物必须只认 physical。
        self.assertTrue(self.gate.check("学校")["ok"])
        self.assertFalse(self.gate.check_located("学校")["ok"])
        self.assertTrue(self.gate.check("学校")["ok"])

    def test_check_target_rejects_abstract_target(self):
        words, pos, dep, sdp = TABLE_EVIDENCE
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target({"index": 1, "word": "上"}, context, gate=self.gate)
        self.assertFalse(verdict["keep"], verdict)
        self.assertEqual(verdict["reason"], "target_not_physical")
        self.assertEqual(verdict["target"]["word"], "证据")
        self.assertFalse(verdict["target_hownet"]["ok"])
        self.assertTrue(verdict["hownet"]["ok"])

    def test_check_target_keeps_physical_target_with_evidence(self):
        words, pos, dep, sdp = BOOK_ON_TABLE
        context = lg.build_context(words, pos, dep, sdp)
        verdict = lg.check_target({"index": 3, "word": "上"}, context, gate=self.gate)
        self.assertTrue(verdict["keep"], verdict)
        self.assertTrue(verdict["target_hownet"]["ok"])
        self.assertIn("physical|物质", verdict["target_hownet"]["path"])
        self.assertTrue(verdict["hownet"]["ok"])

    def test_physical_gate_caches(self):
        first = self.gate.check("桌子")
        second = self.gate.check("桌子")
        self.assertEqual(first, second)


class ApplyGroundingTests(unittest.TestCase):
    def test_keeps_record_and_writes_evidence(self):
        words, pos, dep, sdp = BOOK_ON_TABLE
        row, stats = lg.apply_grounding(
            record(words, [3]), {"上"}, words, pos, dep, sdp, gate=FakeGate()
        )
        self.assertIsNotNone(row)
        self.assertTrue(row["localization_grounding"]["valid"])
        candidate = row["localization_grounding"]["candidates"][0]
        self.assertEqual(candidate["reference"]["word"], "桌子")
        self.assertEqual(candidate["target"]["word"], "书")
        self.assertTrue(candidate["target_hownet"]["ok"])
        self.assertEqual(stats["kept_records"], 1)
        self.assertEqual(stats["kept_candidates"], 1)

    def test_drops_record_when_all_candidates_invalid(self):
        words, pos, dep, sdp = MEETING_BEFORE
        row, stats = lg.apply_grounding(
            record(words, [1]), {"前"}, words, pos, dep, sdp, gate=FakeGate()
        )
        self.assertIsNone(row)
        self.assertEqual(stats["dropped_records"], 1)
        self.assertEqual(stats["reasons"], {"reference_not_loc": 1})

    def test_drops_record_when_only_candidate_has_no_target(self):
        words, pos, dep, sdp = DRAW_NO_AGT
        row, stats = lg.apply_grounding(
            record(words, [2]), {"里"}, words, pos, dep, sdp, gate=FakeGate()
        )
        self.assertIsNone(row)
        self.assertEqual(stats["dropped_records"], 1)
        self.assertEqual(stats["reasons"], {"no_located": 1})
        # 顶层 no_target 专指“规则阶段没有候选方位词”，不能被这里复用。
        self.assertEqual(stats["no_target"], 0)

    def test_prunes_abstract_target_candidate_and_keeps_physical_one(self):
        words, pos, dep, sdp = MIXED_TARGET
        row, stats = lg.apply_grounding(
            record(words, [3, 8]),
            {"上"},
            words,
            pos,
            dep,
            sdp,
            gate=FakeGate(non_physical_target={"证据"}),
        )
        self.assertIsNotNone(row)
        self.assertEqual([node["index"] for node in row["match_tree"]], [3])
        self.assertEqual(stats["candidates"], 2)
        self.assertEqual(stats["kept_candidates"], 1)
        self.assertEqual(stats["dropped_candidates"], 1)
        self.assertEqual(stats["reasons"], {"ok": 1, "target_not_physical": 1})

    def test_drops_when_target_has_no_sense(self):
        words, pos, dep, sdp = TARGET_NO_SENSE
        row, stats = lg.apply_grounding(
            record(words, [3]), {"上"}, words, pos, dep, sdp, gate=lg.PhysicalGate()
        )
        self.assertIsNone(row)
        self.assertEqual(stats["dropped_records"], 1)
        self.assertEqual(stats["reasons"], {"target_no_sense": 1})
        self.assertEqual(stats["no_target"], 0)

    def test_esucc_is_not_treated_as_coordination(self):
        words, pos, dep, sdp = COORDINATED_TARGET
        sdp = {
            "head": list(sdp["head"]),
            "label": list(sdp["label"]),
        }
        # 「书」原本是 eCOO；改成 eSUCC 后不应被当成并列目标物，
        # 否则会拿后一子句的名词去救一个目标物不合格的候选。
        sdp["label"][2] = "eSUCC"
        row, stats = lg.apply_grounding(
            record(words, [7]),
            {"上"},
            words,
            pos,
            dep,
            sdp,
            gate=FakeGate(non_physical_target={"证据"}),
        )
        self.assertIsNone(row)
        self.assertEqual(stats["reasons"], {"target_not_physical": 1})

    def test_coordinated_target_tries_next_physical_candidate(self):
        words, pos, dep, sdp = COORDINATED_TARGET
        row, stats = lg.apply_grounding(
            record(words, [7]),
            {"上"},
            words,
            pos,
            dep,
            sdp,
            gate=FakeGate(non_physical_target={"证据"}),
        )
        self.assertIsNotNone(row)
        candidate = row["localization_grounding"]["candidates"][0]
        self.assertTrue(candidate["keep"], candidate)
        self.assertEqual(candidate["target"]["word"], "书")
        self.assertEqual(candidate["target"]["role"], "eCOO")
        self.assertEqual(stats["kept_candidates"], 1)

    def test_prunes_no_located_candidate_and_keeps_valid_one(self):
        words, pos, dep, sdp = MIXED_NO_LOCATED
        row, stats = lg.apply_grounding(
            record(words, [3, 6]), {"上", "里"}, words, pos, dep, sdp, gate=FakeGate()
        )
        self.assertIsNotNone(row)
        self.assertEqual([node["index"] for node in row["match_tree"]], [3])
        self.assertEqual(stats["candidates"], 2)
        self.assertEqual(stats["kept_candidates"], 1)
        self.assertEqual(stats["dropped_candidates"], 1)
        self.assertEqual(stats["reasons"], {"ok": 1, "no_located": 1})

    def test_prunes_invalid_candidate_and_keeps_valid_one(self):
        words, pos, dep, sdp = MIXED
        row, stats = lg.apply_grounding(
            record(words, [1, 5]), {"前", "上"}, words, pos, dep, sdp, gate=FakeGate()
        )
        self.assertIsNotNone(row)
        self.assertEqual([node["index"] for node in row["match_tree"]], [5])
        self.assertEqual(stats["candidates"], 2)
        self.assertEqual(stats["kept_candidates"], 1)
        self.assertEqual(stats["dropped_candidates"], 1)

    def test_no_target_fails_open(self):
        words, pos, dep, sdp = BOOK_ON_TABLE
        row, stats = lg.apply_grounding(
            record(words, []), {"下"}, words, pos, dep, sdp, gate=FakeGate()
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["localization_grounding"]["source"], "no_target")
        self.assertEqual(stats["no_target"], 1)

    def test_malformed_ltp_fails_open(self):
        words, pos, dep, _sdp = BOOK_ON_TABLE
        row, stats = lg.apply_grounding(
            record(words, [3]), {"上"}, words, pos, dep, {"head": [], "label": []}, gate=FakeGate()
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["localization_grounding"]["source"], "error")
        self.assertEqual(stats["errors"], 1)

    def test_stats_merge_is_additive(self):
        total = lg.empty_stats()
        lg.merge_stats(total, {"records": 2, "reasons": {"no_reference": 1}})
        lg.merge_stats(total, {"records": 3, "reasons": {"no_reference": 2, "ok": 1}})
        self.assertEqual(total["records"], 5)
        self.assertEqual(total["reasons"], {"no_reference": 3, "ok": 1})

    def test_prune_candidates_follows_llm_pruning(self):
        words, pos, dep, sdp = MIXED
        row, _ = lg.apply_grounding(
            record(words, [1, 5]), {"前", "上"}, words, pos, dep, sdp, gate=FakeGate()
        )
        self.assertEqual(len(row["localization_grounding"]["candidates"]), 2)
        lg.prune_candidates(row, {1}, {5})
        self.assertEqual(len(row["localization_grounding"]["candidates"]), 1)
        self.assertTrue(row["localization_grounding"]["valid"])
        lg.prune_candidates(row, {5}, set())
        self.assertEqual(row["localization_grounding"]["candidates"], [])
        self.assertFalse(row["localization_grounding"]["valid"])


def _fake_cache_get(sample):
    words, pos, dep, sdp = sample
    values = {config.CWS: words, config.POS: pos, config.DEP: dep, config.SDP: sdp}

    def get(_sentence, task):
        return values[task]

    return get


class _FakeFilter:
    def __init__(self, index, word):
        self.index = index
        self.word = word

    def explain(self, _sentence):
        node = sentenceFilters.MatchNode(
            name="locality_as_locative_role",
            index=self.index,
            word=self.word,
            conditions={"_vocab": True, "_pos": True, "_syntax": True, "_semantic": True},
            children=[],
        )
        return True, {"methods": ["root", "locality_as_locative_role"], "match_tree": [node]}


class PrepareFamiliesTests(unittest.TestCase):
    def test_direction_family_does_not_load_hownet(self):
        with patch.object(lg, "PhysicalGate", side_effect=AssertionError("不应加载 HowNet")):
            metas, gate = lg.prepare_families(
                ["config_files/direction_expressions.json5"], lambda family: "趋向动词"
            )
        self.assertIsNone(gate)
        self.assertIsNone(metas["direction_expressions"]["vocab"])

    def test_localization_family_builds_vocab_and_gate(self):
        with patch.object(lg, "PhysicalGate", return_value=FakeGate()):
            metas, gate = lg.prepare_families(
                ["config_files/localization_expressions.json5"], lambda family: "方位词"
            )
        self.assertIsInstance(gate, FakeGate)
        self.assertIn("上", metas["localization_expressions"]["vocab"])

    def test_screen_kind_overrides_unknown_file_name(self):
        with patch.object(lg, "PhysicalGate", return_value=FakeGate()):
            metas, gate = lg.prepare_families(
                ["config_files/localization_expressions.json5"],
                lambda family: None,
                kind="方位词",
            )
        self.assertIsInstance(gate, FakeGate)
        self.assertIn("上", metas["localization_expressions"]["vocab"])


class GateFailureTests(unittest.TestCase):
    def test_hownet_failure_aborts_instead_of_failing_open(self):
        class BrokenGate:
            def check(self, _word):
                raise lg.HowNetUnavailable("词典损坏")

        words, pos, dep, sdp = BOOK_ON_TABLE
        with self.assertRaises(lg.HowNetUnavailable):
            lg.apply_grounding(
                record(words, [3]), {"上"}, words, pos, dep, sdp, gate=BrokenGate()
            )

    def test_target_hownet_failure_aborts(self):
        class BrokenTargetGate:
            def check(self, _word):
                return {"ok": True, "reason": "ok", "sense_id": None, "sememe": None, "path": []}

            def check_located(self, _word):
                raise lg.HowNetUnavailable("目标物查询失败")

        words, pos, dep, sdp = BOOK_ON_TABLE
        with self.assertRaises(lg.HowNetUnavailable):
            lg.apply_grounding(
                record(words, [3]), {"上"}, words, pos, dep, sdp, gate=BrokenTargetGate()
            )

    def test_broken_sememe_data_raises_hownet_unavailable(self):
        class BadSememe:
            @property
            def en_zh(self):
                raise RuntimeError("坏数据")

        class FakeHowNet:
            sense_dic = {"x": object()}
            zh_map = {"x": [object()]}

            def get_sense(self, _word, language=None, strict=True):
                return [SimpleNamespace(No="No.x", sememes=[BadSememe()])]

        gate = lg.PhysicalGate(hownet=FakeHowNet())
        with self.assertRaises(lg.HowNetUnavailable):
            gate.check("桌子")


class EntryWiringTests(unittest.TestCase):
    def run_batch(self, module, sample, target_index, word, vocab):
        with patch.object(module, "_WORKER_FILTERS", [("localization_expressions", _FakeFilter(target_index, word))]), patch.object(
            sentenceFilters.cache, "prefill", lambda _sentences: 0
        ), patch.object(sentenceFilters.cache, "get_task_value", side_effect=_fake_cache_get(sample)):
            return module._filter_sentence_batch([{"sentence": "样本句"}])

    def test_linux_attaches_analysis_for_localization(self):
        batches, _ = self.run_batch(self._module("entry_in_linux"), BOOK_ON_TABLE, 3, "上", {"上"})
        row = batches[0][0]
        self.assertIn("_locative_analysis", row)
        kept, stats = lg.gate_rows(batches[0], {"上"}, gate=FakeGate())
        self.assertEqual(len(kept), 1)
        self.assertNotIn("_locative_analysis", kept[0])
        self.assertEqual(kept[0]["localization_grounding"]["candidates"][0]["reference"]["word"], "桌子")
        self.assertEqual(stats["kept_candidates"], 1)

    def test_windows_attaches_analysis_for_localization(self):
        batches, _ = self.run_batch(self._module("entry_in_win"), BOOK_ON_TABLE, 3, "上", {"上"})
        self.assertIn("_locative_analysis", batches[0][0])

    def test_screen_kind_override_enables_attachment_for_unknown_family_name(self):
        module = self._module("entry_in_linux")
        with patch.object(module, "_WORKER_KIND_OVERRIDE", "方位词"), patch.object(
            module, "_WORKER_FILTERS", [("custom_rules", _FakeFilter(3, "上"))]
        ), patch.object(sentenceFilters.cache, "prefill", lambda _sentences: 0), patch.object(
            sentenceFilters.cache, "get_task_value", side_effect=_fake_cache_get(BOOK_ON_TABLE)
        ):
            batches, _ = module._filter_sentence_batch([{"sentence": "书在桌子上。"}])
        self.assertIn("_locative_analysis", batches[0][0])

    def test_direction_family_is_not_attached(self):
        module = self._module("entry_in_linux")
        with patch.object(module, "_WORKER_FILTERS", [("direction_expressions", _FakeFilter(3, "上"))]), patch.object(
            sentenceFilters.cache, "prefill", lambda _sentences: 0
        ), patch.object(sentenceFilters.cache, "get_task_value", side_effect=_fake_cache_get(BOOK_ON_TABLE)):
            batches, hits = module._filter_sentence_batch([{"sentence": "书在桌子上。"}])
        self.assertEqual(hits, 1)
        self.assertNotIn("_locative_analysis", batches[0][0])

    def test_streaming_gate_drops_before_writing(self):
        import entry_in_linux

        row = record(MEETING_BEFORE[0], [1])
        row["_locative_analysis"] = {
            "pos": MEETING_BEFORE[1],
            "dep": MEETING_BEFORE[2],
            "sdp": MEETING_BEFORE[3],
        }
        out = io.StringIO()
        metas = {"localization_expressions": {"kind": "方位词", "vocab": {"前"}}}
        with patch.object(entry_in_linux, "_init_filter_worker", lambda paths: None), patch.object(
            entry_in_linux, "_filter_sentence_batch", lambda batch: ([[row]], 1)
        ), patch.object(entry_in_linux.localization_grounding, "prepare_families", lambda paths, classify, kind=None: (metas, FakeGate())), patch.object(
            entry_in_linux, "_print_batch_progress", lambda *args, **kwargs: None
        ):
            entry_in_linux.filter_sentences_streaming(
                [{"sentence": "会议前三十五分钟，他到了。"}],
                ["localization_expressions.json5"],
                [out],
                filter_workers=1,
            )
        self.assertEqual(out.getvalue(), "")

    @staticmethod
    def _module(name):
        import importlib

        return importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
