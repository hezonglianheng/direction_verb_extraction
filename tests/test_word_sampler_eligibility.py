"""抽样阶段只统计 true/no_llm 语料，避免不可判定记录挤占逐词配额。"""

import json
import tempfile
import unittest
from pathlib import Path

from word_sampler import scan_by_word, write_sample


def record(status=None):
    row = {
        "sentence": ["在", "桌子", "上", "。"],
        "source_file": "demo.jsonl",
        "filter_methods": ["root"],
        "match_tree": [{"index": 2, "word": "上", "children": []}],
    }
    if status is not None:
        row["screening"] = {"status": status, "source": "llm" if status != "no_llm" else "no_llm"}
    return row


class WordSamplerEligibilityTests(unittest.TestCase):
    def test_scan_and_write_skip_ineligible_records(self):
        with tempfile.TemporaryDirectory(prefix="sampler-eligible-") as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            rows = [record("false"), record("true"), record("error"), record(), record("limit")]
            corpus.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            lines_by_word, counts = scan_by_word(corpus, {"上"})
            self.assertEqual(counts["total"], 5)
            self.assertEqual(counts["ineligible"], 3)
            self.assertEqual(lines_by_word["上"], [2, 4])

            chosen = {2: ["上"], 4: ["上"]}
            out = root / "sampled.jsonl"
            self.assertEqual(write_sample(corpus, chosen, out), 2)
            written = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["source_line_no"] for row in written], [2, 4])


if __name__ == "__main__":
    unittest.main()
