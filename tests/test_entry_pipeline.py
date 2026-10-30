"""入口层的并行抽取与 LLM 筛选衔接（不加载 LTP 模型、不调 LLM）。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import entry_in_linux


class ExtractionWorkerTests(unittest.TestCase):
    def test_parallel_extraction_preserves_article_order(self):
        with tempfile.TemporaryDirectory(prefix="extract-workers-") as directory:
            source = Path(directory) / "src.jsonl"
            source.write_text(
                "\n".join(json.dumps({"content": text}, ensure_ascii=False) for text in ("第一篇", "第二篇", "第三篇")) + "\n",
                encoding="utf-8",
            )
            with patch.object(entry_in_linux, "get_sentences_list", side_effect=lambda article, vocabs=None: [article]):
                rows = list(entry_in_linux.iter_jsonl_file_sentences(str(source), workers=3))
        self.assertEqual([row["sentence"] for row in rows], ["第一篇", "第二篇", "第三篇"])


class ScreeningIntegrationTests(unittest.TestCase):
    def test_filter_stream_applies_screener_before_writing(self):
        class FakeScreener:
            def __init__(self):
                self.calls = 0

            def screen_rows(self, rows):
                self.calls += 1
                return [
                    {
                        **row,
                        "screening": {
                            "status": "false" if row["sentence"] == "drop" else "true",
                            "source": "llm",
                            "extractable": row["sentence"] != "drop",
                            "verdicts": [],
                        },
                    }
                    for row in rows
                ]

            def describe(self):
                return "fake screener"

        def fake_batch(batch):
            rows = [[{**item, "sentence": item["sentence"]} for item in batch]]
            return rows, len(batch)

        screener = FakeScreener()
        out = io.StringIO()
        sentences = [{"sentence": "keep"}, {"sentence": "drop"}, {"sentence": "keep2"}]
        with patch.object(entry_in_linux, "_init_filter_worker", lambda paths: None), patch.object(
            entry_in_linux, "_filter_sentence_batch", fake_batch
        ):
            entry_in_linux.filter_sentences_streaming(
                sentences,
                ["rule.json5"],
                [out],
                screeners={"rule": screener},
                filter_workers=1,
            )
        written = [json.loads(line)["sentence"] for line in out.getvalue().splitlines()]
        self.assertEqual(written, ["keep", "drop", "keep2"])
        statuses = [json.loads(line)["screening"]["status"] for line in out.getvalue().splitlines()]
        self.assertEqual(statuses, ["true", "false", "true"])
        self.assertGreaterEqual(screener.calls, 1)

    def test_duplicate_identical_rows_count_as_two_hits(self):
        def fake_batch(batch):
            rows = [[{**item} for item in batch]]
            return rows, len(batch)

        out = io.StringIO()
        logs = io.StringIO()
        sentences = [
            {"sentence": "同一句", "source_file": "same.jsonl"},
            {"sentence": "同一句", "source_file": "same.jsonl"},
        ]
        with patch.object(entry_in_linux, "_init_filter_worker", lambda paths: None), patch.object(
            entry_in_linux, "_filter_sentence_batch", fake_batch
        ), redirect_stdout(logs):
            entry_in_linux.filter_sentences_streaming(sentences, ["rule.json5"], [out], filter_workers=1)
        self.assertEqual(len(out.getvalue().splitlines()), 2)
        self.assertIn("hit：2", logs.getvalue())
        self.assertNotIn("_corpus_row_id", out.getvalue())


if __name__ == "__main__":
    unittest.main()
