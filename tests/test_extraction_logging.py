"""语料筛选漏斗日志：计数口径、日志落盘与失败降级（不加载 LTP、不调 LLM）。"""

import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import entry_in_linux
import extraction_log


def _tagged(rows):
    """模拟入口里的 tag_rows：给每条记录补整轮唯一键"""
    return [{**row, "_corpus_row_id": index} for index, row in enumerate(rows)]


class _DropScreener:
    """把所有记录都返回；sentence == "drop" 的标 false，其余标 true"""

    def screen_rows(self, rows):
        result = []
        for row in rows:
            status = "false" if row["sentence"] == "drop" else "true"
            result.append(
                {
                    **row,
                    "screening": {
                        "kind": "方位词",
                        "source": "llm",
                        "status": status,
                        "extractable": status == "true",
                        "verdicts": [],
                    },
                }
            )
        return result

    def describe(self):
        return "drop screener"


class LogWriterTests(unittest.TestCase):
    def test_creates_log_dir_and_timestamped_file_with_funnel_counts(self):
        with tempfile.TemporaryDirectory(prefix="extract-log-") as directory:
            log_dir = Path(directory) / "log"
            stats = extraction_log.new_stats()
            stats.update(
                read_sentences=10,
                vocab_candidates=6,
                program_selected=3,
                corpus_written=3,
                llm_kept=2,
                llm_excluded=1,
            )
            stats["families"]["direction_expressions"] = {
                "program_selected": 3,
                "written": 3,
                "llm_kept": 2,
                "llm_excluded": 1,
            }
            path = extraction_log.write_extract_log(
                stats,
                src_path="input.jsonl",
                filter_configs=["direction_expressions.json5"],
                output_paths=["out.jsonl"],
                vocab_count=4,
                log_dir=log_dir,
            )

            self.assertIsNotNone(path)
            self.assertTrue(path.exists())
            self.assertTrue(path.parent.is_dir())
            self.assertRegex(path.name, r"^extract-log-\d{8}-\d{6}-\d{6}\.log$")
            text = path.read_text(encoding="utf-8")
            self.assertIn("总共读入的句子数量: 10", text)
            self.assertIn("初步程序筛选筛出的句子数量: 3", text)
            self.assertIn("写出的语料条数（全部标注）: 3", text)
            self.assertIn("可用于判定工作流的句子数量（true/no_llm）: 2", text)
            self.assertIn("被 LLM 判为不可提取的句子数量（已落盘）: 1", text)
            self.assertIn("词表初筛后进入规则筛选的句子数量: 6", text)
            self.assertIn(
                "direction_expressions: 初步程序筛选=3, 落盘=3, 可判定=2, LLM 判假=1", text
            )

    def test_write_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory(prefix="extract-log-fail-") as directory:
            blocker = Path(directory) / "log"
            blocker.write_text("我是文件，不是目录", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                path = extraction_log.write_extract_log(
                    extraction_log.new_stats(),
                    src_path="input.jsonl",
                    filter_configs=["rules.json5"],
                    output_paths=["out.jsonl"],
                    log_dir=blocker,
                )
            self.assertIsNone(path)
            self.assertIn("写入", stderr.getvalue())


class ReadCountTests(unittest.TestCase):
    def _source(self, directory, texts):
        source = Path(directory) / "src.jsonl"
        source.write_text(
            "\n".join(json.dumps({"content": text}, ensure_ascii=False) for text in texts) + "\n",
            encoding="utf-8",
        )
        return source

    def test_read_count_includes_sentences_dropped_by_vocab_prefilter(self):
        with tempfile.TemporaryDirectory(prefix="extract-read-") as directory:
            source = self._source(directory, ["第一篇"])
            stats = extraction_log.new_stats()
            with patch.object(
                entry_in_linux,
                "get_sentences_list",
                return_value=["含目标词上的句子", "不含目标词的句子"],
            ):
                rows = list(
                    entry_in_linux.iter_jsonl_file_sentences(
                        str(source), vocabs=["上"], workers=1, stats=stats
                    )
                )
        self.assertEqual(stats["read_sentences"], 2)
        self.assertEqual([row["sentence"] for row in rows], ["含目标词上的句子"])

    def test_parallel_extraction_also_counts_before_vocab_prefilter(self):
        with tempfile.TemporaryDirectory(prefix="extract-read-parallel-") as directory:
            source = self._source(directory, ["第一篇", "第二篇", "第三篇"])
            stats = extraction_log.new_stats()
            with patch.object(
                entry_in_linux,
                "get_sentences_list",
                return_value=["含目标词上的句子", "不含目标词的句子"],
            ):
                rows = list(
                    entry_in_linux.iter_jsonl_file_sentences(
                        str(source), vocabs=["上"], workers=3, stats=stats
                    )
                )
        self.assertEqual(stats["read_sentences"], 6)
        self.assertEqual(len(rows), 3)


class ScreeningFunnelTests(unittest.TestCase):
    def setUp(self):
        self.stats = extraction_log.new_stats()
        self.stats["read_sentences"] = 5
        self.out = io.StringIO()

    @staticmethod
    def _batch(batch):
        return [[{**item} for item in batch]], len(batch)

    def _run(self, sentences, families, screeners):
        with patch.object(entry_in_linux, "_init_filter_worker", lambda paths: None), patch.object(
            entry_in_linux, "_filter_sentence_batch", self._batch
        ), redirect_stdout(io.StringIO()):
            return entry_in_linux.filter_sentences_streaming(
                _tagged(sentences),
                families,
                [self.out],
                screeners=screeners,
                filter_workers=1,
                stats=self.stats,
            )

    def test_program_and_llm_counts(self):
        returned = self._run(
            [{"sentence": "keep"}, {"sentence": "drop"}, {"sentence": "keep2"}],
            ["rule.json5"],
            {"rule": _DropScreener()},
        )
        self.assertIs(returned, self.stats)
        self.assertEqual(self.stats["read_sentences"], 5)
        self.assertEqual(self.stats["vocab_candidates"], 3)
        self.assertEqual(self.stats["program_selected"], 3)
        self.assertEqual(self.stats["corpus_written"], 3)
        self.assertEqual(self.stats["llm_kept"], 2)
        self.assertEqual(self.stats["llm_excluded"], 1)
        self.assertEqual(
            self.stats["families"]["rule"],
            {
                "program_selected": 3,
                "written": 3,
                "llm_kept": 2,
                "llm_excluded": 1,
                "llm_error": 0,
                "llm_limit": 0,
                "no_llm": 0,
            },
        )
        self.assertEqual(len(self.out.getvalue().splitlines()), 3)

    def test_no_screening_means_zero_llm_exclusions(self):
        self._run([{"sentence": "keep"}, {"sentence": "also"}], ["rule.json5"], None)
        self.assertEqual(self.stats["program_selected"], 2)
        self.assertEqual(self.stats["corpus_written"], 2)
        self.assertEqual(self.stats["llm_kept"], 2)
        self.assertEqual(self.stats["llm_excluded"], 0)
        self.assertEqual(self.stats["no_llm"], 2)

    def test_multi_family_dedup_and_per_family_breakdown(self):
        self.out = io.StringIO()
        out_a, out_b = io.StringIO(), io.StringIO()

        def two_families(batch):
            return [[{**item} for item in batch], [{**item} for item in batch]], len(batch)

        with patch.object(entry_in_linux, "_init_filter_worker", lambda paths: None), patch.object(
            entry_in_linux, "_filter_sentence_batch", two_families
        ), redirect_stdout(io.StringIO()):
            entry_in_linux.filter_sentences_streaming(
                _tagged([{"sentence": "drop"}]),
                ["a_expressions.json5", "b_expressions.json5"],
                [out_a, out_b],
                screeners={"a_expressions": _DropScreener()},
                filter_workers=1,
                stats=self.stats,
            )
        # 同一句命中两族：全局只算一句，a 族被 LLM 排除、b 族保留 → 全局排除 0
        self.assertEqual(self.stats["program_selected"], 1)
        self.assertEqual(self.stats["llm_kept"], 1)
        self.assertEqual(self.stats["llm_excluded"], 0)
        self.assertEqual(
            self.stats["families"]["a_expressions"],
            {
                "program_selected": 1,
                "written": 1,
                "llm_kept": 0,
                "llm_excluded": 1,
                "llm_error": 0,
                "llm_limit": 0,
                "no_llm": 0,
            },
        )
        self.assertEqual(
            self.stats["families"]["b_expressions"],
            {
                "program_selected": 1,
                "written": 1,
                "llm_kept": 1,
                "llm_excluded": 0,
                "llm_error": 0,
                "llm_limit": 0,
                "no_llm": 1,
            },
        )


class WindowsFunnelTests(unittest.TestCase):
    def test_windows_counts_match_linux_semantics(self):
        import entry_in_win

        class FakePbar:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def update(self, count):
                return None

            def set_postfix(self, *args, **kwargs):
                return None

        def batch(items):
            return [[{**item} for item in items]], len(items)

        stats = extraction_log.new_stats()
        stats["read_sentences"] = 5
        out = io.StringIO()
        with patch.object(entry_in_win, "_init_filter_worker", lambda paths: None), patch.object(
            entry_in_win, "_filter_sentence_batch", batch
        ), patch.object(entry_in_win, "tqdm", lambda *args, **kwargs: FakePbar()), redirect_stdout(
            io.StringIO()
        ):
            entry_in_win.filter_sentences_streaming(
                _tagged([{"sentence": "keep"}, {"sentence": "drop"}]),
                ["rule.json5"],
                [out],
                screeners={"rule": _DropScreener()},
                filter_workers=1,
                stats=stats,
            )
        self.assertEqual(stats["read_sentences"], 5)
        self.assertEqual(stats["vocab_candidates"], 2)
        self.assertEqual(stats["program_selected"], 2)
        self.assertEqual(stats["corpus_written"], 2)
        self.assertEqual(stats["llm_kept"], 1)
        self.assertEqual(stats["llm_excluded"], 1)
        self.assertEqual(len(out.getvalue().splitlines()), 2)


class RunPipelineTests(unittest.TestCase):
    def test_run_counts_read_before_vocab_and_merges_old_streaming_shape(self):
        import entry_in_linux

        with tempfile.TemporaryDirectory(prefix="extract-run-") as directory:
            root = Path(directory)
            source = root / "src.jsonl"
            source.write_text(
                json.dumps({"content": "一篇文章"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            def batch(items):
                return [[{**item} for item in items]], len(items)

            with patch.object(
                entry_in_linux,
                "get_sentences_list",
                side_effect=lambda article, vocabs=None: ["含目标词上的句子", "不含目标词的句子"],
            ), patch.object(entry_in_linux, "_init_filter_worker", lambda *args, **kwargs: None), patch.object(
                entry_in_linux, "_filter_sentence_batch", batch
            ), redirect_stdout(
                io.StringIO()
            ):
                stats = entry_in_linux.run(
                    str(source),
                    ["config_files/direction_expressions.json5"],
                    [root / "out.jsonl"],
                    vocabs=["上"],
                )
        self.assertEqual(stats["read_sentences"], 2)
        self.assertEqual(stats["vocab_candidates"], 1)
        self.assertEqual(stats["program_selected"], 1)
        self.assertEqual(stats["llm_kept"], 1)
        self.assertEqual(stats["llm_excluded"], 0)
        self.assertEqual(stats["no_llm"], 1)


class LogNameTests(unittest.TestCase):
    def test_filename_uses_timestamp(self):
        from datetime import datetime

        path = extraction_log.log_path(
            "/tmp", now=datetime(2026, 10, 29, 15, 30, 0, 123456)
        )
        self.assertEqual(str(path), "/tmp/extract-log-20261029-153000-123456.log")


class MainWiringTests(unittest.TestCase):
    def test_main_writes_one_log_after_successful_run(self):
        import sentence_extract

        class FakeScreener:
            def describe(self):
                return "fake screener"

            def warm(self):
                return None

            def stats(self):
                return {"enabled": False}

        with tempfile.TemporaryDirectory(prefix="extract-main-") as directory:
            root = Path(directory)
            log_dir = root / "log"
            stats = extraction_log.new_stats()
            stats.update(
                read_sentences=9,
                vocab_candidates=5,
                program_selected=3,
                corpus_written=3,
                llm_kept=2,
                llm_excluded=1,
            )
            stats["families"]["direction_expressions"] = {
                "program_selected": 3,
                "written": 3,
                "llm_kept": 2,
                "llm_excluded": 1,
            }
            with patch.object(
                sentence_extract.sentence_screening,
                "build_screeners",
                return_value={"direction_expressions": FakeScreener()},
            ), patch.object(sentence_extract, "run_linux_entry", return_value=stats), redirect_stdout(io.StringIO()):
                sentence_extract.main(
                    str(root / "src.jsonl"),
                    str(root),
                    "config_files/direction_expressions.json5",
                    out_name="out",
                    log_dir=str(log_dir),
                )
            logs = list(log_dir.glob("extract-log-*.log"))
            self.assertEqual(len(logs), 1)
            text = logs[0].read_text(encoding="utf-8")
            self.assertIn("总共读入的句子数量: 9", text)
            self.assertIn("初步程序筛选筛出的句子数量: 3", text)
            self.assertIn("可用于判定工作流的句子数量（true/no_llm）: 2", text)
            self.assertIn("被 LLM 判为不可提取的句子数量（已落盘）: 1", text)


if __name__ == "__main__":
    unittest.main()
