# encoding: utf8
"""针对本轮修复的离线回归：路径安全、原子落盘、行号契约、LTP 对齐与签名完整性。"""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import batch_inference as bi
import config
import export_pairs_csv
import file_paths
import sentenceTable as st_table
import spatial_runner as sr
import spatial_tools as st
import word_sampler as ws


class PathSafetyTests(unittest.TestCase):
    def test_batch_sidecar_cannot_alias_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            out = Path(tmp) / "out.jsonl"
            with self.assertRaises(ValueError):
                file_paths.validate_run_outputs(
                    corpus,
                    out,
                    extra_outputs=[corpus],
                    overwrite=True,
                )

    def test_batch_cache_path_resolution_is_shared_and_model_scoped(self):
        out = Path("results") / "run.jsonl"
        default = bi.resolve_cache_path(None, out)
        self.assertEqual(default, Path("results") / "run.jsonl.batch-cache.jsonl")
        scoped = bi.resolve_cache_path(None, out, model_key="m1", multi_model=True)
        self.assertEqual(scoped, Path("results") / "run.jsonl.batch-cache.m1.jsonl")
        cache, jobs, meta = bi.sidecar_paths(scoped)
        self.assertEqual(cache, scoped)
        self.assertEqual(jobs, scoped.with_suffix(scoped.suffix + ".jobs.json"))
        self.assertEqual(meta, scoped.with_suffix(scoped.suffix + ".meta.json"))

        # 多模型预检必须按各模型实际输出（raw_paths[name]）推导，而不是最终投票文件
        args = argparse.Namespace(batch_cache=None, models=["a", "b"])
        raw = Path("results") / "run.a.jsonl"
        self.assertEqual(
            sr._batch_cache_path(args, raw, "a"),
            Path("results") / "run.a.jsonl.batch-cache.a.jsonl",
        )


class WordSamplerTests(unittest.TestCase):
    def test_resample_preserves_source_line_no_and_writes_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            record = {
                "screening": {"status": "no_llm"},
                "sentence": ["书", "放", "在", "桌子", "上"],
                "source_line_no": 42,
            }
            corpus.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            out = Path(tmp) / "sampled.jsonl"

            written = ws.write_sample(corpus, {1: ["上"]}, out)

            self.assertEqual(written, 1)
            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(saved["source_line_no"], 42)
            self.assertEqual(saved["line_no"], 1)
            self.assertEqual(saved["sampled_words"], ["上"])
            self.assertEqual(list(Path(tmp).glob(".*.tmp")), [])

    def test_output_cannot_overwrite_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "sampled_corpus.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            # out_paths 对 sampled_corpus.jsonl 仍会生成 sampled_sampled_corpus.jsonl，
            # 这里直接验证底层检查能识别同一路径。
            with self.assertRaises(ValueError):
                file_paths.validate_output_paths([corpus], [corpus])

    def test_multi_input_preflight_catches_output_that_is_another_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "foo.jsonl"
            # 第一份的输出名恰好是第二份输入：必须在任何扫描/写入前拒绝
            second = Path(tmp) / "sampled_foo.jsonl"
            first.write_text("{}\n", encoding="utf-8")
            second.write_text("{}\n", encoding="utf-8")
            # 必须匹配「路径冲突」本身，而不是后续 vocab_for 的未知语料报错；
            # 同时 mock 掉处理函数，保证一旦预检漏掉，测试不会因别的原因仍然通过。
            with mock.patch.object(ws, "run_corpus") as run_corpus:
                with self.assertRaisesRegex(SystemExit, "输出路径.*指向同一文件"):
                    ws.main(
                        [
                            "--corpus",
                            str(first),
                            "--corpus",
                            str(second),
                            "--out-dir",
                            tmp,
                            "--dry-run",
                        ]
                    )
            run_corpus.assert_not_called()


class SentenceTableAlignmentTests(unittest.TestCase):
    def test_mixed_short_and_long_rerun_discards_split(self):
        class FakeLTP:
            def pipeline(self, sentences, tasks):
                # pos 比词表长、dep head 比词表短：旧写法先截长、跳过短检查，
                # 会把长度仍不一致的结果写进缓存。
                return {
                    config.POS: [["n", "nl", "x"]],
                    config.DEP: [{"head": [2], "label": ["ATT", "LOC"]}],
                    config.SDP: [{"head": [2, 0], "label": ["LOC", "Root"]}],
                }

        values = [
            {
                config.CWS: ["桌子上"],
                config.POS: ["nl"],
                config.DEP: {"head": [0], "label": ["ROOT"]},
                config.SDP: {"head": [0], "label": ["Root"]},
            }
        ]
        with mock.patch.object(st_table, "get_suffix_vocab", return_value=("上",)), mock.patch.object(
            st_table, "split_tokens_by_suffix", return_value=["桌子", "上"]
        ):
            st_table.SentenceTaskCache()._correct_splits(FakeLTP(), values)

        self.assertEqual(values[0][config.CWS], ["桌子上"])
        self.assertEqual(values[0][config.POS], ["nl"])
        self.assertEqual(values[0][config.DEP], {"head": [0], "label": ["ROOT"]})


class SignatureTests(unittest.TestCase):
    def test_signature_tracks_vocab_content_and_exclusion_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            vocab = Path(tmp) / "vocab.json5"
            vocab.write_text('["上"]', encoding="utf-8")
            with mock.patch.object(config, "LOCATIVE_SPLIT_VOCAB_FILE", str(vocab)):
                first = config.segmentation_signature()
                vocab.write_text('["上", "下"]', encoding="utf-8")
                second = config.segmentation_signature()

            self.assertNotEqual(first["locative_split_vocab_sha256"], second["locative_split_vocab_sha256"])
            self.assertIn("locative_split_suffix_exclude", first)
            self.assertIn("locative_split_token_blacklist", first)
            self.assertEqual(first["locative_split_allowed_pos"], sorted(config.LOCATIVE_SPLIT_ALLOWED_POS))


class StreamingAndWriterTests(unittest.TestCase):
    def test_iter_jsonl_matches_read_jsonl_and_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
            self.assertEqual(list(st.iter_jsonl(path)), [{"a": 1}, {"a": 2}])
            self.assertEqual(st.read_jsonl(path), [{"a": 1}, {"a": 2}])

    def test_result_writer_line_is_visible_before_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            writer = sr.ResultWriter(
                path,
                counter_names=(),
                tally=lambda record, counters: None,
                overwrite=True,
            )
            writer.write({"line_no": 1, "candidate": {"index": 0}})
            self.assertEqual(len(st.read_jsonl(path)), 1)
            writer._close()

    def test_result_writer_closes_handle_when_run_raises(self):
        class BoomSpec:
            csv_columns = ()
            counter_names = ()
            captured = None

            @staticmethod
            def tally(record, counters):
                return None

            def build(self, agents, writer):
                # 保住强引用：否则 assertRaises 清理 traceback 后 writer 可能靠 __del__
                # 才关闭，旧实现也会让测试通过。
                type(self).captured = writer
                raise RuntimeError("boom")

        args = argparse.Namespace(
            no_llm=True,
            no_pairs_csv=True,
            overwrite=True,
            workers=1,
            batch="off",
            models=None,
            batch_cache=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            with self.assertRaises(RuntimeError):
                sr._execute_run(
                    args,
                    BoomSpec(),
                    "方位词",
                    Path(tmp) / "corpus.jsonl",
                    [],
                    {},
                    path,
                    model_key="fake",
                    prefill_ltp=False,
                    verbose=False,
                )
            writer = BoomSpec.captured
            self.assertIsNotNone(writer)
            # 直接断言生产路径已经关闭，而不是只靠 Windows 上 unlink 的成功/失败；
            # 这样在没有 finally 的实现上，测试一定失败（不依赖 __del__ 的时机）。
            self.assertIsNone(writer._handle)
            path.unlink()
            self.assertFalse(path.exists())

    def test_stream_export_keeps_old_csv_when_input_breaks_midway(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "records.jsonl"
            target = Path(tmp) / "records.csv"
            source.write_text('{"a": 1}\nnot-json\n', encoding="utf-8")
            target.write_text("old", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                export_pairs_csv.main([str(source), "-o", str(target)])
            self.assertEqual(target.read_text(encoding="utf-8"), "old")


if __name__ == "__main__":
    unittest.main(verbosity=2)
