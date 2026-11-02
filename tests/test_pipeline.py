import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pipeline


class PipelineTests(unittest.TestCase):
    def test_all_six_contiguous_ranges_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="stage paths ") as directory:
            root = Path(directory)
            for name in ("source.jsonl", "extracted.jsonl", "pairs.jsonl"):
                (root / name).write_text("{}\n", encoding="utf-8")
            for start, end, expected in (
                ("extract", "extract", ["extract"]),
                ("extract", "replace", ["extract", "replace"]),
                ("extract", "annotate", ["extract", "replace", "annotate"]),
                ("replace", "replace", ["replace"]),
                ("replace", "annotate", ["replace", "annotate"]),
                ("annotate", "annotate", ["annotate"]),
            ):
                with self.subTest(start=start, end=end):
                    args = pipeline.parse_args([
                        "--from-stage", start, "--to-stage", end,
                        "--source", str(root / "source.jsonl"),
                        "--extracted", str(root / "extracted.jsonl"),
                        "--pairs", str(root / "pairs.jsonl"),
                        "--output", str(root / "annotated.jsonl"), "--overwrite",
                    ])
                    commands = pipeline.build_commands(args)
                    self.assertEqual([stage for stage, cmd in commands], expected)
                    for stage, cmd in commands:
                        self.assertEqual(cmd[0], sys.executable)
                        self.assertEqual(Path(cmd[1]).name, stage)
                    if start == "annotate":
                        self.assertIn(str(root / "pairs.jsonl"), commands[0][1])

    def test_invalid_ranges_missing_input_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.jsonl"
            source.write_text("{}\n", encoding="utf-8")
            for argv in (
                ["--from-stage", "annotate", "--to-stage", "extract"],
                ["--from-stage", "replace", "--extracted", str(source / "missing")],
                ["--source", str(source), "--extracted", str(source)],
                ["--source", str(source), "--extracted", str(source.parent / "old.jsonl")],
            ):
                (source.parent / "old.jsonl").write_text("old", encoding="utf-8")
                with self.subTest(argv=argv), self.assertRaises(ValueError):
                    pipeline.build_commands(pipeline.parse_args(argv))
            self.assertEqual(source.read_text(encoding="utf-8"), "{}\n")

    def test_json_passthrough_and_engine_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "pairs.jsonl"
            pairs.touch()
            args = pipeline.parse_args([
                "--from-stage", "annotate", "--pairs", str(pairs),
                "--output", str(Path(directory) / "out.jsonl"),
                "--annotation-engine", "all-in-one",
                "--annotate-args", json.dumps(["--model", "model with spaces", "--limit", "3"]),
            ])
            cmd = pipeline.build_commands(args)[0][1]
            self.assertEqual(cmd[-4:], ["--model", "model with spaces", "--limit", "3"])
            self.assertIn("all-in-one", cmd)
            for extra in ('"not a list"', '[1]', '["--corpus=other.jsonl"]', '["--out-file", "other"]'):
                args.annotate_args = extra
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    pipeline.build_commands(args)

    def test_dry_run_has_no_filesystem_or_child_process_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jsonl"
            source.touch()
            output_dir = root / "not created"
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                code = pipeline.main([
                    "--source", str(source), "--extracted", str(output_dir / "extracted.jsonl"),
                    "--pairs", str(output_dir / "pairs.jsonl"), "--output", str(output_dir / "out.jsonl"),
                    "--dry-run",
                ])
            self.assertEqual(code, 0)
            self.assertIn("extract", stdout.getvalue())
            self.assertFalse(output_dir.exists())

    def test_child_failure_stops_following_stage_and_preserves_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            commands = [
                ("replace", [sys.executable, "-c", "raise SystemExit(7)"]),
                ("annotate", [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.execute_commands(commands), 7)
            self.assertFalse(marker.exists())

    def test_abbreviated_managed_options_cannot_override_preflight(self):
        for stage, tokens in (
            ("replace", ["--out-f", "unplanned.jsonl", "--overw"]),
            ("replace", ["--out-f=unplanned.jsonl"]),
            ("extract", ["--out_n", "unplanned"]),
            ("extract", ["--filter_config_p", "unplanned.json5"]),
            ("annotate", ["--corp", "unplanned.jsonl"]),
            ("annotate", ["--batch-c", "unplanned.jsonl"]),
            ("annotate", ["--no-pairs-c"]),
        ):
            with self.subTest(stage=stage, tokens=tokens), self.assertRaises(ValueError):
                pipeline._extra_args(json.dumps(tokens), stage)

    def test_all_annotation_write_targets_participate_in_cross_stage_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "out.jsonl"
            cases = (
                (root / "source.jsonl", ["--batch-cache", str(root / "source.jsonl")]),
                (root / "cache.jsonl.jobs.json", ["--batch-cache", str(root / "cache.jsonl")]),
                (root / "out.jsonl.batch-cache.jsonl", []),
                (root / "out.models" / "00_model-a.jsonl", ["--models", "model-a", "model-b"]),
            )
            for source, extra in cases:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("original data\n", encoding="utf-8")
                args = pipeline.parse_args([
                    "--source", str(source), "--extracted", str(root / "extracted.jsonl"),
                    "--pairs", str(root / "pairs.jsonl"), "--output", str(output),
                    "--annotate-args", json.dumps(extra), "--overwrite",
                ])
                with self.subTest(source=source), self.assertRaises(ValueError):
                    pipeline.build_commands(args)
                self.assertEqual(source.read_text(encoding="utf-8"), "original data\n")

    def test_uppercase_jsonl_is_rejected_before_extraction_renames_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.touch()
            args = pipeline.parse_args([
                "--source", str(source), "--to-stage", "extract",
                "--extracted", str(root / "extracted.JSONL"),
            ])
            with self.assertRaises(ValueError):
                pipeline.build_commands(args)


if __name__ == "__main__":
    unittest.main()
