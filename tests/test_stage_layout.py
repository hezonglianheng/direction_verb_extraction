import importlib
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StageLayoutTests(unittest.TestCase):
    def test_directories_are_executable(self):
        for stage in ("extract", "replace", "annotate"):
            with self.subTest(stage=stage):
                self.assertTrue((ROOT / stage / "__main__.py").is_file(), stage)

    def test_directory_cli_help_and_engine_dispatch(self):
        for argv, marker in (
            (["extract", "--help"], "--filter_config_path"),
            (["replace", "--help"], "--alter-rules"),
            (["annotate", "--help"], "--list-graphs"),
            (["annotate", "--engine", "all-in-one", "--help"], "--print-prompt"),
        ):
            with self.subTest(argv=argv):
                result = subprocess.run([sys.executable, *argv], cwd=ROOT, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=180)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(marker, result.stdout)

    def test_new_and_legacy_imports_share_state(self):
        for old, new in (
            ("sentence_split", "extract.sentence_split"),
            ("sentence_screening", "extract.sentence_screening"),
            ("sentence_replace", "replace.sentence_replace"),
            ("spatial_runner", "annotate.spatial_runner"),
            ("spatial_tools", "annotate.spatial_tools"),
            ("all_in_one", "annotate.all_in_one"),
        ):
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_resources_still_resolve_to_project_root(self):
        from annotate import spatial_tools, spatial_runner, all_in_one, prompt_examples
        from extract import word_sampler, sentence_screening
        for module in (spatial_tools, spatial_runner, all_in_one, word_sampler):
            self.assertEqual(module._PROJECT_ROOT, ROOT)
        self.assertEqual(prompt_examples.EXAMPLES_DIR, ROOT / "system_prompt" / "examples")
        self.assertTrue(sentence_screening.spatial_criteria_path("方位词").is_file())

    def test_spawn_can_import_moved_worker(self):
        # Spawn uses a fresh interpreter, as on Windows and both extraction entries.
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            from extract.sentence_split import split_sentences
            self.assertEqual(pool.apply(split_sentences, ("甲。乙。",), {"lower_bound": 1}), ["甲。", "乙。"])


if __name__ == "__main__":
    unittest.main()
