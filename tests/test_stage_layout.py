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

    def test_packages_load_without_any_legacy_module_names(self):
        modules = [f"{stage}.{path.stem}" for stage in ("extract", "replace", "annotate")
                   for path in sorted((ROOT / stage).glob("*.py"))
                   if path.stem not in ("__init__", "__main__", "temp")]
        old_names = {name.split(".")[1] for name in modules} | {"temp", "legacy_entry"}
        script = f"""
import importlib
import importlib.abc
import sys
blocked = {old_names!r}
class NoLegacyModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise ModuleNotFoundError('Legacy import forbidden: ' + fullname)
sys.meta_path.insert(0, NoLegacyModules())
for name in {modules!r}:
    importlib.import_module(name)
assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
print('PACKAGE IMPORTS OK')
"""
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=240)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PACKAGE IMPORTS OK", result.stdout)

    def test_legacy_forwarding_files_are_removed(self):
        names = {path.name for stage in ("extract", "replace", "annotate")
                 for path in (ROOT / stage).glob("*.py") if not path.name.startswith("__")}
        for name in names | {"legacy_entry.py"}:
            with self.subTest(name=name):
                self.assertFalse((ROOT / name).exists(), "旧入口应被删除：" + name)

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
