"""不加载 LTP 或调用 API，验证真实入口在写盘前保护输入文件。"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from file_paths import collect_jsonl_inputs, validate_output_paths


ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def read_jsonlines(path):
    with open(path, encoding="utf-8") as source:
        yield (json.loads(line) for line in source if line.strip())


def load_entry(filename):
    """仅替换外部推理依赖，路径检查、输入读取和输出打开执行真实代码。"""
    ltp = ModuleType("ltp")
    ltp.StnSplit = Mock()
    jsonlines = ModuleType("jsonlines")
    jsonlines.open = read_jsonlines
    tqdm = ModuleType("tqdm")
    tqdm.tqdm = lambda items, **kwargs: items
    st = ModuleType("spatial_tools")
    st.read_jsonl = Mock(return_value=[])
    st.write_pairs_csv = Mock(return_value=(0, 0))
    replacements = {
        "ltp": ltp,
        "jsonlines": jsonlines,
        "tqdm": tqdm,
        "sentenceFilters": ModuleType("sentenceFilters"),
        "spatial_tools": st,
    }
    spec = importlib.util.spec_from_file_location(f"_test_{Path(filename).stem}", ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class OutputPathTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="corpus-path-tests-")))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.source = self.root / "source.jsonl"
        self.original = '{"content": "原始语料必须保留"}\n'
        self.source.write_text(self.original, encoding="utf-8")
        self.linux = load_entry("entry_in_linux.py")
        self.windows = load_entry("entry_in_win.py")
        self.export = load_entry("export_pairs_csv.py")
        for module in (self.linux, self.windows):
            module.get_sentences_list = lambda article, vocabs=None: [article]

        def write_rows(sentences, configs, outputs):
            for sentence in sentences:
                for output in outputs:
                    output.write(json.dumps(sentence, ensure_ascii=False) + "\n")

        self.linux.filter_sentences_streaming = Mock(side_effect=write_rows)
        self.windows.filter_sentences_streaming = Mock(side_effect=write_rows)

    def run_filter(self, platform, source, outputs):
        configs = [f"rules_{index}.json5" for index in range(len(outputs))]
        if platform == "linux":
            self.linux.run(str(source), configs, outputs)
        else:
            self.windows.run_windows_entry(str(source), configs, outputs)

    def assert_source_preserved(self):
        self.assertEqual(self.source.read_text(encoding="utf-8"), self.original)

    def test_same_input_output_is_rejected_by_both_entries(self):
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform), self.assertRaisesRegex(ValueError, "同一文件"):
                self.run_filter(platform, self.source, [self.source])
            self.assert_source_preserved()
        self.linux.filter_sentences_streaming.assert_not_called()
        self.windows.filter_sentences_streaming.assert_not_called()

    def test_relative_and_absolute_aliases_are_rejected(self):
        with contextlib.chdir(self.root):
            for platform in ("linux", "windows"):
                with self.subTest(platform=platform), self.assertRaises(ValueError):
                    self.run_filter(platform, Path("source.jsonl"), [self.source.resolve()])
        self.assert_source_preserved()

    def test_hardlink_alias_is_rejected_by_entries_and_export(self):
        alias = self.root / "alias.csv"
        try:
            alias.hardlink_to(self.source)
        except OSError as exc:
            self.skipTest(f"当前文件系统不支持硬链接：{exc}")
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform), self.assertRaises(ValueError):
                self.run_filter(platform, self.source, [alias])
        with self.assertRaises(SystemExit):
            self.export.main([str(self.source), "-o", str(alias)])
        self.assert_source_preserved()

    def test_symlink_alias_is_rejected(self):
        alias = self.root / "alias.jsonl"
        try:
            alias.symlink_to(self.source)
        except OSError as exc:
            self.skipTest(f"当前用户不能创建符号链接：{exc}")
        with self.assertRaises(ValueError):
            validate_output_paths([self.source], [alias])
        self.assert_source_preserved()

    def test_later_collision_does_not_truncate_earlier_output(self):
        earlier = self.root / "existing-output.jsonl"
        earlier.write_text("已有结果", encoding="utf-8")
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform), self.assertRaises(ValueError):
                self.run_filter(platform, self.source, [earlier, self.source])
            self.assertEqual(earlier.read_text(encoding="utf-8"), "已有结果")
        self.assert_source_preserved()

    def test_duplicate_outputs_are_rejected_before_writing(self):
        output = self.root / "output.jsonl"
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform), self.assertRaises(ValueError):
                self.run_filter(platform, self.source, [output, output])
            self.assertFalse(output.exists())

    def test_directory_collision_is_rejected_without_excluding_raw_input(self):
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform), self.assertRaises(ValueError):
                self.run_filter(platform, self.root, [self.source])
            self.assert_source_preserved()

    def test_new_output_in_input_directory_is_not_read_as_input(self):
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform):
                directory = self.root / platform
                directory.mkdir()
                source = directory / "input.jsonl"
                source.write_text(self.original, encoding="utf-8")
                output = directory / "filtered.jsonl"
                self.run_filter(platform, directory, [output])
                rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["sentence"], "原始语料必须保留")
                self.assertEqual(source.read_text(encoding="utf-8"), self.original)
                # 再次运行时无法把已有 JSONL 安全认定为可覆盖产物，必须拒绝。
                previous = output.read_bytes()
                with self.assertRaises(ValueError):
                    self.run_filter(platform, directory, [output])
                self.assertEqual(output.read_bytes(), previous)

    def test_linux_directory_snapshot_is_taken_before_iteration(self):
        sentences = self.linux.iter_path_sentences(str(self.root))
        (self.root / "late-output.jsonl").write_text('{"sentence": "不是原始文章"}\n', encoding="utf-8")
        self.assertEqual(len(list(sentences)), 1)

    def test_missing_input_does_not_truncate_output(self):
        output = self.root / "output.jsonl"
        output.write_text("已有结果", encoding="utf-8")
        for platform in ("linux", "windows"):
            with self.subTest(platform=platform), self.assertRaises(ValueError):
                self.run_filter(platform, self.root / "missing.jsonl", [output])
            self.assertEqual(output.read_text(encoding="utf-8"), "已有结果")

    def test_csv_export_rejects_same_path_before_reading_or_writing(self):
        with self.assertRaisesRegex(SystemExit, "同一文件"):
            self.export.main([str(self.source), "-o", str(self.source)])
        self.export.st.read_jsonl.assert_not_called()
        self.export.st.write_pairs_csv.assert_not_called()
        self.assert_source_preserved()

    def test_csv_default_output_remains_available(self):
        self.assertEqual(self.export.main([str(self.source)]), 0)
        self.export.st.write_pairs_csv.assert_called_once_with([], self.source.with_suffix(".csv"))
        self.assert_source_preserved()

    def test_nonconflicting_outputs_in_another_directory_are_allowed(self):
        output = self.root / "results" / "output.jsonl"
        self.assertEqual(collect_jsonl_inputs(self.root, [output]), [self.source])


if __name__ == "__main__":
    unittest.main()
