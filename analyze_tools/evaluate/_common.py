# encoding: utf8
"""evaluate 分析工具的公共部分：一致率、CSV、Markdown 与输出目录发布。

这里只依赖标准库；不读取模型配置，不导入 ``evaluate`` / ``annotate`` 的推理模块。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
import csv
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYZE_RESULTS_DIR = ROOT_DIR / "analyze_results"
SUMMARY_FILENAME = "summary.md"
DEFAULT_MAX_FILES = 20

BINARY_LABELS = ("same", "different")
UNRESOLVED_LABEL = "（无结论）"

DETAIL_COLUMNS = [
    "比较结果",
    "对齐键",
    "变化字段",
    "原句",
    "替换句",
    "关注词a",
    "关注词b",
    "左记录类型",
    "右记录类型",
    "左模型/层",
    "右模型/层",
    "左最终标签",
    "右最终标签",
    "左同义字段",
    "右同义字段",
    "左参考标签",
    "右参考标签",
    "左状态",
    "右状态",
    "左投票状态",
    "右投票状态",
    "左步骤",
    "右步骤",
    "左标注图",
    "右标注图",
    "左行号",
    "右行号",
    "左物理行号",
    "右物理行号",
    "左来源",
    "右来源",
    "左理由",
    "右理由",
    "左失败原因",
    "右失败原因",
    "左备注",
    "右备注",
]


def label_text(label: str | None) -> str:
    """标签统一展示；None 表示没有形成结论。"""
    return UNRESOLVED_LABEL if label is None else label


def boolean_text(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def agreement_text(same: int, denominator: int) -> str:
    if not denominator:
        return "N/A (0/0)"
    return f"{same / denominator * 100:.2f}% ({same}/{denominator})"


def _md(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def _table(columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(_md(item) for item in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *[
            "| " + " | ".join(_md(item) for item in row) + " |"
            for row in rows
        ],
    ]


def file_label(path: Path, index: int) -> str:
    name = path.name
    if name.lower().endswith(".details.jsonl"):
        name = name[: -len(".details.jsonl")]
    elif name.lower().endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._-")
    name = name.encode("utf-8")[:70].decode("utf-8", errors="ignore") or "result"
    return f"{index:02d}_{name}"


def safe_filename(value: str, limit: int = 40) -> str:
    name = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._-")
    name = name.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    return name or "model"


def _unique_default_output_dir(comparison_name: str) -> Path:
    base = ANALYZE_RESULTS_DIR / comparison_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = base / stamp
    suffix = 1
    while target.exists() or target.is_symlink():
        target = base / f"{stamp}_{suffix:02d}"
        suffix += 1
    return target


def plan_output_dir(value: str | None, comparison_name: str) -> Path:
    target = Path(value) if value else _unique_default_output_dir(comparison_name)
    if target.exists() or target.is_symlink():
        raise ValueError(f"输出目录已存在：{target}；请换一个新目录或删除旧目录")
    return target


def publish_output_dir(staging: Path, target: Path) -> None:
    """独占创建目标目录；``summary.md`` 最后发布，已有目录绝不覆盖。"""
    os.mkdir(target)
    try:
        for path in sorted(
            staging.iterdir(),
            key=lambda item: (item.name == SUMMARY_FILENAME, item.name),
        ):
            os.replace(path, target / path.name)
        staging.rmdir()
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def make_staging_dir(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))


def _same_file(left: Path, right: Path) -> bool:
    try:
        if left.resolve() == right.resolve():
            return True
        return left.samefile(right)
    except (FileNotFoundError, OSError, RuntimeError):
        return False


def input_paths(
    inputs: Sequence[str],
    max_files: int,
    *,
    minimum: int,
    kind: str,
) -> list[Path]:
    """校验输入 JSONL 列表；拒绝目录、CSV、运行汇总与重复路径。"""
    if len(inputs) < minimum:
        raise ValueError(f"至少需要 {minimum} 份{kind}")
    if max_files < 0:
        raise ValueError("--max-files 不能为负数")
    if max_files and len(inputs) > max_files:
        raise ValueError(
            f"输入文件数为 {len(inputs)}，超过 --max-files {max_files}；"
            "请提高上限或分批比较（0 表示不限制）"
        )
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            raise ValueError(
                f"输入是目录：{path}；请逐个传入具体的 JSONL 文件"
            )
        if not path.is_file():
            raise FileNotFoundError(f"找不到输入文件：{path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            raise ValueError(
                f"不支持 CSV：{path}；CSV 只是拍平视图，请改用逐条结果 JSONL"
            )
        if path.name.endswith(".summary.json"):
            raise ValueError(
                f"这是运行汇总，不是逐条结果：{path}；请传入逐条结果 JSONL"
            )
        if suffix != ".jsonl":
            raise ValueError(f"只支持 JSONL：{path}")
        for other in paths:
            if _same_file(path, other):
                raise ValueError(f"存在重复输入文件：{other} 与 {path}")
        paths.append(path)
    return paths


@lru_cache(maxsize=4096)
def normalize_source_path(value: str) -> str:
    """把 details 里的 input_file 规范成可比较的绝对路径。"""
    try:
        path = Path(value).resolve()
    except OSError:
        path = Path(os.path.normpath(value))
    return os.path.normcase(str(path))


@dataclass(frozen=True)
class SideView:
    """CSV/Markdown 里一侧记录的展示投影。"""

    layer: str
    kind: str
    label: str | None
    same_meaning: bool | None = None
    reference: str | None = None
    status: str = ""
    voting_status: str = ""
    step: str = ""
    mode: str = ""
    line_no: str = ""
    input_line: str = ""
    source_file: str = ""
    sentence_a: str = ""
    sentence_b: str = ""
    focus_a: str = ""
    focus_b: str = ""
    reason: str = ""
    error: str = ""
    note: str = ""


def _side_cells(prefix: str, view: SideView) -> dict[str, str]:
    return {
        f"{prefix}记录类型": view.kind,
        f"{prefix}模型/层": view.layer,
        f"{prefix}最终标签": label_text(view.label),
        f"{prefix}同义字段": boolean_text(view.same_meaning),
        f"{prefix}参考标签": view.reference or "",
        f"{prefix}状态": view.status,
        f"{prefix}投票状态": view.voting_status,
        f"{prefix}步骤": view.step,
        f"{prefix}标注图": view.mode,
        f"{prefix}行号": view.line_no,
        f"{prefix}物理行号": view.input_line,
        f"{prefix}来源": view.source_file,
        f"{prefix}理由": view.reason,
        f"{prefix}失败原因": view.error,
        f"{prefix}备注": view.note,
    }


def _context(view: SideView) -> tuple[str, str, str, str]:
    return (view.sentence_a, view.sentence_b, view.focus_a, view.focus_b)


def _changes(left: SideView, right: SideView) -> dict[str, bool]:
    return {
        "最终标签": left.label != right.label,
        "同义字段": left.same_meaning != right.same_meaning,
        "参考标签": left.reference != right.reference,
        "状态": left.status != right.status,
        "上下文": _context(left) != _context(right),
    }


@dataclass
class Comparison:
    """一对（或一个 details 文件对一个 annotate 层）的统计。"""

    left_label: str
    right_label: str
    left_map: dict[str, SideView]
    right_map: dict[str, SideView]
    right_unavailable: int = 0
    left_ambiguous: int = 0
    identity_conflicts: int = 0
    reference_comparable: int = 0
    reference_same: int = 0
    csv_name: str = ""
    conflict_keys: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        conflicts = self.conflict_keys
        self.common_keys = [
            key
            for key in self.left_map
            if key in self.right_map and key not in conflicts
        ]
        self.only_left_keys = [
            key
            for key in self.left_map
            if key not in self.right_map and key not in conflicts
        ]
        self.only_right_keys = [
            key
            for key in self.right_map
            if key not in self.left_map and key not in conflicts
        ]
        self.identity_conflicts = len(conflicts)
        self.common = len(self.common_keys)
        self.only_left = len(self.only_left_keys)
        self.only_right = len(self.only_right_keys)
        self.union = self.common + self.only_left + self.only_right

        self.same = 0
        self.different = 0
        self.defined_comparable = 0
        self.defined_same = 0
        self.binary_comparable = 0
        self.binary_same = 0
        self.unresolved = 0
        self.left_null = 0
        self.right_null = 0
        self.label_transitions: Counter[tuple[str | None, str | None]] = Counter()

        for key in self.common_keys:
            left = self.left_map[key]
            right = self.right_map[key]
            same_label = left.label == right.label
            self.same += same_label
            self.different += not same_label
            self.label_transitions[(left.label, right.label)] += 1
            if left.label is None:
                self.left_null += 1
            if right.label is None:
                self.right_null += 1
            if left.label is not None and right.label is not None:
                self.defined_comparable += 1
                self.defined_same += same_label
            else:
                self.unresolved += 1
            if left.label in BINARY_LABELS and right.label in BINARY_LABELS:
                self.binary_comparable += 1
                self.binary_same += same_label
            if (
                left.reference is not None
                and right.reference is not None
            ):
                self.reference_comparable += 1
                self.reference_same += left.reference == right.reference

    @property
    def common_agreement_text(self) -> str:
        return agreement_text(self.same, self.common)

    @property
    def defined_agreement_text(self) -> str:
        return agreement_text(self.defined_same, self.defined_comparable)

    @property
    def binary_agreement_text(self) -> str:
        return agreement_text(self.binary_same, self.binary_comparable)

    def release_maps(self) -> None:
        """CSV 写完后释放逐条映射与键列表，只保留计数供 summary 使用。"""
        self.left_map = {}
        self.right_map = {}
        self.common_keys = []
        self.only_left_keys = []
        self.only_right_keys = []


def compare_maps(
    left_map: dict[str, SideView],
    right_map: dict[str, SideView],
    *,
    left_label: str,
    right_label: str,
    check_context: bool = False,
) -> Comparison:
    conflict_keys: set[str] = set()
    if check_context:
        for key in left_map.keys() & right_map.keys():
            if _context(left_map[key]) != _context(right_map[key]):
                conflict_keys.add(key)
    return Comparison(
        left_label=left_label,
        right_label=right_label,
        left_map=left_map,
        right_map=right_map,
        conflict_keys=conflict_keys,
    )


def _detail_row(
    key: str,
    left: SideView | None,
    right: SideView | None,
) -> dict[str, str]:
    source = left or right
    assert source is not None
    result: dict[str, str] = {
        "对齐键": key,
        "变化字段": "",
        "原句": source.sentence_a,
        "替换句": source.sentence_b,
        "关注词a": source.focus_a,
        "关注词b": source.focus_b,
    }
    if left is not None and right is not None:
        result["比较结果"] = "相同" if left.label == right.label else "不同"
        flags = _changes(left, right)
        result["变化字段"] = ", ".join(
            name for name, changed in flags.items() if changed
        )
    else:
        result["比较结果"] = "仅左侧" if left is not None else "仅右侧"
    if left is not None:
        result.update(_side_cells("左", left))
    if right is not None:
        result.update(_side_cells("右", right))
    return result


def write_pair_csv(comparison: Comparison, path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for key in comparison.common_keys:
            writer.writerow(
                _detail_row(
                    key,
                    comparison.left_map[key],
                    comparison.right_map[key],
                )
            )
        for key in comparison.only_left_keys:
            writer.writerow(_detail_row(key, comparison.left_map[key], None))
        for key in comparison.only_right_keys:
            writer.writerow(_detail_row(key, None, comparison.right_map[key]))


def transition_rows(
    counter: Counter[tuple[Any, Any]],
) -> list[list[Any]]:
    return [
        [label_text(left), label_text(right), count]
        for (left, right), count in sorted(
            counter.items(), key=lambda item: (-item[1], str(item[0]))
        )
    ]


def distribution(counter: Counter[str]) -> str:
    if not counter:
        return "—"
    return "，".join(
        f"{name}={count}"
        for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


def format_matrix_lines(
    labels: Sequence[str],
    lookup: dict[tuple[int, int], Comparison],
) -> list[str]:
    lines = [
        "| " + " | ".join(_md(cell) for cell in ["文件"] + list(labels)) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(labels)) + " |",
    ]
    for left_index, left_label in enumerate(labels):
        cells = [_md(left_label)]
        for right_index, _ in enumerate(labels):
            if left_index == right_index:
                cells.append("—")
                continue
            comparison = lookup.get(
                (min(left_index, right_index), max(left_index, right_index))
            )
            cells.append(
                comparison.common_agreement_text if comparison else "N/A (0/0)"
            )
        lines.append("| " + " | ".join(_md(cell) for cell in cells) + " |")
    return lines


def comparison_summary_rows(comparisons: Sequence[Comparison]) -> list[list[Any]]:
    return [
        [
            index,
            comparison.left_label,
            comparison.right_label,
            comparison.common,
            comparison.union,
            comparison.common_agreement_text,
            comparison.defined_agreement_text,
            comparison.binary_agreement_text,
            comparison.only_left,
            comparison.only_right,
            comparison.right_unavailable,
            comparison.left_ambiguous,
            comparison.identity_conflicts,
            comparison.csv_name,
        ]
        for index, comparison in enumerate(comparisons, 1)
    ]


COMPARISON_COLUMNS = [
    "#",
    "左",
    "右",
    "共同",
    "并集",
    "结论一致率（含未判定）",
    "有效结论一致率",
    "同义二分类一致率",
    "仅左",
    "仅右",
    "右侧缺票/不适用",
    "左侧身份不唯一",
    "身份内容冲突",
    "明细 CSV",
]
