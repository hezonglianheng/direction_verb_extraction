# encoding: utf8
"""读取 evaluate details 与 annotate 投票结果，并建立稳定的逐条身份。

对齐口径：

- evaluate details 没有写 ``candidate_key``，但写了 ``line_no`` 与 ``candidate``。
  两份 details 之间用 ``(规范化 input_file, 身份)`` 对齐；身份优先用
  ``(line_no, candidate.index, candidate.focus_index, replacement)``
  （与 ``annotate.voting.record_identity`` 同口径），没有 candidate 时退回
  ``id``。
- details 对 annotate 结果时用同一个身份 join。annotate 侧身份由
  ``line_no + candidate`` 重算，不依赖 ``id``；只有双方身份相同且句对内容
  一致才算对齐。
- 不用 JSONL 行号、记录出现顺序或单独 ``id`` 兜底跨语料对齐。

本模块只读文件；annotate 侧的正式记录校验复用
``analyze_tools.annotate._records.record_row``，不导入 LTP/torch。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

if __package__:
    from . import _common as common
    from ..annotate import _records as annotate_records
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _common as common  # type: ignore

    _ANNOTATE_DIR = Path(__file__).resolve().parents[1] / "annotate"
    if str(_ANNOTATE_DIR) not in sys.path:
        sys.path.insert(0, str(_ANNOTATE_DIR))
    import _records as annotate_records  # type: ignore


def _fail(source: Path, line_no: int, message: str):
    raise ValueError(f"{source} 第 {line_no} 行：{message}")


def _text(
    record: dict[str, Any],
    field: str,
    source: Path,
    line_no: int,
    *,
    required: bool = False,
    allow_empty: bool = True,
) -> str:
    value = record.get(field)
    if value is None:
        if required:
            _fail(source, line_no, f"缺少 {field}")
        return ""
    if not isinstance(value, str):
        _fail(source, line_no, f"{field} 必须是字符串，实际是 {type(value).__name__}")
    if not allow_empty and not value.strip():
        _fail(source, line_no, f"{field} 不能为空")
    return value


def _optional_text(
    record: dict[str, Any], field: str, source: Path, line_no: int
) -> str:
    value = record.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        _fail(source, line_no, f"{field} 必须是字符串或 null")
    return value


def _integer(
    record: dict[str, Any], field: str, source: Path, line_no: int
) -> int | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(source, line_no, f"{field} 必须是整数或 null")
    return value


def _strict_integer(
    record: dict[str, Any], field: str, source: Path, line_no: int
) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(source, line_no, f"{field} 必须是整数")
    return value


_MISSING = object()


def _boolean(
    record: dict[str, Any], field: str, source: Path, line_no: int
) -> bool | None:
    value = record.get(field, _MISSING)
    if value is _MISSING:
        return None
    if value is None or type(value) is bool:
        return value
    _fail(source, line_no, f"{field} 只接受 true/false/null，实际是 {value!r}")


def _words(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        word = value.get("word")
        return word if isinstance(word, str) else ""
    return ""


def inner_key(record: dict[str, Any], source: Path, line_no: int) -> str | None:
    """一条记录在 details / annotate 之间的稳定身份。

    有 candidate 时使用与投票一致的四元组；缺任何一项都报错，不静默填默认值。
    没有 candidate 时退回 ``id``（外部导入的句对只有 id）。
    """
    candidate = record.get("candidate")
    if candidate is not None:
        if not isinstance(candidate, dict):
            _fail(source, line_no, "candidate 必须是 JSON 对象或 null")
        record_line = record.get("line_no")
        if isinstance(record_line, bool) or not isinstance(record_line, int):
            _fail(source, line_no, "有 candidate 但 line_no 不是整数，无法建立候选身份")
        index = candidate.get("index")
        focus_index = candidate.get("focus_index")
        replacement = candidate.get("replacement")
        if isinstance(index, bool) or not isinstance(index, int):
            _fail(source, line_no, "candidate.index 必须是整数")
        if isinstance(focus_index, bool) or not isinstance(focus_index, int):
            _fail(source, line_no, "candidate.focus_index 必须是整数")
        if not isinstance(replacement, str):
            _fail(source, line_no, "candidate.replacement 必须是字符串")
        return f"cand|{record_line}|{index}|{focus_index}|{replacement}"
    value = record.get("id")
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(source, line_no, "id 必须是字符串或 null")
    return f"id|{value}" if value.strip() else None


@dataclass(frozen=True)
class DetailRow:
    input_line: int
    input_file: str
    input_line_no: int
    record_id: str
    line_no: int | None
    key: str | None
    inner_key: str | None
    sentence_a: str
    sentence_b: str
    focus_a: str
    focus_b: str
    reference: str
    reference_source: str
    reference_model: str
    annotation_status: str
    voting_status: str
    prediction: str | None
    same_meaning: bool | None
    source: str
    status: str
    reason: str
    error: str
    error_kind: str
    correct: bool | None
    step: int
    mode: str
    allow_lookup: bool
    corpus_type: str

    def side(self, layer: str) -> common.SideView:
        return common.SideView(
            layer=layer,
            kind="evaluate detail",
            label=self.prediction,
            same_meaning=self.same_meaning,
            reference=self.reference,
            status=self.status,
            voting_status=self.voting_status,
            step=str(self.step),
            mode=self.mode,
            line_no="" if self.line_no is None else str(self.line_no),
            input_line=str(self.input_line),
            source_file=self.input_file,
            sentence_a=self.sentence_a,
            sentence_b=self.sentence_b,
            focus_a=self.focus_a,
            focus_b=self.focus_b,
            reason=self.reason,
            error=self.error or self.error_kind,
            note=self.annotation_status,
        )


@dataclass
class LoadedDetails:
    index: int
    path: Path
    label: str
    rows: list[DetailRow]
    rows_by_key: dict[str, common.SideView]
    inner_counts: Counter[str]
    step: int
    mode: str
    allow_lookup: bool
    corpus_type: str
    prediction_counts: Counter[str]
    status_counts: Counter[str]
    error_counts: Counter[str]
    reference_counts: Counter[str]
    source_files: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def keyed(self) -> int:
        return sum(1 for row in self.rows if row.key is not None)

    @property
    def unkeyed(self) -> int:
        return self.total - self.keyed

    @property
    def route_text(self) -> str:
        return f"step={self.step}；mode={self.mode}；corpus_type={self.corpus_type}"


def _reject_non_details(record: dict[str, Any], source: Path, line_no: int) -> None:
    if "annotation_mode" in record or "candidate_key" in record or "voting" in record:
        _fail(
            source,
            line_no,
            "这是 annotate 结果，不是 evaluate details；"
            "annotate 结果请用 --annotate 传入",
        )
    if "pair_schema_version" in record:
        _fail(source, line_no, "这是 replace 句对结果，不是 evaluate details")
    if "screening" in record:
        _fail(source, line_no, "这是 extract 原句结果，不是 evaluate details")
    if "prediction" not in record:
        _fail(
            source,
            line_no,
            "缺少 prediction；只接受 python -m evaluate 写出的 .details.jsonl",
        )


def _route(
    record: dict[str, Any], source: Path, line_no: int
) -> tuple[int, str, bool, str]:
    step = _strict_integer(record, "step", source, line_no)
    mode = _text(record, "mode", source, line_no, required=True, allow_empty=False)
    allow_lookup = _boolean(record, "allow_lookup", source, line_no)
    if allow_lookup is None:
        _fail(source, line_no, "allow_lookup 必须是 true/false")
    corpus_type = _text(
        record, "corpus_type", source, line_no, required=True, allow_empty=False
    )
    return step, mode, allow_lookup, corpus_type


def detail_row(
    record: dict[str, Any], *, source: Path, line_no: int
) -> DetailRow:
    _reject_non_details(record, source, line_no)
    step, mode, allow_lookup, corpus_type = _route(record, source, line_no)

    prediction = record.get("prediction")
    if prediction is not None and prediction not in common.BINARY_LABELS:
        _fail(
            source,
            line_no,
            f"prediction 只能是 same/different/null，实际是 {prediction!r}",
        )

    reference = record.get("reference")
    if reference not in common.BINARY_LABELS:
        _fail(
            source,
            line_no,
            f"reference 只能是 same/different，实际是 {reference!r}；"
            "evaluate 只评测二分类参考标签",
        )

    same_meaning = _boolean(record, "same_meaning", source, line_no)
    correct = _boolean(record, "correct", source, line_no)

    status = _text(record, "status", source, line_no, required=True, allow_empty=False)
    if prediction is None and status != "error":
        _fail(
            source,
            line_no,
            f"prediction 为空时 status 应为 error，实际是 {status!r}",
        )
    if prediction is not None and status != "ok":
        _fail(
            source,
            line_no,
            f"prediction 非空时 status 应为 ok，实际是 {status!r}",
        )

    sentence_a = _text(
        record, "sentence_a", source, line_no, required=True, allow_empty=False
    )
    sentence_b = _text(
        record, "sentence_b", source, line_no, required=True, allow_empty=False
    )
    input_file = _text(
        record, "input_file", source, line_no, required=True, allow_empty=False
    )
    input_line_no = _strict_integer(record, "input_line_no", source, line_no)
    record_id = _optional_text(record, "id", source, line_no)
    record_line_no = _integer(record, "line_no", source, line_no)
    identity = inner_key(record, source, line_no)
    key = (
        f"{common.normalize_source_path(input_file)}\x1f{identity}"
        if identity is not None
        else None
    )

    return DetailRow(
        input_line=line_no,
        input_file=input_file,
        input_line_no=input_line_no,
        record_id=record_id,
        line_no=record_line_no,
        key=key,
        inner_key=identity,
        sentence_a=sentence_a.strip(),
        sentence_b=sentence_b.strip(),
        focus_a=_words(record.get("focus_word")),
        focus_b=_words(record.get("focus_word_b")),
        reference=reference,
        reference_source=_optional_text(record, "reference_source", source, line_no),
        reference_model=_optional_text(record, "reference_model", source, line_no),
        annotation_status=_optional_text(
            record, "annotation_status", source, line_no
        ),
        voting_status=_optional_text(record, "voting_status", source, line_no),
        prediction=prediction,
        same_meaning=same_meaning,
        source=_optional_text(record, "source", source, line_no),
        status=status,
        reason=_optional_text(record, "reason", source, line_no),
        error=_optional_text(record, "error", source, line_no),
        error_kind=_optional_text(record, "error_kind", source, line_no),
        correct=correct,
        step=step,
        mode=mode,
        allow_lookup=allow_lookup,
        corpus_type=corpus_type,
    )


def _rerun_placeholder(
    record: dict[str, Any], source: Path, line_no: int
) -> DetailRow:
    """evaluate.rerun 对原始损坏行的补齐记录：没有 route/句对，只能计为无身份错误行。"""
    prediction = record.get("prediction")
    if prediction is not None:
        _fail(
            source,
            line_no,
            "rerun 补齐记录缺少 route/句对，但 prediction 非空；无法安全比较",
        )
    input_file = record.get("input_file")
    input_line_no = record.get("input_line_no")
    return DetailRow(
        input_line=line_no,
        input_file=input_file if isinstance(input_file, str) else str(source),
        input_line_no=(
            input_line_no
            if isinstance(input_line_no, int) and not isinstance(input_line_no, bool)
            else line_no
        ),
        record_id="",
        line_no=None,
        key=None,
        inner_key=None,
        sentence_a="",
        sentence_b="",
        focus_a="",
        focus_b="",
        reference="",
        reference_source="",
        reference_model="",
        annotation_status="",
        voting_status="",
        prediction=None,
        same_meaning=None,
        source="",
        status=str(record.get("status") or "error"),
        reason=_optional_text(record, "reason", source, line_no),
        error=_optional_text(record, "error", source, line_no),
        error_kind=_optional_text(record, "error_kind", source, line_no),
        correct=None,
        step=0,
        mode="",
        allow_lookup=False,
        corpus_type="",
    )


def _is_rerun_placeholder(record: dict[str, Any]) -> bool:
    """严格识别 rerun 对原始损坏行的补齐记录，避免绕过正常输入校验。

    生产形状（``evaluate.rerun._merged_record`` 的 ``row.record is None`` 分支）：
    只有 input_file / input_line_no / id=None / prediction=None / same_meaning=None /
    status="error" / reason / error / error_kind / rerun，没有 route、句对与 candidate。
    """
    if not isinstance(record.get("rerun"), dict):
        return False
    if any(
        key in record
        for key in (
            "annotation_mode",
            "candidate",
            "candidate_key",
            "pair_schema_version",
            "screening",
            "voting",
            "step",
            "mode",
            "allow_lookup",
            "corpus_type",
            "reference",
            "reference_source",
            "reference_model",
            "annotation_status",
            "voting_status",
            "sentence_a",
            "sentence_b",
            "focus_word",
            "focus_word_b",
            "line_no",
        )
    ):
        return False
    if "prediction" not in record or record["prediction"] is not None:
        return False
    if record.get("status") != "error":
        return False
    if record.get("same_meaning") is not None:
        return False
    if record.get("id") is not None:
        return False
    if not isinstance(record.get("input_file"), str) or not record["input_file"].strip():
        return False
    input_line_no = record.get("input_line_no")
    if isinstance(input_line_no, bool) or not isinstance(input_line_no, int):
        return False
    if "error_kind" not in record:
        return False
    return True


def load_details(
    path: str | Path,
    *,
    index: int,
    label: str,
    progress_every: int = 50_000,
    progress_stream=None,
) -> LoadedDetails:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    if source.suffix.lower() != ".jsonl":
        raise ValueError(f"只支持 evaluate details JSONL：{source}")

    rows: list[DetailRow] = []
    rows_by_key: dict[str, common.SideView] = {}
    seen: dict[str, int] = {}
    inner_counts: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    reference_counts: Counter[str] = Counter()
    source_files: set[str] = set()
    route: tuple[int, str, bool, str] | None = None
    route_line = 0

    with source.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source} 第 {line_no} 行不是合法 JSON：{exc}"
                ) from exc
            if not isinstance(record, dict):
                _fail(source, line_no, "记录必须是 JSON 对象")

            if "rerun" in record and "step" not in record:
                if not _is_rerun_placeholder(record):
                    _fail(
                        source,
                        line_no,
                        "含有 rerun 但不符合 evaluate.rerun 损坏行补齐记录格式"
                        "（缺少 route/句对，或 prediction/status 不符）；无法比较",
                    )
                row = _rerun_placeholder(record, source, line_no)
                rows.append(row)
                prediction_counts[common.label_text(row.prediction)] += 1
                status_counts[row.status] += 1
                error_counts[row.error_kind or "unknown"] += 1
                source_files.add(common.normalize_source_path(row.input_file))
                continue

            row = detail_row(record, source=source, line_no=line_no)
            current_route = (row.step, row.mode, row.allow_lookup, row.corpus_type)
            if route is None:
                route = current_route
                route_line = line_no
            elif current_route != route:
                raise ValueError(
                    f"{source} 第 {line_no} 行的 route（step/mode/allow_lookup/"
                    f"corpus_type）与第 {route_line} 行不一致；"
                    "一次 evaluate 跑批只会写一套 route"
                )

            if row.key is not None:
                if row.key in seen:
                    raise ValueError(
                        f"{source} 第 {line_no} 行身份重复：{row.inner_key!r}"
                        f"（首次位于第 {seen[row.key]} 行）；不能按出现顺序配对"
                    )
                seen[row.key] = line_no
            if row.inner_key is not None:
                inner_counts[row.inner_key] += 1

            rows.append(row)
            if row.key is not None:
                rows_by_key[row.key] = row.side(label)
            prediction_counts[common.label_text(row.prediction)] += 1
            status_counts[row.status] += 1
            if row.prediction is None:
                error_counts[row.error_kind or "unknown"] += 1
            reference_counts[row.reference] += 1
            source_files.add(common.normalize_source_path(row.input_file))

            if progress_every and len(rows) % progress_every == 0:
                print(
                    f"{source.name}：已读取 {len(rows)} 条记录",
                    file=progress_stream or sys.stderr,
                    flush=True,
                )

    step, mode, allow_lookup, corpus_type = route or (0, "", False, "")
    return LoadedDetails(
        index=index,
        path=source,
        label=label,
        rows=rows,
        rows_by_key=rows_by_key,
        inner_counts=inner_counts,
        step=step,
        mode=mode,
        allow_lookup=allow_lookup,
        corpus_type=corpus_type,
        prediction_counts=prediction_counts,
        status_counts=status_counts,
        error_counts=error_counts,
        reference_counts=reference_counts,
        source_files=tuple(sorted(source_files)),
    )


@dataclass(frozen=True)
class Ballot:
    model: str
    label: str | None
    reason: str
    weight: int
    missing: bool
    invalid: bool = False

    @property
    def note(self) -> str:
        if self.missing:
            return "缺票（该模型未提供结果）"
        if self.invalid:
            return "无效票（模型未形成有效结论）"
        if self.label is None:
            return "弃权（模型未形成有效票）"
        return ""


@dataclass(frozen=True)
class AnnotateRow:
    input_line: int
    candidate_key: str
    inner_key: str | None
    sentence_a: str
    sentence_b: str
    focus_a: str
    focus_b: str
    total_label: str | None
    same_meaning: bool | None
    status: str
    voting_status: str
    reason: str
    ballots: tuple[Ballot, ...]

    def ballot(self, model: str) -> Ballot | None:
        for ballot in self.ballots:
            if ballot.model == model:
                return ballot
        return None

    def total_side(self, label: str) -> common.SideView:
        return common.SideView(
            layer=f"{label} 总结果",
            kind="annotate 总结果",
            label=self.total_label,
            same_meaning=self.same_meaning,
            status=self.status,
            voting_status=self.voting_status,
            line_no="",
            input_line=str(self.input_line),
            source_file=label,
            sentence_a=self.sentence_a,
            sentence_b=self.sentence_b,
            focus_a=self.focus_a,
            focus_b=self.focus_b,
            reason=self.reason,
        )

    def model_side(self, ballot: Ballot, label: str) -> common.SideView:
        return common.SideView(
            layer=f"{label}/{ballot.model}",
            kind="annotate 模型票",
            label=ballot.label,
            same_meaning=None,
            status="missing" if ballot.missing else ("abstain" if ballot.label is None else "ok"),
            voting_status=self.voting_status,
            line_no="",
            input_line=str(self.input_line),
            source_file=label,
            sentence_a=self.sentence_a,
            sentence_b=self.sentence_b,
            focus_a=self.focus_a,
            focus_b=self.focus_b,
            reason=ballot.reason,
            note=ballot.note,
        )


@dataclass
class LoadedAnnotate:
    path: Path
    label: str
    rows: list[AnnotateRow]
    rows_by_inner: dict[str, common.SideView]
    row_by_inner: dict[str, AnnotateRow]
    mode: str
    is_voting: bool
    models: tuple[str, ...]
    total_counts: Counter[str]
    status_counts: Counter[str]
    voting_status_counts: Counter[str]
    missing_by_model: Counter[str]
    invalid_by_model: Counter[str]

    @property
    def kind(self) -> str:
        return "投票汇总" if self.is_voting else "单模型"

    @property
    def total(self) -> int:
        return len(self.rows)


def load_annotate(
    path: str | Path,
    *,
    label: str = "annotate",
    progress_every: int = 50_000,
    progress_stream=None,
) -> LoadedAnnotate:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到 annotate 结果文件：{source}")
    if source.suffix.lower() != ".jsonl":
        raise ValueError(f"annotate 结果只支持 JSONL：{source}")

    rows: list[AnnotateRow] = []
    rows_by_inner: dict[str, common.SideView] = {}
    row_by_inner: dict[str, AnnotateRow] = {}
    seen: dict[str, int] = {}
    modes: set[str] = set()
    voting_flags: set[bool] = set()
    models: set[str] = set()
    total_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    voting_status_counts: Counter[str] = Counter()
    missing_by_model: Counter[str] = Counter()
    invalid_by_model: Counter[str] = Counter()

    with source.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{source} 第 {line_no} 行不是合法 JSON：{exc}"
                ) from exc
            if not isinstance(record, dict):
                _fail(source, line_no, "记录必须是 JSON 对象")
            if "annotation_mode" not in record and "prediction" in record:
                _fail(
                    source,
                    line_no,
                    "这是 evaluate details，不是 annotate 结果；"
                    "details 请作为位置参数传入",
                )

            projected = annotate_records.record_row(
                record, source=source, line_no=line_no
            )
            identity = inner_key(record, source, line_no)
            if identity is not None:
                if identity in seen:
                    raise ValueError(
                        f"{source} 第 {line_no} 行身份重复：{identity!r}"
                        f"（首次位于第 {seen[identity]} 行）；不能按出现顺序配对"
                    )
                seen[identity] = line_no

            raw_voting = record.get("voting")
            ballots: tuple[Ballot, ...] = ()
            if isinstance(raw_voting, dict):
                raw_ballots = raw_voting.get("ballots")
                if isinstance(raw_ballots, dict):
                    missing = set(projected.voting.missing_models if projected.voting else ())
                    invalid = set(projected.voting.invalid_models if projected.voting else ())
                    ballots = tuple(
                        Ballot(
                            model=model,
                            label=ballot.get("label"),
                            reason=ballot.get("reason") or "",
                            weight=(
                                ballot.get("weight")
                                if isinstance(ballot.get("weight"), int)
                                and not isinstance(ballot.get("weight"), bool)
                                else 1
                            ),
                            missing=model in missing,
                            invalid=model in invalid,
                        )
                        for model, ballot in raw_ballots.items()
                    )
                    models.update(ballot.model for ballot in ballots)

            row = AnnotateRow(
                input_line=line_no,
                candidate_key=projected.key or "",
                inner_key=identity,
                sentence_a=projected.sentence_a.strip(),
                sentence_b=projected.sentence_b.strip(),
                focus_a=projected.focus_a,
                focus_b=projected.focus_b,
                total_label=projected.label,
                same_meaning=projected.same_meaning,
                status=projected.status,
                voting_status=projected.voting_status_text,
                reason=projected.judgement_reason,
                ballots=ballots,
            )
            rows.append(row)
            if identity is not None:
                row_by_inner[identity] = row
                rows_by_inner[identity] = row.total_side(label)

            modes.add(projected.mode)
            voting_flags.add(projected.is_voting)
            total_counts[common.label_text(projected.label)] += 1
            status_counts[projected.status] += 1
            if projected.voting is not None:
                voting_status_counts[projected.voting.status] += 1
                for model in projected.voting.missing_models:
                    missing_by_model[model] += 1
                for model in projected.voting.invalid_models:
                    invalid_by_model[model] += 1
                models.update(projected.voting.models)
            elif projected.model_name:
                models.add(projected.model_name)

            if progress_every and len(rows) % progress_every == 0:
                print(
                    f"{source.name}：已读取 {len(rows)} 条记录",
                    file=progress_stream or sys.stderr,
                    flush=True,
                )

    if len(modes) > 1:
        raise ValueError(
            f"{source} 混有多个 annotation_mode：{', '.join(sorted(modes))}"
        )
    if len(voting_flags) > 1:
        raise ValueError(
            f"{source} 同时含投票与非投票记录；投票汇总文件应每条都带 voting 块"
        )

    return LoadedAnnotate(
        path=source,
        label=label,
        rows=rows,
        rows_by_inner=rows_by_inner,
        row_by_inner=row_by_inner,
        mode=next(iter(modes), ""),
        is_voting=next(iter(voting_flags), False),
        models=tuple(sorted(models)),
        total_counts=total_counts,
        status_counts=status_counts,
        voting_status_counts=voting_status_counts,
        missing_by_model=missing_by_model,
        invalid_by_model=invalid_by_model,
    )


def _same_context(left: common.SideView, right: common.SideView) -> bool:
    return (
        left.sentence_a.strip(),
        left.sentence_b.strip(),
    ) == (
        right.sentence_a.strip(),
        right.sentence_b.strip(),
    )


def build_cross_comparison(
    details: LoadedDetails,
    annotate: LoadedAnnotate,
    *,
    model: str | None = None,
) -> common.Comparison:
    """把一份 details 对 annotate 总结果或某个模型票做逐条比较。

    ``model=None`` 表示总结果；否则只看 ``voting.ballots[model]``。
    缺票（``missing_models`` 或有记录但没有该模型 ballot）不计入共同项，
    单独计入 ``right_unavailable``；弃权（有 ballot、label=null）仍计入共同项。
    """
    left_map: dict[str, common.SideView] = {}
    excluded: set[str] = set()
    for row in details.rows:
        key = row.inner_key
        if key is None:
            continue
        if details.inner_counts.get(key, 0) > 1:
            excluded.add(key)
            continue
        left_map[key] = row.side(details.label)
    ambiguous = sum(1 for count in details.inner_counts.values() if count > 1)

    right_map: dict[str, common.SideView] = {}
    unavailable = 0
    right_label = f"{annotate.label} 总结果" if model is None else f"{annotate.label}/{model}"
    for key, annotate_row in annotate.row_by_inner.items():
        if model is None:
            view = annotate_row.total_side(annotate.label)
        else:
            ballot = annotate_row.ballot(model)
            if ballot is None or ballot.missing:
                if key in left_map:
                    unavailable += 1
                    excluded.add(key)
                continue
            view = annotate_row.model_side(ballot, annotate.label)
        right_map[key] = view

    for key in excluded:
        left_map.pop(key, None)
        right_map.pop(key, None)

    conflicts: set[str] = set()
    for key in left_map.keys() & right_map.keys():
        if not _same_context(left_map[key], right_map[key]):
            conflicts.add(key)

    return common.Comparison(
        left_label=details.label,
        right_label=right_label,
        left_map=left_map,
        right_map=right_map,
        right_unavailable=unavailable,
        left_ambiguous=ambiguous,
        conflict_keys=conflicts,
    )
