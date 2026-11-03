# encoding: utf8

"""评测输入：读取 annotate/ 的 JSONL，提取参考答案与句对。

参考标签的优先级（见 ``extract_reference``）：

    多模型投票记录（model_name=voting，或带 voting/voting_label）
        仅在 ``voting.status=accepted`` 时看顶层 ``voting_label``：
        same/different 才可评测；未采纳、平票、无有效票、
        discarded、not_spatial、not_fluent*、schema:*、fluent:* 都跳过。
        刻意**不回退**到代表模型的 ``judgement.same_meaning``——投票平票时
        代表记录可能残留一个布尔值，回退会把它误当成正式结论。

    单模型记录
        先看现实空间/通顺性闸门；再看 ``judgement.same_meaning`` 是否为
        真正的 bool。字符串 "true"、整数 1 都不接受，避免把脏数据静默算进指标。

同一份输入里重复的记录按「输入文件 + id」去重；没有 id 时按
``voting.record_identity``（line_no / candidate.index / focus_index / replacement）
去重。重复项计入 skipped，不参与指标，避免同一条语料被重复加权。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from annotate.voting import VOTING_ACCEPTED, record_identity

LABEL_SAME = "same"
LABEL_DIFFERENT = "different"
LABELS = (LABEL_SAME, LABEL_DIFFERENT)

SKIP_INVALID_JSON = "invalid_json"
SKIP_INVALID_RECORD = "invalid_record"
SKIP_INVALID_PAIR = "invalid_pair"
SKIP_DUPLICATE = "duplicate_record"
SKIP_VOTING_TIE = "voting_tie"
SKIP_VOTING_NO_VALID_VOTES = "voting_no_valid_votes"
SKIP_VOTING_NOT_ACCEPTED = "voting_not_accepted"
SKIP_DISCARDED = "discarded"
SKIP_NOT_SPATIAL = "not_spatial"
SKIP_NOT_FLUENT = "not_fluent"
SKIP_FEATURE_ERROR = "feature_error"
SKIP_JUDGEMENT_UNKNOWN = "judgement_unknown"
SKIP_JUDGEMENT_MISSING = "judgement_missing"
SKIP_VOTING_LABEL_MISSING = "voting_label_missing"
SKIP_UNSUPPORTED = "unsupported_reference_label"
SKIP_REFERENCE_CONFLICT = "reference_conflict"


@dataclass(frozen=True)
class RawRecord:
    """JSONL 的一行：解析成功时 ``record`` 是对象，失败时 ``error_kind`` 说明原因。"""

    input_file: Path
    input_line_no: int
    record: dict | None = None
    error_kind: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ReferenceResult:
    """一条记录的参考答案；``label=None`` 时 ``reason`` 是跳过原因。"""

    label: str | None
    source: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class EvalSample:
    """一条可评测样本：参考答案、句对与来源身份。"""

    key: str
    input_file: str
    input_line_no: int
    record_id: str | None
    line_no: int | None
    sentence_a: str
    sentence_b: str
    focus_word: object | None
    focus_word_b: object | None
    reference_label: str
    reference_source: str
    reference_model: str | None
    annotation_status: str | None
    voting_status: str | None
    candidate: dict | None


@dataclass
class Dataset:
    """一次输入扫描的结果。"""

    samples: list[EvalSample]
    skipped: Counter
    records_read: int = 0
    invalid_json: int = 0
    invalid_record: int = 0
    duplicates: int = 0

    def skipped_total(self) -> int:
        return sum(self.skipped.values())


def read_records(path: str | Path) -> Iterator[RawRecord]:
    """逐行读 JSONL；坏行不中断整个文件，交给跳过统计记一笔。"""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                yield RawRecord(path, line_no, None, SKIP_INVALID_JSON, f"JSON 解析失败：{exc}")
                continue
            if not isinstance(record, dict):
                yield RawRecord(
                    path,
                    line_no,
                    None,
                    SKIP_INVALID_RECORD,
                    f"记录不是 JSON 对象，而是 {type(record).__name__}",
                )
                continue
            yield RawRecord(path, line_no, record)


def _as_dict(value) -> dict | None:
    return value if isinstance(value, dict) else None


def _non_fluent_sides(record: dict) -> list[str]:
    semantic = _as_dict(record.get("semantic")) or {}
    sides = []
    for side in ("a", "b"):
        item = _as_dict(semantic.get(side)) or {}
        if item.get("is_fluent") is False:
            sides.append(side)
    return sides


def _int_or_none(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _reference_conflict(record: dict, reality_false: bool, non_fluent: list[str]) -> str | None:
    """有明确 same/different 却同时被闸门否定：口径冲突，跳过而不是硬算。"""
    if reality_false:
        return SKIP_REFERENCE_CONFLICT
    if non_fluent:
        return SKIP_REFERENCE_CONFLICT
    return None


def extract_reference(record: dict) -> ReferenceResult:
    """从一条 annotate 结果里取出可评测的参考答案。"""
    if not isinstance(record, dict):
        return ReferenceResult(None, reason=SKIP_INVALID_RECORD)

    voting = _as_dict(record.get("voting")) or {}
    is_voting = (
        record.get("model_name") == "voting"
        or "voting" in record
        or "voting_label" in record
    )
    reality = _as_dict(record.get("reality")) or {}
    reality_false = reality.get("is_spatial_scene") is False
    non_fluent = _non_fluent_sides(record)
    not_fluent_reason = (
        SKIP_NOT_FLUENT + "_" + "".join(non_fluent) if non_fluent else SKIP_NOT_FLUENT
    )

    if is_voting:
        voting_status = str(voting.get("status") or "")
        if voting_status == "tie":
            return ReferenceResult(None, reason=SKIP_VOTING_TIE)
        if voting_status == "no_valid_votes":
            return ReferenceResult(None, reason=SKIP_VOTING_NO_VALID_VOTES)
        if record.get("status") == "discarded" or voting_status == "discarded":
            return ReferenceResult(None, reason=SKIP_DISCARDED)
        if voting_status != VOTING_ACCEPTED:
            return ReferenceResult(None, reason=SKIP_VOTING_NOT_ACCEPTED)

        label = record.get("voting_label")
        if label in LABELS:
            conflict = _reference_conflict(record, reality_false, non_fluent)
            if conflict:
                return ReferenceResult(None, reason=conflict)
            return ReferenceResult(label, source="voting_label")

        if label is None:
            if reality_false:
                return ReferenceResult(None, reason=SKIP_NOT_SPATIAL)
            if non_fluent:
                return ReferenceResult(None, reason=not_fluent_reason)
            return ReferenceResult(None, reason=SKIP_VOTING_LABEL_MISSING)

        # 投票标签还可能是 not_spatial / not_fluent_ab / schema:... / fluent:...
        text = str(label)
        if text == "not_spatial" or reality_false:
            return ReferenceResult(None, reason=SKIP_NOT_SPATIAL)
        if text.startswith("not_fluent"):
            # 保留投票标签里的具体侧别（not_fluent_a / not_fluent_ab），便于审计。
            return ReferenceResult(None, reason=text)
        if non_fluent:
            return ReferenceResult(None, reason=not_fluent_reason)
        return ReferenceResult(None, reason=SKIP_UNSUPPORTED)

    if reality_false:
        return ReferenceResult(None, reason=SKIP_NOT_SPATIAL)
    if non_fluent:
        return ReferenceResult(None, reason=not_fluent_reason)

    status = str(record.get("status") or "")
    if status.startswith(SKIP_FEATURE_ERROR):
        return ReferenceResult(None, reason=SKIP_FEATURE_ERROR)
    if status == "discarded":
        return ReferenceResult(None, reason=SKIP_DISCARDED)

    judgement = _as_dict(record.get("judgement")) or {}
    value = judgement.get("same_meaning")
    if isinstance(value, bool):
        return ReferenceResult(LABEL_SAME if value else LABEL_DIFFERENT, source="judgement")
    if value is None:
        if status == SKIP_JUDGEMENT_UNKNOWN:
            return ReferenceResult(None, reason=SKIP_JUDGEMENT_UNKNOWN)
        return ReferenceResult(None, reason=SKIP_JUDGEMENT_MISSING)
    # 字符串 "true"/"false"、整数 1/0 等都不接受：参考标签必须是真正的 bool。
    return ReferenceResult(None, reason=SKIP_UNSUPPORTED)


def _sentences(record: dict) -> tuple[str | None, str | None]:
    pair = _as_dict(record.get("pair")) or {}
    side_a = _as_dict(pair.get("a")) or {}
    side_b = _as_dict(pair.get("b")) or {}
    sentence_a = side_a.get("sentence")
    sentence_b = side_b.get("sentence")
    if sentence_a is None:
        sentence_a = record.get("sentence_a")
    if sentence_b is None:
        sentence_b = record.get("sentence_b")
    return sentence_a, sentence_b


def _sample_key(raw: RawRecord, record: dict, record_id: str | None) -> str:
    prefix = str(Path(raw.input_file).resolve())
    if record_id:
        return f"{prefix}::{record_id}"
    identity = record_identity(record)
    return f"{prefix}::{'|'.join(str(part) for part in identity)}"


def build_sample(raw: RawRecord) -> tuple[EvalSample | None, str | None]:
    """把一行原始记录变成可评测样本；返回 ``(sample, 跳过原因)``。"""
    if raw.record is None:
        return None, raw.error_kind or SKIP_INVALID_RECORD

    record = raw.record
    reference = extract_reference(record)
    if reference.label is None:
        return None, reference.reason or SKIP_JUDGEMENT_MISSING

    sentence_a, sentence_b = _sentences(record)
    if not isinstance(sentence_a, str) or not sentence_a.strip():
        return None, SKIP_INVALID_PAIR
    if not isinstance(sentence_b, str) or not sentence_b.strip():
        return None, SKIP_INVALID_PAIR

    raw_id = record.get("id")
    record_id = str(raw_id) if raw_id is not None else None
    voting = _as_dict(record.get("voting")) or {}
    return (
        EvalSample(
            key=_sample_key(raw, record, record_id),
            input_file=str(Path(raw.input_file).resolve()),
            input_line_no=raw.input_line_no,
            record_id=record_id,
            line_no=_int_or_none(record.get("line_no")),
            sentence_a=sentence_a.strip(),
            sentence_b=sentence_b.strip(),
            focus_word=record.get("focus_word"),
            focus_word_b=record.get("focus_word_b"),
            reference_label=reference.label,
            reference_source=reference.source or "unknown",
            reference_model=record.get("model_name")
            if isinstance(record.get("model_name"), str)
            else None,
            annotation_status=record.get("status")
            if isinstance(record.get("status"), str)
            else None,
            voting_status=str(voting.get("status") or "") or None,
            candidate=_as_dict(record.get("candidate")),
        ),
        None,
    )


def collect_samples(paths: Iterable[str | Path]) -> Dataset:
    """读取全部输入文件，按参考标签与身份筛选出可评测样本。"""
    dataset = Dataset(samples=[], skipped=Counter())
    seen: set[str] = set()
    for path in paths:
        for raw in read_records(path):
            dataset.records_read += 1
            if raw.error_kind:
                dataset.skipped[raw.error_kind] += 1
                if raw.error_kind == SKIP_INVALID_JSON:
                    dataset.invalid_json += 1
                elif raw.error_kind == SKIP_INVALID_RECORD:
                    dataset.invalid_record += 1
                continue

            sample, reason = build_sample(raw)
            if sample is None:
                dataset.skipped[reason or SKIP_JUDGEMENT_MISSING] += 1
                continue
            if sample.key in seen:
                dataset.skipped[SKIP_DUPLICATE] += 1
                dataset.duplicates += 1
                continue
            seen.add(sample.key)
            dataset.samples.append(sample)
    return dataset
