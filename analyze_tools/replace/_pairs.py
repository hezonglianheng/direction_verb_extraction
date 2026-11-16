# encoding: utf8
"""replace 分析的共同数据契约；不导入生产模型或读取继承的整句筛选。"""

from __future__ import annotations

from collections import Counter
from contextlib import closing, contextmanager
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterator


PAIR_SCHEMA_VERSION = "1.0"
PAIR_STATUSES = ("true", "false", "error", "limit", "no_llm")
SIDE_COLUMNS = (
    "句子", "切片", "关注词", "关注词下标", "目标物", "参照物", "动词",
    "判定", "语法正确", "空间自洽", "理由", "判定来源",
)
PAIR_COLUMNS = [
    "JSONL行号", "句对版本", "candidate_key", "源文件", "语料类型", "替换词", "句对状态",
    *[side + column for side in ("A", "B") for column in SIDE_COLUMNS],
]


@dataclass(frozen=True, slots=True)
class SideRow:
    sentence: str
    tokens: tuple[str, ...] | None
    word: str
    index: int
    target: str
    reference: str
    verb: str
    extractable: bool | None
    grammatical: bool | None
    spatial_coherent: bool | None
    reason: str
    source: str

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.target, self.reference, self.verb)


@dataclass(frozen=True, slots=True)
class PairRow:
    line_no: str
    schema_version: str
    key: str
    source_file: str
    corpus_type: str
    replacement: str
    status: str
    a: SideRow
    b: SideRow

    @property
    def signature(self) -> tuple[bool | None, bool | None, str]:
        """主比较口径；空值相同是字段相同，不代表模型已确认通过。"""
        return (self.a.extractable, self.b.extractable, self.status)


def cell(value: Any) -> str:
    return "" if value is None else str(value)


def bool_cell(value: bool | None) -> str:
    if value is None:
        return ""
    return "true" if value else "false"


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是 JSON 对象")
    return value


def _boolean(value: Any, field: str) -> bool | None:
    if value is None or type(value) is bool:
        return value
    raise ValueError(f"{field} 非法：{value!r}（只接受 true/false/null）")


def _index(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} 必须是非负整数 token 下标")
    return value


def _tokens(value: Any, field: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(token, str) for token in value):
        raise ValueError(f"{field} 必须是非空字符串列表或 null")
    return tuple(value)


def _word(node: Any) -> str:
    return cell(node.get("word")) if isinstance(node, dict) else ""


def _side_row(record: dict[str, Any], side: str) -> SideRow:
    focus_key = "focus_word" if side == "a" else "focus_word_b"
    focus = _object(record.get(focus_key), focus_key)
    data = _object(record["pair"].get(side), f"pair.{side}")
    check = _object(record["pair_screening"].get(side), f"pair_screening.{side}")
    word = _required_text(focus.get("word"), f"{focus_key}.word")
    index = _index(focus.get("index"), f"{focus_key}.index")
    tokens = _tokens(data.get("slice"), f"pair.{side}.slice")
    sentence = data.get("sentence")
    if sentence is None and tokens is not None:
        sentence = "".join(tokens)
    sentence = _required_text(sentence, f"pair.{side}.sentence")
    if tokens is not None:
        if index >= len(tokens) or tokens[index] != word:
            raise ValueError(f"{focus_key} 与 pair.{side}.slice 的关注词位置不一致")
        if sentence != "".join(tokens):
            raise ValueError(f"pair.{side}.sentence 与 slice 不一致")
    for field, expected in (("index", index), ("word", word)):
        if field in check and (type(check[field]) is not type(expected) or check[field] != expected):
            raise ValueError(f"pair_screening.{side}.{field} 与 {focus_key} 不一致")

    grounding = focus.get("grounding")
    if grounding is None:
        grounding = data.get("grounding")
    if grounding is not None:
        grounding = _object(grounding, f"{focus_key}.grounding / pair.{side}.grounding")
    grounding = grounding or {}
    return SideRow(
        sentence=sentence, tokens=tokens, word=word, index=index,
        target=_word(grounding.get("target")), reference=_word(grounding.get("reference")),
        verb=_word(grounding.get("verb")),
        extractable=_boolean(check.get("extractable"), f"pair_screening.{side}.extractable"),
        grammatical=_boolean(check.get("grammatical"), f"pair_screening.{side}.grammatical"),
        spatial_coherent=_boolean(check.get("spatial_coherent"), f"pair_screening.{side}.spatial_coherent"),
        reason=cell(check.get("reason")), source=cell(check.get("source")),
    )


def pair_from_record(record: dict[str, Any], line_no: str | int) -> PairRow:
    if record.get("pair_schema_version") != PAIR_SCHEMA_VERSION:
        raise ValueError(
            f"pair_schema_version 必须是 {PAIR_SCHEMA_VERSION!r}；输入必须是 replace 句对，不是 extract 原句"
        )
    key = _required_text(record.get("candidate_key"), "candidate_key")
    corpus_type = record.get("corpus_type")
    if corpus_type not in ("方位词", "趋向动词"):
        raise ValueError(f"corpus_type 非法：{corpus_type!r}")
    candidate = _object(record.get("candidate"), "candidate")
    replacement = _required_text(candidate.get("replacement"), "candidate.replacement")
    _object(record.get("pair"), "pair")
    checks = _object(record.get("pair_screening"), "pair_screening")
    status = checks.get("status")
    if status not in PAIR_STATUSES:
        raise ValueError(f"pair_screening.status 非法：{status!r}")
    a, b = _side_row(record, "a"), _side_row(record, "b")
    if b.word != replacement or b.index != a.index:
        raise ValueError("focus_word_b 与 candidate.replacement / focus_word.index 不一致")
    if "focus_index" in candidate and _index(candidate["focus_index"], "candidate.focus_index") != a.index:
        raise ValueError("candidate.focus_index 与 focus_word.index 不一致")
    return PairRow(
        line_no=cell(line_no), schema_version=PAIR_SCHEMA_VERSION, key=key,
        source_file=cell(record.get("source_file")), corpus_type=corpus_type,
        replacement=replacement, status=status, a=a, b=b,
    )


def pair_to_csv_row(row: PairRow) -> dict[str, str]:
    result = {
        "JSONL行号": row.line_no, "句对版本": row.schema_version, "candidate_key": row.key,
        "源文件": row.source_file, "语料类型": row.corpus_type, "替换词": row.replacement,
        "句对状态": row.status,
    }
    for label, side in (("A", row.a), ("B", row.b)):
        values = {
            "句子": side.sentence,
            "切片": json.dumps(side.tokens, ensure_ascii=False) if side.tokens is not None else "",
            "关注词": side.word, "关注词下标": str(side.index),
            "目标物": side.target, "参照物": side.reference, "动词": side.verb,
            "判定": bool_cell(side.extractable), "语法正确": bool_cell(side.grammatical),
            "空间自洽": bool_cell(side.spatial_coherent), "理由": side.reason, "判定来源": side.source,
        }
        result.update({label + column: value for column, value in values.items()})
    return result


def _csv_boolean(value: str, field: str) -> bool | None:
    value = value.strip().lower()
    if not value:
        return None
    if value not in ("true", "false"):
        raise ValueError(f"{field} 非法：{value!r}（只接受 true/false/空串）")
    return value == "true"


def _pair_from_csv(raw: dict[str, str]) -> PairRow:
    record: dict[str, Any] = {
        "pair_schema_version": raw["句对版本"], "candidate_key": raw["candidate_key"],
        "source_file": raw["源文件"], "corpus_type": raw["语料类型"],
        "candidate": {"replacement": raw["替换词"]},
        "pair": {}, "pair_screening": {"status": raw["句对状态"]},
    }
    for label, name, focus_key in (("A", "a", "focus_word"), ("B", "b", "focus_word_b")):
        index_text = raw[label + "关注词下标"]
        if not index_text.isascii() or not index_text.isdigit():
            raise ValueError(f"{label}关注词下标 非法：{index_text!r}")
        grounding = {role: {"word": raw[label + column]} for role, column in
                     (("target", "目标物"), ("reference", "参照物"), ("verb", "动词"))}
        record[focus_key] = {"word": raw[label + "关注词"], "index": int(index_text), "grounding": grounding}
        try:
            tokens = json.loads(raw[label + "切片"]) if raw[label + "切片"] else None
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}切片 不是合法 JSON：{exc}") from exc
        record["pair"][name] = {"sentence": raw[label + "句子"], "slice": tokens}
        record["pair_screening"][name] = {
            "extractable": _csv_boolean(raw[label + "判定"], label + "判定"),
            "grammatical": _csv_boolean(raw[label + "语法正确"], label + "语法正确"),
            "spatial_coherent": _csv_boolean(raw[label + "空间自洽"], label + "空间自洽"),
            "reason": raw[label + "理由"], "source": raw[label + "判定来源"],
        }
    return pair_from_record(record, raw["JSONL行号"])


def _iter_jsonl(source: Path) -> Iterator[tuple[int, PairRow]]:
    with source.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("不是 JSON 对象")
                row = pair_from_record(record, line_no)
            except ValueError as exc:
                raise ValueError(f"{source} 第 {line_no} 行：{exc}") from exc
            yield line_no, row


@contextmanager
def _large_csv_fields():
    """标准库默认仅允许 128 KiB；C long 的 32 位上限也兼容 Windows。"""
    previous = csv.field_size_limit()
    csv.field_size_limit(max(previous, 2**31 - 1))
    try:
        yield
    finally:
        csv.field_size_limit(previous)


def _iter_csv(source: Path) -> Iterator[tuple[int, PairRow]]:
    with _large_csv_fields(), source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        try:
            columns = reader.fieldnames or []
            missing = sorted(set(PAIR_COLUMNS) - set(columns))
            if missing:
                raise ValueError(f"缺列：{', '.join(missing)}；请使用 replace 句对导出 CSV")
            if len(columns) != len(set(columns)):
                raise ValueError("CSV 列名重复")
            for raw in reader:
                if None in raw or any(raw.get(column) is None for column in PAIR_COLUMNS):
                    raise ValueError("CSV 行字段数与表头不一致")
                try:
                    row = _pair_from_csv(raw)
                except ValueError as exc:
                    raise ValueError(f"{source} 第 {reader.line_num} 行：{exc}") from exc
                yield reader.line_num, row
        except csv.Error as exc:
            raise ValueError(f"{source} 第 {reader.line_num} 行 CSV 解析失败：{exc}") from exc
        except ValueError as exc:
            if str(exc).startswith(str(source)):
                raise
            raise ValueError(f"{source} 第 {reader.line_num} 行：{exc}") from exc


def iter_pair_rows(
    path: str | Path, *, progress_every: int = 50_000, progress_stream=None,
) -> Iterator[PairRow]:
    """逐行读 JSONL 或本工具导出的 CSV；只为查重保留 key 与首次行号。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    if source.suffix.lower() == ".jsonl":
        iterator = _iter_jsonl(source)
    elif source.suffix.lower() == ".csv":
        iterator = _iter_csv(source)
    else:
        raise ValueError(f"只支持 replace .jsonl / .csv：{source}")
    if progress_every < 0:
        raise ValueError("progress_every 不能为负数")
    seen: dict[str, int] = {}
    with closing(iterator):
        for count, (line_no, row) in enumerate(iterator, 1):
            if row.key in seen:
                raise ValueError(
                    f"{source} 第 {line_no} 行 candidate_key 重复：{row.key!r}"
                    f"（首次位于第 {seen[row.key]} 行）"
                )
            seen[row.key] = line_no
            yield row
            if progress_every and count % progress_every == 0:
                print(f"{source.name}：已读取 {count} 条句对", file=progress_stream or sys.stderr, flush=True)


def update_stats(stats: Counter[str], row: PairRow) -> None:
    stats["records"] += 1
    stats[f"pair_status_{row.status}"] += 1
    for label, side in (("a", row.a), ("b", row.b)):
        for field, suffix in (("extractable", ""), ("grammatical", "_grammatical"),
                              ("spatial_coherent", "_spatial_coherent")):
            state = bool_cell(getattr(side, field)) or "null"
            stats[f"{label}{suffix}_{state}"] += 1


def same_file(left: Path, right: Path) -> bool:
    try:
        if left.resolve() == right.resolve():
            return True
        return left.samefile(right)
    except (OSError, RuntimeError):
        return False
