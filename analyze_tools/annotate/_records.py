# encoding: utf8
"""annotate 分析工具的记录契约：读取、校验与轻量投影。

只支持正式图 ``pair`` / ``synonym`` / ``all_in_one`` 的结果 JSONL（schema 1.3）：

- ``--model`` 的单模型结果：没有 ``voting`` 块，结论看
  ``judgement.same_meaning`` 与 ``reality`` / ``semantic`` 闸门；
- ``--models`` 的最终投票结果：带 ``voting`` 块，**最终结论看
  ``voting.winner_label``**；``judgement.same_meaning`` 只是代表模型继承下来的字段，
  采纳 ``not_spatial`` / ``not_fluent_*`` 时它甚至可能是 null，不能当投票共识；
- ``--models`` 各模型的原始结果位于 ``<最终文件主名>.models/00_<模型>.jsonl``，
  与单模型结果同形，可直接传给比较脚本。

本模块不导入 ``spatial_tools`` / ``spatial_runner``（导入会连带加载 LTP/torch），
只复用纯函数模块 ``annotate.voting`` 的取票口径（见其模块 docstring）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from annotate import voting as voting_rules  # noqa: E402  # 纯函数，不加载 LTP/torch


RESULT_SCHEMA_VERSION = "1.3"
"""与 ``annotate.spatial_tools.RESULT_SCHEMA_VERSION`` 保持一致。

刻意不导入该模块（会加载 LTP/torch）；``tests/test_annotate_analysis.py`` 会读源码
比对这一常量，版本升了却没更新比较工具时测试会失败。
"""

FORMAL_MODES = ("pair", "synonym", "all_in_one")
VOTING_STATUSES = (
    voting_rules.VOTING_ACCEPTED,
    voting_rules.VOTING_TIE,
    voting_rules.VOTING_NO_VALID_VOTES,
)
UNRESOLVED_LABEL = "（无结论）"


def label_text(label: str | None) -> str:
    """结论标签的统一文本；None 表示这次没有形成结论（弃权/平票/未判定）。"""
    return UNRESOLVED_LABEL if label is None else label


def boolean_text(value: bool | None) -> str:
    """三态布尔：true / false / 空串（空串只用于 CSV，避免把空值写成 false）。"""
    if value is None:
        return ""
    return "true" if value else "false"


def _fail(source: Path, line_no: int, message: str):
    raise ValueError(f"{source} 第 {line_no} 行：{message}")


def _object(
    value: Any, field: str, source: Path, line_no: int, *, required: bool = False
) -> dict[str, Any]:
    if value is None:
        if required:
            _fail(source, line_no, f"缺少 {field}")
        return {}
    if not isinstance(value, dict):
        _fail(
            source,
            line_no,
            f"{field} 必须是 JSON 对象，实际是 {type(value).__name__}",
        )
    return value


def _boolean(value: Any, field: str, source: Path, line_no: int) -> bool | None:
    if value is None or type(value) is bool:
        return value
    _fail(source, line_no, f"{field} 非法：{value!r}（只接受 true/false/null）")


def _text(
    value: Any,
    field: str,
    source: Path,
    line_no: int,
    *,
    required: bool = False,
    allow_empty: bool = True,
) -> str:
    if value is None:
        if required:
            _fail(source, line_no, f"缺少 {field}")
        return ""
    if not isinstance(value, str):
        _fail(
            source,
            line_no,
            f"{field} 必须是字符串，实际是 {type(value).__name__}",
        )
    if required and not allow_empty and not value.strip():
        _fail(source, line_no, f"{field} 不能为空")
    return value


def _string_list(value: Any, field: str, source: Path, line_no: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _fail(source, line_no, f"{field} 必须是字符串数组或 null")
    return tuple(value)


def _display(value: Any) -> str:
    """展示字段：容忍旧数据里的非字符串，但不参与核心比较。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _dig(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _schema_name(
    record: dict[str, Any], side: str, source: Path, line_no: int
) -> str:
    schema = _object(record.get("schema"), "schema", source, line_no)
    item = _object(schema.get(side), f"schema.{side}", source, line_no)
    name = item.get("name")
    if name is None:
        return ""
    if not isinstance(name, str):
        _fail(source, line_no, f"schema.{side}.name 必须是字符串或 null")
    return name


@dataclass(frozen=True, slots=True)
class VotingView:
    """``voting`` 块的比较投影。

    ``winner_label`` 是投票共识；``discarded``（tie / no_valid_votes）时没有结论，
    不能退回代表模型的继承字段。
    """

    status: str
    winner_label: str | None
    winner_model: str | None
    incomplete: bool | None
    missing_models: tuple[str, ...]
    invalid_models: tuple[str, ...]
    models: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.status == voting_rules.VOTING_ACCEPTED


def _voting_view(
    record: dict[str, Any], source: Path, line_no: int
) -> VotingView | None:
    voting = record.get("voting")
    if voting is None:
        return None
    if not isinstance(voting, dict):
        _fail(source, line_no, "voting 必须是 JSON 对象")
    status = voting.get("status")
    if status not in VOTING_STATUSES:
        _fail(
            source,
            line_no,
            f"voting.status 非法：{status!r}（只接受 {'/'.join(VOTING_STATUSES)}）",
        )
    winner_label = voting.get("winner_label")
    if winner_label is not None and not isinstance(winner_label, str):
        _fail(source, line_no, "voting.winner_label 必须是字符串或 null")
    if status == voting_rules.VOTING_ACCEPTED:
        if not winner_label:
            _fail(source, line_no, "voting.status=accepted 却没有 winner_label")
    elif winner_label is not None:
        _fail(
            source,
            line_no,
            f"voting.status={status} 时 winner_label 应为 null，实际是 {winner_label!r}",
        )

    winner_model = voting.get("winner_model")
    if winner_model is not None and not isinstance(winner_model, str):
        _fail(source, line_no, "voting.winner_model 必须是字符串或 null")
    incomplete = _boolean(
        voting.get("incomplete"), "voting.incomplete", source, line_no
    )
    missing = _string_list(
        voting.get("missing_models"), "voting.missing_models", source, line_no
    )
    invalid = _string_list(
        voting.get("invalid_models"), "voting.invalid_models", source, line_no
    )

    ballots = voting.get("ballots")
    models: tuple[str, ...] = ()
    if ballots is not None:
        if not isinstance(ballots, dict):
            _fail(source, line_no, "voting.ballots 必须是 JSON 对象")
        for model, ballot in ballots.items():
            if not isinstance(model, str) or not model:
                _fail(source, line_no, "voting.ballots 的模型名必须是非空字符串")
            if not isinstance(ballot, dict):
                _fail(source, line_no, f"voting.ballots.{model} 必须是 JSON 对象")
            ballot_label = ballot.get("label")
            if ballot_label is not None and not isinstance(ballot_label, str):
                _fail(
                    source,
                    line_no,
                    f"voting.ballots.{model}.label 必须是字符串或 null",
                )
        models = tuple(ballots)

    expected_top = winner_label if status == voting_rules.VOTING_ACCEPTED else None
    if "voting_label" in record and record.get("voting_label") != expected_top:
        _fail(
            source,
            line_no,
            "顶层 voting_label 与 voting.status/winner_label 不一致；"
            "结果可能被手工改过",
        )
    record_status = record.get("status")
    if status == voting_rules.VOTING_ACCEPTED and record_status == "discarded":
        _fail(source, line_no, "voting.status=accepted 但记录 status=discarded")
    if status != voting_rules.VOTING_ACCEPTED and record_status != "discarded":
        _fail(
            source,
            line_no,
            f"voting.status={status} 时记录 status 应为 discarded，"
            f"实际是 {record_status!r}",
        )

    return VotingView(
        status=status,
        winner_label=winner_label,
        winner_model=winner_model,
        incomplete=incomplete,
        missing_models=missing,
        invalid_models=invalid,
        models=models,
    )


@dataclass(frozen=True, slots=True)
class RecordRow:
    """一条 annotate 结果的比较投影；不保留特征、历史等大字段。"""

    input_line: int
    key: str | None
    mode: str
    model_name: str
    voting: VotingView | None
    label: str | None
    same_meaning: bool | None
    judgement_reason: str
    status: str
    status_reason: str
    schema_a: str
    schema_b: str
    admitted: bool | None
    review_required: bool | None
    reasons: tuple[str, ...]
    sentence_id: str
    source_file: str
    sentence_a: str
    sentence_b: str
    focus_a: str
    focus_b: str
    line_no: str

    @property
    def is_voting(self) -> bool:
        return self.voting is not None

    @property
    def voting_status_text(self) -> str:
        return self.voting.status if self.voting else ""

    @property
    def winner_label_text(self) -> str:
        if self.voting is None or self.voting.winner_label is None:
            return ""
        return self.voting.winner_label

    @property
    def winner_model_text(self) -> str:
        if self.voting is None or self.voting.winner_model is None:
            return ""
        return self.voting.winner_model

    @property
    def incomplete_text(self) -> str:
        if self.voting is None:
            return ""
        return boolean_text(self.voting.incomplete)

    @property
    def missing_text(self) -> str:
        return ", ".join(self.voting.missing_models) if self.voting else ""

    @property
    def invalid_text(self) -> str:
        return ", ".join(self.voting.invalid_models) if self.voting else ""


def record_row(
    record: dict[str, Any], *, source: Path, line_no: int
) -> RecordRow:
    """把一条结果记录校验并投影成 RecordRow；错误信息带文件与行号。"""
    if "annotation_mode" not in record:
        if "pair_schema_version" in record:
            _fail(
                source,
                line_no,
                "这是 replace 句对结果，不是 annotate 标注结果；"
                "请使用 analyze_tools/replace/compare_replace_pairs.py",
            )
        if "screening" in record:
            _fail(
                source,
                line_no,
                "这是 extract 原句结果，不是 annotate 标注结果；"
                "请使用 analyze_tools/extract/compare_extract_verdicts.py",
            )
        _fail(
            source,
            line_no,
            "缺少 annotation_mode；只支持正式图 pair/synonym/all_in_one 的结果，"
            "诊断图 single/alter_check 没有候选唯一键，不能这样比较",
        )

    mode = record.get("annotation_mode")
    if mode not in FORMAL_MODES:
        _fail(
            source,
            line_no,
            f"annotation_mode={mode!r} 不支持；只接受 {'/'.join(FORMAL_MODES)}"
            "（诊断图 single/alter_check 的结果不是正式句对契约）",
        )
    version = record.get("schema_version")
    if version != RESULT_SCHEMA_VERSION:
        _fail(
            source,
            line_no,
            f"schema_version 是 {version!r}，比较工具只支持 {RESULT_SCHEMA_VERSION!r}；"
            "旧结果请先迁移",
        )

    key_value = record.get("candidate_key")
    candidate = record.get("candidate")
    if key_value is None:
        if candidate is not None:
            _fail(
                source,
                line_no,
                "有 candidate 却缺 candidate_key；不能按行号或出现顺序兜底对齐",
            )
        key = None
    elif isinstance(key_value, str) and key_value.strip():
        key = key_value
    else:
        _fail(
            source,
            line_no,
            f"candidate_key 非法：{key_value!r}（必须是非空字符串或 null）",
        )

    status = _text(
        record.get("status"),
        "status",
        source,
        line_no,
        required=True,
        allow_empty=False,
    )
    judge = _object(record.get("judgement"), "judgement", source, line_no)
    same_meaning = _boolean(
        judge.get("same_meaning"), "judgement.same_meaning", source, line_no
    )

    admission = _object(
        record.get("admission"), "admission", source, line_no, required=True
    )
    admitted = _boolean(
        admission.get("is_admitted"), "admission.is_admitted", source, line_no
    )
    review_required = _boolean(
        admission.get("review_required"), "admission.review_required", source, line_no
    )
    reasons = _string_list(admission.get("reasons"), "admission.reasons", source, line_no)

    voting = _voting_view(record, source, line_no)
    if voting is None:
        # 单模型结果没有 voting 块：用生产侧同一份取票口径还原最终结论，
        # 这样和投票结果的 winner_label 才是同一套标签。
        # ballot_label 会直接读这些闸门字段；先按契约校验类型，避免 AttributeError
        # 或把 0/"false" 这类非法值静默当成“不是非空间/不是不通顺”。
        reality = _object(record.get("reality"), "reality", source, line_no)
        _boolean(
            reality.get("is_spatial_scene"),
            "reality.is_spatial_scene",
            source,
            line_no,
        )
        semantic = _object(record.get("semantic"), "semantic", source, line_no)
        for side in ("a", "b"):
            item = _object(semantic.get(side), f"semantic.{side}", source, line_no)
            _boolean(
                item.get("is_fluent"),
                f"semantic.{side}.is_fluent",
                source,
                line_no,
            )
            # _pair_ballot 会把失败侧的理由用 "；".join(...) 拼起来；
            # 非字符串理由会在生产函数内部抛 TypeError，必须在这里拦下。
            _text(
                item.get("reason"),
                f"semantic.{side}.reason",
                source,
                line_no,
            )
        label, _ = voting_rules.ballot_label(mode, record)
        if label is not None and not isinstance(label, str):
            _fail(source, line_no, f"无法从记录还原结论标签：{label!r}")
    else:
        label = voting.winner_label if voting.accepted else None

    return RecordRow(
        input_line=line_no,
        key=key,
        mode=mode,
        model_name=_display(record.get("model_name")),
        voting=voting,
        label=label,
        same_meaning=same_meaning,
        judgement_reason=_display(judge.get("reason")),
        status=status,
        status_reason=_display(record.get("status_reason")),
        schema_a=_schema_name(record, "a", source, line_no),
        schema_b=_schema_name(record, "b", source, line_no),
        admitted=admitted,
        review_required=review_required,
        reasons=reasons,
        sentence_id=_display(record.get("sentence_id")),
        source_file=_display(_dig(record, "source.source_file")),
        sentence_a=_display(_dig(record, "pair.a.sentence")),
        sentence_b=_display(_dig(record, "pair.b.sentence")),
        focus_a=_display(_dig(record, "focus_word.word")),
        focus_b=_display(_dig(record, "focus_word_b.word")),
        line_no=_display(record.get("line_no")),
    )


@dataclass
class LoadedFile:
    """一份已加载的结果文件；只保留比较投影与轻量统计。"""

    index: int
    path: Path
    label: str
    rows: list[RecordRow]
    mode: str
    is_voting: bool | None
    models: tuple[str, ...]
    voting_statuses: Counter[str]
    label_counts: Counter[str]
    status_counts: Counter[str]
    incomplete_count: int
    missing_by_model: Counter[str]
    invalid_by_model: Counter[str]

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
    def has_schema(self) -> bool:
        return self.mode in ("pair", "all_in_one")

    @property
    def kind(self) -> str:
        if self.is_voting is None:
            return "空文件"
        return "投票汇总" if self.is_voting else "单模型"


def _distribution(counter: Counter[str]) -> str:
    if not counter:
        return "—"
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return "，".join(f"{name}={count}" for name, count in items)


def file_distribution(counter: Counter[str]) -> str:
    return _distribution(counter)


def load_file(
    path: str | Path,
    *,
    index: int,
    label: str,
    progress_every: int = 50_000,
    progress_stream=None,
) -> LoadedFile:
    """流式读取一份结果 JSONL，校验并建立比较投影。"""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    if source.suffix.lower() != ".jsonl":
        raise ValueError(f"只支持 annotate 结果 JSONL：{source}")

    rows: list[RecordRow] = []
    seen: dict[str, int] = {}
    modes: set[str] = set()
    voting_flags: set[bool] = set()
    models: set[str] = set()
    voting_statuses: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    incomplete_count = 0
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

            row = record_row(record, source=source, line_no=line_no)
            if row.key is not None:
                if row.key in seen:
                    raise ValueError(
                        f"{source} 第 {line_no} 行 candidate_key 重复：{row.key!r}"
                        f"（首次位于第 {seen[row.key]} 行）；不能按出现顺序配对"
                    )
                seen[row.key] = line_no

            rows.append(row)
            modes.add(row.mode)
            voting_flags.add(row.is_voting)
            label_counts[label_text(row.label)] += 1
            status_counts[row.status] += 1
            if row.voting is not None:
                voting_statuses[row.voting.status] += 1
                if row.voting.incomplete:
                    incomplete_count += 1
                models.update(row.voting.models)
                if row.voting.winner_model:
                    models.add(row.voting.winner_model)
                for model in row.voting.missing_models:
                    missing_by_model[model] += 1
                for model in row.voting.invalid_models:
                    invalid_by_model[model] += 1
            elif row.model_name:
                # 单模型结果的 model_name 就是 --model / 单名 --models 的配置名。
                models.add(row.model_name)

            if progress_every and len(rows) % progress_every == 0:
                print(
                    f"{source.name}：已读取 {len(rows)} 条记录",
                    file=progress_stream or sys.stderr,
                    flush=True,
                )

    if len(modes) > 1:
        raise ValueError(
            f"{source} 混有多个 annotation_mode：{', '.join(sorted(modes))}；"
            "一次跑批只会写一张正式图，请先确认文件是否被拼接"
        )
    if len(voting_flags) > 1:
        raise ValueError(
            f"{source} 同时含投票与非投票记录；投票汇总文件应每条都带 voting 块，"
            "疑似中断后拼接或被改动"
        )

    return LoadedFile(
        index=index,
        path=source,
        label=label,
        rows=rows,
        mode=next(iter(modes), ""),
        is_voting=next(iter(voting_flags), None),
        models=tuple(sorted(models)),
        voting_statuses=voting_statuses,
        label_counts=label_counts,
        status_counts=status_counts,
        incomplete_count=incomplete_count,
        missing_by_model=missing_by_model,
        invalid_by_model=invalid_by_model,
    )
