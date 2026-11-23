# encoding: utf8

"""已评测明细的定点补跑：只重跑失败条，并给出补跑/合计两份统计。

输入是 ``python -m evaluate`` 写出的 ``<模型>_step<N>_<时间>.details.jsonl``。
默认只挑 ``status=error`` 或 ``prediction=None`` 的行重跑，原有成功行保持原样；
补跑结果不会改变原明细，而是另写：

    <原明细前缀>_rerun_<UTC 时间>.json          补跑汇总（含补跑前基线、补跑、合计）
    <原明细前缀>_rerun_<UTC 时间>.details.jsonl  逐条明细（原顺序，补跑结果已回填）

用法：

    python -m evaluate.rerun --details results/evaluate/glm_step0_20261123.details.jsonl \\
        --model glm --timeout-seconds 240

模型配置名优先取 ``--model``；省略时读取同名汇总报告里 ``model.config_key``。
``--timeout-seconds`` / ``--max-retries`` 只作用于本次补跑，不修改模型配置文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import config
import file_paths
from annotate.material_paths import annotation_material_paths
from corpus_metadata import normalize_corpus_type
from evaluate import data, metrics
from evaluate.inference import (
    actual_mode,
    cache_identity_path,
    run_model,
    sanitize_exception,
    sanitize_text,
)
from evaluate.progress import ProgressManager
from evaluate.runner import (
    EVALUATION_NOTE,
    ModelPlan,
    _exclusive_lock,
    _prediction_for,
    _reserve_outputs,
    _safe_model_info,
    _sanitize_export,
    _validate_paths,
    _write_details,
    _write_report,
    build_spec,
)

RERUN_SCHEMA_VERSION = 1
"""补跑报告自身的 schema 版本；与原评测报告的 SCHEMA_VERSION 无关。"""

REPORT_SUFFIX = ".details.jsonl"
"""原评测明细的固定后缀（见 evaluate.runner._build_plans）。"""

_CANDIDATE_STATE_PENDING = "pending"
_STATE_UNCHANGED = "unchanged"
_STATE_SUCCESS = "success"
_STATE_FAILED = "failed"
_STATE_SKIPPED = "skipped"
_STATE_NOT_STARTED = "not_started"
_STATE_UNKNOWN = "unknown"
"""Batch 提交后写盘失败等：可能已经发过请求，但拿不到 outcome。"""


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"需要数字，收到 {value!r}") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(f"必须是有限正数，收到 {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"需要整数，收到 {value!r}") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"不能为负数，收到 {parsed}")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _non_negative_int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须 >= 1")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"需要数字，收到 {value!r}") from None
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(f"必须是非负有限数，收到 {value!r}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluate.rerun",
        description="只重跑原评测明细里失败的条，并写补跑/合计两份统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument(
        "--details", required=True,
        help="原评测写出的 .details.jsonl（必须是单文件；不会就地修改）",
    )
    parser.add_argument(
        "--model", default=None,
        help="待测模型配置名；省略时读取同名汇总报告里的 model.config_key",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="补跑输出目录（默认与输入明细同目录）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="允许覆盖同名补跑输出（默认拒绝；输入明细本身始终不会被覆盖）",
    )
    parser.add_argument(
        "--workers", type=_positive_int, default=config.EVALUATE_WORKERS,
        help=f"补跑并发数（默认 {config.EVALUATE_WORKERS}）",
    )
    parser.add_argument(
        "--timeout-seconds", type=_positive_float, default=None,
        help="本次请求的 SDK 超时秒数（默认沿用模型配置）；"
             "原 120 秒超时的失败建议先用 180–300",
    )
    parser.add_argument(
        "--max-retries", type=_non_negative_int, default=None,
        help="首次请求之外最多再试几次（默认沿用模型配置）；只影响本次补跑",
    )
    batch = parser.add_argument_group("批量推理")
    batch.add_argument(
        "--batch", choices=("off", "auto", "native"), default="off",
        help="批量推理模式（默认 off；补跑通常只有少量条，同步最直接）",
    )
    batch.add_argument("--batch-cache", default=None, help="批量结果 sidecar 路径（默认 <报告文件>.batch-cache.jsonl）")
    batch.add_argument(
        "--batch-poll-seconds", type=_non_negative_float, default=config.BATCH_POLL_SECONDS,
        help=f"轮询原生 batch 状态的间隔秒数（默认 {config.BATCH_POLL_SECONDS}）",
    )
    batch.add_argument(
        "--batch-timeout-seconds", type=_non_negative_float, default=config.BATCH_TIMEOUT_SECONDS,
        help=f"等待单个原生 batch 完成的上限秒数（默认 {config.BATCH_TIMEOUT_SECONDS}，0 = 不限）",
    )
    batch.add_argument(
        "--batch-chunk-size", type=_positive_int, default=config.BATCH_CHUNK_SIZE,
        help=f"一个原生 batch 最多包含多少条请求（默认 {config.BATCH_CHUNK_SIZE}）",
    )
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=True,
        help="是否显示补跑进度（默认显示；--no-progress 关闭）",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# 读取明细与路由
# ---------------------------------------------------------------------------


@dataclass
class DetailRow:
    """一行明细在补跑过程中的全部状态。"""

    line_no: int
    record: dict | None
    error_kind: str | None
    error: str | None
    reference: str | None = None
    candidate: bool = False
    state: str = _STATE_UNCHANGED
    reason: str = ""
    route_error: str | None = None
    sample: data.EvalSample | None = None
    prediction: dict | None = None

    @property
    def previous_status(self) -> str | None:
        return self.record.get("status") if self.record is not None else None

    @property
    def previous_prediction(self) -> str | None:
        value = self.record.get("prediction") if self.record is not None else None
        return value if value in data.LABELS else None

    @property
    def previous_error_kind(self) -> str | None:
        return self.record.get("error_kind") if self.record is not None else self.error_kind


@dataclass(frozen=True)
class Route:
    step: int
    corpus_type: str
    mode: str
    allow_lookup: bool

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "corpus_type": self.corpus_type,
            "mode": self.mode,
            "allow_lookup": self.allow_lookup,
        }


_EXPECTED_ROUTE = {
    0: ("synonym", False),
    1: ("all_in_one", False),
    2: ("all_in_one", True),
    3: ("pair", False),
    4: ("pair", True),
}
"""step -> 预期的 (graph mode, allow_lookup)；与 evaluate.annotation 的五档一致。"""


def _route_from_record(record: dict) -> Route:
    """从一条明细里取路由；缺字段或值非法时抛 ValueError（不猜默认档位）。"""
    step = record.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= 4:
        raise ValueError(f"step 必须是 0–4 的整数，收到 {step!r}")
    corpus_raw = record.get("corpus_type")
    if not isinstance(corpus_raw, str) or not corpus_raw.strip():
        raise ValueError(f"corpus_type 缺失或不是字符串，收到 {corpus_raw!r}")
    corpus_type = normalize_corpus_type(corpus_raw)
    mode = record.get("mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError(f"mode 缺失或不是字符串，收到 {mode!r}")
    allow_lookup = record.get("allow_lookup")
    if not isinstance(allow_lookup, bool):
        raise ValueError(f"allow_lookup 必须是布尔值，收到 {allow_lookup!r}")
    expected_mode, expected_lookup = _EXPECTED_ROUTE[step]
    if (mode, allow_lookup) != (expected_mode, expected_lookup):
        raise ValueError(
            f"step={step} 的预期路由是 mode={expected_mode} allow_lookup={expected_lookup}，"
            f"明细里是 mode={mode} allow_lookup={allow_lookup}"
        )
    return Route(step, corpus_type, mode, allow_lookup)


def _load_rows(details: Path) -> tuple[list[DetailRow], dict]:
    """从同一份字节快照解析行并计算哈希，报告里不会出现「旧行 + 新哈希」。"""
    raw = details.read_bytes()
    rows: list[DetailRow] = []
    for record in data.read_records(details, text=raw.decode("utf-8-sig")):
        rows.append(DetailRow(
            line_no=record.input_line_no,
            record=record.record,
            error_kind=record.error_kind,
            error=record.error,
        ))
    return rows, {
        "path": str(details.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rows": len(rows),
    }


def _resolve_route(rows: list[DetailRow]) -> Route:
    """整份明细只能有一个路由；单行路由坏掉只跳过该行的候选。"""
    routes: set[Route] = set()
    for row in rows:
        if row.record is None:
            continue
        try:
            routes.add(_route_from_record(row.record))
        except ValueError as exc:
            row.route_error = str(exc)
    if not routes:
        raise ValueError("明细里没有可识别的评测路由（step/corpus_type/mode/allow_lookup）")
    if len(routes) > 1:
        described = "；".join(
            f"step={route.step} corpus_type={route.corpus_type} mode={route.mode} "
            f"allow_lookup={route.allow_lookup}"
            for route in sorted(routes, key=lambda item: (item.step, item.corpus_type, item.mode))
        )
        raise ValueError(f"明细里混入了多个评测路由（{described}），无法确定补跑用哪一档")
    return routes.pop()


def _sibling_report(details: Path) -> Path | None:
    """`X.details.jsonl` 对应的原汇总报告 `X.json`。"""
    if not details.name.endswith(REPORT_SUFFIX):
        return None
    return details.with_name(details.name[: -len(REPORT_SUFFIX)] + ".json")


def _reported_batch_cache(report: Path | None) -> Path | None:
    """原报告里记录过的 Batch 缓存路径；用于把它当只读输入保护起来。"""
    if report is None or not report.is_file():
        return None
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    outputs = payload.get("outputs") if isinstance(payload, dict) else None
    cache = outputs.get("batch_cache") if isinstance(outputs, dict) else None
    if not isinstance(cache, str) or not cache.strip():
        return None
    return Path(cache)


def _cache_group(cache: Path) -> list[Path]:
    """一个 Batch 缓存会写/读的全部附属路径（含锁与评测身份文件）。"""
    return [*file_paths.sidecar_paths(cache), cache_identity_path(cache), Path(f"{cache}.lock")]


def _existing_cache_artifacts(cache: Path) -> list[Path]:
    """缓存组里已存在的旧产物；锁文件本身不算（它长期保留，进程退出自动解锁）。"""
    lock = Path(f"{cache}.lock")
    return [path for path in _cache_group(cache) if path != lock and path.exists()]


def _assert_fresh_cache(cache: Path) -> None:
    """在持有缓存锁之后再确认一次：本进程要用 resume=False 新建缓存。

    预先检查只能挡住「运行前就已存在」；两个进程用不同报告名、同一个显式
    --batch-cache 时，第二个进程必须等前一个释放锁后再查，否则会清空刚写好的结果。
    """
    existing = _existing_cache_artifacts(cache)
    if existing:
        shown = "、".join(str(path) for path in existing[:3])
        raise ValueError(
            f"--batch-cache 指向已有缓存（{shown}），补跑不复用它且会清空旧结果；"
            "请换一个新的缓存路径，或确认后手工删除"
        )


def _resolve_model(explicit: str | None, details: Path) -> str:
    report = _sibling_report(details)
    reported = None
    if report is not None and report.is_file():
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
            model = payload.get("model") if isinstance(payload, dict) else None
            value = model.get("config_key") if isinstance(model, dict) else None
            if isinstance(value, str) and value.strip():
                reported = value
        except (OSError, ValueError):
            reported = None
    if explicit and reported and explicit != reported:
        raise ValueError(
            f"--model {explicit!r} 与同名报告里的 model.config_key {reported!r} 不一致；"
            "请确认要补跑的是哪一份明细"
        )
    model = explicit or reported
    if not model:
        raise ValueError(
            "无法确定待测模型：请显式给 --model，或保证同目录存在原汇总报告 "
            f"{report or '<明细前缀>.json'} 且包含 model.config_key"
        )
    return model


# ---------------------------------------------------------------------------
# 候选分类与样本重建
# ---------------------------------------------------------------------------


def _classify(rows: list[DetailRow], digest: str, details: Path) -> list[data.EvalSample]:
    """标记每行状态，返回需要真正发给模型的样本（保持原明细顺序）。"""
    samples: list[data.EvalSample] = []
    for row in rows:
        if row.record is None:
            row.state = _STATE_SKIPPED
            row.reason = row.error_kind or data.SKIP_INVALID_RECORD
            continue

        record = row.record
        reference = record.get("reference")
        row.reference = reference if reference in data.LABELS else None

        row.candidate = record.get("status") == "error" or record.get("prediction") is None
        if not row.candidate:
            row.state = _STATE_UNCHANGED
            continue

        if row.route_error is not None:
            row.state = _STATE_SKIPPED
            row.reason = "invalid_route"
            continue
        if row.reference is None:
            row.state = _STATE_SKIPPED
            row.reason = "invalid_reference"
            continue
        sentence_a, sentence_b = record.get("sentence_a"), record.get("sentence_b")
        if (not isinstance(sentence_a, str) or not sentence_a.strip()
                or not isinstance(sentence_b, str) or not sentence_b.strip()):
            row.state = _STATE_SKIPPED
            row.reason = data.SKIP_INVALID_PAIR
            continue

        raw_id = record.get("id")
        line_no = record.get("line_no")
        input_line_no = record.get("input_line_no")
        input_file = record.get("input_file")
        row.sample = data.EvalSample(
            # 原 EvalSample.key 没有写进明细；用「输入指纹 + 明细物理行号」保证可复算、
            # 且同一行重复补跑不会换键。
            key=f"{digest}:{row.line_no}",
            input_file=input_file if isinstance(input_file, str) else str(details),
            input_line_no=input_line_no if isinstance(input_line_no, int) else row.line_no,
            record_id=str(raw_id) if raw_id is not None else None,
            line_no=line_no if isinstance(line_no, int) else None,
            sentence_a=sentence_a.strip(),
            sentence_b=sentence_b.strip(),
            focus_word=record.get("focus_word"),
            focus_word_b=record.get("focus_word_b"),
            reference_label=row.reference,
            reference_source=record.get("reference_source") or "details",
            reference_model=record.get("reference_model") if isinstance(record.get("reference_model"), str) else None,
            annotation_status=record.get("annotation_status") if isinstance(record.get("annotation_status"), str) else None,
            voting_status=record.get("voting_status") if isinstance(record.get("voting_status"), str) else None,
            candidate=record.get("candidate") if isinstance(record.get("candidate"), dict) else None,
        )
        row.state = _CANDIDATE_STATE_PENDING
        samples.append(row.sample)
    return samples


# ---------------------------------------------------------------------------
# 结果合并与计分
# ---------------------------------------------------------------------------


def _prediction_fields(prediction: dict, reference: str | None) -> dict:
    label = prediction.get("label")
    return {
        "prediction": label,
        "same_meaning": prediction.get("same_meaning"),
        "source": prediction.get("source") or "unknown",
        "features": prediction.get("features") or {},
        "schema": prediction.get("schema") or {},
        "history": prediction.get("history") or [],
        "correct": None if label is None or reference is None else label == reference,
        "status": "ok" if label else "error",
        "reason": prediction.get("reason") or "",
        "error": prediction.get("error"),
        "error_kind": prediction.get("error_kind"),
    }


def _merged_record(row: DetailRow, details: Path) -> dict:
    """保留原行与所有扩展字段，只替换预测相关字段并附 rerun 诊断块。"""
    if row.record is None:
        merged = {
            "input_file": str(details),
            "input_line_no": row.line_no,
            "id": None,
            "prediction": None,
            "same_meaning": None,
            "status": "error",
            "reason": row.error or "",
            "error": row.error,
            "error_kind": row.error_kind,
        }
    else:
        merged = dict(row.record)
        if row.prediction is not None:
            merged.update(_prediction_fields(row.prediction, row.reference))
    merged["rerun"] = {
        "line": row.line_no,
        "candidate": row.candidate,
        "state": row.state,
        "reason": row.reason,
        "previous_status": row.previous_status,
        "previous_prediction": row.previous_prediction,
        "previous_error_kind": row.previous_error_kind,
    }
    return merged


def _metric_label(value) -> str | None:
    return value if value in data.LABELS else None


def _pairs(records: list[dict]) -> tuple[list[tuple[str, str | None]], Counter]:
    """按合并后的行构造指标对；无合法参考的行计入不可评分。"""
    pairs: list[tuple[str, str | None]] = []
    unscorable: Counter = Counter()
    for record in records:
        reference = record.get("reference")
        if reference not in data.LABELS:
            rerun = record.get("rerun") if isinstance(record.get("rerun"), dict) else {}
            unscorable[rerun.get("reason") or "invalid_reference"] += 1
            continue
        pairs.append((reference, _metric_label(record.get("prediction"))))
    return pairs, unscorable


def _original_records(rows: list[DetailRow]) -> tuple[list[dict], Counter]:
    records = []
    unscorable: Counter = Counter()
    for row in rows:
        if row.record is None:
            unscorable[row.error_kind or data.SKIP_INVALID_RECORD] += 1
            continue
        reference = row.record.get("reference")
        if reference not in data.LABELS:
            unscorable["invalid_reference"] += 1
            continue
        records.append({
            "reference": reference,
            "prediction": row.record.get("prediction"),
        })
    return records, unscorable


def _change_counts(rows: list[DetailRow], merged: list[dict]) -> dict:
    improved = regressed = 0
    for row, record in zip(rows, merged):
        reference = record.get("reference")
        if reference not in data.LABELS:
            continue
        before = row.previous_prediction
        after = record.get("prediction") if record.get("prediction") in data.LABELS else None
        if before != reference and after == reference:
            improved += 1
        elif before == reference and after != reference:
            regressed += 1
    return {"improved": improved, "regressed": regressed}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _run_and_merge(
    rows: list[DetailRow],
    samples: list[data.EvalSample],
    spec,
    route: Route,
    plan: ModelPlan,
    args: argparse.Namespace,
    progress: ProgressManager,
) -> tuple[dict | None, str | None, float | None, dict | None, bool]:
    """执行补跑并把预测回填到行上。

    Returns:
        (outcome, error, elapsed, batch 摘要, 重试是否可能重复计费)。
        最后一个布尔值只在结果不完整时为 True：它表示「可能有已受理但未对账的
        请求」，再次补跑前应先到供应商控制台确认。
    """
    if not samples:
        return None, None, None, None, False

    def log(message: str) -> None:
        progress.write(f"[rerun] {sanitize_text(message, spec.cfg)}")

    tracker = progress.start(spec.key)
    outcome = None
    error = None
    elapsed = None
    invoke_started = False

    def invoke():
        nonlocal invoke_started
        invoke_started = True
        return run_model(
            spec,
            samples,
            step=route.step,
            corpus_type=route.corpus_type,
            workers=args.workers,
            batch_mode=args.batch,
            cache_path=plan.cache_path,
            poll_seconds=args.batch_poll_seconds,
            timeout_seconds=args.batch_timeout_seconds,
            chunk_size=args.batch_chunk_size,
            resume=False,
            log=log,
            on_progress=tracker.update,
        )

    needs_cache_lock = (
        plan.cache_path is not None
        and spec.batch is not None
        and bool(getattr(spec.batch, "enabled", False))
    )
    try:
        if needs_cache_lock:
            with _exclusive_lock(Path(f"{plan.cache_path}.lock")):
                # 持有锁后再查一次：否则两个进程用不同报告路径、同一缓存路径时，
                # 后者会在前者释放锁之后把刚写好的缓存当成「新路径」清空。
                _assert_fresh_cache(plan.cache_path)
                outcome = invoke()
        else:
            outcome = invoke()
        elapsed = outcome.elapsed_seconds
    except Exception as exc:  # noqa: BLE001 —— 失败要落进报告，不能只留 traceback
        error = sanitize_exception(exc, spec.cfg)
        log(f"失败：{error[:300]}")
    finally:
        tracker.close()

    by_sample = {row.sample.key: row for row in rows if row.sample is not None}
    if outcome is None:
        # 抢锁/锁内预检失败发生在 invoke() 之前，一条请求都没发；invoke() 一旦
        # 开始就可能已经发出（同步线程池或 Batch 提交），只能保守记为「未知」。
        state = _STATE_UNKNOWN if invoke_started else _STATE_NOT_STARTED
        reason = "run_error_unknown" if invoke_started else "run_not_started"
        for sample in samples:
            row = by_sample[sample.key]
            row.state = state
            row.reason = reason
        return None, error, elapsed, None, invoke_started
    for sample in samples:
        row = by_sample[sample.key]
        row.prediction = _prediction_for(outcome, sample)
        row.state = _STATE_SUCCESS if row.prediction.get("label") else _STATE_FAILED
    batch = outcome.batch
    # Batch 已受理但结果缺失 / 受理状态不明时，失败行再次补跑可能重复计费；
    # 预受理失败回退同步（batches=0）则没有这个风险。
    accepted = bool(batch and (batch.get("batches") or batch.get("items_collected")))
    duplicate_risk = bool(
        batch and batch.get("resume_blocked")
    ) or bool(outcome.batch_error and accepted)
    return outcome, error, elapsed, batch, duplicate_risk


def run(args: argparse.Namespace) -> int:
    details = Path(args.details)
    if not details.is_file():
        raise ValueError(f"--details 不是文件：{details}")

    progress = ProgressManager(enabled=args.progress)
    try:
        rows, details_info = _load_rows(details)
        if not rows:
            raise ValueError(f"明细里没有任何非空记录：{details}")
        route = _resolve_route(rows)
        model = _resolve_model(args.model, details)
        digest = details_info["sha256"]
        samples = _classify(rows, digest, details)

        # 先建 spec：模型配置/生成参数非法时在创建任何输出文件之前退出。
        spec = build_spec(model, batch_mode=args.batch)
        configured_policy = dict(spec.policy)
        policy = dict(spec.policy)
        if args.timeout_seconds is not None:
            policy["timeout"] = float(args.timeout_seconds)
        if args.max_retries is not None:
            policy["max_retries"] = int(args.max_retries)
        if policy != spec.policy:
            spec = replace(spec, policy=policy)

        out_dir = Path(args.out_dir) if args.out_dir else details.parent
        if out_dir.exists() and not out_dir.is_dir():
            raise ValueError(f"--out-dir 不是目录：{out_dir}")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = details.name[: -len(REPORT_SUFFIX)] if details.name.endswith(REPORT_SUFFIX) else details.stem
        report_path = out_dir / f"{stem}_rerun_{timestamp}.json"
        details_path = out_dir / f"{stem}_rerun_{timestamp}.details.jsonl"
        cache_path = None
        if args.batch != "off":
            cache_path = file_paths.resolve_cache_path(args.batch_cache, report_path)
        plan = ModelPlan(model, report_path, details_path, cache_path)

        input_files = [details]
        sibling = _sibling_report(details)
        if sibling is not None and sibling.is_file() and sibling != details:
            input_files.append(sibling)
        # 原报告指向的 Batch 缓存属于输入证据，补跑一律当只读路径保护；
        # 绝不能因为 --overwrite 或路径别名把它当成可写输出。
        old_cache = _reported_batch_cache(sibling)
        if old_cache is not None:
            input_files.extend(_cache_group(old_cache))
        _validate_paths(
            [*input_files, *annotation_material_paths(route.corpus_type)], [plan], args.overwrite)

        if cache_path is not None:
            # 补跑固定 resume=False；指向任何已存在的缓存组都会把它截断/清理。
            # 持锁后的 _run_and_merge 里还会再查一次，堵住两个进程的竞态。
            _assert_fresh_cache(cache_path)
        _reserve_outputs([plan], args.overwrite)

        progress.write(
            f"输入 {len(rows)} 行，候选 {sum(1 for row in rows if row.candidate)} 条，"
            f"实际补跑 {len(samples)} 条；模型 {model}，路由 step={route.step} "
            f"{route.corpus_type}，timeout={policy.get('timeout')}s "
            f"max_retries={policy.get('max_retries')}"
        )
        if args.batch == "off" and args.batch_cache:
            progress.write("提示：--batch off 时 --batch-cache 不会使用")

        started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        wall = datetime.now(timezone.utc)
        outcome, error, elapsed, batch, duplicate_risk = _run_and_merge(
            rows, samples, spec, route, plan, args, progress)
        finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
        total_elapsed = round((datetime.now(timezone.utc) - wall).total_seconds(), 3)

        merged = [_merged_record(row, details) for row in rows]

        original_records, original_unscorable = _original_records(rows)
        original_metrics = metrics.compute_metrics(
            [(record["reference"], record["prediction"] if record["prediction"] in data.LABELS else None)
             for record in original_records]
        )

        attempted = [row for row in rows if row.state in (_STATE_SUCCESS, _STATE_FAILED)]
        rerun_pairs = [
            (row.reference, _metric_label(row.prediction.get("label") if row.prediction else None))
            for row in attempted
        ]
        rerun_metrics = metrics.compute_metrics(rerun_pairs)
        not_started = [row for row in rows if row.state == _STATE_NOT_STARTED]
        unknown = [row for row in rows if row.state == _STATE_UNKNOWN]
        skipped = [row for row in rows if row.state == _STATE_SKIPPED and row.candidate]
        unattempted = [*skipped, *not_started, *unknown]

        total_pairs, total_unscorable = _pairs(merged)
        total_metrics = metrics.compute_metrics(total_pairs)
        changes = _change_counts(rows, merged)

        candidates = sum(1 for row in rows if row.candidate)
        successes = sum(1 for row in rows if row.state == _STATE_SUCCESS)
        failures = sum(1 for row in rows if row.state == _STATE_FAILED)
        if error:
            run_status = "error"
        elif candidates == 0:
            run_status = "no_candidates"
        elif not attempted:
            run_status = "skipped"
        elif failures == 0 and not skipped:
            run_status = "ok"
        elif successes:
            run_status = "partial"
        else:
            run_status = "failed"

        errors_by_kind = Counter(
            row.prediction.get("error_kind") or "unknown"
            for row in rows
            if row.state == _STATE_FAILED
        )
        try:
            mode = actual_mode(args.batch, outcome)
        except Exception:  # noqa: BLE001 —— 只影响诊断字段
            mode = None
        model_block = _safe_model_info(spec, batch_mode=args.batch, actual_mode=mode)

        report = {
            "schema_version": RERUN_SCHEMA_VERSION,
            "kind": "evaluate_rerun",
            "run_status": run_status,
            "input": {
                **details_info,
                "rows": len(rows),
                "report": str(sibling) if sibling is not None and sibling.is_file() else None,
            },
            "model": model_block,
            "route": route.as_dict(),
            "run": {
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": total_elapsed,
                "model_elapsed_seconds": elapsed,
                "timeout_seconds": policy.get("timeout"),
                "max_retries": policy.get("max_retries"),
                "configured_timeout_seconds": configured_policy.get("timeout"),
                "configured_max_retries": configured_policy.get("max_retries"),
                "batch": args.batch,
                "actual_mode": mode,
                "requests_may_have_been_sent": bool(duplicate_risk),
                "workers": args.workers,
                "batch_poll_seconds": args.batch_poll_seconds,
                "batch_timeout_seconds": args.batch_timeout_seconds,
                "batch_chunk_size": args.batch_chunk_size,
            },
            "original": {
                "metrics": original_metrics,
                "unscorable": sum(original_unscorable.values()),
                "unscorable_by_reason": dict(original_unscorable.most_common()),
            },
            "rerun": {
                "candidates": candidates,
                "attempted": len(attempted),
                "success": successes,
                "failed": failures,
                "skipped": len(unattempted),
                "not_started": len(not_started),
                "unknown": len(unknown),
                "skipped_by_reason": dict(Counter(row.reason for row in unattempted).most_common()),
                "errors_by_kind": dict(errors_by_kind.most_common()),
                "metrics": rerun_metrics,
                "elapsed_seconds": elapsed,
            },
            "total": {
                "metrics": total_metrics,
                "unscorable": sum(total_unscorable.values()),
                "unscorable_by_reason": dict(total_unscorable.most_common()),
                **changes,
            },
            "batch": batch,
            "outputs": {
                "report": str(report_path),
                "details": str(details_path),
                "batch_cache": str(cache_path) if cache_path is not None else None,
            },
            "evaluation_note": EVALUATION_NOTE,
            "error": error,
        }

        depth_ok = f"rerun {successes}成功/{failures}失败，合计正确 {total_metrics.get('correct')}/{total_metrics.get('total')}"
        progress.write(
            f"[{model}] {run_status}：{depth_ok}"
            f"（补跑前 {original_metrics.get('correct')}/{original_metrics.get('total')}）；"
            f"报告 {report_path}"
        )

        records = [_sanitize_export(record, spec.cfg) for record in merged]
        try:
            _write_details(details_path, records)
        except Exception as exc:  # noqa: BLE001 —— 明细失败仍要留下汇总
            error = sanitize_exception(exc, spec.cfg)
            report["run_status"] = "error"
            report["error"] = error
            report["outputs"]["details"] = None
            run_status = "error"
            progress.write(f"[{model}] 明细写出失败：{error[:300]}")
        try:
            _write_report(report_path, _sanitize_export(report, spec.cfg))
        except Exception as exc:  # noqa: BLE001 —— 只能靠终端提示，仍返回非零
            error = sanitize_exception(exc, spec.cfg)
            run_status = "error"
            progress.write(f"[{model}] 报告写出失败：{error[:300]}")
    finally:
        progress.finalize()

    return 1 if run_status in ("error", "failed", "partial") else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        print(f"补跑失败：{exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
