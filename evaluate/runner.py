# encoding: utf8

"""五档同义判断评测：以 annotate/ 既有标注为参考，逐模型独立报告。

step 0：直接同义判断（默认）；1/2：一次完整标注，禁用/启用查表；
3/4：特征、图式、判定分步标注，禁用/启用查表。默认语料类型为方位词，
趋向动词须显式指定 --corpus-type direction，不从文件名推断。

用法：

    python evaluate --step 0 --corpus results/location_annotated.jsonl --models deepseek-flash
    python evaluate --step 3 --corpus results/direction_annotated.jsonl --corpus-type direction --models deepseek-flash qwen3.8-max
    python -m evaluate --step 4 --corpus results/location_annotated.jsonl --print-prompt

原始 SpaCE 句对（context1/context2/judge）先转成评测用 JSONL（样例文件不随仓库
分发，请把下面的路径换成你本地的原始句对文件）：

    python -m evaluate.import_sentence_pairs <原始句对文件>.json \\
        --output results/evaluate/sample_sentence_pairs.jsonl

每个模型独立写 ``results/evaluate/<模型名>_step<N>_<YYYYMMDDTHHMMSSZ>.json``；
时间戳为 UTC（ISO 8601 基本格式，精确到秒），同一天可多次运行。默认另写同名
``.details.jsonl`` 逐条预测（``--no-details`` 可关）。模型之间并行、单模型内部
按 ``--workers`` 并发；``--batch auto/native/off`` 与 annotate 同一套语义。
auto 只在原生 Batch **受理前**不可用时回退同步；一旦受理，失败/缺失项只记
错误，不在评测层偷偷同步重发（避免改变口径和重复计费）。

注意：参考答案来自 annotate/ 的既有标注，不是人工金标准；报告里称「与参考标注
的一致率」，不把准确率表述成真实语言能力。待测模型可能参与过参考投票，解读时
应保留这一点。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import config
import file_paths
import sync_cache
import temperature_policy as tp
from annotate.material_paths import annotation_material_paths
from corpus_metadata import normalize_corpus_type
from evaluate import data, metrics
from evaluate.inference import (
    ModelSpec,
    actual_mode,
    cache_sidecar_paths,
    public_model_info,
    public_reproducibility,
    run_model,
    sanitize_exception,
    sanitize_text,
)
from evaluate.progress import ProgressManager

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = _PROJECT_ROOT / "results" / "evaluate"
SCHEMA_VERSION = 3
REFERENCE_POLICY = (
    "单模型记录取 judgement.same_meaning（必须是真正的 bool）；"
    "多模型投票记录仅在 voting.status=accepted 时取顶层 voting_label 的 same/different；"
    "未采纳、平票、无有效票、discarded、非空间、不通顺、schema/fluent 标签与未判定一律跳过，"
    "不把投票代表记录里可能残留的 judgement 当成参考答案。"
)
EVALUATION_NOTE = (
    "参考答案来自 annotate/ 的既有标注，不是人工金标准；本报告衡量待测模型与参考标注的一致率。"
    "若待测模型同时参与过参考标注/投票，指标会偏乐观。"
)


@dataclass(frozen=True)
class ModelPlan:
    model: str
    report_path: Path
    details_path: Path | None
    cache_path: Path | None
    sync_path: Path | None = None


@dataclass
class RunResult:
    model: str
    report_path: Path
    report: dict
    error: str | None = None


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


def _corpus_type(value: str) -> str:
    # 只依赖 corpus_metadata 的纯函数，不导入 annotate.spatial_tools 及其 LTP/torch 链。
    try:
        return normalize_corpus_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_batch_args(parser: argparse.ArgumentParser) -> None:
    """与 annotate 相同的 --batch 参数族。

    这里刻意不 import annotate.spatial_runner：那条导入链会加载 LTP/torch
    （实测约 100 秒），仅仅为了取一个 argparse 辅助函数不值得。
    """
    group = parser.add_argument_group("批量推理")
    group.add_argument(
        "--batch",
        choices=("off", "auto", "native"),
        default=config.BATCH_MODE,
        help=f"批量推理：auto（默认 {config.BATCH_MODE}，受理前原生入口不可用则同步回退；"
        "受理后失败/缺失不再重发）／native（必须用原生，不可用即报错）／off（完全走同步链路）",
    )
    group.add_argument(
        "--batch-cache",
        default=None,
        help="批量结果 sidecar 路径（默认 <报告文件>.batch-cache.jsonl）；"
        "--batch-resume 时必须显式指定旧缓存路径",
    )
    group.add_argument(
        "--batch-resume",
        action="store_true",
        help="复用显式 --batch-cache 里的旧结果，并继续收 .jobs.json 里未终结的 batch；生成新时间戳报告",
    )
    group.add_argument(
        "--batch-poll-seconds",
        type=float,
        default=config.BATCH_POLL_SECONDS,
        help=f"轮询原生 batch 状态的间隔秒数（默认 {config.BATCH_POLL_SECONDS}）",
    )
    group.add_argument(
        "--batch-timeout-seconds",
        type=float,
        default=config.BATCH_TIMEOUT_SECONDS,
        help=f"等待单个原生 batch 完成的上限秒数（默认 {config.BATCH_TIMEOUT_SECONDS}，0 = 不限）",
    )
    group.add_argument(
        "--batch-chunk-size",
        type=int,
        default=config.BATCH_CHUNK_SIZE,
        help=f"一个原生 batch 最多包含多少条请求（默认 {config.BATCH_CHUNK_SIZE}）",
    )
    group.add_argument(
        "--sync-cache",
        default=None,
        help="可选的同步成功节点结果 sidecar 路径（默认不写）。与 --batch-cache 分开，"
        "只保存已通过业务校验的成功结果；配合 --sync-resume 可在中断后只补缺失节点。"
        "原生 Batch 成功结果不会写入本 sidecar，请继续用 --batch-resume",
    )
    group.add_argument(
        "--sync-resume",
        action="store_true",
        help="复用 --sync-cache 的旧成功结果并继续本轮；必须显式指定 --sync-cache",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluate",
        description="五档同义判断评测：用 annotate/ 的既有标注作参考，逐模型独立报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument(
        "--corpus",
        "--input",
        dest="corpus",
        required=True,
        help="annotate/ 输出的 JSONL 文件或目录（传目录时读取其中 *.jsonl；"
        "单模型结果、多模型投票结果均可）",
    )
    parser.add_argument(
        "--models",
        "--model",
        dest="models",
        nargs="+",
        metavar="模型",
        default=None,
        help="待测模型配置名（model_config/api_keys.json 里的键），可给多个分别评测；"
        "除 --print-prompt 外必填",
    )
    parser.add_argument(
        "--step", type=int, choices=(0, 1, 2, 3, 4), default=0,
        help="0=直接同义判断（默认）；1/2=一次完整标注（禁用/启用查表）；"
             "3/4=分步特征/图式/判定（禁用/启用查表）",
    )
    parser.add_argument(
        "--corpus-type", default="方位词", metavar="类型",
        help="默认方位词（localization/location/loc）；趋向动词须显式指定 "
             "direction/dir/direction_verb；不按文件名推断",
    )
    parser.add_argument("--workers", type=_positive_int, default=config.EVALUATE_WORKERS,
                        help=f"单个模型内部的并发数（默认 {config.EVALUATE_WORKERS}）")
    parser.add_argument(
        "--model-workers",
        type=_non_negative_int,
        default=config.EVALUATE_MODEL_WORKERS,
        help=f"同时评测几个模型；0 = 有几个跑几个（默认 {config.EVALUATE_MODEL_WORKERS}）",
    )
    parser.add_argument("--limit", type=_non_negative_int, default=None,
                        help="只评测筛选后有效记录的前多少条（默认全部；0 表示一条都不选）")
    parser.add_argument("--offset", type=_non_negative_int, default=0,
                        help="跳过筛选后有效记录的前多少条（默认 0）")
    parser.add_argument("--out-dir", default=None, help=f"报告目录（默认 {DEFAULT_OUT_DIR}）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名已有报告（默认拒绝）")
    parser.add_argument(
        "--details",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否另写同名 .details.jsonl 逐条预测（默认写；--no-details 关闭）",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否显示每个模型的推理进度（默认显示；--no-progress 关闭，"
             "不影响报告和诊断日志）",
    )
    parser.add_argument("--prompt-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--print-prompt", action="store_true",
                        help="离线打印本 step 的 agent 提示语与可渲染消息；"
                             "下游动态依赖只作说明，不读模型配置、不调用模型、不写文件")
    add_batch_args(parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prompt_dir is not None:
        parser.error("--prompt-dir 已停用：五档评测复用 annotate 的 agent 提示语/配置；"
                     "请用 --step、--corpus-type 和 --print-prompt 检查，不再加载旧 evaluate 模板")
    if not args.print_prompt and not args.models:
        parser.error("--models 至少需要一个模型；只有 --print-prompt 时可以省略")
    if args.batch_resume and not args.batch_cache:
        parser.error(
            "--batch-resume 必须显式指定 --batch-cache 的旧缓存路径；"
            "每次运行生成新时间戳，不能用新默认缓存代替续跑"
        )
    if args.sync_resume and not args.sync_cache:
        parser.error("--sync-resume 必须显式指定 --sync-cache 的旧缓存路径")
    try:
        args.corpus_type = _corpus_type(args.corpus_type)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    return args


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: str | Path) -> dict:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def build_spec(model: str, *, batch_mode: str) -> ModelSpec:
    """读取一个模型的配置/重试策略/批处理能力；配置错误在花钱前抛出。"""
    from agents import (
        disable_thinking_kwargs, load_model_config, proxy_policy, request_policy,
    )

    cfg = load_model_config(model)
    try:
        # 生成参数在创建报告占位和调用模型之前校验。畸形 disable_thinking
        # 若拖到报告阶段才报，会留下 0 字节报告并中止后续模型。
        disable_thinking_kwargs(cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"模型 {model!r} 的生成参数配置非法：{exc}") from None
    policy = request_policy(model)
    proxy_mode, _ = proxy_policy(cfg, model)
    profile = None
    if batch_mode != "off":
        import batch_inference as bi

        profile = bi.batch_profile(model)
        if batch_mode == "native" and not profile.enabled:
            raise ValueError(f"模型 {model!r} 的 --batch native 不可用：{profile.describe()}")
    return ModelSpec(key=model, cfg=cfg, policy=policy, proxy_mode=proxy_mode, batch=profile)


def model_info(spec: ModelSpec, *, batch_mode: str, actual_mode: str | None = None) -> dict:
    """报告里的模型信息固定白名单：只给复现所需，不复制配置字典。"""
    return public_model_info(spec, batch_mode=batch_mode, actual_mode=actual_mode)


def _fallback_model_info(spec: ModelSpec, error: str) -> dict:
    """模型元数据无法生成时仍保留固定白名单结构，不重试失败的配置辅助函数。"""
    return {
        "config_key": spec.key,
        "provider": None,
        "model_name": None,
        "endpoint": None,
        "temperature": None,
        "temperature_enabled": tp.enabled(),
        "proxy_mode": spec.proxy_mode,
        "request": {},
        "batch": {"enabled": False, "protocol": None, "endpoint": None, "disabled_reason": ""},
        "summary_error": error,
    }


def _safe_model_info(spec: ModelSpec, *, batch_mode: str, actual_mode: str | None = None) -> dict:
    """构造模型信息；失败时降级为最小白名单，绝不阻止错误报告落盘。"""
    try:
        return model_info(spec, batch_mode=batch_mode, actual_mode=actual_mode)
    except Exception as exc:  # noqa: BLE001 —— 元数据失败不能吞掉整份报告
        return _fallback_model_info(spec, sanitize_exception(exc, spec.cfg))


def _safe_reproducibility(
    spec: ModelSpec, pipeline: dict | None, *, batch_mode: str, workers: int,
    poll_seconds: float, timeout_seconds: float, chunk_size: int, outcome,
) -> dict:
    """构造复现信息；配置畸形或序列化失败时保留最小执行账，不阻塞报告。"""
    try:
        return public_reproducibility(
            spec, pipeline, batch_mode=batch_mode, workers=workers,
            poll_seconds=poll_seconds, timeout_seconds=timeout_seconds,
            chunk_size=chunk_size, outcome=outcome,
        )
    except Exception as exc:  # noqa: BLE001 —— 同 _safe_model_info
        error = sanitize_exception(exc, spec.cfg)
        try:
            actual = actual_mode(batch_mode, outcome)
        except Exception:  # noqa: BLE001 —— 只影响诊断字段
            actual = None
        return {
            "execution": {
                "requested_batch_mode": batch_mode,
                "actual_mode": actual,
                "fallback_reason": None,
                "workers": workers,
                "batch_poll_seconds": poll_seconds,
                "batch_timeout_seconds": timeout_seconds,
                "batch_chunk_size": chunk_size,
                "summary_error": error,
            },
            "invocation": {},
            "generation": {
                "thinking_policy_source": "unknown",
                "has_unpublished_parameters": True,
                "error": error,
            },
            "configuration_fingerprint": None,
        }


def _build_plans(models: list[str], out_dir: Path, args: argparse.Namespace, run_timestamp: str) -> list[ModelPlan]:
    from annotate.voting import safe_model_stem

    stems: dict[str, str] = {}
    plans = []
    multi_model = len(models) > 1
    for model in models:
        stem = safe_model_stem(model)
        collision_key = stem.lower()
        if collision_key in stems and stems[collision_key] != model:
            raise ValueError(
                f"模型名 {stems[collision_key]!r} 与 {model!r} 清洗后都变成 {stem!r}，"
                "会写同一个报告文件；请改用可区分的配置名"
            )
        stems[collision_key] = model
        report_path = out_dir / f"{stem}_step{args.step}_{run_timestamp}.json"
        details_path = report_path.with_suffix(".details.jsonl") if args.details else None
        cache_path = None
        if args.batch != "off":
            cache_path = file_paths.resolve_cache_path(
                args.batch_cache,
                report_path,
                model_key=stem if args.batch_cache and multi_model else None,
                multi_model=bool(args.batch_cache) and multi_model,
            )
        sync_path = None
        sync_cache_arg = getattr(args, "sync_cache", None)
        if sync_cache_arg:
            sync_path = file_paths.resolve_cache_path(
                sync_cache_arg,
                report_path,
                model_key=stem if multi_model else None,
                multi_model=multi_model,
            )
        plans.append(ModelPlan(model, report_path, details_path, cache_path, sync_path))
    return plans


def _outputs(plans: list[ModelPlan]) -> tuple[list[Path], list[Path]]:
    outputs: list[Path] = []
    sidecars: list[Path] = []
    for plan in plans:
        outputs.append(plan.report_path)
        if plan.details_path is not None:
            outputs.append(plan.details_path)
        if plan.cache_path is not None:
            sidecars.extend(cache_sidecar_paths(plan.cache_path))
            sidecars.append(Path(f"{plan.cache_path}.lock"))
        if plan.sync_path is not None:
            sidecars.extend(sync_cache.sidecar_paths(plan.sync_path))
    return outputs, sidecars


def _validate_paths(input_files: list[Path], plans: list[ModelPlan], overwrite: bool) -> None:
    outputs, sidecars = _outputs(plans)
    file_paths.validate_output_paths(input_files, outputs + sidecars)
    if not overwrite:
        for path in outputs:
            if path.exists():
                raise ValueError(
                    f"输出文件 {path} 已存在（同名输出默认拒绝覆盖）；"
                    "请换 --out-dir，或显式加 --overwrite"
                )
    for path in sidecars:
        if path.exists() and not path.is_file():
            raise ValueError(f"Batch 附属路径不是文件：{path}")


@contextlib.contextmanager
def _exclusive_lock(path: Path):
    """跨进程独占锁：同一份 Batch 缓存同一时间只允许一个评测进程。

    报告占位只能防同名报告；两个进程用不同 --out-dir、却指向同一个
    --batch-cache 时，仍会互相截断/追加缓存。这里用操作系统文件锁（Windows
    msvcrt / POSIX fcntl），进程退出时锁自动释放，不会留下需要手工删除的
    stale lock；锁文件本身保留是正常的。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                raise ValueError(
                    f"Batch 缓存 {path} 正被另一个评测进程使用；"
                    "请等它结束后重试，或换一个 --batch-cache"
                ) from None
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                raise ValueError(
                    f"Batch 缓存 {path} 正被另一个评测进程使用；"
                    "请等它结束后重试，或换一个 --batch-cache"
                ) from None
        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def _reserve_outputs(plans: list[ModelPlan], overwrite: bool) -> None:
    """在调用模型前独占创建报告/明细，堵住「exists 检查后才被另一进程抢先」的竞态。

    与 ResultWriter 的做法一致：先拿到空文件，随后用原子替换写入正式内容。
    ``--overwrite`` 时不预创建，保留旧文件直到收尾原子替换。
    """
    if overwrite:
        return
    for plan in plans:
        for path in (plan.report_path, plan.details_path):
            if path is not None:
                with file_paths.open_exclusive(path):
                    pass


def _select(dataset: data.Dataset, offset: int, limit: int | None) -> tuple[list[data.EvalSample], dict]:
    eligible = dataset.samples
    start = min(max(0, offset), len(eligible))
    end = len(eligible) if limit is None else min(start + limit, len(eligible))
    selected = eligible[start:end]
    return selected, {
        "offset": offset,
        "limit": limit,
        "eligible": len(eligible),
        "selected": len(selected),
        "unselected": len(eligible) - len(selected),
    }


def _prediction_for(outcome, sample: data.EvalSample) -> dict:
    def missing(message: str, kind: str) -> dict:
        return {
            "same_meaning": None, "label": None, "reason": "", "source": "error",
            "error": message, "error_kind": kind, "features": {}, "schema": {}, "history": [],
        }

    if outcome is None:
        return missing("模型未运行", "model_error")
    request = outcome.plan.sample_to_request.get(sample.key)
    if request is None:
        return missing("没有为该样本生成请求", "missing_result")
    return outcome.predictions.get(request) or missing("没有收到该请求的结果", "missing_result")


def _route_metadata(step: int, corpus_type: str, pipeline: dict | None) -> dict:
    """优先报告实际执行元数据；启动失败时仍注明请求的档位，不伪造 provenance。"""
    metadata = pipeline or {}
    return {
        "step": metadata.get("step", step),
        "mode": metadata.get("graph", "synonym" if step == 0 else "all_in_one" if step < 3 else "pair"),
        "allow_lookup": metadata.get("allow_lookup", step in (2, 4)),
        "corpus_type": metadata.get("corpus_type", corpus_type),
    }


def _sanitize_export(value, cfg: dict):
    """入口错误、统计与逐条中间结果均按相同文本规则递归脱敏。"""
    if isinstance(value, str):
        return sanitize_text(value, cfg)
    if isinstance(value, dict):
        return {key: _sanitize_export(item, cfg) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_export(item, cfg) for item in value]
    return value


_BATCH_EXECUTION_KEYS = (
    "mode", "waves", "items_collected", "items_completed", "items_failed",
    "fallback_items", "cache_hits", "batches", "cache_file", "jobs_file",
    "identity_file", "resume_blocked",
)


def _batch_execution(batch: dict | None) -> dict | None:
    """顶层只保留执行计数；模型 provider/端点等配置已由 model.batch 表示。"""
    if batch is None:
        return None
    return {key: batch.get(key) for key in _BATCH_EXECUTION_KEYS if key in batch}


def _build_report(
    spec: ModelSpec,
    plan: ModelPlan,
    input_info: list[dict],
    dataset: data.Dataset,
    selected: list[data.EvalSample],
    selection: dict,
    outcome,
    error: str | None,
    *,
    step: int,
    corpus_type: str,
    model: dict,
    reproducibility: dict,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
) -> dict:
    predictions = {sample.key: _prediction_for(outcome, sample) for sample in selected}
    pairs = [
        (sample.reference_label, predictions[sample.key].get("label"))
        for sample in selected
    ]
    # 模型根本没跑起来时不要构造一份「全错」的指标，否则 accuracy=0.0 会被误读成
    # 模型答错了全部样本；失败原因已由 error / failures_by_reason 说明。
    metric_block = None if error and outcome is None else metrics.compute_metrics(pairs)

    evaluated = sum(1 for item in predictions.values() if item.get("label"))
    failed = len(selected) - evaluated
    failure_reasons = Counter(
        item.get("error_kind") or "unknown"
        for item in predictions.values()
        if not item.get("label")
    )
    sources = Counter(sample.reference_source for sample in selected)
    reference_models = Counter(sample.reference_model or "unknown" for sample in selected)

    if error:
        run_status = "error"
    elif not selected:
        run_status = "no_samples"
    elif failed == 0:
        run_status = "ok"
    elif evaluated:
        run_status = "partial"
    else:
        run_status = "failed"

    pipeline = outcome.pipeline if outcome is not None else None
    prediction_sources = Counter(item.get("source") or "unknown" for item in predictions.values())

    return _sanitize_export({
        "schema_version": SCHEMA_VERSION,
        **_route_metadata(step, corpus_type, pipeline),
        "run_status": run_status,
        "model": model,
        "input": input_info,
        "pipeline": pipeline,
        "reproducibility": reproducibility,
        "agent_stats": outcome.agent_stats if outcome is not None else {},
        "prediction_sources": dict(prediction_sources.most_common()),
        "selection": selection,
        "counts": {
            "records_read": dataset.records_read,
            "eligible": len(dataset.samples),
            "selected": len(selected),
            "evaluated": evaluated,
            "failed": failed,
            "unique_requests": outcome.unique_requests if outcome is not None else 0,
            "skipped": dataset.skipped_total(),
            "duplicates": dataset.duplicates,
        },
        "skipped_by_reason": dict(dataset.skipped.most_common()),
        "reference": {
            "policy": REFERENCE_POLICY,
            "sources": dict(sources.most_common()),
            "models": dict(reference_models.most_common()),
        },
        "metrics": metric_block,
        "failures_by_reason": dict(failure_reasons.most_common()),
        "batch": _batch_execution(outcome.batch if outcome is not None else None),
        "batch_error": outcome.batch_error if outcome is not None else None,
        "sync_cache": outcome.sync_cache if outcome is not None else None,
        "sync_retries": outcome.sync_retries if outcome is not None else 0,
        "model_elapsed_seconds": outcome.elapsed_seconds if outcome is not None else None,
        "run": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
        },
        "outputs": {
            "report": str(plan.report_path),
            "details": str(plan.details_path) if plan.details_path is not None else None,
            "batch_cache": str(plan.cache_path) if plan.cache_path is not None else None,
            "sync_cache": str(plan.sync_path) if plan.sync_path is not None else None,
        },
        "evaluation_note": EVALUATION_NOTE,
        "error": error,
    }, spec.cfg)


def _detail_record(sample: data.EvalSample, prediction: dict, *, route: dict) -> dict:
    label = prediction.get("label")
    correct = None if label is None else label == sample.reference_label
    return {
        **route,
        "input_file": sample.input_file,
        "input_line_no": sample.input_line_no,
        "id": sample.record_id,
        "line_no": sample.line_no,
        "candidate": sample.candidate,
        "sentence_a": sample.sentence_a,
        "sentence_b": sample.sentence_b,
        "focus_word": sample.focus_word,
        "focus_word_b": sample.focus_word_b,
        "reference": sample.reference_label,
        "reference_source": sample.reference_source,
        "reference_model": sample.reference_model,
        "annotation_status": sample.annotation_status,
        "voting_status": sample.voting_status,
        "prediction": label,
        "same_meaning": prediction.get("same_meaning"),
        "source": prediction.get("source") or "unknown",
        "features": prediction.get("features") or {},
        "schema": prediction.get("schema") or {},
        "history": prediction.get("history") or [],
        "correct": correct,
        "status": "ok" if label else "error",
        "reason": prediction.get("reason") or "",
        "error": prediction.get("error"),
        "error_kind": prediction.get("error_kind"),
    }


def _write_details(path: Path, records: list[dict]) -> None:
    with file_paths.atomic_write_text(path) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_report(path: Path, report: dict) -> None:
    with file_paths.atomic_write_text(path) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def _run_one(
    spec: ModelSpec,
    plan: ModelPlan,
    input_info: list[dict],
    dataset: data.Dataset,
    selected: list[data.EvalSample],
    selection: dict,
    args: argparse.Namespace,
    progress: ProgressManager | None = None,
) -> RunResult:
    tracker = progress.start(spec.key) if progress is not None else None

    def log(message: str) -> None:
        text = f"[{spec.key}] {sanitize_text(message, spec.cfg)}"
        if progress is not None:
            progress.write(text)
        else:
            print(text, flush=True)

    started_wall = time.time()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    outcome = None
    error = None

    def invoke():
        return run_model(
            spec,
            selected,
            step=args.step,
            corpus_type=args.corpus_type,
            workers=args.workers,
            batch_mode=args.batch,
            cache_path=plan.cache_path,
            poll_seconds=args.batch_poll_seconds,
            timeout_seconds=args.batch_timeout_seconds,
            chunk_size=args.batch_chunk_size,
            resume=args.batch_resume,
            sync_cache_path=plan.sync_path,
            sync_resume=bool(getattr(args, "sync_resume", False)),
            log=log,
            on_progress=tracker.update if tracker is not None else None,
        )

    needs_cache_lock = (
        plan.cache_path is not None
        and spec.batch is not None
        and bool(getattr(spec.batch, "enabled", False))
    )
    try:
        if needs_cache_lock:
            with _exclusive_lock(Path(f"{plan.cache_path}.lock")):
                outcome = invoke()
        else:
            outcome = invoke()
    except Exception as exc:  # noqa: BLE001 —— 一个模型失败不能拖垮其他模型
        error = sanitize_exception(exc, spec.cfg)
        log(f"失败：{error[:300]}")
    finally:
        # 进度只覆盖推理阶段；报告构造/明细写出的日志仍走同一个安全写出。
        if tracker is not None:
            tracker.close()

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    pipeline = outcome.pipeline if outcome is not None else None
    try:
        route = _route_metadata(args.step, args.corpus_type, pipeline)
    except Exception as exc:  # noqa: BLE001 —— 路由诊断失败仍要保留错误报告
        error = error or sanitize_exception(exc, spec.cfg)
        route = {
            "step": args.step,
            "mode": "unknown",
            "allow_lookup": args.step in (2, 4),
            "corpus_type": args.corpus_type,
        }
    try:
        run_mode = actual_mode(args.batch, outcome)
    except Exception:  # noqa: BLE001 —— actual_mode 只用于诊断字段
        run_mode = None
    model_block = _safe_model_info(spec, batch_mode=args.batch, actual_mode=run_mode)
    reproducibility = _safe_reproducibility(
        spec, pipeline, batch_mode=args.batch, workers=args.workers,
        poll_seconds=args.batch_poll_seconds, timeout_seconds=args.batch_timeout_seconds,
        chunk_size=args.batch_chunk_size, outcome=outcome,
    )
    try:
        report = _build_report(
            spec,
            plan,
            input_info,
            dataset,
            selected,
            selection,
            outcome,
            error,
            step=args.step,
            corpus_type=args.corpus_type,
            model=model_block,
            reproducibility=reproducibility,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=round(time.time() - started_wall, 3),
        )
    except Exception as exc:  # noqa: BLE001 —— 报告构造失败也要留下可读记录
        error = error or sanitize_exception(exc, spec.cfg)
        report = {
            "schema_version": SCHEMA_VERSION,
            **route,
            "run_status": "error",
            "model": model_block,
            "input": input_info,
            "pipeline": pipeline,
            "reproducibility": reproducibility,
            "agent_stats": outcome.agent_stats if outcome is not None else {},
            "prediction_sources": {},
            "selection": selection,
            "counts": {"selected": len(selected)},
            "metrics": None,
            "outputs": {
                "report": str(plan.report_path),
                "details": str(plan.details_path) if plan.details_path is not None else None,
                "batch_cache": str(plan.cache_path) if plan.cache_path is not None else None,
                "sync_cache": str(plan.sync_path) if plan.sync_path is not None else None,
            },
            "evaluation_note": EVALUATION_NOTE,
            "error": error,
        }

    if plan.details_path is not None:
        try:
            records = [
                _sanitize_export(_detail_record(sample, _prediction_for(outcome, sample), route=route), spec.cfg)
                for sample in selected
            ]
            _write_details(plan.details_path, records)
        except Exception as exc:  # noqa: BLE001 —— 明细写失败仍保留汇总
            report["run_status"] = "error"
            report["error"] = sanitize_exception(exc, spec.cfg)
            report["outputs"]["details"] = None
            error = report["error"]  # 让 CLI 返回非零，而不是只写进报告
            log(f"明细写出失败：{report['error'][:300]}")

    try:
        report = _sanitize_export(report, spec.cfg)
        _write_report(plan.report_path, report)
    except Exception as exc:  # noqa: BLE001
        error = sanitize_exception(exc, spec.cfg)
        log(f"报告写出失败：{error[:300]}")

    return RunResult(model=spec.key, report_path=plan.report_path, report=report, error=error)


def print_numbered(title: str, text: str, write=print) -> None:
    write(f"\n===== {title}（共 {len(text.splitlines())} 行）=====")
    for index, line in enumerate(text.splitlines(), 1):
        write(f"{index:>4}| {line}")


def _print_prompt(args: argparse.Namespace, sample: data.EvalSample, write=print) -> None:
    from evaluate.annotation import build_pipeline

    def numbered(title: str, text: str) -> None:
        print_numbered(title, text, write)

    pipeline = build_pipeline(
        args.step, (args.models or ["preview"])[0], args.corpus_type, llm=False)
    state = pipeline.initial_state(sample)
    numbered("pipeline metadata（离线预览，不执行标注）",
             json.dumps(pipeline.metadata(), ensure_ascii=False, indent=2))
    for name, agent in pipeline.agents.items():
        numbered(f"{name} system prompt", agent.system_prompt)
        numbered(f"{name} 提示语/配置路径", json.dumps({
            "templates": agent.prompt_paths,
            "config": getattr(agent, "table_paths", None) or getattr(agent, "table_path", ""),
        }, ensure_ascii=False, indent=2))
        if name in ("synonym", "all_in_one"):
            content = agent._content(
                state["sentence_a"], state["sentence_b"], state["focus_word"], state["focus_word_b"])
            numbered(f"{name} user message（输入第 {sample.input_line_no} 行）", content)
        elif name == "feature":
            for side, focus_key in (("a", "focus_word"), ("b", "focus_word_b")):
                numbered(f"feature user message（{side} 侧）",
                         agent._content(state[f"sentence_{side}"], state[focus_key]))
        elif name == "schema":
            numbered("schema 动态依赖",
                     "两侧 user message 依赖本次 feature 生成的真实语义特征；"
                     "按本侧关注词选择示例和图式候选。离线预览不构造上游输出，不渲染此消息。"
                     + ("启用查表，命中时不调用模型。" if pipeline.allow_lookup
                        else "禁用特征匹配查表，采用模型结论并校验。"))
        elif name == "judge":
            numbered("judge 动态依赖",
                     "user message 依赖两侧真实 feature/schema 结果，示例按生成的图式对动态选择。"
                     "离线预览不构造上游输出，不渲染此消息。"
                     + ("启用判定查表，命中时不调用模型。" if pipeline.allow_lookup
                        else "禁用判定查表，采用模型结论并校验。"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # 所有终端输出（含预览、输入摘要和最终汇总）共用一套安全写出：
    # 下游提前关闭管道时只停止输出，不改变报告和退出码。
    progress = ProgressManager(enabled=getattr(args, "progress", True))

    if args.print_prompt:
        input_files = file_paths.collect_jsonl_inputs(args.corpus)
        dataset = data.collect_samples(input_files)
        selected, _ = _select(dataset, args.offset, args.limit)
        if not selected:
            raise ValueError("没有可评测样本（参考标签、句对或筛选条件不满足），无法预览提示词")
        _print_prompt(args, selected[0], progress.write)
        progress.finalize()
        return 0

    models = _dedupe(args.models or [])
    specs = [build_spec(model, batch_mode=args.batch) for model in models]
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out-dir 不是目录：{out_dir}")
    # UTC 基本格式不含 Windows 文件名禁用的冒号；精确到秒。
    # 只生成一次，确保同次运行的各模型及明细/缓存使用同一时间戳。
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plans = _build_plans(models, out_dir, args, run_timestamp)

    output_paths, sidecars = _outputs(plans)
    input_files = file_paths.collect_jsonl_inputs(args.corpus, output_paths=output_paths + sidecars)
    # 同步 sidecar 也会写文件：把评测实际读取的配置表/示例/提示语一并当作只读输入，
    # 防止 --sync-cache 指到它们时被当成缓存清空。报告里的输入清单仍只列语料。
    _validate_paths(
        [*input_files, *annotation_material_paths(args.corpus_type)], plans, args.overwrite)
    _reserve_outputs(plans, args.overwrite)
    dataset = data.collect_samples(input_files)
    selected, selection = _select(dataset, args.offset, args.limit)
    input_info = [_file_info(path) for path in input_files]

    progress.write(
        f"输入 {len(input_files)} 个文件，{dataset.records_read} 条记录；"
        f"可评测 {len(dataset.samples)} 条，选中 {len(selected)} 条，"
        f"跳过 {dataset.skipped_total()} 条"
    )
    if dataset.skipped:
        top = "，".join(f"{reason}={count}" for reason, count in dataset.skipped.most_common(6))
        progress.write(f"  跳过原因：{top}")
    if args.batch == "off" and args.batch_cache:
        progress.write("提示：--batch off 时 --batch-cache 不会使用")

    model_workers = len(specs) if args.model_workers == 0 else min(args.model_workers, len(specs))
    results: list[RunResult] = []
    if model_workers <= 1:
        for spec, plan in zip(specs, plans):
            results.append(_run_one(
                spec, plan, input_info, dataset, selected, selection, args, progress))
    else:
        with ThreadPoolExecutor(max_workers=model_workers) as pool:
            futures = {
                pool.submit(
                    _run_one, spec, plan, input_info, dataset, selected, selection, args, progress
                ): (spec, plan)
                for spec, plan in zip(specs, plans)
            }
            for future in as_completed(futures):
                spec, plan = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001 —— 一个模型的意外不能中断其他模型
                    error = sanitize_exception(exc, spec.cfg)
                    results.append(RunResult(
                        model=spec.key,
                        report_path=plan.report_path,
                        report={"run_status": "error", "error": error},
                        error=error,
                    ))

    by_model = {result.model: result for result in results}
    failed = False
    for model in models:
        result = by_model.get(model)
        if result is None:
            progress.write(f"[{model}] 没有返回结果")
            failed = True
            continue
        report = result.report
        if result.error:
            failed = True
            progress.write(f"[{model}] 失败：{result.error[:300]}；报告 {result.report_path}")
            continue
        status = report.get("run_status")
        metric = report.get("metrics") or {}
        if status == "no_samples":
            progress.write(f"[{model}] 没有可评测样本；报告 {result.report_path}")
            continue
        if status in ("partial", "failed", "error"):
            failed = True
        accuracy = metric.get("accuracy")
        accuracy_text = "null" if accuracy is None else f"{accuracy:.4f}"
        progress.write(
            f"[{model}] {status}：accuracy={accuracy_text} "
            f"({metric.get('correct')}/{metric.get('total')}，失败 {metric.get('failed')})；"
            f"报告 {result.report_path}"
        )
    progress.finalize()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
