# encoding: utf8

"""补判 extract 标准输出中指定类型的失败材料。

输入是 ``extract`` 模块产出的标准 JSONL。脚本按原 verdict 的 index/word 重新送
LLM，输出是“原有结果 + 后续补充结果”的**总集**：输入有多少行，输出就有多少行，
顺序不变；未命中的、补判后仍失败的和 limit 跳过 的行都按原字节保留。mixed 记录里
已经成功的 verdict 不会重跑。

错误类型由 ``ERROR_RULES`` 注册表选择，默认只处理 402 / Insufficient Balance /
余额不足。后续增加错误类型时，只需增加注册项；核心读写与补判流程与具体错误文案无关。

用法：:

    python extract/rescreen.py 输入.jsonl \
        -f config_files/localization_expressions.json5 \
        -o 输入.rescreened.jsonl \
        --screen-model deepseek-flash \
        --screen-workers 8 --screen-batch-size 20

    # 显式补判其他错误类型（可重复或逗号分隔）
    python extract/rescreen.py 输入.jsonl -f ... -o ... \
        --error-type timeout,http_5xx

    # 自定义关键词（与 --error-type 取并集）
    python extract/rescreen.py 输入.jsonl -f ... -o ... \
        --error-keyword 'rate limit'

本地模型（配置里带 local_service）会复用已启动的服务；服务未启动时由本程序拉起，
并且只关闭本程序自己拉起的服务。当前脚本与 ``python -m extract`` 主流程完全隔离。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from extract import local_model_service, sentence_screening
from file_paths import atomic_write_text, validate_output_paths


@dataclass(frozen=True)
class ErrorRule:
    """一种可补判的错误类型：只匹配 reason 文本，不扫描句子正文。"""

    name: str
    description: str
    pattern: re.Pattern[str]


ERROR_RULES: dict[str, ErrorRule] = {
    "insufficient_balance": ErrorRule(
        "insufficient_balance",
        "402 / Insufficient Balance / 余额不足",
        re.compile(r"(?<!\d)402(?!\d)|insufficient\s+balance|余额不足", re.IGNORECASE),
    ),
    "timeout": ErrorRule(
        "timeout",
        "timeout / timed out / 超时",
        re.compile(r"timeout|timed\s*out|超时", re.IGNORECASE),
    ),
    "http_5xx": ErrorRule(
        "http_5xx",
        "HTTP 5xx / Internal Server Error / Service Unavailable 等",
        re.compile(
            r"(?:error\s*code|http(?:\s*status)?|status(?:\s*code)?|错误码|状态码)"
            r"[^\n]{0,40}\b5\d{2}\b"
            r"|internal\s+server\s+error|service\s+unavailable|bad\s+gateway|gateway\s+timeout",
            re.IGNORECASE,
        ),
    ),
    "incomplete_batch": ErrorRule(
        "incomplete_batch",
        "批量响应不完整",
        re.compile(r"批量响应不完整"),
    ),
    "missing_fields": ErrorRule(
        "missing_fields",
        "模型未完整给出语法和空间自洽判定",
        re.compile(r"模型未完整给出语法和空间自洽判定"),
    ),
}
"""内置错误类型。默认只用 insufficient_balance；新增类型只需在这里加一项。"""

DEFAULT_ERROR_TYPES: tuple[str, ...] = ("insufficient_balance",)
"""不传 --error-type 时仅处理 402 / Insufficient Balance / 余额不足。"""


@dataclass(frozen=True)
class ErrorSelector:
    """按类型和自定义关键词选择要补判的记录/verdict。"""

    types: tuple[str, ...]
    keywords: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = [name for name in self.types if name not in ERROR_RULES]
        if unknown:
            raise ValueError(
                f"未知错误类型：{'、'.join(unknown)}；可用：{'、'.join(ERROR_RULES)}"
            )

    def matches_reason(self, reason: Any) -> bool:
        if not isinstance(reason, str) or not reason.strip():
            return False
        if any(ERROR_RULES[name].pattern.search(reason) for name in self.types):
            return True
        lowered = reason.casefold()
        return any(keyword.casefold() in lowered for keyword in self.keywords if keyword)

    def matches_verdict(self, verdict: Any) -> bool:
        if not isinstance(verdict, dict):
            return False
        if verdict.get("source") == "llm":
            return False
        if sentence_screening.verdict_status(verdict) != sentence_screening.SCREENING_ERROR:
            return False
        return self.matches_reason(verdict.get("reason"))

    def matches_record(self, record: dict) -> bool:
        if sentence_screening.screening_status(record) != sentence_screening.SCREENING_ERROR:
            return False
        block = record.get("screening")
        if not isinstance(block, dict):
            return False
        if self.matches_reason(block.get("reason")):
            return True
        verdicts = block.get("verdicts")
        if isinstance(verdicts, list):
            for verdict in verdicts:
                if self.matches_verdict(verdict):
                    return True
        return False

    def signature(self) -> dict[str, Any]:
        return {
            "types": sorted(set(self.types)),
            "keywords": sorted(set(self.keywords)),
        }

    def describe(self) -> str:
        types = "、".join(sorted(set(self.types)))
        if self.keywords:
            keywords = "、".join(sorted(set(self.keywords)))
            return f"{types}；自定义关键词：{keywords}"
        return types


_CHECKPOINT_VERSION = 2
"""checkpoint 结构版本；错误选择器不同或旧版 checkpoint 都要求重跑。"""


def _parse_error_types(values: list[str] | None) -> tuple[str, ...]:
    raw: list[str] = []
    for value in values or ():
        raw.extend(part.strip() for part in str(value).split(",") if part.strip())
    if not raw:
        return DEFAULT_ERROR_TYPES
    unknown = [name for name in raw if name != "all" and name not in ERROR_RULES]
    if unknown:
        raise ValueError(
            f"未知错误类型：{'、'.join(unknown)}；可用：{'、'.join(ERROR_RULES)} 或 all"
        )
    if "all" in raw:
        return tuple(ERROR_RULES)
    return tuple(sorted(set(raw)))


def _parse_error_keywords(values: list[str] | None) -> tuple[str, ...]:
    result: list[str] = []
    for value in values or ():
        keyword = str(value).strip()
        if keyword and keyword not in result:
            result.append(keyword)
    return tuple(result)


def _line_ending(raw: bytes) -> bytes:
    if raw.endswith(b"\r\n"):
        return b"\r\n"
    if raw.endswith(b"\n"):
        return b"\n"
    return b""


def _encode_record(record: dict, original: bytes) -> bytes:
    """改写记录时保留原行的换行风格；未命中行则完全不重新序列化。"""
    return json.dumps(record, ensure_ascii=False).encode("utf-8") + _line_ending(original)


def _new_stats() -> dict[str, Any]:
    return {
        "input_records": 0,
        "output_records": 0,
        "selected_records": 0,
        "unselected_records": 0,
        "processed_selected_records": 0,
        "attempted_records": 0,
        "repaired_true": 0,
        "repaired_false": 0,
        "still_error": 0,
        "still_error_non_matching": 0,
        "still_limit": 0,
        "still_other": 0,
        "still_matching_records": 0,
        "unresolved_records": 0,
        "limit_skipped": 0,
        "verdicts_retried": 0,
        "verdicts_updated": 0,
        "unmappable_verdicts": 0,
        "unmappable_records": 0,
        "rows_rescreened": 0,
        "resumed_from_input_offset": 0,
        "resumed_from_output_offset": 0,
    }


def _input_signature(
    input_path: Path,
    model: str,
    config_path: Path,
    kind: str,
    system_prompt: str,
    selection: dict[str, Any],
) -> dict:
    stat = input_path.stat()
    config_bytes = config_path.read_bytes()
    return {
        "input_path": str(input_path.resolve()),
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "model": model,
        "filter_config": str(config_path.resolve()),
        "filter_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "kind": kind,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "selection": selection,
    }


def _checkpoint_path(output: Path) -> Path:
    return output.with_name(output.name + ".checkpoint.json")


def _partial_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial")


def _default_summary_path(output: Path) -> Path:
    return output.with_suffix(".summary.json")


def _load_checkpoint(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 —— 损坏的 checkpoint 不能猜着续跑
        raise ValueError(f"checkpoint 无法读取或不是合法 JSON：{path}（{exc!r}）") from exc
    if not isinstance(data, dict) or data.get("version") != _CHECKPOINT_VERSION:
        raise ValueError(f"checkpoint 版本不兼容：{path}；请加 --overwrite 重新开始")
    return data


def _save_checkpoint(path: Path, data: dict) -> None:
    with atomic_write_text(path) as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _build_screener(args: argparse.Namespace) -> sentence_screening.SentenceScreener:
    config_path = Path(args.filter_config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"找不到筛选配置文件：{config_path}")
    family = config_path.stem
    kind = args.screen_kind or sentence_screening.classify_family(family)
    if kind is None:
        raise ValueError(
            f"无法从配置文件名 {family!r} 判断词类型；请用 --screen-kind 指定"
            f"（{'、'.join(sorted(sentence_screening.SCREEN_PROMPT_FILES))}）"
        )
    overrides = None
    if args.max_retries is not None:
        overrides = {"max_retries": args.max_retries}
    screener = sentence_screening.SentenceScreener(
        family=family,
        config_path=str(config_path),
        model_key=args.screen_model,
        llm=True,
        workers=args.screen_workers,
        batch_size=args.screen_batch_size,
        kind=kind,
        budget=sentence_screening.ScreenItemBudget(0),
        retry_policy_overrides=overrides,
    )
    if not screener.enabled:
        raise RuntimeError(f"筛选器未启用，拒绝把未判记录写成 no_llm：{screener.disabled_reason}")
    return screener


def _process(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    checkpoint_path: Path,
    partial_path: Path,
    signature: dict,
    stats: dict[str, Any],
    input_offset: int,
    output_offset: int,
    resume: bool,
    screener: sentence_screening.SentenceScreener,
    selector: ErrorSelector,
    limit: int,
    io_batch_size: int,
    progress_every: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+b" if resume else "wb"
    with open(input_path, "rb") as source, open(partial_path, mode) as target:
        if resume:
            target.seek(output_offset)
            target.truncate()
        else:
            target.seek(0)
            target.truncate()
        source.seek(input_offset)

        line_no = stats.get("input_records", 0)
        chunks_done = 0
        while True:
            raw_lines: list[bytes] = []
            for _ in range(io_batch_size):
                raw = source.readline()
                if not raw:
                    break
                raw_lines.append(raw)
            if not raw_lines:
                break

            records: list[dict] = []
            for raw in raw_lines:
                line_no += 1
                try:
                    record = json.loads(raw)
                except Exception as exc:  # noqa: BLE001 —— 标准输出里出现坏行时不能发布半成品
                    raise ValueError(f"输入第 {line_no} 行不是合法 JSON：{exc!r}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"输入第 {line_no} 行不是 JSON 对象")
                records.append(record)

            # 必须用改写前的原始字节长度推进输入偏移；否则续跑会从半行开始。
            read_bytes = sum(len(raw) for raw in raw_lines)

            selected_indices: list[int] = []
            selected_records: list[dict] = []
            for index, record in enumerate(records):
                if not selector.matches_record(record):
                    stats["unselected_records"] += 1
                    continue
                stats["selected_records"] += 1
                if limit > 0 and stats["processed_selected_records"] + len(selected_records) >= limit:
                    stats["limit_skipped"] += 1
                    continue
                selected_indices.append(index)
                selected_records.append(record)

            if selected_records:
                rescreened_rows, retry_stats = screener.rescreen_rows(
                    selected_records, selector.matches_verdict
                )
                if len(rescreened_rows) != len(selected_records):
                    raise RuntimeError(
                        "补判返回行数与输入不一致，拒绝发布可能缺行的输出："
                        f"{len(rescreened_rows)} != {len(selected_records)}"
                    )
                if any(actual is not expected for actual, expected in zip(rescreened_rows, selected_records)):
                    raise RuntimeError(
                        "补判返回行顺序或对象身份发生变化，拒绝发布可能变序的输出"
                    )
                stats["processed_selected_records"] += len(selected_records)
                stats["attempted_records"] += int(retry_stats.get("rows_rescreened", 0))
                stats["unmappable_records"] += max(
                    0, len(selected_records) - int(retry_stats.get("rows_rescreened", 0))
                )
                for key in ("verdicts_retried", "verdicts_updated", "unmappable_verdicts", "rows_rescreened"):
                    stats[key] += int(retry_stats.get(key, 0))
                for index, record in zip(selected_indices, rescreened_rows):
                    status = sentence_screening.screening_status(record)
                    if status == sentence_screening.SCREENING_TRUE:
                        stats["repaired_true"] += 1
                    elif status == sentence_screening.SCREENING_FALSE:
                        stats["repaired_false"] += 1
                    elif status == sentence_screening.SCREENING_LIMIT:
                        stats["still_limit"] += 1
                    elif status == sentence_screening.SCREENING_ERROR:
                        stats["still_error"] += 1
                        if not selector.matches_record(record):
                            stats["still_error_non_matching"] += 1
                    else:
                        stats["still_other"] += 1
                    if status not in (
                        sentence_screening.SCREENING_TRUE,
                        sentence_screening.SCREENING_FALSE,
                    ):
                        stats["unresolved_records"] += 1
                    if selector.matches_record(record):
                        stats["still_matching_records"] += 1
                    raw_lines[index] = _encode_record(record, raw_lines[index])

            written = 0
            for raw in raw_lines:
                target.write(raw)
                written += 1
            if written != len(records):
                raise RuntimeError(
                    f"写出行数与读入不一致：{written} != {len(records)}；拒绝发布"
                )
            target.flush()
            os.fsync(target.fileno())

            input_offset += read_bytes
            output_offset = target.tell()
            stats["input_records"] += len(raw_lines)
            stats["output_records"] += written
            chunks_done += 1
            _save_checkpoint(
                checkpoint_path,
                {
                    "version": _CHECKPOINT_VERSION,
                    "signature": signature,
                    "input_offset": input_offset,
                    "output_offset": output_offset,
                    "stats": stats,
                },
            )
            if progress_every > 0 and chunks_done % progress_every == 0:
                print(
                    f"[补判] 已读 {stats['input_records']} 行，"
                    f"选中 {stats['selected_records']} 条，"
                    f"已尝试 {stats['attempted_records']} 条，"
                    f"修复 {stats['repaired_true'] + stats['repaired_false']} 条，"
                    f"仍失败 {stats['still_error']} 条",
                    flush=True,
                )

    stats["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    stats["limit"] = limit or None
    stats["selection"] = selector.signature()
    stats["selection_description"] = selector.describe()
    stats["remaining_records"] = stats["unresolved_records"] + stats["limit_skipped"]
    stats["remaining_matching_records"] = stats["still_matching_records"] + stats["limit_skipped"]
    stats["screener"] = screener.stats()

    # 总集与顺序的不变式：所有输入行都必须原位置写出一条输出。
    if stats["input_records"] != stats["output_records"]:
        raise RuntimeError(
            f"输出不是输入总集：输入 {stats['input_records']} 行，"
            f"输出 {stats['output_records']} 行；拒绝发布"
        )
    if stats["selected_records"] + stats["unselected_records"] != stats["input_records"]:
        raise RuntimeError("selected + unselected 与输入行数不一致；拒绝发布")
    if input_offset != input_path.stat().st_size:
        raise RuntimeError(
            f"输入未完整读完：偏移 {input_offset} != 文件大小 {input_path.stat().st_size}"
        )

    # 收尾顺序：先写 summary，再写带 complete 的 checkpoint，最后发布最终文件。
    # 这样 rename 后中断时，checkpoint 足以让下一次 --resume 只做幂等清理。
    with atomic_write_text(summary_path) as handle:
        handle.write(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    _save_checkpoint(
        checkpoint_path,
        {
            "version": _CHECKPOINT_VERSION,
            "signature": signature,
            "input_offset": input_offset,
            "output_offset": output_offset,
            "complete": True,
            "stats": stats,
        },
    )
    os.replace(partial_path, output_path)
    checkpoint_path.unlink(missing_ok=True)
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="补判 extract 标准输出中指定错误类型的未完成记录"
    )
    parser.add_argument("input", type=str, help="extract 标准输出 JSONL")
    parser.add_argument(
        "-f",
        "--filter-config-path",
        required=True,
        type=str,
        help="筛选规则配置，用来推导目标词表和词类型；例如 "
        "config_files/localization_expressions.json5",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 JSONL；默认在输入旁生成 <文件名>.rescreened.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="统计 JSON；默认 <输出文件名>.summary.json",
    )
    parser.add_argument(
        "--error-type",
        action="append",
        default=None,
        metavar="NAME",
        help="要补判的错误类型，可重复或逗号分隔；默认 insufficient_balance。可用："
        + "、".join(ERROR_RULES)
        + "；all 表示全部内置类型",
    )
    parser.add_argument(
        "--error-keyword",
        action="append",
        default=None,
        metavar="TEXT",
        help="额外按 reason 中的字面关键词选择错误，可重复；与 --error-type 取并集",
    )
    parser.add_argument(
        "--screen-model",
        default=config.SCREEN_MODEL,
        help=f"LLM 筛选使用的模型配置名，默认 {config.SCREEN_MODEL}",
    )
    parser.add_argument(
        "--screen-workers",
        type=int,
        default=None,
        help="并发线程数；不传时用模型配置或全局默认",
    )
    parser.add_argument(
        "--screen-batch-size",
        type=int,
        default=None,
        help="每次请求包含多少条判定项；不传时用模型配置或全局默认",
    )
    parser.add_argument(
        "--screen-kind",
        choices=sorted(sentence_screening.SCREEN_PROMPT_FILES),
        default=None,
        help="强制指定词类型；默认按配置文件名推断",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="覆盖模型配置里的传输层重试次数（0 表示不重试）；"
        "具体错误是否自动重试仍由 retry policy 决定",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="本次最多尝试多少条选中记录，0=不限；适合先小样本验证",
    )
    parser.add_argument(
        "--io-batch-size",
        type=int,
        default=500,
        help="每次读入并写出的行数，默认 500（避免把 1.1GB 文件整体读进内存）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="每处理多少个 I/O chunk 打印一次进度，0=关闭，默认 10",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 <输出>.checkpoint.json 和 <输出>.partial 续跑；签名不一致会拒绝",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许覆盖已有输出、partial、checkpoint 和 summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"输入文件不存在：{input_path}")
    if args.limit < 0:
        parser.error("--limit 不能为负数")
    if args.max_retries is not None and args.max_retries < 0:
        parser.error("--max-retries 不能为负数")
    if args.io_batch_size < 1:
        parser.error("--io-batch-size 必须是正整数")
    if args.progress_every < 0:
        parser.error("--progress-every 不能为负数")
    try:
        selector = ErrorSelector(
            _parse_error_types(args.error_type),
            _parse_error_keywords(args.error_keyword),
        )
    except ValueError as exc:
        parser.error(str(exc))

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + ".rescreened.jsonl")
    )
    summary_path = Path(args.summary) if args.summary else _default_summary_path(output_path)
    partial_path = _partial_path(output_path)
    checkpoint_path = _checkpoint_path(output_path)

    validate_output_paths(
        [input_path],
        [output_path, summary_path, partial_path, checkpoint_path],
    )

    screener = _build_screener(args)
    signature = _input_signature(
        input_path,
        args.screen_model,
        Path(args.filter_config_path),
        screener.kind or "",
        screener.system_prompt,
        selector.signature(),
    )

    if args.overwrite:
        for path in (output_path, summary_path, partial_path, checkpoint_path):
            path.unlink(missing_ok=True)

    stats = _new_stats()
    input_offset = 0
    output_offset = 0
    resume = bool(args.resume and checkpoint_path.exists())
    if resume:
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.get("signature") != signature:
            raise ValueError(
                "checkpoint 与当前输入/模型/规则/提示语/错误选择器签名不一致，拒绝续跑；"
                "请加 --overwrite 从头开始"
            )
        input_offset = int(checkpoint.get("input_offset") or 0)
        output_offset = int(checkpoint.get("output_offset") or 0)
        input_size = input_path.stat().st_size
        if not 0 <= input_offset <= input_size:
            raise ValueError(
                f"checkpoint 的输入偏移越界：{input_offset}（文件大小 {input_size}）；"
                "请加 --overwrite 从头开始"
            )
        if output_offset < 0:
            raise ValueError(f"checkpoint 的输出偏移非法：{output_offset}")
        saved_stats = checkpoint.get("stats")
        if isinstance(saved_stats, dict):
            stats.update(saved_stats)

        if output_path.exists():
            # rename 后、summary/checkpoint 清理前中断的窗口：partial 已变成最终输出，
            # checkpoint 已标记 complete。此时只补齐收尾，不重新发请求。
            if checkpoint.get("complete") is True and not partial_path.is_file():
                if not summary_path.exists():
                    with atomic_write_text(summary_path) as handle:
                        handle.write(
                            json.dumps(saved_stats or stats, ensure_ascii=False, indent=2) + "\n"
                        )
                checkpoint_path.unlink(missing_ok=True)
                print(
                    f"[补判] 最终输出已发布，仅清理收尾状态：{output_path}",
                    flush=True,
                )
                return 0
            raise ValueError(
                f"输出文件已存在，但 checkpoint 不在可恢复的已完成状态：{output_path}；"
                "请加 --overwrite 从头开始，或换一个 -o 路径"
            )

        if not partial_path.is_file():
            raise ValueError(
                f"checkpoint 存在但 partial 文件缺失：{partial_path}；"
                "请加 --overwrite 从头开始"
            )
        if partial_path.stat().st_size < output_offset:
            raise ValueError(
                f"partial 文件比 checkpoint 记录的输出偏移短："
                f"{partial_path}（{partial_path.stat().st_size} < {output_offset}）；"
                "请加 --overwrite 从头开始"
            )
        stats["resumed_from_input_offset"] = input_offset
        stats["resumed_from_output_offset"] = output_offset
        print(
            f"[补判] 从 checkpoint 续跑：输入偏移 {input_offset}，"
            f"输出偏移 {output_offset}，已尝试 {stats.get('attempted_records', 0)} 条",
            flush=True,
        )
    elif args.resume:
        if partial_path.exists():
            raise ValueError(
                "--resume 指定续跑，但 checkpoint 缺失而 partial 已存在；"
                "为避免丢掉已有进度，拒绝从头截断。请恢复 checkpoint，"
                "或显式加 --overwrite 重新开始"
            )
        print("[补判] 未找到 checkpoint，按新任务从头开始", flush=True)

    if not resume:
        existing = [
            path
            for path in (output_path, partial_path, checkpoint_path, summary_path)
            if path.exists()
        ]
        if existing:
            raise ValueError(
                "检测到已有输出/断点文件，未加 --resume 或 --overwrite，拒绝覆盖或截断："
                + "、".join(str(path) for path in existing)
            )
        # 清理可能残留的旧 checkpoint（正常情况下上面的检查已确保不存在）。
        checkpoint_path.unlink(missing_ok=True)

    print(f"[补判] 输入：{input_path}")
    print(f"[补判] 输出：{output_path}")
    print(f"[补判] 模型：{args.screen_model}；配置：{args.filter_config_path}")
    print(f"[补判] 错误选择：{selector.describe()}")
    print(f"[补判] {screener.describe()}")

    # 本地模型服务复用/启动范围包住 warm 与全部补判；远程模型是空上下文。
    with local_model_service.for_model(args.screen_model, enabled=screener.enabled):
        screener.warm()
        stats = _process(
            input_path=input_path,
            output_path=output_path,
            summary_path=summary_path,
            checkpoint_path=checkpoint_path,
            partial_path=partial_path,
            signature=signature,
            stats=stats,
            input_offset=input_offset,
            output_offset=output_offset,
            resume=resume,
            screener=screener,
            selector=selector,
            limit=args.limit,
            io_batch_size=args.io_batch_size,
            progress_every=args.progress_every,
        )

    print(
        f"[补判] 完成：输入 {stats['input_records']} 行，输出 {stats['output_records']} 行；"
        f"选中 {stats['selected_records']} 条，"
        f"修复 true={stats['repaired_true']} / false={stats['repaired_false']}，"
        f"未解决 {stats['remaining_records']}，limit 跳过 {stats['limit_skipped']}",
        flush=True,
    )
    print(f"[补判] 统计：{summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
