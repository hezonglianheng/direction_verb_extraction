# encoding: utf8

"""句对替换日志：把 inputs/outputs、句对状态和 LLM 筛选性能写进 log/。

与 extract/extraction_log.py 同一风格：文本可读、字段可 grep；日志失败只警告，
不影响已经原子写出的句对文件。
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from extract import screening_metrics
from file_paths import atomic_write_text

DEFAULT_LOG_DIR = Path(config._PROJECT_ROOT) / "log"


def log_path(log_dir: str | Path | None = None, *, now: datetime | None = None) -> Path:
    moment = now or datetime.now()
    directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    return directory / f"replace-log-{moment.strftime('%Y%m%d-%H%M%S-%f')}.log"


def _iso(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(float(timestamp)).isoformat(timespec="seconds")


def _render_limit_stats(stats: dict[str, Any]) -> list[str]:
    """限额抽样统计：模式/种子/候选数/逐词配额，便于复现和核对。"""
    lines = ["", "限额抽样:"]
    if stats.get("mode") == "aligned" and stats.get("limit") is None:
        lines.append("  limit: 全部")
    for key in (
        "version", "mode", "limit", "seed", "corpus_type", "scanned", "blank",
        "skipped_ineligible", "skipped_bad_record", "skipped_no_word",
        "candidates", "available", "selected", "stopped_early",
        "reference_file", "reference_rows", "reference_blank", "reference_duplicates",
        "reference_records", "matched_reference_records", "unmatched_reference_records",
        "matched_source_records", "content_matches", "line_collisions", "limit_excluded",
    ):
        if key in stats and stats[key] is not None:
            lines.append(f"  {key}: {stats[key]}")
    key_counts = stats.get("key_counts")
    if isinstance(key_counts, dict) and key_counts:
        lines.append("  对齐键统计:")
        for key, value in key_counts.items():
            lines.append(f"    {key}: {value}")
    line_nos = stats.get("selected_line_nos")
    if isinstance(line_nos, list) and line_nos:
        preview = "、".join(str(number) for number in line_nos)
        more = "（仅预览前 %d 条）" % len(line_nos) if stats.get("selected", 0) > len(line_nos) else ""
        lines.append(f"  selected_line_nos: {preview}{more}")
    totals = stats.get("word_totals")
    quotas = stats.get("word_quotas")
    selected = stats.get("word_selected")
    if isinstance(totals, dict) and totals:
        lines.append("  逐词统计:")
        for word, total in totals.items():
            quota = (quotas or {}).get(word, 0)
            chosen = (selected or {}).get(word, 0)
            lines.append(f"    {word}: total={total} quota={quota} selected={chosen}")
    return lines


def render_log(
    *,
    corpus: str,
    output: str,
    corpus_type: str,
    model_key: str,
    counts: dict[str, Any],
    screening_stats: dict[str, Any] | None = None,
    limit_stats: dict[str, Any] | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> str:
    lines = [
        "# 句对替换日志",
        f"开始时间: {_iso(started_at)}",
        f"结束时间: {_iso(finished_at if finished_at is not None else time.time())}",
        f"输入语料: {corpus}",
        f"输出文件: {output}",
        f"语料类型: {corpus_type}",
        f"模型配置: {model_key}",
        "",
        "句对状态统计:",
    ]
    if counts:
        for key, value in sorted(counts.items()):
            lines.append(f"  {key}: {value}")
    else:
        lines.append("  （无）")

    lines.append("")
    lines.append("LLM 筛选器统计:")
    if screening_stats:
        for family, detail in screening_stats.items():
            lines.append(f"  {family}:")
            if isinstance(detail, dict):
                lines.extend(screening_metrics.format_stats_lines(detail))
                lines.append(f"    stats_json: {screening_metrics.stats_json(detail)}")
            else:
                lines.append(f"    {detail}")
    else:
        lines.append("  （无；可能使用了 --no-llm 或未提供筛选器）")
    if limit_stats:
        lines.extend(_render_limit_stats(limit_stats))
    return "\n".join(lines) + "\n"


def write_log(
    *,
    corpus: str,
    output: str,
    corpus_type: str,
    model_key: str,
    counts: dict[str, Any],
    screening_stats: dict[str, Any] | None = None,
    limit_stats: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> Path | None:
    """写 replace-log-*.log；失败只打印警告并返回 None。"""
    try:
        target = log_path(log_dir)
        text = render_log(
            corpus=corpus,
            output=output,
            corpus_type=corpus_type,
            model_key=model_key,
            counts=counts,
            screening_stats=screening_stats,
            limit_stats=limit_stats,
            started_at=started_at,
            finished_at=finished_at,
        )
        with atomic_write_text(target) as handle:
            handle.write(text)
    except Exception as exc:  # noqa: BLE001 —— 日志是附属产物，不能反过来吞掉句对产物
        print(
            f"[替换日志] 写入 {log_dir or DEFAULT_LOG_DIR} 下的日志失败：{exc!r}；句对输出不受影响",
            file=sys.stderr,
        )
        return None
    print(f"[替换日志] 已写入 {target}")
    return target
