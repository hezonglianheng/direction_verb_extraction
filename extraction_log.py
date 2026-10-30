# encoding: utf8

"""语料筛选日志：把「读入 → 程序初筛 → LLM 筛选 → 落盘」的漏斗数量写进 log/。

为什么单独一个模块：
    两个平台入口（entry_in_linux / entry_in_win）的筛选取数逻辑必须一致，日志字段与
    文件名也不该各写一份；入口只负责在成功收尾时把统计交过来，渲染与落盘集中在这里。

统计口径（与 filter_sentences_streaming 的计数点对应）：
    read_sentences     切句 + 长度/空串/字符/缺文过滤后、**词表子串初筛前**的有效句数。
                       词表初筛掉的句子仍算「读入」，只是没进入规则筛选。
    program_selected   规则引擎命中、再经过方位词语义闸门后、**送 LLM 前**的句子数；
                       多套规则同时命中同一句时按句去重（用批内 _corpus_row_id）。
    corpus_written     实际落盘的语料数；false/error/limit/no_llm 也全部落盘。
    llm_kept           可用于判定工作流的句子数（LLM 判 true + 未使用 LLM）。
    llm_excluded       仅指 LLM 明确判为不可提取的句子数（已落盘，但不进入判定）。
    llm_error          LLM 调用/输出异常的句子数（已落盘，不进入判定）。
    llm_limit          超出 LLM 发送上限的句子数（已落盘，不进入判定）。
    no_llm             未使用 LLM 筛选的句子数（已落盘，进入判定）。

多套规则时另给每族的 program_selected / llm_kept / llm_excluded 明细；同一句同时命中
两族会在两族各计一次，因此分族数字之和可能大于全局去重数字，这是刻意的。

日志写失败（目录不可写、磁盘满等）只打印警告，不把已经成功落盘的语料任务改判为失败。
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import config
from file_paths import atomic_write_text

DEFAULT_LOG_DIR = Path(config._PROJECT_ROOT) / "log"
"""默认日志目录：项目根下的 log/，不依赖运行时 cwd。"""


def new_stats() -> dict[str, Any]:
    """一份新的漏斗统计；入口可以把它透传给抽取/筛选两段分别累加。"""
    return {
        "read_sentences": 0,
        "vocab_candidates": 0,
        "program_selected": 0,
        "corpus_written": 0,
        "llm_kept": 0,
        "llm_excluded": 0,
        "llm_error": 0,
        "llm_limit": 0,
        "no_llm": 0,
        "screening_status": {},
        "families": {},
        "grounding": {},
    }


def ensure_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    """补齐旧调用方/测试桩传进来的不完整统计，保证后续累加不会 KeyError。"""
    if stats is None:
        return new_stats()
    for key, value in new_stats().items():
        stats.setdefault(key, value)
    return stats


def family_stats(stats: dict[str, Any], family: str) -> dict[str, int]:
    """取（必要时创建）某一套规则的明细计数器。"""
    families = stats.setdefault("families", {})
    return families.setdefault(
        family,
        {
            "program_selected": 0,
            "written": 0,
            "llm_kept": 0,
            "llm_excluded": 0,
            "llm_error": 0,
            "llm_limit": 0,
            "no_llm": 0,
        },
    )


def log_path(log_dir: str | Path | None = None, *, now: datetime | None = None) -> Path:
    """本轮日志路径：log/extract-log-YYYYMMDD-HHMMSS-ffffff.log

    带微秒是为了同一秒内连跑两次（测试/小样本重跑）不会互相覆盖。
    """
    moment = now or datetime.now()
    directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    return directory / f"extract-log-{moment.strftime('%Y%m%d-%H%M%S-%f')}.log"


def _iso(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(float(timestamp)).isoformat(timespec="seconds")


def render_log(
    stats: dict[str, Any],
    *,
    src_path: str,
    filter_configs: Iterable[str],
    output_paths: Iterable[str | Path],
    vocab_count: int = 0,
    screening_stats: dict[str, Any] | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> str:
    """把统计渲染成便于人工阅读、也便于 grep 的文本。"""
    stats = ensure_stats(stats)
    read = int(stats.get("read_sentences", 0))
    candidates = int(stats.get("vocab_candidates", 0))
    program = int(stats.get("program_selected", 0))
    written = int(stats.get("corpus_written", 0))
    kept = int(stats.get("llm_kept", 0))
    excluded = int(stats.get("llm_excluded", 0))
    errors = int(stats.get("llm_error", 0))
    limited = int(stats.get("llm_limit", 0))
    no_llm = int(stats.get("no_llm", 0))

    lines = [
        "# 语料筛选日志",
        f"开始时间: {_iso(started_at)}",
        f"结束时间: {_iso(finished_at if finished_at is not None else time.time())}",
        f"输入路径: {src_path}",
        f"筛选配置: {', '.join(str(item) for item in filter_configs) or '（无）'}",
        f"输出文件: {', '.join(str(item) for item in output_paths) or '（无）'}",
        f"初筛词表词数: {int(vocab_count)}",
        "口径: 读入=切句及空串/长度/缺文过滤后、词表初筛前；"
        "程序筛选=规则命中并过方位词语义闸门后、送 LLM 前（按句去重）；"
        "所有筛选结果都落盘，false/error/limit 只标注、不进入判定工作流。",
        "",
        f"总共读入的句子数量: {read}",
        f"初步程序筛选筛出的句子数量: {program}",
        f"写出的语料条数（全部标注）: {written}",
        f"可用于判定工作流的句子数量（true/no_llm）: {kept}",
        f"被 LLM 判为不可提取的句子数量（已落盘）: {excluded}",
        f"LLM 输出错误的句子数量（已落盘）: {errors}",
        f"超出 LLM 发送上限的句子数量（已落盘）: {limited}",
        f"未使用 LLM 筛选的句子数量（已落盘）: {no_llm}",
        "",
        "详细:",
        f"  词表初筛排除的句子数量: {max(0, read - candidates)}",
        f"  词表初筛后进入规则筛选的句子数量: {candidates}",
        f"  规则/语义闸门排除的句子数量: {max(0, candidates - program)}",
        "分规则族（同一句命中多族时各计一次）:",
    ]
    families = stats.get("families") or {}
    if families:
        for family, counts in families.items():
            lines.append(
                f"  {family}: 初步程序筛选={int(counts.get('program_selected', 0))}, "
                f"落盘={int(counts.get('written', 0))}, "
                f"可判定={int(counts.get('llm_kept', 0))}, "
                f"LLM 判假={int(counts.get('llm_excluded', 0))}, "
                f"LLM 错误={int(counts.get('llm_error', 0))}, "
                f"超限={int(counts.get('llm_limit', 0))}, "
                f"未用 LLM={int(counts.get('no_llm', 0))}"
            )
    else:
        lines.append("  （无）")

    if screening_stats:
        lines.append("")
        lines.append("LLM 筛选器统计:")
        for family, detail in screening_stats.items():
            lines.append(f"  {family}: {detail}")

    status_counts = stats.get("screening_status") or {}
    if status_counts:
        lines.append("")
        lines.append("筛选状态分布（按句去重）:")
        for status, count in sorted(status_counts.items()):
            if count:
                lines.append(f"  {status}: {int(count)}")

    grounding = stats.get("grounding") or {}
    if grounding.get("records"):
        lines.append("")
        lines.append(f"方位词参照物语义闸门: {grounding}")
    return "\n".join(lines) + "\n"


def write_extract_log(
    stats: dict[str, Any] | None,
    *,
    src_path: str,
    filter_configs: Iterable[str],
    output_paths: Iterable[str | Path],
    vocab_count: int = 0,
    screening_stats: dict[str, Any] | None = None,
    log_dir: str | Path | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
) -> Path | None:
    """在 log/ 下写一份带时间戳的筛选日志；失败不抛异常，返回 None。

    Returns:
        Path | None: 成功时返回日志路径，写失败时返回 None（只向 stderr 警告）。
    """
    try:
        target = log_path(log_dir)
        text = render_log(
            stats,
            src_path=src_path,
            filter_configs=filter_configs,
            output_paths=output_paths,
            vocab_count=vocab_count,
            screening_stats=screening_stats,
            started_at=started_at,
            finished_at=finished_at,
        )
        # atomic_write_text 会创建父目录（log/ 不存在时自动建），并先写临时文件再替换
        with atomic_write_text(target) as handle:
            handle.write(text)
    except Exception as exc:  # noqa: BLE001 —— 日志是附属产物，不能反过来吞掉成功结果
        print(
            f"[筛选日志] 写入 {log_dir or DEFAULT_LOG_DIR} 下的日志失败：{exc!r}；本轮语料输出不受影响",
            file=sys.stderr,
        )
        return None
    print(f"[筛选日志] 已写入 {target}")
    return target
