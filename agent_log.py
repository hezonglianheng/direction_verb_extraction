# encoding: utf8

"""Agent 执行日志：每跑完一个标注 agent，就在 log/ 下留下一条可回查的记录。

为什么单独一个模块：
    extraction_log.py 记的是「读入 → 程序初筛 → LLM 筛选 → 落盘」的漏斗，粒度是一轮跑批；
    本模块记的是「哪个 agent、哪条记录、哪一侧、花了多久、给出什么结论」，粒度是一次
    agent.run。两者都放 log/，但用途不同，字段也不该挤在同一张表里。

一轮跑批会写两份同名的文件：
    log/agent-log-YYYYMMDD-HHMMSS-ffffff.log     人读：一次执行一段，收尾带汇总
    log/agent-log-YYYYMMDD-HHMMSS-ffffff.jsonl   机器读：一次执行一个 JSON 对象

两份文件用同一个时间戳，拿到任意一份都能找到另一份。JSONL 适合事后用 jq/脚本统计，
.log 适合直接 tail 与人工翻看。

记录口径：
    status  ok        agent 正常给出结论（含查表命中、规则改写成功）
            skipped   没有调用或无从判断（--no-llm、提示语未填、没有关注词等）
            error     调用失败、结构化输出缺失、模型给出的结果不可用
    source  原结果里的 source 字段，原样保留；alter 的纯规则路径记 "rule"。
    elapsed 一次 agent.run 的墙上时间（不含日志写入本身）。
    input/output 只记摘要：长文本截断、嵌套列表限量，避免把整份匹配树或特征表写进日志。

异常策略：
    日志是附属产物。目录不可写、磁盘满、序列化遇到奇怪对象等，都只向 stderr 警告一次，
    绝不把已经跑成功的标注改判为失败；调用方也不需要用 try/except 包住日志调用。

并发口径：
    所有写入都在一把线程锁内完成。--workers N 的多线程与 --models 的多模型并行都安全；
    多模型共用同一份日志，记录里的 model 字段区分来源。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import config

DEFAULT_LOG_DIR = Path(config._PROJECT_ROOT) / "log"
"""默认日志目录：项目根下的 log/，不依赖运行时 cwd。"""

DEFAULT_PREFIX = "agent-log"
"""日志文件名前缀；与 extraction_log 的 extract-log 区分。"""

_MAX_TEXT_CHARS = 500
"""单条字符串在日志里的最大长度；超出的截断并注明省略了多少字符。"""

_MAX_ITEMS = 20
"""字典键、列表元素的最大保留数量；超出的记一个省略计数。"""

_MAX_DEPTH = 4
"""嵌套结构最多展开几层，避免递归写进异常对象或循环引用。"""

_SKIP_SOURCES = frozenset({
    "no_llm",
    "unfilled_prompt",
    "no_sentence",
    "no_focus",
    "no_target",
    "bad_record",
    "alter_missed",
    "limit",
    "no_llm_screen",
})
"""这些 source 表示 agent 没有真正调用模型、或前置条件不成立。"""

_ERROR_SOURCES = frozenset({"error", "llm_invalid"})
"""这些 source 表示模型调用失败，或模型给出的结果无法被下游使用。"""

_OUTPUT_KEYS: dict[str, tuple[str, ...]] = {
    "reality": ("is_spatial_scene", "source", "reason"),
    "semantic": (
        "is_fluent",
        "grammar_ok",
        "spatial_ok",
        "grammar_reason",
        "spatial_reason",
        "source",
        "reason",
    ),
    "feature": ("source", "reason", "validation"),
    "schema": ("name", "source", "reason"),
    "judge": ("same_meaning", "source", "lookup", "reason"),
    "synonym": ("same_meaning", "source", "reason"),
    "alter": ("applied", "source", "detail"),
}
"""各 agent 最值得写进日志的输出字段；其余字段（整棵匹配树、全部候选句等）不落日志。"""

_INPUT_TEXT_KEYS = frozenset({"sentence", "sentence_a", "sentence_b", "sentence_slice"})
"""这些输入在摘要里按文本处理（sentence_slice 先拼成句子）。"""


def _now() -> datetime:
    return datetime.now()


def _iso(timestamp: float | None = None) -> str:
    """统一的时间戳格式：本地时间、秒级 ISO 8601。"""
    moment = datetime.fromtimestamp(timestamp) if timestamp is not None else _now()
    return moment.isoformat(timespec="seconds")


def log_paths(
    log_dir: str | Path | None = None,
    *,
    now: datetime | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> tuple[Path, Path]:
    """本轮的 ``(人读 .log, 机器读 .jsonl)`` 路径。

    带微秒是为了同一秒内连跑两次（测试/小样本重跑）不会互相覆盖；极端情况下仍然撞名时，
    :class:`AgentLog` 会再补一个序号（见 ``_unique_paths``）。
    """
    moment = now or _now()
    directory = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    stem = f"{prefix}-{moment.strftime('%Y%m%d-%H%M%S-%f')}"
    return directory / f"{stem}.log", directory / f"{stem}.jsonl"


def _unique_paths(
    log_dir: str | Path | None = None,
    *,
    now: datetime | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> tuple[Path, Path]:
    """返回一对尚未被占用的路径；同一微秒内重复创建时追加 -1、-2……"""
    text_path, jsonl_path = log_paths(log_dir, now=now, prefix=prefix)
    if not text_path.exists() and not jsonl_path.exists():
        return text_path, jsonl_path
    counter = 1
    while True:
        candidate_text = text_path.with_name(f"{text_path.stem}-{counter}{text_path.suffix}")
        candidate_jsonl = jsonl_path.with_name(f"{jsonl_path.stem}-{counter}{jsonl_path.suffix}")
        if not candidate_text.exists() and not candidate_jsonl.exists():
            return candidate_text, candidate_jsonl
        counter += 1


def _compact(value: Any, *, depth: int = 0) -> Any:
    """把任意对象压成可 JSON 序列化、长度可控的摘要。

    字符串截断、容器限量、嵌套限深；Path/集合等转成字符串或列表，无法识别的对象用 repr。
    这里不修改原对象，调用方传进来的结果仍会原样交给下游。
    """
    if depth > _MAX_DEPTH:
        return "…（嵌套过深，已省略）"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_TEXT_CHARS:
            return value
        return value[:_MAX_TEXT_CHARS] + f"…（省略 {len(value) - _MAX_TEXT_CHARS} 字）"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        items = list(value.items())
        result: dict[str, Any] = {}
        for key, item in items[:_MAX_ITEMS]:
            result[str(key)] = _compact(item, depth=depth + 1)
        if len(items) > _MAX_ITEMS:
            result["__省略键数__"] = len(items) - _MAX_ITEMS
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = [_compact(item, depth=depth + 1) for item in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            result.append(f"…（省略 {len(items) - _MAX_ITEMS} 项）")
        return result
    try:
        rendered = repr(value)
    except Exception:  # noqa: BLE001 —— 连 repr 都失败的怪对象也不能让日志中断标注
        rendered = f"<无法打印的 {type(value).__name__} 对象>"
    return _compact(rendered, depth=depth + 1)


def _feature_summary(features: Any) -> Any:
    """特征结果只记「有几类、每类几个值、键是什么」，不把整份特征表塞进日志。"""
    if not isinstance(features, Mapping):
        return _compact(features)
    value_counts: dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            value_counts[str(key)] = len(value)
        elif isinstance(value, Mapping):
            value_counts[str(key)] = len(value)
        else:
            value_counts[str(key)] = 1
    return {
        "keys": [str(key) for key in list(features)[:_MAX_ITEMS]],
        "value_counts": value_counts,
        "preview": _compact(features),
    }


def _summarize_inputs(inputs: Any) -> Any:
    """把 agent.run 的实参压成摘要；句子类输入拼成可读文本。"""
    if not isinstance(inputs, Mapping):
        return _compact(inputs)
    summary: dict[str, Any] = {}
    for key, value in inputs.items():
        name = str(key)
        if name in _INPUT_TEXT_KEYS and isinstance(value, (list, tuple)):
            summary[name] = _compact(" ".join(str(item) for item in value))
        elif name in {"features", "features_a", "features_b"}:
            summary[name] = _feature_summary(value)
        else:
            summary[name] = _compact(value)
    return summary


def _summarize_output(agent_name: str, result: Any) -> Any:
    """按 agent 类型挑输出字段，避免把整棵匹配树/全部候选句写进日志。"""
    if not isinstance(result, Mapping):
        return _compact(result)
    output: dict[str, Any] = {}
    for key in _OUTPUT_KEYS.get(agent_name, ()):
        if key in result:
            output[key] = _compact(result[key])
    if agent_name == "feature":
        output["features"] = _feature_summary(result.get("features"))
    if agent_name == "alter":
        output["candidate_sentences"] = len(result.get("sentences") or ())
        output["candidate_slices"] = len(result.get("slices") or ())
        detail = result.get("detail")
        if isinstance(detail, Mapping):
            output["detail"] = {
                key: _compact(detail.get(key))
                for key in ("word", "index", "replacement", "applied", "reason", "source", "group")
                if key in detail
            }
    return output


def _classify(agent_name: str, result: Any) -> tuple[str, str | None]:
    """从 agent 返回的 source 推出统一的 status 与 source。"""
    if not isinstance(result, Mapping):
        return "ok", None
    source = result.get("source")
    if source in _ERROR_SOURCES:
        return "error", source
    if source in _SKIP_SOURCES:
        return "skipped", source
    if agent_name == "alter":
        return ("ok" if result.get("applied") else "skipped"), source or "rule"
    if source:
        return "ok", source
    return "ok", None


class AgentLog:
    """一轮跑批的 agent 日志写入器（线程安全，失败不抛异常）。

    典型用法（跑批入口）::

        log = AgentLog(run_meta={"graph": "pair", ...})
        for agent in agents.values():
            agent.attach_log(log)
        ...
        log.finish(agent_stats={name: agent.stats() for name, agent in agents.items()})

    每个 agent 节点通过 ``agent.log`` 取到本对象后调用 :meth:`agent_finished`；
    纯规则节点（alter）也走同一个入口，只是 source 记 "rule"。
    """

    def __init__(
        self,
        *,
        log_dir: str | Path | None = None,
        prefix: str = DEFAULT_PREFIX,
        run_meta: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ):
        self.run_id = ""
        self.path: Path | None = None
        self.jsonl_path: Path | None = None
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._closed = False
        self._summary: dict[str, Any] | None = None
        self._records = 0
        self._seq = 0
        self._stats: dict[str, dict[str, Any]] = {}
        self._text = None
        self._jsonl = None
        self._error = ""
        self._warned = False

        try:
            text_path, jsonl_path = _unique_paths(log_dir, now=now, prefix=prefix)
            self.path = text_path
            self.jsonl_path = jsonl_path
            self.run_id = text_path.stem
            text_path.parent.mkdir(parents=True, exist_ok=True)
            self._text = open(text_path, "w", encoding="utf-8")
            self._jsonl = open(jsonl_path, "w", encoding="utf-8")
            self._write_header(run_meta or {})
        except Exception as exc:  # noqa: BLE001 —— 日志写不了不能影响标注主流程
            self._error = repr(exc)
            self._warn_once()
            self._close_files()

    @property
    def enabled(self) -> bool:
        """是否成功打开了日志文件；失败时仍是可调用的空对象。"""
        return self._text is not None and self._jsonl is not None and not self._closed

    @property
    def error(self) -> str:
        """初始化失败的原因；成功时为空串。"""
        return self._error

    def _warn_once(self) -> None:
        if self._warned:
            return
        self._warned = True
        target = self.path.parent if self.path is not None else DEFAULT_LOG_DIR
        print(
            f"[Agent日志] 写入 {target} 下的日志失败：{self._error}；本轮标注结果不受影响",
            file=sys.stderr,
        )

    # -- 写文件 ------------------------------------------------------------
    def _close_files(self) -> None:
        for handle in (self._text, self._jsonl):
            if handle is None:
                continue
            try:
                handle.close()
            except Exception:  # noqa: BLE001 —— 关闭失败不值得覆盖原始异常
                pass
        self._text = None
        self._jsonl = None

    def _write_json(self, payload: Mapping[str, Any]) -> None:
        if self._jsonl is None:
            return
        self._jsonl.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def _write_text(self, line: str) -> None:
        if self._text is None:
            return
        self._text.write(line + "\n")
        self._text.flush()

    def _write_header(self, meta: Mapping[str, Any]) -> None:
        payload = {
            "ts": _iso(self._started_at),
            "run_id": self.run_id,
            "event": "run_start",
            "pid": os.getpid(),
            **{str(key): _compact(value) for key, value in meta.items()},
        }
        self._write_json(payload)
        lines = [
            "# Agent 执行日志",
            f"运行 ID: {self.run_id}",
            f"开始时间: {_iso(self._started_at)}",
        ]
        for key, value in meta.items():
            rendered = value if isinstance(value, str) else json.dumps(_compact(value), ensure_ascii=False)
            lines.append(f"{key}: {rendered}")
        lines.append("---")
        self._write_text("\n".join(lines))

    # -- 对外写入 ----------------------------------------------------------
    def event(self, event: str, **fields: Any) -> None:
        """记一条非 agent 执行事件（run_start / stage_start / prefill_finished / run_error ...）。"""
        with self._lock:
            if self._closed:
                return
            self._seq += 1
            payload = {
                "ts": _iso(),
                "run_id": self.run_id,
                "event": event,
                "seq": self._seq,
                **{str(key): _compact(value) for key, value in fields.items()},
            }
            try:
                self._write_json(payload)
                self._write_text(self._render_event(payload))
            except Exception as exc:  # noqa: BLE001 —— 日志失败不影响主流程
                self._error = repr(exc)
                self._warn_once()

    def _render_event(self, payload: Mapping[str, Any]) -> str:
        """人读日志里的事件；agents_ready 的清单单独排版，避免一行里塞几千字 JSON。"""
        event = payload.get("event")
        prefix = f"{payload['ts']} [{event}]"
        if event == "agents_ready":
            agents = payload.get("agents") or {}
            lines = [f"{prefix} 共 {len(agents)} 个 agent"]
            for name, item in agents.items():
                if not isinstance(item, Mapping):
                    lines.append(f"  {name}: {item}")
                    continue
                lines.append(
                    f"  {name}: llm={item.get('llm')} model={item.get('model')} "
                    f"table={item.get('table_path') or '-'} prompt="
                    f"{json.dumps(item.get('prompt_paths') or {}, ensure_ascii=False)}"
                )
            return "\n".join(lines)
        detail = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in payload.items()
            if key not in {"ts", "run_id", "event", "seq"}
        )
        return f"{prefix} {detail}".rstrip()

    def agents_ready(self, agents: Mapping[str, Any]) -> None:
        """记录本轮实际建了哪些 agent、各自读的提示语/配置表（只写一次清单）。"""
        manifest: dict[str, Any] = {}
        for name, agent in agents.items():
            item: dict[str, Any] = {
                "model": getattr(agent, "model_key", None),
                "llm": bool(getattr(agent, "llm", False)),
                "prompt_paths": getattr(agent, "prompt_paths", None),
                "table_path": getattr(agent, "table_path", None),
                "cacheable_sources": list(getattr(agent, "CACHEABLE_SOURCES", ()) or ()),
            }
            describe = getattr(agent, "describe", None)
            if callable(describe):
                try:
                    item["describe"] = describe()
                except Exception as exc:  # noqa: BLE001 —— 清单描述失败不阻断日志
                    item["describe"] = f"<describe 失败：{exc!r}>"
            manifest[str(name)] = item
        self.event("agents_ready", agents=manifest)

    def agent_finished(
        self,
        *,
        agent: str,
        node: str,
        result: Any = None,
        status: str | None = None,
        source: str | None = None,
        elapsed_seconds: float = 0.0,
        model: str | None = None,
        llm: bool | None = None,
        inputs: Any = None,
        output: Any = None,
        error: str | None = None,
        side: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """记一次 agent 执行完成。

        Args:
            agent/node: agent 名与图节点名（多数时候相同，synonym 图里才可能分开）。
            result: agent.run 的原始返回值；用来推导 status/source 与 output 摘要。
            status/source: 显式指定时优先于推导（alter 的坏记录/未命中走这里）。
            elapsed_seconds: 本次 run 的墙上时间。
            inputs: agent.run 的实参摘要；不传则 input 为空。
            output: 显式输出摘要；不传则从 result 按 agent 类型挑字段。
            error: 异常文本；非空时 status 记为 error。
            side: "a"/"b"/None，成对图区分原句与改写句。
            context: 行号、候选等记录级上下文，会平铺到日志对象顶层。
        """
        derived_status, derived_source = _classify(agent, result)
        if status is None:
            status = derived_status
        if source is None:
            source = derived_source
        if error:
            status = "error"
        if status not in {"ok", "skipped", "error"}:
            status = "error" if source in _ERROR_SOURCES else "ok"

        elapsed_seconds = max(0.0, float(elapsed_seconds or 0.0))
        output_summary = output if output is not None else _summarize_output(agent, result)
        input_summary = _summarize_inputs(inputs) if inputs is not None else None
        context_summary = {
            str(key): _compact(value) for key, value in (context or {}).items()
        }

        with self._lock:
            if self._closed:
                return
            self._seq += 1
            self._records += 1
            stats = self._stats.setdefault(
                agent,
                {
                    "calls": 0,
                    "ok": 0,
                    "skipped": 0,
                    "error": 0,
                    "elapsed_seconds": 0.0,
                    "sources": {},
                },
            )
            stats["calls"] += 1
            stats[status] = stats.get(status, 0) + 1
            stats["elapsed_seconds"] = round(
                float(stats["elapsed_seconds"]) + elapsed_seconds, 3
            )
            if source:
                sources = stats["sources"]
                sources[source] = sources.get(source, 0) + 1

            payload = {
                "ts": _iso(),
                "run_id": self.run_id,
                "seq": self._seq,
                "event": "agent",
                "agent": agent,
                "node": node,
                "side": side,
                "status": status,
                "source": source,
                "elapsed_ms": round(elapsed_seconds * 1000, 1),
                "model": model,
                "llm": llm,
                **context_summary,
                "input": input_summary,
                "output": _compact(output_summary),
            }
            if error:
                payload["error"] = _compact(error)
            try:
                self._write_json(payload)
                self._write_text(self._render_agent(payload))
            except Exception as exc:  # noqa: BLE001 —— 日志失败不影响主流程
                self._error = repr(exc)
                self._warn_once()

    def _render_agent(self, payload: Mapping[str, Any]) -> str:
        """人读日志的一行（必要时跟两行缩进详情）。"""
        side = payload.get("side") or "-"
        pieces = [
            f"{payload['ts']} [{payload.get('agent')}]",
            f"node={payload.get('node')}",
            f"side={side}",
        ]
        for key in ("line_no", "candidate"):
            if payload.get(key) is not None:
                pieces.append(f"{key}={json.dumps(payload[key], ensure_ascii=False)}")
        pieces.extend(
            [
                f"status={payload.get('status')}",
                f"source={payload.get('source') or '-'}",
                f"elapsed={float(payload.get('elapsed_ms') or 0) / 1000:.3f}s",
            ]
        )
        if payload.get("model"):
            pieces.append(f"model={payload['model']}")
        lines = [" ".join(pieces)]
        if payload.get("input") is not None:
            lines.append(f"    in: {json.dumps(payload['input'], ensure_ascii=False)}")
        if payload.get("output") is not None:
            lines.append(f"    out: {json.dumps(payload['output'], ensure_ascii=False)}")
        if payload.get("error"):
            lines.append(f"    error: {json.dumps(payload['error'], ensure_ascii=False)}")
        return "\n".join(lines)

    def finish(
        self,
        *,
        agent_stats: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """写收尾汇总并关闭文件；可重复调用（第二次返回同一份汇总）。失败不抛异常。"""
        with self._lock:
            if self._closed:
                return dict(self._summary or {})
            finished_at = time.time()
            summary = {
                "run_id": self.run_id,
                "event": "run_end",
                "ts": _iso(finished_at),
                "started_at": _iso(self._started_at),
                "finished_at": _iso(finished_at),
                "elapsed_seconds": round(finished_at - self._started_at, 3),
                "records": self._records,
                "log_file": str(self.path) if self.path is not None else None,
                "jsonl_file": str(self.jsonl_path) if self.jsonl_path is not None else None,
                "agents": _compact(self._stats),
                "agent_stats": _compact(agent_stats or {}),
                **{str(key): _compact(value) for key, value in (extra or {}).items()},
            }
            if self._error:
                summary["log_error"] = self._error
            try:
                self._write_json(summary)
                self._write_text(self._render_summary(summary))
            except Exception as exc:  # noqa: BLE001 —— 日志失败不影响主流程
                self._error = repr(exc)
                self._warn_once()
            self._closed = True
            self._summary = summary
            self._close_files()
        return dict(summary)

    def _render_summary(self, summary: Mapping[str, Any]) -> str:
        lines = [
            "---",
            "# 汇总",
            f"结束时间: {summary.get('finished_at')}",
            f"总耗时: {summary.get('elapsed_seconds', 0)} 秒",
            f"agent 执行记录: {summary.get('records', 0)} 条",
            "按 agent:",
        ]
        agents = summary.get("agents") or {}
        if agents:
            for name, stats in agents.items():
                sources = json.dumps(stats.get("sources") or {}, ensure_ascii=False)
                lines.append(
                    f"  {name}: 调用={stats.get('calls', 0)}, "
                    f"ok={stats.get('ok', 0)}, skipped={stats.get('skipped', 0)}, "
                    f"error={stats.get('error', 0)}, "
                    f"耗时={stats.get('elapsed_seconds', 0)}s, source={sources}"
                )
        else:
            lines.append("  （无）")
        agent_stats = summary.get("agent_stats") or {}
        if agent_stats:
            lines.append("模型调用统计（BaseAgent.stats）:")
            for name, stats in agent_stats.items():
                lines.append(f"  {name}: {json.dumps(stats, ensure_ascii=False)}")
        if summary.get("log_error"):
            lines.append(f"日志自身错误: {summary['log_error']}")
        if summary.get("log_file"):
            lines.append(f"日志文件: {summary['log_file']}")
        if summary.get("jsonl_file"):
            lines.append(f"JSONL 文件: {summary['jsonl_file']}")
        return "\n".join(lines)

    def __enter__(self) -> "AgentLog":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.event("run_error", error=repr(exc))
        self.finish()
        return False
