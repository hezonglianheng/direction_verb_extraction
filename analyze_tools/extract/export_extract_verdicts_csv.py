#!/usr/bin/env python3
# encoding: utf8

"""把 extract 模块的方位词筛选结果导出为逐词判定 CSV。

输入是 ``python extract`` / ``pipeline.py`` 落盘的原句 JSONL。模型的逐词判定在
``screening.verdicts[]`` 里：``extractable`` 为 ``True``/``False`` 表示模型判定过；
实际存在的 verdict 中 ``None``（典型为 limit/error）导出为 ``unknown``。真实
``no_llm`` 记录是记录级状态，``screening.verdicts`` 为空，只计入读取记录数，不产生
CSV 行。CSV 同时写出方位词在拼接原句中的字符偏移，供比较脚本稳定对齐（不受重新分词
影响）。

目标物、参照物、动词来自 ``localization_grounding.candidates[]``，按
``verdict.index == candidate.index`` 对应。混合判定的整句里，被判 false 的候选会被
前序流程裁掉，或 grounding 本身失败（``candidates=[]``），此时角色留空，不猜测补全。

一行对应一个方位词判定，因此同一句可能有多行。输出为 UTF-8 BOM 的 CSV，方便 Excel
直接打开；不按句子或词形去重，保持输入顺序。

只支持 extract 原句；replace 句对请使用
``analyze_tools/replace/export_replace_pairs_csv.py``，本脚本会拒绝句对输入。

用法（不传输入时自动选 results/extract_result 下最新的
``location_corpus_with_locative_split_*.jsonl``）：
    .venv/bin/python analyze_tools/extract/export_extract_verdicts_csv.py
    .venv/bin/python analyze_tools/extract/export_extract_verdicts_csv.py \
        results/extract_result/location_corpus_with_locative_split_20261031.jsonl
    ./analyze_tools/extract/export_extract_verdicts_csv.py

默认输出到输入文件旁：
    results/extract_result/location_corpus_with_locative_split_20261031.screening_verdicts.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results" / "extract_result"
CORPUS_GLOB = "location_corpus_with_locative_split_*.jsonl"

CSV_COLUMNS = [
    "JSONL行号",
    "原句",
    "源文件",
    "方位词",
    "方位词下标",
    "方位词字符偏移",
    "目标物",
    "参照物",
    "动词",
    "模型判定",
    "判定理由",
    "判定来源",
    "整句筛选状态",
]


def default_output_path(input_path: str | Path) -> Path:
    """与输入同目录、同名加 ``.screening_verdicts.csv``，避免覆盖已有结果。"""
    path = Path(input_path)
    return path.with_suffix(".screening_verdicts.csv")


def latest_corpus_path(results_dir: str | Path | None = None) -> Path:
    """返回结果目录下最新的 ``location_corpus_with_locative_split_*.jsonl``。

    直接运行脚本且不传输入时使用；按修改时间排序，时间相同再按文件名兜底，避免依赖
    当前工作目录。
    """
    directory = Path(results_dir) if results_dir is not None else RESULTS_DIR
    candidates = sorted(
        directory.glob(CORPUS_GLOB),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 {directory} 下找不到 {CORPUS_GLOB}；请显式传入输入 JSONL 路径"
        )
    return candidates[-1]


def sentence_text(value: Any) -> str:
    """还原原句：新结果存 token 列表，旧结果可能直接存字符串。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(token for token in value if isinstance(token, str))
    if value is None:
        return ""
    return str(value)


def _cell(value: Any) -> str:
    """CSV 空值统一为空串，不写 None/null。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _node_word(node: Any) -> str:
    if isinstance(node, dict):
        return _cell(node.get("word"))
    return ""


def _locative_offset(
    sentence: Any, index: Any, word: Any, source_file: Any, line_no: int
) -> str:
    """方位词在拼接原句中的字符偏移；无法唯一确定时直接报错。"""
    needle = _cell(word)
    where = f"{_cell(source_file) or '<未知源文件>'} 第 {line_no} 行"
    if isinstance(sentence, list):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(sentence)
        ):
            raise ValueError(f"{where} verdict index 非法：{index!r}")
        token = sentence[index]
        if not isinstance(token, str):
            raise ValueError(f"{where} index={index} 的 token 不是字符串：{token!r}")
        if token != needle:
            raise ValueError(
                f"{where} index={index} 的 token={token!r} 与 word={needle!r} 不一致"
            )
        return str(
            sum(len(item) for item in sentence[:index] if isinstance(item, str))
        )
    if isinstance(sentence, str):
        positions = []
        start = 0
        while needle:
            found = sentence.find(needle, start)
            if found < 0:
                break
            positions.append(found)
            start = found + 1
        if len(positions) == 1:
            return str(positions[0])
        raise ValueError(
            f"{where} 无法从字符串原句确定方位词字符偏移："
            f"word={needle!r} 出现 {len(positions)} 次；请提供 token 列表"
        )
    raise ValueError(f"{where} sentence 类型非法：{type(sentence).__name__}")


def _verdict_rows(record: dict[str, Any], line_no: int) -> Iterator[dict[str, str]]:
    """产出一条记录里所有逐词判定行（true/false/unknown）。"""
    screening = record.get("screening")
    if not isinstance(screening, dict):
        return
    verdicts = screening.get("verdicts")
    if not isinstance(verdicts, list):
        return

    grounding = record.get("localization_grounding")
    candidates = grounding.get("candidates") if isinstance(grounding, dict) else None
    if not isinstance(candidates, list):
        candidates = []
    candidates_by_index = {
        candidate["index"]: candidate
        for candidate in candidates
        if isinstance(candidate, dict) and "index" in candidate
    }

    sentence = sentence_text(record.get("sentence"))
    source_file = _cell(record.get("source_file"))
    status = _cell(screening.get("status"))

    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        extractable = verdict.get("extractable")
        # true/false 是模型判定；实际 verdict 里的 None（典型为 limit/error）是 unknown。
        # 真实 no_llm 记录没有逐词 verdict，不会走到这里，也不会被补造成 unknown 行。
        if extractable is True:
            verdict_text = "true"
        elif extractable is False:
            verdict_text = "false"
        elif extractable is None:
            verdict_text = "unknown"
        else:
            continue
        index = verdict.get("index")
        try:
            candidate = candidates_by_index.get(index)
        except TypeError:
            candidate = None
        if not isinstance(candidate, dict):
            candidate = {}

        yield {
            "JSONL行号": str(line_no),
            "原句": sentence,
            "源文件": source_file,
            "方位词": _cell(verdict.get("word")),
            "方位词下标": _cell(index),
            "方位词字符偏移": _locative_offset(
                record.get("sentence"),
                index,
                verdict.get("word"),
                record.get("source_file"),
                line_no,
            ),
            "目标物": _node_word(candidate.get("target")),
            "参照物": _node_word(candidate.get("reference")),
            "动词": _node_word(candidate.get("verb")),
            "模型判定": verdict_text,
            "判定理由": _cell(verdict.get("reason")),
            "判定来源": _cell(verdict.get("source") or screening.get("source")),
            "整句筛选状态": status,
        }


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except (FileNotFoundError, OSError):
        return False


def export_jsonl_to_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    progress_every: int = 50_000,
    progress_stream=None,
) -> dict[str, Any]:
    """流式读取 JSONL 并写出 CSV，返回统计信息。

    大文件不整体读入内存；输出先写同目录临时文件，成功后再原子替换，异常时不会留下
    半份 CSV，也不会破坏已有输出。
    """
    source = Path(input_path)
    target = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"找不到输入文件：{source}")
    if _same_file(source, target):
        raise ValueError(f"输出路径 {target} 与输入文件 {source} 是同一文件，请指定其他输出路径")
    if target.exists() and not overwrite:
        raise FileExistsError(f"输出文件 {target} 已存在；如需覆盖请显式加 --overwrite")

    target.parent.mkdir(parents=True, exist_ok=True)
    # 即使一条判定都没写出、甚至输入为空，返回的普通 dict 也始终带这两个键；
    # main 才能安全打印零行统计，而不是在写完 header-only CSV 后抛 KeyError。
    stats: Counter[str] = Counter(records=0, rows_written=0)
    fd, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with open(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            with source.open("r", encoding="utf-8") as source_handle:
                for line_no, line in enumerate(source_handle, 1):
                    if not line.strip():
                        stats["blank_lines"] += 1
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{source} 第 {line_no} 行不是合法 JSON：{exc}") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"{source} 第 {line_no} 行不是 JSON 对象")
                    if "pair_schema_version" in record or "pair_screening" in record:
                        raise ValueError(
                            f"{source} 第 {line_no} 行是 replace 句对，不能按 extract 原句导出；"
                            "请使用 analyze_tools/replace/export_replace_pairs_csv.py"
                        )
                    stats["records"] += 1

                    screening = record.get("screening")
                    status = screening.get("status") if isinstance(screening, dict) else None
                    stats[f"record_status_{_cell(status) or 'missing'}"] += 1
                    verdicts = screening.get("verdicts") if isinstance(screening, dict) else None
                    if isinstance(verdicts, list):
                        for verdict in verdicts:
                            if not isinstance(verdict, dict):
                                continue
                            extractable = verdict.get("extractable")
                            if extractable is True:
                                stats["verdict_true"] += 1
                            elif extractable is False:
                                stats["verdict_false"] += 1
                            else:
                                stats["verdict_null"] += 1

                    for row in _verdict_rows(record, line_no):
                        if not row["目标物"] or not row["参照物"]:
                            stats["rows_missing_role"] += 1
                        if row["模型判定"] == "unknown":
                            stats["rows_unknown"] += 1
                        writer.writerow(row)
                        stats["rows_written"] += 1

                    if progress_every and line_no % progress_every == 0:
                        print(
                            f"已读取 {line_no} 行，导出 {stats['rows_written']} 行判定",
                            file=progress_stream or sys.stderr,
                            flush=True,
                        )
        if overwrite:
            os.replace(temp_path, target)
        else:
            # os.replace 会静默替换导出期间才出现的同名文件；link 在目标已存在时原子失败，
            # 保证「默认不覆盖」在提交阶段仍然成立（也能识别悬空符号链接）。
            try:
                os.link(temp_path, target)
            except FileExistsError:
                raise FileExistsError(
                    f"输出文件 {target} 在导出期间已存在；如需覆盖请显式加 --overwrite"
                ) from None
            temp_path.unlink()
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    stats["input_bytes"] = source.stat().st_size
    stats["output_bytes"] = target.stat().st_size
    return dict(stats)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出 extract 原句的逐词 true/false 判定（不支持 replace 句对）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "筛选结果 JSONL；省略时自动选 "
            f"{RESULTS_DIR}/{CORPUS_GLOB} 中最新的文件"
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        help="输出 CSV 路径（默认与输入同目录，名为 <输入名>.screening_verdicts.csv）",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出 CSV")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="每读取多少行打印一次进度；0 表示关闭（默认 50000）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source = Path(args.input) if args.input else latest_corpus_path()
        target = Path(args.out) if args.out else default_output_path(source)
        stats = export_jsonl_to_csv(
            source,
            target,
            overwrite=args.overwrite,
            progress_every=max(0, args.progress_every),
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"已写出 {target}：{stats['rows_written']} 行 "
        f"(true={stats.get('verdict_true', 0)}, false={stats.get('verdict_false', 0)}, "
        f"unknown={stats.get('rows_unknown', 0)})；"
        f"读取 {stats['records']} 条记录"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
