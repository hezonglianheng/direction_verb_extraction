# encoding: utf8

"""把 SpaCE 原始句对数据转成 ``evaluate`` 可读的 JSONL。

``config_files/sample_sentence_pairs.json`` 这类文件是原始参考数据（JSON 数组，字段为
``context1/context2/judge`` 等），并不是 ``annotate`` 产出的评测输入：``evaluate`` 只扫描
JSONL，参考标签取 ``judgement.same_meaning``，句对取 ``pair.a.sentence`` / ``pair.b.sentence``。
直接把它传给 ``--corpus`` 会得到 0 条可评测样本。

本脚本做一层显式、可审计的转换：保留原始身份与提示字段，把 ``judge`` 映射为严格的 bool
参考标签；不把参考 ``reason`` 注入模型提示（``evaluate`` 只把参考记录用于计分与明细）。
每条输入生成唯一的 ``id``，避免原始 ``qid`` 重复时被评测入口当成重复记录跳过。

用法：

    python -m evaluate.import_sentence_pairs config_files/sample_sentence_pairs.json \
        --output results/evaluate/sample_sentence_pairs.jsonl
    python -m evaluate --corpus results/evaluate/sample_sentence_pairs.jsonl --print-prompt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

_META_FIELDS = (
    "source",
    "qid",
    "oid",
    "context1_hint",
    "context2_hint",
    "template",
    "subjects",
    "event",
    "difference",
)
"""原样保留到 ``reference_meta`` 的来源字段；只用于审计，不参与模型提示。"""


def read_rows(path: str | Path) -> list[Any]:
    """读 JSON 数组或逐行 JSONL；两种输入都返回行列表"""
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"{path}：JSON 顶层必须是数组")
        return rows
    rows = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON：{exc}") from exc
    return rows


def convert_row(row: Any, index: int) -> dict:
    """把一条原始句对转成 ``evaluate.data.build_sample`` 认识的形状"""
    if not isinstance(row, dict):
        raise ValueError(f"第 {index} 条不是 JSON 对象")
    sentence_a, sentence_b = row.get("context1"), row.get("context2")
    judge = row.get("judge")
    if not isinstance(sentence_a, str) or not sentence_a.strip():
        raise ValueError(f"第 {index} 条的 context1 必须是非空字符串")
    if not isinstance(sentence_b, str) or not sentence_b.strip():
        raise ValueError(f"第 {index} 条的 context2 必须是非空字符串")
    if not isinstance(judge, bool):
        raise ValueError(f"第 {index} 条的 judge 必须是布尔值 true/false，收到 {judge!r}")

    identity = "-".join(
        str(row[field]) for field in ("source", "qid", "oid") if row.get(field) not in (None, "")
    )
    # 原始 qid 有重复（同一来源下也可能出现），补序号保证评测入口不会按 id 去重丢掉样本。
    record_id = f"{identity or 'pair'}-{index}"

    judgement: dict[str, Any] = {"same_meaning": judge}
    reason = row.get("reason")
    if isinstance(reason, str) and reason.strip():
        judgement["reason"] = reason
    return {
        "id": record_id,
        "pair": {
            "a": {"sentence": sentence_a.strip()},
            "b": {"sentence": sentence_b.strip()},
        },
        "judgement": judgement,
        "reference_meta": {field: row[field] for field in _META_FIELDS if field in row},
    }


def convert_rows(rows: Iterable[Any]) -> list[dict]:
    return [convert_row(row, index) for index, row in enumerate(rows, 1)]


def write_jsonl(records: Iterable[dict], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluate.import_sentence_pairs",
        description="把 SpaCE 原始句对 JSON/JSONL 转成 evaluate 可读的 JSONL（严格校验 judge 为布尔值）",
    )
    parser.add_argument("input", help="原始句对文件：JSON 数组或 JSONL")
    parser.add_argument("--output", "-o", required=True, help="输出的评测用 JSONL 路径")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的输出文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_rows(args.input)
    records = convert_rows(rows)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"输出文件已存在：{output}；请换一个路径或加 --overwrite")
    count = write_jsonl(records, output)
    print(f"已转换 {count} 条 → {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
