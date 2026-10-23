# encoding: utf8

"""把旧结果 jsonl 的顶层 id 迁移成
「来源文档编号-源行号-第几个目标词位置-第几个替换句」。

为什么需要它：新口径只改顶层 id（见 spatial_graphs._record_id），旧文件里的 id 是
`<sentence_id>-<focus_order>-<candidate_order>`、`<源行号>-<位置>-<替换句>` 或更早的裸行号
形式，两份产物不能直接按 id 对账。本脚本读旧 jsonl、按新口径重算 id，写出**新的 jsonl 与
新的 csv**：

    新 jsonl  除 id（以及 schema_version）外，逐字段照抄旧记录，行序按
              (line_no, candidate.index) 重排，与跑批收尾一致
    新 csv    列序与单元格口径完全走 spatial_tools.PAIR_CSV_COLUMNS / write_pairs_csv，
              与 spatial_runner 产出的成对信息 csv 相同

来源文档编号按语料首次出现顺序（1 起），与 st.read_corpus 赋值同一口径：旧结果排过序，
按 (line_no, candidate.index) 扫一遍就是当时的语料顺序；给 `--corpus` 时直接按那份语料
的首次出现顺序还原。行号取 source_line_no，其次 input_line_no / line_no。

第三段按**全部目标词位置**计数（展开不出候选的位置也占号，见 spatial_agents.expand_positions）。
结果记录没有落 sampled_words，抽样语料的「全部位置」无法仅凭结果文件还原：
带 `--corpus` 时从原语料按 (来源文档, 源行号) 取回 sampled_words 再重算；否则对
`focus_word.source == "sampled_word"` 的记录沿用旧 id 里的位置序号并告警。

用法：
    python migrate_result_ids.py results/old.jsonl
    python migrate_result_ids.py old.jsonl -o new.jsonl --csv-out new.csv --corpus sampled.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import spatial_graphs as sg
import spatial_tools as st


def _is_pair_record(record: dict) -> bool:
    """这条记录是否要迁移：成对结果有 pair 块；带 id 的旧成对记录也认。"""
    return "pair" in record or "id" in record


def _source_key(record: dict) -> str:
    """迁移分组用的来源标识：与 sentence_id 首段同一口径，但先看 source 块。"""
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    for key in st.SOURCE_DOCUMENT_KEYS:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return st.source_document_id(record) or ""


def _old_order_parts(old_id) -> tuple[int | None, int | None]:
    """从旧 id 末尾拆出 (focus_order, candidate_order)

    旧 id 形如 `path#1#hash-2-3` 或更早的 `1-2-3`；`rsplit("-", 2)` 只看最后两段，
    路径里的 `-` 不会干扰。0-0（无候选）按缺省处理，由调用方单独记。
    """
    if not isinstance(old_id, str) or not old_id:
        return None, None
    parts = old_id.rsplit("-", 2)
    if len(parts) != 3:
        return None, None
    try:
        focus_order = int(parts[1])
        candidate_order = int(parts[2])
    except ValueError:
        return None, None
    if focus_order < 1 or candidate_order < 1:
        return None, None
    return focus_order, candidate_order


def _line_from_old_id(old_id) -> int | None:
    """兜底：从旧 id 里取行号。

    更早的 `1-2-3` 形式首段就是行号；`1-42-2-3` 形式首段是文档号、第二段是行号；
    带 sentence_id 的旧形式首段形如 `path#42#hash`，取两个 `#` 之间的那一段
    （`?` 占位时返回 None）。
    """
    if not isinstance(old_id, str) or not old_id:
        return None
    head = old_id.rsplit("-", 2)[0]
    if head.isdigit():
        return int(head)
    # 新口径四段：head 是「文档号-行号」；旧 sentence_id 的 head 里也可能有 '-'，
    # 但最后一段是十六进制指纹、不是纯数字，不会误判
    doc_part, _, line_part = head.rpartition("-")
    if doc_part.isdigit() and line_part.isdigit():
        return int(line_part)
    parts = head.rsplit("#", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return int(parts[1])
    return None


def _document_order(record: dict) -> tuple[int, int]:
    """按语料顺序排记录：顶层 line_no 是语料里的物理行号，跨文档也单调

    不能用 st.record_line_no：它优先 source_line_no，而源行号在每篇文档里都从 1 重新数，
    拿它排序会把不同文档交错的记录打乱。坏记录缺 line_no 时退回 source.input_line_no，
    都没有就排到最后，免得抢走第一篇文档的编号。
    """
    line_no = record.get("line_no")
    if not isinstance(line_no, int) or isinstance(line_no, bool):
        source = record.get("source") if isinstance(record.get("source"), dict) else {}
        line_no = source.get("input_line_no")
    index = (record.get("candidate") or {}).get("index")
    return (
        line_no if isinstance(line_no, int) and not isinstance(line_no, bool) else sys.maxsize,
        index if isinstance(index, int) and not isinstance(index, bool) else 0,
    )


def _document_numbers(records: list[dict]) -> dict[str, int]:
    """来源文档编号：按语料首次出现顺序 1、2、3…（与 st.read_corpus 同一口径）

    旧结果按 (line_no, candidate.index) 排过序，按这个顺序扫一遍就是当时的语料顺序，
    --offset 跑出来的文件也会从它遇到的第一篇文档重新数 1。来源标识缺失的记录不占号，
    对应记录迁移后 id 为 None（与程序侧一致）。
    """
    numbers: dict[str, int] = {}
    ordered = sorted(enumerate(records), key=lambda item: (_document_order(item[1]), item[0]))
    for _, record in ordered:
        key = _source_key(record)
        if key and key not in numbers:
            numbers[key] = len(numbers) + 1
    return numbers


def _focus_identity(record: dict):
    """这条记录改的是哪个位置：优先 focus_index，缺了就用词的 (index, word) 兜底。"""
    candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
    focus_index = candidate.get("focus_index")
    if focus_index is None:
        focus_index = (record.get("focus_word") or {}).get("index")
    if focus_index is not None:
        return focus_index
    word = candidate.get("word") or (record.get("focus_word") or {}).get("word")
    return ("word", word) if word else None


def _focus_ordinal(record: dict, sampled_words: list[str] | None) -> int | None:
    """按新口径重算第三段：该位置在 pick_focus_words 返回的全部位置里排第几（1 起）。

    探针记录用结果里的 source.match_tree + filter_methods；有 sampled_words 时一并带上，
    与跑批时的抽样口径一致。找不到位置时返回 None，由调用方回退旧 id。
    """
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    match_tree = source.get("match_tree") or record.get("match_tree")
    filter_methods = source.get("filter_methods") or record.get("filter_methods") or []
    focus_index = _focus_identity(record)
    if focus_index is None or isinstance(focus_index, tuple) or not match_tree:
        return None
    probe = {"match_tree": match_tree, "filter_methods": filter_methods}
    if sampled_words:
        probe["sampled_words"] = sampled_words
    try:
        entries = st.pick_focus_words(probe)
    except (KeyError, TypeError, ValueError):
        return None
    for ordinal, entry in enumerate(entries, start=1):
        if entry.get("index") == focus_index:
            return ordinal
    return None


def build_corpus_index(
    corpus_path: Path,
) -> tuple[dict[tuple[str, int], list[str]], dict[str, int]]:
    """从原语料读回两样东西

    - (来源文档, 源行号) -> sampled_words，供抽样记录重算位置序号；
    - 来源文档 -> 文档编号：st.read_corpus 已按首次出现顺序写好 source_document_no，
      直接照抄，保证与真跑一次这份语料得到的编号一致。
    """
    index: dict[tuple[str, int], list[str]] = {}
    numbers: dict[str, int] = {}
    for record in st.read_corpus(corpus_path):
        source = st.source_document_id(record)
        number = st.record_document_no(record)
        if source and number is not None:
            numbers[source] = number
        words = record.get("sampled_words")
        if not isinstance(words, list) or not words:
            legacy = record.get("sampled_word")
            words = [legacy] if isinstance(legacy, str) and legacy else None
        if not words:
            continue
        line_no = st.record_line_no(record)
        if source and line_no is not None:
            index[(source, line_no)] = [str(word) for word in words if isinstance(word, str) and word]
    return index, numbers


def _merge_document_numbers(
    records: list[dict], doc_numbers: dict[str, int] | None
) -> dict[str, int]:
    """以语料给出的编号为准；语料里没有的文档接在后面，避免和无编号文档撞号"""
    computed = _document_numbers(records)
    if doc_numbers is None:
        return computed
    merged = dict(doc_numbers)
    next_no = max(merged.values(), default=0) + 1
    for key, _ in sorted(computed.items(), key=lambda item: item[1]):
        if key not in merged:
            merged[key] = next_no
            next_no += 1
    return merged


def _new_id(
    document_no: int | None,
    line_no: int | None,
    focus_order: int | None,
    candidate_order: int | None,
) -> str | None:
    if document_no is None or line_no is None or focus_order is None or candidate_order is None:
        return None
    return f"{document_no}-{line_no}-{focus_order}-{candidate_order}"


def migrate_records(
    records: list[dict],
    sampled_index: dict[tuple[str, int], list[str]] | None = None,
    doc_numbers: dict[str, int] | None = None,
) -> tuple[list[dict], dict]:
    """返回（迁移后的记录, 统计）。未迁移的记录原样保留。

    doc_numbers 是「来源文档 -> 编号」；不给就按旧结果的行序现算（见 _document_numbers），
    给了则以它为准（--corpus 走这条，能精确复现真跑一次的编号）。
    """
    numbers = _merge_document_numbers(records, doc_numbers)
    migrated: list[dict] = []
    pending: list[tuple[int, dict, int, int, object, int | None, int | None]] = []
    stats = Counter()
    for record in records:
        if not _is_pair_record(record):
            migrated.append(record)
            stats["skipped_non_pair"] += 1
            continue

        document_no = numbers.get(_source_key(record))
        line_no = st.record_line_no(record)
        old_focus, old_candidate = _old_order_parts(record.get("id"))
        if line_no is None:
            line_no = _line_from_old_id(record.get("id"))

        if document_no is None:
            # 来源文档缺失就编不出首段；与程序侧一致，写 None 而不是硬塞一个号
            stats["no_document"] += 1
            migrated.append(_replace_id(record, None))
            continue
        if line_no is None:
            stats["no_line"] += 1
            migrated.append(_replace_id(record, None))
            continue

        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        has_candidate = isinstance(candidate.get("index"), int) and not isinstance(candidate.get("index"), bool)
        if not has_candidate:
            migrated.append(
                _replace_id(record, f"{document_no}-{line_no}-0-0", document_no)
            )
            stats["no_candidate"] += 1
            continue

        focus_order = None
        sampled_words = None
        if sampled_index is not None:
            sampled_words = sampled_index.get((_source_key(record), line_no))
        if (record.get("focus_word") or {}).get("source") == "sampled_word" and not sampled_words:
            stats["sampled_keep_old"] += 1
            focus_order = old_focus
        else:
            focus_order = _focus_ordinal(record, sampled_words)
            if focus_order is None:
                stats["focus_recompute_failed"] += 1
                focus_order = old_focus

        candidate_order = old_candidate
        if candidate_order is None:
            value = candidate.get("candidate_order")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
                candidate_order = value

        if focus_order is None or candidate_order is None:
            pending.append(
                (
                    len(migrated),
                    record,
                    document_no,
                    line_no,
                    _focus_identity(record),
                    focus_order,
                    candidate_order,
                )
            )
            migrated.append(record)  # 第二遍再替换
        else:
            migrated.append(
                _replace_id(
                    record,
                    _new_id(document_no, line_no, focus_order, candidate_order),
                    document_no,
                )
            )
            stats["migrated_ok"] += 1

    _fill_pending(pending, migrated, stats)
    return migrated, dict(stats)


def _replace_id(record: dict, new_id: str | None, document_no: int | None = None) -> dict:
    """保持 id 在第一个键上（与跑批产物一致），其余字段顺序不变。

    同时把来源文档编号写回 `source.source_document_no`（紧跟 source_document_id），
    迁移后的记录因此与 1.2 版程序产物同形，不必再靠 id 反推。
    """
    out = {"id": new_id}
    for key, value in record.items():
        if key == "id":
            continue
        if key == "source" and isinstance(value, dict):
            source = {name: cell for name, cell in value.items() if name != "source_document_no"}
            if document_no is not None:
                rebuilt = {}
                for name, cell in source.items():
                    rebuilt[name] = cell
                    if name == "source_document_id":
                        rebuilt["source_document_no"] = document_no
                if "source_document_no" not in rebuilt:
                    rebuilt["source_document_no"] = document_no
                source = rebuilt
            value = source
        out[key] = value
    if "schema_version" in out:
        out["schema_version"] = st.RESULT_SCHEMA_VERSION
    return out


def _fill_pending(
    pending: list[tuple[int, dict, int, int, object, int | None, int | None]],
    migrated: list[dict],
    stats: Counter,
) -> None:
    """第二遍：缺第三/第四段的记录按旧输出内的位置关系回填。

    分组键 (来源, 行号)；组内按 candidate.index 排序。第三段取 focus 位置首次出现的顺序，
    第四段取同一位置内 candidate.index 的顺序；已经算出的段原样保留。
    注意：输出里看不到「展开不出候选的位置」，这一步只能按旧口径回填，
    条数会记进统计并在收尾告警。
    """
    groups: dict[
        tuple[str, int],
        list[tuple[int, dict, int, object, int | None, int | None, int | None]],
    ] = defaultdict(list)
    for slot, record, document_no, line_no, focus_key, focus_order, candidate_order in pending:
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        index = candidate.get("index")
        groups[(_source_key(record), line_no)].append(
            (slot, record, document_no, focus_key, index, focus_order, candidate_order)
        )

    for rows in groups.values():
        rows.sort(key=lambda row: row[4] if isinstance(row[4], int) else sys.maxsize)
        focus_ordinals: dict[object, int] = {}
        for _, _, _, focus_key, _, _, _ in rows:
            focus_ordinals.setdefault(focus_key, len(focus_ordinals) + 1)
        per_focus_counter: Counter = Counter()
        for slot, record, document_no, focus_key, _, known_focus, known_candidate in rows:
            per_focus_counter[focus_key] += 1
            focus_order = known_focus if known_focus is not None else focus_ordinals[focus_key]
            candidate_order = (
                known_candidate if known_candidate is not None else per_focus_counter[focus_key]
            )
            line_no = st.record_line_no(record) or _line_from_old_id(record.get("id"))
            migrated[slot] = _replace_id(
                record,
                _new_id(document_no, line_no, focus_order, candidate_order),
                document_no,
            )
            stats["grouped_fallback"] += 1


def _duplicate_ids(records: list[dict]) -> dict[str, int]:
    counter = Counter(
        record.get("id")
        for record in records
        if isinstance(record.get("id"), str) and record.get("id")
    )
    return {key: count for key, count in counter.items() if count > 1}


def _check_paths(input_path: Path, out_path: Path, csv_path: Path, overwrite: bool) -> str | None:
    if not input_path.is_file():
        return f"输入文件不存在：{input_path}"
    if out_path == input_path or csv_path == input_path:
        return "输出路径不能与输入文件相同"
    if out_path == csv_path:
        return "jsonl 与 csv 输出不能是同一个文件"
    if not overwrite:
        for path in (out_path, csv_path):
            if path.exists():
                return f"输出已存在：{path}（如需覆盖请加 --overwrite）"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把旧结果 jsonl 的顶层 id 迁移为"
        "「来源文档编号-源行号-第几个目标词位置-第几个替换句」"
    )
    parser.add_argument("input", type=Path, help="旧结果 jsonl")
    parser.add_argument("-o", "--out", type=Path, help="新 jsonl（默认 <stem>.migrated.jsonl）")
    parser.add_argument("--csv-out", type=Path, help="新 csv（默认与新 jsonl 同名）")
    parser.add_argument(
        "--corpus",
        type=Path,
        help="原/抽样语料：按首次出现顺序还原来源文档编号，并按 (来源文档, 源行号) 取回 "
        "sampled_words 供抽样记录重算位置序号",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出")
    parser.add_argument("--strict", action="store_true", help="有重复 id 或算不出的段时以非零码退出")
    args = parser.parse_args(argv)

    out_path = args.out or args.input.with_name(args.input.stem + ".migrated.jsonl")
    csv_path = args.csv_out or st.pairs_csv_path(out_path)
    problem = _check_paths(args.input, out_path, csv_path, args.overwrite)
    if problem:
        print(f"错误：{problem}", file=sys.stderr)
        return 2

    records = st.read_jsonl(args.input)
    sampled_index = doc_numbers = None
    if args.corpus:
        sampled_index, doc_numbers = build_corpus_index(args.corpus)
    migrated, stats = migrate_records(records, sampled_index, doc_numbers)
    migrated.sort(key=sg.record_order_key)

    duplicates = _duplicate_ids(migrated)
    unresolved = stats.get("no_line", 0) + stats.get("no_document", 0)
    print(f"读入 {len(records)} 条，迁移 {len(records) - stats.get('skipped_non_pair', 0)} 条")
    if stats:
        print("统计：" + "，".join(f"{key}={value}" for key, value in sorted(stats.items())))
    if stats.get("grouped_fallback"):
        print(
            f"注意：{stats['grouped_fallback']} 条缺位置序号，按旧输出内的位置关系回填；"
            "输出里看不到空位置，第三段可能与新跑批不一致"
        )
    if stats.get("sampled_keep_old"):
        print(
            f"注意：{stats['sampled_keep_old']} 条是抽样记录且未提供 --corpus，"
            "位置序号沿用旧 id；需要严格对齐请补 --corpus"
        )
    if stats.get("no_document") or stats.get("no_line"):
        print(
            f"注意：{stats.get('no_document', 0)} 条缺来源文档、{stats.get('no_line', 0)} 条缺行号，"
            "这些记录的新 id 写 null"
        )
    if duplicates:
        print(
            f"注意：有 {len(duplicates)} 个重复 id（同一文档同一句的编号撞了；"
            "多是未给 --corpus 时文档编号与原跑批不一致，属已知限制）"
        )
        for id_value, count in list(duplicates.items())[:5]:
            print(f"  {id_value} × {count}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = out_path.with_name(out_path.name + ".tmp")
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    try:
        st.write_jsonl(migrated, json_tmp)
        total, judged = st.write_pairs_csv(migrated, csv_tmp, st.PAIR_CSV_COLUMNS)
        json_tmp.replace(out_path)
        csv_tmp.replace(csv_path)
    except Exception:
        for path in (json_tmp, csv_tmp):
            path.unlink(missing_ok=True)
        raise

    print(f"新 jsonl：{out_path}（{len(migrated)} 行）")
    print(f"新 csv：{csv_path}（{total} 行，其中 {judged} 行有同义判断）")

    if args.strict and (duplicates or unresolved):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
