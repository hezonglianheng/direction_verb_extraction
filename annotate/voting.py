# encoding: utf8

"""多模型投票：把每个模型跑出的同一条候选汇总成一条最终标注。

本模块是**纯函数**，不 import langchain / LTP，也不读文件：
    - ``normalize_weight``      校验模型配置里的 voting_weight
    - ``ballot_label``          从一条结果记录里取出该模型的“票”
    - ``aggregate_votes``       按权重做简单多数；平票、无有效票单独标出
    - ``combine_model_records`` 按 (line_no, candidate.index, …) 对齐多个模型的记录并合成最终记录
    - ``summarize_voting``      汇总 .summary.json 里的投票统计
    - ``model_output_paths``    多模型各自的原始结果文件路径（不碰文件系统）

“简单多数”采用**唯一最高票**（plurality）：得分最高的标签只有一个就采纳，
出现两个及以上同分最高标签即平票，记为 discarded，不把任何一方写成最终标注。
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path

VOTING_WEIGHT_FIELD = "voting_weight"
"""模型配置里的权重字段名；缺省为 1，必须是 >= 1 的整数"""

VOTING_ACCEPTED = "accepted"
"""投票状态：有唯一最高票，采纳该标签"""
VOTING_TIE = "tie"
"""投票状态：最高票同分，数据舍弃"""
VOTING_NO_VALID_VOTES = "no_valid_votes"
"""投票状态：没有任何模型给出有效票（全部失败/未走到判定）"""


def normalize_weight(value, *, model_name: str = "") -> int:
    """把配置里的 voting_weight 规范成 >= 1 的整数

    缺省（None/未配置）为 1。刻意拒绝 bool、字符串、0、负数与小数：
    票数写成字符串或 0 时，投票结果会静默偏离“权重代表票数”的直觉。
    """
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int):
        where = f"模型 {model_name!r} 的 " if model_name else ""
        raise ValueError(f"{where}{VOTING_WEIGHT_FIELD} 必须是整数，收到 {value!r}")
    if value < 1:
        where = f"模型 {model_name!r} 的 " if model_name else ""
        raise ValueError(f"{where}{VOTING_WEIGHT_FIELD} 必须 >= 1，收到 {value!r}")
    return value


def record_identity(record: dict) -> tuple[int, int, int, str]:
    """一条结果记录在多个模型之间的对齐键

    不用 ``id``：它是“来源文档编号-源行号-第几个目标词位置-第几个替换句”的次序编号，
    换规则表就会错位；这里用落盘侧本来就唯一的
    ``(line_no, candidate.index, candidate.focus_index, candidate.replacement)``。
    其中 candidate.index 是“整条语料全位置展开后的第几个待选句”，同一份输入下各模型一致。

    缺 line_no 的坏记录仍给一个可排序的键（-1），不抛异常打断投票。
    """
    line_no = record.get("line_no")
    candidate = record.get("candidate") or {}
    index = candidate.get("index")
    focus_index = candidate.get("focus_index")
    replacement = candidate.get("replacement")
    return (
        line_no if isinstance(line_no, int) and not isinstance(line_no, bool) else -1,
        index if isinstance(index, int) and not isinstance(index, bool) else 0,
        focus_index if isinstance(focus_index, int) and not isinstance(focus_index, bool) else -1,
        replacement if isinstance(replacement, str) else "",
    )


def _pair_ballot(record: dict) -> tuple[str | None, str]:
    """pair / synonym 图的票：先看闸门与语义闸门，再看同义判定

    把“不是现实空间用法”“句子不通顺”也当成一种**有效标签**，而不是弃权：
    模型之间对“这条语料该不该进入同义判定”的分歧，本身就是标注结果的一部分；
    若把它们当弃权，两个模型判 not_spatial、一个模型判 same 时，最终会错误地采纳 same。
    """
    reality = record.get("reality") or {}
    if reality.get("is_spatial_scene") is False:
        return "not_spatial", reality.get("reason") or "关注词不是现实空间用法"

    semantic = record.get("semantic") or {}
    failed = [
        side
        for side in ("a", "b")
        if (semantic.get(side) or {}).get("is_fluent") is False
    ]
    if failed:
        return "not_fluent_" + "".join(failed), "；".join(
            (semantic.get(side) or {}).get("reason") or f"{side} 侧不通顺" for side in failed
        )

    status = str(record.get("status") or "")
    if status.startswith("feature_error"):
        # 特征失败时 judge 的结论缺少下游依据，不能当成与其他模型同口径的一票
        return None, record.get("status_reason") or "特征提取失败，未形成有效判定"

    judge = record.get("judgement") or {}
    same_meaning = judge.get("same_meaning")
    if same_meaning is True:
        return "same", judge.get("reason") or ""
    if same_meaning is False:
        return "different", judge.get("reason") or ""
    return None, record.get("status_reason") or f"未给出同义判定（status={status or 'unknown'}）"


def _single_ballot(record: dict) -> tuple[str | None, str]:
    """single 图的票：闸门/通顺性之外，最终标注就是 schema 名"""
    reality = record.get("reality") or {}
    if reality.get("is_spatial_scene") is False:
        return "not_spatial", reality.get("reason") or "关注词不是现实空间用法"

    semantic = record.get("semantic") or {}
    if semantic.get("is_fluent") is False:
        return "not_fluent", semantic.get("reason") or "句子不通顺"
    if semantic and semantic.get("is_fluent") is None:
        return None, semantic.get("reason") or "语义闸门未给出结论"

    status = str(record.get("status") or "")
    if status.startswith("feature_error"):
        # 与 pair 路径同口径：特征提取失败时 schema 缺少下游依据，不能当成有效票
        return None, record.get("status_reason") or "特征提取失败，未形成有效 schema"

    schema = record.get("schema") or {}
    name = schema.get("name")
    if isinstance(name, str) and name:
        return f"schema:{name}", schema.get("reason") or ""
    return None, record.get("status_reason") or f"未给出 schema（status={record.get('status')}）"


def _alter_check_ballot(record: dict) -> tuple[str | None, str]:
    """alter_check 图的票：两侧通顺性组成一个复合标签，例如 ``fluent:a=1,b=0``"""
    reality = record.get("reality") or {}
    if reality.get("is_spatial_scene") is False:
        return "not_spatial", reality.get("reason") or "关注词不是现实空间用法"

    semantic = record.get("semantic") or {}
    verdicts = {}
    reasons = []
    for side in ("a", "b"):
        item = semantic.get(side) or {}
        value = item.get("is_fluent")
        if value is None:
            return None, item.get("reason") or f"{side} 侧未给出通顺性判断"
        verdicts[side] = "1" if value else "0"
        if item.get("reason"):
            reasons.append(f"{side}: {item['reason']}")
    return f"fluent:a={verdicts['a']},b={verdicts['b']}", "；".join(reasons)


def ballot_label(graph_name: str, record: dict) -> tuple[str | None, str]:
    """取一条记录在该图下的投票标签；返回 ``(label, reason)``，label 为 None 表示弃权

    各图的标签口径：
        pair / synonym   not_spatial / not_fluent_a|b|ab / same / different
        single           not_spatial / not_fluent / schema:<名字>
        alter_check      not_spatial / fluent:a=1,b=0 这类复合标签

    弃权只发生在“模型确实没给出结论”的情形（调用失败、判定为空、特征失败）；
    明确的否定结论（非空间、不通顺）是有效票，不是弃权。
    """
    if graph_name in ("pair", "synonym"):
        return _pair_ballot(record)
    if graph_name == "single":
        return _single_ballot(record)
    if graph_name == "alter_check":
        return _alter_check_ballot(record)
    raise ValueError(f"没有为图 {graph_name!r} 定义投票口径")


def aggregate_votes(
    ballots: dict[str, dict],
    weights: dict[str, int],
    model_order: list[str],
) -> dict:
    """按权重聚合一次投票

    Args:
        ballots: {模型名: {"label": str|None, "reason": str}}；label=None 表示弃权
        weights: {模型名: 正整数票数}
        model_order: 配置顺序；平票时按它决定“代表模型”的先后（只用于展示/取记录）

    Returns:
        dict: 可直接放进结果记录的 ``voting`` 块，含 status / scores / winner / ballots 等。

    “简单多数”= 加权得分唯一最高。两个及以上标签同分最高即平票。
    """
    scores: Counter[str] = Counter()
    for model_name, ballot in ballots.items():
        label = (ballot or {}).get("label")
        if label is None:
            continue
        scores[label] += int(weights.get(model_name, 1))

    valid_weight = sum(scores.values())
    configured_weight = sum(int(weights.get(model, 1)) for model in model_order)
    missing_models = [model for model in model_order if model not in ballots]
    invalid_models = [
        model for model in model_order if model in ballots and (ballots[model] or {}).get("label") is None
    ]

    voting = {
        "scores": dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))),
        "valid_weight": valid_weight,
        "configured_weight": configured_weight,
        "missing_models": missing_models,
        "invalid_models": invalid_models,
        "incomplete": bool(missing_models or invalid_models),
        "ballots": {
            model: {
                "label": (ballots.get(model) or {}).get("label"),
                "reason": (ballots.get(model) or {}).get("reason") or "",
                "weight": int(weights.get(model, 1)),
            }
            for model in model_order
        },
    }

    if not scores:
        voting.update(
            {
                "status": VOTING_NO_VALID_VOTES,
                "winner_label": None,
                "winner_model": None,
                "tie_labels": [],
            }
        )
        return voting

    top = max(scores.values())
    top_labels = [label for label, score in scores.items() if score == top]
    if len(top_labels) > 1:
        voting.update(
            {
                "status": VOTING_TIE,
                "winner_label": None,
                "winner_model": None,
                "tie_labels": sorted(top_labels),
            }
        )
        return voting

    winner_label = top_labels[0]
    winner_model = next(
        (model for model in model_order if (ballots.get(model) or {}).get("label") == winner_label),
        None,
    )
    voting.update(
        {
            "status": VOTING_ACCEPTED,
            "winner_label": winner_label,
            "winner_model": winner_model,
            "tie_labels": [],
        }
    )
    return voting


def _discard_reason(voting: dict) -> str:
    """把平票/无有效票写成人能读的一句 status_reason"""
    scores = "，".join(f"{label}={score}" for label, score in (voting.get("scores") or {}).items())
    if voting.get("status") == VOTING_NO_VALID_VOTES:
        return f"多模型投票：没有有效票（{scores or '全部模型弃权或缺失'}），数据舍弃"
    return f"多模型投票平票（{scores or '无有效票'}），数据舍弃"


def _mark_admission(record: dict, reason_code: str, reason: str) -> None:
    """把投票结论写进 admission：剔除准入并转人工复核

    不改各维度原有的取值（reality/fluency/...），只加一条复核理由；
    这样既保留“这条记录实际判成了什么”，又不会让下游误把它当成定版数据。
    """
    admission = dict(record.get("admission") or {})
    admission["is_admitted"] = False
    admission["review_required"] = True
    reasons = list(admission.get("reasons") or [])
    if reason_code not in reasons:
        reasons.append(reason_code)
    admission["reasons"] = reasons
    if reason:
        admission["voting_reason"] = reason
    record["admission"] = admission


def combine_model_records(
    graph_name: str,
    model_records: dict[str, list[dict]],
    weights: dict[str, int],
    model_order: list[str],
    *,
    output_file: str | Path | None = None,
) -> list[dict]:
    """把多个模型的记录按候选对齐、投票、合成最终记录

    Args:
        graph_name: pair / single / alter_check / synonym
        model_records: {模型名: 该模型的全部结果记录}
        weights: {模型名: 正整数票数}
        model_order: 配置顺序（决定代表记录与输出顺序的稳定性）
        output_file: 最终投票结果文件路径；会写进每条记录的 output_file

    Returns:
        list[dict]: 按 ``record_identity`` 排序的最终记录。每条都带 ``voting`` 块；
        采纳的记录保留代表模型的完整字段，平票/无有效票的记录 status=discarded。

    代表记录的选择是**确定性的**：按 model_order 取第一个投出获胜标签的模型。
    这样同一份输入重跑时最终记录的 features/schema 等内容不会随机漂移。
    """
    model_order = list(model_order)
    indexed: dict[str, dict[tuple, dict]] = {}
    duplicates: dict[str, set[tuple]] = {}
    for model in model_order:
        by_key: dict[tuple, dict] = {}
        for record in model_records.get(model) or []:
            key = record_identity(record)
            if key in by_key:
                # 同一模型对同一候选出现多条记录：不能静默取第一条（结果会由文件先后决定）。
                # 该模型在这个候选上直接弃权，并让最终记录按“投票不完整”转人工复核。
                duplicates.setdefault(model, set()).add(key)
                continue
            by_key[key] = record
        indexed[model] = by_key

    all_keys = sorted({key for by_key in indexed.values() for key in by_key})
    combined: list[dict] = []

    for key in all_keys:
        present: dict[str, dict] = {
            model: indexed[model][key] for model in model_order if key in indexed[model]
        }
        ballots: dict[str, dict] = {}
        duplicate_models: list[str] = []
        for model in model_order:
            record = present.get(model)
            if record is None:
                continue
            if key in duplicates.get(model, ()):
                duplicate_models.append(model)
                ballots[model] = {
                    "label": None,
                    "reason": "同一模型对同一候选出现重复记录，已按弃权处理",
                }
                continue
            label, reason = ballot_label(graph_name, record)
            ballots[model] = {"label": label, "reason": reason}

        voting = aggregate_votes(ballots, weights, model_order)
        voting["graph"] = graph_name
        voting["duplicate_models"] = duplicate_models
        voting["identity"] = {
            "line_no": key[0],
            "candidate_index": key[1],
            "focus_index": key[2],
            "replacement": key[3],
        }

        # 代表模型：采纳时取第一个获胜者；平票/无有效票时取配置顺序里第一个有记录的模型，
        # 只为给记录保留句对、来源等共同字段，标注结论已由 status=discarded 明确否定。
        representative = voting.get("winner_model") or next(
            (model for model in model_order if model in present), None
        )
        if representative is None:
            continue
        final = copy.deepcopy(present[representative])
        final["model_name"] = "voting"
        final["voting"] = voting
        if output_file is not None:
            final["output_file"] = str(output_file)

        if voting["status"] == VOTING_ACCEPTED:
            final["voting_label"] = voting["winner_label"]
            if voting["incomplete"]:
                _mark_admission(final, "voting_incomplete", "部分模型缺失或弃权，投票不完整")
        else:
            reason = _discard_reason(voting)
            final["voting_label"] = None
            final["status"] = "discarded"
            final["status_reason"] = reason

            if graph_name in ("pair", "synonym"):
                judge = dict(final.get("judgement") or {})
                judge["same_meaning"] = None
                judge["source"] = f"voting_{voting['status']}"
                judge["reason"] = reason
                final["judgement"] = judge
            elif graph_name == "single":
                schema = dict(final.get("schema") or {})
                schema["name"] = None
                schema["source"] = f"voting_{voting['status']}"
                schema["reason"] = reason
                final["schema"] = schema
            elif graph_name == "alter_check":
                # alter_check 的最终标注就是两侧通顺性：平票时不能留下代表模型的布尔值，
                # 否则 alter_check_tally 会把它当成正式结论统计。
                semantic = dict(final.get("semantic") or {})
                for side in ("a", "b"):
                    item = dict(semantic.get(side) or {})
                    item["is_fluent"] = None
                    item["source"] = f"voting_{voting['status']}"
                    item["reason"] = reason
                    semantic[side] = item
                final["semantic"] = semantic
            _mark_admission(final, f"voting_{voting['status']}", reason)

        combined.append(final)

    return combined


def summarize_voting(records: list[dict]) -> dict:
    """汇总投票结果，写进 .summary.json 的 voting 块"""
    statuses: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    duplicate: Counter[str] = Counter()
    for record in records:
        voting = record.get("voting") or {}
        status = voting.get("status") or "unknown"
        statuses[status] += 1
        if voting.get("incomplete"):
            statuses["incomplete"] += 1
        if status == VOTING_ACCEPTED and voting.get("winner_label"):
            labels[voting["winner_label"]] += 1
        for model in voting.get("missing_models") or []:
            missing[model] += 1
        for model in voting.get("invalid_models") or []:
            invalid[model] += 1
        for model in voting.get("duplicate_models") or []:
            duplicate[model] += 1
    return {
        "records": len(records),
        "accepted": statuses.get(VOTING_ACCEPTED, 0),
        "tie": statuses.get(VOTING_TIE, 0),
        "no_valid_votes": statuses.get(VOTING_NO_VALID_VOTES, 0),
        "discarded": statuses.get(VOTING_TIE, 0) + statuses.get(VOTING_NO_VALID_VOTES, 0),
        "incomplete": statuses.get("incomplete", 0),
        "labels": dict(labels.most_common()),
        "missing_by_model": dict(missing.most_common()),
        "invalid_by_model": dict(invalid.most_common()),
        "duplicate_by_model": dict(duplicate.most_common()),
    }


def safe_model_stem(model_name: str) -> str:
    """把模型配置名变成安全的文件名主干（防路径分隔符/相对路径注入）"""
    stem = re.sub(r"[^0-9A-Za-z._-]+", "_", str(model_name)).strip("._-")
    return stem or "model"


def model_output_paths(out_path: str | Path, model_names: list[str]) -> tuple[Path, dict[str, Path]]:
    """多模型原始结果的路径：``<输出主干>.models/00_<模型>.jsonl``

    前缀序号保证 sanitize 之后重名（例如 ``a/b`` 与 ``a_b``）也不会互相覆盖；
    原始结果放在单独目录，最终投票文件仍是用户 ``--out-file`` 指定的那一个。
    """
    out_path = Path(out_path)
    raw_dir = out_path.parent / f"{out_path.stem}.models"
    paths: dict[str, Path] = {}
    for index, model_name in enumerate(model_names):
        paths[model_name] = raw_dir / f"{index:02d}_{safe_model_stem(model_name)}.jsonl"
    return raw_dir, paths
