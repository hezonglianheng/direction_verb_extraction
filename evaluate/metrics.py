# encoding: utf8

"""同义判断的评测指标：纯函数，不读文件、不依赖模型。"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

LABEL_SAME = "same"
LABEL_DIFFERENT = "different"
LABELS = (LABEL_SAME, LABEL_DIFFERENT)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_metrics(pairs: Iterable[tuple[str, str | None]]) -> dict:
    """按 (参考答案, 模型预测) 计算准确率、混淆矩阵与分类指标

    ``prediction=None`` 表示模型没有给出可用结论（调用失败、结果缺失、结构化输出
    不合法等）。这类样本计入 ``total`` 与 ``failed``，并在主 ``accuracy`` 里按
    **未答对**处理，避免「失败越多、准确率越高」。另给 ``accuracy_on_success``
    与 ``coverage``，把「只统计成功预测」的口径单独标出。

    混淆矩阵的行是参考答案，列是模型预测；标签顺序固定为 same/different。
    失败样本计入对应参考类别的 support，并作为该类的 false negative——与主
    accuracy 的口径一致，避免「只统计成功预测」把 recall 抬高。
    零分母的 P/R/F1 一律记 0.0，保证报告字段稳定；没有样本时各比率记 None。
    """
    pairs = list(pairs)
    confusion = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    failed_by_reference = Counter()
    predicted_counts = Counter()
    total = success = failed = correct = 0

    for reference, prediction in pairs:
        if reference not in LABELS:
            raise ValueError(f"参考答案只能是 {LABELS}，收到 {reference!r}")
        total += 1
        if prediction is None:
            failed += 1
            failed_by_reference[reference] += 1
            continue
        if prediction not in LABELS:
            raise ValueError(f"预测只能是 {LABELS} 或 None，收到 {prediction!r}")
        success += 1
        predicted_counts[prediction] += 1
        confusion[reference][prediction] += 1
        if reference == prediction:
            correct += 1

    per_label: dict[str, dict] = {}
    for label in LABELS:
        support = sum(confusion[label].values()) + failed_by_reference[label]
        predicted = predicted_counts[label]
        true_positive = confusion[label][label]
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = _rate(true_positive, true_positive + false_positive) or 0.0
        recall = _rate(true_positive, true_positive + false_negative) or 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = sum(item["f1"] for item in per_label.values()) / len(LABELS) if total else None
    weighted_f1 = (
        sum(item["f1"] * item["support"] for item in per_label.values()) / total
        if total
        else None
    )

    return {
        "labels": list(LABELS),
        "total": total,
        "success": success,
        "failed": failed,
        "correct": correct,
        "accuracy": _rate(correct, total),
        "accuracy_on_success": _rate(correct, success),
        "coverage": _rate(success, total),
        "confusion": confusion,
        "failed_by_reference": {label: failed_by_reference[label] for label in LABELS},
        "per_label": per_label,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
