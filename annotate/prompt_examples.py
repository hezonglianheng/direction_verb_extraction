# encoding: utf8
"""按标注步骤、词性和关注词／图式对选取 JSON5 示例。"""

from pathlib import Path

import json5

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "system_prompt" / "annotate" / "examples"
_KINDS = {"方位词": "localization", "趋向动词": "direction"}


def load_examples(stage: str, kind: str, *, root: str | Path | None = None) -> list[dict]:
    if stage not in ("feature", "schema", "judge") or kind not in _KINDS:
        raise ValueError(f"不支持的示例步骤/词性：{stage!r}/{kind!r}")
    path = (Path(root) if root is not None else EXAMPLES_DIR) / f"{stage}_{_KINDS[kind]}.json5"
    with path.open(encoding="utf-8-sig") as handle:
        groups = json5.load(handle)
    if not isinstance(groups, list):
        raise ValueError(f"{path}：顶层必须是字典列表")
    if stage == "judge":
        _validate_judge_groups(groups, path)
        return groups
    for i, group in enumerate(groups):
        where = f"{path} 第 {i + 1} 组"
        if not isinstance(group, dict) or set(group) != {"words", "examples"}:
            raise ValueError(f"{where}：必须且只能有 words/examples 两个字段")
        words, examples = group["words"], group["examples"]
        if not isinstance(words, list) or not words or not all(isinstance(w, str) and w.strip() == w and w for w in words):
            raise ValueError(f"{where}：words 必须是非空词形列表")
        if not isinstance(examples, list):
            raise ValueError(f"{where}：examples 必须是例句/说明字典列表")
        for example in examples:
            if (not isinstance(example, dict) or set(example) != {"sentence", "instruction"}
                    or not all(isinstance(v, str) and v.strip() for v in example.values())):
                raise ValueError(f"{where}：每条例句必须且只能有非空 sentence/instruction 字符串")
    return groups


def _validate_judge_groups(groups: list[dict], path: Path) -> None:
    for i, group in enumerate(groups):
        where = f"{path} 第 {i + 1} 组"
        if not isinstance(group, dict) or set(group) != {"schema_pair", "sentence_pairs"}:
            raise ValueError(f"{where}：必须且只能有 schema_pair/sentence_pairs 两个字段")
        schemas, pairs = group["schema_pair"], group["sentence_pairs"]
        if (not isinstance(schemas, list) or len(schemas) != 2
                or not all(isinstance(s, str) and s.strip() == s and s for s in schemas)):
            raise ValueError(f"{where}：schema_pair 必须是恰好两个非空、无首尾空白的图式名列表")
        if not isinstance(pairs, list):
            raise ValueError(f"{where}：sentence_pairs 必须是句对字典列表")
        for j, pair in enumerate(pairs):
            location = f"{where} 第 {j + 1} 个句对"
            if not isinstance(pair, dict) or set(pair) != {"context1", "context2", "judge", "instruction"}:
                raise ValueError(f"{location}：必须且只能有 context1/context2/judge/instruction 四个字段")
            for field in ("context1", "context2", "instruction"):
                if not isinstance(pair[field], str) or not pair[field].strip():
                    raise ValueError(f"{location}：{field} 必须是非空字符串")
            if not isinstance(pair["judge"], bool):
                raise ValueError(f"{location}：judge 必须是布尔值 true/false")


def render_examples(groups: list[dict], word: str) -> str:
    """精确匹配词形；命中多组全部拼接，无命中则不注入任何例句。"""
    return "\n\n".join(
        f"例句：{example['sentence']}\n说明：{example['instruction']}"
        for group in groups if word and word in group["words"]
        for example in group["examples"]
    )


def example_block(groups: list[dict], focus: dict | None, stage: str) -> str:
    word = (focus or {}).get("word") or ""
    content = render_examples(groups, word)
    title = "空间特征" if stage == "feature" else "空间图式"
    return f"针对关注词「{word}」的{title}示例：\n{content}" if content else ""


def judge_example_block(groups: list[dict], schema_a: str | None, schema_b: str | None) -> str:
    """精确双向匹配图式对；保留示例句序和分析，多组命中按文件顺序合并。"""
    if not schema_a or not schema_b:
        return ""
    examples = [
        pair
        for group in groups
        if tuple(group["schema_pair"]) in ((schema_a, schema_b), (schema_b, schema_a))
        for pair in group["sentence_pairs"]
    ]
    if not examples:
        return ""
    blocks = ["阅读下列示例句对、判定结果和理由，判定当前输入的两个句子是否能表示相同的空间场景（true/false）。"]
    for i, example in enumerate(examples, 1):
        lines = [f"example{i}:"]
        for field in ("context1", "context2", "judge", "instruction"):
            value = ("true" if example[field] else "false") if field == "judge" else example[field]
            value = value.replace("\n", "\n        ")
            lines.append(f"    {field}: {value}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
