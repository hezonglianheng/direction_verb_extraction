# encoding: utf8
"""按标注步骤、词性和本侧关注词选取 JSON5 示例。"""

from pathlib import Path

import json5

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "system_prompt" / "examples"
_KINDS = {"方位词": "localization", "趋向动词": "direction"}


def load_examples(stage: str, kind: str, *, root: str | Path | None = None) -> list[dict]:
    if stage not in ("feature", "schema") or kind not in _KINDS:
        raise ValueError(f"不支持的示例步骤/词性：{stage!r}/{kind!r}")
    path = (Path(root) if root is not None else EXAMPLES_DIR) / f"{stage}_{_KINDS[kind]}.json5"
    with path.open(encoding="utf-8-sig") as handle:
        groups = json5.load(handle)
    if not isinstance(groups, list):
        raise ValueError(f"{path}：顶层必须是字典列表")
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
