# encoding: utf8
"""按标注步骤、词性和关注词／图式对选取 JSON5 示例。

示例与四张配置表同一口径：随词性放在 `config_files/<词性目录>/` 下，各词性目录里
同名（`<步骤>_examples.json5`）。判据随词类不同，示例也按词类独立维护。
"""

from pathlib import Path

import json5

from corpus_metadata import CORPUS_TYPE_DIRS

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "config_files"
"""示例的根目录；每类语料一个子目录（见 corpus_metadata.CORPUS_TYPE_DIRS）"""

EXAMPLE_FILES: dict[str, str] = {
    # 逻辑步骤名 -> 各词性目录里的同名文件（与四张配置表一样，文件名只有一处定义）
    "feature": "feature_examples.json5",
    "schema": "schema_examples.json5",
    "judge": "judge_examples.json5",
}
"""本流水线读的三份示例；值同时是各词性目录里的文件名（单一真源）"""


def examples_path(stage: str, kind: str, *, root: str | Path | None = None) -> Path:
    """示例文件路径：<root>/<词性目录>/<步骤>_examples.json5

    Args:
        stage (str): feature / schema / judge
        kind (str): 语料类型（方位词 / 趋向动词）
        root (str | Path | None): 覆盖 config_files 根目录；测试注入临时目录时仍须按词性
            建子目录。None = 仓库的 config_files/
    """
    if stage not in EXAMPLE_FILES:
        raise ValueError(f"不支持的示例步骤：{stage!r}；可用：{'、'.join(EXAMPLE_FILES)}")
    if kind not in CORPUS_TYPE_DIRS:
        raise ValueError(f"不支持的示例词性：{kind!r}；可用：{'、'.join(CORPUS_TYPE_DIRS)}")
    base = Path(root) if root is not None else EXAMPLES_ROOT
    return base / CORPUS_TYPE_DIRS[kind] / EXAMPLE_FILES[stage]


def load_examples(stage: str, kind: str, *, root: str | Path | None = None) -> list[dict]:
    path = examples_path(stage, kind, root=root)
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
