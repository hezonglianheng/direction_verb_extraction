# encoding: utf8

"""标注阶段实际读取的材料路径（轻量、无 LTP/torch 依赖）。

只做路径枚举，不加载配置内容。annotate、pipeline 与路径预检共用这一份口径，
防止 --sync-cache / --batch-cache 指到配置表或提示语时把它们当成可清空的缓存。
"""

from pathlib import Path

from corpus_metadata import CORPUS_TYPE_DIRS, normalize_corpus_type

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def annotation_material_paths(
    corpus_type: str | None, *, root: str | Path | None = None
) -> tuple[Path, ...]:
    """返回本语料类型标注会读取的配置表、示例与提示语文件。

    目录不存在时返回空表；只列**现存文件**，避免把不存在的路径误报成输入。
    """
    base = Path(root) if root is not None else _PROJECT_ROOT
    suffix = CORPUS_TYPE_DIRS[normalize_corpus_type(corpus_type)]
    paths: list[Path] = []

    config_dir = base / "config_files" / suffix
    if config_dir.is_dir():
        paths.extend(sorted(item for item in config_dir.iterdir() if item.is_file()))

    prompt_dir = base / "system_prompt" / "annotate"
    if prompt_dir.is_dir():
        paths.extend(sorted(prompt_dir.rglob("*.txt")))

    extract_dir = base / "system_prompt" / "extract"
    if extract_dir.is_dir():
        paths.extend(sorted(extract_dir.glob("screen_*.txt")))

    return tuple(dict.fromkeys(paths))
