# encoding: utf8
"""初筛词表路径解析：加载端与流水线预检共用同一份优先级。

只依赖标准库，不读取 JSON5、不创建文件，也不导入 LTP/torch 等重依赖。
``sentenceFilters.load_vocab_file`` 与实际读取走这里，``pipeline`` 的只读输入预检
也走这里——否则预检保护的是一个候选，extract 实际读的却是另一个，输出冲突检查
会误报或漏报。
"""

from pathlib import Path


def vocab_path_candidates(
    path: str | Path, *, project_root: str | Path, base_dir: str | Path | None = None
) -> tuple[Path, ...]:
    """按加载顺序列出词表候选路径（不检查存在性）

    顺序与 ``sentenceFilters.load_vocab_file`` 的原口径一致：

    1. 给了 ``base_dir`` 时先试 ``base_dir / path``（规则配置的 ``_vocab_file`` 用）；
    2. ``path`` 是绝对路径时只认它；
    3. 相对路径先试项目根，再试当前工作目录。

    绝对路径与 ``base_dir`` 组合时，``Path(base_dir) / path`` 仍是那个绝对路径，
    因此候选里可能出现重复项；保持原样，不在这里去重。
    """
    candidate = Path(path)
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / candidate)
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(Path(project_root) / candidate)
        candidates.append(Path.cwd() / candidate)
    return tuple(candidates)


def resolve_vocab_path(
    path: str | Path, *, project_root: str | Path, base_dir: str | Path | None = None
) -> Path:
    """返回第一个实际存在的词表文件；都不存在时抛 ``FileNotFoundError``

    只认 ``is_file()``：同名目录不是可读取的词表，与加载端原行为一致。
    异常里列出全部尝试路径，便于判断是路径写错还是放错了目录。
    """
    candidates = vocab_path_candidates(path, project_root=project_root, base_dir=base_dir)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到词表文件 {path}，已尝试：{[str(c) for c in candidates]}")
