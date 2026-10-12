# encoding: utf8

"""语料读写的路径检查，不依赖模型或规则引擎。"""

from pathlib import Path
from typing import Iterable


def same_file_path(left: str | Path, right: str | Path) -> bool:
    """识别相对路径、符号链接和现存文件的硬链接别名。"""
    left, right = Path(left), Path(right)
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        # 新输出尚不存在时，只需比较上面的实际路径。
        return False


def validate_output_paths(
    input_paths: Iterable[str | Path], output_paths: Iterable[str | Path]
) -> None:
    """在打开任何输出前，拒绝覆盖输入或让多个输出指向同一文件。"""
    inputs = [Path(path) for path in input_paths]
    outputs = [Path(path) for path in output_paths]
    for index, output in enumerate(outputs):
        for source in inputs:
            if same_file_path(source, output):
                raise ValueError(
                    f"输出路径 {output} 与输入文件 {source} 指向同一文件，不能覆盖原始语料；"
                    "请指定其他输出路径（目录输入建议使用单独的输出目录）"
                )
        for previous in outputs[:index]:
            if same_file_path(previous, output):
                raise ValueError(f"输出路径 {previous} 与 {output} 指向同一文件，请分别指定输出路径")


def collect_jsonl_inputs(
    src_path: str | Path, output_paths: Iterable[str | Path] = ()
) -> list[Path]:
    """写盘前固定输入清单，避免同目录新建的输出被本次扫描读入。

    已存在的同名文件仍视为输入并拒绝覆盖：无法仅凭文件名判断它是原始语料
    还是上次的输出，不能静默排除后再将它清空。
    """
    source = Path(src_path)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = [path for path in source.glob("*.jsonl") if path.is_file()]
    else:
        raise ValueError(f"输入路径 {src_path} 既不是文件也不是目录")
    validate_output_paths(files, output_paths)
    return files
