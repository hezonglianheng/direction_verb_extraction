# encoding: utf8

"""语料读写的路径检查，不依赖模型或规则引擎。"""

import contextlib
import tempfile
from pathlib import Path
from typing import Callable, Iterable


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


@contextlib.contextmanager
def open_exclusive(path: str | Path, encoding: str = "utf-8"):
    """独占创建输出文件：目标已存在就报错，绝不在确认覆盖前截断它

    `open(path, "w")` 会**先截断再写**：异常发生在写入过程中时，旧的那份输出已经没了、
    新的还没写成，两边都拿不到。跑批与筛选入口都改成先走这里——「拒绝覆盖已有输出」
    因此发生在首次写入之前，而不是写在文件被清空之后。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # x 由文件系统一次完成「不存在才创建」，避免 exists() 与 w 之间的竞态。
        handle = target.open("x", encoding=encoding)
    except FileExistsError:
        raise FileExistsError(f"输出文件 {target} 已存在；如需覆盖请显式给出覆盖选项，或换一个输出路径") from None
    try:
        yield handle
    finally:
        handle.close()


@contextlib.contextmanager
def atomic_write_text(path: str | Path, encoding: str = "utf-8"):
    """先写同目录临时文件、正常退出后才原子替换目标：中途失败时旧输出保持原样

    与 open_exclusive 是两件事：那个管「不许覆盖已有文件」（在写之前拦），这个管
    「写的过程中崩了不要把已有文件留下个半截」（在写之后才替换）。筛选入口要的是后者
    ——它的输出文件常常是上一次跑出来的，本轮就是要重写它，但要保证写失败时旧结果还在。

    临时文件与目标同目录：跨设备 rename 不是原子操作。后缀用 .tmp，不会被
    collect_jsonl_inputs 的 *.jsonl 扫描当成输入。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    temp = Path(temp_name)
    try:
        with open(fd, "w", encoding=encoding, closefd=True) as handle:
            yield handle
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def iter_path_sentences(src_path: str | Path, per_file: Callable[[Path, int], Iterable], *, output_paths=()):
    """目录（或单文件）下 jsonl 的「收集 + 遍历」：**不能自身包含 yield**（E4）

    两个平台入口（entry_in_win / entry_in_linux）这一步原本各写一份，差别只在逐文件怎么切句；
    路径检查、目录扫描、输出避让这些**必须一致**的部分却跟着抄了两遍，改一处就容易漏另一处。
    这里把它们收成一份：入口只带自己的 per_file 进来。

    为什么本函数不能有 yield：生成器体的代码只在第一次 next() 时才跑，于是
    `collect_jsonl_inputs` 里的输入扫描与"输出不能落在输入目录里"的检查会被推迟到
    **输出文件已经打开之后**——那时输出已经把目标截断了，报错也来不及。
    所以这里先算清文件清单，再返回一个惰性生成器表达式。

    Args:
        src_path: 输入文件或目录
        per_file: 单文件迭代器工厂，形如 lambda file, index: iter_jsonl_file_sentences(...)
        output_paths: 本轮输出，用来把输出从输入清单里排除（见 collect_jsonl_inputs）
    """
    files = collect_jsonl_inputs(src_path, output_paths)
    return (sentence for index, file in enumerate(files) for sentence in per_file(file, index))


def validate_run_outputs(
    corpus_path: str | Path,
    out_path: str | Path,
    csv_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """跑批入口的输出路径总检查：在任何写入之前一次查完（E1）

    与 validate_output_paths 的分工：那个只查「输入与输出、输出与输出是不是同一个文件」，
    这个是跑批专属的那层——把一次跑批会写出的三个文件（jsonl、成对 csv、summary）凑齐后
    交给它，再补一条「已有输出默认拒绝」。放在 file_paths 而不放在跑批入口里，是为了让
    spatial_runner 与 all_in_one 两个入口共用同一份口径（E4：入口之间不该各写一套）。

    查三件事，任一件不成立就在建文件之前退出：
      1. 输入与任何一个输出不能是同一个文件——`--out-file` 指向输入时，旧行为会把结果追加
         进输入，收尾再按 (line_no, candidate.index) 排序重写它，原始语料与结果混在一起。
      2. 三个输出之间不能互撞：`--out-file result.csv` 时 `with_suffix(".csv")` 换出来还是
         它自己，收尾导 csv 会覆盖刚写好的 jsonl（路径由调用方用 st.pairs_csv_path 错开）。
      3. 已有输出默认拒绝（overwrite=True 才允许）。

    Args:
        corpus_path: 输入语料路径（可以是目录吗——不行，跑批只吃单个 jsonl，传目录进来
            会在这里被当成「输出撞上一个目录」而报出来）
        out_path: 结果 jsonl 路径
        csv_path: 成对信息 csv 路径；本次不导出时给 None
        overwrite: 是否显式允许覆盖已有输出
    """
    out_path = Path(out_path)
    outputs: list[Path] = [out_path]
    if csv_path is not None:
        outputs.append(Path(csv_path))
    outputs.append(out_path.with_suffix(".summary.json"))
    validate_output_paths([corpus_path], outputs)
    if not overwrite:
        for output in outputs:
            if output.exists():
                raise ValueError(
                    f"输出文件 {output} 已存在（默认拒绝覆盖，免得混入历史记录与重复 id）；"
                    "请换一个输出路径，或显式加 --overwrite 覆盖它"
                )


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
