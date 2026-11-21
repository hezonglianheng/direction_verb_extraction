"""按连续起止阶段执行语料流水线；--config 读取 JSON5 模块参数，--dry-run 只打印命令。"""

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

from file_paths import collect_jsonl_inputs, validate_output_paths, resolve_cache_path, same_file_path, sidecar_paths
import sync_cache
from annotate.cli_args import ConsistentSelection, resolve_graph_selection
from annotate.cli_constants import FORMAL_GRAPHS
from annotate.material_paths import annotation_material_paths
from annotate.voting import model_output_paths, safe_model_stem
from corpus_metadata import CORPUS_TYPE_DIRS, DEFAULT_CORPUS_TYPE, normalize_corpus_type
from extract.vocab_paths import resolve_vocab_path
from filter_funcs import LIMIT_MODES


ROOT = Path(__file__).resolve().parent
STAGES = ("extract", "replace", "annotate")
STAGE_MODEL_OPTIONS = {
    "extract": ("--extract-model", "--screen-model"),
    "replace": ("--replace-model", "--model"),
}
"""一等模型参数到各阶段子命令参数的映射；annotate 仍走 --annotate-args 的 --model/--models。"""

ANNOTATE_GRAPHS = ("pair", "synonym", "all_in_one", "single", "alter_check")
DIAGNOSTIC_GRAPHS = tuple(name for name in ANNOTATE_GRAPHS if name not in FORMAL_GRAPHS)
CONFIG_MANAGED_CLI = (
    "source", "extracted", "pairs", "output", "filter_config",
    "extract_model", "replace_model", "annotation_engine",
    "extract_args", "replace_args", "annotate_args",
)
"""--config 模式下不能再由 CLI 覆盖的旧模块参数；阶段选择/词性/覆盖/预检仍走 CLI。"""

# JSON5 配置里的字段表。kind 含义：
#   pipeline     -> 由流水线自己规划的路径/模型/引擎，不直接透传
#   extra_*      -> 转成该阶段子命令的参数
#   unsupported  -> 只能放在流水线 CLI，模块字典里出现即报错
#   list_graphs  -> 仅允许 false（配置模式跑完整链路，不支持只列图）
EXTRACT_FIELDS = {
    "src_path": {"kind": "pipeline", "dest": "source", "type": "path"},
    "filter_config_path": {"kind": "pipeline", "dest": "filter_config", "type": "path"},
    "vocab_config_path": {"kind": "extra_append", "type": "paths", "input": True,
                          "option": "--vocab_config_path"},
    "tgt_path": {"kind": "pipeline", "dest": "tgt_path", "type": "path"},
    "out_name": {"kind": "pipeline", "dest": "out_name", "type": "filename"},
    "no_llm_screen": {"kind": "extra_flag"},
    "screen_model": {"kind": "pipeline", "dest": "extract_model", "type": "model"},
    "screen_workers": {"kind": "extra_value", "type": "int"},
    "screen_batch_size": {"kind": "extra_value", "type": "int"},
    "screen_kind": {"kind": "extra_value", "type": "choice", "choices": ("方位词", "趋向动词")},
    "extract_workers": {"kind": "extra_value", "type": "int"},
    "filter_workers": {"kind": "extra_value", "type": "int"},
    "sentence_dedup_window": {"kind": "extra_value", "type": "int"},
    "corpus_type": {"kind": "unsupported", "message": "这是全局参数，请用 --corpus-type 指定"},
    "overwrite": {"kind": "unsupported", "message": "这是全局参数，请用 --overwrite 指定"},
}
REPLACE_FIELDS = {
    "corpus": {"kind": "pipeline", "dest": "corpus", "type": "path"},
    "out_file": {"kind": "pipeline", "dest": "out_file", "type": "output_path"},
    "alter_rules": {"kind": "extra_value", "type": "path", "input": True},
    "model": {"kind": "pipeline", "dest": "replace_model", "type": "model"},
    "workers": {"kind": "extra_value", "type": "int"},
    "chunk_size": {"kind": "extra_value", "type": "int"},
    "batch_size": {"kind": "extra_value", "type": "int"},
    "limit": {"kind": "extra_value", "type": "int"},
    "limit_mode": {"kind": "extra_value", "type": "choice", "choices": LIMIT_MODES},
    "seed": {"kind": "extra_value", "type": "int"},
    "reference_file": {"kind": "extra_value", "type": "path", "input": True},
    "log_dir": {"kind": "extra_value", "type": "path"},
    "no_llm": {"kind": "extra_flag"},
    "corpus_type": {"kind": "unsupported", "message": "这是全局参数，请用 --corpus-type 指定"},
    "overwrite": {"kind": "unsupported", "message": "这是全局参数，请用 --overwrite 指定"},
}
ANNOTATE_FIELDS = {
    "graph": {"kind": "extra_value", "type": "choice", "choices": ANNOTATE_GRAPHS},
    "engine": {"kind": "pipeline", "dest": "annotation_engine", "type": "choice",
               "choices": ("staged", "all-in-one")},
    "list_graphs": {"kind": "list_graphs"},
    "corpus": {"kind": "pipeline", "dest": "corpus", "type": "path"},
    "alter_rules": {"kind": "extra_value", "type": "path", "input": True},
    "out_dir": {"kind": "pipeline", "dest": "out_dir", "type": "path"},
    # 后缀在确定“本轮真的会写标注结果”时再查：--print-prompt 不写这个文件。
    "out_file": {"kind": "pipeline", "dest": "out_file", "type": "path"},
    "max_requests": {"kind": "extra_value", "type": "int"},
    "batch_max_requests": {"kind": "extra_value", "type": "int"},
    "limit": {"kind": "extra_value", "type": "int"},
    "limit_mode": {"kind": "extra_value", "type": "choice", "choices": LIMIT_MODES},
    "seed": {"kind": "extra_value", "type": "int"},
    "reference_file": {"kind": "extra_value", "type": "path", "input": True},
    "offset": {"kind": "extra_value", "type": "int"},
    "model": {"kind": "extra_value", "type": "model"},
    "models": {"kind": "extra_nargs", "type": "models"},
    "model_workers": {"kind": "extra_value", "type": "int"},
    "workers": {"kind": "extra_value", "type": "int"},
    "batch_inference": {"kind": "extra_flag"},
    "no_llm": {"kind": "extra_flag"},
    "feature_deep": {"kind": "extra_flag"},
    "print_prompt": {"kind": "extra_flag"},
    "allow_retry": {"kind": "extra_tri"},
    "all_candidates": {"kind": "extra_flag"},
    "no_syntax": {"kind": "extra_flag"},
    "no_pairs_csv": {"kind": "extra_flag"},
    "batch": {"kind": "extra_value", "type": "choice", "choices": ("auto", "native", "off")},
    "batch_cache": {"kind": "extra_value", "type": "path"},
    "batch_resume": {"kind": "extra_flag"},
    "sync_cache": {"kind": "extra_value", "type": "path"},
    "sync_resume": {"kind": "extra_flag"},
    "batch_poll_seconds": {"kind": "extra_value", "type": "float"},
    "batch_timeout_seconds": {"kind": "extra_value", "type": "float"},
    "batch_chunk_size": {"kind": "extra_value", "type": "int"},
    "corpus_type": {"kind": "unsupported", "message": "这是全局参数，请用 --corpus-type 指定"},
    "overwrite": {"kind": "unsupported", "message": "这是全局参数，请用 --overwrite 指定"},
}
CONFIG_FIELDS = {"extract": EXTRACT_FIELDS, "replace": REPLACE_FIELDS, "annotate": ANNOTATE_FIELDS}


def _corpus_type_arg(value: str) -> str:
    """把中文类型名或英文别名规范化；未知值走 argparse 的参数错误（退出码 2）。"""
    try:
        return normalize_corpus_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=None,
                        help="JSON5 配置文件：顶层为 extract/replace/annotate 三个字典；"
                             "只读取本次选中阶段的字典，未选中阶段即使填写也忽略")
    parser.add_argument("--from-stage", choices=STAGES, default="extract")
    parser.add_argument("--to-stage", choices=STAGES, default="annotate")
    parser.add_argument(
        "--corpus-type", type=_corpus_type_arg, default=DEFAULT_CORPUS_TYPE,
        help="语料类型：方位词（别名 localization/location/loc）或 趋向动词"
             "（别名 direction/dir/direction_verb）；英文别名不区分大小写",
    )
    parser.add_argument("--source", help="原始文章 JSONL 文件或目录；从 extract 开始时必填")
    parser.add_argument("--extracted", help="提取阶段输出/替换阶段输入 JSONL")
    parser.add_argument("--pairs", help="替换阶段输出/标注阶段输入 JSONL")
    parser.add_argument("--output", help="标注结果 JSONL")
    parser.add_argument("--filter-config", help="提取规则，默认按 --corpus-type 选择")
    parser.add_argument("--extract-model", default=None,
                        help="extract 阶段 LLM 筛选模型配置名，映射到 extract --screen-model；"
                             "仅在 extract 被选中时生效，不能与 --extract-args 里的 --screen-model 同时使用")
    parser.add_argument("--replace-model", default=None,
                        help="replace 阶段 LLM 检查模型配置名，映射到 replace --model；"
                             "仅在 replace 被选中时生效，不能与 --replace-args 里的 --model 同时使用")
    parser.add_argument("--annotation-engine", choices=("staged", "all-in-one"), default=None, action=ConsistentSelection,
                        help="兼容引擎选择；未指定时由 --annotate-args 的 --graph 选择（默认 pair）")
    for stage in STAGES:
        parser.add_argument(f"--{stage}-args", default=None,
                            help="透传给本阶段的 JSON 字符串数组，不允许覆盖流水线管理的路径/词性；"
                                 "与 --config 不能同时使用")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖各阶段产物（绝不允许覆盖输入）")
    parser.add_argument("--dry-run", action="store_true", help="预检并打印命令，不执行/建目录/调用模型")
    return parser.parse_args(argv)


def _extra_args(value: str, stage: str) -> list[str]:
    try:
        tokens = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"--{stage}-args 必须是 JSON 字符串数组") from exc
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise ValueError(f"--{stage}-args 必须是 JSON 字符串数组")
    reserved = {
        "--corpus", "--corpus-type", "--out-file", "--out-dir", "--overwrite", "--engine",
        "-t", "--tgt_path", "-o", "--out_name", "-f", "--filter_config_path",
    }
    # Legacy CLIs accept long-option abbreviations. Any option used by path
    # planning must be spelled in full here, or the child could bypass preflight.
    planned = {
        "--batch-cache", "--batch", "--models", "--model", "--no-llm", "--no-pairs-csv",
        "--batch-resume", "--graph", "--reference-file", "--vocab_config_path",
        "--alter-rules", "--print-prompt", "--sync-cache", "--sync-resume",
    }
    for token in tokens:
        option = token.split("=", 1)[0]
        if option in reserved or (stage == "extract" and token.startswith(("-t", "-o", "-f")) and not token.startswith("--")):
            raise ValueError(f"--{stage}-args 不允许覆盖流水线参数：{token}")
        if option.startswith("--") and option not in planned and any(name.startswith(option) for name in reserved | planned):
            raise ValueError(f"--{stage}-args 中涉及路径规划的参数不能缩写：{token}")
    return tokens


def _annotation_options(tokens: list[str], engine):
    """解析 annotate 透传参数中与预检有关的选项，并返回解析结果与图选择。"""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    parser.add_argument("--batch-cache")
    parser.add_argument("--batch", choices=("auto", "native", "off"), default="auto")
    parser.add_argument("--batch-resume", action="store_true")
    parser.add_argument("--sync-cache")
    parser.add_argument("--sync-resume", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-pairs-csv", action="store_true")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--print-prompt", action="store_true")
    parser.add_argument("--graph", choices=ANNOTATE_GRAPHS, action=ConsistentSelection)
    try:
        options, _ = parser.parse_known_args(tokens)
    except argparse.ArgumentError as exc:
        raise ValueError(str(exc)) from exc
    graph = resolve_graph_selection(options.graph, engine, engine_option="--annotation-engine")
    return options, graph


def _reject_diagnostic_graph(graph: str, *, source: str = "") -> None:
    if graph in DIAGNOSTIC_GRAPHS:
        where = f"{source} " if source else ""
        raise ValueError(
            f"{where}pipeline 仅支持正式句对标注图 {FORMAL_GRAPHS}；诊断图 {graph} 请直接运行 "
            f"python annotate --graph {graph} --corpus <原句.jsonl>"
        )


def _uses_option(token: str, target: str) -> bool:
    """判断透传 token 是否指向 target，兼容 --opt=value 与 argparse 前缀缩写。"""
    option = token.split("=", 1)[0]
    return option == target or (len(option) > 2 and target.startswith(option))


def _option_value(tokens: list[str], target: str) -> str | None:
    """从透传 token 里取最终生效的 --opt value / --opt=value 值；没给返回 None。

    argparse 对重复的单值参数取**最后一次**；只取首次会把被覆盖的旧值当成
    实际输入，预检与子命令看到的内容就不一致。缺值时返回 None，交给子命令
    自己报参数错误；``--opt --next-option`` 中的 ``--next-option`` 视为缺值，
    不能被当成路径。
    """
    value = None
    for index, token in enumerate(tokens):
        if token == target:
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            value = None if following is None or _looks_like_option(following) else following
        elif token.startswith(target + "="):
            value = token.split("=", 1)[1]
    return value


_NEGATIVE_NUMBER = re.compile(r"^-\d+$|^-\d*\.\d+$")
"""与 argparse ``_negative_number_matcher`` 同口径；两个子命令都没有负数选项。"""


def _looks_like_option(token: str) -> bool:
    """透传 token 是否会被 argparse 当成选项，而不是一个待读取的值。

    argparse 允许没有“负数选项”的解析器把 ``-1`` / ``-1.5`` 当作普通值；
    预检必须沿用同一规则，否则名为 ``-1`` 的 reference 会被漏掉。
    """
    if not token.startswith("-") or token == "-":
        return False
    return not _NEGATIVE_NUMBER.fullmatch(token)


def _reference_input_path(value: str) -> Path:
    """透传 --reference-file 的预检路径：与子命令的 ``Path(value)`` 同口径。

    子命令（replace / annotate）只做 ``Path(value)``，不会 ``expanduser()``；
    预检若自行展开 ``~``，就会检查用户主目录里的文件，而子命令实际读取当前
    目录下的同名相对路径，留下“预检保护 A、子命令读 B”的覆盖缺口。
    """
    return Path(value).resolve()


def _option_values(tokens: list[str], options: tuple[str, ...]) -> list[str]:
    """收集可重复选项的值，兼容长选项、短选项、紧邻写法和 ``=`` 写法。"""
    values: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in options:
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if following is not None and not _looks_like_option(following):
                values.append(following)
                index += 1
        else:
            for option in options:
                if option.startswith("--"):
                    prefix = option + "="
                    if token.startswith(prefix):
                        values.append(token[len(prefix):])
                        break
                elif token.startswith(option) and len(token) > len(option):
                    value = token[len(option):]
                    values.append(value[1:] if value.startswith("=") else value)
                    break
        index += 1
    return [value for value in values if value]


def _vocab_input_paths(value: str) -> list[Path]:
    """词表路径里**实际会被读取**的那一份（与 extract 加载端共用同一优先级）

    不能把项目根与当前目录两个候选都当输入：加载端只打开第一个存在的文件，另一个
    候选即使被 ``--batch-cache`` 指到也不会被读取，按输入保护会误报覆盖。
    相对词表在两边都不存在时直接报错——extract 随后也只会抛同一个找不到文件的错，
    预检不该替它把某个不存在的候选说成“与输出冲突”。
    """
    try:
        return [resolve_vocab_path(value, project_root=ROOT).resolve()]
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from None


def _load_pipeline_config(value: str) -> tuple[Path, dict]:
    """读取并做结构检查；模块内部的参数由 _convert_stage_config 按选中阶段处理。"""
    config_path = Path(value).expanduser().resolve()
    try:
        import json5
    except ImportError as exc:
        raise ValueError("读取 --config 需要安装 json5（requirements.txt 已包含）") from exc
    try:
        with config_path.open(encoding="utf-8") as reader:
            data = json5.load(reader, allow_duplicate_keys=False)
    except Exception as exc:
        raise ValueError(f"无法读取 pipeline 配置 {config_path}：{exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"pipeline 配置 {config_path} 顶层必须是字典")
    unknown = sorted(str(key) for key in data if key not in STAGES)
    if unknown:
        raise ValueError(
            f"pipeline 配置 {config_path} 顶层只能包含 extract/replace/annotate，发现：{', '.join(unknown)}"
        )
    return config_path, data


def _resolve_config_path(base: Path, value, config_path: Path, stage: str, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是非空字符串路径")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _coerce_config_value(config_path: Path, stage: str, key: str, value, spec: dict, base: Path):
    kind = spec.get("type")
    if kind == "path":
        return _resolve_config_path(base, value, config_path, stage, key)
    if kind == "output_path":
        path = _resolve_config_path(base, value, config_path, stage, key)
        if path.suffix != ".jsonl":
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是 .jsonl 文件：{path}")
        return path
    if kind == "filename":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是非空字符串文件名")
        if Path(value).name != value:
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 只能是文件名，不能包含目录：{value}")
        if not value.endswith(".jsonl"):
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须以小写 .jsonl 结尾：{value}")
        return value
    if kind == "model":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是非空模型配置名")
        return value
    if kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是整数")
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是有限数值")
        return float(value)
    if kind == "choice":
        choices = spec.get("choices", ())
        if value not in choices:
            raise ValueError(
                f"配置文件 {config_path} 的 {stage}.{key} 必须是 {'/'.join(choices)} 之一，得到：{value!r}"
            )
        return value
    if kind == "paths":
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or not all(isinstance(item, str) and item.strip() for item in values):
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是字符串或字符串数组")
        return [_resolve_config_path(base, item, config_path, stage, key) for item in values]
    if kind == "models":
        if not isinstance(value, list) or not value:
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是非空模型名数组")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 的每一项都必须是非空模型配置名")
        if len(set(value)) != len(value):
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 不能包含重复模型名")
        return list(value)
    raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 类型未实现：{kind!r}")


def _convert_stage_config(stage: str, section: dict, config_path: Path, base: Path):
    """把一个模块字典转成流水线值、子命令透传 token 和只读输入路径。"""
    fields = CONFIG_FIELDS[stage]
    values: dict[str, object] = {}
    extras: list[str] = []
    inputs: list[Path] = []
    graph = None
    if stage == "annotate":
        graph = resolve_graph_selection(
            section.get("graph"), section.get("engine"), engine_option="annotate.engine",
        )
        _reject_diagnostic_graph(graph, source=f"配置文件 {config_path} 的 annotate：")
    for key, value in section.items():
        if key.startswith("--"):
            raise ValueError(f"配置文件 {config_path} 的 {stage} 键名不要带 --：{key}")
        spec = fields.get(key)
        if spec is None:
            raise ValueError(
                f"配置文件 {config_path} 的 {stage} 不支持参数 {key!r}；"
                f"可用参数：{', '.join(sorted(fields))}"
            )
        if value is None:
            continue
        kind = spec["kind"]
        option = spec.get("option") or "--" + key.replace("_", "-")
        if kind == "unsupported":
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 不能在这里配置：{spec['message']}")
        if kind == "list_graphs":
            if not isinstance(value, bool):
                raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是布尔值")
            if value:
                raise ValueError(
                    f"配置文件 {config_path} 的 {stage}.{key} 不能为 true："
                    "配置模式会跑完整链路，请直接运行 python annotate --list-graphs"
                )
            continue
        if kind == "pipeline":
            values[spec["dest"]] = _coerce_config_value(config_path, stage, key, value, spec, base)
        elif kind == "extra_flag":
            if not isinstance(value, bool):
                raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是布尔值")
            if value:
                extras.append(option)
        elif kind == "extra_value":
            coerced = _coerce_config_value(config_path, stage, key, value, spec, base)
            extras.extend((option, str(coerced)))
            if spec.get("input"):
                inputs.append(Path(coerced))
        elif kind == "extra_append":
            coerced = _coerce_config_value(config_path, stage, key, value, spec, base)
            for item in coerced:
                extras.extend((option, str(item)))
                if spec.get("input"):
                    inputs.append(Path(item))
        elif kind == "extra_nargs":
            coerced = _coerce_config_value(config_path, stage, key, value, spec, base)
            extras.append(option)
            extras.extend(coerced)
        elif kind == "extra_tri":
            if not isinstance(value, bool):
                raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 必须是 true/false/null")
            extras.append("--allow-retry" if value else "--no-allow-retry")
        else:
            raise ValueError(f"配置文件 {config_path} 的 {stage}.{key} 类型未实现：{kind!r}")

    if stage == "annotate":
        if section.get("model") is not None and section.get("models") is not None:
            raise ValueError(f"配置文件 {config_path} 的 annotate.model 与 annotate.models 不能同时设置")
        if section.get("out_dir") is not None and section.get("out_file") is not None:
            raise ValueError(f"配置文件 {config_path} 的 annotate.out_dir 与 annotate.out_file 不能同时设置")
        if graph in FORMAL_GRAPHS and section.get("alter_rules") is not None:
            raise ValueError(
                f"配置文件 {config_path} 的 annotate.alter_rules 只用于 alter_check 诊断图；"
                f"正式图 {graph} 不在这里替换"
            )
        if graph == "all_in_one" and section.get("feature_deep") is True:
            raise ValueError(f"配置文件 {config_path} 的 annotate.feature_deep 不适用于 all_in_one")
        models = section.get("models")
        if (
            models is not None
            and len(models) > 1
            and section.get("no_llm") is True
            and section.get("print_prompt") is not True
        ):
            raise ValueError(
                f"配置文件 {config_path} 的 annotate.models 在多模型投票时不能与 annotate.no_llm 同时使用；"
                "只有 print_prompt 预览可以同时开启"
            )
        if section.get("print_prompt") is True and graph not in FORMAL_GRAPHS:
            raise ValueError(f"配置文件 {config_path} 的 annotate.print_prompt 只支持正式图 {FORMAL_GRAPHS}")
        if graph not in FORMAL_GRAPHS and (
            section.get("all_candidates") is True or section.get("no_syntax") is True
        ):
            raise ValueError(
                f"配置文件 {config_path} 的 annotate.all_candidates/no_syntax 只支持正式图 {FORMAL_GRAPHS}；"
                f"当前图是 {graph}"
            )
        if graph not in FORMAL_GRAPHS and any(
            section.get(key) is not None for key in ("limit_mode", "seed", "reference_file")
        ):
            raise ValueError(
                f"配置文件 {config_path} 的 annotate.limit_mode/seed/reference_file 只支持正式图 {FORMAL_GRAPHS}；"
                f"当前图是 {graph}"
            )
        limit_mode = section.get("limit_mode")
        if section.get("reference_file") is not None:
            if limit_mode not in (None, "aligned"):
                raise ValueError(
                    f"配置文件 {config_path} 的 annotate.reference_file 只能配合 limit_mode=aligned；"
                    f"当前是 {limit_mode!r}"
                )
            if section.get("seed") is not None:
                raise ValueError(
                    f"配置文件 {config_path} 的 annotate.reference_file 不对齐随机过程，不能与 seed 同时设置"
                )
        elif limit_mode == "aligned":
            raise ValueError(
                f"配置文件 {config_path} 的 annotate.limit_mode=aligned 必须同时设置 reference_file"
            )
        limit_value = section.get("limit")
        if isinstance(limit_value, int) and not isinstance(limit_value, bool) and limit_value <= 0:
            raise ValueError(f"配置文件 {config_path} 的 annotate.limit 必须是大于 0 的整数")
        offset_value = section.get("offset")
        if isinstance(offset_value, int) and not isinstance(offset_value, bool) and offset_value < 0:
            raise ValueError(f"配置文件 {config_path} 的 annotate.offset 必须是大于等于 0 的整数")
        seed_value = section.get("seed")
        if isinstance(seed_value, int) and not isinstance(seed_value, bool) and seed_value < 0:
            raise ValueError(f"配置文件 {config_path} 的 annotate.seed 必须是大于等于 0 的整数")
        if limit_mode in ("sequential", "random", "proportional") and limit_value is None:
            raise ValueError(
                f"配置文件 {config_path} 的 annotate.limit_mode={limit_mode} 需要同时设置 limit（X>0）"
            )
        if section.get("seed") is not None and limit_value is None:
            raise ValueError(f"配置文件 {config_path} 的 annotate.seed 只能在设置 limit 时使用")
    elif stage == "replace":
        reference = section.get("reference_file")
        limit_mode = section.get("limit_mode")
        limit_value = section.get("limit")
        seed_value = section.get("seed")
        if reference is not None:
            if limit_mode not in (None, "aligned"):
                raise ValueError(
                    f"配置文件 {config_path} 的 replace.reference_file 只能配合 limit_mode=aligned；"
                    f"当前是 {limit_mode!r}"
                )
            if seed_value is not None:
                raise ValueError(
                    f"配置文件 {config_path} 的 replace.reference_file 不对齐随机过程，不能与 seed 同时设置"
                )
        elif limit_mode == "aligned":
            raise ValueError(
                f"配置文件 {config_path} 的 replace.limit_mode=aligned 必须同时设置 reference_file"
            )
        elif limit_value is None and (limit_mode is not None or seed_value is not None):
            raise ValueError(
                f"配置文件 {config_path} 的 replace.limit_mode 与 replace.seed 只能在设置 replace.limit 时使用"
            )
        if isinstance(limit_value, int) and not isinstance(limit_value, bool) and limit_value <= 0:
            raise ValueError(f"配置文件 {config_path} 的 replace.limit 必须是大于 0 的整数")
        if isinstance(seed_value, int) and not isinstance(seed_value, bool) and seed_value < 0:
            raise ValueError(f"配置文件 {config_path} 的 replace.seed 必须是大于等于 0 的整数")
    return values, extras, inputs


def _apply_config(args) -> argparse.Namespace:
    config_path, data = _load_pipeline_config(args.config)
    for name in CONFIG_MANAGED_CLI:
        if getattr(args, name) is not None:
            raise ValueError(
                f"使用 --config 时不能再显式传入 --{name.replace('_', '-')}；"
                "请在配置文件对应模块中填写"
            )

    start, end = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if start > end:
        raise ValueError("起点不能晚于终点：extract → replace → annotate")
    selected = STAGES[start:end + 1]
    suffix = CORPUS_TYPE_DIRS[args.corpus_type]
    prefix = "location" if suffix == "localization" else "direction"
    base = config_path.parent

    # 诊断图不属于接力链路；先于任何输入存在性检查拒绝，避免被上游缺失文件掩盖。
    if "annotate" in selected:
        annotate_section = data.get("annotate", {})
        if not isinstance(annotate_section, dict):
            raise ValueError(f"pipeline 配置 {config_path} 的 annotate 必须是字典")
        _reject_diagnostic_graph(
            resolve_graph_selection(
                annotate_section.get("graph"), annotate_section.get("engine"),
                engine_option="annotate.engine",
            ),
            source=f"配置文件 {config_path} 的 annotate：",
        )

    stage_values: dict[str, dict] = {}
    stage_extras: dict[str, list[str]] = {}
    stage_inputs: dict[str, list[Path]] = {}
    for stage in selected:
        section = data.get(stage, {})
        if not isinstance(section, dict):
            raise ValueError(f"pipeline 配置 {config_path} 的 {stage} 必须是字典")
        values, extras, inputs = _convert_stage_config(stage, section, config_path, base)
        stage_values[stage], stage_extras[stage], stage_inputs[stage] = values, extras, inputs
        for path in inputs:
            if not path.is_file():
                raise ValueError(f"配置文件 {config_path} 的 {stage} 输入文件不存在：{path}")

    extracted = None
    source = None
    filter_config = None
    extract_model = None
    if "extract" in selected:
        values = stage_values["extract"]
        source = values.get("source")
        if source is None:
            raise ValueError(f"配置文件 {config_path} 的 extract.src_path 必填（从 extract 开始时）")
        tgt_path = values.get("tgt_path") or (ROOT / "results")
        out_name = values.get("out_name") or f"{prefix}_corpus.jsonl"
        extracted = (tgt_path / out_name).resolve()
        filter_config = values.get("filter_config")
        extract_model = values.get("extract_model")

    pairs = None
    extracted_arg = extracted
    replace_model = None
    if "replace" in selected:
        values = stage_values["replace"]
        pairs = values.get("out_file") or (ROOT / "results" / f"{prefix}_pairs.jsonl").resolve()
        corpus = values.get("corpus")
        if corpus is not None:
            if "extract" in selected:
                if not same_file_path(corpus, extracted):
                    raise ValueError(
                        f"配置文件 {config_path} 的 replace.corpus 与 extract 输出不一致："
                        f"{corpus} != {extracted}"
                    )
            else:
                extracted_arg = corpus
        replace_model = values.get("replace_model")

    output = None
    pairs_arg = pairs
    annotation_engine = None
    if "annotate" in selected:
        values = stage_values["annotate"]
        out_file = values.get("out_file")
        out_dir = values.get("out_dir")
        if out_file is not None:
            output = out_file
        elif out_dir is not None:
            output = (out_dir / f"{prefix}_annotated.jsonl").resolve()
        else:
            output = (ROOT / "results" / f"{prefix}_annotated.jsonl").resolve()
        corpus = values.get("corpus")
        if corpus is not None:
            if "replace" in selected:
                if not same_file_path(corpus, pairs):
                    raise ValueError(
                        f"配置文件 {config_path} 的 annotate.corpus 与 replace 输出不一致："
                        f"{corpus} != {pairs}"
                    )
            else:
                pairs_arg = corpus
        annotation_engine = values.get("annotation_engine")

    new_args = argparse.Namespace(**vars(args))
    new_args.source = source
    new_args.extracted = extracted_arg
    new_args.pairs = pairs_arg
    new_args.output = output
    new_args.filter_config = filter_config
    new_args.extract_model = extract_model
    new_args.replace_model = replace_model
    new_args.annotation_engine = annotation_engine
    new_args.extract_args = json.dumps(stage_extras.get("extract", []), ensure_ascii=False)
    new_args.replace_args = json.dumps(stage_extras.get("replace", []), ensure_ascii=False)
    new_args.annotate_args = json.dumps(stage_extras.get("annotate", []), ensure_ascii=False)
    new_args._config_inputs = [
        config_path,
        *[path for stage in selected for path in stage_inputs.get(stage, [])],
    ]
    new_args.config = None
    return new_args


def build_commands(args) -> list[tuple[str, list[str]]]:
    if getattr(args, "config", None):
        args = _apply_config(args)
    start, end = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if start > end:
        raise ValueError("起点不能晚于终点：extract → replace → annotate")
    selected = STAGES[start:end + 1]
    suffix = CORPUS_TYPE_DIRS[args.corpus_type]
    prefix = "location" if suffix == "localization" else "direction"
    extracted = Path(args.extracted or ROOT / "results" / f"{prefix}_corpus.jsonl").resolve()
    pairs = Path(args.pairs or ROOT / "results" / f"{prefix}_pairs.jsonl").resolve()
    output = Path(args.output or ROOT / "results" / f"{prefix}_annotated.jsonl").resolve()
    paths = {"extract": extracted, "replace": pairs, "annotate": output}
    extras = {}
    for stage in selected:
        value = getattr(args, stage + "_args")
        extras[stage] = _extra_args("[]" if value is None else value, stage)
    for stage in selected:
        pipeline_option, child_option = STAGE_MODEL_OPTIONS.get(stage, (None, None))
        if pipeline_option is None:
            continue
        model = getattr(args, stage + "_model")
        if model is None:
            continue
        if not model.strip():
            raise ValueError(f"{pipeline_option} 不能为空")
        if any(_uses_option(token, child_option) for token in extras[stage]):
            raise ValueError(f"{pipeline_option} 与 --{stage}-args 中的 {child_option} 不能同时使用；请只保留一种写法")
        extras[stage] = [child_option, model, *extras[stage]]
    options = None
    graph = None
    preview = False
    if "annotate" in selected:
        options, graph = _annotation_options(extras["annotate"], args.annotation_engine)
        _reject_diagnostic_graph(graph, source="--from-stage/--to-stage 选中的 annotate：")
        if getattr(options, "sync_resume", False) and not getattr(options, "sync_cache", None):
            raise ValueError("annotate 的 --sync-resume 必须同时指定 --sync-cache")
        preview = bool(options.print_prompt)
    inputs = []
    if start == 0:
        if not args.source:
            raise ValueError("从 extract 开始必须提供 --source")
        source = Path(args.source).resolve()
        inputs = collect_jsonl_inputs(source)
        if not inputs:
            raise ValueError(f"输入目录没有 JSONL 文件：{source}")
        filter_config = Path(
            args.filter_config or ROOT / "config_files" / suffix / f"{suffix}_expressions.json5"
        ).resolve()
        if not filter_config.is_file():
            raise ValueError(f"找不到提取规则文件：{filter_config}")
        inputs.append(filter_config)
    else:
        first_input = extracted if start == 1 else pairs
        if not first_input.is_file():
            raise ValueError(f"找不到起始阶段输入：{first_input}")
        inputs = [first_input]
    # 配置文件本身，以及配置里显式声明的词表/替换规则，都是只读输入。
    inputs.extend(getattr(args, "_config_inputs", []))
    if "annotate" in selected and not preview:
        # 标注实际读取的配置表、示例与提示语同样是只读输入；否则 --sync-cache
        # 指到这些文件时，预检会放行，真正运行再把它截断。
        inputs.extend(annotation_material_paths(args.corpus_type))
        alter_rules = _option_value(extras["annotate"], "--alter-rules")
        if alter_rules:
            inputs.append(Path(alter_rules).resolve())
    # 透传参数里的词表、替换规则和 reference 同样会在子阶段被读取；它们必须参加
    # 输入/输出别名检查，否则后续 Batch 缓存/jobs/meta 可能覆盖这些资源。
    # 词表按加载端优先级只保护实际读取的那一份，见 _vocab_input_paths。
    for stage in selected:
        if stage == "extract":
            for value in _option_values(extras[stage], ("--vocab_config_path", "-v")):
                for item in value.split(","):
                    item = item.strip()
                    if item:
                        inputs.extend(_vocab_input_paths(item))
        elif stage == "replace":
            rules = _option_value(extras[stage], "--alter-rules")
            if rules:
                inputs.append(Path(rules).resolve())
            else:
                default_rules = (ROOT / "config_files" / suffix / "alter_rules.json5").resolve()
                if default_rules.is_file():
                    inputs.append(default_rules)
    # --replace-args / --annotate-args 里的 --reference-file 都是只读输入，必须参加
    # 输出别名检查；否则后续阶段可能把前一阶段读过的 reference 覆盖掉。
    for stage in ("replace", "annotate"):
        if stage in selected:
            reference = _option_value(extras[stage], "--reference-file")
            if reference:
                reference_path = _reference_input_path(reference)
                if not reference_path.is_file():
                    raise ValueError(f"找不到 reference 文件：{reference_path}")
                inputs.append(reference_path)

    extra_outputs: list[Path] = []
    sidecars: list[Path] = []
    # Share the exact naming functions with the actual annotation runners.
    # --print-prompt 在 runner 里早于输出规划返回；标注主产物、CSV/summary、
    # 多模型 raw 与 Batch sidecar 都不会写，因此不参加本阶段的写入检查。
    if "annotate" in selected and not preview:
        models = options.models or []
        multi = len(models) > 1
        raw = model_output_paths(output, models)[1] if multi else {}
        results = [output, *raw.values()]
        extra_outputs.extend(raw.values())
        for result in results:
            if not options.no_pairs_csv and graph in FORMAL_GRAPHS:
                extra_outputs.append(result.with_suffix(".csv"))
            extra_outputs.append(result.with_suffix(".summary.json"))
        if not options.no_llm and options.batch != "off":
            cache_outputs = raw.items() if multi else [(None, output)]
            for model, result in cache_outputs:
                cache = resolve_cache_path(options.batch_cache, result, model_key=model, multi_model=multi)
                sidecars.extend(sidecar_paths(cache))
        if options.sync_cache and not options.no_llm:
            sync_outputs = raw.items() if multi else [(None, output)]
            for model, result in sync_outputs:
                sync_path = resolve_cache_path(
                    options.sync_cache,
                    result,
                    model_key=safe_model_stem(model) if multi and model else model,
                    multi_model=multi,
                )
                sidecars.extend(sync_cache.sidecar_paths(sync_path))

    outputs = [paths[stage] for stage in selected if not (stage == "annotate" and preview)]
    outputs.extend(extra_outputs)
    validate_output_paths(inputs, outputs + sidecars)
    for path in sidecars:
        if path.exists() and not path.is_file():
            raise ValueError(f"Batch 附属路径不是文件：{path}")
    for path in outputs:
        if path.suffix != ".jsonl" and path in [paths[s] for s in selected]:
            raise ValueError(f"阶段产物必须使用小写 .jsonl 后缀：{path}")
        if path.exists() and (not args.overwrite or not path.is_file()):
            raise ValueError(f"输出已存在或不是文件：{path}；如需覆盖文件请加 --overwrite")

    commands = []
    for stage in selected:
        command = [sys.executable, str(ROOT / stage)]
        if stage == "extract":
            command += [str(source), "-f", str(filter_config), "-t", str(extracted.parent), "-o", extracted.name]
        else:
            if stage == "annotate" and args.annotation_engine is not None:
                command += ["--engine", args.annotation_engine]
            command += ["--corpus", str(extracted if stage == "replace" else pairs),
                        "--corpus-type", args.corpus_type, "--out-file", str(paths[stage])]
            if args.overwrite:
                command.append("--overwrite")
        command += extras[stage]
        commands.append((stage, command))
    return commands


def execute_commands(commands: list[tuple[str, list[str]]]) -> int:
    for stage, command in commands:
        print(f"[{stage}] {shlex.join(command)}", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"[{stage}] 失败，退出码 {result.returncode}；后续阶段未执行。", file=sys.stderr)
            return result.returncode
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        commands = build_commands(args)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        for stage, command in commands:
            print(f"[{stage}] {shlex.join(command)}")
        return 0
    # 提取旧入口要求目录已存在，只在完整预检通过后创建。
    if args.from_stage == "extract":
        command = commands[0][1]
        Path(command[command.index("-t") + 1]).mkdir(parents=True, exist_ok=True)
    return execute_commands(commands)


if __name__ == "__main__":
    raise SystemExit(main())
