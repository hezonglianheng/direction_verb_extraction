"""按连续起止阶段执行语料流水线；--config 读取 JSON5 模块参数，--dry-run 只打印命令。"""

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from file_paths import collect_jsonl_inputs, validate_output_paths, resolve_cache_path, same_file_path, sidecar_paths
from annotate.cli_args import ConsistentSelection, resolve_graph_selection
from annotate.voting import model_output_paths


ROOT = Path(__file__).resolve().parent
STAGES = ("extract", "replace", "annotate")
KINDS = {"方位词": "localization", "趋向动词": "direction"}
STAGE_MODEL_OPTIONS = {
    "extract": ("--extract-model", "--screen-model"),
    "replace": ("--replace-model", "--model"),
}
"""一等模型参数到各阶段子命令参数的映射；annotate 仍走 --annotate-args 的 --model/--models。"""

ANNOTATE_GRAPHS = ("pair", "synonym", "all_in_one", "single", "alter_check")
FORMAL_GRAPHS = ("pair", "synonym", "all_in_one")
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
    "screen_kind": {"kind": "extra_value", "type": "choice", "choices": ("方位词", "趋向动词")},
    "extract_workers": {"kind": "extra_value", "type": "int"},
    "filter_workers": {"kind": "extra_value", "type": "int"},
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
    "out_file": {"kind": "pipeline", "dest": "out_file", "type": "output_path"},
    "max_requests": {"kind": "extra_value", "type": "int"},
    "batch_max_requests": {"kind": "extra_value", "type": "int"},
    "limit": {"kind": "extra_value", "type": "int"},
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
    "batch_poll_seconds": {"kind": "extra_value", "type": "float"},
    "batch_timeout_seconds": {"kind": "extra_value", "type": "float"},
    "batch_chunk_size": {"kind": "extra_value", "type": "int"},
    "corpus_type": {"kind": "unsupported", "message": "这是全局参数，请用 --corpus-type 指定"},
    "overwrite": {"kind": "unsupported", "message": "这是全局参数，请用 --overwrite 指定"},
}
CONFIG_FIELDS = {"extract": EXTRACT_FIELDS, "replace": REPLACE_FIELDS, "annotate": ANNOTATE_FIELDS}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=None,
                        help="JSON5 配置文件：顶层为 extract/replace/annotate 三个字典；"
                             "只读取本次选中阶段的字典，未选中阶段即使填写也忽略")
    parser.add_argument("--from-stage", choices=STAGES, default="extract")
    parser.add_argument("--to-stage", choices=STAGES, default="annotate")
    parser.add_argument("--corpus-type", choices=KINDS, default="方位词")
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
    planned = {"--batch-cache", "--batch", "--models", "--model", "--no-llm", "--no-pairs-csv", "--batch-resume", "--graph"}
    for token in tokens:
        option = token.split("=", 1)[0]
        if option in reserved or (stage == "extract" and token.startswith(("-t", "-o", "-f")) and not token.startswith("--")):
            raise ValueError(f"--{stage}-args 不允许覆盖流水线参数：{token}")
        if option.startswith("--") and option not in planned and any(name.startswith(option) for name in reserved | planned):
            raise ValueError(f"--{stage}-args 中涉及路径规划的参数不能缩写：{token}")
    return tokens


def _uses_option(token: str, target: str) -> bool:
    """判断透传 token 是否指向 target，兼容 --opt=value 与 argparse 前缀缩写。"""
    option = token.split("=", 1)[0]
    return option == target or (len(option) > 2 and target.startswith(option))


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
        graph = section.get("graph")
        if graph is None and section.get("engine") == "all-in-one":
            graph = "all_in_one"
        graph = graph or "pair"
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
    suffix = KINDS[args.corpus_type]
    prefix = "location" if suffix == "localization" else "direction"
    base = config_path.parent

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
    suffix = KINDS[args.corpus_type]
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
    inputs = []
    if start == 0:
        if not args.source:
            raise ValueError("从 extract 开始必须提供 --source")
        source = Path(args.source).resolve()
        inputs = collect_jsonl_inputs(source)
        if not inputs:
            raise ValueError(f"输入目录没有 JSONL 文件：{source}")
        filter_config = Path(args.filter_config or ROOT / "config_files" / f"{suffix}_expressions.json5").resolve()
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

    outputs = [paths[stage] for stage in selected]
    sidecars = []
    # Share the exact naming functions with the actual annotation runners.
    if "annotate" in selected:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
        parser.add_argument("--batch-cache")
        parser.add_argument("--batch", choices=("auto", "native", "off"), default="auto")
        parser.add_argument("--batch-resume", action="store_true")
        parser.add_argument("--no-llm", action="store_true")
        parser.add_argument("--no-pairs-csv", action="store_true")
        parser.add_argument("--models", nargs="+")
        parser.add_argument("--graph", choices=("pair", "synonym", "all_in_one", "single", "alter_check"),
                            action=ConsistentSelection)
        try:
            options, _ = parser.parse_known_args(extras["annotate"])
        except argparse.ArgumentError as exc:
            raise ValueError(str(exc)) from exc
        graph = resolve_graph_selection(options.graph, args.annotation_engine, engine_option="--annotation-engine")
        models = options.models or []
        multi = len(models) > 1
        raw = model_output_paths(output, models)[1] if multi else {}
        results = [output, *raw.values()]
        outputs.extend(raw.values())
        for result in results:
            if not options.no_pairs_csv and graph in ("pair", "synonym", "all_in_one"):
                outputs.append(result.with_suffix(".csv"))
            outputs.append(result.with_suffix(".summary.json"))
        if not options.no_llm and options.batch != "off":
            cache_outputs = raw.items() if multi else [(None, output)]
            for model, result in cache_outputs:
                cache = resolve_cache_path(options.batch_cache, result, model_key=model, multi_model=multi)
                sidecars.extend(sidecar_paths(cache))
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
