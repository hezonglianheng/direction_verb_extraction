"""按连续起止阶段执行语料流水线；--dry-run 只打印命令，不加载模型。"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from file_paths import collect_jsonl_inputs, validate_output_paths, resolve_cache_path, sidecar_paths
from annotate.cli_args import ConsistentSelection, resolve_graph_selection
from annotate.voting import model_output_paths


ROOT = Path(__file__).resolve().parent
STAGES = ("extract", "replace", "annotate")
KINDS = {"方位词": "localization", "趋向动词": "direction"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--from-stage", choices=STAGES, default="extract")
    parser.add_argument("--to-stage", choices=STAGES, default="annotate")
    parser.add_argument("--corpus-type", choices=KINDS, default="方位词")
    parser.add_argument("--source", help="原始文章 JSONL 文件或目录；从 extract 开始时必填")
    parser.add_argument("--extracted", help="提取阶段输出/替换阶段输入 JSONL")
    parser.add_argument("--pairs", help="替换阶段输出/标注阶段输入 JSONL")
    parser.add_argument("--output", help="标注结果 JSONL")
    parser.add_argument("--filter-config", help="提取规则，默认按 --corpus-type 选择")
    parser.add_argument("--annotation-engine", choices=("staged", "all-in-one"), default=None, action=ConsistentSelection,
                        help="兼容引擎选择；未指定时由 --annotate-args 的 --graph 选择（默认 pair）")
    for stage in STAGES:
        parser.add_argument(f"--{stage}-args", default="[]", help="透传给本阶段的 JSON 字符串数组，不允许覆盖流水线管理的路径/词性")
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


def build_commands(args) -> list[tuple[str, list[str]]]:
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
    extras = {stage: _extra_args(getattr(args, stage + "_args"), stage) for stage in selected}
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
