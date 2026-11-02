# 三阶段入口、流水线与服务器同步

## 目录与兼容性

在项目根目录运行（Windows 可将 `python` 换成 `.\.venv\Scripts\python.exe`）：

```sh
python extract --help
python replace --help
python annotate --help
python annotate --engine all-in-one --help
```

- `extract/`：提取入口、Windows/Linux 执行器、规则与 LTP 缓存、切句、语义闸门、原句筛选、提取日志、按词采样及 LTP 诊断脚本。
- `replace/`：机械替换、两侧检查和句对数据契约。
- `annotate/`：分步/单次标注、空间 agent/图/工具、提示示例选择、投票、CSV 导出/抽样和结果 ID 迁移。
- 根目录同名 `.py` 文件是兼容转发层，不是第二份实现。原来的脚本命令、import 和测试补丁继续有效；新旧导入共享同一个模块对象。
- `config.py`、`agents.py`、`batch_inference.py`、`agent_log.py`、`file_paths.py`、`pipeline_utils.py`、`retry.py`、`utils.py` 属于共享基础设施，仍在根目录。
- `config_files/`、`model_config/`、`system_prompt/`、`LTP_model/`、`results/`、`log/` 的位置不变。移动不改变 tokens、schema、候选键或模型缓存键；分词/规则配置若另有变化，仍需自行判断旧数据是否可复用。
- 建议只用目录入口、`python -m extract/replace/annotate` 或旧根目录入口，不直接运行目录内的实现文件。
- 被 Git 忽略的 `*copy*` 历史副本和既有 `.docs/` 私人说明不移动、不删除；本说明和实施计划采用精确白名单提交。

## 单阶段调用

```sh
python extract raw.jsonl -f config_files/localization_expressions.json5 -t results -o location_corpus
python replace --corpus results/location_corpus.jsonl --corpus-type 方位词 --out-file results/location_pairs.jsonl
python annotate --corpus results/location_pairs.jsonl --corpus-type 方位词 --out-file results/location_annotated.jsonl --limit 100
python annotate --engine all-in-one --corpus results/location_pairs.jsonl --corpus-type 方位词 --limit 100
```

独立提取入口仍要求 `-t` 目录已存在，且保留原有允许重写提取结果的行为。替换和标注默认拒绝覆盖已有结果，需要 `--overwrite`。正式标注只读取两侧检查均合格的句对，不在标注时补做替换/检查。`--no-llm` 不会把未经检查的句对变成合格数据。

## 选择起点和终点

顺序固定为 `extract → replace → annotate`，起点和终点均包含在内，不能反向或跳过中间步骤。

```sh
# 只提取
python pipeline.py --from-stage extract --to-stage extract --source raw.jsonl

# 提取、替换，不标注
python pipeline.py --from-stage extract --to-stage replace --source raw.jsonl

# 只替换，使用已有原句文件
python pipeline.py --from-stage replace --to-stage replace --extracted results/location_corpus.jsonl

# 替换到标注
python pipeline.py --from-stage replace --to-stage annotate --extracted results/location_corpus.jsonl

# 只标注，使用已有句对文件
python pipeline.py --from-stage annotate --to-stage annotate --pairs results/location_pairs.jsonl

# 完整链路；先预检/打印，确认后去掉 --dry-run
python pipeline.py --source raw.jsonl --corpus-type 方位词 --dry-run
```

默认：方位词使用 `results/location_corpus.jsonl`、`location_pairs.jsonl`、`location_annotated.jsonl`；趋向动词使用相应 `direction_*` 文件，规则也自动选 `config_files/direction_expressions.json5`。传 `--filter-config` 可覆盖提取规则；请确保其研究词性与 `--corpus-type` 一致，程序不猜测自定义规则的词性。

使用 `--extracted`、`--pairs`、`--output` 指定各阶段 JSONL 路径。路径按调用时的当前目录解析，内部资源始终按项目根解析。驱动在预检通过后创建提取输出目录，其余目录由原阶段创建。文件名须以小写 `.jsonl` 结尾（不接受 `.JSONL`，避免旧提取器再次追加后缀）。

默认拒绝覆盖本次选中阶段的产物及标注 CSV/summary；`--overwrite` 明确允许覆盖，但不能让输入输出或多份产物指向同一个文件（包括链接别名）。中途起跑不会重跑上游。某阶段非零退出时立即停止并保留退出码；已完成阶段的产物保留，不回滚，也不假装全链路成功。

Batch 缓存/jobs/meta（含自定义 `--batch-cache`）和多模型原始结果也参与全链路路径冲突检查，不能覆盖原始输入或其他阶段产物。缓存路径推导由预检与实际会话共用。缓存允许已存在以便续跑；仍沿用原接口约定：只有 `--batch-resume` 才复用已有任务，不加它可能清空旧缓存并发起新任务。

`--dry-run` 不创建目录、不运行子进程、不调用模型，但仍检查起始输入是否存在和路径冲突；它不是语料内容、API 密钥或每个子命令业务参数的完整验证。

### 阶段专属参数

用 JSON 字符串数组透传，不经 shell 拆词；包含空格的模型名/路径仍是一个参数。以下写法适用于 PowerShell 和常见 POSIX shell：

```sh
python pipeline.py --source raw.jsonl --annotation-engine all-in-one --extract-args '["--screen-model","screen-model","--filter-workers","1"]' --replace-args '["--model","screen-model","--workers","4"]' --annotate-args '["--model","annotation-model","--limit","500","--batch","off"]'
```

流水线管理的输入、输出、词性、覆盖和 engine 参数不能通过透传覆盖；请使用驱动的对应选项。涉及路径规划的透传选项必须写全名，不能用 argparse 缩写（例如 `--out-f`、`--batch-c`）。独立阶段仍接受全部旧参数。标注的默认 limit 沿用原程序：分步 100、all-in-one 1；要处理更多句对，务必显式指定 `--limit`。未选中阶段的透传参数不执行。

这次整理没有填写语言学配置。趋向动词替换规则仍可能没有启用组，运行前应检查 `config_files/direction/alter_rules.json5`，或在替换参数中传 `--alter-rules`；没有候选并不代表已构建出有效语料。

## 服务器同步（仅说明，本次未推送或登录服务器）

Git 会同步已跟踪的代码移动及兼容文件，不会同步被忽略的密钥、模型、结果和日志。远端另一份代码是否有本地修改/本地提交，目前无法从本机确认，这是部署前必须检查的事项。

1. 本机检查本次提交和测试后，由你决定是否 `git push origin main`。不要只 cherry-pick 配置提交而漏掉后续流程/迁移提交；中间快照不保证独立可运行。
2. 服务器暂停定时任务，等待正在运行的 Python/Batch 工作安全结束。不要在长跑进程仍可能延迟导入模块时更新代码。记录未完成 Batch 的缓存路径和任务 ID，避免重发付费任务。
3. 在服务器项目根目录检查 `git status --short`、`git log -5 --oneline`、`git branch --show-current`、`git remote -v`。确认远程和分支确实对应本仓库。
4. **先备份**服务器独有的代码、配置和未跟踪文件；`git diff --binary` 不能覆盖未跟踪/忽略文件。密钥 `model_config/api_keys.json`、`LTP_model/`、`results/`、`log/` 另行安全备份，勿提交密钥。
5. 有未提交修改时，可先 `git stash push -u -m before-stage-layout`；忽略文件不会进入该 stash。确认备份和 stash 后再继续，不要用 reset/clean 丢弃服务器工作。
6. `git fetch origin` 后使用 `git pull --ff-only origin main`。若有服务器本地提交导致不能快进，停下来人工决定 merge/rebase，不强制覆盖。
7. 如需恢复 stash，先 `git stash apply`，确认内容和测试后再决定是否删除 stash。Git 通常能识别移动，但根目录新增兼容壳后，服务器对旧大文件的修改可能冲突或应用到错误位置：对照本说明，把业务改动落入阶段目录里的实现，根目录保留转发层。不要直接用服务器旧文件覆盖兼容入口。
8. 确认原虚拟环境、模型路径和 API 配置仍可用；本次未新增第三方依赖。运行下方测试和四个目录 `--help`，然后小批量验证现有模型接口。旧 cron/nohup 命令可继续使用；逐步迁移到新入口即可。
9. 若需回滚部署，优先切换到事先备份的独立代码目录，不要在未备份的共享工作区强制重置。根目录的兼容壳依赖新目录，不能只复制壳文件。

若采用文件复制而非 Git，必须把新目录、共享文件和兼容壳作为同一版本完整同步；**禁止使用带删除的广域同步命令**（例如未审查的 `rsync --delete`），以免删掉服务器数据、模型或密钥。

## 验证

```sh
python -m unittest discover -s tests -v
```

测试覆盖既有句对/提示流程、目录入口及 engine 分派、资源根路径、新旧模块共享、多进程 spawn、全部六种连续区间、参数透传、路径与覆盖检查、dry-run 无副作用和失败即停。不包含真实在线模型请求及服务器端运行；这些需在自己的环境小样本验证。
