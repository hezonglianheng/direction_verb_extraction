# 三阶段入口、流水线与服务器同步

## 目录与入口（已移除根目录兼容层）

在项目根目录运行（Windows 可将 `python` 换成 `.\.venv\Scripts\python.exe`）：

```sh
python extract --help
python replace --help
python annotate --help
python annotate --engine all-in-one --help
```

- `extract/`：提取入口、Windows/Linux 执行器、规则与 LTP 缓存、切句、语义闸门、原句筛选、提取日志及按词采样。
- `replace/`：机械替换、两侧检查和句对数据契约。
- `annotate/`：分步/单次标注、空间 agent/图/工具、提示示例选择、投票、CSV 导出/抽样和结果 ID 迁移。
- **本版是入口兼容性变更。** 上一版（`2770a92`、`f6e2936`）暂存的根目录同名 `.py` 转发入口和 `legacy_entry.py` 已彻底删除。`python sentence_extract.py`、`python spatial_runner.py` 等旧命令及 `import spatial_tools` 等旧导入不再支持，部署前必须更新外部调用。
- `config.py`、`agents.py`、`batch_inference.py`、`agent_log.py`、`file_paths.py`、`pipeline_utils.py`、`retry.py`、`utils.py` 属于共享基础设施，仍在根目录。
- `config_files/`、`model_config/`、`system_prompt/`、`LTP_model/`、`results/`、`log/` 的位置不变。移动不改变 tokens、schema、候选键或模型缓存键；分词/规则配置若另有变化，仍需自行判断旧数据是否可复用。
- 支持目录入口 `python extract`、`python replace`、`python annotate`，或对应 `python -m extract`、`python -m replace`、`python -m annotate`。不要 `cd extract` 后运行实现文件，也不要用 `python extract/sentence_extract.py` 绕开包入口。
- 被 Git 忽略的 `*copy*` 历史副本和既有 `.docs/` 私人说明不移动、不删除；这些副本可能仍引用旧模块名，只作为历史材料，不属于正式测试/执行路径，重新运行前需自行迁移。历史审查报告中的旧文件位置不是当前调用说明。

### 内部导入与辅助命令

生产代码与测试统一使用完整包路径，包括函数内延迟导入和 `TYPE_CHECKING` 类型导入。不再用 `sys.modules` 注册旧名，不再依赖根目录兼容文件；配置、结果位置和语料 schema 不变。

```python
from extract import sentence_screening
from extract.sentenceTable import SentenceTaskCache
from replace import sentence_replace, corpus_pairs
from annotate import spatial_tools, spatial_runner, all_in_one
# 外部测试的字符串补丁路径也要改为完整包路径：
# patch("annotate.spatial_tools.some_function", ...)
```

共享基础模块仍按原名导入，如 `import config`、`import batch_inference`。不要将它们误移入阶段包，也不要将各阶段目录单独加入 `PYTHONPATH`，否则可能重新产生顶层同名模块及多份缓存。外部脚本应从项目根启动，或仅将项目根加入 `PYTHONPATH`。

| 旧调用（不再支持） | 新调用（在项目根目录运行） |
| --- | --- |
| `python sentence_extract.py ...` | `python extract ...` |
| `python sentence_replace.py ...` | `python replace ...` |
| `python spatial_runner.py ...` | `python annotate ...` |
| `python all_in_one.py ...` | `python annotate --engine all-in-one ...` |
| `python word_sampler.py ...` | `python -m extract.word_sampler ...` |
| `python spatial_agents.py --list` | `python -m annotate.spatial_agents --list` |
| `python export_pairs_csv.py ...` | `python -m annotate.export_pairs_csv ...` |
| `python sample_pairs_csv.py ...` | `python -m annotate.sample_pairs_csv ...` |
| `python migrate_result_ids.py ...` | `python -m annotate.migrate_result_ids ...` |
| `python sentenceFilters.py` | `python -m extract.sentenceFilters`（会执行 LTP 自测） |

此前已删除 25 个阶段模块的根目录转发文件及 `legacy_entry.py`。`pipeline.py`、上述共享模块和三个阶段目录中的生产实现都保留；旧入口不是可单独恢复的运行方案。

### 最小运行版的提交范围

本次另追加取消跟踪提交，排除非运行必需内容：`extract/temp.py`、`.docs/plans/2026-11-02-stage-layout.md`、`tests/`、`filter_sentence/`、`filtered_sentences.csv`、`note.txt` 和 `项目审查与改进建议.md`。本地副本保留供后续开发，并通过定向忽略规则防止再次被批量加入；服务器更新后不能依赖这些文件。

运行所需的三个阶段包、共享基础模块、`annotate/prompt_examples.py`、`replace/corpus_pairs.py`、`replace/sentence_replace.py`、`system_prompt/judgement_criteria.txt` 和四个示例 JSON5 均继续跟踪。不要把运行时提示示例误当作本地语料样例删除，也不要忽略所有 CSV，配置目录中的判定表仍是运行依赖。

这是追加删除提交，不改写已有历史，因此上述文件仍可能出现在先前提交中。忽略文件不受 Git 跟踪，也不会进入 `git stash -u`；后续开发前应另行备份或有选择地提交到开发分支，勿用 `git clean -fdx` 清理它们。

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

Git 会同步已跟踪的包路径改动和旧入口删除，不会同步被忽略的密钥、模型、结果和日志。远端另一份代码是否有本地修改/本地提交，目前无法从本机确认，这是部署前必须检查的事项。

1. 本机检查本次提交和测试后，由你决定是否 `git push origin main`。不要只 cherry-pick 配置提交而漏掉后续流程/迁移提交；中间快照不保证独立可运行。
2. 服务器暂停定时任务，等待正在运行的 Python/Batch 工作安全结束。不要在长跑进程仍可能延迟导入模块时更新代码。记录未完成 Batch 的缓存路径和任务 ID，避免重发付费任务。
3. 在服务器项目根目录检查 `git status --short`、`git log -5 --oneline`、`git branch --show-current`、`git remote -v`。确认远程和分支确实对应本仓库。
4. **先备份**服务器独有的代码、配置和未跟踪文件；`git diff --binary` 不能覆盖未跟踪/忽略文件。密钥 `model_config/api_keys.json`、`LTP_model/`、`results/`、`log/` 另行安全备份，勿提交密钥。
5. 有未提交修改时，可先 `git stash push -u -m before-stage-layout`；忽略文件不会进入该 stash。确认备份和 stash 后再继续，不要用 reset/clean 丢弃服务器工作。
6. `git fetch origin` 后使用 `git pull --ff-only origin main`。若有服务器本地提交导致不能快进，停下来人工决定 merge/rebase，不强制覆盖。
7. 如需恢复 stash，先 `git stash apply`，确认内容和测试后再决定是否删除 stash。服务器对旧大文件/兼容壳的修改可能产生 modify/delete 冲突，或让旧文件重新出现：把业务改动迁入阶段包中的实现，并同步修改 import，**不要恢复已删除的根目录转发文件**。
8. **恢复任务前，先完成外部调用迁移：**检查 cron、systemd、nohup/批处理脚本、IDE 启动配置、Notebook 及自定义 Python 脚本，按上表替换旧命令、import 和字符串补丁路径。设置工作目录为项目根目录，使用原虚拟环境解释器；服务器上仅 `git pull` 不会自动修改仓库外的定时任务。`pipeline.py` 的现有调用不需要改名。
9. 确认原模型路径、API 配置和依赖环境仍可用；本次未新增第三方依赖。按下方部署验证执行命令，再小批量验证现有模型接口。**旧 cron/nohup 命令现在会失败，验证前不能恢复调度。**
10. 若需回滚部署，优先切换到事先备份的完整旧版本代码目录，同时恢复与该版本对应的外部启动命令。不要只复制几个旧入口，也不要在未备份的共享工作区强制重置。

若采用文件复制而非 Git，必须把三个包目录和共享文件作为同一版本完整同步；还应对照本次 Git 删除清单，仅移除根目录的 25 个转发文件和 `legacy_entry.py`。不要直接删除全部根目录 `.py` 文件。**禁止使用带删除的广域同步命令**（例如未审查的 `rsync --delete`），以免删掉共享代码、服务器数据、模型或密钥。

## 最小运行版的部署验证

在装有项目依赖的虚拟环境中，从项目根目录运行：

```sh
python pipeline.py --help
python extract --help
python replace --help
python annotate --help
python annotate --engine all-in-one --help
```

这些命令检查入口与依赖能否加载，不发送模型请求，也不能证明模型配置、语料内容或最终标注质量正确。再用独立的小样本输入执行 `pipeline.py ... --dry-run` 检查实际路径，确认后按前文命令进行小批量验证；真实模型验证可能产生费用。

完整回归测试曾在精简前通过 46 项；测试源码当前仅保留在本地供后续开发，不随本次部署提交。因此不要在服务器运行 `python -m unittest discover -s tests -v` 作为本版本的验证：没有本项目 `tests/` 时可能报错或发现第三方同名测试包。需要复现完整回归时，应取得相应的开发分支/测试文件，并确认测试确实来自本项目。
