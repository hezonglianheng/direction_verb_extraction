# 汉语空间语义语料构建与评测流水线

面向汉语空间语言研究（**方位词 `localization`** 与 **趋向动词 `direction`**）的语料抽取、句对构建、空间语义标注与模型评测工具集。

本仓库结合 **LTP 句法/语义分析**、**JSON5 规则树**、**OpenHowNet 义原约束** 与 **LLM 结构化推理**，把原始中文文本加工为可用于语言学分析与模型评测的句对语料。

---

## 目录

- [一图看懂流程](#一图看懂流程)
- [环境安装与运行准备](#环境安装与运行准备)
- [快速开始](#快速开始)
- [流水线与四大模块用法](#流水线与四大模块用法)
- [离线分析与常用工具](#离线分析与常用工具)
- [仓库结构](#仓库结构)
- [常见坑点与延伸文档](#常见坑点与延伸文档)

---

## 一图看懂流程

仓库包含四个核心模块。前三个阶段由 [`pipeline.py`](pipeline.py) 串联执行，[`evaluate/`](evaluate/) 作为独立评测入口运行：

```text
原始文本 JSONL ({"content": "..."})
   │
   ▼ 1. extract   切句 ─▶ 词表初筛 ─▶ LTP 分析 ─▶ 规则 & HowNet 筛选 ─▶ LLM 逐词筛选
   │              产出：候选原句 JSONL（含目标词三元组与筛选状态）
   ▼ 2. replace   单位置机械替换（alter_rules.json5）─▶ LLM 双侧语法与空间义检查
   │              产出：候选句对 JSONL（pair_schema_version = "1.0"）
   ▼ 3. annotate  准入合格句对 ─▶ 空间特征 ─▶ 图式匹配 ─▶ 同义判断 / 多模型投票
                  产出：正式标注 JSONL（schema_version = "1.3"）+ CSV + Summary
   │
   ├──▶ 4. evaluate       以 annotate 结果为参考，对模型运行 0–4 五档同义一致性评测
   └──▶ analyze_tools     离线导出 CSV、对比多轮运行差异（不调用模型）
```

### 支持的两类研究对象

所有命令需通过 `--corpus-type` 明确指定研究对象（**不根据文件名自动判断**）：

| 研究对象 | `--corpus-type` 可选值（英文不区分大小写） | 配置目录 | 当前就绪状态 |
| --- | --- | --- | --- |
| **方位词**（默认） | `方位词`、`localization`、`location`、`loc` | [`config_files/localization/`](config_files/localization/) | 抽取、替换、特征、图式、判定表均已配置 |
| **趋向动词** | `趋向动词`、`direction`、`dir`、`direction_verb` | [`config_files/direction/`](config_files/direction/) | 仅完成抽取词表与规则；`alter_rules.json5`、`feature_classes.json5`、`schema_table.json5`、`judge_table.csv`、`judge_examples.json5` 当前为空骨架，**跑完整替换/标注前需自行补齐** |

---

## 环境安装与运行准备

### 1. 安装依赖

- **环境要求**：Python `>=3.11,<3.12`（见 [`.python-version`](.python-version)），支持 Linux `x86_64`/`aarch64`、Windows `x86_64`、macOS `arm64`（见 [`pyproject.toml`](pyproject.toml)）。
- **必须从仓库根目录运行**：项目未打包为系统命令（`package = false`），请始终在根目录执行脚本或模块。

**方式 A：使用 `uv`（推荐）**

```bash
uv sync --locked --extra llm
```

`uv sync` 会创建 `.venv` 但不会自动激活终端。后续命令可二选一：
- 不激活终端，直接在命令前加 `uv run --extra llm`（如 `uv run --extra llm python -m extract --help`）；
- 先激活环境再直接用 `python`：Linux/macOS 执行 `source .venv/bin/activate`，Windows PowerShell 执行 `.\.venv\Scripts\Activate.ps1`。

**方式 B：使用 `venv` + `pip`**

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> **提示**：[`requirements.txt`](requirements.txt) 中的 CUDA/Triton 依赖带有 Linux `x86_64` 标记，跨平台安装时请勿删除。以下示例默认你**已激活 `.venv`**。

### 2. 准备模型、词典与 API Key

以下三项未随 Git 分发（见 [`.gitignore`](.gitignore)），首次运行前需在本地准备好：

1. **LTP 句法模型**：默认加载项目根目录的 `LTP_model/base2`；也可设置环境变量 `LTP_MODEL_PATH` 指向本地模型目录或 Hugging Face 名称（如 `LTP/base`）。默认使用 CPU，如需 GPU 请修改 [`config.py`](config.py) 的 `LTP_DEVICE`（支持 `cpu`/`gpu`/`cuda`/`auto`，不读取环境变量；指定 CUDA 但不可用时会直接报错）。
2. **OpenHowNet 义原词典**（`extract` 阶段检查缺失会中止）：
   ```bash
   python -c "import OpenHowNet; OpenHowNet.download()"
   ```
3. **LLM 接口配置**：复制 [`model_config/api_keys.example.json`](model_config/api_keys.example.json) 为 `model_config/api_keys.json` 并填写真实服务信息：
   ```json
   {
     "my-model": {
       "model_name": "实际模型ID",
       "provider": "deepseek",
       "base_url": "https://your-endpoint.example/v1",
       "api_key": "YOUR_API_KEY",
       "timeout": 120,
       "max_retries": 6
     }
   }
   ```
   - 命令行 `--model` / `--models` / `--screen-model` 传的是这里的**字典键**（如 `"my-model"`）。
   - 代码默认模型键为 `deepseek-flash`（示例文件未内置，请自行添加或在命令行显式传入自己的键）。
   - 默认不发送 `temperature`（由 [`config.py`](config.py) 的 `LLM_TEMPERATURE_ENABLED = False` 统一控制）；若需在 `extract`/`replace` 中自动启停本地服务（仅 POSIX），见 [`LOCAL_MODEL_SERVICE.md`](LOCAL_MODEL_SERVICE.md)。

---

## 快速开始

以**方位词（`localization`）**为例，三步完成端到端预检与运行：

### 第 1 步：准备输入 JSONL

每行需包含 **`content`** 字段（注意不是 `text`），保存为本地文件（如 `data/raw.jsonl`）：

```json
{"content": "小猫正安静地趴在桌子下面，桌子上放着一本书。"}
{"content": "图书馆大楼北面是一片草坪，教学楼就在操场东边。"}
```

### 第 2 步：预检命令（`--dry-run` 不调用模型）

```bash
# Bash (Linux / macOS / Git Bash)
python pipeline.py \
  --source data/raw.jsonl \
  --corpus-type localization \
  --extract-model my-model \
  --replace-model my-model \
  --annotate-args '["--model", "my-model", "--limit", "10", "--batch", "off"]' \
  --dry-run
```

```powershell
# Windows PowerShell 5.1（单行写法，JSON 数组内双引号需用 \" 转义；PowerShell 7.3+ 直接用 '["--model", ...]'）
python pipeline.py --source data/raw.jsonl --corpus-type localization --extract-model my-model --replace-model my-model --annotate-args '[\"--model\", \"my-model\", \"--limit\", \"10\", \"--batch\", \"off\"]' --dry-run
```

> 后文 `\` 折行命令均以 Bash 为例；Windows 用户可合并为单行，或直接使用下节的 `--config` JSON5 配置文件免去命令行引号转义。

### 第 3 步：正式运行

去掉 `--dry-run` 即可依次执行 `extract → replace → annotate`，默认结果输出在 `results/`：
- 提取原句：`results/location_corpus.jsonl`
- 替换句对：`results/location_pairs.jsonl`
- 空间标注：`results/location_annotated.jsonl`（同步生成 `.csv` 与 `.summary.json`）

---

## 流水线与四大模块用法

### 端到端编排（`pipeline.py`）

[`pipeline.py`](pipeline.py) 按固定顺序 `extract → replace → annotate` 执行，可用 `--from-stage` / `--to-stage` 截取连续片段（不能反向或跳过中间阶段）。某阶段退出码非零时立即中止，已完成的产物保持保留：

```bash
# 仅提取
python pipeline.py --from-stage extract --to-stage extract --source data/raw.jsonl --extract-model my-model

# 从已有原句接着跑替换和标注
python pipeline.py --from-stage replace --to-stage annotate --extracted results/location_corpus.jsonl --replace-model my-model --annotate-args '["--model", "my-model"]'

# 使用 JSON5 配置文件（参考并复制 pipeline_config/full.example.json5 后修改路径与模型）
python pipeline.py --config path/to/your_pipeline.json5 --corpus-type localization --dry-run
```

关键约定：
- **路径与覆盖**：各阶段输出文件须以小写 `.jsonl` 结尾；预检通过后会自动创建 `extract` 输出目录。默认拒绝覆盖已存在输出（含 CSV/summary/多模型原始文件），需加 `--overwrite`，且绝不允许输入与输出指向同一文件。
- **透传参数（`--*-args`）**：须为合法 JSON 字符串数组，且不能覆盖流水线托管的路径/词性参数；涉及路径规划的选项（如 `--out-file`、`--batch-cache`、`--graph`、`--vocab_config_path`、`--alter-rules`、`--print-prompt`）须写全名，不能缩写。
- **`--config` 模式**：配置内相对路径按**该文件所在目录**解析；`allow_retry` 支持 `true`/`false`/`null` 三态，`vocab_config_path` 可写字符串或数组，`annotate.out_dir` 与 `out_file` 二选一。`--corpus-type` 与 `--overwrite` 仍从 CLI 传入，且不能再混用 `--source`、`--extracted`、`--pairs`、`--output`、`--filter-config`、`--extract-model`、`--replace-model`、`--annotation-engine` 或 `--*-args`。

### 四大模块独立入口

所有模块支持 `python -m <模块>`（与 `python <模块>` 等价），`--help` 极快且不加载 LTP/PyTorch。
点击下表的模块名，可查看各模块的主要功能、实现流程、使用示例与代码导航。

| 模块 | 主要作用 | 典型命令 | 核心注意点 |
| --- | --- | --- | --- |
| **[`extract`](extract/README.md)** | 切句、LTP 分析、规则 & HowNet 过滤、LLM 初筛 | `python -m extract data/raw.jsonl -f config_files/localization/localization_expressions.json5 -t results -o location_corpus --screen-model my-model` | `-f` 必填；独立运行时 `-t` 目录须已存在并允许重写；未通过/出错的句也会落盘，记录级状态看 `screening.status`（`true`/`false`/`error`/`limit`/`no_llm`），逐词细节看 `screening.verdicts`（`no_llm` 时为 `[]`） |
| **[`replace`](replace/README.md)** | 按 `alter_rules.json5` 单位置机械替换并做双侧检查 | `python -m replace --corpus results/location_corpus.jsonl --corpus-type localization --out-file results/location_pairs.jsonl --model my-model` | 默认拒绝覆盖（需 `--overwrite`）；`--limit` 默认限制**可用原句数**（在 `aligned` + `--reference-level verdict` 下按展开的 verdict 计数）；支持 `sequential`/`random`/`proportional`/`aligned`；`--no-llm` 产物不可入标注 |
| **[`annotate`](annotate/README.md)** | 抽取空间特征、匹配图式并做同义判断或投票 | `python -m annotate --graph pair --corpus results/location_pairs.jsonl --corpus-type localization --model my-model --limit 100 --batch off` | 正式图默认 `--limit`：`pair`(100)、`synonym`(100)、`all_in_one`(1)，禁传 `0`（仅 `aligned` 省略时对齐全量）；兼容 `--engine staged\|all-in-one` 与 `python -m annotate.all_in_one`；下游只采纳 `admission.is_admitted=true` 记录 |
| **[`evaluate`](evaluate/README.md)** | 以 `annotate` 结果为参考，对模型做五档同义评测 | `python -m evaluate --corpus results/location_annotated.jsonl --corpus-type localization --models my-model --step 0 --limit 100 --batch off` | `--step` 0~4：0 直接判断，1/2 单次全标注（禁/启用查表），3/4 分步标注（禁/启用查表）；默认跑全量（`--limit 0` 为 0 条）；参考标签并非人工金标准，参与投票的模型结果会偏乐观 |

**`annotate` 补充说明**：
- **结构化错误重试**：`pair` / `synonym` 默认允许补发，`all_in_one` 默认禁止，可用 `--allow-retry` / `--no-allow-retry` 显式覆盖。
- **单模型 vs 多模型投票**：`--model A` 或 `--models A` 直接写入目标文件（不建 `.models/`）；`--models A B ...` 按 `voting_weight` 加权投票，将各模型原始结果存入 `<输出主名>.models/`，最终沿用代表模型完整记录（平票/无效票记 `discarded`，缺票转人工复核）。未传 `--out-file` 时输出至 `--out-dir`（默认 `results/<图前缀>/`）。
- **1.3 契约与抽样**：跨文件溯源请使用稳定的 `sentence_id` / `candidate_key`（四段 `id` 仅本次输入内有效）；`synonym` 的特征/图式准入为 `not_applicable`；`proportional` 按无向关注词对分层（`上-下` 与 `下-上` 同层）。
- **诊断图（`single` / `alter_check`）**：读取 `extract` 原句 JSONL，仅可在 `python -m annotate --graph single|alter_check --corpus <原句.jsonl>` 独立运行，不被 `pipeline.py` 编排。

**常用补充子命令**：
- 离线预览提示词（不调模型、不读密钥）：`python -m annotate --graph pair --corpus results/location_pairs.jsonl --print-prompt`
- 导入外部原始 SpaCE 句对（需自备含 `context1`/`context2`/`judge` 的 JSON/JSONL）：`python -m evaluate.import_sentence_pairs <原始文件>.json --output results/evaluate/imported.jsonl`
- 补跑评测失败条目：`python -m evaluate.rerun --details results/evaluate/<模型>_step<N>_<时间戳>.details.jsonl --timeout-seconds 240`

---

## 离线分析与常用工具

离线审查脚本**不调用模型、不修改输入文件**，更多参数见 [`analyze_tools/README.md`](analyze_tools/README.md)（下例中的 `run_a.jsonl`、`run_b.jsonl`、`old.jsonl` 均为用户自备的本地输入，`new.jsonl` / `new.csv` 为输出）：

```bash
# 导出逐词判定 / 句对 CSV
python analyze_tools/extract/export_extract_verdicts_csv.py results/location_corpus.jsonl
python analyze_tools/replace/export_replace_pairs_csv.py results/location_pairs.jsonl

# 对比两次运行结果（--output-dir 必须尚不存在）
python analyze_tools/extract/compare_extract_verdicts.py run_a.jsonl run_b.jsonl --output-dir analyze_results/extract_check
python analyze_tools/replace/compare_replace_pairs.py run_a.jsonl run_b.jsonl --output-dir analyze_results/replace_check
python analyze_tools/annotate/compare_annotate_results.py run_a.jsonl run_b.jsonl --output-dir analyze_results/annotate_check

# 其他实用工具：按词分层抽样 / 补筛 extract 错误条目 / 仅迁移 ID 与重排 CSV（保留原 schema_version，不补齐缺失证据）/ 查看 agent 列表
python -m extract.word_sampler --help
python -m extract.rescreen --help
python -m annotate.migrate_result_ids old.jsonl -o new.jsonl --csv-out new.csv
python -m annotate.spatial_agents --list
```

---

## 仓库结构

以下仅列出 Git 版本库中已跟踪的文件与目录（`LTP_model/`、`results/`、`log/`、`analyze_results/` 与 `model_config/api_keys.json` 需本地准备或运行时生成）：

```text
.
├── .python-version / pyproject.toml / requirements.txt / uv.lock   # Python 3.11 与依赖锁定
├── pipeline.py                                                     # 三阶段串联编排入口
├── pipeline_config/full.example.json5                              # pipeline --config 完整模板
├── model_config/api_keys.example.json                              # 模型配置与密钥模板
├── config.py / corpus_metadata.py                                  # 全局默认参数与语料类型元数据
├── agents.py / batch_inference.py / sync_cache.py / retry.py       # 模型调用、Batch 与同步缓存/重试
├── structured_schema.py / temperature_policy.py / pipeline_utils.py / file_paths.py / filter_funcs.py / agent_log.py / utils.py
│
├── extract/                        # 阶段 1：切句、LTP 分析、规则匹配、HowNet 约束与初筛
├── replace/                        # 阶段 2：机械替换与双侧检查
├── annotate/                       # 阶段 3：特征、图式、同义判断与多模型投票
├── evaluate/                       # 独立评测：0–4 五档同义评测、导入与补跑
│
├── config_files/
│   ├── localization/               # 方位词词表、匹配/替换规则、特征/图式/判定表与示例
│   └── direction/                  # 趋向动词词表、匹配规则及当前空骨架配置
├── system_prompt/                  # 各阶段提示词模板（维护索引见 system_prompt/README.md）
├── analyze_tools/                  # extract / replace / annotate 离线 CSV 导出与对比工具
└── LOCAL_MODEL_SERVICE.md          # extract / replace 本地推理服务自动启停说明
```

---

## 常见坑点与延伸文档

1. **包入口与导入规范（旧脚本已删除）**：旧根目录转发脚本（`sentence_extract.py`、`sentence_replace.py`、`spatial_runner.py`、`all_in_one.py`、`word_sampler.py`、`migrate_result_ids.py` 等）已移除。请勿 `cd` 进子包目录运行或把子目录单独加入 `PYTHONPATH`；请统一在仓库根目录运行 `python -m extract`、`python -m replace`、`python -m annotate`、`python -m evaluate` 或 `python pipeline.py`，外部 Python 脚本应使用完整包路径导入（如 `from annotate import spatial_runner`）。
2. **调整分词切分配置须整批重跑**：[`config.py`](config.py) 的 `LOCATIVE_SPLIT_ENABLED` 默认开启（`True`）。改变切分开关、后缀排除集或黑名单会引起 token 下标位移，旧数据不可增量续跑。
3. **两类语料的 HowNet 语义边界**：规则与词表严格面向**汉语**。虽 [`config_files/localization/reality.txt`](config_files/localization/reality.txt) 包含物理/心理/信息空间判据，但前置 HowNet grounding 会过滤不满足义原树的句子：`localization` 检查目标物与参照物（见 `extract/localization_grounding.py`），`direction` 仅检查目标物是否为 `physical|物质` 子树、不筛选动词与参照物（见 `extract/direction_grounding.py`）。
4. **提示词、Batch 续跑与本地服务范围**：
   - `annotate` 提示词按链路存放在 [`system_prompt/annotate/`](system_prompt/annotate/) 下的 `pair/`、`synonym/`、`all_in_one/` 及共享片段 `annotate_shared_feature_*.txt`；[`system_prompt/evaluate/`](system_prompt/evaluate/) 为历史留存模板，当前 `evaluate` 统一复用 `annotate` 提示词与查表资源。
   - `evaluate` 在原生 Batch 受理后出现失败/缺失项时不会自动改走同步重发，需用 `python -m evaluate.rerun` 补跑；`--batch-resume` 必须搭配显式 `--batch-cache`（`evaluate` 每次默认报告路径含新时间戳）。
   - `local_service` 自动启停仅挂在 `extract`、`replace` 与 `extract.rescreen`，在 `annotate`/`evaluate` 中使用本地模型前请自行启动服务。

**延伸文档：**
- 本地模型服务生命周期管理：[`LOCAL_MODEL_SERVICE.md`](LOCAL_MODEL_SERVICE.md)
- 提示词模板索引与占位符规范：[`system_prompt/README.md`](system_prompt/README.md)
- 离线导出与对比工具手册：[`analyze_tools/README.md`](analyze_tools/README.md)
