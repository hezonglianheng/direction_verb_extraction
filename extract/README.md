# extract：从原始文章中提取候选句

本模块从中文文章中找出包含**方位词**或**趋向动词**的句子，记录目标词位置、空间角色和筛选依据，供下一步构建句对。

[返回项目说明](../README.md) · [下一步：replace](../replace/README.md)

## 主要功能

- 按标点和空白切句，过滤不符合长度要求的片段。
- 用词表和 LTP 分析结果匹配语言学规则，定位句中的目标词。
- 整理“目标物、参照物、动词”等空间角色，并用 OpenHowNet 词典检查其语义。
- 通过大语言模型（LLM）检查目标词所在句子的语法、空间义与自洽性。
- 保存候选句、逐词判定和提取统计，便于复查或补筛。

## 实现流程

```text
原始文章 JSONL
  → 切句、长度过滤
  → 词表初筛、窗口内去重
  → LTP 分词、词性与依存分析
  → JSON5 规则匹配
  → 空间角色整理、HowNet 语义检查
  → LLM 逐词筛选
  → 原句 JSONL + 提取日志
```

1. **先用低成本条件缩小范围**：读取文章的 `content` 字段，按句末标点及空白切分，默认保留 10–1000 字符的片段。初筛词表通常从规则自动推导；去重只记住有限窗口内的原句，不保证全语料去重。
2. **再用句法规则定位**：LTP 提供分词、词性、句法依存（DEP）和语义依存（SDP）结果。规则树组合词汇、词性与依存条件，并保留命中位置和匹配树，而不只是返回“通过/不通过”。
3. **整理空间角色**：方位词链路检查目标物和参照物；趋向动词链路仅对目标物做义原检查，动词、参照物仍会记录，但不参与该语义筛选。
4. **最后调用模型**：一次请求可包含多个“句子＋目标词”判定项，多个请求可并发执行。这种组批不是供应商的原生 Batch API。

为减少重复计算，LTP 结果按句缓存并批量预填充；切句、规则筛选与 LLM 等待之间使用有界预取，避免把全部原文同时放进内存。入口会自动选择 Windows 或非 Windows 执行器，用户命令不变。

## 输入与输出

**输入**：单个 JSONL 文件，或目录直属的 `*.jsonl` 文件。每行是一篇文章，例如：

```json
{"content": "小猫正安静地趴在桌子下面，桌子上放着一本书。"}
```

**输出**：每行一个候选原句，主要信息如下。

| 字段 | 含义 |
| --- | --- |
| `sentence` | 分词后的词列表，不是原始文章字符串 |
| `source_file` | 原始输入文件 |
| `filter_methods`、`match_tree` | 命中的规则及词位证据 |
| `localization_grounding` / `direction_grounding` | 对应链路的空间角色和语义检查信息 |
| `screening.status` | 记录级筛选状态 |
| `screening.verdicts` | 各目标词的判定与理由 |

`screening.status` 分为 `true`、`false`、`error`、`limit`、`no_llm`。**保存下来不等于筛选合格**；未调用模型时，`verdicts` 为空列表。提取结束后另在项目根目录的 `log/` 中写入统计日志。

有逐词判定时，记录状态按 `error > limit > true > false` 汇总：没有错误或限额、且至少一个目标词通过时，记录可为 `true`，未通过的词位会从匹配树中裁掉；这不表示句中所有目标词都通过。

## 使用示例

先完成[环境与模型配置](../README.md#环境安装与运行准备)。以下命令均在**仓库根目录、已激活的虚拟环境**中执行；`my-model` 是本地密钥配置中的模型键，`data/raw.jsonl` 是自备输入，`results/` 是本地输出目录。

```bash
# 独立入口要求输出目录已存在
python -c "from pathlib import Path; Path('results').mkdir(exist_ok=True)"

# 提取方位词句子，并调用模型筛选
python -m extract data/raw.jsonl -f config_files/localization/localization_expressions.json5 -t results -o location_corpus --screen-model my-model

# 提取趋向动词句子，暂不调用 LLM（仍执行 LTP 与 HowNet 检查）
python -m extract data/raw.jsonl -f config_files/direction/direction_expressions.json5 -t results -o direction_corpus --no-llm-screen

# 查看完整参数
python -m extract --help
```

### 常用参数

| 参数 | 用途 |
| --- | --- |
| `-f` / `--filter_config_path` | 必须显式选择规则；重复传入可同时运行多套规则 |
| `-v` / `--vocab_config_path` | 覆盖初筛词表，通常建议省略以自动推导 |
| `-t`、`-o` | 输出目录和文件名；文件名自动补 `.jsonl` |
| `--screen-model` | LLM 筛选模型键 |
| `--screen-workers`、`--screen-batch-size` | 请求并发数和每请求的判定项数 |
| `--extract-workers`、`--filter-workers` | 切句线程数、规则筛选进程数，默认分别为 4、1 |
| `--sentence-dedup-window` | 默认记住最近 200000 个初筛后唯一句的哈希；`0` 关闭 |
| `--no-llm-screen` | 跳过 LLM 筛选，不跳过本地分析 |

单套规则未指定 `-o` 时，输出为 `filtered_sentences.jsonl`；多套规则各写一个文件，`-o` 会作为文件名前缀。

## 代码导航

| 文件 | 负责什么 |
| --- | --- |
| [`cli.py`](cli.py)、[`sentence_extract.py`](sentence_extract.py) | 解析参数、加载规则、规划输出并选择平台执行器 |
| [`entry_in_linux.py`](entry_in_linux.py)、[`entry_in_win.py`](entry_in_win.py) | 串联流式读取、规则筛选、组批与写出 |
| [`sentence_split.py`](sentence_split.py) | 两个平台共用的切句与词表初筛 |
| [`sentenceFilters.py`](sentenceFilters.py) | 规则树解析、匹配及命中解释 |
| [`sentenceTable.py`](sentenceTable.py)、[`locative_split.py`](locative_split.py) | LTP 加载、分析缓存与方位词后缀切分 |
| [`localization_grounding.py`](localization_grounding.py)、[`direction_grounding.py`](direction_grounding.py) | 空间角色整理与 HowNet 语义检查 |
| [`sentence_screening.py`](sentence_screening.py) | 逐词 LLM 筛选、判定状态与请求组批 |
| [`extraction_log.py`](extraction_log.py)、[`screening_metrics.py`](screening_metrics.py) | 提取漏斗统计与筛选指标 |
| [`word_sampler.py`](word_sampler.py)、[`rescreen.py`](rescreen.py) | 按词抽样，以及对失败条目定点补筛 |
| [`local_model_service.py`](local_model_service.py) | 本地推理服务的自动启停，详见[接入说明](../LOCAL_MODEL_SERVICE.md) |

规则和词表位于 [`config_files/localization/`](../config_files/localization/) 与 [`config_files/direction/`](../config_files/direction/)，筛选提示词位于 [`system_prompt/extract/`](../system_prompt/extract/)。

## 注意事项

- **独立入口可能重写同名输出**，不会像 `replace`、`annotate` 那样要求 `--overwrite`；正式运行前请确认输出路径。
- LTP 模型和 HowNet 词典需本地准备；`--no-llm-screen` 不能绕过这些依赖。
- 调整分词模型或后缀切分配置可能改变 token 下标，相关语料应重新提取，不要混用新旧词位。
- 检查结果可交给 [extract 离线分析工具](../analyze_tools/extract/README.md)；原句进入 [replace](../replace/README.md) 后，缺少完整判定的目标词还会补筛。
