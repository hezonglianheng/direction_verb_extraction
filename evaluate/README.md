# evaluate：评测模型与参考标注的一致性

本模块读取已有句对及参考答案，让待测模型重新判断是否同义，再统计一致率、覆盖率和分类指标。它是**独立评测入口**，不包含在 `pipeline.py` 的三阶段构建流程中。

[返回项目说明](../README.md) · [参考标注：annotate](../annotate/README.md)

## 主要功能

- 读取单模型标注、多模型投票结果，或经导入器转换的外部句对。
- 提供直接判断、单次完整标注、分步标注共五档评测。
- 同时评测多个模型，各自输出报告，不在评测阶段重新投票。
- 保存逐条预测、错误与指标，支持失败补跑和缓存续跑。

**指标衡量的是“与参考答案的一致性”**。使用 `annotate` 生成的模型标注时，它不是人工金标准；待测模型若参与过参考标注或投票，结果可能偏乐观。

## 实现流程

```text
参考标注 JSONL
  → 校验句对与参考标签，过滤无效项并去重
  → 按 offset / limit 选择样本
  → 仅把句子与关注词交给待测模型
  → 按指定 step 执行评测
  → 对照参考答案计分
  → 每模型 JSON 报告 + 逐条明细
```

1. **准备参考样本**：`data.py` 解析两侧句子、参考标签与来源身份。无效标签、非空间或不通顺记录、未采纳的投票等跳过并计数。
2. **隔离答案与预测**：`annotation.py` 只向模型传入文本和清理后的关注词信息，不复制参考判断、参考特征或参考图式，避免把答案喂回模型。
3. **复用标注实现**：五档评测调用 `annotate` 的 agent 和节点，通过评测专用适配器控制是否启用查表，不修改生产标注的全局注册表。
4. **执行与汇总**：`inference.py` 管同步、原生 Batch、请求去重、缓存身份检查和脱敏；`runner.py` 管模型并行、进度与报告；`metrics.py` 负责纯本地计分。

### 五档评测怎么选

| `--step` | 执行路径 | 本地图式 / 判定查表 |
| ---: | --- | --- |
| `0`（默认） | 直接同义判断 | 不使用 |
| `1` | 单次完整标注：特征、图式、同义关系 | 禁用 |
| `2` | 单次完整标注：特征、图式、同义关系 | 启用 |
| `3` | 分步执行：特征 → 图式 → 判定 | 禁用 |
| `4` | 分步执行：特征 → 图式 → 判定 | 启用 |

“禁用查表”指不采用本地匹配得到的直接结论，**不代表移除所有配置或示例**；图式候选、示例与输出校验仍可能使用对应资源。当前评测复用 `annotate` 提示词，不读取历史的 `system_prompt/evaluate/` 模板。

## 输入与输出

**输入**：单个参考标注 JSONL，或目录直属的 `*.jsonl` 文件。两侧句子不能为空，参考标签按以下规则提取：

- **单模型记录**：`judgement.same_meaning` 必须是真正的布尔值，不接受字符串 `"true"` 或数字 `1`。
- **投票记录**：只在 `voting.status=accepted` 时读取 `voting_label` 的 `same` / `different`；不会回退到代表模型残留的 `judgement`。
- **外部原始句对**：先用 `import_sentence_pairs.py` 将 `context1`、`context2`、布尔 `judge` 转为评测格式，不要求伪装成完整的正式标注。

**输出**：默认写入项目根目录的 `results/evaluate/`，每个模型独立生成：

```text
<模型名>_step<N>_<UTC时间戳>.json
<模型名>_step<N>_<UTC时间戳>.details.jsonl
```

JSON 报告包含样本选择、跳过统计、模型与运行信息及指标；明细包含参考答案、预测和错误信息，使用 `--no-details` 可关闭明细。

### 如何阅读指标

| 指标 | 含义 |
| --- | --- |
| `accuracy` | 主一致率：一致数量 / 全部所选样本，预测失败也计入分母 |
| `accuracy_on_success` | 只在成功预测中计算的一致率 |
| `coverage` | 成功给出有效预测的样本比例 |
| `confusion` | 混淆矩阵，行是参考标签，列是预测标签 |
| `per_label`、`macro_f1`、`weighted_f1` | 各类别的 precision / recall / F1 及总体汇总 |

不要只比较成功项一致率而忽略覆盖率。失败样本还会计入对应类别的 support 和 false negative，避免“失败越多，指标反而越高”。

## 使用示例

先完成[环境与模型配置](../README.md#环境安装与运行准备)。以下命令在**仓库根目录、已激活的虚拟环境**中执行；`my-model` 是已配置模型键，参考文件需预先生成或自备，`results/evaluate/` 为运行时输出。

```bash
# 直接同义判断，最多评测 100 个有效参考样本
python -m evaluate --corpus results/location_annotated.jsonl --corpus-type localization --models my-model --step 0 --limit 100 --batch off

# 离线查看分步查表评测的提示词，不读模型配置、不调用模型、不写报告
python -m evaluate --corpus results/location_annotated.jsonl --corpus-type localization --step 4 --print-prompt

# 查看完整参数
python -m evaluate --help
```

### 导入外部句对与补跑失败项

下例 `data/reference_pairs.json` 是用户自备的原始句对，`previous_run.details.jsonl` 需替换为某次实际评测明细；两者均不是仓库附带样例。

```bash
python -m evaluate.import_sentence_pairs data/reference_pairs.json --output results/evaluate/imported.jsonl
python -m evaluate --corpus results/evaluate/imported.jsonl --corpus-type localization --models my-model --step 0 --batch off

python -m evaluate.rerun --details results/evaluate/previous_run.details.jsonl --model my-model --timeout-seconds 240
```

补跑默认使用同步模式，只重试失败条目，保留原成功项的结果；另写回填后的明细和含补跑、合计统计的报告，不就地修改原文件。

### 常用参数

| 参数 | 用途 |
| --- | --- |
| `--corpus` / `--input` | 参考标注文件或目录，必填 |
| `--models` / `--model` | 一个或多个待测模型键；仅提示词预览时可省略 |
| `--step`、`--corpus-type` | 评测档位及语料类型，默认 `0`、方位词 |
| `--limit`、`--offset` | 有效样本选择范围；省略上限为全部，`0` 表示不选样本 |
| `--workers`、`--model-workers` | 单模型内部并发默认 8，模型并发默认 `0` 即全部并行 |
| `--batch auto\|native\|off` | 自动选择原生 Batch、强制原生或只走同步 |
| `--batch-cache`、`--batch-resume` | 原生 Batch 缓存与任务续跑 |
| `--sync-cache`、`--sync-resume` | 可选的同步成功节点缓存与续跑 |
| `--out-dir`、`--no-details`、`--overwrite` | 报告目录、关闭逐条明细、允许覆盖同名报告 |

## 代码导航

| 文件 | 负责什么 |
| --- | --- |
| [`runner.py`](runner.py) | CLI、路径预检、样本选择、模型调度与报告 |
| [`data.py`](data.py) | 参考标签提取、句对解析、去重和跳过原因 |
| [`annotation.py`](annotation.py) | 五档评测适配器、干净输入与阶段结果校验 |
| [`inference.py`](inference.py) | 同步 / Batch 执行、请求去重、缓存身份和脱敏 |
| [`metrics.py`](metrics.py) | 一致率、覆盖率、混淆矩阵和分类指标 |
| [`progress.py`](progress.py) | 多模型运行进度显示 |
| [`import_sentence_pairs.py`](import_sentence_pairs.py) | 外部原始句对格式转换 |
| [`rerun.py`](rerun.py) | 失败项补跑与合计报告 |

## 注意事项

- `evaluate --limit 0` 表示零样本，与 `annotate` 正式图禁止 `0` 的规则不同。
- 原生 Batch **受理后**，失败或缺失项不会偷偷改为同步重发；需要时显式使用补跑工具，避免重复计费或改变评测口径。
- Batch 续跑必须同时指定 `--batch-resume` 和旧 `--batch-cache`，因为新报告的默认文件名带新时间戳；同步续跑也必须显式指定旧 `--sync-cache`。
- 本模块不自动启停本地推理服务；使用本地模型前请自行确保服务就绪。
- 报告可用 [evaluate 离线分析工具](../analyze_tools/evaluate/README.md) 对齐明细、比较模型或对照参考标注；help 和提示词预览不能证明真实 API 可用。
