# annotate：为空间表达句对添加语义标注

本模块读取经过检查的句对，提取两侧的空间特征、匹配空间图式，并判断它们能否表达相同的空间场景。也可让多个模型分别标注，再加权投票。

[返回项目说明](../README.md) · [上一步：replace](../replace/README.md) · [下一步：evaluate](../evaluate/README.md)

## 主要功能

- 提供分步标注、直接同义判断和单次完整标注三条正式链路。
- 结合 LLM 与本地特征、图式、判定表，保存结论及来源。
- 支持多模型加权投票、限额抽样、提示词预览和断点续跑。
- 输出 JSONL、CSV、汇总与日志，并标记哪些结果可正式采用。

**正式标注只消费合格句对**：不会重新替换、补筛或准备句法证据。这些工作由 [`extract`](../extract/README.md) 和 [`replace`](../replace/README.md) 完成。

## 实现流程

```text
replace 句对 JSONL
  → 校验两侧检查与句对结构
  → 抽样并选择标注图
  → 单模型执行，或各模型独立执行后投票
  → 结果准入检查
  → 排序写出 JSONL / CSV / 汇总
```

### 三条正式标注链路

这里的“图”就是一组按顺序连接的处理步骤，由 LangGraph 编排：

| `--graph` | 如何处理 | 默认 `--limit` |
| --- | --- | ---: |
| `pair`（默认） | 两侧特征抽取 → 两侧图式匹配 → 同义判断 | 100 |
| `synonym` | 直接判断同义，不抽特征，不查图式或判定表 | 100 |
| `all_in_one` | 一次模型响应给出两侧特征、图式及同义关系，再做本地解析 | 1 |

- **特征**：描述空间场景的属性，如范围、竖直方向、水平位置、距离。
- **图式**：按关注词及特征匹配得到的空间关系分类。
- **本地表优先**：`pair` 在查表未得到直接结论时调用模型兜底；`all_in_one` 在模型返回后仍优先采用本地图式表、判定表的直接结论。`synonym` 不走这些表。

处理职责分为三层：`spatial_runner.py` 管输入、并发与落盘，`spatial_graphs.py` 管步骤连接，`spatial_agents.py` 管每一步的模型调用和业务判断。三条链路共用执行、缓存、日志和结果格式。

### 多模型投票

`--models A B` 会让两个模型独立处理**同一批句对**，再按配置里的 `voting_weight` 加权投票。汇总保留一个代表模型的完整记录，不跨模型拼接特征和图式。

平票、无有效票或缺票不能当作可靠共识，需查看 `voting` 状态并复核。`--models A` 等价于 `--model A`，不会启动投票或创建多模型原始结果目录。

## 输入与输出

**输入**：[`replace`](../replace/README.md) 生成的 1.0 句对 JSONL。正式图会复核两侧逐词判定、关注词位置、文本切片和单位置替换关系，不只信任 `pair_screening.status`。

**输出**：正式结果的 `schema_version="1.3"`，`annotation_mode` 标明所用链路。

| 信息 | 主要字段或文件 |
| --- | --- |
| 稳定身份与来源 | `sentence_id`、`candidate_key`、来源信息 |
| 两侧语义与判断 | `features`、`schema`、`judgement` |
| 结果是否可采用 | `admission.is_admitted`、复核说明 |
| 多模型结论 | `voting`、`voting_label` |
| 附属输出 | 同名 `.csv`、`.summary.json`，以及 `log/` 下的日志 |

跨文件对账应使用 `sentence_id` / `candidate_key`，而不是仅在本次输入中有效的 `id`。**`status=ok` 不代表结果已准入**；未知、错误、预算耗尽或未调用模型的结果需要复核。`synonym` 不要求特征和图式证据，对应准入项为 `not_applicable`。

## 使用示例

先完成[环境与模型配置](../README.md#环境安装与运行准备)。以下命令在**仓库根目录、已激活的虚拟环境**中执行；`results/location_pairs.jsonl` 是上游输出，`my-model`、`model-a`、`model-b` 均需替换为已配置模型键，其他结果路径为运行时输出。

```bash
# 分步标注最多 100 个合格句对
python -m annotate --graph pair --corpus results/location_pairs.jsonl --corpus-type localization --model my-model --limit 100 --batch off --out-file results/location_annotated.jsonl

# 使用两个模型直接判断同义，再加权投票
python -m annotate --graph synonym --corpus results/location_pairs.jsonl --corpus-type localization --models model-a model-b --limit 100 --batch off --out-file results/location_vote.jsonl

# 离线预览单次完整标注提示词，不读密钥、不调用模型、不写输出
python -m annotate --graph all_in_one --corpus results/location_pairs.jsonl --corpus-type localization --print-prompt

# 查看完整参数
python -m annotate --help
```

指定 `--out-file` 可固定输出路径；否则默认在 `results/<图前缀>/` 下生成带时间戳的文件。投票时，各模型原始文件另存于 `<输出主名>.models/`。

### 常用参数

| 参数 | 用途 |
| --- | --- |
| `--graph` | 选择 `pair`、`synonym` 或 `all_in_one` |
| `--corpus-type` | 默认方位词；趋向动词需显式指定 `direction` |
| `--model` / `--models` | 单模型或多模型标注，二者互斥 |
| `--limit`、`--offset` | 处理数量上限、跳过的合格句对数 |
| `--limit-mode`、`--seed` | 顺序、随机或按无向关注词对比例抽样 |
| `--reference-file` | 按既有标注结果的 `candidate_key` 对齐 |
| `--workers`、`--model-workers` | 模型内部的记录并发数、同时运行的模型数 |
| `--feature-deep` | 在 `pair` / `single` 中按类别分工抽取特征，默认使用单次 flat 抽取 |
| `--allow-retry` / `--no-allow-retry` | 是否补发结构化输出错误；默认 `pair` / `synonym` 允许，`all_in_one` 禁止 |
| `--max-requests`、`--batch-max-requests` | 候选和整批请求预算 |
| `--batch auto\|native\|off` | 自动选择原生 Batch、强制原生或只走同步 |
| `--out-file`、`--no-pairs-csv`、`--overwrite` | 指定结果、关闭 CSV、允许覆盖已有输出 |

正式图的 `--limit` 必须大于 0；省略时按上表默认值处理。提供 `--reference-file` 后自动进入 `aligned` 模式，此时省略 `--limit` 表示对齐参考文件的全部句对，不能再配合 `--seed` 或其他抽样模式。

原生 Batch 续跑用 `--batch-cache` 配合 `--batch-resume`；同步成功节点缓存用 `--sync-cache` 配合 `--sync-resume`，两种缓存不混用。同步缓存默认不写，续跑必须显式指定原缓存路径。

## 代码导航

| 文件 | 负责什么 |
| --- | --- |
| [`cli.py`](cli.py)、[`cli_args.py`](cli_args.py)、[`cli_constants.py`](cli_constants.py) | 参数解析、图选择与默认数量 |
| [`spatial_runner.py`](spatial_runner.py) | 调度、并发、缓存、日志、投票及结果写出 |
| [`spatial_graphs.py`](spatial_graphs.py) | 图状态、节点连接、结果序列化与准入检查 |
| [`spatial_agents.py`](spatial_agents.py) | 特征、图式、同义判断等 agent |
| [`spatial_tools.py`](spatial_tools.py)、[`annotation_tables.py`](annotation_tables.py) | 输入读取、身份、配置解析和本地查表 |
| [`limit_sampler.py`](limit_sampler.py)、[`voting.py`](voting.py) | 句对抽样与多模型加权投票 |
| [`prompt_examples.py`](prompt_examples.py)、[`prompt_cache.py`](prompt_cache.py)、[`token_usage.py`](token_usage.py) | 示例选择、提示词缓存和 token 统计 |
| [`export_pairs_csv.py`](export_pairs_csv.py)、[`sample_pairs_csv.py`](sample_pairs_csv.py)、[`migrate_result_ids.py`](migrate_result_ids.py) | CSV 导出、抽样及 ID 迁移 |
| [`all_in_one.py`](all_in_one.py) | 兼容入口，转发到统一 runner 的 `all_in_one` 图 |

特征与图式规则位于 [`config_files/`](../config_files/)，提示词位于 [`system_prompt/annotate/`](../system_prompt/annotate/)；修改模板前请阅读[提示词维护索引](../system_prompt/README.md)。

## 注意事项

- `single` 和 `alter_check` 是读取原句的**诊断图**，不属于正式句对标注，也不由 `pipeline.py` 编排。
- `--no-llm` 不会补造语义证据；`--print-prompt` 是提示词预览，不是标注完成证明。`pair` 的下游动态输入依赖前序结果，预览只说明依赖，不伪造内容。
- 趋向动词的默认特征与图式表仍为空骨架，不能据此认为已有完整分类体系。
- 结果默认拒绝覆盖；ID 迁移工具只重编 ID、排序和导出 CSV，**不会升级旧 schema 或补齐缺失证据**。
- 单模型与投票结果可交给 [annotate 离线分析工具](../analyze_tools/annotate/README.md) 比较，再由 [evaluate](../evaluate/README.md) 做独立评测。
