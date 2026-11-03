# 提示词维护索引

正式标注有三条链路：`pair` 分步标注、`synonym` 独立同义判断、`all_in_one` 单次完整标注。
三者只读取已通过两侧筛选的合格句对，不重新替换、筛选或准备句法证据。
文件按链路和步骤命名；类别片段、示例和配置表共享，不为链路复制近似正文，不保留旧名副本。

## 正式链路主提示词

| 链路 | 步骤/条件 | 文件 | 读取与渲染 |
| --- | --- | --- | --- |
| pair | 默认 flat 特征抽取 | `annotate_pair_feature_flat.txt` | `FeatureAgent`；`{{n}}`、`{{classes}}` |
| pair | 可选 deep 特征抽取 | `annotate_pair_feature_deep.txt` | `FeatureAgent`；`{{n}}`、`{{roster}}`；每类交给 subagent |
| pair | 特征类别表为空时的自由抽取 | `annotate_pair_feature_unfilled.txt` | `FeatureAgent`；flat/deep 均走此兜底 |
| pair | 按本侧关注词收窄并查图式表，未命中时 LLM 兜底 | `annotate_pair_schema.txt` | `SchemaAgent`；候选名单与特征放在 user message |
| pair | 判定表没有直接结论时 LLM 兜底 | `annotate_pair_judgement.txt` | `JudgeAgent`；表的直接结论优先 |
| synonym | 不抽特征、不查图式或判定表的独立同义判断 | `annotate_synonym_judgement.txt` | `SynonymAgent`；正文由用户维护，空提示语不调用模型 |
| all_in_one | 一次给出两侧特征、两侧图式与同义关系 | `annotate_all_in_one.txt` | `AllInOneAgent`；`{{n}}`、`{{roster}}` |

`all_in_one` 使用自己的主提示词，不把 pair 的三个步骤提示词拼成新判据，也不重新判断语法或空间性。
完成单次模型输出后，仍优先采用本地图式表和判定表的直接结论；没有直接结论才采用本次模型结果。

## 共享类别模板

| 文件 | 共享关系 | 占位符/用途 |
| --- | --- | --- |
| `annotate_shared_feature_class.txt` | pair flat/deep 与 all_in_one 共用一份 | `{{name}}`、`{{description}}`、`{{values}}`、`{{examples}}`、`{{usage}}` |
| `annotate_shared_feature_usage_deep.txt` | pair deep 的每类 subagent | 填入共享类别模板的 `{{usage}}` |
| `annotate_shared_feature_usage_flat.txt` | pair flat 与 all_in_one | 填入同一模板的 `{{usage}}` |

类别名称、判据与取值来自本词性的 `feature_classes.json5`。`render_class_blocks` 渲染 flat/all_in_one
共用的类别段；deep 用同一个类别模板，仅工作方式不同。模板中的 `{{examples}}` 当前留空，
示例统一按本侧关注词从下列 JSON5 选取并放入 user message，不读取类别表中遗留的无筛选示例。

## single 诊断与非正式标注资源

- `single` 是诊断图，复用 pair 的 `FeatureAgent`（flat/deep/未填兜底）和 `SchemaAgent`，不另建提示词副本。
- `screen_localization.txt`、`screen_direction.txt` 是上游筛选模板；reality/semantic 诊断回退也复用它们。
- `judgement_criteria.txt` 是筛选模板注入的语法/空间自洽判据片段。
- `config_files/localization/reality.txt`、`config_files/direction/reality.txt` 是按词性分组的空间判据片段，
  由筛选模板注入；正式三条标注链路不注入这些判据、不重筛。
- `semantic.txt` 保留原文件名，是旧资源而非三条正式标注主提示词；当前 `SemanticAgent` 使用筛选模板。

## 示例来源（不按链路复制）

示例由 `annotate/prompt_examples.py` 按步骤、词性和本侧关注词精确匹配；同一词命中的多个组按文件顺序共享。

| 步骤 | 方位词 | 趋向动词 |
| --- | --- | --- |
| 特征 | `examples/feature_localization.json5` | `examples/feature_direction.json5` |
| 图式 | `examples/schema_localization.json5` | `examples/schema_direction.json5` |

pair/single 每侧每步选自己的示例；all_in_one 的 user message 分成原句/改写句 × 特征/图式四个独立块。
synonym 不读取这两类示例。

## 配置表与维护入口

方位词使用 `config_files/localization/`，趋向动词使用 `config_files/direction/`；两组表同名、内容独立。

| 配置文件 | 使用方 |
| --- | --- |
| `feature_classes.json5` | pair/single 特征及 all_in_one 的共享类别段、输出字段与取值校验 |
| `schema_table.json5` | pair/single 图式收窄与查表；all_in_one 候选及后置查表 |
| `judge_table.csv` | pair 判定查表；all_in_one 后置判定查表 |
| `alter_rules.json5` | 上游 replace 或改写诊断；三条正式标注链路不重新替换 |

synonym 不依赖上述特征、图式或判定表。配置路径统一由 `annotate/spatial_tools.py` 解析，不把判据移入代码。

- 文件名常量集中在 `annotate/spatial_agents.py` 的提示语区；`FEATURE_*` 常量名不变，值指向本索引文件。
  `ALL_IN_ONE_PROMPT_FILE = "annotate_all_in_one"` 同样在此处，由 `AllInOneAgent` 读取主提示词。
- `utils.prompt_path` / `utils.read_prompt` 接口不变，名称参数不带 `.txt`；路径以项目根为准，不依赖 cwd。
- `python -m annotate.spatial_agents --list` 列出 agent 的实际提示词路径；三条正式图均支持 `--print-prompt`。
  pair 展示实际特征输入与各步骤 system prompt，并说明 schema/judge 对前序真实输出的依赖，绝不伪造结果。
  synonym/all_in_one 展示完整渲染消息。预览不读模型密钥或投票权重，不建模型、日志或输出。
- 仅改文件名不改变 agent 名、system/user 正文或渲染输入，因此不应使内容缓存无意义失效，也不跨链路串用缓存。
- 同步部署应完整包含十个 `annotate_*.txt`、本索引、四个示例 JSON5 与配置表；旧名文件须随重命名移除。
  历史审查报告保留当时路径，不作为当前维护入口。

### 部署重命名映射（旧名必须删除）

下表路径均相对 `system_prompt/`；只改文件名，正文、渲染输入和内容缓存基线不变。

| 旧文件 | 当前文件 |
| --- | --- |
| `feature.txt` | `annotate_pair_feature_deep.txt` |
| `feature_flat.txt` | `annotate_pair_feature_flat.txt` |
| `feature_unfilled.txt` | `annotate_pair_feature_unfilled.txt` |
| `schema.txt` | `annotate_pair_schema.txt` |
| `judge.txt` | `annotate_pair_judgement.txt` |
| `synonym.txt` | `annotate_synonym_judgement.txt` |
| `all_in_one.txt` | `annotate_all_in_one.txt` |
| `feature_class.txt` | `annotate_shared_feature_class.txt` |
| `feature_class_usage.txt` | `annotate_shared_feature_usage_deep.txt` |
| `feature_class_usage_flat.txt` | `annotate_shared_feature_usage_flat.txt` |

代码、资源和旧文件删除须作为同一版本部署。文件复制部署也须按表逐项删除旧名，不能保留新旧双副本，
也不能广域删除提示词目录；筛选判据及四个示例文件不在删除范围内。

## 共同执行入口

三图统一由 `annotate/spatial_runner.py` 执行，同步、Batch 波次及多模型投票共用同一 runner：

```sh
python annotate --graph pair --corpus results/location_pairs.jsonl --out-file results/pair.jsonl --limit 100
python annotate --graph synonym --corpus results/location_pairs.jsonl --out-file results/synonym.jsonl --limit 100
python annotate --graph all_in_one --corpus results/location_pairs.jsonl --out-file results/all_in_one.jsonl --limit 1
```

三者均可加 `--corpus-type 趋向动词` 并提供对应合格句对；未指定词性仍为方位词，不按文件名猜测。
上述正式运行会使用模型；仅离线检查时使用 `--print-prompt`，不要把 `--no-llm` 当成已完成模型标注。

```sh
python annotate --graph pair --corpus results/location_pairs.jsonl --print-prompt
python annotate --graph all_in_one --models model-a model-b --limit 10
python annotate --engine all-in-one --limit 1
python -m annotate.all_in_one --print-prompt
```

目录入口与 `python -m annotate` 等价；旧 `--engine staged`（默认 pair）、`--engine all-in-one` 和
`python -m annotate.all_in_one` 继续可用，后两者只转发到 `all_in_one` 图，不保留另一套执行器。
显式相冲突的 engine/graph 组合及重复的不同选择会报错；重复同一选择允许。
未指定 `--limit` 时，pair/synonym 保留 100，all_in_one 保留 1。
结构化补发默认 pair/synonym 允许、all_in_one 禁止，可统一用 `--allow-retry` / `--no-allow-retry` 覆盖；
传输重试仍按模型配置。`--feature-deep` 不适用于 all_in_one，会报错；旧 `--all-candidates`、`--no-syntax`、
`--batch-inference` 在正式标注中仅保留兼容并明确提示，不会展开或准备句法。

共同设置还包括 `--model` / `--models`、`--workers` / `--model-workers`、`--offset`、
`--max-requests` / `--batch-max-requests`、`--batch auto|native|off`（默认 auto）、
`--batch-cache` / `--batch-resume`、`--no-pairs-csv` 和 `--overwrite`。
三路均支持多模型加权投票；代表结果保留一个模型的完整记录，不跨模型拼接特征/图式，平票或缺票需复核。
JSONL 与默认同名 CSV 共用收尾排序（`line_no`、`candidate.index`），summary 记录汇总、日志路径和实际重试策略。
默认拒绝覆盖已有产物；即使 `--overwrite` 也不能覆盖输入或让结果/CSV/summary/模型原始结果/Batch sidecar 路径冲突（含链接别名）。

## 1.3 结果与旧数据边界

正式结果的 `schema_version` 为 `"1.3"`，`annotation_mode` 为 `pair`、`synonym` 或 `all_in_one`。
三路共享 `id`、`sentence_id`、`content_hash`、`candidate_key`、来源和 preparation 证据；同一输入可按稳定键对账。
四段 `id` 是本次输入内的位置编号，跨文件溯源应看 `sentence_id` / `candidate_key`，不要只比较编号。

准入按任务判断：pair/all_in_one 仍要求特征、图式与判定证据；synonym 的特征/图式准入为 `not_applicable`，
不伪造抽取结果，CSV 对应图式/查表列为空。已确认两侧筛选并取得真实布尔独立判断时，正例和负例均可准入；
未知、未调用模型、空提示语、错误或预算拒绝不准入，需人工复核。`status=ok` 不等于数据必定准入。

`python -m annotate.migrate_result_ids old.jsonl -o new.jsonl` **只迁移 ID/来源文档编号并重排、导出 CSV**，
保留原 `schema_version`（包括缺失或已是 1.3 的值）。它不会把旧 1.1/1.2 或缺字段的 synonym 自动升级为 1.3，
不会补造 `annotation_mode` / admission 或重新判断；旧结果应只读保留，需新契约时另行标注/验证完整证据。

调用账中 `requests` 是同步逻辑执行数，`actual_requests` 是回调观测的同步模型开始数；
原生 Batch 指标单独由 BatchSession 记录。all_in_one 的旧 `pair_llm_calls` 只是步骤比较诊断，不能当真实请求数。

## 离线验证与发布范围

本地测试使用 mock 模型验证三路 CLI、JSONL/CSV/summary/log、排序、身份和路径保护，
并固定重命名前的提示词/渲染消息/缓存摘要；不要为通过测试重新生成这些基线。
`tests/` 和私有 `.docs/superpowers/` 仍是被忽略的本地开发资料，不随最小运行版发布，不修改忽略策略。
测试临时目录可用 `ANNOTATION_TEST_TEMP` 指向**已存在**的专用临时根；Windows 优先已有的
`%LOCALAPPDATA%/Temp/opencode`，其他情况回退到 `TemporaryDirectory` 的平台默认目录，不创建生产路径。

CLI help、离线预览、mock Batch 及本地回归均不能证明真实供应商的 Batch、密钥、计费或标注质量。
部署后需由维护者在备份数据、确认输入输出路径后另行授权小批量在线验证；本次未访问真实模型接口。
