# annotate 标注结果分析

专门分析正式图 `pair` / `synonym` / `all_in_one` 的 annotate 结果 JSONL（`schema_version="1.3"`）。
支持两种产物，可任意混比：

| 产物 | 来源 | 形态 |
| --- | --- | --- |
| 单模型结果 | `python -m annotate --model <配置名>` | 每条记录直接写最终标注，**没有** `voting` 块 |
| 投票最终结果 | `python -m annotate --models <配置名...>` | 每条记录带 `voting` 块，`model_name="voting"` |
| 各模型原始结果 | 同上，另存于 `<最终文件主名>.models/00_<模型>.jsonl` | 与单模型结果同形，可直接参与比较 |

入口只读输入，使用 Python 标准库；`--help` 不加载 LTP/torch，也不读模型密钥。
可直接执行脚本，也可用 `python -m analyze_tools.annotate.compare_annotate_results`。

## 对比：compare_annotate_results.py

```bash
# 两次单模型结果
python analyze_tools/annotate/compare_annotate_results.py run_a.jsonl run_b.jsonl

# 投票最终结果 vs 其中一个模型的原始结果
python analyze_tools/annotate/compare_annotate_results.py \
    results/spatial_pairs/vote.jsonl \
    results/spatial_pairs/vote.models/00_model-a.jsonl

# 三份结果 → 三份两两明细 + 一份总体报告
python analyze_tools/annotate/compare_annotate_results.py run_a.jsonl run_b.jsonl run_c.jsonl \
    --output-dir analyze_results/annotate_check
```

### 对齐口径

唯一键是 **`candidate_key`**（`句子ID|关注词位置|替换词`）。
`id`、`line_no`、JSONL 行号和记录出现顺序都**不参与**对齐：它们会随 `--offset`、抽样和批次变化。

- 没有候选的记录（`bad_record` / `alter_missed`）合法地没有 key，单独计入输入表的「无候选」，不强行配对。
- 同一文件内 `candidate_key` 重复会直接报错，不按出现顺序取第一条。
- 多出/缺失的 key 记为仅左/仅右，不进入共同项分母。

### 比较口径

主比较字段是**最终结论标签**：

- 单模型结果按 `annotate.voting.ballot_label` 的同一套口径还原：
  `same` / `different` / `not_spatial` / `not_fluent_a|b|ab` / 无结论；
- 投票最终结果取 **`voting.winner_label`**。`accepted` 时为获胜标签；
  `tie` / `no_valid_votes` 记为 `（无结论）`。

**不要把投票记录的 `judgement.same_meaning` 当作投票共识**：它来自代表模型，
采纳 `not_spatial` / `not_fluent_*` 时可能是 `null`，即使投票本身已有明确结论。
该字段仍单独比较，用于观察代表模型的原始判定。同理，投票记录的 `schema`、`features`
也是代表模型继承的字段，不是投票共识。

只给一个配置名时 `--models A` 与 `--model A` 等价，结果同样没有 `voting` 块，
本工具按单模型结果处理。

统计分两个分母，避免把「没判」混成「判了不同」：

- **结论一致率（含未判定）**：共同 key 中最终标签相同的比例；无结论对无结论计为相同状态。
- **有效结论一致率**：只在双方最终标签都不是「无结论」时计算。

其他单列统计（均不进入上述一致率）：

- `judgement.same_meaning` 的三态矩阵（true / false / 未判定）；**空值不当作 false**。
- 最终 `status` 分布与转移。
- 两侧 `schema.a.name` / `schema.b.name`：只在双方都有名字时统计；`synonym` 不写图式，
  记为「不适用」，不会把不适用算成不同；单侧有名字单独计数。
- `admission.is_admitted`、`admission.review_required`、复核原因。
- `voting.status` / `winner_label` / `winner_model` / `incomplete` / `missing_models` / `invalid_models`。
  单模型结果没有投票，记为「不适用」，**不当作缺票**。
- 句子、关注词、来源文件等上下文变化；判定理由只展示，不按措辞计算一致性。

### 输出

默认目录：`analyze_results/compare_annotate_results/<YYYYmmdd_HHMMSS>/`。
显式 `--output-dir` 也必须是尚不存在的目录。

```text
<输出目录>/
  summary.md
  01_run_a__vs__02_run_b.csv
  ...
```

- 两两 CSV：UTF-8 BOM；共同项按左文件顺序，随后仅左、仅右。
  列含 `candidate_key`、变化字段、左右最终结论/同义判定/状态/图式/准入/投票信息、句子上下文和理由。
  布尔单元格为小写 `true` / `false` / 空串，空串不表示 false。
  `比较结果=相同` **只指最终结论标签相同**；状态、图式、准入等是否变化看「变化字段」和各列。
- `summary.md`：输入清单（类型、模型、记录数、无候选、结论与投票状态分布、缺票/无效票）、
  两两汇总、一致性矩阵、逐对结论转移、同义判定矩阵、状态转移、图式、准入与复核、投票状态转移。
- 使用同目录暂存区，独占创建目标目录后移动文件，`summary.md` 最后作为完成标志。
  可捕获的写入错误会清理本次输出；强制终止可能留下无 `summary.md` 的不完整目录。

| 参数 | 说明 |
| --- | --- |
| `INPUT ...` | 至少两份 annotate 结果 JSONL；投票汇总与各模型原始结果可混比 |
| `-o, --output-dir DIR` | 尚不存在的输出目录 |
| `--max-files N` | 最多加载的文件数，默认 20；0 不限制 |
| `--progress-every N` | 每 N 条记录显示读取进度，默认 50000；0 关闭 |

退出码：0 表示所有文件对均有共同 key；1 表示至少一对无共同 key（报告仍生成）；
2 表示输入/输出错误。判定不同不算执行错误。
常驻内存与所有输入记录投影之和成正比，报告大小和比较次数为 `O(N²)`。

## 输入保护

以下情况会报错，并带文件名与行号：

- `pair_schema_version` 的 replace 句对、顶层 `screening` 的 extract 原句；
- 诊断图 `single` / `alter_check` 的结果（没有正式候选契约）；
- 缺少 `annotation_mode` / `schema_version`，或 `schema_version` 不是 `1.3`；
- 有 `candidate` 却缺 `candidate_key`，或同一文件内 key 重复；
- `judgement.same_meaning` 不是真正的 true/false/null；
- `voting` 块状态、`voting_label` 或 `status=discarded` 互相矛盾；
- 损坏 JSON、非对象行、混合投票/非投票记录、混合 `annotation_mode`。

支持直接传 JSONL；**不接受** annotate 配套 `.csv`（只是拍平视图，缺少 voting 与完整记录）
和 `.summary.json`（运行汇总，不是逐条结果）。各模型原始结果请传
`<最终文件主名>.models/*.jsonl` 里的具体文件，不要传目录或最终投票文件代替。
