# evaluate 结果分析

专门分析 `python -m evaluate` 写出的逐条明细
（`<模型>_step<N>_<时间>.details.jsonl`），并可与 `annotate` 结果对账。
两个入口都只读输入，使用 Python 标准库；不读模型配置，不调用模型，
不导入 LTP/torch。

| 入口 | 用途 |
| --- | --- |
| [`compare_evaluate_details.py`](compare_evaluate_details.py) | 多份 evaluate details 之间逐条标注一致性 |
| [`compare_evaluate_annotate.py`](compare_evaluate_annotate.py) | evaluate details vs annotate 总结果 / voting 各模型票 |

可直接执行脚本，也可用 `python -m analyze_tools.evaluate.<模块名>`。

## 对齐口径

evaluate details 没有写 `candidate_key`，但保留了 `line_no` 与 `candidate`。
本工具用同一套候选身份对齐：

```text
(line_no, candidate.index, candidate.focus_index, replacement)
```

这与 `annotate.voting.record_identity` 同口径。没有 `candidate` 时退回 `id`
（外部导入的句对只有 id）。具体规则：

- 两份 details 之间：身份 + details 的 `input_file`。JSONL 行号、记录出现顺序
  和单看 `id` 都不参与对齐；`--offset`、抽样、文件行序变化不影响结果。
- details 对 annotate：只用身份 join。annotate 侧身份由 `line_no + candidate`
  重算，不依赖 `id`；只有双方身份相同、句对内容（`sentence_a` / `sentence_b`
  去首尾空白后）一致才算对齐。
- 同一文件内身份重复直接报错，不按出现顺序取第一条。
- details↔details 与 details↔annotate 都核对句对内容；身份相同但句对内容不同记
  「身份内容冲突」，单独计数，不进入共同项分母，也不强行配对。
- 同一份 details 里身份不唯一（例如一次 `--corpus` 目录包含多个语料、行号/候选
  恰好重叠）时，这些行在 details↔annotate 比较中计为「左侧身份不唯一」，
  不参与共同项；details↔details 仍按 `input_file + 身份` 正常比较。
- 没有候选也没有 `id` 的记录无法对齐，单独计入「无身份」，不参与分母。

## 比较口径

evaluate 侧主字段是 `prediction`：

- `same` / `different` / 无结论（`null`，模型调用失败或未形成结论）。

annotate 侧：

- **总结果**：只取 `voting.status=accepted` 的 `voting.winner_label`；
  `tie` / `no_valid_votes` 记无结论，**不退回**代表模型继承的
  `judgement.same_meaning`。
- **各模型结果**：直接取 `voting.ballots[model].label`，不重新投票。
  这是本次投票实际采用的票，不需要另外传 `.models/*.jsonl`。
- `missing_models` 中的模型记**缺票**，不计入共同项分母；有 ballot 但
  `label=null` 记**弃权/无效票**，仍计入共同项并按无结论统计。
- 单模型 annotate 结果（没有 `voting` 块）允许作为「总结果」比较，
  模型列不适用。

每项比较同时给出三种口径：

| 口径 | 分母 | 说明 |
| --- | --- | --- |
| 结论一致率（含未判定） | 共同身份 | null/null 计为相同状态 |
| 有效结论一致率 | 双方标签都非 null | `not_spatial` / `not_fluent_*` 也算有效标签 |
| 同义二分类一致率 | 双方都是 `same`/`different` | evaluate 只预测二分类，主看这一列 |

零分母统一写 `N/A (0/0)`。差异本身不是错误；工具只报告一致性，不改写输入。

## 用法

```bash
# 两份 details 直接比较
python analyze_tools/evaluate/compare_evaluate_details.py \
    results/evaluate/model-a_step0_20261123.details.jsonl \
    results/evaluate/model-b_step0_20261123.details.jsonl

# 三份 details → 三份两两明细 + 一份总体报告
python analyze_tools/evaluate/compare_evaluate_details.py \
    a.details.jsonl b.details.jsonl c.details.jsonl \
    --output-dir analyze_results/evaluate_details_check

# details vs annotate 投票总结果 + 各模型票
python analyze_tools/evaluate/compare_evaluate_annotate.py \
    model-a.details.jsonl model-b.details.jsonl \
    --annotate results/spatial_pairs/vote.jsonl \
    --output-dir analyze_results/evaluate_annotate_check
```

两个工具都支持 `--max-files N`（默认 20，0 不限制）和
`--progress-every N`（默认 50000，0 关闭）。`compare_evaluate_annotate.py`
的 `--max-files` 只限制 details 文件数；annotate 文件固定一份。

## 输出

默认目录分别是：
`analyze_results/compare_evaluate_details/<YYYYmmdd_HHMMSS>/`、
`analyze_results/compare_evaluate_annotate/<YYYYmmdd_HHMMSS>/`。
显式 `--output-dir` 也必须是尚不存在的目录。

```text
<输出目录>/
  summary.md
  01_a__vs__02_b.csv
  01_a__vs__annotate_total.csv
  01_a__vs__model_01_model-a.csv
  ...
```

- CSV 为 UTF-8 BOM；共同身份按左文件顺序，随后仅左、仅右。
  列含对齐身份、变化字段、句对/关注词、左右标签、同义字段、参考标签、
  状态、投票状态、route、物理行号、理由与失败原因。
- `summary.md`：输入清单（route、记录数、prediction 分布、无结论原因）、
  两两/跨层汇总、一致性矩阵、逐项标签转移。
- 使用同目录暂存区，独占创建目标目录后移动文件，`summary.md` 最后作为完成标志。
  可捕获的写入错误会清理本次输出；强制终止可能留下无 `summary.md` 的不完整目录。

## 退出码

- `0`：所有比较都有共同身份；
- `1`：至少一项比较没有共同身份（报告仍生成）；
- `2`：输入/输出错误（格式不符、重复身份、损坏 JSON、输出目录已存在等）。

## 输入保护与限制

以下情况会报错，并带文件名与物理行号：

- 误把 annotate 结果传给 details 位置参数、把 details 传给 `--annotate`；
- replace 句对 / extract 原句 / 诊断图结果、运行汇总 `.summary.json`、CSV、
  目录或重复路径；
- details 缺少 `prediction` / route 字段，`prediction` 不是
  `same`/`different`/null，`status` 与 `prediction` 矛盾；
- 同一文件内身份重复、损坏 JSON、非对象行、同一 details 文件混 route。

需要知道的限制：

- **details 不记录被评测的模型名**，文件身份只能靠 `input_file` 和文件名。
  请保留 `evaluate` 生成的 `<模型>_step<N>_<时间>.details.jsonl` 命名。
- evaluate 只评测参考标签为 `same` / `different` 的样本，因此这里比较的是
  **已进入 evaluate 的样本**，不是整份 annotate 语料的一致率。
- `evaluate.rerun` 对原始损坏行的补齐记录没有 route / 句对，按「无身份」错误行计数，
  不参与对齐，也不会让整份文件失败。
- 跨 annotate 源比较依赖 `line_no + candidate` 身份。若生产侧更换规则表、
  重新编号或对语料重排，身份可能漂移；本工具不会按句子文本或 `id` 猜配。
- 首版只用 `voting.ballots` 取各模型票，不读取旁路 `.models/*.jsonl`；
  若需要核对模型原始记录里的 `same_meaning` / 图式等字段，请用
  `analyze_tools/annotate/compare_annotate_results.py`。
- 理由、参考标签、`same_meaning` 只展示/审计，不进入一致率。

## 测试

```bash
python -m unittest tests.test_evaluate_analysis -v
```

专项测试只用临时文件和离线构造记录，不依赖实际大结果文件或模型配置。
