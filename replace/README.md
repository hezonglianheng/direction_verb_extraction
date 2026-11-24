# replace：替换目标词并生成句对

本模块把提取出的原句转换为“**原句 A＋替换句 B**”，检查两侧是否通顺、是否保留有效空间表达，供后续语义标注使用。

[返回项目说明](../README.md) · [上一步：extract](../extract/README.md) · [下一步：annotate](../annotate/README.md)

## 主要功能

- 从原句中定位目标词，复用已有检查，必要时补筛。
- 按替换规则生成候选句，**每条候选只替换一个 token 位置**。
- 调用大语言模型（LLM）检查替换句，保存两侧判定与稳定的句对标识。
- 支持限额、随机、按词比例以及参考文件对齐抽样。

这里的替换是**规则驱动的机械替换**，不是让模型自由改写整句话；两侧检查也不负责判定它们是否同义，同义关系由 `annotate` 判断。

## 实现流程

```text
extract 原句 JSONL
  → 选择可用原句与目标词位置
  → 复用原句判定，缺失时补筛
  → 按替换规则展开候选
  → 检查替换句
  → 句对 JSONL + 替换日志
```

1. **读取与抽样**：分批读取原句。随机和比例抽样先计数再选择，避免为抽样把整个大文件装入内存。
2. **确认原句可用**：按目标词位置读取已有筛选证据；没有完整判定时，复用 `extract` 的筛选器补判。未通过的目标词不生成候选。
3. **展开替换**：按 `alter_rules.json5` 的 `words → replacements` 映射生成句对。一个原句的多个词位分别展开，不组合修改多个位置，也不重新运行句法分析。
4. **检查 B 侧并写出**：沿用 A 侧证据，对 B 侧重新做语法与空间义检查；逐条写入候选句对，包括检查失败或未判定的候选，便于审查。

`PairPreparer` 负责上述准备过程；实际替换复用 `annotate.spatial_agents.AlterAgent` 的本地规则逻辑，双侧检查复用 `extract.sentence_screening.SentenceScreener`。机械替换本身不调用模型。

## 输入与输出

**输入**：[`extract`](../extract/README.md) 输出的原句 JSONL。`sentence` 应为 token 列表，并带有可定位目标词的规则或筛选证据；不能直接传入普通文章或已生成的句对。

**输出**：一行一个候选句对，契约版本为 `pair_schema_version="1.0"`。

| 字段 | 含义 |
| --- | --- |
| `candidate_key` | 用于跨步骤对齐句对的稳定标识 |
| `corpus_type`、`candidate` | 语料类型、替换词、词位及候选编号 |
| `focus_word`、`focus_word_b` | A/B 两侧关注的目标词 |
| `pair.a`、`pair.b` | 两侧文本、token 切片和空间角色 |
| `pair_screening.a`、`pair_screening.b` | 两侧检查证据 |
| `pair_screening.status` | 句对的汇总检查状态 |

汇总状态分为 `true`、`false`、`error`、`limit`、`no_llm`。**输出文件不是纯合格集**；下游还会复核两侧判定、词位、切片及单位置替换关系，不能只看行级 `status`。

## 使用示例

先完成[环境与模型配置](../README.md#环境安装与运行准备)。以下命令在**仓库根目录、已激活的虚拟环境**中执行；`my-model` 是已配置的模型键，`results/` 下的输入需先由上游生成，输出由本模块创建。

```bash
# 从方位词原句生成句对
python -m replace --corpus results/location_corpus.jsonl --corpus-type localization --model my-model --out-file results/location_pairs.jsonl

# 随机选择最多 100 条可用原句，写入另一份输出
python -m replace --corpus results/location_corpus.jsonl --corpus-type localization --model my-model --limit 100 --limit-mode random --seed 0 --out-file results/location_pairs_sample.jsonl

# 查看完整参数
python -m replace --help
```

### 常用参数

| 参数 | 用途 |
| --- | --- |
| `--corpus` | 原句 JSONL，必填 |
| `--corpus-type` | 默认方位词；趋向动词需显式指定 `direction` |
| `--alter-rules` | 覆盖该类型默认的机械替换规则 |
| `--model` | 原句补筛与替换句检查使用的模型键 |
| `--workers`、`--batch-size` | 检查请求并发数、每请求的判定项数 |
| `--chunk-size` | 每批读取的原句数，默认 100 |
| `--limit`、`--limit-mode`、`--seed` | 限额、抽样模式和随机种子 |
| `--reference-file`、`--reference-level` | 按参考文件对齐，选择记录级或目标词判定级 |
| `--out-file`、`--log-dir` | 句对输出和替换日志位置 |
| `--no-llm`、`--overwrite` | 跳过模型检查、允许覆盖已有输出 |

默认输出为项目根目录的 `results/location_pairs.jsonl` 或 `results/direction_pairs.jsonl`，日志默认写入 `log/`。

### 抽样模式怎么选

- **`sequential`**：按源文件顺序选择，适合小规模试跑。
- **`random`**：随机选择可用原句；固定输入与 `--seed` 可复现。
- **`proportional`**：按主关注词比例分层选择原句，不限制该句其他可用词位的展开。
- **`aligned`**：提供 `--reference-file` 后自动启用；按参考文件对回源语料，不能同时指定 `--seed` 或其他抽样模式。未匹配或状态不可用的参考项会跳过并汇总，一对多歧义仍报错。

`--limit` 必须大于 0，省略时处理全部可用输入；一般按**送入替换的原句数**计数，不是最终句对数。唯一例外是 `aligned` 配合 `--reference-level verdict`，此时按展开的目标词判定数计数。

## 代码导航

| 文件 | 负责什么 |
| --- | --- |
| [`cli.py`](cli.py) | 参数解析、类型别名和抽样组合校验 |
| [`sentence_replace.py`](sentence_replace.py) | `PairPreparer`、候选展开、双侧检查和分批写出 |
| [`corpus_pairs.py`](corpus_pairs.py) | 1.0 句对契约、汇总状态与合格句对校验 |
| [`limit_sampler.py`](limit_sampler.py) | 顺序、随机、比例与参考对齐抽样 |
| [`replacement_log.py`](replacement_log.py) | 替换数量、筛选统计和对齐遗漏日志 |

共享实现见 [`extract/sentence_screening.py`](../extract/sentence_screening.py) 与 [`annotate/spatial_agents.py`](../annotate/spatial_agents.py)；规则位于 [`config_files/localization/alter_rules.json5`](../config_files/localization/alter_rules.json5) 和 [`config_files/direction/alter_rules.json5`](../config_files/direction/alter_rules.json5)。

## 注意事项

- 趋向动词默认替换规则目前为空，需要先填写规则或传入 `--alter-rules`；“没有候选”不等于完成了有效语料构建。
- `--no-llm` 不会把未检查的数据变成合格句对；原句缺少可复用的合格判定时，甚至可能无法生成候选。
- 默认拒绝覆盖已有结果；即使使用 `--overwrite`，也不允许覆盖输入或参考文件。
- 结果可用 [replace 离线工具](../analyze_tools/replace/README.md) 导出 CSV、比较差异；合格句对交给 [annotate](../annotate/README.md) 继续处理。
