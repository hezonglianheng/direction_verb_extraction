# replace 句对分析

专门分析 `python replace` 输出的 `pair_schema_version="1.0"` 句对。
支持方位词和趋向动词；不读取从原句继承的顶层 `screening`、`localization_grounding` 或 `direction_grounding`。

两个入口均只读输入，使用 Python 标准库；可直接执行脚本，也可用 `python -m analyze_tools.replace.<模块名>`。
**输入必须显式指定**，不根据旧目录或修改时间自动选择结果。

## 导出：export_replace_pairs_csv.py

```bash
python analyze_tools/replace/export_replace_pairs_csv.py results/new_replace_result_ds/replace-20261110.jsonl

python analyze_tools/replace/export_replace_pairs_csv.py results/location_pairs.jsonl --out analyze_results/pairs.csv
```

一行一个句对，保持输入顺序；同一句的不同替换词/位置各自成行，绝不展开继承的全部逐词 verdict。
所有状态都保留，包括改写失败或未判定的句对，不只导出通过项。

默认输出为输入文件旁的 `<输入名>.replace_pairs.csv`，UTF-8 BOM 编码，便于 Excel 打开。

### CSV 列

公共列：

```text
JSONL行号, 句对版本, candidate_key, 源文件, 语料类型, 替换词, 句对状态
```

A 为原句侧，B 为改写句侧；以下每列均有 `A`、`B` 前缀，并列存储：

```text
句子, 切片, 关注词, 关注词下标, 目标物, 参照物, 动词,
判定, 语法正确, 空间自洽, 理由, 判定来源
```

例如 `A句子`、`B句子`、`A判定`、`B判定`、`B理由`。切片写为 JSON 数组，token 下标从 0 开始。
理由中的逗号/换行由标准 CSV 引号保留，不能按文本行数估计 CSV 记录数。

### 字段来源

| CSV 信息 | JSONL 来源 |
| --- | --- |
| 替换词 | `candidate.replacement` |
| 句对状态 | `pair_screening.status` |
| A/B 判定 | `pair_screening.a/b.extractable` |
| A/B 语法正确、空间自洽、理由、来源 | `pair_screening.a/b.grammatical`、`spatial_coherent`、`reason`、`source` |
| A/B 关注词、下标 | `focus_word` / `focus_word_b` 的 `word`、`index` |
| A/B 句子、切片 | `pair.a/b.sentence`、`slice`；句子缺失时可由同侧切片拼接 |
| A/B 三元组 | 同侧关注词的 `grounding`，缺失/null 时回退同侧 `pair.a/b.grounding` |

布尔值只接受真正的 `true`/`false`/`null`。CSV 中 `true`、`false` 为小写文本，`null` 为空串，
**不会写成 false**。缺失角色也为空，不猜测补齐；趋向动词的 verb 为空不导致整条句对被丢弃。
保存的是原始模型字段与原始句对状态，不根据分析器自行重算通过/失败。
CSV 读取支持超过标准库默认 128 KiB 的长字段（上限 `2^31-1` 字符），结束或报错后恢复进程原来的字段上限。

### 参数与退出码

| 参数 | 说明 |
| --- | --- |
| `input` | 必填，replace JSONL |
| `-o, --out PATH` | 可选输出 CSV |
| `--overwrite` | 显式允许覆盖已有输出；不允许与输入是同一文件 |
| `--progress-every N` | 每 N 条句对显示进度，默认 50000，0 关闭 |

成功返回 0；输入/输出错误返回 2。终端显示 A/B true、false、未判定数量及句对状态分布。
流式写入临时文件，完整校验成功后发布；默认拒绝覆盖，后续行损坏也不会破坏已有 CSV。
为检查重复 key，会在内存保留 key 与首次行号，不把整条原始 JSONL 全量加载。

## 对比：compare_replace_pairs.py

```bash
# 直接比较两个 JSONL
python analyze_tools/replace/compare_replace_pairs.py run_a.jsonl run_b.jsonl

# 可混用 JSONL 与上面的导出 CSV
python analyze_tools/replace/compare_replace_pairs.py run_a.jsonl run_b.replace_pairs.csv --output-dir analyze_results/replace_check

# 三份结果会生成三份两两明细和一份总体报告
python analyze_tools/replace/compare_replace_pairs.py run_a.jsonl run_b.jsonl run_c.jsonl
```

每份文件只读取一次，按唯一 **`candidate_key`** 对齐。JSONL 行号不参与对齐；输入重排不会改变比较结果。
多出/缺失的 key 记为仅左/仅右，不会按同一句出现顺序或替换词列表顺序强行配对。

### 比较口径

- **主字段一致率**：共同 key 中 `(A.extractable, B.extractable, pair_screening.status)` 三项全部相同的比例。
  两侧相同的空值也属于字段相同，**这不代表句对合格率，更不表示两次模型均已判定**。
- **A/B 布尔一致率**：分别对 extractable、grammatical、spatial_coherent 计算；只纳入左右文件均为布尔值的项，
  单独列出涉及未判定数量、true→false 和 false→true。无可比较布尔项时为 `N/A (0/0)`。
- 句对状态逐值比较，单列状态分布和状态转移，不把 error/limit/no_llm 简化成 false。
- A/B 三元组文本、句子、切片、标识上下文变化单独统计，不纳入主字段一致率。
  标识上下文包含源文件、语料类型、替换词、两侧关注词及下标；相同 key 的文本/上下文变化需要人工检查。
- 理由和判定来源保留在左右明细中，不按措辞计算一致性；不套用 extract 的三角色必须齐全口径。
- 仅单侧项不进入共同项分母；完全无共同项时为 `N/A (0/0)`，不会伪装成 100%。

### 输出

默认目录：`analyze_results/compare_replace_pairs/<YYYYmmdd_HHMMSS>/`。
显式 `--output-dir` 也必须是尚不存在的目录。

```text
<输出目录>/
  summary.md
  01_run_a__vs__02_run_b.csv
  ...
```

- 两两 CSV：共同项按左文件顺序，随后仅左、仅右；保留两份文件的全部导出字段，加比较结果和逐字段变化标志。
  `左/右` 表示不同结果文件，`A/B` 表示句对的原句/改写侧，例如 `左B判定`、`右B判定`。
  `比较结果=相同` 仅指主三项一致，辅助变化需查看“变化字段”及对应标志。
- `summary.md`：输入数量、两侧 true/false/未判定、状态和检查项分布、主一致性矩阵、逐字段变化/布尔一致率、状态转移、按替换组合的统计。
- 使用同目录暂存区，独占创建目标目录后移动文件，`summary.md` 最后作为完成标志。
  可捕获的写入错误会清理本次输出；强制终止可能留下无 `summary.md` 的不完整目录。

| 参数 | 说明 |
| --- | --- |
| `INPUT ...` | 至少两个 replace JSONL / 句对导出 CSV，可混用格式 |
| `-o, --output-dir DIR` | 尚不存在的输出目录 |
| `--max-files N` | 最多加载的文件数，默认 20；0 不限制 |
| `--progress-every N` | 每 N 条句对显示读取进度，默认 50000；0 关闭 |

退出码：0 表示所有文件对均有共同 key；1 表示至少一对无共同 key，其他报告仍生成；2 表示输入/输出错误。
判定不同不算执行错误。常驻内存与所有输入句对投影之和成正比，报告大小和比较次数为 `O(N²)`。

## 输入保护

缺失或重复 `candidate_key`、extract 原句、未知 schema、非法布尔值、损坏 JSON/CSV、关注词与切片/替换词不一致会报错，
包含文件和行号；CSV 报错使用物理行号。不要把旧工具错误导出的 replace 逐词 CSV 交给此工具，必须从原始句对重新导出。
