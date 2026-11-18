# extract 结果分析

本目录的两个脚本只用于 **extract 原句筛选结果**，由原 `analyze_tools/` 下的脚本迁移而来。
以下命令均从仓库根目录执行。

**不要用于 replace 句对。** 检测到 `pair_schema_version` 或 `pair_screening` 的 JSONL 会明确报错；
句对请使用 [`../replace/`](../replace/README.md) 下的独立工具。旧版工具曾从句对中导出的逐词 CSV 已丢失替换侧信息，不能补救为句对 CSV，必须从原 replace JSONL 重新导出。

迁移后仍保留原默认发现路径 `results/extract_result` 与原文件名规则，它们可能选到旧数据，
**建议一律显式指定当前输入路径**。本次没有改变 extract 趋向动词的 grounding/角色完整性口径；以下说明针对方位词。

| 文件 | 用途 |
| --- | --- |
| `compare_extract_verdicts.py` | 对比 2 个或更多 extract 筛选结果，输出两两明细 CSV 和总体一致性矩阵 Markdown |
| `export_extract_verdicts_csv.py` | 从 extract 结果 JSONL 导出逐方位词判定（true/false/unknown）CSV |

---

## compare_extract_verdicts.py

对比多个方位词筛选结果。脚本会把每个输入文件加载一次，对所有文件对做两两比较：

- 每一对文件生成一个逐项明细 **CSV**；
- 另生成一份总体报告 **`summary.md`**，包含每个文件的统计、两两汇总表和一致性矩阵。

### 输入

支持两种同构结果文件，同一次运行中必须全部为同一种格式：

- `*.screening_verdicts.csv`：每行一个逐方位词判定，必须包含 `方位词字符偏移` 列；
- `*.jsonl`：直接从 JSONL 的 `screening.verdicts[]` 和 `localization_grounding.candidates[]` 投影。

至少需要两个输入；重复文件（包括符号链接指向同一文件）会被拒绝。

不传位置参数时，自动发现 `results/extract_result` 下全部
`location_corpus_with_locative_split_*.screening_verdicts.csv`，按文件名中的日期、再按文件名排序。

### 输出

默认输出到：

```text
analyze_results/compare_verdicts/<YYYYmmdd_HHMMSS>/
```

也可以用 `--output-dir` 指定一个**尚不存在**的目录；已存在会报错，避免覆盖旧结果。

发布优先使用 Linux `renameat2(RENAME_NOREPLACE)` 做一次原子目录重命名，目标已存在（包括空目录和悬空符号链接）会失败。平台或文件系统不支持时，降级为“先占位创建目录、再移动文件、最后移动 `summary.md`”；降级路径若中途被强制终止，可能留下带 `summary.md` 缺失的不完整目录，需要人工清理。

输出示例（三个输入文件）：

```text
<run-dir>/
├── summary.md
├── 01_20261029__vs__02_20261031.csv
├── 01_20261029__vs__03_20261103.csv
└── 02_20261031__vs__03_20261103.csv
```

文件名中的 `01_`、`02_` 是本次运行中的输入序号，后面的标签由输入文件名生成。标签只用于本次运行，不承诺跨运行稳定。

#### 两两明细 CSV

每个 pair CSV 的列固定为：

```text
比较结果, 左侧行号, 右侧行号, 源文件, 原句, 方位词,
方位词字符偏移, 左方位词下标, 右方位词下标,
左目标物, 左参照物, 左动词, 右目标物, 右参照物, 右动词,
左判定, 右判定, 三元组是否变化, 左理由, 右理由
```

行顺序为：共同相同/不同项（按左侧原始顺序）→ 仅左侧 → 仅右侧。CSV 使用 UTF-8 BOM，便于 Excel 直接打开。

#### summary.md

包含：

1. **输入文件表**：编号、路径、格式、判定数、true/false/unknown、完整三元组数、重复 base 组数；
2. **两两对比汇总表**：共同、并集、相同、不同、一致率、仅左、仅右、三元组变化和明细 CSV 链接；
3. **一致性矩阵**；
4. **重复 base 歧义警告**；
5. **口径说明**。

一致性矩阵的单元格格式为：

```text
一致率（相同/并集）
```

- 分母是两侧 key 的并集 `共同 + 仅左 + 仅右`，仅单侧项也进入分母；
- `unknown`（没拿到 true/false 判定）参与共同项和分母统计，不会被当成 `false`；
- 共同项中 `unknown` 对 `unknown` 计为“相同状态”，`unknown` 对 true/false 计为不同；因此这里的一致率是**三态状态一致率**，不只是有效模型判定一致率；
- 对角线表示“不与自己比较”，写 `—`；
- 两侧并集为 0 时写 `N/A (0/0)`，不会伪装成 0% 或 100%。

示例（数值仅示意格式，实际以当次并集分母为准）：

```text
| 文件 | 01_20261029 | 02_20261031 | 03_20261103 |
| --- | ---: | ---: | ---: |
| 01_20261029 | — | 92.82% (7649/8241) | 89.67% (7182/8009) |
| 02_20261031 | 92.82% (7649/8241) | — | 91.03% (88544/97269) |
| 03_20261103 | 89.67% (7182/8009) | 91.03% (88544/97269) | — |
```

### 对齐口径

```text
base = (源文件, 原句, 方位词, 方位词字符偏移)
key  = base + (同一 base 在该文件中按原始顺序出现的序号,)
```

- 字符偏移是方位词在拼接原句中的起始位置，可由 token 列表 `sum(len(token) for token in tokens[:index])` 得到；重新分词不会改变它；
- `JSONL行号` 不参与对齐：不同批次记录增删会让行号整体平移；
- 三元组 `(目标物, 参照物, 动词)` 不参与 key，角色为空或 grounding 变化不会拆散同一条判定；
- 实际存在的 verdict 中 `extractable=None`（典型 limit/error；旧数据可能含逐词 no_llm）作为 `unknown` 参与统计，不跳过、不当成 `false`；真实 `no_llm` 记录 `verdicts=[]`，不产生比较行；
- 同一 base 重复出现时按原始顺序配对；一侧多出/缺少时存在配对歧义，`summary.md` 会给出警告和相同判断数范围。

### 用法

```bash
# 自动发现全部筛选 CSV，输出到默认时间戳目录
python analyze_tools/extract/compare_extract_verdicts.py

# 只比较最新两份
python analyze_tools/extract/compare_extract_verdicts.py --latest 2

# 显式比较三个 CSV
python analyze_tools/extract/compare_extract_verdicts.py \
  results/extract_result/location_corpus_with_locative_split_20261029.screening_verdicts.csv \
  results/extract_result/location_corpus_with_locative_split_20261031.screening_verdicts.csv \
  results/extract_result/location_corpus_with_locative_split_20261103.screening_verdicts.csv

# 比较 JSONL
python analyze_tools/extract/compare_extract_verdicts.py \
  results/extract_result/location_corpus_with_locative_split_20261031.jsonl \
  results/extract_result/location_corpus_with_locative_split_20261103.jsonl

# 指定输出目录（必须尚不存在）
python analyze_tools/extract/compare_extract_verdicts.py \
  --output-dir analyze_results/extract_comparison
```

### 参数

| 参数 | 说明 |
| --- | --- |
| `INPUT ...` | 两个或更多 `.csv` / `.jsonl`；省略时自动发现全部筛选 CSV |
| `-o, --output-dir DIR` | 输出目录；必须尚不存在。省略时写入默认时间戳目录 |
| `--latest N` | 仅无位置参数模式可用，只比较最新 N 份 CSV |
| `--max-files N` | 最多加载多少个文件，默认 20；`0` 表示不限制 |
| `--progress-every N` | 每读取多少行/记录打印一次进度，默认 50000；`0` 关闭 |

### 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 全部文件对的两侧并集都大于 0 |
| `1` | 至少一对两侧都没有判定行（并集为 0）；其余 pair 和输出文件仍然有效，矩阵对应单元格为 `N/A (0/0)` |
| `2` | 输入不足、重复输入、格式混合、文件损坏/缺列、输出目录冲突或写入失败 |

### 资源与限制

- 每个输入文件只加载一次；常驻内存约为所有输入判定投影之和。每个 pair 比较时还会临时复制左右两边的行对象并建立 key 索引，峰值约为“两侧投影 + 索引”的 2 倍左右，pair 结束后释放。
- 输出磁盘和比较时间为 `O(N²)`：N 个文件生成 `N(N-1)/2` 个 pair CSV。
- 全量明细 CSV 可能很大（三份真实结果约 166MB），但不做静默截断。
- 仅比较 `模型判定` 的 true/false/unknown；理由、`grammatical`、`spatial_coherent` 等字段不在比较范围内。
- 重复 base 的顺序配对歧义无法完全消除，需结合 `summary.md` 的警告解释一致率。

---

## export_extract_verdicts_csv.py

把筛选结果 JSONL 中的逐词判定（true/false/unknown）导出为 CSV。

- 模型判定在 `screening.verdicts[]` 中；`True`/`False` 导出为 `true`/`false`；
- 实际存在的 verdict 中，`extractable=None`（典型为 limit/error）导出为 `unknown`，不再跳过；
- 真实 `no_llm` 记录是记录级状态，`screening.verdicts=[]`，只计入读取记录数，不产生 CSV 行，也不会从 grounding 候选补造 `unknown`；
- 空文件或没有任何 verdict 的输入会成功写出仅含表头的 CSV（退出码 0）；
- 目标物、参照物、动词来自 `localization_grounding.candidates[]`，按 `verdict.index == candidate.index` 对应；
- 同时导出 `方位词字符偏移`，供比较脚本稳定对齐；
- 一行对应一个方位词判定，同一句可能有多行。

### 输出列

```text
JSONL行号, 原句, 源文件, 方位词, 方位词下标, 方位词字符偏移,
目标物, 参照物, 动词,
模型判定, 判定理由, 判定来源, 整句筛选状态
```

输出为 UTF-8 BOM CSV，默认写到输入文件旁：

```text
<输入名>.screening_verdicts.csv
```

### 用法

```bash
# 不传输入：自动选 results/extract_result 下最新的 JSONL
python analyze_tools/extract/export_extract_verdicts_csv.py

# 指定输入
python analyze_tools/extract/export_extract_verdicts_csv.py \
  results/new_extract_result_ds/location_corpus_with_locative_split_20261109.jsonl

# 指定输出路径；目标已存在时必须显式加 --overwrite
python analyze_tools/extract/export_extract_verdicts_csv.py \
  results/extract_result/location_corpus_with_locative_split_20261031.jsonl \
  --out analyze_results/extract_verdicts.csv --overwrite
```

### 参数

| 参数 | 说明 |
| --- | --- |
| `input` | 可选，筛选结果 JSONL；省略时自动选最新一份 |
| `-o, --out PATH` | 输出 CSV 路径；默认与输入同目录同名加 `.screening_verdicts.csv` |
| `--overwrite` | 允许覆盖已存在的输出文件 |
| `--progress-every N` | 每读取多少行打印一次进度，默认 50000；`0` 关闭 |

### 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 导出成功 |
| 非零 | 输入不存在、输出已存在但未加 `--overwrite`、JSON 非法、输出与输入同文件等 |

---

## 测试

专项回归统一放在 `tests/`（沿用仓库现有忽略规则，默认不入库）：

```bash
python -m unittest discover -s tests -p test_compare_screening_verdicts.py -v
python -m unittest discover -s tests -p test_export_screening_verdicts_csv.py -v
```

测试使用临时语料，覆盖迁移后的仓库根路径、导出/比较兼容、句对拒绝和输出保护。
