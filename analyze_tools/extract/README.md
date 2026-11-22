# extract 结果分析

本目录的两个脚本只用于 **extract 原句筛选结果**，由原 `analyze_tools/` 下的脚本迁移而来。
以下命令均从仓库根目录执行。

**不要用于 replace 句对。** 检测到 `pair_schema_version` 或 `pair_screening` 的 JSONL 会明确报错；
句对请使用 [`../replace/`](../replace/README.md) 下的独立工具。旧版工具曾从句对中导出的逐词 CSV 已丢失替换侧信息，不能补救为句对 CSV，必须从原 replace JSONL 重新导出。

迁移后仍保留原默认发现路径 `results/extract_result` 与原文件名规则，它们可能选到旧数据，
**建议一律显式指定当前输入路径**。本次没有改变 extract 趋向动词的 grounding/角色完整性口径；以下说明针对方位词。

| 文件 | 用途 |
| --- | --- |
| `compare_extract_verdicts.py` | 对比 2 个或更多 extract 筛选结果，输出原始/严格/句级三层一致率、矩阵和明细 CSV |
| `export_extract_verdicts_csv.py` | 从 extract 结果 JSONL 导出逐方位词判定及可复算严格状态的 CSV |
| `verdict_status.py` | 纯标准库的逐词/整句状态镜像；与生产 `sentence_screening` 有 parity 测试 |

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
比较结果, 左侧行号, 右侧行号, 源文件, 原句, 方位词, 方位词字符偏移, 重复序号,
左方位词下标, 右方位词下标,
左目标物, 左参照物, 左动词, 右目标物, 右参照物, 右动词,
左判定, 右判定, 左严格状态, 右严格状态, 严格比较结果,
左extractable, 左grammatical, 左spatial_coherent,
右extractable, 右grammatical, 右spatial_coherent,
左逐词来源, 右逐词来源, 左整句状态, 右整句状态,
三元组是否变化, 左理由, 右理由
```

`比较结果` 只反映原始 extractable 三态；`严格比较结果` 反映严格五态。
`左/右extractable`、`左/右grammatical`、`左/右spatial_coherent` 写小写
true/false/空串；JSONL 中的非布尔脏值会写成 `非布尔:<repr>`，不会按 Python
真值伪装成 true/false。

行顺序为：共同相同/不同项（按左侧原始顺序）→ 仅左侧 → 仅右侧。CSV 使用 UTF-8 BOM，便于 Excel 直接打开。

#### summary.md

包含：

1. **输入文件与逐词状态分布**：编号、标签、路径、格式、逐词判定数、完整三元组数、
   同词位重复组数，以及原始 true/false/unknown 与严格 true/false/error/limit/no_llm、
   有效判定与覆盖率；
2. **句级状态与下游准入表**（仅 JSONL）：句数、整句 true/false/error/limit/no_llm、
   准入句数与准入率；
3. **两两对比：词级**：共同、并集、原始一致率、严格一致率、有效共同/有效相同/有效一致率、
   仅左、仅右、三元组变化和明细 CSV 链接；
4. **两两对比：句级**：共同句、句并集、句状态一致率、双方准入、仅左/仅右准入、
   准入重合率；
5. **逐词严格状态转移**（非零）：每一种 `左状态 → 右状态` 的数量及占共同项比例；
6. **五组一致性矩阵**：原始 extractable、严格逐词状态、有效判定、句级状态、下游准入句集；
7. **原始 extractable 重复 base 歧义警告**：只量化原始三态的相同数范围；
   严格五态和句级使用同一套序号配对，存在同样歧义，但本表不量化；
8. **口径说明**。

### 三层口径

1. **原始 extractable 三态**：只比较 `extractable` 的 true/false/unknown。这是旧报告
   的“一致率”，保留用于对照，但 `extractable=true` 不代表下游可用。
2. **严格逐词五态**：按生产 `verdict_status` 口径分成
   `true / false / error / limit / no_llm`。`error` 是调用/输出异常或字段不完整，
   `limit` 是额度未送，`no_llm` 是未启用 LLM；三者分别独立，绝不互算相同。
   只有 `true` 和 `false` 是有效判定；严格 true 要求
   `extractable`、`grammatical`、`spatial_coherent` 全为 true，且来源不是
   error/limit/no_llm。
3. **整句状态与下游准入**（仅 JSONL）：按 `screening_status` 统计整句五态，并按
   “true 或 no_llm”比较下游准入句集。CSV 会漏掉 `verdicts=[]` 的 no_llm 句，
   所以 CSV 的句级指标记 `N/A（需 JSONL）`，不用词行数代替句数。

### 一致率与分母

- **原始一致率**：共同项中三态相同数 / 两侧 key 并集 `共同 + 仅左 + 仅右`；
  `unknown→unknown` 记为原始状态相同。
- **严格一致率**：共同项中五态相同数 / 同一并集；`error→error`、`limit→limit`、
  `no_llm→no_llm` 只表示状态相同，不代表已得到有效判定。
- **有效判定一致率**：只在两侧都是严格 true/false 的共同项内计算
  `相同数 / 有效共同项`；error/limit/no_llm 不进入分母。另报两侧有效判定覆盖率，
  避免只看一个高一致率。
- **句级状态一致率**：共同句键中整句状态相同数 / 句键并集。
- **准入重合率**：两侧都准入的句数 / 至少一侧准入的句数；另列仅左/仅右准入。
- 对角线表示“不与自己比较”，写 `—`；分母为 0 时写 `N/A (0/0)`，
  严格列不可用时写 `N/A（缺列，请重新导出）`，句级不可用时写 `N/A（需 JSONL）`。

### 旧版 CSV

旧版 CSV 没有 `extractable`、`grammatical`、`spatial_coherent`、`逐词判定来源`
四列，无法反推严格状态。此时仍输出原始 extractable 一致率；严格指标记
`N/A（缺列，请重新导出）`，不把缺列伪装成生产 `error`。只出现其中一部分列属于
导出损坏，直接报错。新版导出脚本会写全这四列及展示用的 `严格状态`、`整句准入`。

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

- 每个输入文件只加载一次；常驻内存约为所有输入判定投影之和（JSONL 另含句级索引）。
  读取器产出的行已经编号，pair 比较走 `keyed=True` 不再复制整份投影；每对结束显式
  `del comparison`，避免下一对构造期间旧对象仍驻留。N 个文件仍会生成
  `N(N-1)/2` 个 pair CSV，输出与时间为 `O(N²)`，但不额外复制 N 份行对象。
- 全量明细 CSV 可能很大（三份真实结果约 166MB），但不做静默截断。
- 全量明细 CSV 可能很大（三份真实结果约 166MB），但不做静默截断。
- 原始指标只比较 `extractable` 的 true/false/unknown；严格指标另外读取
  `grammatical`、`spatial_coherent` 与逐词来源，理由文本不参与一致率。
- 句级指标只对 JSONL 计算；CSV 缺 `verdicts=[]` 的记录，不能反推句级全量。
- 旧版 CSV 四列严格字段全缺时降级为“原始指标可用、严格指标 N/A”；只缺一部分会报错。
- 重复 base 的顺序配对歧义无法完全消除；`summary.md` 的警告只量化原始 extractable
  的相同数范围，严格五态和句级指标存在同样的配对歧义但未量化，解释时不能假定它们无歧义。

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
模型判定, 判定理由, 判定来源, 整句筛选状态,
extractable, grammatical, spatial_coherent, 逐词判定来源, 严格状态, 整句准入
```

字段口径：

- `模型判定` 仍是原始 `extractable` 三态 true/false/unknown；
- `extractable`/`grammatical`/`spatial_coherent` 写小写 true/false，缺失写空串，
  不把 null 伪装成 false；
- `判定来源` 为兼容列，缺失时回退到整句 `screening.source`；
- `逐词判定来源` 只写该 verdict 自己的 `source`，供比较脚本按生产
  `verdict_status` 复算；
- `严格状态` 是展示用的五态 true/false/error/limit/no_llm；比较脚本从四个原始字段
  复算，不直接信任该列；
- `整句准入` 按“整句 true 或 no_llm”写 true/false。

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
python -m unittest discover -s tests -p "test_extract_analysis.py" -v
python -m unittest discover -s tests -p "test_cli_help.py" -v
```

测试使用临时语料，覆盖迁移后的仓库根路径、导出/比较兼容、句对拒绝和输出保护。
