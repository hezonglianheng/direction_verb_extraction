# analyze_tools

本目录存放结果导出与对比工具。以下命令均从仓库根目录执行。

| 文件 | 用途 |
| --- | --- |
| `compare_screening_verdicts.py` | 对比 2 个或更多筛选结果，输出两两明细 CSV 和总体一致性矩阵 Markdown |
| `export_screening_verdicts_csv.py` | 从筛选结果 JSONL 导出逐方位词 true/false 判定 CSV |

---

## compare_screening_verdicts.py

对比多个方位词筛选结果。脚本会把每个输入文件加载一次，对所有文件对做两两比较：

- 每一对文件生成一个逐项明细 **CSV**；
- 另生成一份总体报告 **`summary.md`**，包含每个文件的统计、两两汇总表和一致性矩阵。

### 输入

支持两种同构结果文件，同一次运行中必须全部为同一种格式：

- `*.screening_verdicts.csv`：每行一个逐方位词 true/false 判定；
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
比较结果, 左侧行号, 右侧行号, 源文件, 原句, 方位词, 方位词下标,
左目标物, 左参照物, 左动词, 右目标物, 右参照物, 右动词,
左判定, 右判定, 三元组是否变化, 左理由, 右理由
```

行顺序为：共同相同/不同项（按左侧原始顺序）→ 仅左侧 → 仅右侧。CSV 使用 UTF-8 BOM，便于 Excel 直接打开。

#### summary.md

包含：

1. **输入文件表**：编号、路径、格式、判定数、true/false、完整三元组数、重复 base 组数；
2. **两两对比汇总表**：共同、相同、不同、一致率、仅左、仅右、三元组变化和明细 CSV 链接；
3. **一致性矩阵**；
4. **重复 base 歧义警告**；
5. **口径说明**。

一致性矩阵的单元格格式为：

```text
一致率（相同/共同）
```

- 只统计两侧都存在的共同判定项，仅单侧项不进入分母；
- 对角线表示“不与自己比较”，写 `—`；
- 没有共同判定项时写 `N/A (0/0)`，不会伪装成 0% 或 100%。

示例：

```text
| 文件 | 01_20261029 | 02_20261031 | 03_20261103 |
| --- | ---: | ---: | ---: |
| 01_20261029 | — | 92.82% (7649/8241) | 89.67% (7182/8009) |
| 02_20261031 | 92.82% (7649/8241) | — | 91.03% (88544/97269) |
| 03_20261103 | 89.67% (7182/8009) | 91.03% (88544/97269) | — |
```

### 对齐口径

```text
base = (源文件, 原句, 方位词, 方位词下标)
key  = base + (同一 base 在该文件中按原始顺序出现的序号,)
```

- `JSONL行号` 不参与对齐：不同批次记录增删会让行号整体平移；
- 三元组 `(目标物, 参照物, 动词)` 不参与 key，角色为空或 grounding 变化不会拆散同一条判定；
- 同一 base 重复出现时按原始顺序配对；一侧多出/缺少时存在配对歧义，`summary.md` 会给出警告和相同判断数范围。

### 用法

```bash
# 自动发现全部筛选 CSV，输出到默认时间戳目录
.venv/bin/python analyze_tools/compare_screening_verdicts.py

# 只比较最新两份
.venv/bin/python analyze_tools/compare_screening_verdicts.py --latest 2

# 显式比较三个 CSV
.venv/bin/python analyze_tools/compare_screening_verdicts.py \
  results/extract_result/location_corpus_with_locative_split_20261029.screening_verdicts.csv \
  results/extract_result/location_corpus_with_locative_split_20261031.screening_verdicts.csv \
  results/extract_result/location_corpus_with_locative_split_20261103.screening_verdicts.csv

# 比较 JSONL
.venv/bin/python analyze_tools/compare_screening_verdicts.py \
  results/extract_result/location_corpus_with_locative_split_20261031.jsonl \
  results/extract_result/location_corpus_with_locative_split_20261103.jsonl

# 指定输出目录（必须尚不存在）
.venv/bin/python analyze_tools/compare_screening_verdicts.py \
  --output-dir /tmp/verdict_comparison
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
| `0` | 全部文件对都至少有一个共同判定项 |
| `1` | 至少一对没有共同判定项；其余 pair 和输出文件仍然有效，矩阵对应单元格为 `N/A (0/0)` |
| `2` | 输入不足、重复输入、格式混合、文件损坏/缺列、输出目录冲突或写入失败 |

### 资源与限制

- 每个输入文件只加载一次；常驻内存约为所有输入判定投影之和。每个 pair 比较时还会临时复制左右两边的行对象并建立 key 索引，峰值约为“两侧投影 + 索引”的 2 倍左右，pair 结束后释放。
- 输出磁盘和比较时间为 `O(N²)`：N 个文件生成 `N(N-1)/2` 个 pair CSV。
- 全量明细 CSV 可能很大（三份真实结果约 166MB），但不做静默截断。
- 仅比较 `模型判定` 的 true/false；理由、`grammatical`、`spatial_coherent` 等字段不在比较范围内。
- 重复 base 的顺序配对歧义无法完全消除，需结合 `summary.md` 的警告解释一致率。

---

## export_screening_verdicts_csv.py

把筛选结果 JSONL 中的逐词 true/false 判定导出为 CSV。

- 模型判定在 `screening.verdicts[]` 中；`extractable` 为 `True`/`False` 才导出；
- `extractable=None` 表示未送模型（limit/error/no_llm），跳过；
- 目标物、参照物、动词来自 `localization_grounding.candidates[]`，按 `verdict.index == candidate.index` 对应；
- 一行对应一个方位词判定，同一句可能有多行。

### 输出列

```text
JSONL行号, 原句, 源文件, 方位词, 方位词下标, 目标物, 参照物, 动词,
模型判定, 判定理由, 判定来源, 整句筛选状态
```

输出为 UTF-8 BOM CSV，默认写到输入文件旁：

```text
<输入名>.screening_verdicts.csv
```

### 用法

```bash
# 不传输入：自动选 results/extract_result 下最新的 JSONL
.venv/bin/python analyze_tools/export_screening_verdicts_csv.py

# 指定输入
.venv/bin/python analyze_tools/export_screening_verdicts_csv.py \
  results/extract_result/location_corpus_with_locative_split_20261031.jsonl

# 指定输出路径；目标已存在时必须显式加 --overwrite
.venv/bin/python analyze_tools/export_screening_verdicts_csv.py \
  results/extract_result/location_corpus_with_locative_split_20261031.jsonl \
  --out /tmp/verdicts.csv --overwrite
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

相关测试仍放在 `tests/`（该目录被 `.gitignore` 忽略，仅本地使用）：

```bash
.venv/bin/python tests/test_compare_screening_verdicts.py
.venv/bin/python tests/test_export_screening_verdicts_csv.py
```

测试通过绝对路径导入 `analyze_tools/` 下的脚本。
