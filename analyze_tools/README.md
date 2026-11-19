# analyze_tools

供人工审查的离线分析工具，不在自动合成工作流中，不调用模型，不修改输入结果。
以下命令从仓库根目录执行，使用项目 Python 环境。

## 目录与入口

| 输入结果 | CSV 导出 | 多结果比较 |
| --- | --- | --- |
| extract 原句 | [`extract/export_extract_verdicts_csv.py`](extract/export_extract_verdicts_csv.py) | [`extract/compare_extract_verdicts.py`](extract/compare_extract_verdicts.py) |
| replace 句对 | [`replace/export_replace_pairs_csv.py`](replace/export_replace_pairs_csv.py) | [`replace/compare_replace_pairs.py`](replace/compare_replace_pairs.py) |
| annotate 正式图结果 | —（比较直接读 JSONL） | [`annotate/compare_annotate_results.py`](annotate/compare_annotate_results.py) |

原来的 `export_screening_verdicts_csv.py`、`compare_screening_verdicts.py` 已**保留实现、迁移并改名**至 `extract/`，旧路径不再提供入口，请更新已有命令。

- **extract**：一行一个三态逐词判定（true/false/unknown），按原句/关注词/字符偏移/重复序号对齐；详见 [extract 说明](extract/README.md)。
- **replace**：一行一个句对，A/B 字段并列；只读 `pair_screening`，按唯一 `candidate_key` 对齐；详见 [replace 说明](replace/README.md)。
- **annotate**：比较 `--model` 单模型结果与 `--models` 投票结果（含各模型 `.models/*.jsonl`），按 `candidate_key` 对齐，主比最终结论标签；投票结果取 `voting.winner_label`，不把代表模型继承的 `judgement.same_meaning` 当共识；详见 [annotate 说明](annotate/README.md)。
- 三类工具不能互换。extract 工具会拒绝句对 JSONL；旧脚本错误导出的 replace 逐词 CSV 无法恢复 B 侧信息，须从 replace JSONL 重新导出。
- extract 自动发现仍沿用旧目录规则，可能选到旧数据；replace 与 annotate 不自动发现。**推荐始终显式指定输入文件。**

## 快速使用

```bash
# extract 方位词原句
python analyze_tools/extract/export_extract_verdicts_csv.py results/location_corpus.jsonl

# replace：一行一个句对，默认输出 results/location_pairs.replace_pairs.csv
python analyze_tools/replace/export_replace_pairs_csv.py results/location_pairs.jsonl

# replace：比较两次结果（输出目录必须尚不存在）
python analyze_tools/replace/compare_replace_pairs.py run_a.jsonl run_b.jsonl --output-dir analyze_results/replace_comparison

# annotate：比较单模型 / 投票结果（可与各模型 .models/*.jsonl 混比）
python analyze_tools/annotate/compare_annotate_results.py run_a.jsonl run_b.jsonl --output-dir analyze_results/annotate_comparison
```

## 测试

```bash
python -m unittest discover -s tests -p "test_compare_screening_verdicts.py" -v
python -m unittest discover -s tests -p "test_export_screening_verdicts_csv.py" -v
python -m unittest tests.test_annotate_analysis -v
```

专项测试放在 `tests/`，只用临时文件和离线构造语料，不依赖实际大结果文件或模型配置。
该目录沿用仓库现有 `.gitignore` 规则，默认不入库。
