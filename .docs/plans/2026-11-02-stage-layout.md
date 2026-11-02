# 三阶段入口与流水线实施计划

> 使用 executing-plans 在当前项目根目录逐项实施。

**Goal:** 保留旧入口，提供 `python extract/replace/annotate [args]` 和可选起止阶段的流水线。
**Architecture:** 阶段实现迁入包目录；根目录旧文件仅转发，并保证新旧 import 是同一模块。共享基础设施、配置、模型、提示词、数据目录位置不变。流水线使用当前 Python 解释器顺序启动子进程，失败即停。
**Tech Stack:** Python 标准库 argparse/subprocess/unittest，现有依赖不变。
**Spec:** 用户附件 task.md 与“按推荐方案执行”的确认。

## Global Constraints
- 在当前根目录操作，只提交本地 Git，不推送、不操作服务器。
- 不改变语言学规则、数据 schema、已有覆盖保护及模型调用逻辑。
- `.docs/` 只为本任务文档添加精确白名单，不纳入既有私有说明。

## Review Focus
- 旧 import 的全局状态与补丁必须影响新模块，避免双份缓存。
- 目录执行及 multiprocessing spawn 不得丢失项目根路径。
- 移动后所有默认资源和结果路径仍指向根目录。
- 中途起跑必须读取已有中间文件，不隐式重跑上游。
- 参数错误、缺失输入、输出冲突在发模型请求前发现；子阶段失败不得运行后续。

## Task 1: 保存已有工作
- [x] 检查工作区、依赖关系与敏感文件排除规则。
- [x] 按配置/提示示例、句对流程、样例/审查记录拆成三个本地提交。
- [x] 运行现有 unittest 基线：32/32 通过。

## Task 2: 目录迁移与兼容入口
**Files:** extract/*.py、replace/*.py、annotate/*.py；根目录同名兼容文件；tests/test_stage_layout.py。
**Interfaces:** 三个目录均通过 __main__.py 接收旧阶段参数；annotate 默认分步，`--engine all-in-one` 选择合并标注。
- [x] 写目录入口、资源根路径、模块身份和多进程回归测试，先验证失败。
- [x] 移动阶段文件；修复基于 __file__ 的路径；增加兼容入口。
- [x] 运行新增测试和完整 unittest；提交迁移（2770a92）。

## Task 3: 起止阶段驱动与同步说明
**Files:** pipeline.py、tests/test_pipeline.py、.docs/stage-migration.md。
**Interfaces:** `pipeline.py --from-stage extract --to-stage annotate --source PATH --corpus-type 方位词`；支持 --extracted/--pairs/--output 指定产物，--extract-args/--replace-args/--annotate-args 接收 JSON 字符串数组透传，--dry-run 仅打印命令。
- [x] 写六种连续范围、反向范围、失败即停、路径安全和参数透传测试，先验证失败。
- [x] 实现参数解析、命令规划、预检和顺序执行；默认不覆盖产物，显式 --overwrite 才允许。
- [x] 写新旧命令、服务器 Git 同步/备份/冲突处理、忽略文件保留说明。
- [x] 完整测试、独立复核及修复，纳入流水线功能提交并确认 Git 状态。

## 执行记录
- Ruling: 按用户要求直接在当前 main 工作区执行，不另建 worktree；不进行 merge/push。
- 初始提交：2a9d4ea、0250f93、0edddbd。
- 目录迁移首轮完整测试 36/36 通过（含 spawn）；流水线专项 5/5 通过。实际目录 CLI 与完整追加测试正在运行。
- 新旧 import 采用 sys.modules 转发而不是星号导入，确保全局状态只有一份；共享模块保持根目录。
- 独立复核发现并验证三项问题：缩写绕过受管参数、Batch/多模型附属输出未纳入全链路检查、大写后缀与旧提取器不一致。回归测试先复现失败，再修复；流水线专项现为 8/8 通过。
- Batch 路径纯函数迁到共享 file_paths.py，batch_inference.py 保留原名重导出；预检不导入模型依赖，路径规划与实际运行保持同源。
- 迁移提交前完整测试 42/42 通过；25 个移动实现与原提交逐一比较，除已声明的资源根路径调整和末尾换行规范化外一致。
- 实际子进程冒烟：替换输出两条句对；分步与 all-in-one 标注各输出两条结果及汇总；全程离线，不请求模型。
- 最终验证：`python -m unittest discover -s tests -v`，45/45 通过（383.312 秒）；实际 CLI 将三项复核问题全部预检拒绝，退出码 2，输入内容不变且未创建其他文件。
- 独立复核的三项代码问题均以 RED→GREEN 回归修复；无待修复的重要问题。文档任务状态已更新；未推送、未连接服务器、未进行在线模型验证。
