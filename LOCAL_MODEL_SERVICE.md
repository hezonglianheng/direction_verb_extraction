# 本地小模型服务接入说明

`extract`（语料提取）与 `replace`（句对替换）的 LLM 筛选现在同时支持：

- **远程 API**：保持原有 `model_config/api_keys.json` 配置方式，不启动任何本地进程；
- **本地部署模型**：在模型配置里加 `local_service` 块，程序启动时自动拉起
  `/home/qinyuhang/model_service`，就绪后连接，程序结束（含异常、Ctrl-C）时自动关闭。

生命周期实现见 [`extract/local_model_service.py`](extract/local_model_service.py)，`replace`
通过同一个模块复用，不各写一套。

## 1. 配置本地模型

在 `model_config/api_keys.json` 中增加一个模型条目（也可复制
`model_config/api_keys.example.json` 里的 `local-qwen`）：

```json
"local-qwen": {
  "model_name": "Qwen/Qwen3.8-27B",
  "provider": "openai",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key": "EMPTY",
  "temperature": 0,
  "proxy": false,
  "timeout": 600,
  "max_retries": 6,
  "disable_thinking": {
    "extra_body": {"chat_template_kwargs": {"enable_thinking": false}}
  },
  "local_service": {
    "enabled": true,
    "service_dir": "/home/qinyuhang/model_service",
    "start_command": ["bash", "service.sh", "start", "--wait"],
    "stop_command": ["bash", "service.sh", "stop"],
    "status_command": ["bash", "service.sh", "status"]
  }
}
```

要点：

- **地址只写一处**：`base_url` 就是本地小模型地址，例如
  `http://127.0.0.1:8000/v1`。健康检查地址由它自动推导为
  `http://127.0.0.1:8000/health`；也可以用 `local_service.health_url` 显式指定。
- 如果 `local_service` 里也写了 `base_url`，必须与模型条目的 `base_url` 一致，否则启动即报错。
- `local_service` 不写、写 `false`、或写 `{"enabled": false}`，都按远程 API 处理。
- `provider` 用 `openai` 即可对接 vLLM 的 OpenAI 兼容接口。`api_key` 不能为空，
  本地服务不鉴权时填 `"EMPTY"`。
- `disable_thinking` 是 `agents.get_model` 的必填兼容项：筛选走 `ToolStrategy` 结构化输出，
  会要求关闭思考模式。若本地模型需要别的关闭参数，按 `agents.py` 的 provider 约定填写；
  也可以把 `"structured_output": "provider"` 加进该条目，改走原生 JSON Schema。
- 默认值（服务目录、命令、超时）集中在 `config.py` 的 `LOCAL_MODEL_SERVICE_*`，模型条目里
  可以逐项覆盖。

## 2. 运行

```bash
# 提取：使用本地模型做 LLM 筛选
.venv/bin/python -m extract data/raw.jsonl -f config_files/direction_expressions.json5 \
  --screen-model local-qwen

# 替换：同样使用本地模型做两侧检查
.venv/bin/python -m replace --corpus results/direction_corpus.jsonl --corpus-type 趋向动词 \
  --model local-qwen
```

启动日志会出现：

```text
[本地模型服务] 准备本地模型 local-qwen：http://127.0.0.1:8000/v1
[本地模型服务] 服务未运行，开始启动
[本地模型服务] 服务已启动并就绪：http://127.0.0.1:8000/health
...
[本地模型服务] 本地模型服务已关闭
```

## 3. 生命周期约定

- 启动前先访问 `/health`：已就绪则直接连接，**不会**在退出时关闭；
- 状态命令显示进程已存在但未就绪（退出码 1）：等待它启动完成，**不会**关闭；
- 状态命令显示未运行（退出码 3）：本次程序负责启动，并在正常结束、异常、Ctrl-C 时执行
  `stop_command` 关闭；同时注册 `atexit` 作为兜底；
- 启动失败不会留下半启动状态：本次尝试拉起的服务会被尽力停止，然后抛出原始错误；
- 远程模型、`--no-llm-screen`、`--no-llm`、提示语为空导致筛选器未启用时，完全不读本地服务配置，
  也不启动任何进程。
- `status_command` 默认用 `service.sh status` 区分服务所有权；如果显式设为 `null`，程序无法
  知道服务是否由外部启动，会按“本次负责启动并关闭”处理。服务可能由其他任务预先启动时，
  请保留默认的 `status_command`，或自行提供等价的状态命令。
- 多个 extract/replace 进程可以同时使用同一服务：每个进程用共享 `flock` 登记“仍在使用”，
  启动决策再用独占锁串行化。先启动的任务退出时，只要还有别的任务在使用，就只断开自己、
  **不立即执行 stop**，而是安排一个 detached reaper 等最后一个任务退出后再关闭服务。
  两个任务同时发现未启动时也只会拉起一份。锁文件位于系统临时目录（按
  `service_dir + base_url` 区分），不会写入被管理的服务目录。
- 默认从 `<service_dir>/run/server.pid` 读取服务 PID（可用 `local_service.pid_file`
  覆盖，写 `null` 表示没有 PID 文件）。`service.sh start` 会通过内部环境变量写回
  “本次启动/复用 + PID”的回执，`stop` 前再核对 PID；服务若已被人工或其他实例替换，
  就跳过 stop，避免误停新服务。

## 4. 批处理与并发调优

`extract` / `replace` 的 LLM 筛选本身已经批量：一个请求最多放
`SCREEN_BATCH_SIZE` 条「句子 + 目标词」判定项，再由 `SCREEN_WORKERS` 个线程并发提交；
本地 vLLM 用 continuous batching 处理这些并发请求，不需要（也不能）走根目录
`batch_inference.py` 的原生 `/batches` 接口。

模型条目可以按模型覆盖这两个值（CLI 显式参数优先级最高）：

```json
"local-qwen": {
  "screen_batch_size": 20,
  "screen_workers": 8,
  "batch": false
}
```

- `batch: false` 避免 `annotate` / `evaluate` 把本地 vLLM 误当成支持 OpenAI 原生 Batch 的端点；
- `--screen-batch-size` / `--screen-workers`（`extract`）与 `--batch-size` / `--workers`（`replace`）
  可临时覆盖模型配置；
- 不建议盲目增大 batch size：一个请求内的多条结果仍是串行解码，批次越大越容易因个别
  `item_id` 缺漏导致整批 `error`。先在固定语料上比较 items/s、输出 token/s、尾延迟和
  `invalid_items`，再决定是否调整；
- 服务端可同步实验 `model_service/config.py` 的 `MAX_NUM_SEQS`（8 → 32）和
  `MAX_NUM_BATCHED_TOKENS`（8192 → 16384），但不要未实测就改成默认；显存和 KV cache
  仍是最先触顶的约束。

## 5. 注意与限制

- 本地服务管理目前只支持 POSIX；默认的 `model_service/service.sh` 面向 Linux
  （使用了 `/proc`、GNU `stat`），macOS 需要换自定义的 start/stop/status 命令。
- `service.sh start --wait` 首次启动可能要下载约 55GB 模型，默认等待窗口 4 小时；
  可用 `local_service.start_timeout` / `health_timeout` 调整。
- 该生命周期只挂在 `extract` 与 `replace` 的入口上。其他入口（例如 `annotate`）如果也使用
  同一个本地模型，需要自行先启动服务；本模块不会替它们管理进程。
- 正常退出、未捕获异常、Ctrl-C 与 `SIGTERM` 都会触发关闭；只有 `SIGKILL` 强杀或断电
  无法执行清理。下一次运行会通过状态命令重新判断，不会误关仍在运行的服务。
