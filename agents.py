# encoding: utf8

"""Agent 工厂：按 model_config/api_keys.json 里的配置创建 langchain / deepagents agent。

本模块只负责"把配置变成模型和 agent"，不写任何业务提示语。业务侧的分工是：
    spatial_agents.py   六个 agent（输入什么、产出什么）
    spatial_graphs.py   把 agent 接成图（状态、节点适配器、各图 builder）
    spatial_runner.py   跑批入口（并发、落盘、命令行）
工具见 spatial_tools.py 与 grammar_agent.py。
"""

import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

import config
import retry

call_with_retry = retry.call_with_retry
"""给业务 agent 用的重试入口（见 spatial_agents.BaseAgent._invoke / all_in_one.call_once）"""

is_retryable_error = retry.is_retryable_error
"""给调用方判断“这次失败值不值得重试”的同一个口径（见 retry.is_retryable_error）"""

_PROJECT_ROOT = Path(__file__).resolve().parent
_MODEL_CONFIG_PATH = _PROJECT_ROOT / "model_config" / "api_keys.json"
"""模型配置文件。用绝对路径而非相对路径，避免换工作目录运行时报找不到文件"""

_REQUIRED_FIELDS = ("model_name", "provider", "base_url", "api_key")

_PROXY_PROVIDERS = frozenset({"openai", "azure_openai", "deepseek"})
"""支持通过 http_client/http_async_client 控制代理的 provider。

这些 provider 的 LangChain Chat 类接受自定义 httpx 客户端；其他 provider
（例如当前的 openrouter）不能安全透传这两个字段，显式代理配置会在启动时报错，
而不是静默退回环境代理。
"""

_DISABLE_THINKING_DEFAULTS: dict[str, dict] = {
    # 各家「关掉思考模式」的参数名不一样，只能按 provider 分别给。想改写就
    # 在 api_keys.json 的该模型下加 "disable_thinking"，会整体覆盖这里的默认值。
    "deepseek": {"extra_body": {"thinking": {"type": "disabled"}}},
    # OpenRouter 没有通用的「关思考」开关：部分端点（例如 claude-opus-5.5）
    # 明确要求 reasoning 必须开启，传 reasoning.effort="none" 会直接 400
    # "Reasoning is mandatory for this endpoint and cannot be disabled"。
    # 这里保持空字典 = 不补任何参数，沿用模型默认；需要压成本/延迟的模型
    # 在自己的配置里写 "disable_thinking": {"reasoning": {"effort": "minimal"}}。
    "openrouter": {},
}
"""关闭思考模式时，往 init_chat_model 里补的参数，按 provider 索引

这里必须用各 provider 建模类自己的字段名，不能统一走 extra_body：
    ChatDeepSeek 继承 BaseChatOpenAI，extra_body 是它声明的字段，直接透传；
    ChatOpenRouter 当前不走 BaseChatOpenAI 的字段表，自己重写了字段，
    没有 extra_body。硬塞的话 pydantic 会把它挪进 model_kwargs 并报警告，
    而且翻不回来——langchain_openrouter 把 model_kwargs 摊平后整个交给
    openrouter SDK 的 chat.send()，而后者不接受 extra_body 这个关键字，
    每次都 TypeError。它的 reasoning 设置走 reasoning 字段。
"""

_NATIVE_STRUCTURED_OUTPUT_PROVIDERS = frozenset({"openrouter"})
"""这些 provider 的结构化输出改用模型原生 JSON Schema，而不是强制 tool_choice

LangChain 的 ToolStrategy 会把 tool_choice 置成 "any"（或具体工具名）来逼模型
调用结构化输出工具；但 OpenRouter 上的 claude-opus-5.5 明确不支持
tool_choice="tool"/"any"（400 "tool_choice: type \"tool\" and \"any\" are not
supported for this model"）。ProviderStrategy 改走 response_format=json_schema，
实测同一条判定能正常返回结构化结果。
"""


def load_model_config(model_name: str) -> dict:
    """读取并校验某个模型的配置

    Args:
        model_name (str): api_keys.json 中的配置名

    Returns:
        dict: 该模型的配置
    """
    with open(_MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        model_config: dict[dict] = json.load(f)

    try:
        model_cfg = model_config[model_name]
    except KeyError:
        raise ValueError(f"模型配置中没有找到 {model_name} 的配置，请检查 {_MODEL_CONFIG_PATH} 文件。")

    for field in _REQUIRED_FIELDS:
        if field not in model_cfg:
            raise ValueError(f"模型配置中缺少必需字段 '{field}'，请检查 {_MODEL_CONFIG_PATH} 文件。")
    return model_cfg


def request_policy(model_name: str) -> dict:
    """读取某个模型的超时/重试策略（模型配置优先，其次 config.py 的默认值）

    返回的 dict 直接交给 ``retry.call_with_retry``，键为：
        timeout               单次请求超时秒数
        max_retries           首次之外最多重试几次（默认 config.LLM_MAX_RETRIES = 6）
        backoff_seconds       首次退避秒数
        backoff_max_seconds   单次退避上限
        jitter_seconds        随机抖动上限

    配置写错（bool、负数、小数次数、字符串）就在启动时抛 ValueError，
    不把一次“重试其实没生效”的跑批留到事后从耗时里猜。
    """
    return _request_policy(load_model_config(model_name), model_name)


def _request_policy(model_cfg: dict, model_name: str) -> dict:
    """request_policy 的实现：从已读出的模型配置构造策略"""
    timeout = model_cfg.get("timeout", config.REQUEST_TIMEOUT_SECONDS)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"模型 {model_name!r} 的 timeout 必须是正数，收到 {timeout!r}")

    def _int(field: str, default: int) -> int:
        value = model_cfg.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"模型 {model_name!r} 的 {field} 必须是非负整数，收到 {value!r}")
        return value

    def _float(field: str, default: float) -> float:
        value = model_cfg.get(field, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"模型 {model_name!r} 的 {field} 必须是非负数，收到 {value!r}")
        return float(value)

    return {
        "timeout": float(timeout),
        "max_retries": _int("max_retries", config.LLM_MAX_RETRIES),
        "backoff_seconds": _float("retry_backoff_seconds", config.LLM_RETRY_BACKOFF_SECONDS),
        "backoff_max_seconds": _float(
            "retry_backoff_max_seconds", config.LLM_RETRY_BACKOFF_MAX_SECONDS
        ),
        "jitter_seconds": _float("retry_jitter_seconds", config.LLM_RETRY_JITTER_SECONDS),
    }


def _model_provider(model_cfg: dict) -> str:
    """模型配置里的 provider（兼容 model_provider 这个 LangChain 名称）"""
    return str(model_cfg.get("model_provider") or model_cfg["provider"])


def _openrouter_client_kwargs(model_cfg: dict, model_name: str, policy: dict) -> dict:
    """为 ChatOpenRouter 显式建 SDK client，消掉两处「秒/毫秒」与隐藏重试的坑

    1. ``ChatOpenRouter.timeout`` 的单位是**毫秒**（字段名 request_timeout，
       透传给 openrouter SDK 的 timeout_ms），而本项目其余 provider 的 timeout
       都是**秒**。直接把 ``policy["timeout"]``（例如 120）传进去会变成 120ms，
       连 TLS 握手都完不成。
    2. ``ChatOpenRouter`` 在 ``max_retries=0`` 时**不会**给 SDK 传 retry_config，
       SDK 的 ``chat.send`` 于是回退到自己的默认值：backoff + max_elapsed_time
       = 3600000ms（整整 1 小时），连连接超时也会重试。业务层的
       retry.call_with_retry 再在外面重试 6 次，一次卡住的调用可以拖 7 小时。
       这里显式传 RetryConfig(strategy="none")，把重试集中到业务层一层。

    返回的 kwargs 直接合并进 init_chat_model：timeout 用毫秒，client 用显式
    client（ChatOpenRouter 看到 client 就不再自己 _build_client）。
    """
    try:
        import openrouter
        from openrouter.utils import RetryConfig
    except ImportError as exc:  # pragma: no cover —— 依赖缺失时给一句能看懂的错
        raise RuntimeError(
            f"模型 {model_name!r} 使用 openrouter provider，但当前环境没有安装 openrouter 包"
        ) from exc

    timeout_ms = max(1, int(round(policy["timeout"] * 1000)))
    client = openrouter.OpenRouter(
        api_key=model_cfg["api_key"],
        server_url=model_cfg["base_url"],
        timeout_ms=timeout_ms,
        retry_config=RetryConfig(
            strategy="none",
            backoff=None,
            retry_connection_errors=False,
        ),
    )
    return {"timeout": timeout_ms, "client": client}


def _structured_output_for(model_name: str, response_format):
    """按模型能力把 ToolStrategy 换成 ProviderStrategy

    绝大多数模型（含 deepseek）走 ToolStrategy：强制模型调用结构化输出工具，
    拿不到 tool call 就算失败，语义最明确。OpenRouter 上的 claude-opus-5.5
    不支持 tool_choice="tool"/"any"，只能走原生 response_format=json_schema；
    对这类 provider 在这里统一换掉，调用方（spatial_agents）不必逐个判断。

    模型配置里可写 ``"structured_output": "provider"`` 或 ``"tool"`` 显式指定；
    不写时按 provider 取默认（openrouter -> provider，其余 -> tool）。
    """
    if not isinstance(response_format, ToolStrategy):
        return response_format
    model_cfg = load_model_config(model_name)
    override = model_cfg.get("structured_output")
    if override is None:
        mode = (
            "provider"
            if _model_provider(model_cfg) in _NATIVE_STRUCTURED_OUTPUT_PROVIDERS
            else "tool"
        )
    elif isinstance(override, str) and override.strip().lower() in {"provider", "tool"}:
        mode = override.strip().lower()
    else:
        raise ValueError(
            f"模型 {model_name!r} 的 structured_output 必须是 \"provider\" 或 \"tool\"，"
            f"收到 {override!r}"
        )
    return ProviderStrategy(response_format.schema) if mode == "provider" else response_format


def proxy_policy(model_cfg: dict, model_name: str) -> tuple[str, str | None]:
    """解析模型配置里的 proxy 字段，返回 ``(模式, 代理 URL)``。

    模式只有三种：

    - ``env``：沿用进程环境里的 HTTP_PROXY/HTTPS_PROXY/ALL_PROXY/NO_PROXY
    - ``direct``：强制直连，httpx 不读取任何环境代理
    - ``url``：使用显式代理 URL，不受环境代理和 NO_PROXY 影响

    配置值语义见 ``get_model`` 的文档。错误信息不回显代理 URL 本身，避免
    URL 里带账号密码时把凭据写进日志。
    """
    if "proxy" not in model_cfg:
        return "env", None

    value = model_cfg["proxy"]
    if isinstance(value, bool):
        return ("env", None) if value else ("direct", None)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(
                f"模型 {model_name!r} 的 proxy 不能为空字符串；"
                "省略、\"env\" 或 true 表示沿用环境代理，false/direct 表示直连，"
                "也可以填 http(s):// 或 socks5:// 代理地址。"
            )
        normalized = text.lower()
        if normalized in {"env", "default", "inherit"}:
            return "env", None
        if normalized in {"none", "off", "direct"}:
            return "direct", None
        try:
            parsed = urlsplit(text)
            # 访问一次 .port：非法端口（例如 host:abc）会在这里抛 ValueError，
            # 不必等到第一次请求才发现配置错误。
            parsed.port
        except ValueError:
            parsed = None
        if (
            parsed is None
            or parsed.scheme.lower() not in {"http", "https", "socks5", "socks5h"}
            or not parsed.hostname
        ):
            raise ValueError(
                f"模型 {model_name!r} 的 proxy 不是合法代理地址：需要 http(s):// 或 socks5:// "
                "前缀，且必须包含代理主机；只想强制直连请用 false 或 \"direct\"。"
            )
        return "url", text

    raise ValueError(
        f"模型 {model_name!r} 的 proxy 必须是布尔值或字符串，收到 {type(value).__name__}；"
        "省略、\"env\" 或 true 表示沿用环境代理，false/direct 表示直连。"
    )


def _close_http_clients(
    clients: tuple[httpx.Client | None, httpx.AsyncClient | None],
) -> None:
    """尽力关闭构造过程中创建的 httpx 客户端（只在模型构造失败时调用）。"""
    sync_client, async_client = clients
    if sync_client is not None:
        try:
            sync_client.close()
        except Exception:  # noqa: BLE001 —— 清理失败不能覆盖原始构造异常
            pass
    if async_client is not None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(async_client.aclose())
            except Exception:  # noqa: BLE001 —— 清理失败不能覆盖原始构造异常
                pass
            return
        # 当前线程已有事件循环：不能在循环里再 asyncio.run，只能把关闭动作
        # 排到该循环上；创建任务失败时显式关闭 coroutine，避免 “never awaited”。
        coroutine = async_client.aclose()
        try:
            loop.create_task(coroutine)
        except Exception:  # noqa: BLE001 —— 循环已关闭/正在关闭时只能放弃清理
            coroutine.close()


def _proxy_client_kwargs(
    model_cfg: dict,
    model_name: str,
    provider: str,
    timeout: float,
) -> tuple[dict, tuple[httpx.Client | None, httpx.AsyncClient | None]]:
    """按 proxy 配置构造传给 ``init_chat_model`` 的客户端参数。

    返回 ``(kwargs, clients)``：``kwargs`` 直接合并进模型参数，``clients`` 仅用于
    模型构造失败时的清理。``env`` 模式返回空 kwargs，完全保持原有行为。
    """
    mode, proxy_url = proxy_policy(model_cfg, model_name)
    if mode == "env":
        return {}, (None, None)

    if provider not in _PROXY_PROVIDERS:
        raise ValueError(
            f"模型 {model_name!r} 的 provider {provider!r} 不支持 proxy 配置；"
            f"当前支持：{'、'.join(sorted(_PROXY_PROVIDERS))}。"
            "请删除 proxy 字段以沿用环境代理，或换用支持自定义 HTTP 客户端的 provider。"
        )

    client_kwargs: dict = {"trust_env": False, "timeout": timeout}
    if proxy_url is not None:
        client_kwargs["proxy"] = proxy_url

    sync_client: httpx.Client | None = None
    async_client: httpx.AsyncClient | None = None
    try:
        sync_client = httpx.Client(**client_kwargs)
        async_client = httpx.AsyncClient(**client_kwargs)
    except Exception:
        _close_http_clients((sync_client, async_client))
        raise ValueError(
            f"模型 {model_name!r} 的 proxy 配置无法创建 HTTP 客户端；请检查代理地址和协议。"
        ) from None

    return (
        {
            "http_client": sync_client,
            "http_async_client": async_client,
            # 显式客户端与 OPENAI_PROXY 不能同时生效；置空以覆盖环境变量默认值。
            "openai_proxy": None,
            # 关闭 langchain-openai 的默认 transport 注入，避免它与自定义客户端叠加。
            "http_socket_options": (),
        },
        (sync_client, async_client),
    )


def disable_thinking_kwargs(model_cfg: dict) -> dict:
    """该模型「关掉思考模式」要补的参数（供 get_model 与原生 Batch 请求共用）

    模型配置里写了 "disable_thinking" 就整体以配置为准，否则按 provider 取默认值。
    原生 Batch 走裸 HTTP，没有 langchain 的 extra_body 包装，调用方需要自行摊平
    （见 batch_inference._flatten_extra）。
    """
    # 兼容 model_provider 这个 LangChain 名称，与 get_model 的取法保持一致。
    provider = _model_provider(model_cfg)
    return dict(model_cfg.get("disable_thinking") or _DISABLE_THINKING_DEFAULTS.get(provider) or {})


def get_model(model_name: str, disable_thinking: bool = False):
    """按配置初始化 chat model

    Args:
        model_name (str): api_keys.json 中的配置名
        disable_thinking (bool): 是否关闭思考模式。deepseek 的思考模式不支持被强制的
            tool_choice（报 400 "Thinking mode does not support this tool_choice"），
            而 ToolStrategy 结构化输出正是靠强制调用结构化输出工具实现的，两者冲突；
            所以凡是要结构化输出就必须关掉思考。代价是失去思维链。
            具体补什么参数按 provider 取 _DISABLE_THINKING_DEFAULTS，配置里写了
            "disable_thinking" 则整个以配置为准。OpenRouter 没有通用的关闭开关
            （部分端点要求 reasoning 必须开启），默认不补参数。

    代理口径：模型配置可选 ``"proxy"`` 字段，不同模型可以各配各的：

        - 省略 / ``"env"`` / ``true``：沿用环境变量代理（原有行为）
        - ``false`` / ``"none"`` / ``"off"`` / ``"direct"``：强制直连
        - ``"http://..."`` / ``"https://..."`` / ``"socks5://..."``：显式代理

    显式代理/直连只对 ``_PROXY_PROVIDERS`` 里的 provider 生效；其他 provider
    如果显式配置会直接报错，避免“以为改了代理、实际仍在走环境变量”。

    超时口径：模型配置里的 ``timeout`` 一律是**秒**。OpenRouter 的 ChatOpenRouter
    要毫秒，由 ``_openrouter_client_kwargs`` 在内部换算，调用方不必区分 provider。

    Returns:
        BaseChatModel: 初始化好的模型。注意不要在其上再调 .with_retry()，
            它返回的 RunnableRetry 没有 bind_tools，会在 agent 绑工具时崩掉。

    重试口径：SDK 自带的重试**显式关掉**（max_retries=0），统一由
    retry.call_with_retry 在业务调用层做。两层都开 6 次会叠成最多 49 次请求，
    一次超时就能把整批拖到不可控；集中到一层后，重试次数也才能如实记进
    agent_stats（见 spatial_agents.BaseAgent.stats）。
    """
    model_cfg = load_model_config(model_name)
    policy = _request_policy(model_cfg, model_name)

    # langchain 1.x 的 init_chat_model 把参数改名了：model_name -> model，
    # provider -> model_provider（旧名会被当成模型字段透传，随后报错），此处做一次映射。
    kwargs = {
        "model": model_cfg.get("model") or model_cfg["model_name"],
        "model_provider": model_cfg.get("model_provider") or model_cfg["provider"],
        "base_url": model_cfg["base_url"],
        "api_key": model_cfg["api_key"],
        "temperature": model_cfg.get("temperature", 0),
        # 超时由这一层显式给；重试统一收在 retry.call_with_retry，SDK 层置 0。
        "timeout": policy["timeout"],
        "max_retries": 0,
    }
    if disable_thinking:
        disable_kwargs = disable_thinking_kwargs(model_cfg) or None
        if disable_kwargs is None:
            raise ValueError(
                f"不知道 provider '{kwargs['model_provider']}' 该怎么关思考模式，"
                f"请在 {_MODEL_CONFIG_PATH} 的 {model_name} 配置里加 \"disable_thinking\" 字段。"
            )
        kwargs.update(disable_kwargs)

    # OpenRouter 的两处 provider 差异在这里统一适配：timeout 用毫秒，
    # 并显式关掉 SDK 的隐藏重试（见 _openrouter_client_kwargs）。
    if str(kwargs["model_provider"]) == "openrouter":
        kwargs.update(_openrouter_client_kwargs(model_cfg, model_name, policy))

    # 代理参数放在 disable_thinking 之后合并：模型配置里的 disable_thinking 是
    # provider 专属参数集合，不应意外覆盖本模型的代理策略。
    proxy_kwargs, proxy_clients = _proxy_client_kwargs(
        model_cfg, model_name, str(kwargs["model_provider"]), policy["timeout"]
    )
    kwargs.update(proxy_kwargs)
    try:
        return init_chat_model(**kwargs)
    except Exception:
        _close_http_clients(proxy_clients)
        raise


def get_agent(model_name: str, tools: list, system_prompt: str, response_format=None):
    """创建一个 langchain agent

    Args:
        model_name (str): 模型名称
        tools (list): 工具列表，每个工具是一个 langchain Tool 对象
        system_prompt (str): 系统提示语，用于指导模型的行为
        response_format: 结构化输出策略，如 ToolStrategy(SomeModel)。传入时会自动关闭
            思考模式（见 get_model）；不传则保留模型的默认行为。

    Returns:
        agent: 创建好的 langchain agent 对象
    """
    # OpenRouter/claude-opus-5.5 不支持强制 tool_choice，结构化输出改走原生
    # JSON Schema（见 _structured_output_for）。其余 provider 保持 ToolStrategy。
    response_format = _structured_output_for(model_name, response_format)
    model = get_model(model_name, disable_thinking=response_format is not None)
    return create_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        **(dict(response_format=response_format) if response_format is not None else {}),
    )


def get_deep_agent(
    model_name: str,
    tools: list,
    system_prompt: str,
    subagents: list | None = None,
    response_format=None,
):
    """创建一个 deepagents agent（带 task 工具，可把子任务派给 subagent）

    Args:
        model_name (str): 模型名称
        tools (list): 主 agent 的工具列表
        system_prompt (str): 主 agent 的系统提示语
        subagents (list, optional): 子 agent 定义，每项形如
            {"name": ..., "description": ..., "system_prompt": ..., "tools": [...]}。
            description 是主 agent 决定何时调用它的唯一依据，要写清"什么时候用它"。
        response_format: 结构化输出策略，见 get_agent

    Returns:
        agent: 创建好的 deepagents agent 对象
    """
    # create_deep_agent 只接受 model=/subagents=/response_format= 这些参数，
    # 模型要先由 init_chat_model 建好再传进来。
    response_format = _structured_output_for(model_name, response_format)
    model = get_model(model_name, disable_thinking=response_format is not None)
    return create_deep_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        subagents=list(subagents or ()),
        response_format=response_format,
    )
