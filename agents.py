# encoding: utf8

"""Agent 工厂：按 model_config/api_keys.json 里的配置创建 langchain / deepagents agent。

本模块只负责"把配置变成模型和 agent"，不写任何业务提示语。业务侧的分工是：
    spatial_agents.py   六个 agent（输入什么、产出什么）
    spatial_graphs.py   把 agent 接成图（状态、节点适配器、各图 builder）
    spatial_runner.py   跑批入口（并发、落盘、命令行）
工具见 spatial_tools.py 与 grammar_agent.py。
"""

import json
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

_PROJECT_ROOT = Path(__file__).resolve().parent
_MODEL_CONFIG_PATH = _PROJECT_ROOT / "model_config" / "api_keys.json"
"""模型配置文件。用绝对路径而非相对路径，避免换工作目录运行时报找不到文件"""

_REQUIRED_FIELDS = ("model_name", "provider", "base_url", "api_key")


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


def get_model(model_name: str, disable_thinking: bool = False):
    """按配置初始化 chat model

    Args:
        model_name (str): api_keys.json 中的配置名
        disable_thinking (bool): 是否关闭思考模式。deepseek 的思考模式不支持被强制的
            tool_choice（报 400 "Thinking mode does not support this tool_choice"），
            而 ToolStrategy 结构化输出正是靠强制调用结构化输出工具实现的，两者冲突；
            所以凡是要结构化输出就必须关掉思考。代价是失去思维链。

    Returns:
        BaseChatModel: 初始化好的模型。注意不要在其上再调 .with_retry()，
            它返回的 RunnableRetry 没有 bind_tools，会在 agent 绑工具时崩掉；
            重试交给这里的 max_retries。
    """
    model_cfg = load_model_config(model_name)

    # langchain 1.x 的 init_chat_model 把参数改名了：model_name -> model，
    # provider -> model_provider（旧名会被当成模型字段透传，随后报错），此处做一次映射。
    kwargs = {
        "model": model_cfg.get("model") or model_cfg["model_name"],
        "model_provider": model_cfg.get("model_provider") or model_cfg["provider"],
        "base_url": model_cfg["base_url"],
        "api_key": model_cfg["api_key"],
        "temperature": model_cfg.get("temperature", 0),
        "max_retries": model_cfg.get("max_retries", 6),
    }
    if disable_thinking:
        kwargs["extra_body"] = model_cfg.get("extra_body", {"thinking": {"type": "disabled"}})
    return init_chat_model(**kwargs)


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
    model = get_model(model_name, disable_thinking=response_format is not None)
    return create_deep_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        subagents=list(subagents or ()),
        response_format=response_format,
    )
