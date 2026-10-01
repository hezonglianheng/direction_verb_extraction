# encoding: utf8

"""提示语的读取与渲染。

提示语正文绝大多数放在项目根目录的 system_prompt/ 下，一个 agent 一份（多份时用 <名字>_<用途>.txt），
目录用 __file__ 解析成绝对路径，与运行时 cwd 无关（与 config.py / spatial_tools.py 同一做法）。
agent 在构造期调用 read_prompt / render_prompt 把正文读进来，见 spatial_agents.BaseAgent。

例外是 reality_agent 那一份：它的判据按语料类型分，所以与它那组配置表一起放在
config_files/<语料类型>/ 下（见 spatial_tools.CONFIG_PROMPT_FILES），读法是 read_prompt_path
——两条路的读法与报错文案是同一套，只有路径的来源不同。
"""

import re
from pathlib import Path
from typing import Any, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parent
"""项目根目录（本模块所在目录）"""

PROMPT_DIR = _PROJECT_ROOT / "system_prompt"
"""提示语目录。注意各调用方要经 read_prompt / prompt_path 取用，不要另存一份引用"""

PROMPT_SUFFIX = ".txt"
"""提示语文件后缀，调用方只传名字（如 "semantic"）"""

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
"""占位符写作 {{名字}}。

刻意不用 str.format：提示语正文里有 JSON 花括号（形如 {"values": [...], "evidence": "..."}），
.format() 会把 "values" 当成字段名而抛 KeyError。也不用 string.Template：$ 在句子示例里可能出现，
且它要求 $$ 转义、可读性更差。
"""


def read_sys_prompt_from_file(file_path: str | Path, encoding: str = "utf-8-sig") -> str:
    """从文件中读取系统提示语，去掉首尾空白

    默认编码用 utf-8-sig：带 BOM 的文件会丢掉 BOM，不带 BOM 的文件行为与 utf-8 完全相同。
    提示语可能在 Windows 上用记事本/VS Code 编辑，一个不可见的 U+FEFF 混进提示语首部不会报错，
    只会悄悄改变送给模型的内容，故这里主动吃掉它。
    """
    with open(file_path, "r", encoding=encoding) as f:
        return f.read().strip()


def prompt_path(name: str) -> Path:
    """提示语文件的绝对路径（name 不含后缀）"""
    return PROMPT_DIR / f"{name}{PROMPT_SUFFIX}"


def read_prompt_path(path: str | Path) -> str:
    """按显式路径读一份提示语

    与 read_prompt 同一口径（同一套编码与报错文案），多出来的只是「路径由调用方给」：
    reality_agent 的提示语不在 system_prompt/ 下，而是与它那组配置表同目录
    （见 spatial_tools.config_prompt_path），读法不能只有 system_prompt/ 那一条。
    """
    path = Path(path)
    try:
        return read_sys_prompt_from_file(path)
    except FileNotFoundError as exc:
        # 读不到时把同目录下现有哪些 .txt 一并列出来——提示语文件名写错（或整组目录还没建）
        # 是启动期就该报的配置错误，报错信息要足以让人当场改对，不用再去翻代码。
        available = "、".join(sorted(p.name for p in path.parent.glob(f"*{PROMPT_SUFFIX}"))) or "（目录为空）"
        raise FileNotFoundError(f"找不到提示语文件 {path}；{path.parent} 下有：{available}") from exc


def read_prompt(name: str) -> str:
    """按名字读 system_prompt/<name>.txt（读不到时的报错见 read_prompt_path）"""
    return read_prompt_path(prompt_path(name))


def render_prompt(template: str, values: Mapping[str, Any], *, source: str = "") -> str:
    """把模板里的 {{名字}} 占位符替换成 values[名字]

    单次扫描替换：值里若含 {{...}} 不会再被替换一遍，替换结果与 values 的迭代顺序无关。
    模板里出现 values 没给的占位符视为写错（多半是 {{n}} 写成了 {{nn}}），直接报错：提示语属于配置，
    写错应该在启动时就报，而不是静默送一份还带着 {{...}} 的提示语给模型。
    代价是模板正文不能出现字面量 {{ —— 现有提示语都没有，真要写时另行约定转义写法。
    """
    unknown = set(_PLACEHOLDER.findall(template)) - set(values)
    if unknown:
        raise ValueError(f"{source or '提示语模板'} 里有未提供的占位符：{'、'.join(sorted(unknown))}")
    return _PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), template)
