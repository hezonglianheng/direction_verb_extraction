# encoding: utf8

"""语料构造阶段的 LLM 筛选：判断「这个句子 + 这个方位词/趋向动词」是否值得被提取。

分工：
    sentence_extract.py / entry_in_linux.py / entry_in_win.py
        规则筛选命中后调用本模块；命中记录在落盘前按批送模型筛选。
    sentence_screening.py（本文件）
        提示语映射、user message 拼装、结构化输出、并发与失败降级。

为什么按词类型分两份提示语：
    方位词（上/里/旁/东…）要看的是「是不是在说一处位置、是不是方位短语/处所成分」；
    趋向动词（上/下/进/出来…）要看的是「是不是真的表示位移方向，还是时体/结果/引申用法」。
    两类判据不同，所以正文分别放在
        system_prompt/screen_localization.txt   （方位词）
        system_prompt/screen_direction.txt      （趋向动词）
    里，由用户填写。文件为空时本模块**不建模型、不发请求**，记录原样通过——这样默认行为
    与加筛选之前完全一致，填好提示语后自动生效。

输出的形式化：
    每个目标词一条判定：``{"index", "word", "extractable": True/False, "reason", "source"}``；
    一条记录汇总在 ``record["screening"]`` 里：
        status       true = LLM 判真（进入判定工作流）
                     false = LLM 判假（落盘，但不进入判定）
                     error = LLM 调用/输出异常（落盘，不进入判定）
                     limit = 超出 LLM 发送上限、未判（落盘，不进入判定）
                     no_llm = 未使用 LLM 筛选（落盘，进入判定；旧语料也归入这一类）
        extractable  true/false 时给布尔值；error/limit/no_llm 时为 None
        verdicts     逐词判定明细。
    所有记录都会落盘；判定工作流按 ``screening_status`` 只取 true/no_llm。

批量请求：
    一次 LLM 请求可以放入 ``config.SCREEN_BATCH_SIZE`` 条「句子 + 目标词」判定项，模型按
    ``item_id`` 返回同样数量、同样顺序无关的结果；单条的判据与逐条请求时完全一致。
    累计发送条数受 ``config.SCREEN_MAX_ITEMS`` 限制（<=0 不限）；达到上限后不再发新请求，
    未发送项按 ``source="limit"`` / ``extractable=None`` 保留，语料继续落盘。

并发与缓存：
    多个批请求由 ``screen_rows`` 用线程池并发；线程数由 ``--screen-workers`` /
    ``config.SCREEN_WORKERS`` 控制。相同 (句子, 词, 相关成分) 的判定按**单条**规范内容缓存
    （上限见 ``config.SCREEN_CACHE_MAX``），因此批组合方式不会影响缓存命中。模型请求的
    超时/重试沿用 agents.request_policy 与 retry.call_with_retry。
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

import config
import utils

SCREEN_PROMPT_FILES: dict[str, str] = {
    "方位词": "screen_localization",
    "趋向动词": "screen_direction",
}
"""词类型 -> system_prompt/ 下的提示语文件名（不含 .txt）。"""

SCREENING_TRUE = "true"
"""LLM 明确判为可提取"""
SCREENING_FALSE = "false"
"""LLM 明确判为不可提取（仍落盘，但不进入判定工作流）"""
SCREENING_ERROR = "error"
"""LLM 调用/输出异常（仍落盘，但不进入判定工作流）"""
SCREENING_LIMIT = "limit"
"""达到 LLM 发送条数上限、未送模型（仍落盘，但不进入判定工作流）"""
SCREENING_NO_LLM = "no_llm"
"""未使用 LLM 筛选（提示语为空、--no-llm-screen，或旧语料无 screening 字段）"""

SCREENING_STATUSES = frozenset(
    {SCREENING_TRUE, SCREENING_FALSE, SCREENING_ERROR, SCREENING_LIMIT, SCREENING_NO_LLM}
)
JUDGEMENT_SCREENING_STATUSES = frozenset({SCREENING_TRUE, SCREENING_NO_LLM})
"""判定工作流只接受这两类语料"""

_TARGET_PREFIXES: dict[str, tuple[str, ...] | None] = {
    # 与 word_sampler.CORPUS_SPECS 的口径一致：
    # 方位词取顶层规则自己的词表；趋向动词只取 direction_verb_ 前缀的规则词表，
    # 避免把 accompany_/cause_/manner_ 里的搭配动词（搬/带/送…）当成目标词。
    "方位词": None,
    "趋向动词": ("direction_verb_",),
}

_FAMILY_HINTS: tuple[tuple[str, str], ...] = (
    ("direction", "趋向动词"),
    ("趋向", "趋向动词"),
    ("local", "方位词"),
    ("location", "方位词"),
    ("方位", "方位词"),
)

_PROGRESS_LOCK = threading.Lock()
"""LLM 批次进度行的输出锁：多套规则、多个筛选线程同时完成批次时，保证整行不交错。

计时/计数仍在各线程内完成；锁只覆盖“累加完成数 + 打印这一行”这一小段，
不包住模型调用，因此不会把并发筛选串行化。
"""


class ExtractionVerdict(BaseModel):
    """单条筛选结果：只允许 true/false，理由另附（批处理里的每个 item 仍是这个形状）"""

    extractable: bool = Field(
        description="这个句子里的这个待提取词是否能够被提取：能提取为 true，不能为 false"
    )
    reason: str = Field(description="判断依据，需结合句子中该词的用法与前序步骤提取到的相关成分")


class BatchExtractionItem(ExtractionVerdict):
    """批量结果中的一条：item_id 对应 user message 里“### 语料 <id>”的编号"""

    item_id: int = Field(
        description="输入里该条语料的 item_id，原样照抄；每个输入 item_id 恰好返回一条"
    )


class BatchExtractionVerdict(BaseModel):
    """一次请求返回多条判定；条数与 item_id 必须和输入对得上"""

    results: list[BatchExtractionItem] = Field(
        description="逐条判定结果；每条输入语料一个 item，item_id 必须原样对应，不能漏项或重复"
    )


class ScreenItemBudget:
    """LLM 筛选的累计发送额度（多套规则共用一份）

    只在**首次发送**缓存未命中的判定项时扣减；缓存命中与失败重试不重复扣。
    ``max_items`` 为 None 表示不限。
    """

    def __init__(self, max_items: int | None = None):
        self.max_items = None if not max_items or int(max_items) <= 0 else int(max_items)
        self.sent = 0
        self._lock = threading.Lock()

    def reserve(self, count: int) -> int:
        """占 count 个名额，返回实际允许发送的数量（可能被剩余额度截短）"""
        count = max(0, int(count))
        with self._lock:
            if self.max_items is None:
                self.sent += count
                return count
            allowed = min(count, max(0, self.max_items - self.sent))
            self.sent += allowed
            return allowed

    def stats(self) -> dict:
        with self._lock:
            return {"max_items": self.max_items, "items_sent": self.sent}


def classify_family(family: str) -> str | None:
    """按规则配置文件名判断词类型；判断不出返回 None（调用方可要求显式指定）

    ``family`` 是配置文件名去扩展名（sentenceFilters.FilterGenerator.family_name），
    默认两份配置分别是 ``localization_expressions`` 与 ``direction_expressions``。
    自定义文件名只要带 direction/趋向 或 local/location/方位 也能认出来。
    """
    name = str(family).lower()
    for marker, kind in _FAMILY_HINTS:
        if marker in name:
            return kind
    return None


def prompt_name(kind: str) -> str:
    try:
        return SCREEN_PROMPT_FILES[kind]
    except KeyError:
        raise ValueError(f"没有为词类型 {kind!r} 配置筛选提示语；可用：{'、'.join(SCREEN_PROMPT_FILES)}") from None


def _sentence_text(record: dict) -> str:
    sentence = record.get("sentence")
    if isinstance(sentence, (list, tuple)):
        return "".join(str(token) for token in sentence)
    return str(sentence or "")


def screening_status(record: dict) -> str:
    """一条语料的筛选状态（五类之一），兼容没有 screening 的旧语料

    新记录以 ``screening.status`` 为准；旧记录（本轮改造前产出）没有 status，
    按 ``extractable`` 与 ``source`` 推导。只有**完全没有 screening 字段**的旧语料才视为
    no_llm；screening 存在但形状/状态无法识别时一律按 error，避免把元数据损坏的语料放进
    判定工作流。
    """
    if "screening" not in record:
        return SCREENING_NO_LLM
    block = record.get("screening")
    if not isinstance(block, dict):
        return SCREENING_ERROR
    if "status" in block:
        status = block.get("status")
        if isinstance(status, str) and status in SCREENING_STATUSES:
            return status
        return SCREENING_ERROR
    extractable = block.get("extractable")
    if extractable is True:
        return SCREENING_TRUE
    if extractable is False:
        return SCREENING_FALSE
    source = block.get("source")
    if source in ("no_llm", "unfilled_prompt"):
        return SCREENING_NO_LLM
    if source == "limit":
        return SCREENING_LIMIT
    if source in ("error", "no_target"):
        return SCREENING_ERROR
    return SCREENING_ERROR


def is_judgement_eligible(record: dict) -> bool:
    """判定工作流是否可以使用这条语料：只接受 true 与 no_llm"""
    return screening_status(record) in JUDGEMENT_SCREENING_STATUSES


def annotate_no_llm(
    rows: Iterable[dict],
    *,
    kind: str | None = None,
    reason: str = "未使用 LLM 筛选（--no-llm-screen、提示语为空或未提供筛选器）",
) -> list[dict]:
    """给未经过 LLM 的语料补上 no_llm 标注（原地修改并返回同一批对象）"""
    rows = list(rows)
    reason = reason or "未使用 LLM 筛选（--no-llm-screen、提示语为空或未提供筛选器）"
    for row in rows:
        row["screening"] = {
            "kind": kind,
            "source": "no_llm",
            "status": SCREENING_NO_LLM,
            "extractable": None,
            "verdicts": [],
            "reason": reason,
        }
    return rows


def _status_from_verdicts(verdicts: list[dict], aggregate: bool | None) -> str:
    """行级状态优先级：error > limit > true > false

    多目标句里只要有一个词的调用/输出出错，整条记录不进入判定工作流；有词因额度未判，
    同样不进入。true 与 false 混合时按 true 处理（false 的节点会被裁掉）。
    """
    sources = {verdict.get("source") for verdict in verdicts}
    if "error" in sources:
        return SCREENING_ERROR
    if "limit" in sources:
        return SCREENING_LIMIT
    if aggregate is True:
        return SCREENING_TRUE
    if aggregate is False:
        return SCREENING_FALSE
    return SCREENING_ERROR


def _target_nodes(record: dict, vocab: set[str]) -> list[dict]:
    """这条记录里真正被规则提取出来的目标词节点（口径见 word_sampler.target_nodes）"""
    try:
        # 延迟 import：入口模块的轻量测试会把 sentenceFilters 换成桩，顶层 import 会连带失败
        import word_sampler

        return word_sampler.target_nodes(record, vocab)
    except Exception:  # noqa: BLE001 —— 单条记录形状异常只降级，不带走整批
        return []


def _prune_tree(tree: Iterable[dict], false_indices: set[int], keep_indices: set[int]) -> list[dict]:
    """把被明确判为不可提取的目标词从 match_tree 里摘掉

    返回列表而不是单节点：一个 false 节点若还带着要保留的下级目标词（明确 true 或
    fail-open 的 unknown），把该节点整个删掉会连它一起丢掉；把它提到根层又会留下一个
    词表内的 false 父节点，而下游取词口径是「父节点在词表里时不再认子节点」
    （见 word_sampler.target_nodes）——被保留的子节点反而会被当成结构上下文忽略。
    所以这里**删掉 false 节点、把含保留目标词的子节点提上一层**。
    """

    def walk(node: Any) -> list[dict]:
        if not isinstance(node, dict):
            return []
        index = node.get("index")
        children = [kept for child in (node.get("children") or []) for kept in walk(child)]
        if index in false_indices and index not in keep_indices:
            # 本节点判 false：自身删掉；只把仍含保留目标词的子树提上来
            return [child for child in children if _contains_index(child, keep_indices)]
        return [{**node, "children": children}]

    return [kept for node in tree for kept in walk(node)]


def _contains_index(node: dict, indices: set[int]) -> bool:
    if node.get("index") in indices:
        return True
    return any(_contains_index(child, indices) for child in node.get("children") or [] if isinstance(child, dict))


class SentenceScreener:
    """一条规则族（一套筛选配置）对应的 LLM 筛选器。

    构造期只读提示语、推导目标词表；**不建模型**。提示语为空或 ``llm=False`` 时
    ``enabled`` 为 False，``screen_rows`` 原样返回输入，连模型配置都不会读。
    """

    def __init__(
        self,
        *,
        family: str,
        config_path: str,
        model_key: str,
        llm: bool = True,
        workers: int | None = None,
        kind: str | None = None,
        budget: ScreenItemBudget | None = None,
    ):
        self.family = family
        self.config_path = str(config_path)
        self.model_key = model_key
        self.llm = bool(llm)
        self.workers = max(1, int(config.SCREEN_WORKERS if workers is None else workers))
        self.batch_size = max(1, int(config.SCREEN_BATCH_SIZE))
        self.budget = budget if budget is not None else ScreenItemBudget(config.SCREEN_MAX_ITEMS)
        self.kind = kind or classify_family(family)
        self.prompt_path: Path | None = None
        self.system_prompt = ""
        self._vocab: set[str] = set()
        self._agent = None
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        self._schedule_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "records": 0,
            "targets": 0,
            "extractable": 0,
            "not_extractable": 0,
            "failed": 0,
            "true_records": 0,
            "false_records": 0,
            "error_records": 0,
            "limit_records": 0,
            "no_llm_records": 0,
            "cache_hits": 0,
            "requests": 0,
            "batches": 0,
            "items_sent": 0,
            "limit_skipped": 0,
            "invalid_items": 0,
        }
        self._limit_warned = False
        self._disabled_reason = ""

        if not self.llm:
            self._disabled_reason = "未启用 LLM 筛选（--no-llm-screen）"
            return
        if self.kind is None:
            self._disabled_reason = (
                f"无法从规则配置名 {family!r} 判断词类型（方位词/趋向动词），LLM 筛选已跳过；"
                "请用 --screen-kind 指定"
            )
            return

        name = prompt_name(self.kind)
        self.prompt_path = utils.prompt_path(name)
        self.system_prompt = utils.read_prompt_path(self.prompt_path)
        if not self.system_prompt.strip():
            self._disabled_reason = f"提示语尚未填写：{self.prompt_path}，LLM 筛选已跳过"
            return

        try:
            # 延迟 import 的理由同 _target_nodes：提示语为空时本模块不该拖入 sentenceFilters/torch
            import word_sampler

            self._vocab = word_sampler.derive_vocab(self.config_path, _TARGET_PREFIXES[self.kind])[0]
        except Exception as exc:  # noqa: BLE001 —— 提示语填了但词表推不出来：说清原因后跳过，不阻断语料构造
            self._disabled_reason = f"无法从 {self.config_path} 推导目标词表：{exc!r}，LLM 筛选已跳过"
            return
        if not self._vocab:
            self._disabled_reason = f"从 {self.config_path} 推导出的目标词表为空，LLM 筛选已跳过"

    @property
    def enabled(self) -> bool:
        return self.llm and self.kind is not None and bool(self.system_prompt.strip()) and bool(self._vocab)

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason

    def describe(self) -> str:
        if self.enabled:
            limit = self.budget.max_items if self.budget.max_items is not None else "不限"
            return (
                f"{self.family}（{self.kind}）：LLM 筛选已启用，提示语 {self.prompt_path}，"
                f"{self.workers} 线程，每批 {self.batch_size} 条，累计上限 {limit}"
            )
        return f"{self.family}：{self._disabled_reason}"

    # -- 模型 --------------------------------------------------------------
    def _model(self):
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    # 延迟 import：提示语为空时连 langchain 都不需要加载
                    from langchain.agents.structured_output import ToolStrategy

                    import agents as agent_factory

                    self._retry_policy = agent_factory.request_policy(self.model_key)
                    self._agent = agent_factory.get_agent(
                        self.model_key,
                        [],
                        self.system_prompt,
                        response_format=ToolStrategy(BatchExtractionVerdict),
                    )
        return self._agent

    def warm(self) -> None:
        """提示语已填时提前建模型，让配置错误在启动时暴露"""
        if self.enabled:
            self._model()

    def _item_content(
        self,
        sentence: str,
        target: dict,
        tree: Any,
        filter_methods: list[str],
        grounding: Any = None,
    ) -> str:
        """单条「句子 + 目标词」的完整判据文本（同时用作缓存键，与批组合无关）"""
        word = target.get("word", "")
        index = target.get("index")
        related = json.dumps(tree, ensure_ascii=False, sort_keys=True)
        methods = "、".join(str(item) for item in filter_methods or ()) or "（无）"
        role_line = ""
        if isinstance(grounding, dict):
            # 只把与当前待提取词同下标的那一条整理结果交给模型，避免同句其他词的
            # 角色信息干扰判定（多目标句子里这一点尤其重要）。
            candidates = [
                candidate
                for candidate in grounding.get("candidates") or []
                if isinstance(candidate, dict) and candidate.get("index") == index
            ]
            if candidates:
                role_line = (
                    "前序步骤整理的空间角色（JSON，目标物/参照物/动词；"
                    "趋向动词的动词允许为空）："
                    + json.dumps(candidates, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        return (
            f"本次语料类型：{self.kind}。\n"
            f"待提取的句子：{sentence}\n"
            f"待提取的词：{word}（分词结果里的下标 {index}）\n"
            f"前序步骤命中的规则：{methods}\n"
            f"前序步骤提取出的匹配树（JSON，含该词及其修饰/搭配成分）：{related}\n"
            f"{role_line}"
            "请判断这个句子里的这个待提取词是否能够被提取；只按结构化输出的字段作答："
            "extractable 为 true 或 false，reason 说明判断依据。"
        )

    def _content_batch(self, tasks: list[tuple[int, str, dict, Any, list[str], Any]]) -> str:
        """把多条判定项拼成一次请求；每条用 item_id 标注，便于按 id 映射结果"""
        lines = [
            f"本次语料类型：{self.kind}。",
            f"本次共 {len(tasks)} 条待判断语料。每条只判断自己的“待提取的词”，相互独立。",
            "返回结果必须为每条输入各一条：item_id 原样照抄，不能漏项、重复或新增 id。",
        ]
        for item_id, (_, sentence, target, tree, methods, grounding) in enumerate(tasks):
            lines.append("")
            lines.append(f"### 语料 {item_id}（item_id={item_id}）")
            lines.append(self._item_content(sentence, target, tree, methods, grounding))
        lines.append("")
        lines.append(
            "请把上述每条语料的判断结果放进 results：每个 item_id 恰好一条，"
            "extractable 为 true 或 false，reason 说明该条自己的判断依据。"
        )
        return "\n".join(lines)

    def _task_key(self, task: tuple[int, str, dict, Any, list[str], Any]) -> str:
        """单条判定的缓存键：不含 batch 编号，批组合变化不会降低命中率"""
        _, sentence, target, tree, methods, grounding = task
        return self._item_content(sentence, target, tree, methods, grounding)

    def _invoke(self, content: str):
        import agents as agent_factory

        agent = self._model()
        result = agent_factory.call_with_retry(
            lambda: agent.invoke({"messages": [{"role": "user", "content": content}]}),
            policy=self._retry_policy,
        )
        verdict = result.get("structured_response")
        if verdict is None:
            raise RuntimeError("模型没有返回结构化筛选结果（structured_response 为空）")
        return verdict

    def _ask_batch(self, tasks: list[tuple[int, str, dict, Any, list[str], Any]]) -> list[dict]:
        """发一次批量请求，按 item_id 映射回每条任务；缺项/重复/越界按 error 记"""
        try:
            verdict: BatchExtractionVerdict = self._invoke(self._content_batch(tasks))
            items = list(verdict.results)
        except Exception as exc:  # noqa: BLE001 —— 整批失败不带走语料，逐条 fail-open
            with self._stats_lock:
                self._stats["failed"] += len(tasks)
            return [
                {"extractable": None, "reason": f"调用失败：{exc!r}", "source": "error"}
                for _ in tasks
            ]

        by_id: dict[int, dict] = {}
        invalid = 0
        for item in items:
            item_id = item.item_id
            if not isinstance(item_id, int) or isinstance(item_id, bool) or not 0 <= item_id < len(tasks) or item_id in by_id:
                invalid += 1
                continue
            by_id[item_id] = {
                "extractable": bool(item.extractable),
                "reason": item.reason,
                "source": "llm",
            }

        missing = [item_id for item_id in range(len(tasks)) if item_id not in by_id]
        if invalid or missing:
            # 重复/越界/缺项都说明这份批量响应不可信：整批转 error、fail-open 保留，
            # 不能拿“第一个重复 id”的结果去丢弃语料，也不能把可疑结果写进缓存。
            with self._stats_lock:
                self._stats["invalid_items"] += invalid + len(missing)
            detail = []
            if invalid:
                detail.append(f"重复/越界 item_id {invalid} 条")
            if missing:
                detail.append(f"缺少 item_id {missing}")
            reason = "批量响应不完整（" + "；".join(detail) + "），整批按 fail-open 保留"
            return [{"extractable": None, "reason": reason, "source": "error"} for _ in tasks]

        return [by_id[item_id] for item_id in range(len(tasks))]

    def _cache_get(self, key: str) -> dict | None:
        with self._cache_lock:
            hit = self._cache.get(key)
        return dict(hit) if hit is not None else None

    def _cache_put(self, key: str, result: dict) -> None:
        if result.get("source") != "llm":
            return
        limit = max(1, int(config.SCREEN_CACHE_MAX))
        with self._cache_lock:
            if len(self._cache) >= limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = dict(result)

    def _resolve_tasks(self, tasks: list[tuple[int, str, dict, Any, list[str], Any]]) -> list[dict]:
        """查缓存 → 预占发送额度 → 组批并发请求 → 按任务顺序还原结果

        额度在**请求发出前**按批占用，且多套规则共用一份（见 ScreenItemBudget），
        因此并发时也不会超过 config.SCREEN_MAX_ITEMS。
        """
        if not tasks:
            return []
        with self._schedule_lock:
            keys = [self._task_key(task) for task in tasks]
            first_task: dict[str, tuple[int, str, dict, Any, list[str], Any]] = {}
            pending: list[str] = []
            resolved: dict[str, dict] = {}
            for key, task in zip(keys, tasks):
                if key in first_task:
                    continue
                first_task[key] = task
                hit = self._cache_get(key)
                if hit is not None:
                    resolved[key] = hit
                else:
                    pending.append(key)

            cache_hits = sum(1 for key in keys if key in resolved)
            if cache_hits:
                with self._stats_lock:
                    self._stats["cache_hits"] += cache_hits

            allowed_count = self.budget.reserve(len(pending))
            allowed = pending[:allowed_count]
            skipped = pending[allowed_count:]
            if allowed:
                with self._stats_lock:
                    self._stats["items_sent"] += len(allowed)
            if skipped:
                with self._stats_lock:
                    self._stats["limit_skipped"] += len(skipped)
                if not self._limit_warned:
                    self._limit_warned = True
                    print(
                        f"[LLM 筛选] {self.family}：累计发送已达上限 "
                        f"{self.budget.max_items} 条，后续语料不再送 LLM，按 fail-open 保留落盘"
                    )

            batches = [allowed[start : start + self.batch_size] for start in range(0, len(allowed), self.batch_size)]
            if batches:
                with self._stats_lock:
                    self._stats["requests"] += len(batches)
                    self._stats["batches"] += len(batches)
                batch_inputs = [[first_task[key] for key in batch_keys] for batch_keys in batches]
                total_batches = len(batch_inputs)
                total_items = sum(len(batch) for batch in batch_inputs)
                progress = {"done_batches": 0, "done_items": 0}

                def ask_with_progress(batch_tasks: list[tuple[int, str, dict, Any, list[str], Any]]) -> list[dict]:
                    """跑完一批后打印进度，再原样返回结果

                    进度行只读计数、不改 verdict/缓存/预算；即使 stdout 被关闭、管道断开，
                    或日志流由自定义对象接管后抛出普通异常，也吞掉输出异常，避免日志故障
                    反过来中断筛选。线程池的 map 按输入顺序返回结果，但本包装函数在工作
                    线程里完成一批就打印一批，因此慢批次不会拖住其他已完成批次的可见性。
                    """
                    results = self._ask_batch(batch_tasks)
                    with _PROGRESS_LOCK:
                        progress["done_batches"] += 1
                        progress["done_items"] += len(batch_tasks)
                        with self._stats_lock:
                            items_sent = self._stats["items_sent"]
                            cache_hits_total = self._stats["cache_hits"]
                        try:
                            print(
                                f"[LLM 筛选] {self.family}：批次完成 "
                                f"{progress['done_batches']}/{total_batches}，"
                                f"本轮完成 {progress['done_items']}/{total_items} 条，"
                                f"累计送判（已预占） {items_sent} 条，"
                                f"累计缓存命中 {cache_hits_total} 条",
                                flush=True,
                            )
                        except Exception:  # noqa: BLE001 —— 进度输出是附属产物，不能反过来
                            # 影响筛选结果：管道断开、stdout 已关闭、或被日志适配器接管后抛出
                            # 的普通异常都在这里止住；KeyboardInterrupt/SystemExit 仍会传播。
                            # 下一批仍会尝试打印。
                            pass
                    return results

                with ThreadPoolExecutor(max_workers=min(self.workers, len(batch_inputs))) as pool:
                    batch_results = list(pool.map(ask_with_progress, batch_inputs))
                for batch_keys, results in zip(batches, batch_results):
                    for key, result in zip(batch_keys, results):
                        resolved[key] = result
                        self._cache_put(key, result)

            if skipped:
                limit_result = {
                    "extractable": None,
                    "reason": (
                        f"已达 LLM 筛选发送上限 {self.budget.max_items} 条，本条未送模型；"
                        "按 fail-open 保留语料"
                    ),
                    "source": "limit",
                }
                for key in skipped:
                    resolved[key] = dict(limit_result)

            return [resolved[key] for key in keys]

    def _note_record_status(self, status: str) -> None:
        key = {
            SCREENING_TRUE: "true_records",
            SCREENING_FALSE: "false_records",
            SCREENING_ERROR: "error_records",
            SCREENING_LIMIT: "limit_records",
            SCREENING_NO_LLM: "no_llm_records",
        }.get(status)
        if key:
            with self._stats_lock:
                self._stats[key] += 1

    # -- 对外 --------------------------------------------------------------
    def screen_rows(self, rows: list[dict]) -> list[dict]:
        """对一批命中记录逐词筛选，返回**与输入等长**的记录（false 也保留）

        提示语为空/未启用时给每条补 no_llm 标注后原样返回；启用时每条记录新增
        ``screening.status``（true/false/error/limit/no_llm），供落盘与判定工作流准入。
        """
        if not rows:
            return rows
        if not self.enabled:
            with self._stats_lock:
                self._stats["records"] += len(rows)
            annotate_no_llm(rows, kind=self.kind, reason=self._disabled_reason or None)
            for _ in rows:
                self._note_record_status(SCREENING_NO_LLM)
            return rows
        with self._stats_lock:
            self._stats["records"] += len(rows)

        tasks: list[tuple[int, str, dict, Any, list[str], Any]] = []
        for row_index, row in enumerate(rows):
            sentence = _sentence_text(row)
            tree = row.get("match_tree") or []
            methods = list(row.get("filter_methods") or [])
            # 两类语料各有一个前序整理块；本行只带本类型的那一块。
            grounding = row.get("direction_grounding") or row.get("localization_grounding")
            targets = _target_nodes(row, self._vocab)
            if not targets:
                row["screening"] = {
                    "kind": self.kind,
                    "source": "no_target",
                    "status": SCREENING_ERROR,
                    "extractable": None,
                    "verdicts": [],
                    "reason": "前序步骤没有提取到属于本词表的目标词，未送 LLM",
                }
                self._note_record_status(SCREENING_ERROR)
                continue
            for target in targets:
                tasks.append((row_index, sentence, target, tree, methods, grounding))

        with self._stats_lock:
            self._stats["targets"] += len(tasks)

        results = self._resolve_tasks(tasks)
        by_row: dict[int, list[dict]] = {}
        for task, result in zip(tasks, results):
            row_index, _, target, _, _, _ = task
            verdict = {
                "index": target.get("index"),
                "word": target.get("word"),
                **result,
            }
            by_row.setdefault(row_index, []).append(verdict)
            with self._stats_lock:
                if verdict.get("extractable") is True:
                    self._stats["extractable"] += 1
                elif verdict.get("extractable") is False:
                    self._stats["not_extractable"] += 1

        kept: list[dict] = []
        for row_index, row in enumerate(rows):
            verdicts = by_row.get(row_index)
            if not verdicts:
                # 没有目标词（或这一行没进任务）：保留，避免因取词失败丢语料
                kept.append(row)
                continue
            true_indices = {v["index"] for v in verdicts if v.get("extractable") is True}
            false_indices = {v["index"] for v in verdicts if v.get("extractable") is False}
            # 判不出来（调用失败）按 fail-open 保留；裁剪 false 父节点时要把它当作要保留的
            # 后代，不能因为父节点被删就把它一起带走。
            unknown_indices = {v["index"] for v in verdicts if v.get("extractable") is None}
            sources = {v.get("source") for v in verdicts}
            if sources == {"llm"}:
                source = "llm"
            elif "llm" in sources:
                source = "mixed"
            else:
                source = next(iter(sources), "error")
            aggregate = True if true_indices else False if false_indices and len(false_indices) == len(verdicts) else None
            status = _status_from_verdicts(verdicts, aggregate)
            row["screening"] = {
                "kind": self.kind,
                "source": source,
                "status": status,
                "extractable": aggregate if status in (SCREENING_TRUE, SCREENING_FALSE) else None,
                "verdicts": verdicts,
            }
            # 只有可判定为 true 的记录才需要裁掉其中被判 false 的词；false/error/limit
            # 整条已不会进入判定工作流，保留原始 match_tree 便于复核。
            if status == SCREENING_TRUE and false_indices:
                row["match_tree"] = _prune_tree(
                    row.get("match_tree") or [], false_indices, true_indices | unknown_indices
                )
                try:
                    import direction_grounding
                    import localization_grounding

                    # 两类语料各有一个前序整理块；不存在的块会被模块内的形状检查直接跳过。
                    localization_grounding.prune_candidates(
                        row, false_indices, true_indices | unknown_indices
                    )
                    direction_grounding.prune_candidates(
                        row, false_indices, true_indices | unknown_indices
                    )
                except Exception:  # noqa: BLE001 —— 整理块缺失/形状异常不影响筛选主流程
                    pass
            self._note_record_status(status)
            kept.append(row)
        return kept

    def stats(self) -> dict:
        with self._stats_lock:
            stats = dict(self._stats)
        stats["enabled"] = self.enabled
        stats["workers"] = self.workers
        stats["batch_size"] = self.batch_size
        stats["budget"] = self.budget.stats()
        stats["limit_reached"] = self._limit_warned
        if self.kind:
            stats["kind"] = self.kind
        if self.prompt_path is not None:
            stats["prompt_file"] = str(self.prompt_path)
        if not self.enabled and self._disabled_reason:
            stats["disabled_reason"] = self._disabled_reason
        return stats


def build_screeners(
    generators: Iterable[Any],
    *,
    model_key: str,
    llm: bool = True,
    workers: int | None = None,
    kind: str | None = None,
) -> dict[str, SentenceScreener]:
    """按规则族建筛选器；键用 family_name，与输出文件一一对应

    ``generators`` 是 sentence_extract.py 里已建好的 ``sentenceFilters.FilterGenerator``：直接复用它们的
    family_name / config_path，避免再按输出文件名反推配置路径。
    """
    screeners: dict[str, SentenceScreener] = {}
    # 多套规则共用一份累计额度：上限是“本次跑批发送给 LLM 的语料条数”，不是每套规则各一份
    budget = ScreenItemBudget(config.SCREEN_MAX_ITEMS)
    for generator in generators:
        family = generator.family_name
        screeners[family] = SentenceScreener(
            family=family,
            config_path=generator.config_path,
            model_key=model_key,
            llm=llm,
            workers=workers,
            kind=kind,
            budget=budget,
        )
    return screeners
