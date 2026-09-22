# encoding: utf8

import json5
import abc
from pathlib import Path
from typing import Sequence, Any, Iterable
from dataclasses import dataclass, field
from collections import defaultdict

import config
from sentenceTable import SentenceTaskCache

cache = SentenceTaskCache(max_size=config.CACHE_MAX_SIZE)


def load_vocab_file(path: str, base_dir: str | Path | None = None) -> set[str]:
    """加载 JSON5 词表文件（内容为词条列表）

    解析顺序：base_dir（通常传入规则配置文件所在目录）→ 项目根目录 → 当前工作目录。
    供规则配置的 "_vocab_file" 保留字使用，使词表只有一处真源。

    Args:
        path (str): 词表文件路径，可为相对路径
        base_dir (str | Path | None): 首选的相对路径基准目录

    Returns:
        set[str]: 去重后的词条集合
    """
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(Path(base_dir) / path)
    if Path(path).is_absolute():
        candidates.append(Path(path))
    else:
        candidates.append(Path(config._PROJECT_ROOT) / path)
        candidates.append(Path.cwd() / path)

    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as f:
                words = json5.load(f)
            if not isinstance(words, list):
                raise ValueError(
                    f"词表文件 {candidate} 的内容必须是词条列表（JSON5 数组），实际是 {type(words).__name__}"
                )
            return set(words)

    raise FileNotFoundError(f"找不到词表文件 {path}，已尝试：{[str(c) for c in candidates]}")


def _as_tag_tuple(value: Any) -> tuple:
    """把保留字的取值统一成元组

    "_pos"/"_syntax"/"_semantic" 既接受单个标签（"nd"），也接受标签列表（["n", "r"]），
    列表表示“或”，即多个标签的索引集合取并集。
    """
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def substring_minimize(words: Iterable[str]) -> set[str]:
    """去掉子串冗余的词

    初筛用的是子串匹配（`word in sentence`），若某个词 w 含有集合内另一个更短的词 k，
    则用 k 初筛即可覆盖 w（k in s ⟹ w in s），w 属于冗余条目，删掉可减少初筛开销。
    """
    kept: list[str] = []
    for word in sorted(set(words), key=len):
        if not any(k in word for k in kept):
            kept.append(word)
    return set(kept)


_substring_minimize = substring_minimize
"""兼容旧调用名"""

# =========================
# 匹配树节点
# =========================

@dataclass
class MatchNode:
    name: str
    index: int
    word: str
    conditions: dict
    children: list["MatchNode"] = field(default_factory=list)


# =========================
# 抽象基类
# =========================

class JudgeMethod(abc.ABC):

    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def judge(self, sentence: str) -> bool:
        """判断句子是否符合判断条件的要求

        注意：生产筛选链路只走 explain（其真值判定与 judge 严格等价，且额外提供匹配树）。
        judge 仅用于调试与自测，修改判定逻辑时必须与 explain 同步。

        Args:
            sentence (str): 待判断的句子

        Returns:
            bool: 句子是否符合判断条件要求
        """
        pass

    def judge_with_indices(self, sentence: str) -> tuple[bool, set[int]]:
        """判断句子是否符合判断条件的要求，并返回满足条件的词语索引

        Args:
            sentence (str): 待判断的句子
        
        Returns:
            Tuple[bool, Set[int]]: 句子是否符合判断条件要求，以及满足条件的词语索引集合
        """
        return self.judge(sentence), set()

    def explain(self, sentence: str) -> tuple[bool, dict]:
        """解释句子是否符合判断条件的要求
        Args:
            sentence (str): 待判断的句子
        Returns:
            Tuple[bool, dict]: 句子是否符合判断条件要求，以及解释信息的字典，包含以下字段：
                - matched (bool): 句子是否符合判断条件要求
                - indices (Set[int]): 满足条件的词语索引集合
                - words (List[str]): 满足条件的词语列表
                - methods (List[str]): 参与判断的规则方法名称列表
                - match_tree (List[MatchNode]): 满足条件的词语对应的匹配树节点列表
        """        
        ok = self.judge(sentence)
        return ok, {
            "matched": ok,
            "indices": set(),
            "words": [],
            "methods": [self.name],
            "match_tree": []
        }

    def get_method_names(self):
        return [self.name]


# =========================
# AND 逻辑
# =========================

class AndJudgeMethod(JudgeMethod):

    def __init__(self, name: str, judge_methods: Sequence[JudgeMethod]):
        super().__init__(name)
        self.judge_methods = judge_methods

    def judge(self, sentence: str) -> bool:
        return all(m.judge(sentence) for m in self.judge_methods)

    def judge_with_indices(self, sentence: str):
        sets = []
        for m in self.judge_methods:
            ok, indices = m.judge_with_indices(sentence)
            if not ok:
                return False, set()
            sets.append(indices)
        return True, set.intersection(*sets) if sets else set()

    def explain(self, sentence: str):
        all_nodes = {}
        methods = [self.name]

        for m in self.judge_methods:
            ok, res = m.explain(sentence)
            if not ok:
                return False, {
                    "matched": False,
                    "indices": set(),
                    "words": [],
                    "methods": [],
                    "match_tree": []
                }

            methods.extend(res["methods"])

            for node in res["match_tree"]:
                if node.index not in all_nodes:
                    all_nodes[node.index] = node
                else:
                    all_nodes[node.index].children.extend(node.children)

        return True, {
            "matched": True,
            "indices": set(all_nodes.keys()),
            "words": [n.word for n in all_nodes.values()],
            "methods": methods,
            "match_tree": list(all_nodes.values())
        }


# =========================
# OR 逻辑
# =========================

class OrJudgeMethod(JudgeMethod):

    def __init__(self, name: str, judge_methods: Sequence[JudgeMethod]):
        super().__init__(name)
        self.judge_methods = judge_methods

    def judge(self, sentence: str) -> bool:
        return any(m.judge(sentence) for m in self.judge_methods)

    def judge_with_indices(self, sentence: str):
        result = set()
        for m in self.judge_methods:
            ok, indices = m.judge_with_indices(sentence)
            if ok:
                result.update(indices)
                return True, result
        return False, set()

    def explain(self, sentence: str):
        methods = [self.name]

        for m in self.judge_methods:
            ok, res = m.explain(sentence)
            if ok:
                methods.extend(res["methods"])
                return True, {
                    "matched": True,
                    "indices": res["indices"],
                    "words": res["words"],
                    "methods": methods,
                    "match_tree": res["match_tree"]
                }

        return False, {
            "matched": False,
            "indices": set(),
            "words": [],
            "methods": [],
            "match_tree": []
        }


# =========================
# OR_ALL 逻辑（保留全部命中分支）
# =========================

class OrAllJudgeMethod(JudgeMethod):
    """或（保留全部命中分支）

    真值判定与 OrJudgeMethod 完全等价（任一子条件命中即命中），区别只在于 explain：
    OrJudgeMethod 命中第一个子条件就返回，句子同时满足多条规则时只会记录其中一条；
    本类会遍历全部子条件，把命中分支的方法名与匹配树合并输出。
    适合“同一批句子要按多条并存规则分别统计”的场景（例如方位短语与介词宾语常同时成立），
    这样跑一次即可，事后按 filter_methods 切分子集，不必重跑。

    注意：judge 与 explain 的真值必须保持一致（本类的 judge 为 any 语义，两者等价）。
    """

    def __init__(self, name: str, judge_methods: Sequence[JudgeMethod]):
        super().__init__(name)
        self.judge_methods = judge_methods

    def judge(self, sentence: str) -> bool:
        return any(m.judge(sentence) for m in self.judge_methods)

    def judge_with_indices(self, sentence: str):
        matched = False
        result: set[int] = set()
        for m in self.judge_methods:
            ok, indices = m.judge_with_indices(sentence)
            if ok:
                matched = True
                result.update(indices)
        return matched, result

    def explain(self, sentence: str):
        all_nodes = {}
        methods = [self.name]

        for m in self.judge_methods:
            ok, res = m.explain(sentence)
            if not ok:
                continue

            methods.extend(res["methods"])

            # 与 AndJudgeMethod 相同的合并方式：同一索引的节点合并子节点
            for node in res["match_tree"]:
                if node.index not in all_nodes:
                    all_nodes[node.index] = node
                else:
                    all_nodes[node.index].children.extend(node.children)

        if not all_nodes:
            return False, {
                "matched": False,
                "indices": set(),
                "words": [],
                "methods": [],
                "match_tree": []
            }

        return True, {
            "matched": True,
            "indices": set(all_nodes.keys()),
            "words": [n.word for n in all_nodes.values()],
            "methods": methods,
            "match_tree": list(all_nodes.values())
        }


# =========================
# 单规则（核心）
# =========================

class SingleJudgeMethod(JudgeMethod):

    VOCAB = "_vocab"
    VOCAB_FILE = "_vocab_file"
    POS = "_pos"
    SYNTAX = "_syntax"
    SEMANTIC = "_semantic"
    LINK = "_link"

    BASE_KEYS = (VOCAB, VOCAB_FILE, POS, SYNTAX, SEMANTIC)

    def __init__(self, name: str, config: dict[str, Any], base_dir: str | Path | None = None):
        super().__init__(name)
        self.base_dir = base_dir
        self.config = self._compile_config(config)

    def _compile_config(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        compiled: dict[str, Any] = {}
        for key, value in config_dict.items():
            if key == self.VOCAB_FILE:
                # 常规路径由 FilterGenerator 在加载时解析（相对基目录为规则配置文件所在目录），
                # 这里兜底处理直接构造 SingleJudgeMethod 的场景
                if self.VOCAB in config_dict:
                    raise ValueError(f"规则 {self.name} 不能同时指定 {self.VOCAB} 与 {self.VOCAB_FILE}")
                compiled[self.VOCAB] = load_vocab_file(value, self.base_dir)
            elif key == self.VOCAB:
                compiled[key] = set(value)
            elif key == self.LINK and isinstance(value, dict):
                compiled[key] = {name: self._compile_config(cond) for name, cond in value.items()}
            else:
                compiled[key] = value
        return compiled

    # -------------------------
    # 原 judge
    # -------------------------

    def judge(self, sentence: str) -> bool:
        # 与 _search_with_explain_with_context 是两份等价实现，生产链路只走 explain，
        # 修改任一处的检索语义时必须同步另一处
        context = self._build_sentence_context(sentence)
        indices = self._search_indices_with_context(context, self.config)
        return bool(indices)

    def judge_with_indices(self, sentence: str):
        context = self._build_sentence_context(sentence)
        indices = self._search_indices_with_context(context, self.config)
        return bool(indices), indices

    # -------------------------
    # explain（树）
    # -------------------------

    def explain(self, sentence: str):
        context = self._build_sentence_context(sentence)
        result_map = self._search_with_explain_with_context(context, self.config)

        return (
            bool(result_map),
            {
                "matched": bool(result_map),
                "indices": set(result_map.keys()),
                "words": [n.word for n in result_map.values()],
                "methods": [self.name],
                "match_tree": list(result_map.values())
            }
        )

    # -------------------------
    # 原逻辑（保持）
    # -------------------------

    def _build_sentence_context(self, sentence: str) -> dict[str, Any]:
        words = cache.get_task_value(sentence, config.CWS)
        pos_tags = cache.get_task_value(sentence, config.POS)
        dep = cache.get_task_value(sentence, config.DEP)
        sdp = cache.get_task_value(sentence, config.SDP)

        word_indices_map: dict[str, set[int]] = defaultdict(set)
        for idx, w in enumerate(words):
            word_indices_map[w].add(idx)

        pos_indices_map: dict[str, set[int]] = defaultdict(set)
        for idx, p in enumerate(pos_tags):
            pos_indices_map[p].add(idx)

        dep_label_indices_map: dict[str, set[int]] = defaultdict(set)
        dep_child_map: dict[tuple[int, str], set[int]] = defaultdict(set)
        for idx, (head, label) in enumerate(zip(dep["head"], dep["label"])):
            dep_label_indices_map[label].add(idx)
            dep_child_map[(head - 1, label)].add(idx)

        sdp_label_indices_map: dict[str, set[int]] = defaultdict(set)
        sdp_child_map: dict[tuple[int, str], set[int]] = defaultdict(set)
        for idx, (head, label) in enumerate(zip(sdp["head"], sdp["label"])):
            sdp_label_indices_map[label].add(idx)
            sdp_child_map[(head - 1, label)].add(idx)

        return {
            "words": words,
            "word_indices_map": word_indices_map,
            "pos_indices_map": pos_indices_map,
            "dep_label_indices_map": dep_label_indices_map,
            "dep_child_map": dep_child_map,
            "sdp_label_indices_map": sdp_label_indices_map,
            "sdp_child_map": sdp_child_map,
        }

    def _get_vocab_indices(self, context: dict[str, Any], vocab_set: set[str]) -> set[int]:
        word_indices_map = context["word_indices_map"]
        result: set[int] = set()
        # 必须按固定顺序遍历：set[str] 的迭代顺序受 PYTHONHASHSEED 随机化影响，
        # 而此处写入 result 的顺序会决定该 int 集合的内部布局，进而影响
        # final_indices 的迭代顺序与 match_tree 的节点顺序——同一份输入会在不同进程
        # 得到内容相同但顺序不同的 match_tree（判定结果不受影响，但产物无法逐字节比对）。
        for vocab in sorted(vocab_set):
            result.update(word_indices_map.get(vocab, set()))
        return result

    def _collect_base_indices(self, context: dict[str, Any], config_dict: dict[str, Any], specific_idx=None) -> dict[str, set[int] | None]:
        vocab_indices = None
        pos_indices = None
        syntax_indices = None
        semantic_indices = None

        if self.VOCAB in config_dict:
            vocab_indices = self._get_vocab_indices(context, config_dict[self.VOCAB])

        if self.POS in config_dict:
            pos_indices = set()
            for pos_tag in _as_tag_tuple(config_dict[self.POS]):
                pos_indices |= context["pos_indices_map"].get(pos_tag, set())

        if self.SYNTAX in config_dict:
            syntax_indices = set()
            for syntax_label in _as_tag_tuple(config_dict[self.SYNTAX]):
                if specific_idx is None:
                    syntax_indices |= context["dep_label_indices_map"].get(syntax_label, set())
                else:
                    # 带 specific_idx 时表示“本词（specific_idx）的依存标签为该标签的子节点”
                    syntax_indices |= context["dep_child_map"].get((specific_idx, syntax_label), set())

        if self.SEMANTIC in config_dict:
            semantic_indices = set()
            for semantic_label in _as_tag_tuple(config_dict[self.SEMANTIC]):
                if specific_idx is None:
                    semantic_indices |= context["sdp_label_indices_map"].get(semantic_label, set())
                else:
                    semantic_indices |= context["sdp_child_map"].get((specific_idx, semantic_label), set())

        return {
            self.VOCAB: vocab_indices,
            self.POS: pos_indices,
            self.SYNTAX: syntax_indices,
            self.SEMANTIC: semantic_indices,
        }

    def _search_indices_with_context(self, context: dict[str, Any], config_dict: dict[str, Any], specific_idx=None):

        base_indices = self._collect_base_indices(context, config_dict, specific_idx)
        sets = [s for s in base_indices.values() if s is not None]

        if not sets:
            return set()

        result = set.intersection(*sets)
        if not result:
            return set()

        if self.LINK in config_dict:
            filtered = set()
            for idx in result:
                ok = True
                for _, cond in config_dict[self.LINK].items():
                    if not self._search_indices_with_context(context, cond, idx):
                        ok = False
                        break
                if ok:
                    filtered.add(idx)
            return filtered

        return result

    def _search_indices(self, sentence, config_dict, specific_idx=None):
        context = self._build_sentence_context(sentence)
        return self._search_indices_with_context(context, config_dict, specific_idx)

    # -------------------------
    # 🌳 核心：树结构解释
    # -------------------------

    def _search_with_explain_with_context(self, context: dict[str, Any], config_dict: dict[str, Any], specific_idx=None):

        words = context["words"]
        result = {}
        base_indices = self._collect_base_indices(context, config_dict, specific_idx)
        sets = [s for s in base_indices.values() if s is not None]

        if not sets:
            return {}

        final_indices = set.intersection(*sets)
        if not final_indices:
            return {}

        for idx in final_indices:
            result[idx] = MatchNode(
                name=self.name,
                index=idx,
                word=words[idx],
                conditions={
                    "_vocab": idx in base_indices[self.VOCAB] if base_indices[self.VOCAB] is not None else None,
                    "_pos": idx in base_indices[self.POS] if base_indices[self.POS] is not None else None,
                    "_syntax": idx in base_indices[self.SYNTAX] if base_indices[self.SYNTAX] is not None else None,
                    "_semantic": idx in base_indices[self.SEMANTIC] if base_indices[self.SEMANTIC] is not None else None,
                },
                children=[]
            )

        if self.LINK in config_dict:
            filtered = {}

            for idx, node in result.items():
                children = []
                ok = True

                for _, cond in config_dict[self.LINK].items():
                    sub_res = self._search_with_explain_with_context(context, cond, idx)

                    if not sub_res:
                        ok = False
                        break

                    children.extend(sub_res.values())

                if ok:
                    node.children = children
                    filtered[idx] = node

            return filtered

        return result

    def _search_with_explain(self, sentence, config_dict, specific_idx=None):
        context = self._build_sentence_context(sentence)
        return self._search_with_explain_with_context(context, config_dict, specific_idx)


# =========================
# 生成器
# =========================

class FilterGenerator:

    LOGIC = "_logic"
    LOGIC_VALUES = ("or", "or_all", "and")

    VOCAB = SingleJudgeMethod.VOCAB
    VOCAB_FILE = SingleJudgeMethod.VOCAB_FILE
    LINK = SingleJudgeMethod.LINK
    BASE_KEYS = SingleJudgeMethod.BASE_KEYS

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_dir = Path(config_path).resolve().parent
        self._config_data: dict[str, Any] | None = None

    @property
    def family_name(self) -> str:
        """规则族名称：取配置文件名（不含扩展名），用于输出文件命名与日志"""
        return Path(self.config_path).stem

    # -------------------------
    # 加载与预处理
    # -------------------------

    def _resolve_vocab_files(self, node: Any) -> Any:
        """把 "_vocab_file" 就地展开为 "_vocab"

        这样规则编译与初筛推导共用同一份已解析的配置，避免两处各读一次文件而出现不一致。
        相对路径以规则配置文件所在目录为基准。
        """
        if isinstance(node, dict):
            if self.VOCAB_FILE in node:
                if self.VOCAB in node:
                    raise ValueError(f"规则节点不能同时指定 {self.VOCAB} 与 {self.VOCAB_FILE}：{node[self.VOCAB_FILE]}")
                words = load_vocab_file(node[self.VOCAB_FILE], self.config_dir)
                node = {k: v for k, v in node.items() if k != self.VOCAB_FILE}
                node[self.VOCAB] = sorted(words)
            return {k: self._resolve_vocab_files(v) for k, v in node.items()}
        if isinstance(node, list):
            return [self._resolve_vocab_files(item) for item in node]
        return node

    def load_config(self) -> dict[str, Any]:
        """读取并预处理规则配置（进程内缓存，只读一次文件）"""
        if self._config_data is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json5.load(f)
            if not isinstance(config_data, dict):
                raise ValueError(
                    f"筛选配置文件 {self.config_path} 的顶层必须是字典（规则集合），"
                    f"实际是 {type(config_data).__name__}。"
                    f"若这是词表文件，请用 -v 传入或写成规则里的 {self.VOCAB_FILE}"
                )
            self._config_data = self._resolve_vocab_files(config_data)
        return self._config_data

    # -------------------------
    # 生成判定树
    # -------------------------

    def __generate(self, method_data: Any, name: str):

        if not isinstance(method_data, dict):
            raise ValueError(f"规则节点 {name!r} 必须是字典，实际是 {type(method_data).__name__}")

        sub = {k: v for k, v in method_data.items() if not k.startswith("_")}

        if any(k in method_data for k in self.BASE_KEYS):
            if sub:
                # 顶层误写 _vocab 会让整份配置塌缩成一条规则，把下面所有命名规则静默丢掉，
                # 这里直接报错而不是让它悄悄发生
                raise ValueError(
                    f"规则节点 {name!r} 同时含有保留字（{self.VOCAB}/{self.VOCAB_FILE}/...）与子条件 "
                    f"{list(sub)}，会被解释成单条规则从而使子条件失效；请把子条件放进独立规则，"
                    f"或把保留字写进每个子规则"
                )
            return SingleJudgeMethod(name, method_data, base_dir=self.config_dir)

        if not sub:
            raise ValueError(
                f"规则节点 {name!r} 既没有 {self.VOCAB}/{self.VOCAB_FILE}/_pos/_syntax/_semantic，"
                f"也没有子条件，永远不会命中（常见原因：条件只写了 _link）"
            )

        logic = method_data.get(self.LOGIC, "or")
        if logic not in self.LOGIC_VALUES:
            raise ValueError(
                f"规则节点 {name!r} 的 {self.LOGIC} 取值 {logic!r} 不被支持，仅支持 {list(self.LOGIC_VALUES)}"
            )

        methods = [self.__generate(v, k) for k, v in sub.items()]

        if logic == "and":
            return AndJudgeMethod(name, methods)
        if logic == "or_all":
            return OrAllJudgeMethod(name, methods)
        return OrJudgeMethod(name, methods)

    def generate_filter(self):
        return self.__generate(self.load_config(), "root")

    # -------------------------
    # 初筛词表推导
    # -------------------------

    def collect_rule_screens(self) -> list[tuple[str, set[str] | None]]:
        """按规则结构逐条推导初筛词表

        原理：规则命中要求其**所有**合取项（自身的词表与每个 _link 子条件）同时成立，
        因此任一含词表的合取项都可单独充当初筛，取其中最小者即可。
        再做子串最小化去掉冗余词。

        正确性：规则用的是 CWS 分词后的精确匹配，句子若被某规则命中，必然含该规则某合取项的
        词作为完整词，因而必含该词作为子串，从而必过由该合取项派生的初筛。
        所以派生初筛是**超集过滤**：只影响速度，不会漏掉任何命中句。

        Returns:
            list[tuple[str, set[str] | None]]: [(规则名, 该规则的最小可用初筛词表)]，
                词表为 None 表示这条规则的条件里根本没有词表（例如只写了 _pos），
                此时任何初筛都可能漏掉该规则的命中句，调用方应放弃初筛。
        """
        screens: list[tuple[str, set[str]]] = []
        unscreenable: list[str] = []
        self._walk_screens(self.load_config(), "root", screens, unscreenable)
        return screens + [(name, None) for name in unscreenable]

    def collect_screen_vocab(self) -> tuple[set[str] | None, list[str]]:
        """把逐条规则推导的结果合并为一份初筛词表

        Returns:
            tuple[set[str] | None, list[str]]: (初筛词表, 推导不出初筛的规则名列表)；
                只要有规则推导不出词表，第一条就返回 None，调用方应放弃初筛。
        """
        rule_screens = self.collect_rule_screens()
        unscreenable = [name for name, screen in rule_screens if screen is None]
        if unscreenable:
            return None, unscreenable

        screens = [screen for _, screen in rule_screens if screen]
        if not screens:
            return set(), []

        return substring_minimize(set().union(*screens)), []

    def _walk_screens(
        self,
        node: Any,
        name: str,
        screens: list[tuple[str, set[str]]],
        unscreenable: list[str],
    ) -> None:
        if not isinstance(node, dict):
            unscreenable.append(name)
            return

        if any(k in node for k in self.BASE_KEYS):
            screen = self._leaf_screen(node)
            if screen is None:
                unscreenable.append(name)
            else:
                screens.append((name, screen))
            return

        children = [(k, v) for k, v in node.items() if not k.startswith("_")]
        if not children:
            unscreenable.append(name)
            return

        # 各子规则之间是“或”关系，初筛取并集是安全的超集
        for key, value in children:
            self._walk_screens(value, key, screens, unscreenable)

    def _leaf_screen(self, node: dict[str, Any]) -> set[str] | None:
        """单条规则的最小可用初筛词表；推导不出时返回 None"""
        candidates: list[set[str]] = []

        if self.VOCAB in node:
            candidates.append(substring_minimize(node[self.VOCAB]))

        link = node.get(self.LINK)
        if isinstance(link, dict):
            for cond in link.values():
                if isinstance(cond, dict):
                    sub = self._leaf_screen(cond)
                    if sub:
                        candidates.append(sub)

        if not candidates:
            return None

        return min(candidates, key=len)


# =========================
# 调试打印树
# =========================

def print_tree(node: MatchNode, indent=0):
    print("  " * indent + f"{node.word} ({node.name})")
    for c in node.children:
        print_tree(c, indent + 1)

def simplify_tree(tree: list[MatchNode]) -> list[dict[str, Any]]:
    """将 MatchNode 树简化为字典列表，方便输出和查看
    
    Args:
        tree (list[MatchNode]): MatchNode 树

    Returns:
        list[dict[str, Any]]: 简化后的字典列表
    """

    simplified = []
    for node in tree:
        simplified.append({
            # "name": node.name,
            "index": node.index,
            "word": node.word,
            # "conditions": node.conditions,
            "children": simplify_tree(node.children)
        })
    return simplified
    
# =========================
# 测试
# =========================

if __name__ == "__main__":
    # 自测：python sentenceFilters.py（需要 LTP 模型，秒级）
    # 注意：路径基于 __file__ 解析，与运行时 cwd 无关（旧写法用 Windows 反斜杠，在 Linux 下必然失败）
    config_dir = Path(__file__).resolve().parent / "config_files"
    direction_config = config_dir / "direction_expressions.json5"
    locality_config = config_dir / "locality_expressions.json5"
    failures: list[str] = []

    def check(desc: str, condition: bool) -> None:
        print(f"  [{'OK' if condition else 'FAIL'}] {desc}")
        if not condition:
            failures.append(desc)

    def tree_words(nodes: list[MatchNode]) -> set[str]:
        """匹配树里只有规则命中的核心词在顶层，其余命中词挂在 children 上，需递归收集"""
        words: set[str] = set()
        for node in nodes:
            words.add(node.word)
            words |= tree_words(node.children)
        return words

    # ---- 1. 趋向动词规则（回归）----
    print("趋向动词规则：")
    direction_generator = FilterGenerator(str(direction_config))
    direction_filter = direction_generator.generate_filter()
    s = "魏仲时这时从里间的重症病房走出来,又走进了病房。"
    check("命中动趋式例句", direction_filter.judge(s))
    ok, res = direction_filter.explain(s)
    check("explain 与 judge 真值一致", ok == direction_filter.judge(s))
    check("命中词含“出来”“进”", {"出来", "进"} <= tree_words(res["match_tree"]))
    print(f"  filter_methods = {res['methods']}")
    for node in res["match_tree"]:
        print_tree(node)

    # ---- 2. 初筛词表自动派生：必须与手写词表一致 ----
    print("初筛词表派生：")
    derived, unscreenable = direction_generator.collect_screen_vocab()
    manual = set(json5.load(open(config_dir / "direction_vocab.json5", encoding="utf-8")))
    check("无推导不出初筛的规则", not unscreenable)
    check(f"派生词表 {sorted(derived or [])} == 手写词表 {sorted(manual)}", derived == manual)

    # ---- 3. 方位词规则 ----
    print("方位词规则：")
    locality_filter = FilterGenerator(str(locality_config)).generate_filter()

    should_match = [
        "他把书放在桌子上。",
        "屋子里面很暖和。",
        "村子东边有一条小河,西边是山坡。",
        "他站在窗户前面,望着远处的群山。",
        "从门缝里往外看,街上空无一人。",
        "他往后退了两步,然后向左转。",
        "她的房间在三楼,我的在下面。",
        "桌子上摆着一本翻开的书。",
        "我在图书馆里查了一下午的资料。",
    ]
    should_not_match = [
        "经过三个月的努力,他终于成功了。",        # 经过：不是方位词
        "这就是问题的关键所在。",                  # 所在：不是方位词
        "会议由王主任主持,下午继续进行。",          # 下午：LTP 切为一个词，不等于词表里的“下”
        "他中了一次奖,高兴了好几天。",              # 中：此处是动词（中奖），非方位词
    ]
    for sentence in should_match:
        check(f"命中：{sentence}", locality_filter.judge(sentence))
    for sentence in should_not_match:
        check(f"排除：{sentence}", not locality_filter.judge(sentence))

    # ---- 4. or_all：同句命中多条规则时全部记录 ----
    multi = "从门缝里往外看,街上空无一人。"
    ok, res = locality_filter.explain(multi)
    methods = set(res["methods"])
    print(f"  or_all 例句 {multi} -> filter_methods = {res['methods']}")
    check("同时记录“方位短语”与“介词+方位词”两条规则",
          {"locality_phrase", "locality_as_prep_object"} <= methods)

    # ---- 5. 多值 _pos（["n", "r"] 之类）不再报 unhashable ----
    multi_pos = SingleJudgeMethod("probe", {
        "_vocab": ["之间"], "_pos": "nd",
        "_link": {"m": {"_syntax": "ATT", "_pos": ["n", "r"]}},
    })
    check("多值 _pos 可用（我们之间 = 代词 + 方位词）", multi_pos.judge("我们之间没有秘密。"))

    print()
    if failures:
        print(f"自测失败 {len(failures)} 项：")
        for desc in failures:
            print(f"  - {desc}")
        raise SystemExit(1)
    print("全部自测通过")