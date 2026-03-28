# encoding: utf8

import json5
import abc
from typing import Sequence, Any
from dataclasses import dataclass, field

import config
from sentenceTable import SentenceTaskCache

cache = SentenceTaskCache(max_size=config.CACHE_MAX_SIZE)

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
# 单规则（核心）
# =========================

class SingleJudgeMethod(JudgeMethod):

    VOCAB = "_vocab"
    POS = "_pos"
    SYNTAX = "_syntax"
    SEMANTIC = "_semantic"
    LINK = "_link"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name)
        self.config = self._compile_config(config)

    def _compile_config(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        compiled: dict[str, Any] = {}
        for key, value in config_dict.items():
            if key == self.VOCAB:
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
        indices = self._search_indices(sentence, self.config)
        return bool(indices)

    def judge_with_indices(self, sentence: str):
        indices = self._search_indices(sentence, self.config)
        return bool(indices), indices

    # -------------------------
    # explain（树）
    # -------------------------

    def explain(self, sentence: str):
        result_map = self._search_with_explain(sentence, self.config)

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

    def _search_indices(self, sentence, config_dict, specific_idx=None):

        words = cache.get_task_value(sentence, config.CWS)

        vocab_indices = None
        pos_indices = None
        syntax_indices = None
        semantic_indices = None

        if self.VOCAB in config_dict:
            vocab_indices = {i for i, w in enumerate(words) if w in config_dict[self.VOCAB]}

        if self.POS in config_dict:
            pos_tags = cache.get_task_value(sentence, config.POS)
            pos_indices = {i for i, p in enumerate(pos_tags) if p == config_dict[self.POS]}

        if self.SYNTAX in config_dict:
            dep = cache.get_task_value(sentence, config.DEP)
            labels = dep["label"]
            heads = dep["head"]

            if specific_idx is not None:
                syntax_indices = {
                    i for i, (h, l) in enumerate(zip(heads, labels))
                    if l == config_dict[self.SYNTAX] and h - 1 == specific_idx
                }
            else:
                syntax_indices = {i for i, l in enumerate(labels) if l == config_dict[self.SYNTAX]}

        if self.SEMANTIC in config_dict:
            sdp = cache.get_task_value(sentence, config.SDP)
            labels = sdp["label"]
            heads = sdp["head"]

            if specific_idx is not None:
                semantic_indices = {
                    i for i, (h, l) in enumerate(zip(heads, labels))
                    if l == config_dict[self.SEMANTIC] and h - 1 == specific_idx
                }
            else:
                semantic_indices = {i for i, l in enumerate(labels) if l == config_dict[self.SEMANTIC]}

        sets = [s for s in [vocab_indices, pos_indices, syntax_indices, semantic_indices] if s is not None]

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
                    if not self._search_indices(sentence, cond, idx):
                        ok = False
                        break
                if ok:
                    filtered.add(idx)
            return filtered

        return result

    # -------------------------
    # 🌳 核心：树结构解释
    # -------------------------

    def _search_with_explain(self, sentence, config_dict, specific_idx=None):

        words = cache.get_task_value(sentence, config.CWS)
        result = {}

        vocab_indices = None
        pos_indices = None
        syntax_indices = None
        semantic_indices = None

        if self.VOCAB in config_dict:
            vocab_indices = {i for i, w in enumerate(words) if w in config_dict[self.VOCAB]}

        if self.POS in config_dict:
            pos_tags = cache.get_task_value(sentence, config.POS)
            pos_indices = {i for i, p in enumerate(pos_tags) if p == config_dict[self.POS]}

        if self.SYNTAX in config_dict:
            dep = cache.get_task_value(sentence, config.DEP)
            labels = dep["label"]
            heads = dep["head"]

            if specific_idx is not None:
                syntax_indices = {
                    i for i, (h, l) in enumerate(zip(heads, labels))
                    if l == config_dict[self.SYNTAX] and h - 1 == specific_idx
                }
            else:
                syntax_indices = {i for i, l in enumerate(labels) if l == config_dict[self.SYNTAX]}

        if self.SEMANTIC in config_dict:
            sdp = cache.get_task_value(sentence, config.SDP)
            labels = sdp["label"]
            heads = sdp["head"]

            if specific_idx is not None:
                semantic_indices = {
                    i for i, (h, l) in enumerate(zip(heads, labels))
                    if l == config_dict[self.SEMANTIC] and h - 1 == specific_idx
                }
            else:
                semantic_indices = {i for i, l in enumerate(labels) if l == config_dict[self.SEMANTIC]}

        sets = [s for s in [vocab_indices, pos_indices, syntax_indices, semantic_indices] if s is not None]

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
                    "_vocab": idx in vocab_indices if vocab_indices else None,
                    "_pos": idx in pos_indices if pos_indices else None,
                    "_syntax": idx in syntax_indices if syntax_indices else None,
                    "_semantic": idx in semantic_indices if semantic_indices else None,
                },
                children=[]
            )

        if self.LINK in config_dict:
            filtered = {}

            for idx, node in result.items():
                children = []
                ok = True

                for subname, cond in config_dict[self.LINK].items():
                    sub_res = self._search_with_explain(sentence, cond, idx)

                    if not sub_res:
                        ok = False
                        break

                    children.extend(sub_res.values())

                if ok:
                    node.children = children
                    filtered[idx] = node

            return filtered

        return result


# =========================
# 生成器
# =========================

class FilterGenerator:

    LOGIC = "_logic"

    def __init__(self, config_path: str):
        self.config_path = config_path

    def __generate(self, method_data: dict, name: str):

        if any(k in method_data for k in ["_vocab", "_pos", "_syntax", "_semantic"]):
            return SingleJudgeMethod(name, method_data)

        logic = method_data.get(self.LOGIC, "or")

        sub = {k: v for k, v in method_data.items() if not k.startswith("_")}

        methods = [self.__generate(v, k) for k, v in sub.items()]

        if logic == "and":
            return AndJudgeMethod(name, methods)
        else:
            return OrJudgeMethod(name, methods)

    def generate_filter(self):

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = json5.load(f)

        return self.__generate(config_data, "root")


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
    fg = FilterGenerator(r"config_files\direction_expressions.json5")
    f = fg.generate_filter()

    s = "魏仲时这时从里间的重症病房走出来,又走进了病房。"

    print(f.judge(s))
    print(f.judge_with_indices(s))
    print(f.explain(s))

    ok, res = f.explain(s)

    print(ok)
    print(res["indices"])

    for node in res["match_tree"]:
        print_tree(node)