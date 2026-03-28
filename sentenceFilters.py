# encoding: utf8

import json5
import abc
from typing import Sequence, Any
from dataclasses import dataclass, field
from collections import defaultdict

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
        for vocab in vocab_set:
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
            pos_indices = context["pos_indices_map"].get(config_dict[self.POS], set())

        if self.SYNTAX in config_dict:
            syntax_label = config_dict[self.SYNTAX]
            if specific_idx is None:
                syntax_indices = context["dep_label_indices_map"].get(syntax_label, set())
            else:
                syntax_indices = context["dep_child_map"].get((specific_idx, syntax_label), set())

        if self.SEMANTIC in config_dict:
            semantic_label = config_dict[self.SEMANTIC]
            if specific_idx is None:
                semantic_indices = context["sdp_label_indices_map"].get(semantic_label, set())
            else:
                semantic_indices = context["sdp_child_map"].get((specific_idx, semantic_label), set())

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