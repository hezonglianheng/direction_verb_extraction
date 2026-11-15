# encoding: utf8

"""评测专用的五档标注路径；不改生产 agent 或全局注册表。"""

from __future__ import annotations

import json
from dataclasses import dataclass

from annotate import annotation_tables as tables
from annotate import prompt_examples as pe
from annotate import spatial_agents as sa
from annotate import spatial_graphs as sg
from annotate import spatial_tools as st

from evaluate.data import EvalSample, LABEL_DIFFERENT, LABEL_SAME


def _model_only_judgement() -> dict:
    return {
        "same_meaning": None, "source": "no_llm", "lookup": "disabled",
        "reason": "评测禁用判定查表，且未调用 LLM（--no-llm）",
    }


class ModelOnlySchemaAgent(sa.SchemaAgent):
    """保留父类候选、示例、结构化输出和名字校验，只跳过特征匹配查表。"""

    def run(self, sentence: str, features: dict, focus_word: dict | None) -> dict:
        if not self.llm:
            return {"name": None, "source": "no_llm",
                    "reason": "评测禁用图式查表，且未调用 LLM（--no-llm）"}
        word = (focus_word or {}).get("word") or ""
        class_word = st.schema_group_key(self.table, focus_word)
        candidates = st.schema_candidates(self.table, focus_word)
        note = "评测禁用特征匹配查表，请直接根据句意和语义特征选择图式"
        content = self._content(sentence, features or {}, focus_word, word, class_word, note, candidates)
        return self._cached_or_ask(content, lambda c: self._ask(c, word, class_word, candidates))


class ModelOnlyJudgeAgent(sa.JudgeAgent):
    """不能调用父类 run：它的 judgement_situation 也会隐式查判定表。"""

    def run(
        self,
        sentence_a: str,
        sentence_b: str,
        schema_a: dict,
        schema_b: dict,
        features_a: dict,
        features_b: dict,
        focus_word: dict | None,
        focus_word_b: dict | None = None,
    ) -> dict:
        local = _model_only_judgement()
        if not self.llm:
            return local
        schema_a, schema_b = schema_a or {}, schema_b or {}
        examples = pe.judge_example_block(self.example_groups, schema_a.get("name"), schema_b.get("name"))
        example_section = f"{examples}\n\n" if examples else ""
        content = (
            f"{sa.pair_head_block(focus_word, focus_word_b, corpus_type=self.corpus_type)}\n"
            f"原句：{sentence_a}\n"
            f"原句 schema：{schema_a.get('name')}（来源 {schema_a.get('source')}）\n"
            f"原句语义特征：{json.dumps(features_a or {}, ensure_ascii=False)}\n"
            f"改写句：{sentence_b}\n"
            f"改写句 schema：{schema_b.get('name')}（来源 {schema_b.get('source')}）\n"
            f"改写句语义特征：{json.dumps(features_b or {}, ensure_ascii=False)}\n"
            "判定表情况：评测禁用判定查表，直接采用模型结论。\n"
            f"{example_section}"
            "请判断这两句话的意义是否相同。"
        )
        try:
            verdict = self._invoke(content)
        except Exception as exc:
            return tables.judgement_from_model(local, None, error=f"调用失败：{exc!r}")
        return tables.judgement_from_model(local, verdict.model_dump())


class EvaluationAllInOneAgent(sa.AllInOneAgent):
    """只调整评测消息：不声称句对已通过未验证的准备检查，查表解析仍沿用父类。"""

    def _content(
        self, sentence_a: str, sentence_b: str, focus_word_a: dict | None, focus_word_b: dict | None
    ) -> str:
        parts = [
            sa.pair_head_block(focus_word_a, focus_word_b, corpus_type=self.corpus_type),
            f"原句：{sentence_a or ''}", f"改写句：{sentence_b or ''}",
            "本次只标注，不重复筛选。",
        ]
        for stage in ("feature", "schema"):
            title = "空间特征" if stage == "feature" else "空间图式"
            for label, focus in (("原句", focus_word_a), ("改写句", focus_word_b)):
                block = pe.example_block(self.example_groups[stage], focus, stage)
                if block:
                    parts.append(f"{label}的{title}示例（只供本侧本步骤参考）：\n{block}")
        for label, focus in (("原句", focus_word_a), ("改写句", focus_word_b)):
            parts.append(self._candidates_block(label, focus))
        parts.append("请依次给出两侧空间特征（features）、两侧图式（schema）及同义关系（judgement）。")
        return "\n\n".join(part for part in parts if part)


class ModelOnlyAllInOneAgent(EvaluationAllInOneAgent):
    """模型调用、示例、输出模型和缓存沿用父类；解析时不作本地覆盖。"""

    def _resolve(
        self, dumps: dict | None, focus_a: dict | None, focus_b: dict | None, *, error: str | None
    ) -> dict:
        if not self.llm:
            source, reason = "no_llm", "未调用 LLM（--no-llm）"
        elif error is not None:
            source, reason = "error", error
        else:
            source, reason = "llm", ""
        feature_results, schemas = {}, {}
        for side, focus in (("a", focus_a), ("b", focus_b)):
            features = ((dumps or {}).get("features") or {}).get(side) or {}
            if source == "llm":
                feature_results[side] = {
                    "features": features, "source": self.table_source, "reason": "",
                    "validation": sa.validate_feature_values(features, self.feature_classes),
                }
                verdict = ((dumps or {}).get("schema") or {}).get(side)
                schemas[side] = (
                    tables.schema_from_model(self.schema_table, focus, verdict) if verdict is not None
                    else {"name": None, "source": "error", "reason": "模型未给出图式"}
                )
            else:
                feature_results[side] = {"features": {}, "source": source, "reason": reason}
                schemas[side] = {"name": None, "source": source, "reason": reason}
        judgement = tables.judgement_from_model(
            _model_only_judgement(), (dumps or {}).get("judgement"), error=error)
        return {"feature_results": feature_results, "schema": schemas, "judgement": judgement,
                "source": source, "reason": reason}


@dataclass
class AnnotationPipeline:
    step: int
    model_key: str
    corpus_type: str
    llm: bool
    graph_name: str
    needs: tuple[str, ...]
    allow_lookup: bool
    agents: sa.AgentSet

    def initial_state(self, sample: EvalSample) -> dict:
        """只取文本和关注词定位；不复制参考标注、筛选结论或候选记录。"""
        return {
            "sentence_a": sample.sentence_a, "sentence_b": sample.sentence_b,
            "focus_word": _clean_focus(sample.focus_word, self.corpus_type, replacement=False),
            "focus_word_b": _clean_focus(sample.focus_word_b, self.corpus_type, replacement=True),
            "status_reason": "", "history": [],
        }

    def _nodes(self):
        # 只复用 read_pair 之后的真实节点，不调用 prepared_pair_node，也不造通过检查的状态。
        if self.graph_name == "synonym":
            return (("synonym", sg.pair_synonym_node(self.agents.need("synonym"))),)
        if self.graph_name == "all_in_one":
            return (("all_in_one", sg.all_in_one_node(self.agents.need("all_in_one"))),)
        return (
            ("feature", sg.pair_feature_node(self.agents.need("feature"))),
            ("schema", sg.pair_schema_node(self.agents.need("schema"))),
            ("judge", sg.pair_judge_node(self.agents.need("judge"))),
        )

    def _feature_failure(self, state: dict) -> tuple[str, str, str] | None:
        name = "all_in_one" if self.graph_name == "all_in_one" else "feature"
        classes = self.agents.need(name).feature_classes
        for side in ("a", "b"):
            detail = (state.get("features_detail") or {}).get(side) or {}
            source = detail.get("source")
            if source in ("error", "no_llm") or not source:
                kind = "no_llm" if source == "no_llm" else "feature_error"
                reason = detail.get("reason") or "特征提取未给出结果"
                return kind, source or "error", f"{side} 侧特征提取失败：{reason}"
            features = (state.get("features") or {}).get(side)
            if not isinstance(features, dict):
                return "invalid_features", "invalid", f"{side} 侧特征不是对象"
            validation = sa.validate_feature_values(features, classes)
            missing = sorted({c["key"] for c in classes} - features.keys())
            if missing:
                validation["ok"] = False
                validation["missing"] = sorted(set(validation["missing"]) | set(missing))
            if not validation["ok"]:
                reason = json.dumps(validation, ensure_ascii=False)
                return "invalid_features", "invalid", f"{side} 侧特征校验失败：{reason}"
        return None

    def _schema_failure(self, state: dict) -> tuple[str, str, str] | None:
        for side in ("a", "b"):
            schema = (state.get("schema") or {}).get(side) or {}
            source, name = schema.get("source"), schema.get("name")
            if source == "error":
                return "schema_error", "error", f"{side} 侧图式失败：{schema.get('reason') or '未提供原因'}"
            if source == "no_llm":
                return "no_llm", source, f"{side} 侧图式未生成：{schema.get('reason') or '未调用 LLM'}"
            if source == "llm_invalid" or not isinstance(name, str) or not name.strip():
                return "invalid_schema", source or "invalid", (
                    f"{side} 侧图式无效：{schema.get('reason') or '未给出合法图式名'}")
        return None

    def _prediction(self, state: dict, failure: tuple[str, str, str] | None = None) -> dict:
        verdict = state.get("synonym" if self.graph_name == "synonym" else "judge") or {}
        same = verdict.get("same_meaning")
        source = verdict.get("source") or "unknown"
        reason = verdict.get("reason") or state.get("status_reason") or ""
        if failure is not None:
            kind, source, reason = failure
            same = None
        elif not isinstance(same, bool):
            kind = "call_failed" if source == "error" else (
                source if source in ("no_llm", "unfilled_prompt") else "judgement_unknown")
            reason = reason or "模型未给出有效的同义判定"
            same = None
        else:
            kind = None
        return {
            "label": (LABEL_SAME if same else LABEL_DIFFERENT) if same is not None else None,
            "same_meaning": same, "reason": reason, "source": source,
            "error": reason if kind is not None else None, "error_kind": kind,
            "features": state.get("features") or {}, "schema": state.get("schema") or {},
            "history": state["history"],
        }

    def run(self, sample: EvalSample) -> dict:
        """从干净状态重放；普通异常记为失败，BatchPending/BaseException 原样传播。"""
        state = self.initial_state(sample)
        with self.agents.budget_scope():
            for name, node in self._nodes():
                try:
                    changes = node(state)
                except Exception as exc:
                    return self._prediction(state, ("call_failed", "error", f"{name} 节点失败：{exc!r}"))
                history = state["history"] + changes.get("history", [])
                state.update(changes)
                state["history"] = history
                if name in ("feature", "all_in_one"):
                    failure = self._feature_failure(state)
                    if failure is not None:
                        return self._prediction(state, failure)
                if name in ("schema", "all_in_one"):
                    failure = self._schema_failure(state)
                    if failure is not None:
                        return self._prediction(state, failure)
        return self._prediction(state)

    def metadata(self) -> dict:
        return {
            "step": self.step, "graph": self.graph_name, "needs": list(self.needs),
            "allow_lookup": self.allow_lookup, "model_key": self.model_key,
            "corpus_type": self.corpus_type, "llm": self.llm,
        }


def _clean_focus(value: object | None, corpus_type: str, *, replacement: bool) -> dict | None:
    focus = {"word": value} if isinstance(value, str) else value if isinstance(value, dict) else {}
    word = focus.get("word")
    if not isinstance(word, str) or not word.strip():
        return None
    clean = {"word": word.strip()}
    if type(focus.get("index")) is int:
        clean["index"] = focus["index"]
    clean["kind"] = focus.get("kind") if isinstance(focus.get("kind"), str) else corpus_type
    clean["source"] = st.FOCUS_SOURCE_ALTER if replacement else (
        focus.get("source") if isinstance(focus.get("source"), str) else "vocab")
    return clean


_STEPS = (
    ("synonym", ("synonym",), False),
    ("all_in_one", ("all_in_one",), False),
    ("all_in_one", ("all_in_one",), True),
    ("pair", ("feature", "schema", "judge"), False),
    ("pair", ("feature", "schema", "judge"), True),
)


def build_pipeline(
    step: int, model_key: str, corpus_type: str, *, llm: bool, policy: dict | None = None
) -> AnnotationPipeline:
    if type(step) is not int or not 0 <= step < len(_STEPS):
        raise ValueError(f"非法评测 step {step!r}；可用：0、1、2、3、4")
    corpus_type = st.normalize_corpus_type(corpus_type)
    graph_name, needs, allow_lookup = _STEPS[step]
    classes = {
        "synonym": sa.SynonymAgent,
        "all_in_one": EvaluationAllInOneAgent if allow_lookup else ModelOnlyAllInOneAgent,
        "feature": sa.FeatureAgent,
        "schema": sa.SchemaAgent if allow_lookup else ModelOnlySchemaAgent,
        "judge": sa.JudgeAgent if allow_lookup else ModelOnlyJudgeAgent,
    }
    agents = sa.AgentSet(
        (name, classes[name](model_key=model_key, corpus_type=corpus_type, llm=llm, retry_policy=policy))
        for name in needs
    )
    return AnnotationPipeline(step, model_key, corpus_type, bool(llm), graph_name, needs, allow_lookup, agents)
