# encoding: utf8
"""离线验证：批量推理、LTP 一次准备、三元组注入（不联网、不建真实模型）"""

import json
import tempfile
import unittest
from pathlib import Path

import batch_inference as bi
import config
import spatial_agents as sa
import spatial_graphs as sg
import spatial_runner as sr
import spatial_tools as st


def _chat_body(payload: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": bi.TOOL_NAME, "arguments": json.dumps(payload, ensure_ascii=False)}}
                    ]
                }
            }
        ]
    }


class FakeClient:
    """按 agent 名返回一条结构化结果；记录每次提交的条目，供断言波次与去重"""

    def __init__(self):
        self.calls = []
        self.messages = []

    def run(self, items, *, on_job=None, on_status=None, on_results=None):
        self.calls.append([item.agent_name for item in items])
        self.messages.extend(item.content for item in items)
        results = {}
        for item in items:
            if item.agent_name == "reality":
                payload = {"is_spatial_scene": True, "reason": "现实空间"}
            elif item.agent_name == "semantic":
                payload = {
                    "grammar_ok": True,
                    "grammar_reason": "通顺",
                    "spatial_ok": True,
                    "spatial_reason": "合理",
                }
            elif item.agent_name == "feature":
                payload = {"area": ["内"]}
            elif item.agent_name == "schema":
                payload = {"schema_name": "上1", "reason": "接近"}
            elif item.agent_name == "judge":
                payload = {"same_meaning": True, "reason": "同义"}
            elif item.agent_name == "all_in_one":
                payload = _all_in_one_payload()
            else:
                raise AssertionError(f"未预期的 agent {item.agent_name}")
            results[item.key] = (item.parse_response(_chat_body(payload)), None)
        if on_results is not None:
            on_results(results)
        return results


def _all_in_one_payload() -> dict:
    return {
        "reality": {"is_spatial_scene": True, "reason": "现实空间"},
        "semantic": {
            "a": {"grammar_ok": True, "grammar_reason": "通顺", "spatial_ok": True, "spatial_reason": "合理"},
            "b": {"grammar_ok": True, "grammar_reason": "通顺", "spatial_ok": True, "spatial_reason": "合理"},
        },
        "features": {"a": {"area": ["内"]}, "b": {"area": ["内"]}},
        "schema": {
            "a": {"schema_name": "上1", "reason": "接近"},
            "b": {"schema_name": "上1", "reason": "接近"},
        },
        "judgement": {"same_meaning": True, "reason": "同义"},
    }


def _profile() -> bi.BatchProfile:
    return bi.BatchProfile(
        model_key="fake",
        provider="fake",
        model="fake-model",
        base_url="http://fake.invalid/v1",
        api_key="test",
        enabled=True,
    )


def _record() -> dict:
    return {
        "line_no": 1,
        "sentence": ["书", "放", "在", "桌子", "上", "。"],
        "filter_methods": ["locality_phrase"],
        "match_tree": [
            {"index": 3, "word": "桌子", "children": [{"index": 4, "word": "上", "children": []}]}
        ],
        "screening": {"status": "no_llm"},
        "source_file": "unit.jsonl",
        "source_document_no": 1,
        "localization_grounding": {
            "candidates": [
                {
                    "index": 4,
                    "word": "上",
                    "keep": True,
                    "reference": {"index": 3, "word": "桌子", "sdp": "LOC"},
                    "target": {"index": 0, "word": "书", "role": "PAT"},
                    "verb": {"index": 1, "word": "放"},
                }
            ]
        },
    }


def _agents():
    """不读 api_keys.json、不建模型的一套 agent（同步构造参数显式给）"""
    return sa.AgentSet(
        {
            "alter": sa.AlterAgent(rules={"groups": [{"words": ["上"], "replacements": ["里"]}]}, corpus_type="方位词"),
            "reality": sa.RealityAgent(corpus_type="方位词", retry_policy={"max_retries": 0}),
            "semantic": sa.SemanticAgent(corpus_type="方位词", retry_policy={"max_retries": 0}),
            "feature": sa.FeatureAgent(
                corpus_type="方位词",
                feature_classes=[{"key": "area", "name": "区域", "values": ["内"], "description": "区域"}],
                retry_policy={"max_retries": 0},
            ),
            "schema": sa.SchemaAgent(
                corpus_type="方位词",
                table={
                    "schemas": {
                        "上": {"1": {"description": "内部", "match": [{"area": ["内"]}]}},
                        "里": {"1": {"description": "内部", "match": [{"area": ["内"]}]}},
                    }
                },
                feature_keys={"area"},
                retry_policy={"max_retries": 0},
            ),
            "judge": sa.JudgeAgent(corpus_type="方位词", table={}, retry_policy={"max_retries": 0}),
        }
    )


class LtpAndGroundingTests(unittest.TestCase):
    def setUp(self):
        st.reset_syntax_evidence()

    def test_syntax_block_reads_evidence_without_inferring(self):
        sentence = "书放在桌子上。"
        st._SYNTAX_EVIDENCE[sentence] = {
            "text": "分词/词性: 0:书/n 1:放/v\n依存句法:\n  0:书 —SBV—> 1:放",
            "error": None,
        }
        original = st._cache.get_task_value

        def boom(*args, **kwargs):
            raise AssertionError("不应在拼消息时调用 LTP 推理")

        st._cache.get_task_value = boom
        try:
            self.assertIn("0:书/n", sa.syntax_block(sentence) or "")
        finally:
            st._cache.get_task_value = original

    def test_grounding_line_uses_exact_position(self):
        record = _record()
        focus = st.pick_focus_words(record)[0]
        self.assertEqual(focus["grounding"]["target"]["word"], "书")
        text = sa.head_block(focus, corpus_type="方位词")
        self.assertIn("目标物「书」", text)
        self.assertIn("参照物「桌子」", text)
        self.assertIn("动词「放」", text)
        # 既有锚点不移动：关注词第 1 行、语料类型第 3 行
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("本次需要重点关注的词"))
        self.assertEqual(lines[2], "本次跑批的语料类型：方位词。")

    def test_grounding_absent_for_direction(self):
        record = _record()
        record.pop("localization_grounding")
        focus = st.pick_focus_words(record)[0]
        self.assertNotIn("grounding", focus)
        self.assertNotIn("目标物", sa.head_block(focus))


class BatchProfileTests(unittest.TestCase):
    def test_provider_and_override(self):
        import agents as agent_factory

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "api_keys.json"
            path.write_text(
                json.dumps(
                    {
                        "openai-like": {
                            "model_name": "gpt-test",
                            "provider": "openai",
                            "base_url": "https://api.example/v1",
                            "api_key": "x",
                        },
                        "deepseek-like": {
                            "model_name": "deepseek-test",
                            "provider": "deepseek",
                            "base_url": "https://api.deepseek/v1",
                            "api_key": "x",
                        },
                        "openrouter-like": {
                            "model_name": "or-test",
                            "provider": "openrouter",
                            "base_url": "https://openrouter.ai/api/v1",
                            "api_key": "x",
                        },
                        "gateway": {
                            "model_name": "gateway-test",
                            "provider": "custom",
                            "base_url": "https://gw.example/v1",
                            "api_key": "x",
                            "batch": {"enabled": True, "base_url": "https://gw.example/v1"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            original = agent_factory._MODEL_CONFIG_PATH
            agent_factory._MODEL_CONFIG_PATH = path
            try:
                self.assertTrue(bi.batch_profile("openai-like").enabled)
                self.assertEqual(bi.batch_profile("openai-like").protocol, "openai")
                self.assertFalse(bi.batch_profile("deepseek-like").enabled)
                self.assertEqual(bi.batch_profile("openrouter-like").protocol, "openrouter")
                self.assertTrue(bi.batch_profile("gateway").enabled)
            finally:
                agent_factory._MODEL_CONFIG_PATH = original


class BatchItemTests(unittest.TestCase):
    def test_request_body_and_parse(self):
        item = bi.BatchItem(
            key="k1",
            agent_name="semantic",
            system_prompt="sys",
            content="user",
            response_model=sa.SemanticVerdict,
        )
        body = item.request_body(_profile())
        self.assertEqual(body["tool_choice"]["function"]["name"], bi.TOOL_NAME)
        self.assertEqual(body["messages"][1]["content"], "user")
        parsed = item.parse_response(
            _chat_body(
                {
                    "grammar_ok": True,
                    "grammar_reason": "ok",
                    "spatial_ok": False,
                    "spatial_reason": "异常",
                }
            )
        )
        self.assertFalse(parsed["spatial_ok"])
        # content 里给 JSON 的兼容分支
        parsed2 = item.parse_response(
            {"choices": [{"message": {"content": '```json\n{"grammar_ok": true, "grammar_reason": "ok", '
                                                        '"spatial_ok": true, "spatial_reason": "ok"}\n```'}}]}
        )
        self.assertTrue(parsed2["grammar_ok"])


class NativeClientProtocolTests(unittest.TestCase):
    def test_submit_poll_collect(self):
        class FakeResponse:
            def __init__(self, payload, text=""):
                self._payload = payload
                self.text = text
                self.status_code = 200

            def json(self):
                return self._payload

        state = {"polls": 0}

        def fake_request(method, url, **kwargs):
            if method == "POST" and url.endswith("/files"):
                return FakeResponse({"id": "file1"})
            if method == "POST" and url.endswith("/batches"):
                return FakeResponse({"id": "batch1", "status": "validating"})
            if method == "GET" and "/batches/batch1" in url:
                state["polls"] += 1
                if state["polls"] == 1:
                    return FakeResponse({"id": "batch1", "status": "in_progress"})
                return FakeResponse({"id": "batch1", "status": "completed", "output_file_id": "out1"})
            if method == "GET" and url.endswith("/files/out1/content"):
                payload = {
                    "id": "r1",
                    "custom_id": "k1",
                    "response": {
                        "status_code": 200,
                        "body": _chat_body(
                            {
                                "grammar_ok": True,
                                "grammar_reason": "ok",
                                "spatial_ok": True,
                                "spatial_reason": "ok",
                            }
                        ),
                    },
                }
                return FakeResponse({}, text=json.dumps(payload, ensure_ascii=False) + "\n")
            raise AssertionError(f"未预期的请求 {method} {url}")

        item = bi.BatchItem(
            key="k1",
            agent_name="semantic",
            system_prompt="sys",
            content="user",
            response_model=sa.SemanticVerdict,
        )
        client = bi.NativeBatchClient(_profile(), poll_seconds=1, timeout_seconds=5)
        client.poll_seconds = 0.01
        # 生产代码走 requests.Session.request（保留 proxy/session 配置），
        # 测试必须替换同一个入口；替换模块级 requests.request 不会生效。
        original = client.session.request
        client.session.request = fake_request
        try:
            jobs = []
            results = client.run([item], on_job=jobs.append)
        finally:
            client.session.request = original
        self.assertEqual([job.batch_id for job in jobs], ["batch1"])
        self.assertTrue(results["k1"][0]["grammar_ok"])
        self.assertIsNone(results["k1"][1])


class OpenRouterProtocolTests(unittest.TestCase):
    def test_inline_batch_submit_and_results(self):
        class FakeResponse:
            def __init__(self, payload, text=""):
                self._payload = payload
                self.text = text
                self.status_code = 200

            def json(self):
                return self._payload

        submitted = {}
        state = {"posted": False}

        def fake_request(method, url, **kwargs):
            if method == "POST" and url.endswith("/batches"):
                submitted.update(kwargs["json"])
                state["posted"] = True
                return FakeResponse({"id": "or-1", "status": "validating", "results": None})
            if method == "GET" and "/batches/or-1" in url:
                if not state["posted"]:
                    raise AssertionError("应先提交")
                return FakeResponse(
                    {
                        "id": "or-1",
                        "status": "completed",
                        "results": [
                            {
                                "custom_id": "k1",
                                "response": {
                                    "status_code": 200,
                                    "body": _chat_body(
                                        {
                                            "grammar_ok": True,
                                            "grammar_reason": "ok",
                                            "spatial_ok": True,
                                            "spatial_reason": "ok",
                                        }
                                    ),
                                },
                                "error": None,
                            }
                        ],
                    }
                )
            raise AssertionError(f"未预期的请求 {method} {url}")

        item = bi.BatchItem(
            key="k1",
            agent_name="semantic",
            system_prompt="sys",
            content="user",
            response_model=sa.SemanticVerdict,
        )
        profile = bi.BatchProfile(
            model_key="or",
            provider="openrouter",
            model="or-model",
            base_url="https://openrouter.ai/api/v1",
            api_key="x",
            protocol="openrouter",
            enabled=True,
        )
        client = bi.NativeBatchClient(profile, poll_seconds=1, timeout_seconds=5)
        client.poll_seconds = 0.01
        original = client.session.request
        client.session.request = fake_request
        try:
            results = client.run([item])
        finally:
            client.session.request = original
        self.assertEqual(submitted["endpoint"], "/v1/chat/completions")
        self.assertEqual(submitted["model"], "or-model")
        self.assertEqual(submitted["requests"][0]["custom_id"], "k1")
        self.assertNotIn("input_file_id", submitted)
        self.assertTrue(results["k1"][0]["spatial_ok"])


class PrepareSyntaxTests(unittest.TestCase):
    def setUp(self):
        st.reset_syntax_evidence()

    def test_prepare_dedups_and_never_infers_twice(self):
        calls = []

        def fake_prefill(sentences, chunk_size=None):
            sentences = list(sentences)
            calls.append(sentences)
            for sentence in sentences:
                st._cache.cache[sentence] = {
                    config.CWS: list(sentence),
                    config.POS: ["n"] * len(sentence),
                    config.DEP: {"head": [0] * len(sentence), "label": ["ROOT"] * len(sentence)},
                }
            return len(sentences)

        original = st.prefill_ltp
        st.prefill_ltp = fake_prefill
        try:
            stat = st.prepare_syntax(["甲句", "甲句", "乙句"])
            self.assertEqual(stat["sentences"], 2)
            self.assertEqual(stat["inferred"], 2)
            self.assertEqual(len(calls), 1)
            self.assertIn("分词/词性", st.syntax_text("甲句") or "")
            # 第二次全是证据表命中，不再推理
            stat2 = st.prepare_syntax(["甲句", "乙句"])
            self.assertEqual(stat2["inferred"], 0)
            self.assertEqual(stat2["cached"], 2)
            self.assertEqual(len(calls), 1)
        finally:
            st.prefill_ltp = original
            for sentence in ("甲句", "乙句"):
                st._cache.cache.pop(sentence, None)


class RunnerBatchIntegrationTests(unittest.TestCase):
    def test_execute_run_uses_batch_session_and_skips_reality_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            agent_set = _agents()
            spec = sg.GRAPHS["pair"]
            records = [_record()]
            args = sr.parse_args(
                [
                    "--graph",
                    "pair",
                    "--batch",
                    "native",
                    "--no-pairs-csv",
                    "--overwrite",
                    "--model",
                    "fake",
                ]
            )
            fake_client = FakeClient()
            real_session = bi.BatchSession

            def session_factory(profile, **kwargs):
                return real_session(profile, client=fake_client, **kwargs)

            original_profile = bi.batch_profile
            original_session = bi.BatchSession
            bi.batch_profile = lambda model_key: _profile()
            bi.BatchSession = session_factory
            try:
                result = sr._execute_run(
                    args,
                    spec,
                    "方位词",
                    Path(tmp) / "corpus.jsonl",
                    records,
                    agent_set,
                    out,
                    model_key="fake",
                    prefill_ltp=False,
                    verbose=False,
                )
            finally:
                bi.batch_profile = original_profile
                bi.BatchSession = original_session
            summary = result["summary"]
            self.assertEqual(summary["count"], len(result["states"]))
            self.assertIn("batch", summary)
            self.assertEqual(summary["batch"]["mode"], "native")
            self.assertEqual(summary["batch"]["waves"], 4)
            self.assertNotIn("reality_prefill", summary)
            self.assertTrue(fake_client.calls)


class AllInOnePrecomputedTests(unittest.TestCase):
    def test_precomputed_result_skips_sync_call(self):
        import argparse

        import all_in_one as aio

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            agents = _agents()
            ctx = aio._Context(
                agents=agents,
                corpus_type="方位词",
                model_key="fake",
                llm=True,
                out_path=out,
                system_prompt="system",
                verdict_model=aio.build_verdict_model([{"key": "area"}]),
                feature_classes=[{"key": "area"}],
                prompt_path="",
                reality_path="",
                retry_policy={"max_retries": 0},
            )
            args = argparse.Namespace(no_syntax=True, print_prompt=False, candidate_index=0, all_candidates=False)
            payload = _all_in_one_payload()
            state = sg.init_state(_record(), str(out))
            record = aio.process_state(
                state,
                ctx=ctx,
                args=args,
                call_agent=None,
                counter=aio.CallCounter(),
                precomputed=(payload, None, 0),
            )
            self.assertEqual(record["all_in_one"]["raw"], payload)
            self.assertTrue(record["reality"]["is_spatial_scene"])
            self.assertIsNone(record["all_in_one"]["error"])


class GroundingSideTests(unittest.TestCase):
    def test_replacement_side_marks_not_reverified(self):
        record = _record()
        focus = st.pick_focus_word(record)
        replacement = st.replacement_focus_word(focus, "里")
        text = sa.head_block(replacement, corpus_type="方位词")
        self.assertIn("改写后未重新验证", text)
        self.assertIn("目标物「书」", text)


class StrictAndMetaTests(unittest.TestCase):
    def test_strict_native_refuses_fallback(self):
        class Unavailable:
            def run(self, items, **kwargs):
                raise bi.BatchUnavailable("no endpoint")

        with tempfile.TemporaryDirectory() as tmp:
            session = bi.BatchSession(
                _profile(),
                cache_path=Path(tmp) / "cache.jsonl",
                client=Unavailable(),
                strict_native=True,
            )
            item = bi.BatchItem(
                key="k",
                agent_name="x",
                system_prompt="s",
                content="c",
                sync_call=lambda: ({"ok": True}, None),
            )
            with self.assertRaises(RuntimeError):
                session.execute([item])

    def test_meta_mismatch_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            cache.write_text(json.dumps({"key": "k", "dumps": {"ok": True}, "error": None}) + "\n", encoding="utf-8")
            cache.with_suffix(cache.suffix + ".meta.json").write_text(
                json.dumps({"model": "another"}, ensure_ascii=False), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                bi.BatchSession(_profile(), cache_path=cache, resume=True)

    def test_reservation_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = bi.BatchSession(_profile(), cache_path=Path(tmp) / "cache.jsonl")
            agent = sa.SemanticAgent(corpus_type="方位词", retry_policy={"max_retries": 0})
            key = bi.item_key(agent.NAME, "user", agent.system_prompt)
            session._reserved.add(key)
            self.assertTrue(session.consume_reservation(agent, "user"))
            self.assertFalse(session.consume_reservation(agent, "user"))


class ResumeSafetyTests(unittest.TestCase):
    def test_new_session_clears_old_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            cache.with_suffix(cache.suffix + ".jobs.json").write_text("[]", encoding="utf-8")
            bi.BatchSession(_profile(), cache_path=cache)
            self.assertFalse(cache.with_suffix(cache.suffix + ".jobs.json").exists())

    def test_resume_failed_job_does_not_cache_error(self):
        from batch_inference import BatchJob

        class FailedJobClient:
            def wait(self, job):
                job.status = "failed"
                return job

            def collect(self, job):
                return {key: (None, "batch 状态为 failed") for key in job.keys}

            def run(self, items, *, on_results=None, **kwargs):
                results = {item.key: ({"ok": True}, None) for item in items}
                if on_results is not None:
                    on_results(results)
                return results

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            cache.write_text("", encoding="utf-8")
            cache.with_suffix(cache.suffix + ".meta.json").write_text(
                json.dumps(_profile().fingerprint(), ensure_ascii=False), encoding="utf-8"
            )
            jobs = [
                BatchJob(
                    batch_id="old",
                    keys=["k"],
                    status="failed",
                    protocol="openai",
                ).to_dict()
            ]
            cache.with_suffix(cache.suffix + ".jobs.json").write_text(
                json.dumps(jobs, ensure_ascii=False), encoding="utf-8"
            )
            session = bi.BatchSession(
                _profile(), cache_path=cache, client=FailedJobClient(), resume=True
            )
            item = bi.BatchItem("k", "x", "s", "c", None, sync_call=lambda: ({"ok": True}, None))
            session.execute([item])
            self.assertTrue(session.results["k"]["dumps"]["ok"])

    def test_create_timeout_stops_instead_of_fallback(self):
        class FakeResponse:
            status_code = 200
            text = ""
            def __init__(self, payload):
                self._payload = payload
            def json(self):
                return self._payload

        def fake_request(method, url, **kwargs):
            if method == "POST" and url.endswith("/files"):
                return FakeResponse({"id": "file1"})
            if method == "POST" and url.endswith("/batches"):
                raise bi.requests.exceptions.Timeout("response lost")
            raise AssertionError(f"未预期的请求 {method} {url}")

        item = bi.BatchItem("k", "x", "s", "c", None)
        client = bi.NativeBatchClient(_profile(), timeout_seconds=1)
        original = client.session.request
        client.session.request = fake_request
        try:
            with self.assertRaises(bi.BatchTimeout):
                client.submit([item])
        finally:
            client.session.request = original


class PartialChunkTests(unittest.TestCase):
    def test_accepted_chunk_results_survive_later_failure(self):
        class ChunkFail:
            def run(self, items, *, on_job=None, on_results=None, **kwargs):
                from batch_inference import BatchJob

                job = BatchJob(batch_id="j1", keys=[item.key for item in items])
                if on_job is not None:
                    on_job(job)
                if on_results is not None:
                    on_results({items[0].key: ({"ok": True}, None)})
                raise TimeoutError("poll connection lost")

        with tempfile.TemporaryDirectory() as tmp:
            session = bi.BatchSession(
                _profile(), cache_path=Path(tmp) / "cache.jsonl", client=ChunkFail()
            )
            item = bi.BatchItem("k", "x", "s", "c", None, sync_call=lambda: ({"ok": False}, None))
            session.pending[item.key] = item
            with self.assertRaises(bi.BatchTimeout):
                session.execute()
            self.assertTrue(session.results["k"]["dumps"]["ok"])
            self.assertEqual(session.fallback_items, 0)


class FailedItemFallbackTests(unittest.TestCase):
    def test_native_item_error_is_retried_synchronously(self):
        class PartlyFailing:
            def run(self, items, *, on_results=None, **kwargs):
                results = {
                    item.key: (
                        ({"ok": True}, None)
                        if item.key == "good"
                        else (None, "rate limited")
                    )
                    for item in items
                }
                if on_results is not None:
                    on_results(results)
                return results

        with tempfile.TemporaryDirectory() as tmp:
            session = bi.BatchSession(
                _profile(), cache_path=Path(tmp) / "cache.jsonl", client=PartlyFailing()
            )
            items = [
                bi.BatchItem("good", "x", "s", "c1", None, sync_call=lambda: ({"ok": False}, None)),
                bi.BatchItem("bad", "x", "s", "c2", None, sync_call=lambda: ({"ok": True}, None)),
            ]
            session.execute(items)
            self.assertTrue(session.results["good"]["dumps"]["ok"])
            self.assertTrue(session.results["bad"]["dumps"]["ok"])
            self.assertEqual(session.mode, "mixed")


class BatchSessionFallbackTests(unittest.TestCase):
    def test_fallback_when_native_unavailable(self):
        class Unavailable:
            def run(self, items, **_kwargs):
                raise bi.BatchUnavailable("no endpoint")

        with tempfile.TemporaryDirectory() as tmp:
            session = bi.BatchSession(
                _profile(),
                cache_path=Path(tmp) / "cache.jsonl",
                client=Unavailable(),
                timeout_seconds=1,
            )
            item = bi.BatchItem(
                key="k",
                agent_name="x",
                system_prompt="s",
                content="c",
                response_model=None,
                sync_call=lambda: ({"ok": True}, None),
            )
            session.execute([item], fallback_workers=1)
            summary = session.summary()
            self.assertEqual(summary["fallback_items"], 1)
            self.assertEqual(session.results["k"]["dumps"], {"ok": True})

    def test_cache_resume_replays_without_resubmit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            cache.write_text(
                json.dumps({"key": "k", "dumps": {"ok": True}, "error": None}) + "\n", encoding="utf-8"
            )
            cache.with_suffix(cache.suffix + ".meta.json").write_text(
                json.dumps(_profile().fingerprint(), ensure_ascii=False), encoding="utf-8"
            )

            class Boom:
                def run(self, items, **_kwargs):
                    raise AssertionError("命中缓存时不应再提交")

            session = bi.BatchSession(
                _profile(), cache_path=cache, client=Boom(), resume=True
            )
            item = bi.BatchItem(key="k", agent_name="x", system_prompt="s", content="c")
            session.execute([item])
            self.assertEqual(session.cache_hits, 1)


class GraphWaveTests(unittest.TestCase):
    def test_pair_graph_runs_in_batch_waves(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.jsonl"
            agents = _agents()
            writer = sr.ResultWriter(
                out,
                counter_names=sg.GRAPHS["pair"].counter_names,
                tally=sg.GRAPHS["pair"].tally,
                corpus_type="方位词",
                config_tables={},
                overwrite=True,
            )
            graph = sg.GRAPHS["pair"].build(agents, writer)
            states = sg.GRAPHS["pair"].init_states([_record()], agents, str(out))
            self.assertTrue(states)
            client = FakeClient()
            session = bi.BatchSession(
                _profile(),
                cache_path=Path(tmp) / "cache.jsonl",
                client=client,
                timeout_seconds=1,
            )
            bi.run_waves(
                states,
                lambda state: graph.invoke(state, {"recursion_limit": 60}),
                session,
                fallback_workers=1,
            )
            records = st.read_jsonl(out)
            writer._close()  # 本测试不走 finish，显式释放常开句柄（Windows 上临时目录才能删除）
            self.assertEqual(len(records), len(states))
            self.assertIn(records[0]["status"], {"ok", "ok_review"})
            # reality -> semantic(a/b 同波) -> feature(a/b 同波) -> judge
            self.assertEqual(len(client.calls), 4, client.calls)
            self.assertEqual(sorted(client.calls[1]), ["semantic", "semantic"])
            self.assertEqual(sorted(client.calls[2]), ["feature", "feature"])
            self.assertTrue(any("目标物「书」" in message for message in client.messages))
            # 离线验证不建真实模型
            self.assertTrue(all(agent._agent is None for agent in agents.values()))
            # 汇总：波次数与完成项
            summary = session.summary()
            self.assertEqual(summary["waves"], 4)
            self.assertEqual(summary["items_completed"], 6)

    def test_batch_pending_survives_agent_except_exception(self):
        # _ask 用 except Exception 兜底；BatchPending 是 BaseException，不能被记成 error
        class FakeSession:
            collecting = True

            def lookup(self, agent, content):
                return None

            def add(self, agent, content):
                return "k"

        token = bi._CURRENT_BATCH.set(FakeSession())
        try:
            with self.assertRaises(bi.BatchPending):
                sa.SemanticAgent(corpus_type="方位词", retry_policy={"max_retries": 0}).run(
                    "书放在桌子上。", {"index": 4, "word": "上", "kind": "方位词"}
                )
        finally:
            bi._CURRENT_BATCH.reset(token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
