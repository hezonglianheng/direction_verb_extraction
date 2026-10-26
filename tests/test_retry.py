"""超时重试与退避的纯函数测试（不依赖 langchain / 网络）。"""

import unittest

import config
import retry


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class RetryClassificationTests(unittest.TestCase):
    def test_default_retry_budget_is_six(self):
        self.assertEqual(config.LLM_MAX_RETRIES, 6)

    def test_timeouts_and_connections_are_retryable(self):
        for exc in (
            TimeoutError("timed out"),
            ConnectionError("reset"),
            _StatusError(429),
            _StatusError(500),
            _StatusError(503),
        ):
            with self.subTest(exc=exc):
                self.assertTrue(retry.is_retryable_error(exc))

    def test_configuration_errors_are_not_retryable(self):
        for exc in (ValueError("bad"), _StatusError(400), _StatusError(401), _StatusError(403), _StatusError(404)):
            with self.subTest(exc=exc):
                self.assertFalse(retry.is_retryable_error(exc))

    def test_class_name_timeout_is_retryable(self):
        class APITimeoutError(Exception):
            pass

        self.assertTrue(retry.is_retryable_error(APITimeoutError("timeout")))

    def test_provider_specific_transient_names_are_retryable(self):
        for name in ("DeadlineExceeded", "ResourceExhausted", "ServiceUnavailable", "OverloadedError"):
            with self.subTest(name=name):
                exc = type(name, (Exception,), {})("transient")
                self.assertTrue(retry.is_retryable_error(exc))

    def test_outer_auth_error_wins_over_inner_timeout_text(self):
        inner = TimeoutError("timed out")
        outer = _StatusError(401)
        outer.__cause__ = inner
        self.assertFalse(retry.is_retryable_error(outer))


class BackoffTests(unittest.TestCase):
    def test_exponential_and_capped(self):
        delays = [
            retry.backoff_delay(
                attempt,
                backoff_seconds=1,
                backoff_max_seconds=10,
                jitter_seconds=0,
            )
            for attempt in range(1, 6)
        ]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 8.0, 10.0])


class CallWithRetryTests(unittest.TestCase):
    def test_retries_until_success(self):
        calls = []
        sleeps = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError("timed out")
            return "ok"

        result = retry.call_with_retry(
            flaky,
            policy={"max_retries": 6, "backoff_seconds": 1, "jitter_seconds": 0},
            sleep=sleeps.append,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_exhaustion_raises_after_max_retries(self):
        calls = []
        sleeps = []

        def always_timeout():
            calls.append(1)
            raise TimeoutError("timed out")

        with self.assertRaises(TimeoutError):
            retry.call_with_retry(
                always_timeout,
                policy={"max_retries": 6, "backoff_seconds": 1, "jitter_seconds": 0},
                sleep=sleeps.append,
            )
        self.assertEqual(len(calls), 7)  # 首发 1 次 + 重试 6 次
        self.assertEqual(len(sleeps), 6)

    def test_non_retryable_fails_immediately(self):
        calls = []
        sleeps = []

        def bad():
            calls.append(1)
            raise ValueError("bad request")

        with self.assertRaises(ValueError):
            retry.call_with_retry(
                bad,
                policy={"max_retries": 6},
                sleep=sleeps.append,
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(sleeps, [])

    def test_zero_retries_means_one_attempt(self):
        calls = []

        def timeout():
            calls.append(1)
            raise TimeoutError("timed out")

        with self.assertRaises(TimeoutError):
            retry.call_with_retry(timeout, policy={"max_retries": 0}, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)

    def test_on_retry_receives_attempt_and_delay(self):
        events = []
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise TimeoutError("timed out")
            return 1

        retry.call_with_retry(
            flaky,
            policy={"max_retries": 3, "backoff_seconds": 2, "jitter_seconds": 0},
            sleep=lambda delay: None,
            on_retry=lambda exc, attempt, delay: events.append((attempt, delay)),
        )
        self.assertEqual(events, [(1, 2.0)])

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            retry.call_with_retry(lambda: 1, policy={"max_retries": -1})
        with self.assertRaises(ValueError):
            retry.call_with_retry(lambda: 1, policy={"max_retries": True})


if __name__ == "__main__":
    unittest.main()
