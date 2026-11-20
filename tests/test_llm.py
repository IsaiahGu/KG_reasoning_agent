import json
from urllib.error import HTTPError

import pytest

from hyperbind.llm import LocalLLMClient, LLMError, StructuredOutputError, api_controller_for_run, summarize_calls, verify_default_thinking_off


SCHEMA = {"type": "object", "properties": {"value": {"type": "integer"}},
          "required": ["value"], "additionalProperties": False}
MESSAGES = [{"role": "user", "content": "Return two."}]


class Transport:
    def __init__(self, content='{"value":2}', finish_reason="stop", reasoning=None):
        self.posts = []
        self.content, self.finish_reason, self.reasoning = content, finish_reason, reasoning

    def __call__(self, method, url, payload, headers, timeout):
        if method == "GET":
            return {"data": [{"id": "local-qwen"}]}
        self.posts.append(payload)
        return {"choices": [{"message": {"content": self.content, "reasoning_content": self.reasoning},
                             "finish_reason": self.finish_reason}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}}


def client(tmp_path, transport, **kwargs):
    return LocalLLMClient(base_url="http://127.0.0.1:18080/v1", cache_dir=tmp_path / "cache",
                          log_path=tmp_path / "calls.jsonl", transport=transport, retries=0, **kwargs)


def test_default_off_and_per_request_override_have_separate_cache(tmp_path):
    transport = Transport()
    reader = client(tmp_path, transport)
    result = reader.complete(MESSAGES, SCHEMA)
    assert result.parsed == {"value": 2} and not result.cache_hit
    assert transport.posts[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert transport.posts[0]["response_format"]["type"] == "json_schema"
    assert reader.complete(MESSAGES, SCHEMA).cache_hit
    reader.complete(MESSAGES, SCHEMA, thinking=True)
    assert len(transport.posts) == 2
    assert transport.posts[1]["chat_template_kwargs"] == {"enable_thinking": True}
    summary = summarize_calls(tmp_path / "calls.jsonl")
    assert summary["logical_calls"] == 3 and summary["cache_hits"] == 1
    assert summary["completion_tokens_generated"] == 10 and summary["prompt_tokens_generated"] == 14


def test_r2_probe_omits_kwargs_and_bypasses_cache(tmp_path):
    transport = Transport(content="2")
    reader = client(tmp_path, transport)
    assert verify_default_thinking_off(reader)["status"] == "passed"
    assert "chat_template_kwargs" not in transport.posts[0]
    verify_default_thinking_off(reader)
    assert len(transport.posts) == 2


@pytest.mark.parametrize("content,reasoning", [("<think>2</think>2", None), ("2", "hidden reasoning")])
def test_r2_rejects_reasoning(tmp_path, content, reasoning):
    with pytest.raises(AssertionError):
        verify_default_thinking_off(client(tmp_path, Transport(content=content, reasoning=reasoning)))


def test_model_identity_generation_prompt_and_schema_invalidate_cache(tmp_path):
    transport = Transport()
    first = client(tmp_path, transport, model_fingerprint={"revision": "a"})
    second = client(tmp_path, transport, model_fingerprint={"revision": "b"})
    first.complete(MESSAGES, SCHEMA)
    second.complete(MESSAGES, SCHEMA)
    second.complete(MESSAGES, SCHEMA, generation_overrides={"max_tokens": 100})
    second.complete([{"role": "user", "content": "Different input."}], SCHEMA)
    changed_schema = {**SCHEMA, "description": "A different constrained format"}
    second.complete(MESSAGES, changed_schema)
    assert len(transport.posts) == 5


@pytest.mark.parametrize("content", ['```json\n{"value":2}\n```', '{"value":"2"}', '{"value":2,"extra":1}', '{"value":NaN}', '{"value":1,"value":2}'])
def test_strict_json_is_not_repaired_and_failure_is_logged_and_cached(tmp_path, content):
    transport = Transport(content)
    reader = client(tmp_path, transport)
    for _ in range(2):
        with pytest.raises(StructuredOutputError):
            reader.complete(MESSAGES, SCHEMA)
    assert len(transport.posts) == 1
    rows = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [x["status"] for x in rows] == ["error", "error"]
    assert rows[1]["cache_hit"]


def test_truncation_is_a_json_failure(tmp_path):
    with pytest.raises(StructuredOutputError):
        client(tmp_path, Transport(finish_reason="length")).complete(MESSAGES, SCHEMA)


def test_key_is_read_from_env_and_not_in_cache_or_log(tmp_path, monkeypatch):
    secret = "test-private-value-do-not-log"
    monkeypatch.setenv("HG_LOCAL_TEST_KEY", secret)
    transport = Transport()
    reader = client(tmp_path, transport, api_key_env="HG_LOCAL_TEST_KEY")
    reader.complete(MESSAGES, SCHEMA)
    for path in tmp_path.rglob("*.json*"):
        assert secret not in path.read_text()


def test_http_failure_is_logged_without_response_body_or_credentials(tmp_path):
    def broken(method, url, payload, headers, timeout):
        if method == "GET":
            return {"data": [{"id": "local-qwen"}]}
        raise HTTPError(url, 400, "bad request", None, None)
    with pytest.raises(LLMError):
        client(tmp_path, broken).complete(MESSAGES)
    row = json.loads((tmp_path / "calls.jsonl").read_text())
    assert row["status"] == "error" and row["http_attempts"][0]["http_status"] == 400


def test_nonlocal_or_secret_bearing_endpoint_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        LocalLLMClient(base_url="https://example.com/v1", cache_dir=tmp_path, log_path=tmp_path / "calls")
    with pytest.raises(ValueError):
        LocalLLMClient(base_url="http://user:secret@127.0.0.1/v1", cache_dir=tmp_path, log_path=tmp_path / "calls")


def test_api_controller_is_explicit_lazy_and_omits_vllm_fields(tmp_path, monkeypatch):
    secret, endpoint = "mock-api-key-never-log", "https://api.example.invalid/v1"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
    monkeypatch.setenv("HYPERBIND_API_MODEL", "gpt-4o-mini")
    transport = Transport(content="Response includes " + secret + " at " + endpoint)
    reader = api_controller_for_run(tmp_path, tmp_path / "run", transport=transport)
    assert transport.posts == []
    result = reader.complete(MESSAGES)
    assert "chat_template_kwargs" not in transport.posts[0]
    assert transport.posts[0]["model"] == "gpt-4o-mini"
    assert secret not in result.content and endpoint not in result.content
    for path in tmp_path.rglob("*.json*"):
        assert secret not in path.read_text() and endpoint not in path.read_text()


def test_remote_key_is_redacted_even_from_exception_text(tmp_path, monkeypatch):
    secret = "mock-api-secret-in-an-error"
    monkeypatch.setenv("KEY_NAME", secret)
    def failed(*args):
        raise RuntimeError("backend echoed " + secret)
    reader = LocalLLMClient(base_url="https://api.example.invalid/v1", model="gpt-4o-mini",
                            cache_dir=tmp_path / "cache", log_path=tmp_path / "calls.jsonl",
                            api_key_env="KEY_NAME", allow_remote=True, transport=failed)
    with pytest.raises(RuntimeError) as caught:
        reader.complete(MESSAGES)
    assert secret not in str(caught.value)
    assert secret not in (tmp_path / "calls.jsonl").read_text()


def test_api_controller_factory_requires_env_and_planned_model(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        api_controller_for_run(tmp_path, tmp_path / "run")
