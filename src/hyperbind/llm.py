"""Logged, cached OpenAI-compatible requests to the local reader service."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

from filelock import FileLock
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def project_model_fingerprint(root: Path) -> dict:
    """Small pinned artifacts identify the weights, template and serving setup."""
    relative = [
        "models/Qwen/Qwen3.5-9B/config.json", "models/Qwen/Qwen3.5-9B/chat_template.jinja",
        "models/Qwen/Qwen3.5-9B/generation_config.json", "configs/env/serve.lock",
        "scripts/serve_local_llm.sh", "configs/serve_runtime.env",
    ]
    hashes = {name: file_sha256(root / name) for name in relative if (root / name).is_file()}
    inventory = root / "reports/model_inventory.json"
    revision = None
    if inventory.exists():
        revision = next((row["revision"] for row in json.loads(inventory.read_text())
                         if row["repo_id"] == "Qwen/Qwen3.5-9B"), None)
    return {"repo_id": "Qwen/Qwen3.5-9B", "revision": revision, "artifact_sha256": hashes}


class LLMError(RuntimeError):
    pass


class StructuredOutputError(LLMError):
    pass


def reject_nonfinite_constant(value: str):
    raise ValueError("Non-finite JSON number: " + value)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key: " + key)
        result[key] = value
    return result


@dataclass(frozen=True)
class LLMResult:
    content: str
    parsed: Any
    reasoning: str | None
    cache_key: str
    cache_hit: bool
    usage: Mapping[str, Any]
    elapsed_seconds: float
    finish_reason: str
    response: Mapping[str, Any]


def http_transport(method: str, url: str, payload: dict | None, headers: dict, timeout: float) -> dict:
    # Insertion order is kept on purpose: a constrained decoder emits JSON-schema properties in the order they are
    # sent, so sorting keys here made the model write `bindings` before `claims` (REVIEW_FOR_CODEX_P2.md, D1).
    data = None if payload is None else json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with urlopen(Request(url, data=data, headers=headers, method=method), timeout=timeout) as response:
        return json.load(response)


class LocalLLMClient:
    """One log row per logical call, with each HTTP retry preserved in the row."""

    def __init__(self, *, base_url: str, cache_dir: Path, log_path: Path,
                 model: str | None = None, model_fingerprint: Mapping[str, Any] | None = None,
                 generation: Mapping[str, Any] | None = None, api_key_env: str | None = None,
                 timeout: float = 900, retries: int = 2, transport: Callable = http_transport,
                 allow_remote: bool = False):
        parsed = urlparse(base_url)
        self.is_local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if self.is_local:
            if parsed.scheme != "http":
                raise ValueError("The local service uses loopback HTTP")
        elif not allow_remote or parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Remote API use requires explicit allow_remote=True and HTTPS")
        if not self.is_local and (not api_key_env or not model):
            raise ValueError("Remote API use requires an explicit model and credential environment variable")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use a plain loopback base URL; credentials come from an environment variable")
        self.base_url = base_url.rstrip("/")
        self.cache_dir, self.log_path = Path(cache_dir), Path(log_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.model_fingerprint = dict(model_fingerprint or {})
        self.generation = {"temperature": 0, "top_p": 1.0, "seed": 42, "max_tokens": 512}
        self.generation.update(generation or {})
        self.headers = {"Content-Type": "application/json"}
        self._secrets = [self.base_url] if not self.is_local else []
        if api_key_env:
            token = os.environ.get(api_key_env)
            if not token:
                raise ValueError(f"Missing required credential environment variable: {api_key_env}")
            self.headers["Authorization"] = "Bearer " + token
            self._secrets.append(token)
        self.timeout, self.retries, self.transport = timeout, retries, transport
        self._model_lock = threading.Lock()

    def _get_model(self) -> str:
        with self._model_lock:
            if self.model is None:
                models = self.transport("GET", self.base_url + "/models", None, self.headers, self.timeout)["data"]
                if len(models) != 1:
                    raise ValueError("Pass model explicitly when the server lists multiple models")
                self.model = models[0]["id"]
        return self.model

    def _log(self, row: dict) -> None:
        with FileLock(str(self.log_path) + ".lock", timeout=120):
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(canonical(self._sanitize(row)) + "\n")

    def _sanitize(self, value):
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, dict):
            return {self._sanitize(key): self._sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(item) for item in value]
        return value

    def complete(self, messages: Sequence[Mapping[str, str]], schema: dict | None = None, *,
                 thinking: bool | None = False, generation_overrides: Mapping[str, Any] | None = None,
                 metadata: Mapping[str, Any] | None = None, use_cache: bool = True) -> LLMResult:
        if not messages or any(set(message) != {"role", "content"} or not isinstance(message["content"], str)
                               for message in messages):
            raise ValueError("Expected nonempty text chat messages with role and content")
        generation = dict(self.generation)
        overrides = dict(generation_overrides or {})
        reserved = {"model", "messages", "response_format", "chat_template_kwargs", "stream"}
        if reserved.intersection(overrides):
            raise ValueError("Use the explicit model/schema/thinking arguments for reserved request fields")
        generation.update(overrides)
        try:
            resolved_model = self._get_model()
        except Exception as error:
            self._log({"call_id": uuid.uuid4().hex, "cache_key": None, "cache_hit": False,
                       "status": "error", "error_type": type(error).__name__, "error": "Model discovery failed",
                       "input_hash": digest(list(messages)), "metadata": dict(metadata or {}),
                       "http_attempts": [], "seconds": 0})
            raise LLMError("Local model discovery failed: " + type(error).__name__) from error
        payload = {"model": resolved_model, "messages": [dict(x) for x in messages], **generation}
        if not self.is_local and thinking is True:
            raise ValueError("Thinking mode overrides are only supported by the local controller")
        if self.is_local and thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if schema is not None:
            Draft202012Validator.check_schema(schema)
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "hyperbind_output", "strict": True, "schema": schema,
            }}
        identity = {"cache_version": 2, "base_url_hash": digest(self.base_url), "model_fingerprint": self.model_fingerprint,
                    "request": payload, "input_hash": digest(list(messages)), "schema_hash": digest(schema)}
        if schema is not None:
            # The declared property order is part of a structured request; responses cached under the old
            # sorted order must not be reused. Unstructured calls keep their old cache keys.
            identity["schema_declared_order_sha256"] = hashlib.sha256(
                json.dumps(schema, ensure_ascii=False).encode("utf-8")).hexdigest()
        key = digest(identity)
        cached_path = self.cache_dir / key[:2] / (key + ".json")
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"call_id": uuid.uuid4().hex, "cache_key": key, "model": self.model,
               "input_hash": identity["input_hash"], "schema_hash": identity["schema_hash"],
               "thinking": thinking, "metadata": dict(metadata or {}), "http_attempts": [],
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cache_hit": False}
        began = time.monotonic()
        response = None
        try:
            with FileLock(str(cached_path) + ".lock", timeout=self.timeout * (self.retries + 1)):
                if use_cache and cached_path.exists():
                    cached = json.loads(cached_path.read_text(encoding="utf-8"))
                    if cached["identity_hash"] != key or cached["response_hash"] != digest(cached["response"]):
                        raise LLMError("Cache identity mismatch")
                    response = cached["response"]
                    row["cache_hit"] = True
                else:
                    for attempt in range(self.retries + 1):
                        request_started = time.monotonic()
                        event = {"attempt": attempt + 1}
                        try:
                            response = self.transport("POST", self.base_url + "/chat/completions", payload,
                                                      self.headers, self.timeout)
                            event["status"] = "success"
                            row["http_attempts"].append(event)
                            break
                        except (HTTPError, URLError, TimeoutError, OSError) as error:
                            event.update(status="error", error_type=type(error).__name__,
                                         http_status=getattr(error, "code", None))
                            row["http_attempts"].append(event)
                            retryable = not isinstance(error, HTTPError) or error.code in {429, 500, 502, 503, 504}
                            if not retryable or attempt == self.retries:
                                raise LLMError(f"Local LLM transport failed: {type(error).__name__}, HTTP {getattr(error, 'code', None)}") from error
                            time.sleep(min(2**attempt, 8))
                        finally:
                            event["seconds"] = round(time.monotonic() - request_started, 6)
                    # Preserve successful raw responses even if subsequent schema validation fails.
                    response = self._sanitize(response)
                    save_json(cached_path, {"identity": self._sanitize(identity), "identity_hash": key,
                                           "response": response, "response_hash": digest(response)})
            choices = response.get("choices", [])
            if len(choices) != 1:
                raise LLMError("Expected exactly one chat completion")
            choice = choices[0]
            content = choice["message"].get("content")
            if not isinstance(content, str):
                raise LLMError("The model did not return textual content")
            reasoning = choice["message"].get("reasoning_content", choice["message"].get("reasoning"))
            finish_reason = choice.get("finish_reason", "unknown")
            parsed_output = None
            if schema is not None:
                if finish_reason != "stop":
                    raise StructuredOutputError("Constrained output did not finish normally: " + str(finish_reason))
                try:
                    parsed_output = json.loads(content, parse_constant=reject_nonfinite_constant,
                                               object_pairs_hook=reject_duplicate_keys)
                    Draft202012Validator(schema).validate(parsed_output)
                except (ValueError, TypeError, ValidationError) as error:
                    # No repair, fence stripping or unconstrained retry is permitted here.
                    raise StructuredOutputError("JSON/schema validation failed: " + type(error).__name__) from error
            usage = response.get("usage", {})
            row.update(status="success", finish_reason=finish_reason, usage=usage,
                       generated_tokens_this_call=0 if row["cache_hit"] else usage.get("completion_tokens", 0),
                       input_tokens_this_call=0 if row["cache_hit"] else usage.get("prompt_tokens", 0))
            return LLMResult(content, parsed_output, reasoning, key, row["cache_hit"], usage,
                             time.monotonic() - began, finish_reason, response)
        except Exception as error:
            row.update(status="error", error_type=type(error).__name__, error=str(error))
            if response is not None:
                row["usage"] = response.get("usage", {})
                row["generated_tokens_this_call"] = 0 if row["cache_hit"] else row["usage"].get("completion_tokens", 0)
                row["input_tokens_this_call"] = 0 if row["cache_hit"] else row["usage"].get("prompt_tokens", 0)
            if not self.is_local or self._sanitize(str(error)) != str(error):
                safe_type = StructuredOutputError if isinstance(error, StructuredOutputError) else LLMError
                raise safe_type(self._sanitize(str(error))) from None
            raise
        finally:
            row["seconds"] = round(time.monotonic() - began, 6)
            self._log(row)


def client_for_run(root: Path, run_dir: Path, *, base_url="http://127.0.0.1:18080/v1",
                   model: str | None = None, generation: dict | None = None,
                   server_profile: Mapping[str, Any] | None = None) -> LocalLLMClient:
    identity = project_model_fingerprint(root)
    identity["server_profile"] = dict(server_profile or {})
    return LocalLLMClient(base_url=base_url, model=model, model_fingerprint=identity,
                          generation=generation, cache_dir=root / "runs/cache/llm",
                          log_path=run_dir / "llm_calls.jsonl")


def api_controller_for_run(root: Path, run_dir: Path, *, transport: Callable = http_transport) -> LocalLLMClient:
    """Construct without calling the API. The runner may use it only after the local G2 failure."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("HYPERBIND_API_MODEL", "gpt-4o-mini")
    if not base_url or not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("G2 fallback requires OPENAI_BASE_URL and OPENAI_API_KEY in the environment")
    if model != "gpt-4o-mini":
        raise ValueError("G2 fallback requires the planned gpt-4o-mini model")
    identity = {"purpose": "G2 API controller retry", "requested_model": model,
                "endpoint_sha256": digest(base_url)}
    return LocalLLMClient(base_url=base_url, model=model, model_fingerprint=identity,
                          generation={"temperature": 0, "top_p": 1.0, "seed": 42, "max_tokens": 2048},
                          cache_dir=root / "runs/cache/llm_api_controller", log_path=run_dir / "llm_calls.jsonl",
                          api_key_env="OPENAI_API_KEY", allow_remote=True, transport=transport)


def summarize_calls(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    return {"logical_calls": len(rows), "cache_hits": sum(bool(row.get("cache_hit")) for row in rows),
            "http_attempts": sum(len(row.get("http_attempts", [])) for row in rows),
            "errors": sum(row.get("status") == "error" for row in rows),
            "prompt_tokens_generated": sum(row.get("input_tokens_this_call", 0) for row in rows),
            "completion_tokens_generated": sum(row.get("generated_tokens_this_call", 0) for row in rows),
            "seconds_sum": sum(row.get("seconds", 0) for row in rows)}


def verify_default_thinking_off(client: LocalLLMClient) -> dict:
    result = client.complete([{"role": "user", "content": "What is 1 + 1? Give only the number."}],
                             thinking=None, generation_overrides={"max_tokens": 64}, use_cache=False,
                             metadata={"phase": "R2", "request_omits_chat_template_kwargs": True})
    assert "<think>" not in result.content.lower() and "</think>" not in result.content.lower(), result.content
    assert not result.reasoning or not result.reasoning.strip(), "Reasoning content was returned with default settings"
    assert result.finish_reason == "stop", result.finish_reason
    return {"request_omits_chat_template_kwargs": True, "response_has_think_tags": False,
            "reasoning_content_empty": True, "content": result.content, "usage": dict(result.usage),
            "cache_key": result.cache_key, "status": "passed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace/hg_reasoning_agent"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--model")
    parser.add_argument("--verify-default-thinking", action="store_true", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir or args.root / "runs/p0_fixes/r2"
    result = verify_default_thinking_off(client_for_run(args.root, run_dir, base_url=args.base_url, model=args.model))
    save_json(run_dir / "default_thinking.json", result)
    print(canonical(result))


if __name__ == "__main__":
    main()
