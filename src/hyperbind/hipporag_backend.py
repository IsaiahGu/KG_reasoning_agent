"""Thin local I/O adapters; graph, OpenIE, prompts and PPR stay upstream."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

STRAY_KEY_DOT = re.compile(r'^(\s*\{\s*")\.(?=[A-Za-z_]+"\s*:)')


def repair_first_key(text):
    """Remove a stray dot in front of the first key of a JSON-object response: {".named_entities": [...]}.

    Since 2026-11-17 the model server forbids whitespace between JSON tokens (structured-outputs setting needed by the
    chain agent). In JSON-object mode Qwen3.5-9B then opens the first key with a dot, which the upstream parser does not
    accept. The development run of this baseline (2026-11-16) used the earlier server setting and had no such responses.
    Only the dot is removed; the content of the response is untouched."""
    return STRAY_KEY_DOT.sub(r'\1', text) if isinstance(text, str) else text


def request(url, payload=None):
    if urlparse(url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('P0 local baseline adapters prohibit external endpoints')
    body = None if payload is None else json.dumps(payload).encode()
    with urlopen(Request(url, data=body, headers={'Content-Type': 'application/json'}), timeout=900) as response:
        return json.load(response)


class LoggedOfficialLLM:
    """Delegate to official CacheOpenAI, preserving prompts and cache semantics."""
    def __init__(self, config, folder, stage, backend=None):
        from hipporag.llm.openai_gpt import CacheOpenAI
        if urlparse(config.llm_base_url).hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Only the local Qwen endpoint is authorized')
        self.global_config = config
        self.stage = stage
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.backend = backend or CacheOpenAI(cache_dir=str(self.folder / 'llm_cache'),
                                             global_config=config, max_retries=0)

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def infer(self, messages=None, **kwargs):
        started = time.time()
        signature = hashlib.sha256(json.dumps({'messages': messages, 'kwargs': kwargs}, sort_keys=True).encode()).hexdigest()
        record = {'stage': self.stage, 'request_sha256': signature, 'model': self.global_config.llm_name,
                  'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            text, metadata, cache_hit = self.backend.infer(messages=messages, **kwargs)
            if getattr(self.global_config, 'response_format', None) == {'type': 'json_object'}:
                repaired = repair_first_key(text)
                record['first_key_repaired'], text = repaired != text, repaired
            record.update({'status': 'success', 'cache_hit': bool(cache_hit), 'usage': metadata})
            return text, metadata, cache_hit
        except Exception as error:
            record.update({'status': 'error', 'error_type': type(error).__name__})
            raise
        finally:
            record['seconds'] = time.time() - started
            with self.lock, (self.folder / 'llm_calls.jsonl').open('a') as log:
                log.write(json.dumps(record) + '\n')

    def close(self):
        self.backend.close()


def call_statistics(path):
    calls = [] if not Path(path).exists() else [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    result = {'logical_calls': len(calls), 'cache_hits': sum(bool(row.get('cache_hit')) for row in calls),
              'successful_requests': sum(row['status'] == 'success' and not row.get('cache_hit') for row in calls),
              'errors': sum(row['status'] == 'error' for row in calls), 'summed_request_seconds': sum(row['seconds'] for row in calls)}
    for token_type in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        result[token_type] = sum(row.get('usage', {}).get(token_type, 0) for row in calls if not row.get('cache_hit'))
    return result


def phase_call_delta(before, after):
    return {key: after[key] - before.get(key, 0) for key in after
            if isinstance(after[key], (int, float))}


def require_clean_phase_calls(*deltas):
    failures = sum(delta.get('errors', 0) for delta in deltas)
    if failures:
        raise RuntimeError(f'{failures} operational LLM errors occurred in this phase; native recognition fallback must not mask them')
    return 0


class FP16Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, instruction TEXT, text TEXT, vector BLOB)')

    @staticmethod
    def key(text, instruction=''):
        return hashlib.sha256(json.dumps([instruction, text], ensure_ascii=False).encode()).hexdigest()

    def get(self, texts, instruction=''):
        with sqlite3.connect(self.path) as connection:
            rows = [connection.execute('SELECT vector FROM vectors WHERE key=?', (self.key(text, instruction),)).fetchone() for text in texts]
        return [None if row is None else np.frombuffer(row[0], dtype='<f2').astype(np.float32) for row in rows]

    def put(self, texts, vectors, instruction=''):
        array = np.asarray(vectors, dtype='<f2')
        if array.ndim != 2 or array.shape[0] != len(texts) or not np.isfinite(array).all():
            raise ValueError('Invalid cache vectors')
        with sqlite3.connect(self.path) as connection:
            connection.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?,?,?)',
                [(self.key(text, instruction), instruction, text, vector.tobytes()) for text, vector in zip(texts, array)])


class CachedNVEmbedding:
    """Official embedding interface, backed by offline FP16 cache and query HTTP."""
    query_instruction_mode = 'distinct'

    def __init__(self, config, cache, endpoint=None):
        from hipporag.embedding_model.base import EmbeddingConfig
        self.global_config = config
        self.embedding_model_name = config.embedding_model_name
        self.embedding_dim = 4096
        self.embedding_config = EmbeddingConfig.from_dict({'norm': True, 'embedding_model_name': self.embedding_model_name})
        self.cache = cache
        self.endpoint = endpoint

    def batch_encode(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else list(texts)
        instruction = kwargs.get('instruction', '')
        vectors = self.cache.get(texts, instruction)
        missing = [index for index, vector in enumerate(vectors) if vector is None]
        if missing and (not instruction or not self.endpoint):
            raise RuntimeError(f'{len(missing)} vectors are not preencoded; offline NV encoding is required')
        for begin in range(0, len(missing), 4):
            indices = missing[begin:begin + 4]
            batch = [texts[index] for index in indices]
            response = request(self.endpoint.rstrip('/') + '/encode',
                               {'texts': batch, 'kind': 'query', 'instruction': instruction})
            result = np.frombuffer(base64.b64decode(response['data']), dtype='<f2').reshape(response['shape']).astype(np.float32)
            self.cache.put(batch, result, instruction)
            for index, vector in zip(indices, result):
                vectors[index] = vector
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.stack(vectors).astype(np.float32)

    def close(self):
        pass
