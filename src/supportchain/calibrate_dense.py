"""Test vectors of the three multi-hop subsets (NV-Embed-v2) and the official dense Recall@5 check. No generation."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

import numpy as np

from .hipporag_upstream import gold_function, query_instruction, recall_metric, score_normalizer

TARGETS = {'musique': 69.7, '2wikimultihopqa': 76.5, 'hotpotqa': 94.5}


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def fetch(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=900) as response:
        return json.load(response)


def encode_to_file(texts, path: Path, kind, endpoint, settings, model_revision, batch_size):
    digest = hashlib.sha256(json.dumps({'texts': texts, 'kind': kind, 'settings': settings,
        'model_revision': model_revision, 'stored_dtype': 'float16'}, sort_keys=True).encode()).hexdigest()
    progress_path = path.with_suffix('.progress.json')
    completed = 0
    shape = (len(texts), settings['dimension'])
    if path.exists() and progress_path.exists():
        previous = json.loads(progress_path.read_text())
        if previous['fingerprint'] != digest:
            raise ValueError(f'Existing embedding cache settings mismatch: {path}')
        completed = previous['completed']
        vectors = np.load(path, mmap_mode='r+')
        assert vectors.shape == shape and vectors.dtype == np.float16
    else:
        if shutil.disk_usage(path.parent).free < 15 * 1024**3:
            raise RuntimeError('Indexing requires at least 15 GiB free disk')
        vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=shape)
    for start in range(completed, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        response = fetch(endpoint + '/encode', {'texts': texts[start:end], 'kind': kind})
        assert response['dtype'] == 'float16' and response['shape'] == [end - start, settings['dimension']]
        result = np.frombuffer(base64.b64decode(response['data']), dtype='<f2').reshape(response['shape'])
        assert np.isfinite(result).all()
        vectors[start:end] = result
        vectors.flush()
        save(progress_path, {'fingerprint': digest, 'completed': end, 'total': len(texts),
                             'kind': kind, 'settings': settings, 'model_revision': model_revision})
        if end == len(texts) or start % (batch_size * 25) == 0:
            print(json.dumps({'file': path.name, 'encoded': end, 'total': len(texts)}), flush=True)
    return vectors


def rank(corpus_vectors, query_vectors, k=5, normalize=None):
    """Same float32 dot, per-query minmax and reversed argsort as upstream."""
    documents = np.asarray(corpus_vectors, dtype=np.float32)
    results = []
    for query in query_vectors:
        scores = np.dot(documents, np.asarray(query, dtype=np.float32).T)
        if normalize is not None:
            scores = normalize(scores)
        results.append(np.argsort(scores)[::-1][:k].tolist())
    return results


def calibrate(root: Path, endpoint: str, batch_size: int):
    started = time.time()
    endpoint = endpoint.rstrip('/')
    health = fetch(endpoint + '/health')
    settings = health['settings']
    assert health['status'] == 'ready'
    assert settings['max_length'] == 2048 and settings['normalize'] is True
    assert settings['query_instruction'] == 'Instruct: ' + query_instruction(root) + '\nQuery: '
    assert settings['passage_instruction'] == '' and settings['dimension'] == 4096
    inventory = json.loads((root / 'configs/models.json').read_text())
    revision = next(x['revision'] for x in inventory if x['repo_id'] == 'nvidia/NV-Embed-v2')
    output = root / 'runs/test_vectors'
    output.mkdir(parents=True, exist_ok=True)
    gold_builder = gold_function(root)
    metric = recall_metric(root)
    normalizer = score_normalizer(root)
    summary = {'scope': 'dense retrieval vectors and Recall@5 check', 'settings': settings,
        'model_revision': revision, 'storage_dtype': 'float16', 'dot_product_dtype': 'float32',
        'target_tolerance_percentage_points': 1.5, 'datasets': {},
        'no_generation_no_api_calls': True}
    for dataset, target in TARGETS.items():
        if shutil.disk_usage(root).free < 15 * 1024**3:
            raise RuntimeError('Indexing requires at least 15 GiB free disk')
        corpus_path = root / 'data/hipporag2' / (dataset + '_corpus.json')
        sample_path = root / 'data/hipporag2' / (dataset + '.json')
        corpus = json.loads(corpus_path.read_text())
        samples = json.loads(sample_path.read_text())
        documents = [row['title'] + '\n' + row['text'] for row in corpus]
        queries = [row['question'] for row in samples]
        gold = gold_builder(samples, dataset)
        missing = sorted({doc for group in gold for doc in group} - set(documents))
        if missing:
            raise ValueError(f'{dataset}: {len(missing)} gold documents missing from corpus')
        data_dir = output / dataset
        data_dir.mkdir(exist_ok=True)
        passage_vectors = encode_to_file(documents, data_dir / 'passages.f16.npy', 'passage', endpoint,
                                        settings, revision, batch_size)
        query_vectors = encode_to_file(queries, data_dir / 'queries.f16.npy', 'query', endpoint,
                                      settings, revision, batch_size)
        indices = rank(passage_vectors, query_vectors, normalize=normalizer)
        retrieved = [[documents[idx] for idx in group] for group in indices]
        aggregate, per_question = metric.calculate_metric_scores(gold, retrieved, k_list=[5])
        recall = aggregate['Recall@5'] * 100
        per_query = []
        for sample, ids, score, gold_docs in zip(samples, indices, per_question, gold):
            per_query.append({'id': sample.get('id', sample.get('_id')), 'top5_corpus_indices': ids,
                              'recall_at_5': score['Recall@5'],
                              'all_supporting_retrieved': set(gold_docs).issubset({documents[i] for i in ids})})
        save(data_dir / 'per_question.json', per_query)
        result = {'questions': len(samples), 'passages': len(corpus), 'recall_at_5_percent': recall,
                  'target_percent': target, 'difference_percentage_points': round(recall - target, 4),
                  'within_tolerance': abs(recall - target) <= 1.5,
                  'all_supporting_at_5_percent': 100 * sum(x['all_supporting_retrieved'] for x in per_query) / len(samples),
                  'gold_missing': len(missing), 'input_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                                                for path in (corpus_path, sample_path)}}
        summary['datasets'][dataset] = result
        save(output / 'summary.json', summary)
        print(json.dumps({'dataset': dataset, **result}), flush=True)
    summary['recall_matches_published'] = all(x['within_tolerance'] for x in summary['datasets'].values())
    summary['seconds'] = time.time() - started
    summary['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(output / 'summary.json', summary)
    print(json.dumps({'recall_matches_published': summary['recall_matches_published']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--endpoint', default='http://127.0.0.1:18081')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    calibrate(args.root, args.endpoint, args.batch_size)
