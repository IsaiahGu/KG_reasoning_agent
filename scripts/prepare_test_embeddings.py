#!/usr/bin/env python
"""NV-Embed-v2 vectors for one more test subset of the HippoRAG 2 suite (NQ, PopQA), through the embedding service.

Writes the same files the P0 calibration wrote for the three multi-hop subsets, so SplitRetriever(split='test')
reads them unchanged: runs/p0/calibration/<dataset>/{passages,queries}.f16.npy with their progress files.
Passages are title + newline + text in corpus order; questions in file order. Resumable. No LLM call.
"""
import argparse
import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind.calibrate_dense import fetch, save  # noqa: E402
from hyperbind.data import dataset_name, load_corpus, load_test_samples  # noqa: E402


def encode(texts, path, kind, endpoint, settings, revision, batch_size):
    digest = hashlib.sha256(json.dumps({'texts': texts, 'kind': kind, 'settings': settings, 'model_revision': revision,
                                        'stored_dtype': 'float16'}, sort_keys=True).encode()).hexdigest()
    progress_path, completed, shape = path.with_suffix('.progress.json'), 0, (len(texts), settings['dimension'])
    if path.exists() and progress_path.exists():
        previous = json.loads(progress_path.read_text())
        if previous['fingerprint'] != digest:
            raise ValueError('Existing vectors were made from different inputs: %s' % path)
        completed, vectors = previous['completed'], np.load(path, mmap_mode='r+')
    else:
        if shutil.disk_usage(path.parent).free < 3 * 1024**3:
            raise RuntimeError('Encoding needs 3 GiB free disk')
        vectors = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=shape)
    for start in range(completed, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        response = fetch(endpoint + '/encode', {'texts': texts[start:end], 'kind': kind})
        result = np.frombuffer(base64.b64decode(response['data']), dtype='<f2').reshape(response['shape'])
        if result.shape != (end - start, settings['dimension']) or not np.isfinite(result).all():
            raise ValueError('Invalid vectors from the embedding service')
        vectors[start:end] = result
        if end == len(texts) or start % (batch_size * 50) == 0:
            vectors.flush()
            save(progress_path, {'fingerprint': digest, 'completed': end, 'total': len(texts), 'kind': kind,
                                 'settings': settings, 'model_revision': revision})
            print(json.dumps({'file': path.name, 'encoded': end, 'total': len(texts)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--endpoint', default='http://127.0.0.1:18081')
    parser.add_argument('--batch-size', type=int, default=4)
    arguments = parser.parse_args()
    root, dataset = arguments.root.resolve(), dataset_name(arguments.dataset)
    health = fetch(arguments.endpoint + '/health')
    reference = json.loads((root / 'runs/p0/calibration/musique/passages.f16.progress.json').read_text())
    settings = dict(health['settings'], batch_size=reference['settings']['batch_size'])
    if health['status'] != 'ready' or settings != reference['settings']:
        raise ValueError('The embedding service differs from the one that made the existing test vectors')
    corpus = load_corpus(root / 'data/hipporag2' / (dataset + '_corpus.json'))
    examples = load_test_samples(root, dataset)
    output = root / 'runs/p0/calibration' / dataset
    output.mkdir(parents=True, exist_ok=True)
    encode([document.content for document in corpus], output / 'passages.f16.npy', 'passage', arguments.endpoint, settings,
           reference['model_revision'], arguments.batch_size)
    encode([example.question for example in examples], output / 'queries.f16.npy', 'query', arguments.endpoint, settings,
           reference['model_revision'], arguments.batch_size)
    print(json.dumps({'status': 'complete', 'dataset': dataset, 'passages': len(corpus), 'questions': len(examples)}), flush=True)


if __name__ == '__main__':
    main()
