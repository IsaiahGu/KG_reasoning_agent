"""Offline NV-Embed-v2 vectors for any (corpus, question file) pair, e.g. train-S. No LLM calls.

Same encoder settings as the dev vectors (prepare_dev_embeddings.py); run it in the embed
environment with vLLM stopped. Output: passages.f16.npy, queries.f16.npy, metadata.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    temporary.replace(path)


def run(root, dataset, corpus_path, queries_path, destination, limit=None):
    allocation = 'expandable_segments:True'
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', allocation)
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                      '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('Offline encoding needs an idle GPU: stop vLLM and the embedding service first')
    if shutil.disk_usage(root).free < 10 * 1024**3:
        raise RuntimeError('Offline encoding needs 10 GiB free disk')
    import numpy as np
    from .nv_adapter import NVEncoder
    encoder = NVEncoder(root, batch_size=4, max_length=2048)
    revision = next(item['revision'] for item in json.loads((root / 'reports/model_inventory.json').read_text())
                    if item['repo_id'] == 'nvidia/NV-Embed-v2')
    corpus = json.loads(corpus_path.read_text())
    queries = json.loads(queries_path.read_text())
    if limit:
        queries = queries[:limit]
    documents = [item['title'] + '\n' + item['text'] for item in corpus]
    query_texts = [item['question'] for item in queries]
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {'dataset': dataset, 'settings': encoder.settings, 'model_revision': revision, 'dtype': 'float16',
                'dimension': 4096, 'passages': len(documents), 'queries': len(queries),
                'sources': {'corpus': {'path': str(corpus_path.relative_to(root)), 'sha256': digest(corpus_path)},
                            'queries': {'path': str(queries_path.relative_to(root)), 'sha256': digest(queries_path)}},
                'passage_ids': [hashlib.sha256(text.encode()).hexdigest() for text in documents],
                'query_ids': [str(item.get('id', item.get('_id'))) for item in queries],
                'passages_file': 'passages.f16.npy', 'queries_file': 'queries.f16.npy'}
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata_path = destination / 'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Existing embedding folder was built from different inputs: ' + str(destination))
    write(metadata_path, {**metadata, 'fingerprint': fingerprint, 'status': 'encoding'})
    for kind, texts, filename in (('passage', documents, 'passages.f16.npy'), ('query', query_texts, 'queries.f16.npy')):
        path = destination / filename
        progress = path.with_suffix('.progress.json')
        start = 0
        if path.exists() and progress.exists():
            start = json.loads(progress.read_text())['completed']
            array = np.load(path, mmap_mode='r+')
            assert array.shape == (len(texts), 4096) and array.dtype == np.float16
        else:
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(len(texts), 4096))
        for begin in range(start, len(texts), 4):
            end = min(begin + 4, len(texts))
            array[begin:end] = encoder.encode(texts[begin:end], kind).astype(np.float16)
            if begin % 400 == 0 or end == len(texts):
                array.flush()
                write(progress, {'completed': end, 'total': len(texts)})
                print(json.dumps({'dataset': dataset, 'kind': kind, 'completed': end, 'total': len(texts)}), flush=True)
    write(metadata_path, {**metadata, 'fingerprint': fingerprint, 'status': 'complete',
                          'files_sha256': {name: digest(destination / name) for name in ('passages.f16.npy', 'queries.f16.npy')}})
    print(json.dumps({'status': 'complete', 'dataset': dataset, 'destination': str(destination)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/hg_reasoning_agent'))
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', default='train-S', help='question file stem and corpus prefix under data/splits/seed42/<dataset>/')
    parser.add_argument('--limit', type=int)
    arguments = parser.parse_args()
    folder = arguments.root / 'data/splits/seed42' / arguments.dataset
    run(arguments.root, arguments.dataset, folder / (arguments.split + '_corpus.json'), folder / (arguments.split + '.json'),
        arguments.root / 'runs/train_s/embeddings' / arguments.dataset, arguments.limit)
