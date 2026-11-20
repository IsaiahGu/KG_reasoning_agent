"""R3 offline NV profile and reusable development embeddings; no LLM calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def run(root, datasets):
    allocation = 'expandable_segments:True'
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF', allocation) != allocation:
        raise ValueError('R3 requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = allocation
    active_gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                         '--format=csv,noheader'], text=True).strip()
    if active_gpu:
        raise RuntimeError('Offline encoding requires the GPU idle before loading NV; stop vLLM and other model services')
    import numpy as np
    import torch
    from .nv_adapter import NVEncoder, MemoryMonitor
    if shutil.disk_usage(root).free < 15 * 1024**3:
        raise RuntimeError('Offline indexing requires 15 GiB free disk')
    encoder = NVEncoder(root, batch_size=4, max_length=2048)
    revision = next(item['revision'] for item in json.loads((root / 'reports/model_inventory.json').read_text())
                    if item['repo_id'] == 'nvidia/NV-Embed-v2')
    source = root / 'data/splits/seed42'
    output = root / 'runs/p0_dev/embeddings'
    output.mkdir(parents=True, exist_ok=True)
    profile = None
    for dataset in datasets:
        if shutil.disk_usage(root).free < 15 * 1024**3:
            raise RuntimeError('Offline indexing requires 15 GiB free disk')
        corpus_path = source / dataset / 'dev_corpus.json'
        queries_path = source / dataset / 'dev-A.json'
        corpus = json.loads(corpus_path.read_text())
        queries = json.loads(queries_path.read_text())
        assert len(queries) == 500
        documents = [item['title'] + '\n' + item['text'] for item in corpus]
        query_texts = [item['question'] for item in queries]
        query_ids = [str(item.get('id', item.get('_id'))) for item in queries]
        destination = output / dataset
        destination.mkdir(exist_ok=True)
        metadata = {'dataset': dataset, 'settings': encoder.settings, 'model_revision': revision,
                    'dtype': 'float16', 'dimension': 4096,
                    'sources': {'corpus': {'path': str(corpus_path.relative_to(root)), 'sha256': digest(corpus_path)},
                                'queries': {'path': str(queries_path.relative_to(root)), 'sha256': digest(queries_path)}},
                    'passage_ids': [hashlib.sha256(text.encode()).hexdigest() for text in documents],
                    'query_ids': query_ids, 'passages_file': 'passages.f16.npy',
                    'queries_file': 'queries_dev-A.f16.npy', 'passages': len(documents), 'queries': len(queries),
                    'allocator_config': allocation, 'status': 'encoding'}
        fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        metadata['fingerprint'] = fingerprint
        metadata_path = destination / 'metadata.json'
        if metadata_path.exists() and json.loads(metadata_path.read_text())['fingerprint'] != fingerprint:
            raise ValueError(f'Existing development embedding cache does not match inputs: {dataset}')
        write(metadata_path, metadata)
        for kind, texts, filename in [('passage', documents, 'passages.f16.npy'),
                                      ('query', query_texts, 'queries_dev-A.f16.npy')]:
            path = destination / filename
            progress = path.with_suffix('.progress.json')
            start = 0
            if path.exists() and progress.exists():
                prior = json.loads(progress.read_text())
                assert prior['fingerprint'] == fingerprint
                start = prior['completed']
                array = np.load(path, mmap_mode='r+')
                assert array.shape == (len(texts), 4096) and array.dtype == np.float16
            else:
                array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(len(texts), 4096))
            if profile is None and kind == 'passage' and start == 0:
                torch.cuda.reset_peak_memory_stats()
                began = time.time()
                with MemoryMonitor() as monitor:
                    for begin in range(0, min(200, len(texts)), 4):
                        end = min(begin + 4, 200, len(texts))
                        array[begin:end] = encoder.encode(texts[begin:end], kind).astype(np.float16)
                    torch.cuda.synchronize()
                    profile = {'dataset': dataset, 'passages': min(200, len(texts)),
                               'source_sha256': digest(corpus_path), 'settings': encoder.settings,
                               'allocator_config': allocation, 'seconds': time.time() - began,
                               'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                profile['nvidia_smi_peak_mib'] = monitor.peak_mib
                profile['vllm_stopped'] = True
                write(root / 'reports/p0_nv_offline_profile.json', profile)
                print(json.dumps({'offline_profile': profile}), flush=True)
                start = min(200, len(texts))
                array.flush()
                write(progress, {'fingerprint': fingerprint, 'completed': start, 'total': len(texts)})
            for begin in range(start, len(texts), 4):
                end = min(begin + 4, len(texts))
                array[begin:end] = encoder.encode(texts[begin:end], kind).astype(np.float16)
                array.flush()
                write(progress, {'fingerprint': fingerprint, 'completed': end, 'total': len(texts)})
                if begin % 200 == 0 or end == len(texts):
                    print(json.dumps({'dataset': dataset, 'kind': kind, 'completed': end, 'total': len(texts)}), flush=True)
        metadata['status'] = 'complete'
        metadata['files_sha256'] = {filename: digest(destination / filename) for filename in
                                    ('passages.f16.npy', 'queries_dev-A.f16.npy')}
        write(metadata_path, metadata)
    print(json.dumps({'status': 'complete', 'datasets': datasets}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/hg_reasoning_agent'))
    parser.add_argument('--datasets', nargs='+', default=['musique', '2wikimultihopqa', 'hotpotqa'])
    arguments = parser.parse_args()
    run(arguments.root, arguments.datasets)
