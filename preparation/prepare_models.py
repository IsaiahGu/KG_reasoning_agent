"""Download and verify upstream model artifacts only; never load or run a model."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path('/workspace/hg_reasoning_agent')
os.environ['HF_HOME'] = str(ROOT / 'preparation/hf_cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_XET_CACHE'] = str(ROOT / 'preparation/xet_cache')
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
from huggingface_hub import HfApi, hf_hub_download

MODELS = [
    'Qwen/Qwen3.5-9B', 'nvidia/NV-Embed-v2',
    'Qwen/Qwen3-Reranker-0.6B', 'urchade/gliner_large-v2.1',
    'Qwen/Qwen3-Embedding-4B', 'Qwen/Qwen3-Embedding-8B',
    'Qwen/Qwen3-Reranker-4B',
]

def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def authenticate():
    tokens = json.loads(Path('/root/.config/hg_prepare/hf_tokens.json').read_text(encoding='utf-8-sig'))
    if isinstance(tokens, str):
        tokens = [tokens]
    for token in tokens:
        try:
            HfApi(token=token).whoami()
            os.environ['HF_TOKEN'] = token
            print('HF authentication: OK', flush=True)
            return
        except Exception:
            continue
    raise RuntimeError('No provided HF token authenticated successfully')

def inventory():
    api = HfApi(token=os.environ['HF_TOKEN'])
    result = []
    for repo in MODELS:
        info = api.model_info(repo, files_metadata=True)
        safe = any(f.rfilename.endswith('.safetensors') for f in info.siblings)
        files = []
        for f in info.siblings:
            name = f.rfilename
            if name.startswith(('onnx/', 'openvino/', 'tensorrt/', 'imgs/', 'images/', 'assets/')):
                continue
            if name.endswith(('.onnx', '.h5', '.msgpack', '.ot', '.png', '.jpg', '.jpeg', '.svg')):
                continue
            if safe and (name.endswith('.bin') or 'pytorch_model.bin.index' in name):
                continue
            lfs = f.lfs
            sha = getattr(lfs, 'sha256', None) if lfs else None
            files.append({'path': name, 'size': f.size, 'lfs_sha256': sha, 'git_blob_sha1': f.blob_id})
        record = {'repo_id': repo, 'revision': info.sha, 'source': 'https://huggingface.co/' + repo,
                  'local_dir': str(ROOT / 'models' / repo), 'files': files,
                  'expected_bytes': sum(f['size'] or 0 for f in files)}
        result.append(record)
        print(f"INVENTORY {repo} {record['expected_bytes']} bytes {len(files)} files", flush=True)
    save(ROOT / 'reports/model_inventory.json', result)
    free = shutil.disk_usage(ROOT).free
    print(f"TOTAL_MODELS {sum(m['expected_bytes'] for m in result)} FREE {free}", flush=True)
    return result

def download(model):
    directory = Path(model['local_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    status = {'repo_id': model['repo_id'], 'revision': model['revision'], 'status': 'downloading',
              'completed_files': 0, 'expected_files': len(model['files']), 'errors': []}
    status_path = ROOT / 'reports' / (model['repo_id'].replace('/', '__') + '.json')
    save(status_path, status)
    try:
        for entry in sorted(model['files'], key=lambda f: f['size'] or 0):
            path = None
            for attempt in range(4):
                try:
                    path = Path(hf_hub_download(model['repo_id'], entry['path'], revision=model['revision'],
                                token=os.environ['HF_TOKEN'], local_dir=directory))
                    if entry['size'] is not None and path.stat().st_size != entry['size']:
                        raise RuntimeError('size mismatch')
                    if entry['lfs_sha256']:
                        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
                        if digest != entry['lfs_sha256']:
                            raise RuntimeError('sha256 mismatch')
                    else:
                        digest_obj = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
                        with path.open('rb') as stream:
                            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                                digest_obj.update(block)
                        if entry['git_blob_sha1'] and digest_obj.hexdigest() != entry['git_blob_sha1']:
                            raise RuntimeError('git blob hash mismatch')
                    break
                except Exception as error:
                    print(f"RETRY {model['repo_id']} {entry['path']} {type(error).__name__} attempt {attempt + 1}", flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(3 * (attempt + 1))
            status['completed_files'] += 1
            status['last_file'] = entry['path']
            save(status_path, status)
        status['status'] = 'complete'
        status['verified_bytes'] = model['expected_bytes']
        print(f"COMPLETE {model['repo_id']} {model['expected_bytes']} bytes", flush=True)
    except Exception as error:
        status['status'] = 'failed'
        status['errors'].append(type(error).__name__)
        print(f"FAILED {model['repo_id']} {type(error).__name__}", flush=True)
    save(status_path, status)
    return status

if __name__ == '__main__':
    authenticate()
    models = inventory()
    if '--inventory-only' not in sys.argv:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            outcomes = list(pool.map(download, models))
        save(ROOT / 'reports/models_complete.json', outcomes)
        sys.exit(0 if all(x['status'] == 'complete' for x in outcomes) else 1)
