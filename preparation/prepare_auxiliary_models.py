"""Cache tokenizer/config dependencies without downloading duplicate base weights."""
import hashlib
import json
import os
from pathlib import Path
from prepare_models import ROOT, authenticate, save
from huggingface_hub import HfApi, hf_hub_download

authenticate()
api = HfApi(token=os.environ['HF_TOKEN'])
results = []
for repo in ['microsoft/deberta-v3-large', 'nvidia/NV-Embed-v2']:
    info = api.model_info(repo, files_metadata=True)
    files = []
    for item in info.siblings:
        name = item.rfilename
        if item.size is None or item.size > 20_000_000:
            continue
        if name.endswith(('.safetensors', '.bin', '.onnx', '.h5', '.msgpack', '.ot')):
            continue
        cached = Path(hf_hub_download(repo, name, revision=info.sha, token=os.environ['HF_TOKEN']))
        expected = getattr(item.lfs, 'sha256', None) if item.lfs else None
        with cached.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if expected and digest != expected:
            raise RuntimeError('Dependency hash mismatch')
        if cached.stat().st_size != item.size:
            raise RuntimeError('Dependency size mismatch')
        files.append({'path': name, 'size': item.size, 'sha256': digest, 'cached_path': str(cached)})
        if repo == 'microsoft/deberta-v3-large':
            local = Path(hf_hub_download(repo, name, revision=info.sha, token=os.environ['HF_TOKEN'],
                         local_dir=ROOT / 'models' / repo))
    # Populate refs/main so hardcoded upstream tokenizer repo IDs also work offline.
    ref = Path(os.environ['HF_HOME']) / 'hub' / ('models--' + repo.replace('/', '--')) / 'refs/main'
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(info.sha)
    results.append({'repo_id': repo, 'revision': info.sha, 'files': files,
                    'scope': 'tokenizer, config and custom source only; no duplicate base weights'})
    print(f'AUX_COMPLETE {repo} {len(files)} files', flush=True)
save(ROOT / 'reports/auxiliary_models.json', results)
(ROOT / 'preparation/offline_env.sh').write_text(
    '# Source only when starting later offline work; no services are started.\n'
    f'export HF_HOME={os.environ["HF_HOME"]}\n'
    'export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\n')
