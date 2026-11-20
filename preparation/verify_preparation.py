"""Static artifact checks only: no model loading, inference, indexing or evaluation."""
import json
import os
from pathlib import Path
import struct
import time

ROOT = Path('/workspace/hg_reasoning_agent')
os.environ['HF_HOME'] = str(ROOT / 'preparation/hf_cache')
os.environ['HF_HUB_OFFLINE'] = '1'
from huggingface_hub import try_to_load_from_cache

models = json.loads((ROOT / 'reports/model_inventory.json').read_text())
checks = []
for model in models:
    local = Path(model['local_dir'])
    status = json.loads((ROOT / 'reports' / (model['repo_id'].replace('/', '__') + '.json')).read_text())
    assert status['status'] == 'complete', model['repo_id']
    for item in model['files']:
        path = local / item['path']
        assert path.is_file() and path.stat().st_size == item['size'], str(path)
        if path.name.endswith('.safetensors.index.json'):
            index = json.loads(path.read_text())
            for shard in set(index['weight_map'].values()):
                assert (local / shard).is_file(), shard
        if path.suffix == '.safetensors':
            with path.open('rb') as stream:
                header_len = struct.unpack('<Q', stream.read(8))[0]
                assert 0 < header_len < 200_000_000
                header = json.loads(stream.read(header_len))
            for name, tensor in header.items():
                if name == '__metadata__':
                    continue
                start, end = tensor['data_offsets']
                assert 0 <= start <= end <= path.stat().st_size - 8 - header_len, name
    checks.append({'repo_id': model['repo_id'], 'status': 'complete',
                   'files': len(model['files']), 'bytes': model['expected_bytes'],
                   'verification': 'upstream SHA256/git blob hash during download; file sizes, shard references and safetensors headers checked'})
for repo, filenames in {
    'nvidia/NV-Embed-v2': ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'tokenizer.model'],
}.items():
    for filename in filenames:
        assert isinstance(try_to_load_from_cache(repo, filename, revision='main'), str), (repo, filename)
incomplete = list((ROOT / 'models').rglob('*.incomplete'))
assert not incomplete, 'Incomplete model transfers remain'
report = {'verified_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'models': checks, 'model_bytes': sum(m['expected_bytes'] for m in models),
          'tokenizer_dependencies_cached': True,
          'scope': 'Static artifact verification only. No model loaded and no experiment started.'}
(ROOT / 'reports/static_verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'model_count': len(checks), 'verified_model_bytes': report['model_bytes'],
                  'tokenizer_dependencies_cached': True, 'experiments_started': False}))
