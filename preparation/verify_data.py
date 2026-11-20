"""Structural validation of downloaded data; no model, index or evaluation calls."""
import json
from pathlib import Path
import time

ROOT = Path('/workspace/hg_reasoning_agent')
DATA = ROOT / 'data'
summary = {'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'scope': 'file integrity and structure only; no experiments executed',
           'datasets': {}, 'limitations': []}

def j(path):
    with path.open() as f:
        return json.load(f)

def add_json(label, path, expected=None):
    rows = j(path)
    count = len(rows)
    if expected is not None:
        assert count == expected, (label, count, expected)
    summary['datasets'][label] = {'path': str(path.relative_to(DATA)), 'records': count}
    return rows

for name, expected in [('musique', 11656), ('2wikimultihopqa', 6119), ('hotpotqa', 9811), ('nq_rear', 9633), ('popqa', 8676)]:
    add_json('hipporag2_' + name + '_test', DATA / 'hipporag2' / (name + '.json'), 1000)
    add_json('hipporag2_' + name + '_corpus', DATA / 'hipporag2' / (name + '_corpus.json'), expected)

for p in sorted((DATA / 'raw/2wikimultihop/extracted').rglob('*.json')):
    if p.name in ['train.json', 'dev.json', 'test.json']:
        rows = add_json('original_2wiki_' + p.stem, p)
        ids = [x['_id'] for x in rows]
        assert len(ids) == len(set(ids))
        summary['datasets']['original_2wiki_' + p.stem]['unique_ids'] = len(set(ids))
        del rows, ids

for p in sorted((DATA / 'raw/musique/extracted').rglob('*.jsonl')):
    if '__MACOSX' in str(p):
        continue
    count = 0
    ids = set()
    duplicate_ids = 0
    with p.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                count += 1
                if 'id' in row:
                    duplicate_ids += row['id'] in ids
                    ids.add(row['id'])
    summary['datasets']['original_musique_' + p.stem] = {'path': str(p.relative_to(DATA)),
        'records': count, 'unique_ids': len(ids), 'duplicate_ids': duplicate_ids}

parquets = sorted((DATA / 'raw/hotpotqa/distractor').glob('train*.parquet'))
assert len(parquets) == 2
for p in parquets:
    with p.open('rb') as f:
        assert f.read(4) == b'PAR1'
        f.seek(-4, 2)
        assert f.read(4) == b'PAR1'
summary['datasets']['original_hotpotqa_train'] = {'paths': [str(p.relative_to(DATA)) for p in parquets],
    'format': 'parquet', 'verified': 'source LFS SHA256, source size, and parquet file markers',
    'row_count': 90447, 'row_count_source': 'official Hugging Face dataset card; not yet parsed locally',
    'source': 'https://huggingface.co/datasets/hotpotqa/hotpot_qa'}

parts = list(DATA.rglob('*.part'))
assert not parts, parts
report = ROOT / 'reports/data_structure_report.json'
report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
