"""Structural validation of downloaded data; no model, index or evaluation calls."""
from collections import Counter
import importlib.util
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

for name, expected in [('musique', 11656), ('2wikimultihopqa', 6119), ('hotpotqa', 9811)]:
    add_json('hipporag2_' + name + '_test', DATA / 'hipporag2' / (name + '.json'), 1000)
    add_json('hipporag2_' + name + '_corpus', DATA / 'hipporag2' / (name + '_corpus.json'), expected)

for name, expected in [('longbook_choice_eng', 229), ('longbook_qa_eng', 351)]:
    p = DATA / 'infinitebench' / (name + '.jsonl')
    count = 0
    with p.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                assert 'context' in row and 'input' in row and 'answer' in row
                count += 1
    assert count == expected
    summary['datasets']['infinitebench_' + name] = {'path': str(p.relative_to(DATA)), 'records': count}

for name, corpus_count, question_count in [('novel', 20, 2010), ('medical', 1, 2062)]:
    add_json('graphrag_bench_' + name + '_corpus', DATA / 'graphrag_bench/Datasets/Corpus' / (name + '.json'), corpus_count)
    rows = add_json('graphrag_bench_' + name + '_questions', DATA / 'graphrag_bench/Datasets/Questions' / (name + '_questions.json'), question_count)
    summary['datasets']['graphrag_bench_' + name + '_questions']['question_types'] = dict(Counter(x['question_type'] for x in rows))
    assert len(summary['datasets']['graphrag_bench_' + name + '_questions']['question_types']) == 4

add_json('hipporag2_narrativeqa_10_doc_questions', DATA / 'hipporag2/narrativeqa_dev_10_doc.json', 293)
add_json('hipporag2_narrativeqa_10_doc_corpus', DATA / 'hipporag2/narrativeqa_dev_10_doc_corpus.json', 4111)

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

for alias, source in [('hgmem', ROOT / 'external/HGMem/data'),
                      ('wikitopics/sample_1pct', ROOT / 'external/HyperRAG/dataset/wikitopics_test_sampled')]:
    p = DATA / alias
    assert source.is_dir(), source
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(source, target_is_directory=True)

hgmem_qas = sorted((DATA / 'hgmem/created_data').rglob('*.jsonl'))
hgmem_questions = sum(sum(1 for line in p.open() if line.strip()) for p in hgmem_qas)
assert hgmem_questions == 1
summary['datasets']['hgmem_public_narrativeqa_example'] = {
    'path': 'hgmem', 'provided_questions': hgmem_questions, 'provided_documents': 1,
    'source': 'https://github.com/Encyclomen/HGMem/tree/main/data'}

sample = DATA / 'wikitopics/sample_1pct'
domain_counts = {}
for p in sorted(sample.glob('*/test_queries.json')):
    d = j(p)
    domain_counts[p.parent.name] = sum(len(v) for v in d.values()) if isinstance(d, dict) else len(d)
summary['datasets']['wikitopics_official_1pct_sample'] = {
    'path': 'wikitopics/sample_1pct', 'domains': domain_counts, 'questions': sum(domain_counts.values()),
    'baseline_reproduction_gate': 'not run, as instructed'}

for label, path in [('wikitopics_kg', DATA / 'wikitopics/kg'), ('wikitopics_nlg', DATA / 'wikitopics/nlg')]:
    files = [p for p in path.rglob('*') if p.is_file()]
    assert files
    summary['datasets'][label] = {'path': str(path.relative_to(DATA)), 'file_count': len(files),
                                 'uncompressed_bytes': sum(p.stat().st_size for p in files)}

summary['limitations'] = [
    {'dataset': 'HyperGraphRAG additional five-domain/medical dataset',
     'status': 'not downloaded: official TeraBox public API requires a verification challenge',
     'source': 'https://1024terabox.com/s/13cV8oy4kXurqTzYl9j94-g',
     'available_filename': 'HyperGraphRAG-contexts&datasets.zip', 'bytes': 3451939,
     'response': {'errno': 400310, 'errmsg': 'need verify_v2'},
     'plan_scope': 'conditional: PLAN_V1 says include medical dataset if provided in repository'},
    {'dataset': 'HGMem NarrativeQA', 'status': 'public repository contains one example question and one document, not the paper evaluation subset; fully downloaded as publicly released'},
    {'dataset': 'WikiTopics', 'status': 'raw datasets and official sample prepared; 1% baseline reproduction gate deliberately pending until experiments are authorized'},
    {'dataset': 'training splits', 'status': 'original training data downloaded; dev-A/dev-B/train-S sampling and deduplication are pending, no experimental split was created'},
    {'dataset': 'LoCoMo', 'status': 'excluded by PLAN_V1 default'},
]
parts = list(DATA.rglob('*.part'))
assert not parts, parts
report = ROOT / 'reports/data_structure_report.json'
report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
