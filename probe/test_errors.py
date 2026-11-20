#!/usr/bin/env python
"""Look at wrong answers of one run where every supporting passage was retrieved. CPU only.

Usage: test_errors.py <dataset> <split> <tag> <stratum> [count]
Purpose: tell a scoring or data problem from a reasoning problem. Nothing is fitted or tuned here.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path('.').resolve()
dataset, split, tag, stratum = sys.argv[1:5]
count = int(sys.argv[5]) if len(sys.argv) > 5 else 20
folder = ROOT / 'runs/chain' / split / dataset / tag
scored = {row['id']: row for row in json.loads((folder / 'scored.json').read_text())}
records = [json.loads(line) for line in (folder / 'records.jsonl').open()]
if split == 'test':
    rows = json.loads((ROOT / 'data/hipporag2' / (dataset + '.json')).read_text())
else:
    rows = json.loads((ROOT / 'data/splits/seed42' / dataset / (split + '.json')).read_text())
gold = {}
for position, row in enumerate(rows):
    key = str(row.get('id', row.get('_id', '%s_%04d' % (dataset, position))))
    gold[key] = (row.get('answer'), row.get('answer_aliases'), row.get('evidences'))
pool = [record for record in records if 'error' not in record and str(record.get('stratum')) == stratum and record['id'] in scored]
wrong = [record for record in pool if scored[record['id']]['f1'] < 0.5]
found = [record for record in wrong if record.get('evidence_all_supporting')]
print('%s %s %s, stratum %s: %d questions, %d with F1 below 0.5, of those %d with every supporting passage retrieved' % (
    dataset, split, tag, stratum, len(pool), len(wrong), len(found)))
kinds = {'empty or no answer': 0, 'gold inside the answer': 0, 'answer inside the gold': 0, 'other': 0}
for record in wrong:
    answer, reference = (record.get('reader_clean') or '').strip().lower(), str(gold[record['id']][0]).strip().lower()
    if not answer or 'unknown' in answer or 'not ' in answer[:40]:
        kinds['empty or no answer'] += 1
    elif reference in answer:
        kinds['gold inside the answer'] += 1
    elif answer in reference:
        kinds['answer inside the gold'] += 1
    else:
        kinds['other'] += 1
print('kinds of wrong answers:', kinds)
random.Random(0).shuffle(found)
for record in found[:count]:
    answer, aliases, evidences = gold[record['id']]
    print('---')
    print('Q:', record['question'])
    print('gold:', answer, '| aliases:', aliases, '| evidences:', evidences)
    print('plan:', record.get('hops'))
    print('hop answers:', [step['candidates'][step['chosen']]['answer'] if step.get('candidates') else None for step in record.get('steps', [])])
    print('chain answer:', record.get('chain_answer'), '| reader:', (record.get('reader_clean') or '')[:160], '| F1 %.2f' % scored[record['id']]['f1'])
