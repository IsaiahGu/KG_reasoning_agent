#!/usr/bin/env python
"""Does the hypergraph-retrieval chain find or answer anything the dense-retrieval chain does not? CPU only.

Usage: hyper_complement.py <dataset> [split ...]
Reads finished runs (tags single = dense retrieval in the loop, hyper_single = hypergraph retrieval in the same loop).
Nothing is fitted or tuned here; the numbers only say whether the two retrieval channels are complementary.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('.').resolve()
DATASET = sys.argv[1]
SPLITS = sys.argv[2:] or ['dev-A', 'test']
TAGS = ['single', 'selfmem_answer', 'selfmem2_answer', 'hyper_single']


def load(split, tag):
    folder = ROOT / 'runs/chain' / split / DATASET / tag
    scored = {row['id']: row for row in json.loads((folder / 'scored.json').read_text())}
    records = {}
    for line in (folder / 'records.jsonl').open():
        record = json.loads(line)
        if 'error' not in record and record['id'] in scored:
            records[record['id']] = record
    return scored, records


def share(values):
    return 100 * float(np.mean(values)) if len(values) else float('nan')


for split in SPLITS:
    runs = {}
    for tag in TAGS:
        try:
            runs[tag] = load(split, tag)
        except FileNotFoundError:
            pass
    if 'single' not in runs or 'hyper_single' not in runs:
        print('== %s %s: runs missing (%s)' % (DATASET, split, sorted(runs)))
        continue
    ids = sorted(set.intersection(*[set(records) for _, records in runs.values()]))
    dense_s, dense_r = runs['single']
    hyper_s, hyper_r = runs['hyper_single']
    print('== %s %s, %d questions in all runs' % (DATASET, split, len(ids)))
    all_d = np.array([bool(dense_r[i].get('evidence_all_supporting')) for i in ids])
    all_h = np.array([bool(hyper_r[i].get('evidence_all_supporting')) for i in ids])
    rec_d = np.array([dense_r[i].get('evidence_recall', 0) for i in ids])
    rec_h = np.array([hyper_r[i].get('evidence_recall', 0) for i in ids])
    f_d = np.array([dense_s[i]['f1'] for i in ids])
    f_h = np.array([hyper_s[i]['f1'] for i in ids])
    print('  evidence recall: dense %.1f, hyper %.1f; all evidence found: dense %.1f%%, hyper %.1f%%' % (
        share(rec_d), share(rec_h), share(all_d), share(all_h)))
    print('  all evidence: both %.1f%%, dense only %.1f%%, hyper only %.1f%%, neither %.1f%%' % (
        share(all_d & all_h), share(all_d & ~all_h), share(~all_d & all_h), share(~all_d & ~all_h)))
    print('  recall higher with hyper %.1f%%, higher with dense %.1f%%, equal %.1f%%' % (
        share(rec_h > rec_d), share(rec_d > rec_h), share(rec_d == rec_h)))
    right_d, right_h = f_d >= 0.5, f_h >= 0.5
    print('  F1: dense %.2f, hyper %.2f; right (F1 at least 0.5): both %.1f%%, dense only %.1f%%, hyper only %.1f%%, neither %.1f%%' % (
        share(f_d), share(f_h), share(right_d & right_h), share(right_d & ~right_h), share(~right_d & right_h),
        share(~right_d & ~right_h)))
    print('  best of dense and hyper %.2f (dense %.2f)' % (share(np.maximum(f_d, f_h)), share(f_d)))
    others = [tag for tag in ('selfmem_answer', 'selfmem2_answer') if tag in runs]
    if others:
        base = np.max([f_d] + [np.array([runs[tag][0][i]['f1'] for i in ids]) for tag in others], axis=0)
        print('  best of dense-retrieval chains (%s) %.2f, with hyper added %.2f' % (
            ', '.join(['single'] + others), share(base), share(np.maximum(base, f_h))))
        union = np.any([all_d] + [np.array([bool(runs[tag][1][i].get('evidence_all_supporting')) for i in ids]) for tag in others],
                       axis=0)
        print('  all evidence in some dense-retrieval chain %.1f%%, with hyper added %.1f%%' % (share(union), share(union | all_h)))
    same = np.array([(dense_r[i].get('reader_clean') or '').strip().lower() == (hyper_r[i].get('reader_clean') or '').strip().lower()
                     for i in ids])
    print('  same final answer text in both chains: %.1f%%' % share(same))
    strata = sorted({str(dense_r[i].get('stratum')) for i in ids})
    for stratum in strata:
        mask = np.array([str(dense_r[i].get('stratum')) == stratum for i in ids])
        print('    %s: n %d, F1 dense %.1f hyper %.1f, all evidence dense %.0f%% hyper %.0f%%, hyper only %.1f%%, dense only %.1f%%' % (
            stratum, mask.sum(), share(f_d[mask]), share(f_h[mask]), share(all_d[mask]), share(all_h[mask]),
            share((~all_d & all_h)[mask]), share((all_d & ~all_h)[mask])))
