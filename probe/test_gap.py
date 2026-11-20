#!/usr/bin/env python
"""Why is the gain on the MuSiQue test subset smaller than on dev-A? CPU only, reads finished runs.

(1) per hop count: F1, loop completion, evidence recall and whether the plan has the right number of hops,
    for the fixed-example plan (A1) and the round-2 memory plan, on dev-A and on test;
(2) how close each dev-A / test question is to its nearest train-S fit question (the questions the plan memory
    is built from), by cosine of the question vectors.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).resolve().parent.name == 'claude_probe' else Path('.').resolve()
DATASET = sys.argv[1] if len(sys.argv) > 1 else 'musique'


def load(split, tag):
    folder = ROOT / 'runs/chain' / split / DATASET / tag
    scored = {row['id']: row for row in json.loads((folder / 'scored.json').read_text())}
    records = {}
    for line in (folder / 'records.jsonl').open():
        record = json.loads(line)
        if 'error' not in record:
            records[record['id']] = record
    return scored, records


for split in ('dev-A', 'test'):
    for tag in ('single', 'selfmem2_answer'):
        try:
            scored, records = load(split, tag)
        except FileNotFoundError:
            continue
        print('== %s %s' % (split, tag))
        groups = {}
        for key, record in records.items():
            if key in scored:
                groups.setdefault(str(record.get('stratum')), []).append((record, scored[key]))
        for stratum in sorted(groups):
            rows = groups[stratum]
            hops = int(stratum) if stratum.isdigit() else None
            cells = ['n %d' % len(rows), 'F1 %.1f' % (100 * np.mean([s['f1'] for _, s in rows])),
                     'complete %.0f%%' % (100 * np.mean([bool(r.get('complete')) for r, _ in rows])),
                     'evidence recall %.0f%%' % (100 * np.mean([r.get('evidence_recall', 0) for r, _ in rows])),
                     'all evidence %.0f%%' % (100 * np.mean([bool(r.get('evidence_all_supporting')) for r, _ in rows]))]
            if hops:
                cells.append('plan hops right %.0f%%' % (100 * np.mean([len(r.get('hops') or []) == hops for r, _ in rows])))
                complete = [s['f1'] for r, s in rows if r.get('evidence_all_supporting')]
                cells.append('F1 when all evidence found %.1f (n %d)' % (100 * np.mean(complete) if complete else 0, len(complete)))
            print('  %s-hop: ' % stratum + ', '.join(cells))


def unit(path):
    vectors = np.load(path).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


sources = {'train-S fit': ROOT / 'runs/train_s/embeddings' / DATASET / 'queries.f16.npy',
           'dev-A': ROOT / 'runs/p0_dev/embeddings' / DATASET / 'queries_dev-A.f16.npy',
           'test': ROOT / 'runs/p0/calibration' / DATASET / 'queries.f16.npy'}
if all(path.exists() for path in sources.values()):
    fit = unit(sources['train-S fit'])[:1500]
    print('== nearest train-S fit question (cosine of question vectors)')
    for name in ('dev-A', 'test'):
        nearest = (unit(sources[name]) @ fit.T).max(axis=1)
        print('  %s: n %d, median %.3f, mean %.3f, share above 0.7: %.0f%%, above 0.8: %.0f%%, above 0.9: %.0f%%' % (
            name, len(nearest), np.median(nearest), nearest.mean(), 100 * (nearest > 0.7).mean(), 100 * (nearest > 0.8).mean(),
            100 * (nearest > 0.9).mean()))
    held = unit(sources['train-S fit'])[1500:2000]
    nearest = (held @ fit.T).max(axis=1)
    print('  train-S held-out: n %d, median %.3f, mean %.3f, share above 0.7: %.0f%%, above 0.8: %.0f%%, above 0.9: %.0f%%' % (
        len(nearest), np.median(nearest), nearest.mean(), 100 * (nearest > 0.7).mean(), 100 * (nearest > 0.8).mean(),
        100 * (nearest > 0.9).mean()))
else:
    print('question vectors missing:', [str(path) for path in sources.values() if not path.exists()])
