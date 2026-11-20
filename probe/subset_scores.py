#!/usr/bin/env python
"""Scores of the final method on a slice of the HippoRAG 2 test files, by position in the file. CPU only.

Usage: subset_scores.py <start> <end> [dataset ...]   (run from the project root, with PYTHONPATH=src)
Purpose: match papers that evaluate on part of the 1,000 questions, e.g. HyperAddress (arXiv 2610.35412), which keeps the
first 200 questions of each file for development and reports the remaining 800. Reads paper/raw/<dataset>/test_predictions.jsonl.
"""
import json
import sys
from pathlib import Path

import numpy as np

from hyperbind.data import load_test_samples

ROOT = Path('.').resolve()
start, end = int(sys.argv[1]), int(sys.argv[2])
datasets = sys.argv[3:] or ['musique', '2wikimultihopqa', 'hotpotqa']
for dataset in datasets:
    order = [example.id for example in load_test_samples(ROOT, dataset)][start:end]
    rows = {}
    for line in (ROOT / 'paper/raw' / dataset / 'test_predictions.jsonl').open():
        row = json.loads(line)
        rows[row['id']] = row
    keys = [key for key in order if key in rows]
    cells = []
    for name in ('final', 'final_on_demand'):
        values = [rows[key][name] for key in keys if name in rows[key]]
        cells.append('%s F1 %.2f EM %.2f' % (name, 100 * np.mean([value['f1'] for value in values]),
                                            100 * np.mean([value['em'] for value in values])))
    fixed = [rows[key]['derivations']['fixed'] for key in keys]
    cells.append('A1 F1 %.2f EM %.2f' % (100 * np.mean([value['f1'] for value in fixed]), 100 * np.mean([value['em'] for value in fixed])))
    print('%s positions %d-%d: %d questions | %s' % (dataset, start, end - 1, len(keys), ' | '.join(cells)))
