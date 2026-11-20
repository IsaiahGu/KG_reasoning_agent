#!/usr/bin/env python
"""Build a plan memory from the agent's own plans on train-S questions (no decomposition annotation). CPU only.

A plan is admitted when its derivation looks sound:
  --admit answer   the derivation's answer matches the training answer (F1 >= threshold, default 0.8)
  --admit support  the reliability model, fitted on the other folds, gives the derivation >= threshold (default 0.7);
                   answers are used only to fit that model, so the same rule can admit plans for unlabeled questions
Output: JSON list of {"id", "hops", "score"} for scripts/run_chain.py --plan knn --plan-cases <file>.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from fit_support_model import load  # noqa: E402
from hyperbind.support import SupportModel, chain_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', default='musique')
    parser.add_argument('--tags', nargs='+', default=['single_fit'], help='train-S runs; the first run that qualifies wins')
    parser.add_argument('--admit', choices=['answer', 'support'], required=True)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--out', type=Path, required=True)
    arguments = parser.parse_args()
    threshold = arguments.threshold if arguments.threshold is not None else (0.8 if arguments.admit == 'answer' else 0.7)
    cases, seen, considered = [], set(), 0
    for tag in arguments.tags:
        scored, records = load(arguments.root, arguments.dataset, tag)
        keys = [key for key in sorted(records) if key in scored and records[key]['plan_ok'] and records[key]['complete']]
        considered += len(keys)
        if arguments.admit == 'answer':
            scores = {key: scored[key]['f1'] for key in keys}
        else:
            X = np.array([chain_features(records[key]) for key in keys])
            y = np.array([float(scored[key]['f1'] >= 0.5) for key in keys])
            folds, scores = np.arange(len(keys)) % 5, {}
            for fold in range(5):
                model = SupportModel.fit(X[folds != fold], y[folds != fold])
                for key, value in zip(np.array(keys)[folds == fold], model.predict(X[folds == fold])):
                    scores[str(key)] = float(value)
        for key in keys:
            if scores[key] >= threshold and key not in seen:
                seen.add(key)
                cases.append({'id': key, 'hops': records[key]['hops'], 'score': round(scores[key], 4),
                              'right': bool(scored[key]['f1'] >= 0.5), 'stratum': str(records[key]['stratum'])})
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(cases, ensure_ascii=False) + '\n')
    lengths = collections.Counter(len(case['hops']) for case in cases)
    print('%d plans admitted out of %d complete derivations (rule %s >= %.2f); %.1f%% of them have a right answer' % (
        len(cases), considered, arguments.admit, threshold, 100 * np.mean([case['right'] for case in cases])))
    print('by plan length: ' + ', '.join('%d hops: %d' % item for item in sorted(lengths.items())))
    print('by gold hop count: ' + ', '.join('%s: %d' % item for item in sorted(collections.Counter(case['stratum'] for case in cases).items())))
    print('saved %s' % arguments.out)


if __name__ == '__main__':
    main()
