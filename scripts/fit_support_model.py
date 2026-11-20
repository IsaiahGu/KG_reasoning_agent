#!/usr/bin/env python
"""Fit the derivation reliability model (src/hyperbind/support.py) on train-S runs with known answers. CPU only.

Label: the derivation's answer has F1 >= 0.5 against the training answer. The dense prior is the share of right
answers of the direct dense read on the same questions. Prints 5-fold calibration so the numbers can be checked.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind.support import FEATURES, SupportModel, chain_features  # noqa: E402


def load(root, dataset, tag, split='train-S'):
    folder = root / 'runs/chain' / split / dataset / tag
    scored = {row['id']: row for row in json.loads((folder / 'scored.json').read_text())}
    records = {}
    for line in (folder / 'records.jsonl').open():
        record = json.loads(line)
        if 'error' not in record:
            records[record['id']] = record
    return scored, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', default='musique')
    parser.add_argument('--tags', nargs='+', default=['single_fit', 'cand3_fit_reader'])
    parser.add_argument('--dense-tag', default='a0_fit')
    parser.add_argument('--out', type=Path, default=ROOT / 'configs/support/musique_chain_v1.json')
    arguments = parser.parse_args()
    X, y, groups = [], [], []
    for tag in arguments.tags:
        scored, records = load(arguments.root, arguments.dataset, tag)
        for key in sorted(records):
            if key in scored:
                X.append(chain_features(records[key]))
                y.append(float(scored[key]['f1'] >= 0.5))
                groups.append(key)
    X, y = np.array(X), np.array(y)
    dense, _ = load(arguments.root, arguments.dataset, arguments.dense_tag)
    prior = float(np.mean([row['f1'] >= 0.5 for row in dense.values()]))
    # 5-fold check, folds by question so that the two runs of one question stay together
    names = sorted(set(groups))
    fold_of = {name: index % 5 for index, name in enumerate(names)}
    folds = np.array([fold_of[name] for name in groups])
    out_of_fold = np.zeros(len(y))
    for fold in range(5):
        model = SupportModel.fit(X[folds != fold], y[folds != fold], prior)
        out_of_fold[folds == fold] = model.predict(X[folds == fold])
    print('%d derivations, %.1f%% right; dense prior %.3f' % (len(y), 100 * y.mean(), prior))
    print('out-of-fold calibration (predicted bin: n, share right):')
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        mask = (out_of_fold >= low) & (out_of_fold < low + 0.2 + 1e-9)
        if mask.any():
            print('  %.1f-%.1f: n=%4d  right %.1f%%' % (low, low + 0.2, mask.sum(), 100 * y[mask].mean()))
    order = np.argsort(-out_of_fold)
    auc = float(np.mean([out_of_fold[i] > out_of_fold[j] for i in np.flatnonzero(y == 1)[:400] for j in np.flatnonzero(y == 0)[:400]]))
    print('AUC (sampled) %.3f; top half precision %.1f%%' % (auc, 100 * y[order[:len(y) // 2]].mean()))
    model = SupportModel.fit(X, y, prior)
    model.save(arguments.out)
    print('weights: ' + '  '.join('%s %+.2f' % pair for pair in zip(FEATURES, model.weights)))
    print('saved %s' % arguments.out)


if __name__ == '__main__':
    main()
