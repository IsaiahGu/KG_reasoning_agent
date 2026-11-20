#!/usr/bin/env python
"""Fit the L2 binding scorer on train-S hops and report held-out selection accuracy. CPU only, no LLM calls.

Fit and validation ranges are positions in the split file, so the two sets of questions never overlap.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind.binding_scorer import FEATURE_SETS, BindingScorer, hop_groups, selection_report  # noqa: E402


def span(text):
    start, end = text.split(':')
    return int(start), int(end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', default='musique')
    parser.add_argument('--split', default='train-S')
    parser.add_argument('--tags', nargs='+', required=True,
                        help='run_chain tags whose records.jsonl hold the labelled candidates; outputs go to the first')
    parser.add_argument('--fit', type=span, default=(0, 1500))
    parser.add_argument('--val', type=span, default=(1500, 2000))
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--out', default='binding_scorer.json')
    parser.add_argument('--records', default='records.jsonl', help='records file name inside each tag folder')
    parser.add_argument('--features', default='v1', choices=sorted(FEATURE_SETS))
    arguments = parser.parse_args()
    FEATURE_NAMES = FEATURE_SETS[arguments.features]
    base = arguments.root / 'runs/chain' / arguments.split / arguments.dataset
    folder = base / arguments.tags[0]
    order = [str(row.get('id', row.get('_id'))) for row in json.loads(
        (arguments.root / 'data/splits/seed42' / arguments.dataset / (arguments.split + '.json')).read_text())]
    records = {}
    for tag in arguments.tags:
        for line in (base / tag / arguments.records).read_text().splitlines():
            record = json.loads(line)
            if 'error' not in record:
                records[record['id']] = record
    parts = {name: [records[key] for key in order[start:end] if key in records]
             for name, (start, end) in (('fit', arguments.fit), ('val', arguments.val))}
    groups = {name: hop_groups(rows) for name, rows in parts.items()}
    train = [(group['rows'], group['labels']) for group in groups['fit'] if len(group['rows']) > 1 and max(group['labels']) == 1]
    scorer = BindingScorer.fit(train, FEATURE_NAMES, l2=arguments.l2)
    report = {'dataset': arguments.dataset, 'split': arguments.split, 'tags': arguments.tags, 'l2': arguments.l2,
              'features': arguments.features, 'records': arguments.records,
              'fit_range': list(arguments.fit), 'val_range': list(arguments.val),
              'questions': {name: len(rows) for name, rows in parts.items()},
              'aligned_questions': {name: sum(bool(row.get('aligned')) for row in rows) for name, rows in parts.items()},
              'training_groups': len(train),
              'selection': {name: selection_report(rows, scorer) for name, rows in groups.items()},
              'selection_by_hop_val': {str(hop): selection_report([group for group in groups['val'] if group['hop'] == hop], scorer)
                                       for hop in sorted({group['hop'] for group in groups['val']})},
              'weights_on_standardised_features': {name: round(float(weight), 3) for name, weight in
                                                   sorted(zip(scorer.names, scorer.weights), key=lambda item: -abs(item[1]))}}
    # Five-fold cross-validation over the fit questions: every hop is scored by a model that never saw its question.
    folds = {key: index % 5 for index, key in enumerate(sorted({group['id'] for group in groups['fit']}))}
    wins = losses = both = neither = 0
    for fold in range(5):
        inside = [(group['rows'], group['labels']) for group in groups['fit']
                  if folds[group['id']] != fold and len(group['rows']) > 1 and max(group['labels']) == 1]
        model = BindingScorer.fit(inside, FEATURE_NAMES, l2=arguments.l2)
        for group in groups['fit']:
            if folds[group['id']] != fold:
                continue
            first = group['labels'][0]
            picked = group['labels'][int(np.argmax(model.logits(group['rows'])))] if len(group['rows']) > 1 else first
            wins += int(picked and not first)
            losses += int(first and not picked)
            both += int(first and picked)
            neither += int(not first and not picked)
    total = wins + losses + both + neither
    report['cross_validation_fit'] = {
        'hops': total, 'first_correct_percent': round(100 * (both + losses) / max(1, total), 2),
        'scorer_correct_percent': round(100 * (both + wins) / max(1, total), 2),
        'scorer_fixes': wins, 'scorer_breaks': losses,
        'any_correct_percent': selection_report(groups['fit']).get('any_correct_percent')}
    held_out = groups['val']
    report['val_changes'] = {
        'scorer_fixes': sum(int(group['labels'][int(np.argmax(scorer.logits(group['rows'])))] and not group['labels'][0])
                            for group in held_out if len(group['rows']) > 1),
        'scorer_breaks': sum(int(group['labels'][0] and not group['labels'][int(np.argmax(scorer.logits(group['rows'])))])
                             for group in held_out if len(group['rows']) > 1)}
    # Ablation: refit without one feature family and score the same validation hops.
    families = {'lookahead': ['has_next', 'next_top1', 'next_top3', 'next_title_match', 'next_answer_in_top3'],
                'retrieval': ['passage_rank', 'passage_score', 'score_gap', 'question_score'],
                'reader_order': ['first', 'position'], 'support': ['explicit', 'quote_verbatim', 'answer_in_quote', 'answer_in_passage'],
                'entity_match': ['title_in_question', 'bound_in_title', 'bound_in_passage', 'answer_in_question', 'support_count']}
    report['ablation_val_scorer_correct_percent'] = {}
    for family, removed in families.items():
        names = [name for name in FEATURE_NAMES if name not in removed]
        partial = BindingScorer.fit(train, names, l2=arguments.l2)
        report['ablation_val_scorer_correct_percent']['without_' + family] = selection_report(groups['val'], partial).get('scorer_correct_percent')
    (folder / arguments.out).write_text(json.dumps(scorer.to_dict(), indent=2) + '\n')
    (folder / (Path(arguments.out).stem + '_report.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
