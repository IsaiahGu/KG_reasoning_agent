#!/usr/bin/env python
"""Paper numbers from finished runs: merging, merge rules, the support threshold and the reliability model. CPU only.

Reads runs/chain/<split>/<dataset>/<tag>/{records.jsonl,scored.json} written by scripts/run_chain.py (and
scripts/run_llm_selector.py for the reader-picks control). No LLM is called. Prints the tables and writes them to
runs/results_<split>.json.

Executions and the run tags that hold them:
  dense   a0               dense read of the top five passages
  single  single           the chain with the fixed plan examples (the single execution)
  plan1   selfmem_answer   plan examples from memory, round 1
  plan2   selfmem2_answer  plan examples from memory, round 2
  t07_k   sc07_k           the single execution resampled at temperature 0.7 (seed k)
  t03_k   sc03_k           the same at temperature 0.3
The full method merges dense, single, plan1 and plan2 by support (Section 3.4 of the paper).

  python scripts/evaluate.py --split test
  python scripts/evaluate.py --split test --refit     # also the reliability-model analyses (needs the train-S runs)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supportchain.support import FEATURES, SupportModel, answer_groups, chain_features, merge  # noqa: E402
from supportchain.textmatch import same_answer  # noqa: E402

DATASETS = ['musique', '2wikimultihopqa', 'hotpotqa', 'nq_rear', 'popqa']
TRANSFER = {'nq_rear': 'hotpotqa', 'popqa': 'hotpotqa'}  # nothing is fitted on NQ and PopQA
TAGS = {'dense': 'a0', 'single': 'single', 'plan1': 'selfmem_answer', 'plan2': 'selfmem2_answer',
        't07_1': 'sc07_1', 't07_2': 'sc07_2', 't07_3': 'sc07_3', 't03_1': 'sc03_1', 't03_2': 'sc03_2'}
FULL = ['dense', 'single', 'plan1', 'plan2']
SETS = {'full method': FULL, 'single + 2 plans': ['single', 'plan1', 'plan2'],
        'single + 2 samples T=0.7': ['single', 't07_1', 't07_2'], 'single + 3 samples T=0.7': ['single', 't07_1', 't07_2', 't07_3'],
        'single + 2 samples T=0.3': ['single', 't03_1', 't03_2'], 'dense + single + 2 samples T=0.7': ['dense', 'single', 't07_1', 't07_2'],
        'dense + single + 2 samples T=0.3': ['dense', 'single', 't03_1', 't03_2'],
        'dense + single + 3 samples T=0.7': ['dense', 'single', 't07_1', 't07_2', 't07_3']}
VARIANTS = {'reader sees notes only': 'ctx_notes', 'reader sees passages only': 'ctx_passages', '4 passages per hop': 'khop4',
            '3 candidates per hop': 'cand3', '+ scorer, backtracking': 'cand3_s1500_bt', 'gold-decomposition plans': 'knn_single'}
GROUPS = {'completion and agreement': ['complete', 'chain_reader_agree', 'marker'],
          'quote evidence': ['min_explicit', 'mean_verbatim', 'min_answer_in_quote'],
          'retrieval scores': ['max_rank', 'min_passage_score', 'min_question_score', 'min_next_top1'],
          'plan shape': ['hops', 'linked', 'coverage', 'answer_in_question', 'backtracked', 'one_hop']}
FIT_TAGS = {'musique': ['single_fit', 'cand3_fit_reader']}  # other datasets: single_fit only
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01)  # 1.01: every execution runs
RESAMPLES = {}


def interval(values):
    """Mean and 95% paired-bootstrap interval in percent: 10,000 resamples of the questions, seed 0."""
    values = np.asarray(values, float)
    if len(values) not in RESAMPLES:
        RESAMPLES[len(values)] = np.random.default_rng(0).integers(0, len(values), (10000, len(values)))
    means = values[RESAMPLES[len(values)]].mean(axis=1)
    return [round(100 * float(values.mean()), 2), round(100 * float(np.percentile(means, 2.5)), 2),
            round(100 * float(np.percentile(means, 97.5)), 2)]


def load(runs, split, dataset, tag):
    folder = runs / split / dataset / tag
    if not (folder / 'scored.json').exists() or not (folder / 'records.jsonl').exists():
        return None
    scored = {row['id']: row for row in json.loads((folder / 'scored.json').read_text(encoding='utf-8'))}
    records = {}
    for line in (folder / 'records.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        if 'error' not in record:
            records[record['id']] = record
    return scored, records


class Bench:
    """The finished runs of one dataset and split and the rules that merge them."""

    def __init__(self, runs, split, dataset, model):
        self.runs = {name: run for name, tag in TAGS.items() if (run := load(runs, split, dataset, tag)) is not None}
        self.variants = {label: run for label, tag in VARIANTS.items() if (run := load(runs, split, dataset, tag)) is not None}
        self.selector = runs / split / dataset / 'llm_select' / 'records.jsonl'
        self.model, self.cache = model, {}
        base = self.runs.get('single')
        self.ids = sorted(set(base[0]) & set(base[1])) if base else []

    def has(self, use):
        return all(name in self.runs for name in use)

    def reliability(self, name, key, model=None):
        model = model or self.model
        slot = (id(model), name, key)
        if slot not in self.cache:
            self.cache[slot] = (model, model.reliability(self.runs[name][1][key], dense=name == 'dense'))
        return self.cache[slot][1]

    def answer(self, name, key):
        return self.runs[name][1][key]['reader_clean'] or ''

    def keys(self, use):
        return [key for key in self.ids if all(key in part for name in use for part in self.runs[name])]

    def evaluate(self, use, rule='support', on_demand=False, model=None, threshold=None, start=2):
        """rule: support (Section 3.4), vote (the largest group of matching answers wins, ties the earliest group; its
        output is the group's first member, as described in the paper's appendix on merge rules), reliable (the most reliable execution, no
        pooling) or oracle (the best execution by F1, which reads the gold answer). on_demand: the executions after the
        first two run only when those two disagree. threshold: executions are added one at a time from `start` on while
        the highest support is below it."""
        if not self.has(use):
            return None
        out = {'keys': self.keys(use), 'f1': {}, 'em': {}, 'support': {}, 'calls': {}, 'tokens': {}, 'used': {}}
        for key in out['keys']:
            names = list(use)
            if on_demand and same_answer(self.answer(names[0], key), self.answer(names[1], key)):
                names = names[:2]
            if threshold is not None:
                count = start
                while True:
                    names = list(use[:count])
                    _, support, _ = merge([self.answer(name, key) for name in names], [self.reliability(name, key, model) for name in names])
                    if support >= threshold or count == len(use):
                        break
                    count += 1
            index, support = 0, None
            if len(names) > 1:
                answers = [self.answer(name, key) for name in names]
                reliabilities = [self.reliability(name, key, model) for name in names]
                if rule == 'oracle':
                    index = int(np.argmax([self.runs[name][0][key]['f1'] for name in names]))
                elif rule == 'reliable':
                    index = int(np.argmax(reliabilities))
                elif rule == 'vote':
                    groups = answer_groups(answers)
                    index = max(groups, key=len)[0] if groups else int(np.argmax(reliabilities))
                else:
                    index, support, _ = merge(answers, reliabilities)
            row = self.runs[names[index]][0][key]
            usage = [self.runs[name][1][key]['usage'] for name in names]
            out['f1'][key], out['em'][key], out['support'][key], out['used'][key] = row['f1'], row['em'], support, len(names)
            out['calls'][key] = sum(item['calls'] for item in usage)
            out['tokens'][key] = sum(item['prompt_tokens'] + item['completion_tokens'] for item in usage)
        return out

    def reader_picks(self):
        """The reader's own choice among the answers of the full method (scripts/run_llm_selector.py). The output is the
        execution the run recorded as its source, the first member of the chosen group."""
        if not self.selector.exists() or not self.has(FULL):
            return None
        rows = {row['id']: row for row in map(json.loads, self.selector.open(encoding='utf-8')) if 'error' not in row}
        keys = [key for key in self.keys(FULL) if key in rows]
        out = {'keys': keys, 'f1': {}, 'em': {}, 'calls': {}, 'tokens': {}}
        for key in keys:
            row = self.runs[{'fixed': 'single', 'memory1': 'plan1', 'memory2': 'plan2'}.get(rows[key]['source'], rows[key]['source'])][0][key]
            usage = [self.runs[name][1][key]['usage'] for name in FULL] + [rows[key]['usage']]
            out['f1'][key], out['em'][key] = row['f1'], row['em']
            out['calls'][key] = sum(item['calls'] for item in usage)
            out['tokens'][key] = sum(item['prompt_tokens'] + item['completion_tokens'] for item in usage)
        return out


def summary(result, against=None):
    if result is None:
        return None
    row = {'n': len(result['keys']), 'f1': interval(list(result['f1'].values())),
           'em': round(100 * float(np.mean(list(result['em'].values()))), 2),
           'calls': round(float(np.mean(list(result['calls'].values()))), 2),
           'tokens': round(float(np.mean(list(result['tokens'].values()))))}
    if against is not None and against is not result:
        shared = [key for key in result['keys'] if key in against['f1']]
        row['minus'] = interval([result['f1'][key] - against['f1'][key] for key in shared])
    return row


def ece(p, y, count=10):
    """Expected calibration error in percent over ten equal-width bins."""
    p, y, error = np.asarray(p, float), np.asarray(y, float), 0.0
    for index in range(count):
        low, high = index / count, (index + 1) / count
        mask = (p >= low) & ((p < high) if index < count - 1 else (p <= 1.0))
        if mask.any():
            error += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return round(100 * float(error), 2)


class Refit:
    """Reliability models refitted like the saved ones, on the train-S runs (scripts/fit_support_model.py)."""

    def __init__(self, runs):
        self.runs, self.rows, self.models = runs, {}, {}

    def training(self, dataset):
        if dataset not in self.rows:
            keys, X, y = [], [], []
            for tag in FIT_TAGS.get(dataset, ['single_fit']):
                scored, records = load(self.runs, 'train-S', dataset, tag)
                for key in sorted(records):
                    if key in scored:
                        keys.append(key)
                        X.append(chain_features(records[key]))
                        y.append(float(scored[key]['f1'] >= 0.5))
            dense = {key: float(row['f1'] >= 0.5) for key, row in load(self.runs, 'train-S', dataset, 'a0_fit')[0].items()}
            self.rows[dataset] = (keys, np.array(X), np.array(y), dense)
        return self.rows[dataset]

    def model(self, datasets, drop=(), limit=None):
        datasets = (datasets,) if isinstance(datasets, str) else tuple(datasets)
        slot = (datasets, tuple(drop), limit)
        if slot not in self.models:
            parts, labels, priors = [], [], []
            for dataset in datasets:
                keys, X, y, dense = self.training(dataset)
                questions = sorted(set(keys))
                if limit is not None:  # a fixed random subset of the training questions
                    questions = [questions[index] for index in np.random.default_rng(0).permutation(len(questions))[:limit]]
                chosen = set(questions)
                mask = np.array([key in chosen for key in keys])
                parts.append(X[mask])
                labels.append(y[mask])
                priors += list(dense.values()) if limit is None else [dense[key] for key in questions if key in dense]
            X = np.concatenate(parts)
            X[:, [FEATURES.index(name) for name in drop]] = 0.0
            self.models[slot] = SupportModel.fit(X, np.concatenate(labels), float(np.mean(priors)))
        return self.models[slot]


def analyse(bench, dataset, refit):
    out = {}
    full, single, dense = bench.evaluate(FULL), bench.evaluate(['single']), bench.evaluate(['dense'])
    samples = ['single', 't07_1', 't07_2']
    out['main'] = {'dense read': summary(dense, full), 'single execution': summary(single, full), 'full method': summary(full),
                   'on demand': summary(bench.evaluate(FULL, on_demand=True), full),
                   'single + 2 samples T=0.7': summary(bench.evaluate(samples), full)}
    out['merge_rules'] = {}
    for label, use in (('full method', FULL), ('single + 2 samples T=0.7', samples)):
        merged = bench.evaluate(use)
        if merged is None:
            continue
        rules = {'majority vote': bench.evaluate(use, 'vote'), 'most reliable execution': bench.evaluate(use, 'reliable'),
                 'oracle': bench.evaluate(use, 'oracle')}
        if use == FULL:
            rules['reader picks'] = bench.reader_picks()
        out['merge_rules'][label] = {'support': summary(merged)}
        for rule, result in rules.items():
            value = summary(result)
            if value is not None and rule != 'oracle':  # support minus the rule, paired
                value['support_minus'] = summary(merged, result)['minus']
            out['merge_rules'][label][rule] = value
    out['sources'] = {label: dict(summary(bench.evaluate(use), full), minus_single=summary(bench.evaluate(use), single)['minus'])
                      for label, use in SETS.items() if bench.has(use)}
    out['all_merge_comparisons'] = {
        label: {'support minus vote': summary(bench.evaluate(use), bench.evaluate(use, 'vote'))['minus'],
                'support minus most reliable': summary(bench.evaluate(use), bench.evaluate(use, 'reliable'))['minus']}
        for label, use in SETS.items() if not label.startswith('dense + ') and bench.has(use)}
    out['threshold_curve'] = {}
    for label, use, start in (('full method', FULL, 2), ('single + 3 samples T=0.7', ['single', 't07_1', 't07_2', 't07_3'], 1)):
        if bench.has(use):
            out['threshold_curve'][label] = {str(tau): summary(bench.evaluate(use, threshold=tau, start=start))
                                             for tau in THRESHOLDS}
    out['chain_variants'] = {}
    for label, run in bench.variants.items():
        keys = [key for key in single['keys'] if key in run[0]]
        out['chain_variants'][label] = interval([run[0][key]['f1'] - single['f1'][key] for key in keys])
    chains = ['single', 'plan1', 'plan2']
    p = [bench.reliability(name, key) for name in chains for key in full['keys']]
    y = [bench.runs[name][0][key]['f1'] >= 0.5 for name in chains for key in full['keys']]
    support = [full['support'][key] or 0.0 for key in full['keys']]
    order = sorted(full['keys'], key=lambda key: -(full['support'][key] or 0.0))
    out['reliability'] = {'ece_single_executions': ece(p, y), 'executions': len(p),
                          'ece_merged_support': ece(support, [full['f1'][key] >= 0.5 for key in full['keys']]),
                          'f1_top_half_by_support': round(100 * float(np.mean([full['f1'][key] for key in order[:len(order) // 2]])), 2)}
    if refit is not None:
        source = TRANSFER.get(dataset, dataset)
        rows = {}
        for name, columns in GROUPS.items():
            rows['without ' + name] = refit.model(source, columns)
        rows['no features (constant)'] = refit.model(source, FEATURES[1:])
        for limit in (100, 250, 500, 1000):
            rows['%d training questions' % limit] = refit.model(source, limit=limit)
        rows['3 datasets pooled'] = refit.model(('musique', '2wikimultihopqa', 'hotpotqa'))
        for other in ('musique', '2wikimultihopqa', 'hotpotqa'):
            if other != source:
                rows['fitted on ' + other] = refit.model(other)
        gap = float(np.max(np.abs(refit.model(source).weights - bench.model.weights)))
        out['reliability_refit'] = {'largest_weight_difference_to_saved_model': round(gap, 6)}
        out['reliability_refit'].update({label: summary(bench.evaluate(FULL, model=model), full)['minus'] for label, model in rows.items()})
    return out


def cell(value):
    return '-' if value is None else '%.2f' % value['f1'][0]


def show(results):
    names = {'musique': 'MuSiQue', '2wikimultihopqa': '2Wiki', 'hotpotqa': 'HotpotQA', 'nq_rear': 'NQ', 'popqa': 'PopQA'}
    datasets = list(results)
    print('%-34s' % 'test F1' + ''.join('%10s' % names[d] for d in datasets))
    for system in ('dense read', 'single execution', 'full method', 'on demand', 'single + 2 samples T=0.7'):
        print('%-34s' % system + ''.join('%10s' % cell(results[d]['main'].get(system)) for d in datasets))
    for label in ('full method', 'single + 2 samples T=0.7'):
        print('\nmerge rules, ' + label)
        for rule in ('majority vote', 'most reliable execution', 'reader picks', 'support', 'oracle'):
            row = [results[d]['merge_rules'].get(label, {}).get(rule) for d in datasets]
            if any(row):
                print('%-34s' % rule + ''.join('%10s' % cell(value) for value in row))
    print('\nsupport threshold (full method): F1 / calls per question')
    for tau in THRESHOLDS:
        row = [results[d]['threshold_curve'].get('full method', {}).get(str(tau)) for d in datasets]
        print('%-34s' % ('tau %.2f' % tau) + ''.join('%10s' % ('-' if v is None else '%.1f/%.1f' % (v['f1'][0], v['calls'])) for v in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/chain')
    parser.add_argument('--split', default='test', choices=['test', 'dev-A'])
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--refit', action='store_true', help='refit the reliability model on train-S runs (Section 6.2)')
    parser.add_argument('--out', type=Path)
    arguments = parser.parse_args()
    refit = Refit(arguments.runs) if arguments.refit else None
    results = {}
    for dataset in arguments.datasets:
        source = TRANSFER.get(dataset, dataset)
        model = SupportModel.load(ROOT / 'configs/support' / (source + '_chain_v1.json'))
        bench = Bench(arguments.runs, arguments.split, dataset, model)
        if bench.has(FULL):
            results[dataset] = analyse(bench, dataset, refit)
    show(results)
    out = arguments.out or ROOT / 'runs' / ('results_%s.json' % arguments.split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1) + '\n')
    print('\nwritten to %s' % out)


if __name__ == '__main__':
    main()
