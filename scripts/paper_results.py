#!/usr/bin/env python
"""Paper tables and per-question raw data from finished runs. CPU only: no GPU job is started and no LLM is called.

The final method (configs/final.json) merges four derivations by calibrated support: dense read, chain A1, memory plans
1 and 2. This pre-specified fallback was selected for reporting after testing.

Writes
  paper/results.json                              every number below, machine readable
  paper/tables/*.md                               the same numbers as tables
  paper/raw/<dataset>/<split>_predictions.jsonl   one line per question: gold answers; for every derivation its answer,
                                                  score, reliability, plan and cost, and both merged answers
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind.baseline_dense import score_answers  # noqa: E402
from hyperbind.data import load_samples, load_test_samples  # noqa: E402
from hyperbind.support import FEATURES, SupportModel, answer_groups, chain_features, merge  # noqa: E402
from hyperbind.textmatch import same_answer  # noqa: E402

NAMES = {'musique': 'MuSiQue', '2wikimultihopqa': '2Wiki', 'hotpotqa': 'HotpotQA', 'nq_rear': 'NQ', 'popqa': 'PopQA'}
LABELS = {'dense': 'dense read', 'fixed': 'chain A1', 'memory1': 'memory plan 1', 'memory2': 'memory plan 2'}
GROUPS = {'completion and agreement': ['complete', 'chain_reader_agree', 'marker'],
          'quote evidence': ['min_explicit', 'mean_verbatim', 'min_answer_in_quote'],
          'retrieval scores': ['max_rank', 'min_passage_score', 'min_question_score', 'min_next_top1'],
          'plan shape': ['hops', 'linked', 'coverage', 'answer_in_question', 'backtracked', 'one_hop']}
FIT_TAGS = {'musique': ['single_fit', 'cand3_fit_reader']}  # as in the queue: the other datasets fit on single_fit only
# Runs that exist on development splits only: baselines with the same reader and embedder, and variants that were dropped.
BASELINES = (('a0', 'dense read'), ('b1_dense', 'B1: dense retrieval + read'), ('b2_hipporag2', 'B2: HippoRAG 2, same reader and embedder'),
             ('b3_ircot', 'B3: IRCoT, same reader and embedder'), ('single', 'chain A1'),
             ('cand3', 'variant: three candidates per hop, first taken'),
             ('cand3_s1500_bt', 'variant: three candidates + learned hop scorer + backtracking'),
             ('knn_single', 'variant: plan memory from gold decompositions (MuSiQue only)'))
# Supplementary runs (2026-11-18): repeated samples of chain A1 for the self-consistency baseline, and design variants of
# the chain. They are loaded next to the derivations when they exist; none of them is part of the final method.
SAMPLES = {0.7: ['sc07_1', 'sc07_2', 'sc07_3'], 0.3: ['sc03_1', 'sc03_2']}
VARIANTS = (('ctx_notes', 'final reader sees the hop notes only'), ('ctx_passages', 'final reader sees the passages only'),
            ('khop4', 'four passages per hop instead of eight'))
RESAMPLES = {}


def interval(values):
    """Mean and 95% bootstrap interval in percent (10,000 resamples, seed 0)."""
    values = np.asarray(values, float)
    if len(values) not in RESAMPLES:
        RESAMPLES[len(values)] = np.random.default_rng(0).integers(0, len(values), (10000, len(values)))
    means = values[RESAMPLES[len(values)]].mean(axis=1)
    return [round(100 * float(values.mean()), 2), round(100 * float(np.percentile(means, 2.5)), 2),
            round(100 * float(np.percentile(means, 97.5)), 2)]


def load(split, dataset, tag):
    folder = ROOT / 'runs/chain' / split / dataset / tag
    if not (folder / 'scored.json').exists() or not (folder / 'records.jsonl').exists():
        return None
    scored = {row['id']: row for row in json.loads((folder / 'scored.json').read_text())}
    records = {}
    for line in (folder / 'records.jsonl').open():
        record = json.loads(line)
        if 'error' not in record:
            records[record['id']] = record
    return scored, records


def training_rows(dataset):
    """The train-S runs a dataset's support model was fitted on: question id, features and label of every chain, and
    whether the dense read of each training question was right."""
    if dataset not in training_rows.cache:
        keys, X, y = [], [], []
        for tag in FIT_TAGS.get(dataset, ['single_fit']):
            scored, records = load('train-S', dataset, tag)
            for key in sorted(records):
                if key in scored:
                    keys.append(key)
                    X.append(chain_features(records[key]))
                    y.append(float(scored[key]['f1'] >= 0.5))
        dense = {key: float(row['f1'] >= 0.5) for key, row in load('train-S', dataset, 'a0_fit')[0].items()}
        training_rows.cache[dataset] = (keys, np.array(X), np.array(y), dense)
    return training_rows.cache[dataset]


training_rows.cache = {}


def fitted(datasets, drop=(), limit=None):
    """A support model refitted like the saved one: on the train-S runs of `datasets` (one name, or several pooled), with
    the features in `drop` set to zero, and on `limit` training questions per dataset (a fixed random subset) if given."""
    datasets = (datasets,) if isinstance(datasets, str) else tuple(datasets)
    slot = (datasets, tuple(drop), limit)
    if slot not in fitted.cache:
        parts, labels, priors = [], [], []
        for dataset in datasets:
            keys, X, y, dense = training_rows(dataset)
            questions = sorted(set(keys))
            if limit is not None:
                questions = [questions[index] for index in np.random.default_rng(0).permutation(len(questions))[:limit]]
            chosen = set(questions)
            mask = np.array([key in chosen for key in keys])
            parts.append(X[mask])
            labels.append(y[mask])
            priors += list(dense.values()) if limit is None else [dense[key] for key in questions if key in dense]
        X = np.concatenate(parts)
        X[:, [FEATURES.index(name) for name in drop]] = 0.0
        fitted.cache[slot] = SupportModel.fit(X, np.concatenate(labels), float(np.mean(priors)))
    return fitted.cache[slot]


fitted.cache = {}


class Bench:
    """The finished derivation runs of one dataset and split, and the rules that combine them."""

    def __init__(self, config, transfer, dataset, split):
        self.dataset, self.split = dataset, split
        self.runs = {}
        for name, tag in config['derivations'].items():
            run = load(split, dataset, tag)
            if run is not None:
                self.runs[name] = run
        for tag in [tag for tags in SAMPLES.values() for tag in tags] + [tag for tag, _ in VARIANTS]:
            run = load(split, dataset, tag)
            if run is not None and run[0]:
                self.runs[tag] = run
        self.source = transfer.get(dataset, dataset)
        path = ROOT / config['support_model'].replace('<dataset>', self.source)
        self.model = SupportModel.load(path) if path.exists() else None
        base = self.runs.get('fixed')
        self.ids = sorted(set(base[0]) & set(base[1])) if base else []
        self.cache = {}
        self.models = {}

    def has(self, use):
        return all(name in self.runs for name in use)

    def reliability(self, name, key, model=None):
        model = model or self.model
        # Keep every model alive: the id of a freed model can be handed to the next one loaded, which would then read
        # the freed model's cached values.
        self.models.setdefault(id(model), model)
        slot = (id(model), name, key)
        if slot not in self.cache:
            self.cache[slot] = model.reliability(self.runs[name][1][key], dense=name == 'dense')
        return self.cache[slot]

    def evaluate(self, use, rule='support', on_demand=False, model=None):
        """rule: support (noisy-OR of reliabilities), vote (equal reliabilities), reliable (most reliable derivation,
        no pooling) or oracle (best F1, an upper bound that reads the gold answer)."""
        if not self.ids or not self.has(use) or (len(use) > 1 and self.model is None and rule != 'oracle'):
            return None
        keys = [key for key in self.ids if all(key in part for name in use for part in self.runs[name])]
        out = {'use': list(use), 'keys': keys, 'f1': {}, 'em': {}, 'choice': {}, 'support': {}, 'calls': {}, 'tokens': {}}
        for key in keys:
            names = list(use)
            if on_demand:  # the first two derivations; the others only when their answers differ
                pair = [self.runs[name][1][key]['reader_clean'] or '' for name in names[:2]]
                if same_answer(pair[0], pair[1]):
                    names = names[:2]
            index, support = 0, None
            if len(names) > 1 and rule == 'oracle':
                index = int(np.argmax([self.runs[name][0][key]['f1'] for name in names]))
            elif len(names) > 1:
                answers = [self.runs[name][1][key]['reader_clean'] or '' for name in names]
                reliabilities = [0.5] * len(names) if rule == 'vote' else [self.reliability(name, key, model) for name in names]
                if rule == 'reliable':
                    index = int(np.argmax(reliabilities))
                else:
                    index, support, _ = merge(answers, reliabilities)
            row = self.runs[names[index]][0][key]
            usage = [self.runs[name][1][key]['usage'] for name in names]
            out['f1'][key], out['em'][key], out['choice'][key], out['support'][key] = row['f1'], row['em'], names[index], support
            out['calls'][key] = sum(item['calls'] for item in usage)
            out['tokens'][key] = sum(item['prompt_tokens'] + item['completion_tokens'] for item in usage)
        return out

    def selected(self, use, tag='llm_select'):
        """Control for the merge rule (scripts/run_llm_selector.py): the reader picks among the distinct answers of `use`.
        Its cost is the derivations plus the selector call. None until that run covers every question."""
        path = ROOT / 'runs/chain' / self.split / self.dataset / tag / 'records.jsonl'
        if not self.ids or not self.has(use) or not path.exists():
            return None
        rows = {}
        for line in path.open(encoding='utf-8'):
            row = json.loads(line)
            if 'error' not in row:
                rows[row['id']] = row
        keys = [key for key in self.ids if all(key in part for name in use for part in self.runs[name])]
        if any(key not in rows or rows[key]['source'] not in use for key in keys):
            return None
        out = {'use': list(use) + [tag], 'keys': keys, 'f1': {}, 'em': {}, 'choice': {}, 'support': {}, 'calls': {}, 'tokens': {}}
        for key in keys:
            source = rows[key]['source']
            row = self.runs[source][0][key]
            usage = [self.runs[name][1][key]['usage'] for name in use] + [rows[key]['usage']]
            out['f1'][key], out['em'][key], out['choice'][key], out['support'][key] = row['f1'], row['em'], source, None
            out['calls'][key] = sum(item['calls'] for item in usage)
            out['tokens'][key] = sum(item['prompt_tokens'] + item['completion_tokens'] for item in usage)
        return out


def summary(result, *others):
    """F1 with interval, EM, cost, and the paired F1 difference to each of `others`."""
    if result is None:
        return None
    row = {'n': len(result['keys']), 'f1': interval(list(result['f1'].values())),
           'em': round(100 * float(np.mean(list(result['em'].values()))), 2),
           'calls': round(float(np.mean(list(result['calls'].values()))), 2),
           'tokens': round(float(np.mean(list(result['tokens'].values()))))}
    for name, other in others:
        shared = [key for key in result['keys'] if other is not None and key in other['f1']]
        row['minus_' + name] = interval([result['f1'][key] - other['f1'][key] for key in shared]) if shared and other is not result else None
    return row


def cell(value):
    return '–' if value is None else '%.2f [%.2f, %.2f]' % tuple(value)


def diff(value):
    return '–' if value is None else '%+.2f [%.2f, %.2f]' % tuple(value)


def score(value):
    return '–' if value is None else '%.2f / %.2f' % (value['f1'][0], value['em'])


def table(title, note, header, rows):
    lines = ['## ' + title, ''] + ([note, ''] if note else []) + ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return lines + ['| ' + ' | '.join(str(item) for item in row) + ' |' for row in rows] + ['']


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    negatives = np.sort(p[~y])
    below = np.searchsorted(negatives, p[y], side='left')
    equal = np.searchsorted(negatives, p[y], side='right') - below
    return round(float((below + 0.5 * equal).sum() / (y.sum() * (~y).sum())), 3)


def calibration(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins, error = [], 0.0
    for low in (0.0, 0.2, 0.4, 0.6, 0.8):
        mask = (p >= low) & ((p < low + 0.2) if low < 0.8 else (p <= 1.0))
        if mask.any():
            bins.append({'bin': '%.1f-%.1f' % (low, low + 0.2), 'n': int(mask.sum()), 'predicted': round(100 * float(p[mask].mean()), 1),
                         'right': round(100 * float(y[mask].mean()), 1)})
            error += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {'bins': bins, 'ece': round(100 * float(error), 2), 'auc': auc(p, y), 'brier': round(float(np.mean((p - y) ** 2)), 4)}


def analyse(bench, final, entry, pages, title):
    """All tables of one dataset and split. Returns the final method's merged result."""
    derivations = [name for name in final if name in bench.runs]
    singles = {name: bench.evaluate([name]) for name in derivations}
    full, fixed = bench.evaluate(final), singles['fixed']
    against = (('final', full), ('fixed', fixed), ('dense', singles['dense']))
    lazy = bench.evaluate(final, on_demand=True)
    entry['main'] = {'dense': summary(singles['dense'], *against), 'fixed': summary(fixed, *against),
                     'final': summary(full, *against), 'on_demand': summary(lazy, *against)}

    # 1. which derivations matter: single derivations, leave one out, smaller sets
    rows, entry['derivations'] = [], {}

    def add(label, result, kind):
        value = summary(result, *against)
        if value is not None:
            entry['derivations'][label] = dict(value, kind=kind, use=result['use'])
            rows.append([label, kind, cell(value['f1']), '%.2f' % value['em'], '%.1f' % value['calls'], value['tokens'],
                         diff(value['minus_final']), diff(value['minus_fixed'])])

    for name in derivations:
        add('single: ' + LABELS[name], singles[name], 'single derivation')
    add('final: four derivations merged by support', full, 'final')
    add('final, expanded on demand', lazy, 'final')
    for name in final:
        add('without ' + LABELS[name], bench.evaluate([other for other in final if other != name]), 'leave one out')
    for order in (['fixed', 'memory2'], ['fixed', 'memory1', 'memory2'], ['dense', 'fixed'], ['dense', 'fixed', 'memory2']):
        if bench.has(order):
            add(' + '.join(LABELS[name] for name in order), bench.evaluate(order), 'subset')
    pages['derivations'] += table(title, None, ['configuration', 'kind', 'F1 [95% CI]', 'EM', 'calls', 'tokens',
                                                'minus final', 'minus A1'], rows)

    # 2. how the answers are combined
    rows, entry['merge_rule'] = [], {}
    for label, rule in (('support: noisy-OR of calibrated reliabilities (final)', 'support'),
                        ('vote: every derivation counts the same', 'vote'),
                        ('most reliable derivation, no pooling', 'reliable'),
                        ('oracle: best of the four (upper bound, reads the gold answer)', 'oracle')):
        value = entry['merge_rule'][rule] = summary(bench.evaluate(final, rule=rule), *against)
        rows.append([label, cell(value['f1']), '%.2f' % value['em'], '%.1f' % value['calls'], diff(value['minus_final']),
                     diff(value['minus_fixed'])])
    value = summary(bench.selected(final), *against)
    if value is not None:  # control added 2026-11-19: one more LLM call on the questions where the derivations disagree
        entry['merge_rule']['llm_select'] = value
        rows.insert(3, ['LLM selector: the reader picks among the distinct answers, seeing every derivation\'s steps and quotes',
                        cell(value['f1']), '%.2f' % value['em'], '%.1f' % value['calls'], diff(value['minus_final']),
                        diff(value['minus_fixed'])])
    pages['merge_rule'] += table(title, None, ['rule', 'F1 [95% CI]', 'EM', 'calls', 'minus final', 'minus A1'], rows)

    # 3. which features the reliability model needs (refitted on the same train-S runs with a feature group zeroed)
    chains = [name for name in final if name != 'dense']
    gap = float(np.max(np.abs(fitted(bench.source).weights - bench.model.weights)))
    rows, entry['support_features'] = [], {'refit_weight_gap': round(gap, 6)}
    options = [('all features (refit)', ())] + [('without ' + name, tuple(columns)) for name, columns in GROUPS.items()]
    options.append(('no features: every chain gets the same reliability', tuple(FEATURES[1:])))
    for label, drop in options:
        model = fitted(bench.source, drop)
        value = entry['support_features'][label] = summary(bench.evaluate(final, model=model), *against)
        p = [bench.reliability(name, key, model) for name in chains for key in full['keys']]
        y = [bench.runs[name][0][key]['f1'] >= 0.5 for name in chains for key in full['keys']]
        value['auc'] = auc(p, y)
        rows.append([label, cell(value['f1']), '%.2f' % value['em'], diff(value['minus_final']), value['auc']])
    pages['support_features'] += table(title, 'Refit on train-S of %s; largest weight difference to the saved model %.5f.' % (
        NAMES[bench.source], gap), ['reliability model', 'F1 [95% CI]', 'EM', 'minus final', 'AUC of chain reliability'], rows)

    # 4. is the reliability calibrated on this split
    p = [bench.reliability(name, key) for name in chains for key in full['keys']]
    y = [bench.runs[name][0][key]['f1'] >= 0.5 for name in chains for key in full['keys']]
    merged_p = [full['support'][key] for key in full['keys'] if full['support'][key] is not None]
    merged_y = [full['f1'][key] >= 0.5 for key in full['keys'] if full['support'][key] is not None]
    entry['calibration'] = {'chain_reliability': calibration(p, y), 'merged_support': calibration(merged_p, merged_y)}
    for name, label in (('chain_reliability', 'reliability of single chains'), ('merged_support', 'support of the merged answer')):
        value = entry['calibration'][name]
        pages['calibration'] += table('%s: %s' % (title, label), 'ECE %.2f, AUC %s, Brier %.4f.' % (
            value['ece'], value['auc'], value['brier']), ['predicted bin', 'n', 'mean predicted %', 'share right %'],
            [[row['bin'], row['n'], row['predicted'], row['right']] for row in value['bins']])

    # 5. reliability model fitted on another dataset
    rows, entry['transfer'] = [], {}
    for source in ('musique', '2wikimultihopqa', 'hotpotqa'):
        path = ROOT / ('configs/support/%s_chain_v1.json' % source)
        if path.exists():
            value = entry['transfer'][source] = summary(bench.evaluate(final, model=SupportModel.load(path)), *against)
            rows.append([NAMES[source] + (' (used in the final method)' if source == bench.source else ''), cell(value['f1']),
                         '%.2f' % value['em'], diff(value['minus_final'])])
    pages['transfer'] += table(title, None, ['reliability model fitted on', 'F1 [95% CI]', 'EM', 'minus final'], rows)

    # 6. by question type or hop count
    rows, entry['breakdown'] = [], {}
    oracle = bench.evaluate(final, rule='oracle')
    for stratum in sorted({str(bench.runs['fixed'][1][key].get('stratum')) for key in full['keys']}):
        keys = [key for key in full['keys'] if str(bench.runs['fixed'][1][key].get('stratum')) == stratum]
        value = entry['breakdown'][stratum] = {'n': len(keys)}
        for name, result in (('dense', singles['dense']), ('fixed', fixed), ('final', full), ('oracle', oracle)):
            value[name] = round(100 * float(np.mean([result['f1'][key] for key in keys])), 2)
        value['evidence_all'] = round(100 * float(np.mean([bool(bench.runs['fixed'][1][key].get('evidence_all_supporting'))
                                                            for key in keys])), 1)
        rows.append([stratum, len(keys), value['dense'], value['fixed'], value['final'], value['oracle'], value['evidence_all']])
    pages['breakdown'] += table(title, None, ['type', 'n', 'dense read F1', 'chain A1 F1', 'final F1', 'oracle of the four F1',
                                              'A1: all evidence found %'], rows)

    return full, lazy, derivations


def supplement(bench, final, entry, pages, title, full, samples):
    """Tables from the supplementary runs and from refits: self-consistency, chain design, training of the reliability
    model, agreement between derivations, plan memory against similarity to the training questions."""
    fixed = bench.evaluate(['fixed'])
    against = (('final', full), ('fixed', fixed))
    line = lambda label, value, *extra: [label, cell(value['f1']), '%.2f' % value['em'], *extra,
                                         diff(value.get('minus_final')), diff(value.get('minus_fixed'))]

    # 1. self-consistency: the same chain sampled again, against different derivations
    rows, entry['self_consistency'] = [], {}
    for temperature, tags in SAMPLES.items():
        for tag in tags:
            if tag in bench.runs:
                value = entry['self_consistency']['sample ' + tag] = summary(bench.evaluate([tag]), *against)
                rows.append(line('one sample at temperature %.1f (%s)' % (temperature, tag), value, '%.1f' % value['calls']))
        for count in (2, 3):
            use = ['fixed'] + tags[:count]
            if count <= len(tags) and bench.has(use):
                for rule, name in (('vote', 'majority vote'), ('support', 'calibrated support')):
                    label = 'A1 + %d samples at temperature %.1f, %s' % (count, temperature, name)
                    value = entry['self_consistency'][label] = summary(bench.evaluate(use, rule=rule), *against)
                    rows.append(line(label, value, '%.1f' % value['calls']))
        use = [name for name in final if not name.startswith('memory')] + tags[:2]
        if len(tags) >= 2 and bench.has(use):  # the final method with its two memory plans replaced by two plain samples
            label = 'dense read + A1 + 2 samples at temperature %.1f, calibrated support' % temperature
            value = entry['self_consistency'][label] = summary(bench.evaluate(use), *against)
            rows.append(line(label, value, '%.1f' % value['calls']))
    if rows:
        head = [line('chain A1 (greedy)', summary(fixed, *against), '%.1f' % summary(fixed)['calls']),
                line('final: dense read + A1 + memory plans 1 and 2, calibrated support', summary(full, *against),
                     '%.1f' % summary(full)['calls'])]
        pages['self_consistency'] += table(title, None, ['configuration', 'F1 [95% CI]', 'EM', 'calls', 'minus final', 'minus A1'],
                                           head + rows)

    # 2. design of the chain: the final read, what it sees, passages per hop
    by_id = {example.id: example for example in samples}
    keys = [key for key in full['keys'] if key in by_id]
    records = bench.runs['fixed'][1]
    _, per = score_answers(ROOT, [by_id[key] for key in keys], [records[key].get('chain_answer') or '' for key in keys])
    value = entry.setdefault('chain_design', {})['no final reader'] = {
        'n': len(keys), 'f1': interval([row['f1'] for row in per]), 'em': round(100 * float(np.mean([row['em'] for row in per])), 2),
        'minus_fixed': interval([row['f1'] - fixed['f1'][key] for row, key in zip(per, keys)]),
        'minus_final': interval([row['f1'] - full['f1'][key] for row, key in zip(per, keys)])}
    rows = [line('chain A1: final reader sees hop notes and passages', summary(fixed, *against), '%.1f' % summary(fixed)['calls']),
            line('no final reader: the answer of the last hop is the answer', value, '–')]
    for tag, label in VARIANTS:
        if tag in bench.runs:
            value = entry['chain_design'][tag] = summary(bench.evaluate([tag]), *against)
            rows.append(line(label, value, '%.1f' % value['calls']))
    pages['chain_design'] += table(title, None, ['configuration', 'F1 [95% CI]', 'EM', 'calls', 'minus final', 'minus A1'], rows)

    # 3. how much training the reliability model needs, and one model for all datasets
    rows, entry['reliability_training'] = [], {}
    for limit in (100, 250, 500, 1000):
        value = entry['reliability_training']['%d questions' % limit] = summary(
            bench.evaluate(final, model=fitted(bench.source, limit=limit)), *against)
        rows.append(line('%d training questions' % limit, value))
    rows.append(line('1,500 training questions (final)', summary(full, *against)))
    value = entry['reliability_training']['pooled'] = summary(
        bench.evaluate(final, model=fitted(('musique', '2wikimultihopqa', 'hotpotqa'))), *against)
    rows.append(line('one model for all datasets (training questions of the three multi-hop sets pooled)', value))
    pages['reliability_training'] += table(title, 'Reliability model of %s, refitted on a fixed random subset of its training '
                                           'questions.' % NAMES[bench.source], ['reliability model', 'F1 [95% CI]', 'EM',
                                                                                 'minus final', 'minus A1'], rows)

    # 4. agreement between the derivations
    answers = {key: [bench.runs[name][1][key].get('reader_clean') or '' for name in final] for key in full['keys']}
    agree = np.array([all(answer.strip() for answer in answers[key]) and len(answer_groups(answers[key])) == 1 for key in full['keys']])
    pair = np.array([same_answer(answers[key][final.index('dense')], answers[key][final.index('fixed')]) for key in full['keys']])
    score_of = lambda result, mask: round(100 * float(np.mean([result['f1'][key] for key, keep in zip(full['keys'], mask) if keep])), 2) \
        if mask.any() else None
    entry['agreement'] = {'all_four_agree_percent': round(100 * float(agree.mean()), 1), 'final_f1_when_all_agree': score_of(full, agree),
                          'final_f1_when_they_differ': score_of(full, ~agree), 'a1_f1_when_they_differ': score_of(fixed, ~agree),
                          'dense_and_a1_agree_percent': round(100 * float(pair.mean()), 1),
                          'final_f1_when_dense_and_a1_agree': score_of(full, pair), 'final_f1_when_dense_and_a1_differ': score_of(full, ~pair),
                          'a1_f1_when_dense_and_a1_differ': score_of(fixed, ~pair)}

    # 5. plan memory against the similarity of the question to the nearest training question
    memory = bench.evaluate(['memory2'])
    train = ROOT / 'runs/train_s/embeddings' / bench.source / 'queries.f16.npy'
    queries = (ROOT / 'runs/p0/calibration' / bench.dataset / 'queries.f16.npy' if bench.split == 'test'
               else ROOT / 'runs/p0_dev/embeddings' / bench.dataset / ('queries_%s.f16.npy' % bench.split))
    if memory is not None and train.exists() and queries.exists():
        unit = lambda path: (lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(np.load(path).astype(np.float32))
        stored = unit(queries)
        if stored.shape[0] == len(samples):
            nearest = dict(zip([example.id for example in samples], (stored @ unit(train)[:1500].T).max(axis=1)))
            rows, entry['plan_memory_similarity'] = [], {}
            for low, high in ((-1.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 2.0)):
                part = [key for key in full['keys'] if key in memory['f1'] and low <= nearest[key] < high]
                if part:
                    label = 'below 0.3' if low < 0 else '0.6 and above' if high > 1 else '%.1f to %.1f' % (low, high)
                    gain = interval([memory['f1'][key] - fixed['f1'][key] for key in part]) if len(part) >= 20 else None
                    entry['plan_memory_similarity'][label] = {
                        'n': len(part), 'a1': round(100 * float(np.mean([fixed['f1'][key] for key in part])), 2),
                        'memory2': round(100 * float(np.mean([memory['f1'][key] for key in part])), 2), 'memory2_minus_a1': gain}
                    rows.append([label, len(part), entry['plan_memory_similarity'][label]['a1'],
                                 entry['plan_memory_similarity'][label]['memory2'], diff(gain)])
            pages['plan_memory_similarity'] += table(
                title, 'Cosine of the question to its nearest training question in the plan memory (%s); median %.3f.' % (
                    NAMES[bench.source], float(np.median([nearest[key] for key in full['keys']]))),
                ['similarity', 'n', 'chain A1 F1', 'memory plan 2 F1', 'memory plan 2 minus A1 [95% CI]'], rows)


def load_samples_of(bench):
    return load_test_samples(ROOT, bench.dataset) if bench.split == 'test' else load_samples(
        ROOT / 'data/splits/seed42' / bench.dataset / (bench.split + '.json'), bench.dataset)


def write_raw(bench, final, full, lazy, derivations, folder, samples):
    gold = {example.id: example for example in samples}
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ('%s_predictions.jsonl' % bench.split)).open('w', encoding='utf-8') as stream:
        for key in full['keys']:
            example = gold.get(key)
            line = {'id': key, 'question': bench.runs['fixed'][1][key]['question'],
                    'gold_answers': list(example.answers) if example else None,
                    'type': str(bench.runs['fixed'][1][key].get('stratum')), 'derivations': {}}
            for name in derivations:
                scored, records = bench.runs[name]
                if key not in records:
                    continue
                record = records[key]
                line['derivations'][name] = {
                    'in_final': name in final, 'answer': record.get('reader_clean') or '', 'f1': round(scored[key]['f1'], 4),
                    'em': scored[key]['em'], 'reliability': round(bench.reliability(name, key), 4), 'plan': record.get('hops') or [],
                    'complete': bool(record.get('complete')), 'evidence_recall': round(record.get('evidence_recall', 0.0), 4),
                    'calls': record['usage']['calls'], 'tokens': record['usage']['prompt_tokens'] + record['usage']['completion_tokens']}
            for name, result in (('final', full), ('final_on_demand', lazy)):
                if result is None or key not in result['f1']:
                    continue
                line[name] = {'from': result['choice'][key], 'answer': bench.runs[result['choice'][key]][1][key].get('reader_clean') or '',
                              'f1': round(result['f1'][key], 4), 'em': result['em'][key],
                              'support': None if result['support'][key] is None else round(result['support'][key], 4),
                              'calls': result['calls'][key], 'tokens': result['tokens'][key]}
            stream.write(json.dumps(line, ensure_ascii=False) + '\n')


def baseline_runs(results, merged):
    """Baselines with the same reader and embedder, and dropped variants, against chain A1 and the final method, on every
    split where they ran (the baselines HippoRAG 2 and IRCoT: dev-A in the development phase, test on 2026-11-18; HippoRAG 2
    also on the single-hop test sets, 2026-11-19)."""
    lines = []
    for split in ('test', 'dev-A', 'dev-B'):
        for dataset in ('musique', '2wikimultihopqa', 'hotpotqa', 'nq_rear', 'popqa'):
            scores = {}
            for tag, _ in BASELINES:
                path = ROOT / 'runs/chain' / split / dataset / tag / 'scored.json'
                if path.exists():
                    scores[tag] = {row['id']: row for row in json.loads(path.read_text())}
            if 'single' not in scores:
                continue
            full, rows = merged.get((split, dataset)), []
            entry = results.setdefault('baseline_runs', {}).setdefault(split, {}).setdefault(dataset, {})
            if full is not None:
                scores['final'] = {key: {'f1': full['f1'][key], 'em': full['em'][key]} for key in full['keys']}
            for tag, label in BASELINES + (('final', 'final: four derivations merged by support'),):
                if tag not in scores:
                    continue
                keys = sorted(set(scores[tag]) & set(scores['single']))
                value = entry[tag] = {'n': len(keys), 'f1': interval([scores[tag][key]['f1'] for key in keys]),
                                      'em': round(100 * float(np.mean([scores[tag][key]['em'] for key in keys])), 2)}
                for name in ('single', 'final'):
                    shared = [key for key in keys if name in scores and key in scores[name]]
                    value['minus_' + name] = interval([scores[tag][key]['f1'] - scores[name][key]['f1'] for key in shared]) \
                        if shared and name != tag else None
                rows.append([label, value['n'], cell(value['f1']), '%.2f' % value['em'], diff(value['minus_single']),
                             diff(value['minus_final'])])
            lines += table('%s, %s' % (NAMES[dataset], split), None, ['system', 'n', 'F1 [95% CI]', 'EM', 'minus A1', 'minus final'], rows)
    return lines


def overview(results, split):
    """The per-dataset tables of one split side by side: one column per dataset."""
    data = results['splits'].get(split, {})
    if not data:
        return []
    names = list(data)
    header = [NAMES[name] for name in names]
    change = lambda value, key='minus_final': '–' if value is None or value.get(key) is None else diff(value[key])
    mean = lambda values: '%.2f' % float(np.mean(values))
    multihop = [name for name in names if name in ('musique', '2wikimultihopqa', 'hotpotqa')]
    rows = []
    for key, label in (('dense', 'dense read'), ('fixed', 'chain A1'), ('final', 'final method'), ('on_demand', 'final, on demand')):
        values = [data[name]['main'][key] for name in multihop]
        rows.append([label] + [score(data[name]['main'][key]) for name in names] + (
            ['%s / %s' % (mean([value['f1'][0] for value in values]), mean([value['em'] for value in values]))]
            if all(value is not None for value in values) and values else ['–']))
    rows.append(['calls per question: A1 / final / on demand'] + ['%.1f / %.1f / %.1f' % tuple(
        data[name]['main'][key]['calls'] for key in ('fixed', 'final', 'on_demand')) for name in names] + ['–'])
    rows.append(['final minus A1'] + [change(data[name]['main']['final'], 'minus_fixed') for name in names] + ['–'])
    rows.append(['final minus dense read'] + [change(data[name]['main']['final'], 'minus_dense') for name in names] + ['–'])
    lines = table('Main results, %s: F1 / EM' % split, 'Mean over the multi-hop sets (MuSiQue, 2Wiki, HotpotQA).',
                  ['system'] + header + ['multi-hop mean'], rows)
    labels = list(dict.fromkeys(label for name in names for label in data[name]['derivations']))
    lines += table('Single derivations and subsets, %s: F1 / EM' % split, None, ['configuration'] + header,
                   [[label] + [score(data[name]['derivations'].get(label)) for name in names] for label in labels])
    lines += table('Leave one derivation out, %s: F1 minus final [95%% CI]' % split, None, ['configuration'] + header,
                   [[label] + [change(data[name]['derivations'].get(label)) for name in names]
                    for label in labels if data[names[0]]['derivations'].get(label, {}).get('kind') == 'leave one out'])
    lines += table('Merge rule, %s: F1 / EM, and F1 minus final' % split, None, ['rule'] + header,
                   [[rule] + ['%s, %s' % (score(data[name]['merge_rule'].get(rule)), change(data[name]['merge_rule'].get(rule)))
                              for name in names]
                    for rule in ('support', 'vote', 'reliable', 'llm_select', 'oracle')
                    if any(rule in data[name]['merge_rule'] for name in names)])
    groups = [label for label in data[names[0]]['support_features'] if label != 'refit_weight_gap']
    lines += table('Reliability features, %s: F1 minus final [95%% CI] (AUC of chain reliability)' % split, None,
                   ['reliability model'] + header,
                   [[label] + ['%s (%s)' % (change(data[name]['support_features'][label]), data[name]['support_features'][label]['auc'])
                               for name in names] for label in groups])
    lines += table('Reliability model fitted on another dataset, %s: final F1 / EM' % split,
                   'Used in the final method: ' + ', '.join('%s for %s' % (NAMES[data[name]['support_model_from']], NAMES[name])
                                                            for name in names) + '.',
                   ['fitted on'] + header, [[NAMES[source]] + [score(data[name]['transfer'].get(source)) for name in names]
                                            for source in ('musique', '2wikimultihopqa', 'hotpotqa')])
    lines += table('Calibration, %s' % split, None, ['quantity'] + header,
                   [['chain reliability: ECE / AUC'] + ['%.2f / %s' % (data[name]['calibration']['chain_reliability']['ece'],
                                                                     data[name]['calibration']['chain_reliability']['auc']) for name in names],
                    ['merged support: ECE / AUC'] + ['%.2f / %s' % (data[name]['calibration']['merged_support']['ece'],
                                                                   data[name]['calibration']['merged_support']['auc']) for name in names]])

    def side_by_side(title, section, cells, note=None):
        labels = list(dict.fromkeys(label for name in names for label in data[name].get(section, {})))
        if labels:
            lines.extend(table(title, note, [section.replace('_', ' ')] + header,
                               [[label] + [cells(data[name].get(section, {}).get(label)) for name in names] for label in labels]))

    both = lambda value: '–' if value is None else '%s, %s' % (score(value), change(value))
    side_by_side('Self-consistency against different derivations, %s: F1 / EM, and F1 minus final' % split, 'self_consistency', both)
    side_by_side('Design of the chain, %s: F1 / EM, and F1 minus A1' % split, 'chain_design',
                 lambda value: '–' if value is None else '%s, %s' % (score(value), change(value, 'minus_fixed')))
    side_by_side('Training of the reliability model, %s: final F1 / EM, and F1 minus final' % split, 'reliability_training', both)
    side_by_side('Agreement between the four derivations, %s' % split, 'agreement', lambda value: '–' if value is None else value)
    side_by_side('Memory plan 2 against the similarity to the nearest training question, %s: n, A1 F1, memory plan 2 F1, difference' % split,
                 'plan_memory_similarity', lambda value: '–' if value is None else 'n %d: %.1f → %.1f, %s' % (
                     value['n'], value['a1'], value['memory2'], diff(value['memory2_minus_a1'])))
    runs = results.get('baseline_runs', {}).get(split, {})
    if runs:
        order = [name for name in names if name in runs]
        labels = [(tag, label) for tag, label in BASELINES + (('final', 'final: four derivations merged by support'),)
                  if any(tag in runs[name] for name in order)]
        lines += table('Same reader and embedder, and dropped variants, %s: F1 / EM' % split, None,
                       ['system'] + [NAMES[name] for name in order],
                       [[label] + [score(runs[name].get(tag)) for name in order] for tag, label in labels])
        lines += table('Final method minus each system, %s: F1 [95%% CI]' % split, None, ['system'] + [NAMES[name] for name in order],
                       [[label] + ['–' if runs[name].get(tag, {}).get('minus_final') is None else diff([
                           -runs[name][tag]['minus_final'][0], -runs[name][tag]['minus_final'][2], -runs[name][tag]['minus_final'][1]])
                           for name in order] for tag, label in labels if tag != 'final'])
    return lines


def cost(results):
    """LLM tokens: to index the corpus, to prepare the trained parts, and per question. Test split."""
    rows, record = [], results.setdefault('cost', {})
    tokens = lambda calls: calls.get('prompt_tokens', 0) + calls.get('completion_tokens', 0)
    for dataset, entry in results['splits'].get('test', {}).items():
        source, main, prepare = entry['support_model_from'], entry['main'], 0
        for tag in ('a0_fit', 'single_fit', 'selfmem_answer_fit') + (('cand3_fit_reader',) if source == 'musique' else ()):
            run = load('train-S', source, tag)
            if run is not None:
                prepare += sum(row['usage']['prompt_tokens'] + row['usage']['completion_tokens'] for row in run[1].values())
        systems = [('dense read', 0, 0, main['dense']['calls'], main['dense']['tokens']),
                   ('chain A1', 0, 0, main['fixed']['calls'], main['fixed']['tokens']),
                   ('final', 0, prepare, main['final']['calls'], main['final']['tokens']),
                   ('final, on demand', 0, prepare, main['on_demand']['calls'], main['on_demand']['tokens'])]
        path = ROOT / 'runs/chain/test' / dataset / 'b3_ircot/summary.json'
        if path.exists():
            summary_ = json.loads(path.read_text())
            systems.append(('IRCoT, same reader', 0, 0, summary_['llm_calls_per_question'], summary_['tokens_per_question']))
        path = ROOT / 'runs/chain/test' / dataset / 'b2_hipporag2/summary.json'
        if path.exists():
            summary_ = json.loads(path.read_text())
            index = tokens(summary_.get('openie', {}).get('extraction_calls', {}))
            report = ROOT / 'runs/b2_test' / dataset / 'openie_report.json'
            if report.exists():
                # The whole extraction ledger. A rerun after a failure answers the earlier requests from the cache,
                # so the last run's own count leaves their tokens out (NQ and PopQA: the entity step).
                index = tokens(json.loads(report.read_text()).get('extraction_calls', {}))
            run = summary_.get('run', {})
            calls = run.get('extraction_calls', {}).get('logical_calls', 0) + run.get('reader_calls', {}).get('logical_calls', 0)
            used = tokens(run.get('extraction_calls', {})) + tokens(run.get('reader_calls', {}))
            questions = max(1, summary_.get('requested', 1))
            systems.append(('HippoRAG 2, same reader', index, 0, calls / questions, used / questions))
        for name, index, prepared, calls, per_question in systems:
            record.setdefault(dataset, {})[name] = {'index_tokens': index, 'preparation_tokens': prepared, 'calls_per_question': calls,
                                                     'tokens_per_question': per_question}
            rows.append([NAMES[dataset], name, '%.1fM' % (index / 1e6) if index else '0', '%.1fM' % (prepared / 1e6) if prepared else '0',
                         '%.1f' % calls, '%d' % per_question, '%.1fM' % ((index + 1000 * per_question) / 1e6)])
    return table('LLM cost, test', 'Tokens are prompt plus completion tokens. Indexing: LLM calls over the corpus before any question. '
                 'Preparation: the runs on %s whose answers fit the reliability model and fill the plan memories, once per dataset '
                 '(NQ and PopQA reuse those of HotpotQA). The last column is indexing plus 1,000 questions.' % 'the 1,500 training questions',
                 ['dataset', 'system', 'indexing', 'preparation (once)', 'calls per question', 'tokens per question',
                  'indexing + 1,000 questions'], rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '--final', type=Path, default=ROOT / 'configs/final.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'paper')
    parser.add_argument('--splits', nargs='+', default=['test', 'dev-A'])
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    final, transfer = config['final'], config['transfer']
    results = {'final': final, 'splits': {}}
    pages = {name: [] for name in ('main', 'derivations', 'merge_rule', 'support_features', 'calibration', 'transfer', 'breakdown',
                                   'self_consistency', 'chain_design', 'reliability_training',
                                   'plan_memory_similarity')}
    merged = {}
    for split in arguments.splits:
        main_rows = []
        for dataset in config['datasets']:
            bench = Bench(config, transfer, dataset, split)
            if not bench.ids or not bench.has(final) or bench.model is None:
                continue
            title = '%s, %s (%d questions)' % (NAMES[dataset], split, len(bench.ids))
            entry = results['splits'].setdefault(split, {})[dataset] = {'support_model_from': bench.source}
            full, lazy, derivations = analyse(bench, final, entry, pages, title)
            merged[(split, dataset)] = full
            samples = load_samples_of(bench)
            supplement(bench, final, entry, pages, title, full, samples)
            write_raw(bench, final, full, lazy, derivations, arguments.out / 'raw' / dataset, samples)
            main = entry['main']
            main_rows.append([NAMES[dataset]] + [score(main[name]) for name in ('dense', 'fixed', 'final', 'on_demand')]
                             + ['%.1f / %.1f' % (main['final']['calls'], main['on_demand']['calls']), diff(main['final']['minus_fixed'])])
        if main_rows:
            pages['main'] += table('Main results, %s' % split, 'F1 / EM. Final method: %s.' % config['label'],
                                   ['dataset', 'dense read', 'chain A1', 'final', 'final, on demand',
                                    'calls: final / on demand', 'final minus A1'], main_rows)
    pages['baseline_runs'] = baseline_runs(results, merged)
    pages['cost'] = cost(results)
    for split in arguments.splits:
        pages['overview_' + split.replace('-', '_')] = overview(results, split)
    (arguments.out / 'tables').mkdir(parents=True, exist_ok=True)
    for name, lines in pages.items():
        (arguments.out / 'tables' / (name + '.md')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (arguments.out / 'results.json').write_text(json.dumps(results, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('\n'.join(pages['main']))
    print('written: %s' % (arguments.out / 'results.json'))


if __name__ == '__main__':
    main()
