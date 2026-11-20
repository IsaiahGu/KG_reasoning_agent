"""Answer support across several derivations: the read-out of the derivation memory.

A derivation is one way of reaching an answer: a plan executed hop by hop, or a direct read of retrieved passages.
Each derivation gets a reliability from features that can be observed without gold answers. The support of an answer
is the probability that at least one of its derivations is sound (noisy-OR over the derivations that end in it).
The reliability model is a logistic regression fitted on training questions with known answers; no LLM weights change.
"""
import json
from pathlib import Path

import numpy as np

from hyperbind.textmatch import same_answer, tokens

STOP = set('a an the of in on at to for from by with and or is was were are be been who whom whose what which when where '
           'why how did does do that this these those it its as into than then also s'.split())
FEATURES = ['bias', 'complete', 'hops', 'chain_reader_agree', 'marker', 'min_explicit', 'mean_verbatim', 'min_answer_in_quote',
            'max_rank', 'min_passage_score', 'min_question_score', 'linked', 'coverage', 'answer_in_question', 'backtracked',
            'min_next_top1', 'one_hop']


def chain_features(record):
    """Features of one executed plan, from its record (see ChainAgent.run). No gold field is read."""
    chosen = [step['candidates'][step['chosen']] for step in record.get('steps', [])
              if step.get('candidates') and step.get('chosen') is not None]
    rows = [candidate['features'] for candidate in chosen] or [{}]
    column = lambda name: [row.get(name, 0.0) for row in rows]
    hops = record.get('hops') or []
    question = set(tokens(record['question'])) - STOP
    covered = {token for hop in hops for token in tokens(hop)}
    answer = set(tokens(record.get('reader_clean') or ''))
    lookahead = [row.get('next_top1', 0.0) for row in rows if row.get('has_next')]
    return np.array([
        1.0, float(record.get('complete', False)), len(hops) / 4.0,
        float(same_answer(record.get('chain_answer') or '', record.get('reader_clean') or '')),
        float(bool(record.get('reader_marker'))),
        min(column('explicit')), float(np.mean(column('quote_verbatim'))), min(column('answer_in_quote')),
        max(column('passage_rank')) / 8.0, min(column('passage_score')), min(column('question_score')),
        float(np.mean(['#' in hop for hop in hops[1:]])) if hops[1:] else 1.0,
        len(question & covered) / max(1, len(question)),
        float(bool(answer) and answer <= set(tokens(record['question']))),
        float(record.get('backtracked', False)), min(lookahead) if lookahead else 0.5, float(len(hops) <= 1)])


class SupportModel:
    def __init__(self, weights, mean, scale, dense_prior, names=FEATURES):
        self.weights, self.mean, self.scale = np.asarray(weights, float), np.asarray(mean, float), np.asarray(scale, float)
        self.dense_prior, self.names = float(dense_prior), list(names)

    @classmethod
    def fit(cls, X, y, dense_prior=0.5, l2=1.0, rounds=3000, rate=0.3):
        X, y = np.asarray(X, float), np.asarray(y, float)
        mean, scale = X.mean(0), X.std(0) + 1e-6
        mean[0], scale[0] = 0.0, 1.0
        Z, weights = (X - mean) / scale, np.zeros(X.shape[1])
        for _ in range(rounds):
            p = 1 / (1 + np.exp(-Z @ weights))
            weights -= rate * (Z.T @ (p - y) / len(y) + l2 * np.r_[0, weights[1:]] / len(y))
        return cls(weights, mean, scale, dense_prior)

    def predict(self, X):
        return 1 / (1 + np.exp(-((np.atleast_2d(X) - self.mean) / self.scale) @ self.weights))

    def reliability(self, record, dense=False):
        return self.dense_prior if dense else float(self.predict(chain_features(record))[0])

    def to_dict(self):
        return {'names': self.names, 'weights': self.weights.tolist(), 'mean': self.mean.tolist(),
                'scale': self.scale.tolist(), 'dense_prior': self.dense_prior}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['names'] != FEATURES:
            raise ValueError('The support model was fitted with a different feature list')
        return cls(data['weights'], data['mean'], data['scale'], data['dense_prior'], data['names'])


def answer_groups(answers):
    """Indices of derivations that end in the same answer (token-level match). Empty answers form no group."""
    groups = []
    for index, answer in enumerate(answers):
        if not (answer or '').strip():
            continue
        for group in groups:
            if same_answer(answers[group[0]], answer):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def merge(answers, reliabilities):
    """Pick the best-supported answer. Returns (index of its most reliable derivation, support, number of answers)."""
    groups = answer_groups(answers)
    if not groups:
        return int(np.argmax(reliabilities)), 0.0, 0
    support = [1 - float(np.prod([1 - reliabilities[index] for index in group])) for group in groups]
    best = groups[int(np.argmax(support))]
    return max(best, key=lambda index: reliabilities[index]), max(support), len(groups)
