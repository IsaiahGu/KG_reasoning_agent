"""Chain variant (learned hop scorer with backtracking): a small trained scorer that picks the binding of a hop among quoted candidates.

Model: conditional logit over the candidates of one hop (softmax of w.x), fitted on train-S hops whose earlier
hops were right. Features are computed in code from the retrieval scores, the quote and the next hop's retrieval;
no LLM call and no gold label is used at inference.
"""
import json
import re

import numpy as np

from .textmatch import contains, same_answer, tokens

FEATURE_NAMES = ['first', 'position', 'explicit', 'quote_verbatim', 'answer_in_quote', 'answer_in_passage', 'answer_in_question',
                 'passage_rank', 'passage_score', 'score_gap', 'title_in_question', 'bound_in_title', 'bound_in_passage',
                 'support_count', 'answer_tokens', 'digit_match', 'question_score',
                 'has_next', 'next_top1', 'next_top3', 'next_title_match']
# Version 2 adds one joint-support feature: how many of the next hop's top passages mention this candidate.
FEATURE_NAMES_V2 = FEATURE_NAMES + ['next_answer_in_top3']
FEATURE_SETS = {'v1': FEATURE_NAMES, 'v2': FEATURE_NAMES_V2}
NUMBER_QUESTION = re.compile(r'\s*(when|what year|in what year|in which year|what date|on what date|how many|how much|how long|'
                             r'how far|how old|what is the population|what was the population)\b', re.I)


def bare_title(title):
    return re.sub(r'\s*\([^)]*\)\s*', ' ', title).strip()


def candidate_features(*, sub_question, candidate, position, count, title, content, top_score, window_contents, bound,
                       hop, hops, question_score, ahead):
    answer, plain = candidate['answer'], bare_title(title)
    wants_number, has_digit = bool(NUMBER_QUESTION.match(sub_question)), any(character.isdigit() for character in answer)
    row = {'first': float(position == 0), 'position': float(position), 'explicit': float(candidate['support'] == 'explicit'),
           'quote_verbatim': float(candidate['quote_verbatim']), 'answer_in_quote': float(contains(candidate['quote'], answer)),
           'answer_in_passage': float(contains(content, answer)), 'answer_in_question': float(contains(sub_question, answer)),
           'passage_rank': float(candidate['passage_rank']), 'passage_score': float(candidate['passage_score']),
           'score_gap': float(top_score - candidate['passage_score']),
           'title_in_question': float(bool(tokens(plain)) and contains(sub_question, plain)),
           'bound_in_title': float(any(contains(title, entity) or contains(entity, plain) for entity in bound)),
           'bound_in_passage': float(any(contains(content, entity) for entity in bound)),
           'support_count': sum(contains(text, answer) for text in window_contents) / max(1, len(window_contents)),
           'answer_tokens': min(len(tokens(answer)), 8) / 8.0, 'digit_match': float(wants_number == has_digit),
           'question_score': float(question_score), 'has_next': 0.0, 'next_top1': 0.0, 'next_top3': 0.0, 'next_title_match': 0.0,
           'next_answer_in_top3': 0.0, 'hop': hop, 'hops': hops, 'count': count}
    if ahead is not None and ahead['scores']:
        row.update(has_next=1.0, next_top1=float(ahead['scores'][0]), next_top3=float(np.mean(ahead['scores'])),
                   next_title_match=float(any(same_answer(bare_title(name), answer) for name in ahead['titles'])))
        if ahead.get('contents'):
            row['next_answer_in_top3'] = sum(contains(text, answer) for text in ahead['contents']) / len(ahead['contents'])
    return row


class BindingScorer:
    def __init__(self, names, weights, mean, scale):
        self.names = list(names)
        self.weights, self.mean, self.scale = (np.asarray(value, dtype=np.float64) for value in (weights, mean, scale))

    def matrix(self, rows):
        values = np.asarray([[row[name] for name in self.names] for row in rows], dtype=np.float64)
        return (values - self.mean) / self.scale

    def logits(self, rows):
        return self.matrix(rows) @ self.weights

    def to_dict(self):
        return {'model': 'conditional logit over the candidates of one hop', 'names': self.names,
                'weights': self.weights.tolist(), 'mean': self.mean.tolist(), 'scale': self.scale.tolist()}

    @classmethod
    def from_dict(cls, value):
        return cls(value['names'], value['weights'], value['mean'], value['scale'])

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(open(path, encoding='utf-8').read()))

    @classmethod
    def fit(cls, groups, names=FEATURE_NAMES, l2=1.0, steps=1500, rate=0.05):
        """groups: list of (feature rows, 0/1 labels) with at least two rows and at least one positive."""
        if not groups:
            raise ValueError('No training groups')
        stacked = np.asarray([[row[name] for name in names] for rows, _ in groups for row in rows], dtype=np.float64)
        mean, scale = stacked.mean(axis=0), stacked.std(axis=0)
        scale[scale < 1e-9] = 1.0
        width = max(len(rows) for rows, _ in groups)
        features = np.zeros((len(groups), width, len(names)))
        mask, positive = np.zeros((len(groups), width), bool), np.zeros((len(groups), width))
        for index, (rows, labels) in enumerate(groups):
            features[index, :len(rows)] = (np.asarray([[row[name] for name in names] for row in rows]) - mean) / scale
            mask[index, :len(rows)] = True
            positive[index, :len(rows)] = labels
        weights, first, second = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
        for step in range(1, steps + 1):  # Adam ascent on the mean log-likelihood with an L2 penalty
            logits = np.where(mask, features @ weights, -np.inf)
            probability = np.exp(logits - logits.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
            posterior = probability * positive
            posterior /= posterior.sum(axis=1, keepdims=True)
            gradient = np.einsum('gc,gcd->d', posterior - probability, features) / len(groups) - l2 * weights / len(groups)
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient ** 2
            weights = weights + rate * (first / (1 - 0.9 ** step)) / (np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
        return cls(names, weights, mean, scale)


def hop_groups(records, require_prefix=True):
    """Labelled candidate groups from chain records (musique_labeler). One group per hop with at least one candidate."""
    groups = []
    for record in records:
        if not record.get('aligned'):
            continue
        for step in record['steps']:
            if not step['candidates'] or (require_prefix and not step.get('prefix_correct')):
                continue
            groups.append({'id': record['id'], 'hop': step['hop'], 'rows': [candidate['features'] for candidate in step['candidates']],
                           'labels': [candidate['label'] for candidate in step['candidates']]})
    return groups


def selection_report(groups, scorer=None):
    """How often the first candidate, the scorer's pick and the best possible pick are right, over the same hops."""
    usable = [group for group in groups if len(group['rows']) > 1]
    report = {'hops': len(groups), 'hops_with_two_or_more_candidates': len(usable),
              'first_correct_percent': round(100 * float(np.mean([group['labels'][0] for group in groups])), 2) if groups else None,
              'any_correct_percent': round(100 * float(np.mean([max(group['labels']) for group in groups])), 2) if groups else None}
    if usable:
        report['multi_first_correct_percent'] = round(100 * float(np.mean([group['labels'][0] for group in usable])), 2)
        report['multi_any_correct_percent'] = round(100 * float(np.mean([max(group['labels']) for group in usable])), 2)
    if scorer is not None and groups:
        picks = [group['labels'][int(np.argmax(scorer.logits(group['rows'])))] if len(group['rows']) > 1 else group['labels'][0]
                 for group in groups]
        report['scorer_correct_percent'] = round(100 * float(np.mean(picks)), 2)
        if usable:
            report['multi_scorer_correct_percent'] = round(100 * float(np.mean(
                [group['labels'][int(np.argmax(scorer.logits(group['rows'])))] for group in usable])), 2)
    return report
