#!/usr/bin/env python
"""Where does a MuSiQue chain first go wrong? Development splits only (dev-A, train-S held-out). CPU only.

Usage: root_cause.py <split> <tag> [other tag ...]
For every wrong answer (F1 below 0.5) of run <tag>, the root is the earliest gold hop (the gold decomposition is in
dependency order) whose gold answer is not among the answers the chain bound. The root is then one of
  retrieval miss      the supporting passage of the root hop was never retrieved, although every hop before it was bound
  passage not bound   the supporting passage was retrieved at some hop, but its answer was not bound
For retrieval misses the script also says whether the supporting passage mentions an entity the chain had already bound, and whether the other runs retrieved that passage.
Nothing is fitted or tuned here.
"""
import collections
import json
import re
import string
import sys
from pathlib import Path

ROOT = Path('.').resolve()
SPLIT, TAG, OTHERS = sys.argv[1], sys.argv[2], sys.argv[3:]
DATASET = 'musique'
folder = ROOT / 'data/splits/seed42' / DATASET
rows = {str(row['id']): row for row in json.loads((folder / (SPLIT + '.json')).read_text())}
corpus = json.loads((folder / ('train-S_corpus.json' if SPLIT == 'train-S' else 'dev_corpus.json')).read_text())
position = {(doc['title'], doc['text']): index for index, doc in enumerate(corpus)}


def normal(text):
    text = ''.join(ch for ch in str(text).lower() if ch not in string.punctuation)
    return ' '.join(word for word in text.split() if word not in ('a', 'an', 'the'))


def same(left, right):
    left, right = normal(left), normal(right)
    return bool(left) and bool(right) and (left == right or left in right or right in left)


def load(tag):
    path = ROOT / 'runs/chain' / SPLIT / DATASET / tag
    scored = {row['id']: row for row in json.loads((path / 'scored.json').read_text())}
    records = {}
    for line in (path / 'records.jsonl').open():
        record = json.loads(line)
        if 'error' not in record and record['id'] in scored:
            records[record['id']] = record
    return scored, records


def bound(record):
    return [step['candidates'][step['chosen']]['answer'] for step in record.get('steps', [])
            if step.get('candidates') and step.get('chosen') is not None]


def retrieved(record):
    found = set()
    for step in record.get('steps', []):
        found.update(step.get('retrieved', []))
    return found


scored, records = load(TAG)
others = {tag: load(tag)[1] for tag in OTHERS}
table = collections.defaultdict(collections.Counter)
detail = collections.defaultdict(collections.Counter)
for key, record in records.items():
    row = rows.get(str(key))
    if row is None:
        continue
    kind = str(key).split('__')[0]
    gold = row['question_decomposition']
    table[kind]['n'] += 1
    table[kind]['f1'] += scored[key]['f1']
    if scored[key]['f1'] >= 0.5:
        table[kind]['right'] += 1
        continue
    answers, seen = bound(record), retrieved(record)
    table[kind]['plan hops differ'] += len(record.get('hops') or []) != len(gold)
    root = None
    for index, hop in enumerate(gold):
        if index == len(gold) - 1 or not any(same(hop['answer'], answer) for answer in answers):
            root = index
            break
    hop = gold[root]
    paragraph = row['paragraphs'][hop['paragraph_support_idx']]
    passage = position.get((paragraph['title'], paragraph['paragraph_text']))
    if passage is None:
        table[kind]['passage not in corpus'] += 1
        continue
    if passage in seen:
        table[kind]['passage not bound'] += 1
        detail['passage not bound']['at the last hop' if root == len(gold) - 1 else 'at an earlier hop'] += 1
        continue
    table[kind]['retrieval miss'] += 1
    references = [int(ref) - 1 for ref in re.findall(r'#(\d+)', hop['question'])]
    entities = [gold[ref]['answer'] for ref in references if ref < len(gold)]
    text = normal(paragraph['title'] + ' ' + paragraph['paragraph_text'])
    detail['retrieval miss']['hop without dependency' if not entities else 'hop with %d dependencies' % len(entities)] += 1
    if entities:
        mentioned = [normal(entity) in text for entity in entities]
        detail['retrieval miss']['passage mentions every bound entity' if all(mentioned) else
                                 'passage mentions some bound entity' if any(mentioned) else 'passage mentions no bound entity'] += 1
        detail['retrieval miss']['title is a bound entity'] += any(same(entity, paragraph['title']) for entity in entities)
    for tag, other in others.items():
        if key in other:
            detail['retrieval miss']['retrieved by ' + tag] += passage in retrieved(other[key])

columns = ['n', 'right', 'retrieval miss', 'passage not bound', 'plan hops differ', 'passage not in corpus']
total = collections.Counter()
print('== %s %s %s: root of each wrong answer (percent of all questions of the kind)' % (DATASET, SPLIT, TAG))
for kind in sorted(table):
    counts = table[kind]
    total.update(counts)
    print('  %-6s n %4d  F1 %5.1f  right %5.1f%%  retrieval miss %5.1f%%  passage not bound %5.1f%%  (plan hop count differs in %4.1f%%)' % (
        kind, counts['n'], 100 * counts['f1'] / counts['n'], 100 * counts['right'] / counts['n'],
        100 * counts['retrieval miss'] / counts['n'], 100 * counts['passage not bound'] / counts['n'],
        100 * counts['plan hops differ'] / counts['n']))
print('  %-6s n %4d  F1 %5.1f  right %5.1f%%  retrieval miss %5.1f%%  passage not bound %5.1f%%  (plan hop count differs in %4.1f%%)' % (
    'all', total['n'], 100 * total['f1'] / total['n'], 100 * total['right'] / total['n'], 100 * total['retrieval miss'] / total['n'],
    100 * total['passage not bound'] / total['n'], 100 * total['plan hops differ'] / total['n']))
for name, counts in detail.items():
    print('  %s:' % name, dict(counts))
