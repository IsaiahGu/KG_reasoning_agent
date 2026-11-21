#!/usr/bin/env python
"""What does a binding error look like? MuSiQue development splits only. CPU only.

Usage: binding_errors.py <split> <tag> [examples]
Takes the wrong answers whose root hop had its supporting passage retrieved (see root_cause.py) and says, for the plan
hop at the same position as the root gold hop: was the supporting passage among that hop's passages and at which rank,
did the hop bind an answer from the supporting passage or from another one, and does the passage it used mention the
entity bound before. Nothing is fitted or tuned here.
"""
import collections
import json
import random
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLIT, TAG = sys.argv[1], sys.argv[2]
SHOW = int(sys.argv[3]) if len(sys.argv) > 3 else 0
folder = ROOT / 'data/splits/seed42/musique'
rows = {str(row['id']): row for row in json.loads((folder / (SPLIT + '.json')).read_text())}
corpus = json.loads((folder / ('train-S_corpus.json' if SPLIT == 'train-S' else 'dev_corpus.json')).read_text())
position = {(doc['title'], doc['text']): index for index, doc in enumerate(corpus)}


def normal(text):
    text = ''.join(ch for ch in str(text).lower() if ch not in string.punctuation)
    return ' '.join(word for word in text.split() if word not in ('a', 'an', 'the'))


def same(left, right):
    left, right = normal(left), normal(right)
    return bool(left) and bool(right) and (left == right or left in right or right in left)


path = ROOT / 'runs/chain' / SPLIT / 'musique' / TAG
scored = {row['id']: row for row in json.loads((path / 'scored.json').read_text())}
counts, examples = collections.Counter(), []
for line in (path / 'records.jsonl').open():
    record = json.loads(line)
    key = record.get('id')
    if 'error' in record or key not in scored or scored[key]['f1'] >= 0.5 or str(key) not in rows:
        continue
    row = rows[str(key)]
    gold, steps = row['question_decomposition'], record.get('steps', [])
    answers = [step['candidates'][step['chosen']]['answer'] if step.get('candidates') and step.get('chosen') is not None else None
               for step in steps]
    root = next((index for index, hop in enumerate(gold)
                 if index == len(gold) - 1 or not any(same(hop['answer'], answer) for answer in answers if answer)), None)
    paragraph = row['paragraphs'][gold[root]['paragraph_support_idx']]
    passage = position.get((paragraph['title'], paragraph['paragraph_text']))
    seen = set(index for step in steps for index in step.get('retrieved', []))
    if passage is None or passage not in seen:
        continue
    counts['binding errors'] += 1
    where = 'last hop' if root == len(gold) - 1 else 'earlier hop'
    counts[where] += 1
    if len(record.get('hops') or []) != len(gold):
        counts[where + ': plan has another number of hops'] += 1
        kind = 'plan has another number of hops'
    elif root >= len(steps):
        counts[where + ': chain stopped before this hop'] += 1
        kind = 'chain stopped before this hop'
    else:
        step = steps[root]
        ranked = step.get('retrieved', [])
        if passage not in ranked:
            kind = 'supporting passage retrieved at another hop only'
        elif not step.get('candidates') or step.get('chosen') is None:
            kind = 'no answer bound, supporting passage at rank %s' % ('1-3' if ranked.index(passage) < 3 else '4-8')
        else:
            chosen = step['candidates'][step['chosen']]
            if chosen.get('passage_index') == passage:
                kind = 'answer taken from the supporting passage, but another answer'
            else:
                used = corpus[chosen['passage_index']] if isinstance(chosen.get('passage_index'), int) else None
                references = [int(ref) - 1 for ref in re.findall(r'#(\d+)', gold[root]['question'])]
                entities = [gold[ref]['answer'] for ref in references if ref < len(gold)]
                text = normal(used['title'] + ' ' + used['text']) if used else ''
                linked = bool(entities) and all(normal(entity) in text for entity in entities)
                kind = 'answer taken from another passage (supporting passage at rank %s)%s' % (
                    '1-3' if ranked.index(passage) < 3 else '4-8',
                    '' if not entities else ', which mentions the bound entity' if linked else ', which does not mention the bound entity')
        counts[where + ': ' + kind] += 1
    examples.append((where, kind, record, row, root, answers))
print('== musique %s %s' % (SPLIT, TAG))
for name, value in sorted(counts.items()):
    print('  %4d  %s' % (value, name))
random.Random(0).shuffle(examples)
for where, kind, record, row, root, answers in examples[:SHOW]:
    print('---', where, '|', kind)
    print('Q:', record['question'], '| gold:', row['answer'], '| reader:', (record.get('reader_clean') or '')[:100])
    print('gold hops:', [(hop['question'], hop['answer']) for hop in row['question_decomposition']])
    print('plan:', record.get('hops'))
    print('bound:', answers, '| root gold hop', root + 1)
