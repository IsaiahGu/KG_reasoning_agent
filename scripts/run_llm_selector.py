#!/usr/bin/env python
"""Control for the merge rule: the reader itself picks among the answers of the final method's derivations.

Nothing is rerun. The derivations of the final method (configs/final.json) are read from their finished runs. On a
question where they give more than one distinct answer, the reader sees the question and, for every distinct answer, the
attempts that reached it (a chain: each hop with its answer and the sentence it quoted; the dense read: a note that it
read the top passages in one go) and returns the number of the answer it judges best supported. Questions on which all
derivations agree need no call. No gold field is read while choosing; the scores are those of the chosen derivation.

Writes runs/chain/<split>/<dataset>/<tag>/{records.jsonl,scored.json,summary.json,llm_calls.jsonl}.
"""
import argparse
import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind.llm import StructuredOutputError, client_for_run  # noqa: E402
from hyperbind.support import answer_groups  # noqa: E402

SYSTEM = (
    'Several attempts answered the same question and they disagree. Each candidate answer is listed with the attempts that '
    'reached it. An attempt either worked step by step (each step shows its sub-question, its answer and the sentence it '
    'quoted from a Wikipedia passage) or read the top retrieved passages in one go (no steps to show). '
    'Decide which candidate answer is best supported: the steps must follow the question, every step must be backed by its '
    'quote, and a step marked "likely", a missing quote or an attempt marked "incomplete" is weaker evidence. '
    'Return JSON with two fields in this order. reason: one or two sentences. choice: the number of the best candidate.')


def schema(count):
    return {'type': 'object', 'properties': {'reason': {'type': 'string'},
                                             'choice': {'type': 'integer', 'enum': list(range(1, count + 1))}},
            'required': ['reason', 'choice'], 'additionalProperties': False}


def load(split, dataset, tag):
    folder = ROOT / 'runs/chain' / split / dataset / tag
    scored = {row['id']: row for row in json.loads((folder / 'scored.json').read_text())}
    records = {}
    for line in (folder / 'records.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        if 'error' not in record:
            records[record['id']] = record
    return scored, records


def describe(record):
    """What one derivation did, from its record. No gold field is read."""
    if not record.get('steps'):
        return 'read the top passages retrieved for the whole question in one go (no steps)'
    lines = ['worked step by step' + ('' if record.get('complete') else ' (incomplete: the plan could not be finished)')]
    for step in record['steps']:
        if step.get('chosen') is None or not step.get('candidates'):
            lines.append('  Step %d: %s -> no answer found' % (step['hop'], step['question']))
            continue
        candidate = step['candidates'][step['chosen']]
        quote = ' '.join((candidate.get('quote') or '').split())[:400]
        lines.append('  Step %d: %s -> %s%s | quote: %s' % (
            step['hop'], step['question'], candidate['answer'],
            '' if candidate.get('support') == 'explicit' else ' (likely, not stated directly)', '"%s"' % quote if quote else 'none'))
    return '\n'.join(lines)


def listing(question, records, answers, groups, order):
    parts = ['Question: %s' % question, '', 'Candidate answers:']
    for number, position in enumerate(order, 1):
        group = groups[position]
        shown = ' '.join(answers[group[0]].split())  # an answer without the reader's answer marker can be a long thought
        parts += ['', '[%d] %s' % (number, shown if len(shown) <= 200 else shown[:200] + ' ...'),
                  'Reached by %d of %d attempts.' % (len(group), len(answers))]
        parts += ['Attempt %s: %s' % (chr(65 + rank), describe(records[index])) for rank, index in enumerate(group)]
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split', default='test', choices=['dev-A', 'dev-B', 'test'])
    parser.add_argument('--tag', default='llm_select')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int, help='only the first N questions (smoke test)')
    parser.add_argument('--seed', type=int, default=42, help='sampling seed sent with every call; part of the LLM cache key')
    parser.add_argument('--base-url', default='http://127.0.0.1:18080/v1')
    parser.add_argument('--dry-run', action='store_true', help='print the first prompts and the number of calls needed; no LLM call')
    arguments = parser.parse_args()
    config = json.loads((ROOT / 'configs/final.json').read_text())
    final = config['final']
    runs = {name: load(arguments.split, arguments.dataset, config['derivations'][name]) for name in final}
    keys = sorted(set.intersection(*[set(scored) & set(records) for scored, records in runs.values()]))
    keys = keys[:arguments.limit] if arguments.limit else keys
    fallback = 'fixed' if 'fixed' in final else final[0]

    def case(key):
        records = [runs[name][1][key] for name in final]
        answers = [record.get('reader_clean') or '' for record in records]
        return records, answers, answer_groups(answers)

    if arguments.dry_run:
        needed = [key for key in keys if len(case(key)[2]) > 1]
        print('%d questions, %d need a call' % (len(keys), len(needed)))
        for key in needed[:3]:
            records, answers, groups = case(key)
            print('-' * 100 + '\n' + listing(records[0]['question'], records, answers, groups, list(range(len(groups)))))
        return
    output = ROOT / 'runs/chain' / arguments.split / arguments.dataset / arguments.tag
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'records.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    rows = [row for row in rows if 'error' not in row]
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    done = {row['id'] for row in rows}
    todo = [key for key in keys if key not in done]
    client = client_for_run(ROOT, output, base_url=arguments.base_url,
                            generation={'temperature': 0, 'top_p': 1.0, 'seed': arguments.seed, 'max_tokens': 2048})
    lock, started, finished = threading.Lock(), time.time(), [0]

    def one(key):
        try:
            records, answers, groups = case(key)
            row = {'id': key, 'answers': dict(zip(final, answers)), 'groups': [[final[index] for index in group] for group in groups],
                   'order': None, 'choice': None, 'source': final[groups[0][0]] if groups else fallback, 'reason': None,
                   'usage': {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'structured_failures': 0}}
            if len(groups) > 1:
                order = list(range(len(groups)))  # a fixed pseudo-random order per question, so that no derivation is always first
                random.Random(int(hashlib.sha256(str(key).encode('utf-8')).hexdigest()[:8], 16)).shuffle(order)
                row['order'], row['source'] = order, fallback
                messages = [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': listing(records[0]['question'], records, answers, groups, order)}]
                try:
                    result = client.complete(messages, schema(len(groups)), thinking=False, generation_overrides={'max_tokens': 400},
                                             metadata={'phase': 'llm_select', 'stage': 'SELECT', 'question_id': key})
                    row['usage'].update(calls=1, prompt_tokens=int(result.usage.get('prompt_tokens', 0) or 0),
                                        completion_tokens=int(result.usage.get('completion_tokens', 0) or 0))
                    position = order[int(result.parsed['choice']) - 1]
                    row.update(choice=position, source=final[groups[position][0]], reason=result.parsed.get('reason'))
                except StructuredOutputError:  # counted; the question keeps chain A1's answer
                    row['usage']['structured_failures'] = 1
        except Exception as error:  # recorded, not hidden; a rerun resumes the failed questions
            row = {'id': key, 'error': repr(error)}
        with lock:
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows.append(row)
            finished[0] += 1
            if finished[0] % 100 == 0 or finished[0] == len(todo):
                print(json.dumps({'done': finished[0], 'of': len(todo), 'seconds': round(time.time() - started),
                                  'errors': sum('error' in item for item in rows)}), flush=True)

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        list(pool.map(one, todo))
    good = {row['id']: row for row in rows if 'error' not in row}
    scored = [dict(runs[good[key]['source']][0][key], complete=False) for key in keys if key in good]
    asked = [row for row in good.values() if row['order'] is not None]
    summary = {'system': 'control: the reader picks among the distinct answers of ' + ', '.join(final), 'dataset': arguments.dataset,
               'split': arguments.split, 'requested': len(keys), 'scored': len(scored), 'errors': len(rows) - len(good),
               'f1': round(100 * float(np.mean([row['f1'] for row in scored])), 2) if scored else None,
               'em': round(100 * float(np.mean([row['em'] for row in scored])), 2) if scored else None,
               'questions_with_a_call': len(asked), 'structured_failures': sum(row['usage']['structured_failures'] for row in asked),
               'selector_calls_per_question': round(sum(row['usage']['calls'] for row in asked) / max(1, len(keys)), 3),
               'selector_tokens_per_question': round(sum(row['usage']['prompt_tokens'] + row['usage']['completion_tokens']
                                                         for row in asked) / max(1, len(keys)), 1),
               'chosen_from': {name: sum(row['source'] == name for row in good.values()) for name in final}}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if summary['errors'] or summary['scored'] != summary['requested']:
        sys.exit(3)
    (output / 'scored.json').write_text(json.dumps(scored) + '\n')


if __name__ == '__main__':
    main()
