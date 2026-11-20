#!/usr/bin/env python
"""Recompute candidate features of saved chain records with the current feature code. Embedding service only, no LLM.

Lets the binding scorer's features change without rerunning the hop reads: everything the features need is in the
record (sub-question, candidates, plan, chosen answers) or comes from retrieval.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind.binding_scorer import candidate_features  # noqa: E402
from hyperbind.chain_agent import SplitRetriever  # noqa: E402


def refeaturize(record, retriever):
    hops, answers = record['hops'], {}
    question_vector = retriever.vector(record['question'])
    for step in record['steps']:
        number, candidates = step['hop'], step['candidates']
        following = None
        if len(candidates) > 1:
            for later in range(number + 1, len(hops) + 1):
                refs = [int(ref) for ref in re.findall(r'#(\d+)', hops[later - 1])]
                if number in refs and all(ref == number or ref in answers for ref in refs):
                    following = hops[later - 1]
                    break
        bound = [answers[int(ref)] for ref in re.findall(r'#(\d+)', hops[number - 1]) if int(ref) in answers]
        window = [retriever.documents[index] for index in step['retrieved']]
        for position, candidate in enumerate(candidates):
            ahead = None
            if following is not None:
                filled = re.sub(r'#(\d+)', lambda match: candidate['answer'] if int(match.group(1)) == number
                                else answers[int(match.group(1))], following)
                hits, _ = retriever.search(filled, 3)
                ahead = {'scores': [hit.score for hit in hits], 'titles': [hit.title for hit in hits],
                         'contents': [hit.content for hit in hits]}
            index, old = candidate['passage_index'], candidate['features']
            candidate['features'] = candidate_features(
                sub_question=step['question'], candidate=candidate, position=position, count=len(candidates),
                title=retriever.titles[index], content=retriever.documents[index],
                top_score=candidate['passage_score'] + old['score_gap'], window_contents=window, bound=bound,
                hop=number, hops=len(hops), question_score=float(retriever.vectors[index] @ question_vector), ahead=ahead)
        if step['chosen'] is None:
            break
        answers[number] = candidates[step['chosen']]['answer']
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', default='musique')
    parser.add_argument('--split', default='train-S')
    parser.add_argument('--tags', nargs='+', required=True)
    parser.add_argument('--out', default='records_refeaturized.jsonl')
    arguments = parser.parse_args()
    retriever = SplitRetriever(arguments.root, arguments.dataset, arguments.split)
    for tag in arguments.tags:
        folder = arguments.root / 'runs/chain' / arguments.split / arguments.dataset / tag
        rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text().splitlines()]
        with (folder / arguments.out).open('w', encoding='utf-8') as stream:
            for count, record in enumerate(rows, 1):
                if 'error' not in record:
                    stream.write(json.dumps(refeaturize(record, retriever), ensure_ascii=False) + '\n')
                if count % 200 == 0:
                    print(json.dumps({'tag': tag, 'done': count, 'of': len(rows)}), flush=True)
        print(json.dumps({'tag': tag, 'written': str(folder / arguments.out)}), flush=True)


if __name__ == '__main__':
    main()
