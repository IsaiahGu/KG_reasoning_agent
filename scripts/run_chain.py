#!/usr/bin/env python
"""Run the quote-bound chain (or the dense read) on one split. Writes runs/chain/<split>/<dataset>/<tag>/.

Examples:
  run_chain.py --dataset musique --split train-S --limit 30 --tag smoke
  run_chain.py --dataset musique --split train-S --tag cand3
  run_chain.py --dataset musique --split train-S --offset 1500 --mode dense --tag a0
  run_chain.py --dataset musique --split train-S --offset 1500 --selector scorer --scorer runs/.../binding_scorer.json --tag cand3_scorer
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supportchain.baseline_dense import score_answers  # noqa: E402
from supportchain.binding_scorer import BindingScorer  # noqa: E402
from supportchain.chain_agent import ChainAgent, ChainConfig, PlanPool, SplitRetriever, musique_labeler  # noqa: E402
from supportchain.llm import client_for_run  # noqa: E402


def interval(values, seed=0, rounds=10000):
    values = np.asarray(values, float)
    generator = np.random.default_rng(seed)
    means = values[generator.integers(0, len(values), (rounds, len(values)))].mean(axis=1)
    return [round(100 * float(values.mean()), 2), round(100 * float(np.percentile(means, 2.5)), 2),
            round(100 * float(np.percentile(means, 97.5)), 2)]


def summarize(root, examples, records, arguments):
    by_id = {record['id']: record for record in records}
    chosen = [example for example in examples if example.id in by_id and 'error' not in by_id[example.id]]
    rows = [by_id[example.id] for example in chosen]
    summary = {'dataset': arguments.dataset, 'split': arguments.split, 'tag': arguments.tag, 'mode': arguments.mode,
               'offset': arguments.offset, 'requested': len(examples), 'scored': len(rows),
               'errors': sum('error' in record for record in records),
               'config': {'candidates': arguments.candidates, 'selector': arguments.selector, 'backtrack': arguments.backtrack,
                          'lookahead': not arguments.no_lookahead, 'k_hop': arguments.k_hop, 'budget': arguments.budget,
                          'reader_context': arguments.reader_context, 'plan': arguments.plan,
                          'retrieval': 'dense', 'plan_dataset': arguments.plan_dataset,
                          'plan_pool_limit': arguments.plan_pool_limit, 'plan_skip': arguments.plan_skip,
                          'plan_cases': str(arguments.plan_cases) if arguments.plan_cases else None, 'plan_loo': arguments.plan_loo,
                          'temperature': arguments.temperature or 0, 'seed': 42 if arguments.seed is None else arguments.seed,
                          'scorer': str(arguments.scorer) if arguments.scorer else None}}
    if not rows:
        return summary
    for name in (() if arguments.no_reader else ('reader_answer', 'reader_clean')):
        _, per_question = score_answers(root, chosen, [row[name] for row in rows])
        for row, score in zip(rows, per_question):
            row[name + '_f1'], row[name + '_em'] = score['f1'], score['em']
        label = 'upstream_extraction' if name == 'reader_answer' else 'first_line_extraction'
        summary[label] = {'f1_ci95': interval([score['f1'] for score in per_question]),
                          'em': round(100 * float(np.mean([score['em'] for score in per_question])), 2)}
    summary.update(
        plan_ok_percent=round(100 * float(np.mean([row['plan_ok'] for row in rows])), 1),
        chain_complete_percent=round(100 * float(np.mean([row['complete'] for row in rows])), 1),
        backtracked_percent=round(100 * float(np.mean([row.get('backtracked', False) for row in rows])), 1),
        evidence_all_supporting_percent=round(100 * float(np.mean([row['evidence_all_supporting'] for row in rows])), 1),
        evidence_recall_percent=round(100 * float(np.mean([row['evidence_recall'] for row in rows])), 1),
        reader_marker_missing=sum(row.get('reader_marker') is False for row in rows),
        llm_calls_per_question=round(float(np.mean([row['usage']['calls'] for row in rows])), 2),
        prompt_tokens_per_question=round(float(np.mean([row['usage']['prompt_tokens'] for row in rows]))),
        completion_tokens_per_question=round(float(np.mean([row['usage']['completion_tokens'] for row in rows]))),
        structured_failures=sum(row['usage'].get('structured_failures', 0) for row in rows))
    for stratum in sorted({row['stratum'] for row in rows}):
        part = [row for row in rows if row['stratum'] == stratum]
        summary['stratum_' + str(stratum)] = {
            'n': len(part), 'complete_percent': round(100 * float(np.mean([row['complete'] for row in part])), 1)}
        if not arguments.no_reader:
            summary['stratum_' + str(stratum)]['f1'] = round(100 * float(np.mean([row['reader_answer_f1'] for row in part])), 2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', default='musique')
    parser.add_argument('--split', default='train-S', choices=['train-S', 'dev-A', 'dev-B', 'test'],
                        help='test = the HippoRAG 2 1,000-question subset; no gold field is read while answering')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--mode', default='chain', choices=['chain', 'dense'])
    parser.add_argument('--candidates', type=int, default=3)
    parser.add_argument('--selector', default='first', choices=['first', 'scorer'])
    parser.add_argument('--scorer', type=Path)
    parser.add_argument('--backtrack', action='store_true')
    parser.add_argument('--no-lookahead', action='store_true')
    parser.add_argument('--no-reader', action='store_true', help='collect hop candidates only; no answer, no F1')
    parser.add_argument('--reader-context', default='both', choices=['both', 'passages', 'notes'])
    parser.add_argument('--plan', default='fixed', choices=['fixed', 'knn'],
                        help='knn: few-shot plan examples are the nearest train-S questions with their gold decompositions')
    parser.add_argument('--plan-pool-limit', type=int,
                        help='use only the first N train-S questions as plan examples (keep a held-out tail of train-S clean)')
    parser.add_argument('--plan-neighbours', type=int, default=6)
    parser.add_argument('--plan-skip', type=int, default=0, help='skip the nearest N plan examples (a second, different plan)')
    parser.add_argument('--plan-cases', type=Path,
                        help='plan memory built from the chain\'s own plans (scripts/build_plan_memory.py) instead of gold plans')
    parser.add_argument('--plan-dataset', help='take the plan memory from another dataset\'s train-S (transfer to a dataset without one)')
    parser.add_argument('--plan-loo', action='store_true',
                        help='train-S only: the pool may contain the evaluated questions; each question skips its own plan')
    parser.add_argument('--temperature', type=float,
                        help='sampling temperature of every LLM call (default: 0, greedy); with --seed this gives the '
                             'repeated samples of the self-consistency baseline')
    parser.add_argument('--seed', type=int, help='sampling seed sent with every LLM call (default: 42)')
    parser.add_argument('--k-hop', type=int, default=8)
    parser.add_argument('--budget', type=int, default=5)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:18080/v1')
    parser.add_argument('--summarize-only', action='store_true')
    arguments = parser.parse_args()
    root = arguments.root
    output = root / 'runs/chain' / arguments.split / arguments.dataset / arguments.tag
    output.mkdir(parents=True, exist_ok=True)
    retriever = SplitRetriever(root, arguments.dataset, arguments.split)
    end = None if arguments.limit is None else arguments.offset + arguments.limit
    examples = retriever.examples[arguments.offset:end]
    records_path = output / 'records.jsonl'
    records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    records = [record for record in records if 'error' not in record]
    done = {record['id'] for record in records}
    todo = [example for example in examples if example.id not in done]
    if todo and not arguments.summarize_only:
        records_path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records))
        # The defaults stay the integers 0 and 42: they are part of the LLM cache key of every earlier run.
        client = client_for_run(root, output, base_url=arguments.base_url,
                                generation={'temperature': 0 if arguments.temperature is None else arguments.temperature, 'top_p': 1.0,
                                            'seed': 42 if arguments.seed is None else arguments.seed, 'max_tokens': 2048})
        scorer = BindingScorer.load(arguments.scorer) if arguments.scorer else None
        config = ChainConfig(k_hop=arguments.k_hop, budget=arguments.budget, candidates=arguments.candidates,
                             selector=arguments.selector, lookahead=not arguments.no_lookahead, backtrack=arguments.backtrack,
                             final_reader=not arguments.no_reader, reader_context=arguments.reader_context,
                             plan_examples=arguments.plan, plan_neighbours=arguments.plan_neighbours,
                             plan_skip=arguments.plan_skip)
        # hop labels (from gold decompositions) are diagnostics for train and dev only; the test run carries none
        labeler = musique_labeler if arguments.dataset == 'musique' and arguments.split != 'test' else None
        pool = None
        if arguments.plan == 'knn':
            pool = PlanPool(root, arguments.plan_dataset or arguments.dataset, arguments.plan_pool_limit, cases=arguments.plan_cases)
            overlap = set(pool.ids) & {example.id for example in examples}
            if arguments.split == 'train-S' and overlap and not arguments.plan_loo:
                raise SystemExit('%d evaluated train-S questions are in the plan memory; use --plan-pool-limit, or --plan-loo '
                                 'to let each question skip its own plan' % len(overlap))
        agent = ChainAgent(root, retriever, client, config, scorer=scorer, labeler=labeler, plan_pool=pool)
        lock, started, finished = threading.Lock(), time.time(), [0]

        def one(example):
            try:
                record = agent.run(example) if arguments.mode == 'chain' else agent.run_dense(example)
            except Exception as error:  # recorded, not hidden; rerun resumes the failed questions
                record = {'id': example.id, 'stratum': example.stratum, 'question': example.question, 'error': repr(error)}
            with lock:
                with records_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                records.append(record)
                finished[0] += 1
                if finished[0] % 25 == 0 or finished[0] == len(todo):
                    print(json.dumps({'done': finished[0], 'of': len(todo), 'seconds': round(time.time() - started),
                                      'errors': sum('error' in row for row in records)}), flush=True)

        with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
            list(pool.map(one, todo))
    summary = summarize(root, examples, records, arguments)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    scored = {record['id']: record for record in records if 'reader_answer_f1' in record}
    (output / 'scored.json').write_text(json.dumps(
        [{'id': key, 'stratum': row['stratum'], 'f1': row['reader_answer_f1'], 'em': row['reader_answer_em'],
          'clean_f1': row['reader_clean_f1'], 'clean_em': row['reader_clean_em'], 'complete': row['complete']}
         for key, row in scored.items()]) + '\n')
    print(json.dumps(summary, indent=2))
    if summary.get('errors') or summary.get('scored') != summary.get('requested'):
        # Failed questions are kept out of the scores; a rerun resumes exactly those.
        sys.exit(3)


if __name__ == '__main__':
    main()
