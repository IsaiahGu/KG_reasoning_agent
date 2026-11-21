#!/usr/bin/env python
"""IRCoT with our reader and embedder on a HippoRAG 2 test subset (same-reader baseline for the paper).

The loop is the development-set IRCoT baseline unchanged (src/supportchain/baseline_ircot.py: upstream IRCoT prompts and
reasoning helper, at most four retrieval rounds of five passages, upstream reader prompt over the five best passages);
only the retriever is pointed at the test corpus. Needs the LLM and the embedding service. Writes
runs/chain/test/<dataset>/b3_ircot/{records.jsonl,per_question.json,scored.json,summary.json}; scored.json has the
format of run_chain's, so that the baseline can be paired with our runs question by question.
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supportchain.baseline_dense import READER_GENERATION, RetrievedDocument, score_answers  # noqa: E402
from supportchain.baseline_ircot import predict_ircot  # noqa: E402
from supportchain.chain_agent import SplitRetriever  # noqa: E402
from supportchain.hipporag_upstream import score_normalizer  # noqa: E402
from supportchain.llm import client_for_run  # noqa: E402
from supportchain.textmatch import clean_answer  # noqa: E402


class TestRetriever:
    """The interface of baseline_dense.DenseRetriever over a test corpus: dense scores, min-max normalised as upstream."""

    def __init__(self, root, dataset):
        self.base = SplitRetriever(root, dataset, 'test')
        self.examples, self.documents = self.base.examples, self.base.documents
        self.normalizer = score_normalizer(root)

    def search(self, query, k=5, *, question_id=None):
        scores = self.normalizer(self.base.vectors @ np.asarray(self.base.vector(query), dtype=np.float32))
        return [RetrievedDocument(int(index), float(scores[index]), self.documents[index]) for index in np.argsort(scores)[::-1][:k]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', required=True, choices=['musique', '2wikimultihopqa', 'hotpotqa'])
    parser.add_argument('--tag', default='b3_ircot')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--max-steps', type=int, default=4)
    parser.add_argument('--base-url', default='http://127.0.0.1:18080/v1')
    arguments = parser.parse_args()
    root = arguments.root
    output = root / 'runs/chain/test' / arguments.dataset / arguments.tag
    output.mkdir(parents=True, exist_ok=True)
    retriever = TestRetriever(root, arguments.dataset)
    examples = retriever.examples[:arguments.limit]
    client = client_for_run(root, output, base_url=arguments.base_url, generation=dict(READER_GENERATION))
    lock = threading.Lock()
    (output / 'records.jsonl').write_text('')

    def one(example):
        try:
            record = {'id': example.id, 'question': example.question, 'status': 'success',
                      **predict_ircot(root, example, client, retriever, max_steps=arguments.max_steps)}
        except Exception as error:  # recorded, not hidden; a rerun answers it again from the LLM cache
            record = {'id': example.id, 'question': example.question, 'status': 'error', 'answer': '', 'error': repr(error)}
        with lock, (output / 'records.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'id': record['id'], 'status': record['status'], 'answer': record.get('answer', '')},
                                    ensure_ascii=False) + '\n')
        return record

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        records = list(pool.map(one, examples))
    _, upstream = score_answers(root, examples, [record.get('answer') or '' for record in records])
    _, first_line = score_answers(root, examples, [clean_answer(record.get('response') or record.get('answer') or '') for record in records])
    scored = []
    for example, record, a, b in zip(examples, records, upstream, first_line):
        usage = [event.get('usage', {}) for event in record.get('trace', [])] + [record.get('usage', {})]
        record['calls'] = len(usage) if record['status'] == 'success' else 0
        record['tokens'] = sum(item.get('prompt_tokens', 0) + item.get('completion_tokens', 0) for item in usage)
        record.update(f1=a['f1'], em=a['em'])
        scored.append({'id': example.id, 'stratum': example.stratum, 'f1': a['f1'], 'em': a['em'], 'clean_f1': b['f1'],
                       'clean_em': b['em'], 'complete': False, 'calls': record['calls'], 'tokens': record['tokens']})
    errors = sum(record['status'] == 'error' for record in records)
    summary = {'system': 'IRCoT, same reader and embedder', 'dataset': arguments.dataset, 'split': 'test',
               'requested': len(examples), 'scored': len(scored), 'errors': errors, 'max_retrieval_steps': arguments.max_steps,
               'f1': round(100 * float(np.mean([row['f1'] for row in scored])), 2),
               'em': round(100 * float(np.mean([row['em'] for row in scored])), 2),
               'recall_at_5': round(100 * float(np.mean([record.get('recall_at_5', 0) for record in records])), 1),
               'all_supporting_at_5': round(100 * float(np.mean([bool(record.get('all_supporting_at_5')) for record in records])), 1),
               'llm_calls_per_question': round(float(np.mean([row['calls'] for row in scored])), 2),
               'tokens_per_question': round(float(np.mean([row['tokens'] for row in scored])))}
    (output / 'per_question.json').write_text(json.dumps(records, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    if not errors and not arguments.limit:  # written only for a complete, clean run
        (output / 'scored.json').write_text(json.dumps(scored) + '\n')
    print(json.dumps(summary))
    if errors:
        raise SystemExit('%d questions failed; run again to retry them' % errors)


if __name__ == '__main__':
    main()
