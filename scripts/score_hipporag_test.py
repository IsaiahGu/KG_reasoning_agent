#!/usr/bin/env python
"""Score the same-reader HippoRAG 2 baseline on a test subset in run_chain's scored.json format. CPU only.

Reads runs/b2_test/<dataset>/predictions.json (written by `python -m hyperbind.baseline_hipporag --split test --phase run`)
and the phase reports next to it; nothing is rerun. The saved answers are scored with the same upstream metric as every
other run. Writes runs/chain/test/<dataset>/b2_hipporag2/{scored.json,summary.json}; the summary also carries the LLM
cost of indexing (OpenIE) and of answering, for the cost table.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind.baseline_dense import score_answers  # noqa: E402
from hyperbind.data import load_test_samples  # noqa: E402
from hyperbind.textmatch import clean_answer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--tag', default='b2_hipporag2')
    arguments = parser.parse_args()
    source = ROOT / 'runs/b2_test' / arguments.dataset
    predictions = json.loads((source / 'predictions.json').read_text())
    examples = load_test_samples(ROOT, arguments.dataset)
    if all(row.get('id') is None for row in predictions):
        # NQ questions carry no id field, so the baseline writes None; its predictions follow the question file's order,
        # which is also the order of load_test_samples (it gives such rows their position as id).
        if len(predictions) != len(examples):
            raise SystemExit('%d predictions for %d questions' % (len(predictions), len(examples)))
        predictions = [dict(row, id=example.id) for row, example in zip(predictions, examples)]
    rows = {str(row['id']): row for row in predictions}
    missing = [example.id for example in examples if example.id not in rows]
    if missing:
        raise SystemExit('%d of %d questions have no prediction' % (len(missing), len(examples)))
    answers = [rows[example.id].get('answer') or '' for example in examples]
    cleaned = [clean_answer(rows[example.id].get('response') or rows[example.id].get('answer') or '') for example in examples]
    _, upstream = score_answers(ROOT, examples, answers)
    _, first_line = score_answers(ROOT, examples, cleaned)
    folder = ROOT / 'runs/chain/test' / arguments.dataset / arguments.tag
    folder.mkdir(parents=True, exist_ok=True)
    recall = [len(set(example.gold_documents) & set(rows[example.id].get('documents') or [])) / len(example.gold_documents)
              for example in examples if example.gold_documents]
    summary = {'system': 'HippoRAG 2 (official code), same reader and embedder (baseline B2)', 'dataset': arguments.dataset,
               'split': 'test', 'requested': len(examples), 'scored': len(examples), 'errors': 0,
               'f1': round(100 * float(np.mean([row['f1'] for row in upstream])), 2),
               'em': round(100 * float(np.mean([row['em'] for row in upstream])), 2),
               'recall_at_5': round(100 * float(np.mean(recall)), 1) if recall else None}
    for phase in ('openie', 'encode-index', 'build', 'run'):
        path = source / (phase + '_report.json')
        if path.exists():
            report = json.loads(path.read_text())
            summary[phase] = {'seconds': round(report['seconds']),
                              'extraction_calls': report.get('extraction_calls_this_phase', {}),
                              'reader_calls': report.get('reader_calls_this_phase', {}),
                              **{name: report[name] for name in ('documents', 'entities', 'facts', 'graph_vertices', 'graph_edges',
                                                                 'questions', 'graph_seeded_queries') if name in report}}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (folder / 'scored.json').write_text(json.dumps([
        {'id': example.id, 'stratum': example.stratum, 'f1': a['f1'], 'em': a['em'], 'clean_f1': b['f1'], 'clean_em': b['em'],
         'complete': False} for example, a, b in zip(examples, upstream, first_line)]) + '\n')
    print(json.dumps({name: summary[name] for name in ('dataset', 'f1', 'em', 'recall_at_5')}))


if __name__ == '__main__':
    main()
