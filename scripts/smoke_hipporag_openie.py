#!/usr/bin/env python
"""Gate before the HippoRAG 2 baseline on a test corpus: run its OpenIE on a few passages and require clean parses.

The baseline's JSON-object requests depend on the model server's structured-output setting (see scripts/with_services.py).
This runs the unchanged extraction on the first passages of a test corpus in a scratch folder and exits non-zero unless
every response parsed, so that a wrong server setting costs minutes rather than the hours of a full extraction.
Usage (interpreter: envs/embed/bin/python, with the LLM answering): smoke_hipporag_openie.py [--dataset musique] [--passages 60]
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind import baseline_hipporag as b2  # noqa: E402
from hyperbind.hipporag_preflight import summarize_audit  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='musique')
    parser.add_argument('--passages', type=int, default=60)
    parser.add_argument('--base-url', default='http://127.0.0.1:18080/v1')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/b2_test/openie_smoke.json')
    arguments = parser.parse_args()
    folder = Path(tempfile.mkdtemp(prefix='b2_smoke_'))
    engine, llms = None, ()
    try:
        documents, _, _ = b2.data(ROOT, arguments.dataset, 'test')
        model = b2.resolve_identity(folder, arguments.base_url, None)
        engine, _, llms = b2.construct(ROOT, arguments.dataset, folder, model, arguments.base_url, endpoint=None, workers=16, split='test')
        ended = None
        try:
            engine.pre_openie(documents[:arguments.passages])
        except RuntimeError as error:  # upstream ends a finished offline extraction with a RuntimeError as well
            ended = str(error)
        state = Path(engine.openie_state_path)
        rows = json.loads(state.read_text()).get('docs', []) if state.exists() else []
        audit = summarize_audit(folder / 'openie_audit.jsonl')
        calls = [json.loads(line) for line in (folder / 'calls/extraction/llm_calls.jsonl').read_text().splitlines() if line]
        report = {'dataset': arguments.dataset, 'passages': arguments.passages, 'extracted_documents': len(rows),
                  'entities_per_passage': round(sum(len(row.get('extracted_entities', [])) for row in rows) / max(1, len(rows)), 1),
                  'triples_per_passage': round(sum(len(row.get('extracted_triples', [])) for row in rows) / max(1, len(rows)), 1),
                  'parse_errors': audit.get('parse_errors'), 'operational_errors': audit.get('operational_errors'),
                  'first_key_repaired': sum(bool(call.get('first_key_repaired')) for call in calls), 'calls': len(calls), 'ended': ended}
        report['passed'] = (len(rows) == arguments.passages and not audit.get('parse_errors') and not audit.get('operational_errors')
                            and report['triples_per_passage'] > 0)
    finally:
        if engine is not None:
            engine.close()
        for llm in llms:
            llm.close()
        shutil.rmtree(folder, ignore_errors=True)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit('HippoRAG 2 extraction is not clean under this server setting')
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
