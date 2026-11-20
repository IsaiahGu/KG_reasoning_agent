"""B2 orchestration around the pinned, unmodified official HippoRAG 2 class.

Stages deliberately separate LLM extraction, offline NV encoding, graph build,
and online retrieval/QA. No stage substitutes dense retrieval for HippoRAG.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from .hipporag_backend import (CachedNVEmbedding, FP16Cache, LoggedOfficialLLM, call_statistics,
                               phase_call_delta, require_clean_phase_calls, request)

DATASETS = ('musique', '2wikimultihopqa', 'hotpotqa')
# The test split also covers the two single-hop sets of the HippoRAG 2 protocol (added 2026-11-19; they have no dev-A split).
TEST_DATASETS = DATASETS + ('nq_rear', 'popqa')
B2_PROTOCOL = 'official_openie_json_object_v2'


def protocol_settings():
    return {'b2_protocol': B2_PROTOCOL, 'extraction_response_format': {'type': 'json_object'},
            'reader_response_format': None}


def extraction_config(settings):
    """Official JSON mode for OpenIE; the independent reader stays unchanged."""
    extraction_settings = deepcopy(settings)
    extraction_settings.response_format = {'type': 'json_object'}
    return extraction_settings


def local_llms(settings, folder):
    return (LoggedOfficialLLM(extraction_config(settings), folder / 'calls/extraction', 'openie_or_recognition'),
            LoggedOfficialLLM(settings, folder / 'calls/reader', 'reader'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


SPLITS = ('dev-A', 'test')


def data(root, dataset, split='dev-A'):
    """dev-A: the 500 development questions and the development corpus (P0.3). test: the HippoRAG 2 subset of 1,000
    questions and its passage pool; added on 2026-11-18 for the same-reader baseline on the test subsets, after the
    method's own test runs were complete. Nothing else in the pipeline differs between the two."""
    if split == 'test':
        corpus_path, samples_path = root / 'data/hipporag2' / (dataset + '_corpus.json'), root / 'data/hipporag2' / (dataset + '.json')
        expected = 1000
    elif split == 'dev-A':
        if dataset not in DATASETS:
            raise ValueError('dev-A exists only for ' + ', '.join(DATASETS))
        folder = root / 'data/splits/seed42' / dataset
        corpus_path, samples_path, expected = folder / 'dev_corpus.json', folder / 'dev-A.json', 500
    else:
        raise ValueError('Unknown split: ' + str(split))
    corpus, samples = json.loads(corpus_path.read_text()), json.loads(samples_path.read_text())
    if len(samples) != expected:
        raise ValueError('B2 runs exactly the %d questions of %s' % (expected, split))
    documents = [row['title'] + '\n' + row['text'] for row in corpus]
    return documents, samples, {'corpus': {'path': str(corpus_path.relative_to(root)), 'sha256': sha(corpus_path)},
                                'queries': {'path': str(samples_path.relative_to(root)), 'sha256': sha(samples_path)}}


def run_folder(root, dataset, split='dev-A'):
    return root / ('runs/b2_test' if split == 'test' else 'runs/p0_dev/b2') / dataset


def dense_passages(root, dataset, split, documents, sources):
    """The offline NV passage vectors of the split's corpus, in corpus order."""
    if split == 'test':
        folder = root / 'runs/p0/calibration' / dataset
        progress = json.loads((folder / 'passages.f16.progress.json').read_text())
        dense = np.load(folder / 'passages.f16.npy', mmap_mode='r')
        if progress.get('completed') != len(documents) or dense.shape != (len(documents), 4096) or dense.dtype != np.float16:
            raise ValueError('Test passage vectors do not match the test corpus')
        return dense
    folder = root / 'runs/p0_dev/embeddings' / dataset
    metadata = json.loads((folder / 'metadata.json').read_text())
    if metadata['status'] != 'complete' or metadata['sources'] != sources:
        raise ValueError('Dense development cache does not match current corpus and questions')
    return np.load(folder / 'passages.f16.npy', mmap_mode='r')


def preparation_audit(root):
    counts = {dataset: len(data(root, dataset)[0]) for dataset in DATASETS}
    upstream = root / 'external/HippoRAG'
    report = {'baseline': 'B2 official HippoRAG 2', **protocol_settings(), 'scope': 'dev-A only', 'passages': counts,
              'official_commit': subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
              'tracked_upstream_changes': subprocess.check_output(['git','-C',str(upstream),'diff','--name-only'],text=True).splitlines(),
              'expected_openie_calls': 2 * sum(counts.values()), 'max_recognition_calls': 1500,
              'reader_calls': 1500, 'external_api_calls': 0,
              'embedding_batch_size': 4, 'embedding_max_length': 2048,
              'allocator_config': 'expandable_segments:True',
              'storage_adapter': 'FP16 SQLite persistence; upstream in-memory FP32 and IDs/order unchanged',
              'algorithm': 'upstream NER/triples, entity/fact embeddings, recognition filter, synonym edges, PPR and original reader prompts',
              'stages': ['official pre_openie with vLLM only', 'offline passage/entity/fact NV encoding with vLLM stopped',
                         'official index using cached OpenIE and FP16 vectors', 'official rag_qa via loopback services'],
              'formal_run_started': False}
    save(root / 'reports/b2_preparation.json', report)
    return report


def config(root, dataset, folder, model, base_url, workers=8):
    from hipporag.utils.config_utils import BaseConfig
    return BaseConfig(save_dir=str(folder / 'index'), dataset=dataset, llm_name=model,
                      llm_base_url=base_url, embedding_model_name='nvidia/NV-Embed-v2',
                      embedding_provider='nvembed', embedding_batch_size=4, embedding_max_seq_len=2048,
                      embedding_return_as_normalized=True, embedding_model_dtype='auto',
                      openie_mode='online', openie_max_workers=workers, seed=42, temperature=0,
                      qa_top_k=5, llm_supports_max_completion_tokens=False)


def resolve_identity(folder, base_url, model=None, allow_discovery=True):
    path = folder / 'run_identity.json'
    if path.exists():
        identity = json.loads(path.read_text())
        if identity.get('b2_protocol') != B2_PROTOCOL:
            raise ValueError('B2 extraction protocol changed; archive the old run and use a fresh directory')
        if identity['base_url'] != base_url or (model and identity['model'] != model):
            raise ValueError('B2 local reader identity changed; use a fresh run directory')
        return identity['model']
    if model is None:
        if not allow_discovery:
            raise ValueError('Run the OpenIE stage first or supply the exact local served model name')
        models = request(base_url.rstrip('/') + '/models')['data']
        if len(models) != 1:
            raise ValueError('Specify the local served model explicitly')
        model = models[0]['id']
    save(path, {'model': model, 'base_url': base_url, **protocol_settings()})
    return model


def construct(root, dataset, folder, model, base_url, endpoint=None, workers=8, llms=None,
              force_graph_rebuild=False, split='dev-A'):
    from hipporag import HippoRAG
    from .hipporag_fp16_store import persistence_adapter
    settings = config(root, dataset, folder, model, base_url, workers)
    settings.force_index_from_scratch = force_graph_rebuild
    cache = FP16Cache(folder / 'nv_cache.fp16.sqlite')
    embedding = CachedNVEmbedding(settings, cache, endpoint=endpoint)
    if llms is None:
        extraction, reader = local_llms(settings, folder)
    else:
        extraction, reader = llms
    _, _, sources = data(root, dataset, split)
    inventory_path=root/'reports/model_inventory.json'
    inventory=json.loads(inventory_path.read_text()) if inventory_path.exists() else []
    reader_revision=next((item['revision'] for item in inventory if item['repo_id']=='Qwen/Qwen3.5-9B'), 'mock-no-inventory')
    identity = json.dumps({'model': model, **protocol_settings(), 'reader_revision':reader_revision, 'corpus_sha256': sources['corpus']['sha256'],
                           'nv_revision': '3fa59658547db50a1e8e3346cf057fd0c77ed6ef',
                           'nv_max_length': 2048, 'storage': 'fp16_sqlite_persistence_v1'}, sort_keys=True)
    with persistence_adapter():
        engine = HippoRAG(global_config=settings, extraction_llm=extraction, qa_llm=reader,
                         embedding_model=embedding, index_identity=identity)
    # Observation only: the overridden methods delegate to the original parser
    # and return its exact result. The running benchmark uses plain OpenIE.
    from .hipporag_preflight import AuditedOpenIE, load_tokenizer
    tokenizer_path=root/'models/Qwen/Qwen3.5-9B'
    tokenizer=load_tokenizer(root) if tokenizer_path.exists() else None
    engine.openie=AuditedOpenIE(extraction,max_workers=workers,
                               ner_max_tokens=settings.openie_ner_max_tokens,
                               triple_max_tokens=settings.openie_triple_max_tokens,
                               audit_path=folder/'openie_audit.jsonl',tokenizer=tokenizer)
    return engine, cache, (extraction, reader)


def benchmark(root, dataset, profile, workers, passages=100, base_url='http://127.0.0.1:18080/v1', model=None,
              server_startup_seconds=None):
    """Real official two-call OpenIE workload; profile caches are independent."""
    from hipporag.information_extraction.openie_openai import OpenIE
    from hipporag.utils.misc_utils import compute_mdhash_id
    if passages != 100 or workers not in (4, 16):
        raise ValueError('R3 profiles use 100 dev passages and workers 4 or 16')
    documents, _, sources = data(root, dataset)
    folder = root / 'runs/p0_dev/openie_benchmark' / profile
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('A benchmark profile must have a fresh independent cache; choose a new profile ID')
    model = resolve_identity(folder, base_url, model)
    settings = extraction_config(config(root, dataset, folder, model, base_url, workers))
    # The same separate one-passage warmup precedes each profile. It is outside
    # the measured first100 and has an independent cache and call ledger.
    warmup_llm = LoggedOfficialLLM(settings, folder / 'warmup', 'benchmark_warmup')
    warmup_openie = OpenIE(warmup_llm, max_workers=workers,
                          ner_max_tokens=settings.openie_ner_max_tokens, triple_max_tokens=settings.openie_triple_max_tokens)
    warmup_began = time.time()
    try:
        warmup_text = documents[100]
        warmup_openie.batch_openie({compute_mdhash_id(warmup_text, prefix='chunk-'): {'content': warmup_text}})
    finally:
        warmup_llm.close()
    warmup_seconds = time.time() - warmup_began
    llm = LoggedOfficialLLM(settings, folder, 'benchmark_openie')
    openie = OpenIE(llm, max_workers=workers,
                    ner_max_tokens=settings.openie_ner_max_tokens, triple_max_tokens=settings.openie_triple_max_tokens)
    chunks = {compute_mdhash_id(text, prefix='chunk-'): {'content': text} for text in documents[:passages]}
    started = time.time()
    try:
        ner, triples = openie.batch_openie(chunks)
    finally:
        llm.close()
    elapsed = time.time() - started
    usage = call_statistics(folder / 'llm_calls.jsonl')
    if usage['cache_hits'] or usage['successful_requests'] != 2 * passages or usage['errors']:
        raise RuntimeError('OpenIE benchmark did not issue exactly 200 uncached successful calls')
    total_passages = sum(len(data(root, name)[0]) for name in DATASETS)
    report = {'profile': profile, **protocol_settings(), 'dataset': dataset, 'sources': sources, 'model': model, 'base_url': base_url,
              'workers': workers, 'passages': passages, 'wall_seconds': elapsed, 'calls': usage,
              'server_startup_seconds': server_startup_seconds,
              'warmup': {'passages': 1, 'corpus_index': 100, 'wall_seconds': warmup_seconds,
                         'calls': call_statistics(folder / 'warmup/llm_calls.jsonl')},
              'prompt_tokens_per_second': usage['prompt_tokens'] / elapsed,
              'completion_tokens_per_second': usage['completion_tokens'] / elapsed,
              'total_tokens_per_second': usage['total_tokens'] / elapsed,
              'extracted_entities': sum(len(row.unique_entities) for row in ner.values()),
              'extracted_triples': sum(len(row.triples) for row in triples.values()),
              'offline_openie_estimate_seconds': elapsed * total_passages / passages,
              'offline_passages': total_passages, 'offline_openie_calls': 2 * total_passages,
              'online_calls_max': 3000,
              'online_sequential_proxy_seconds': usage['summed_request_seconds'] / usage['logical_calls'] * 3000,
              'estimation_limitations': 'Mu first100 OpenIE-call workload; dataset length mix, embedding/graph work and retrieval/reader latency differ. Online value is only an OpenIE-call-latency proxy; combine with R3 chat timings.',
              'thinking': 'server default disabled; no per-request template override', 'status': 'passed'}
    save(folder / 'report.json', report)
    return report


def official_answers(root, samples):
    source = root / 'external/HippoRAG/main.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'get_gold_answers')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['get_gold_answers'](samples)


def execute(root, dataset, phase, base_url, model=None, endpoint='http://127.0.0.1:18081', workers=8, split='dev-A'):
    documents, samples, sources = data(root, dataset, split)
    folder = run_folder(root, dataset, split)
    folder.mkdir(parents=True, exist_ok=True)
    extraction_log=folder/'calls/extraction/llm_calls.jsonl'
    reader_log=folder/'calls/reader/llm_calls.jsonl'
    extraction_before=call_statistics(extraction_log)
    reader_before=call_statistics(reader_log)
    engine,llms=None,()
    began = time.time()
    try:
        model = resolve_identity(folder, base_url, model, allow_discovery=phase in ('openie', 'run'))
        engine, cache, llms = construct(root, dataset, folder, model, base_url,
                                      endpoint=endpoint if phase == 'run' else None, workers=workers,
                                      force_graph_rebuild=phase == 'build', split=split)
        if phase == 'openie':
            from .hipporag_preflight import preflight_documents
            bounds=preflight_documents(documents,engine.openie.tokenizer)
            save(folder/'input_preflight.json',bounds)
            if not bounds['passed_definite_bounds']:
                raise RuntimeError('OpenIE input exceeds the fixed context limit; no document is truncated or skipped')
            try:
                engine.pre_openie(documents)
            except RuntimeError as error:
                if str(error) != "Offline OpenIE completed. Run indexing again with openie_mode='online' to build the graph.":
                    raise
            outcome = {'openie_state_path': engine.openie_state_path, 'documents': len(documents)}
        elif phase == 'encode-index':
            if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True).strip():
                raise RuntimeError('All model services must be stopped before offline NV indexing')
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            from .nv_adapter import NVEncoder
            from hipporag.utils.misc_utils import reformat_openie_results, text_processing, extract_entity_nodes, flatten_facts, compute_mdhash_id
            chunk_ids = [compute_mdhash_id(text, prefix='chunk-') for text in documents]
            info, missing = engine.load_existing_openie(chunk_ids)
            if missing:
                raise RuntimeError('Complete official OpenIE extraction before offline vector encoding')
            _, triple_results = reformat_openie_results(info)
            chunk_triples = [[text_processing(triple) for triple in triple_results[key].triples] for key in chunk_ids]
            entities, _ = extract_entity_nodes(chunk_triples)
            facts = [str(fact) for fact in flatten_facts(chunk_triples)]
            encoder = NVEncoder(root, batch_size=4, max_length=2048)
            cache.put(documents, dense_passages(root, dataset, split, documents, sources))
            for label, texts in [('entities', entities), ('facts', facts)]:
                pending = [text for text, vector in zip(texts, cache.get(texts)) if vector is None]
                for begin in range(0, len(pending), 4):
                    if shutil.disk_usage(root).free < 15 * 1024**3:
                        raise RuntimeError('Offline encoding requires 15 GiB free disk')
                    batch = pending[begin:begin + 4]
                    cache.put(batch, encoder.encode(batch, 'passage'))
                    if begin % 200 == 0:
                        print(json.dumps({'dataset': dataset, 'kind': label, 'completed': begin + len(batch), 'total': len(pending)}), flush=True)
            outcome = {'passages': len(documents), 'entities': len(entities), 'facts': len(facts), 'cache': str(cache.path)}
        elif phase == 'build':
            engine.index(documents)
            outcome = {'graph_vertices': engine.graph.vcount(), 'graph_edges': engine.graph.ecount(),
                       'entities': len(engine.entity_embedding_store.get_all_ids()), 'facts': len(engine.fact_embedding_store.get_all_ids())}
        elif phase == 'run':
            from .hipporag_upstream import gold_function
            gold = gold_function(root)(samples, dataset)
            results, responses, metadata, retrieval, qa = engine.rag_qa([row['question'] for row in samples],
                                                                      gold_docs=gold, gold_answers=official_answers(root, samples))
            predictions = [{'id': row.get('id', row.get('_id')), 'answer': result.answer,
                            'documents': result.docs[:5], 'graph_seeds': result.graph_seeds,
                            'response': response, 'metadata': extra} for row, result, response, extra in
                           zip(samples, results, responses, metadata)]
            save(folder / 'predictions.json', predictions)
            outcome = {'retrieval': retrieval, 'qa': qa, 'questions': len(results),
                       'graph_seeded_queries': sum(bool(result.graph_seeds) for result in results),
                       'dense_fallback_queries': sum(not result.graph_seeds for result in results),
                       'ppr_seconds': engine.ppr_time}
        else:
            raise ValueError(phase)
        from .hipporag_preflight import summarize_audit
        extraction_after=call_statistics(extraction_log)
        reader_after=call_statistics(reader_log)
        extraction_delta=phase_call_delta(extraction_before,extraction_after)
        reader_delta=phase_call_delta(reader_before,reader_after)
        operational_errors=require_clean_phase_calls(extraction_delta,reader_delta)
        report = {'baseline': 'B2 official HippoRAG 2', **protocol_settings(), 'dataset': dataset, 'split': split, 'phase': phase,
                  'sources': sources, 'status':'passed','operational_errors':operational_errors,
                  'model': model, 'base_url': base_url, 'seconds': time.time() - began,
                  'openie_max_workers': workers,
                  'storage': 'FP16 SQLite persistence; same upstream IDs/order and FP32 computation', **outcome,
                  'openie_quality': summarize_audit(folder/'openie_audit.jsonl'),
                  'extraction_calls':extraction_after,'reader_calls':reader_after,
                  'extraction_calls_this_phase':extraction_delta,'reader_calls_this_phase':reader_delta,
                  'metric_units':{'qa':'0-1','retrieval':'0-1'}}
        save(folder / (phase + '_report.json'), report)
        return report
    except Exception as error:
        from .hipporag_preflight import summarize_audit
        save(folder/(phase+'_failed.json'),{'dataset':dataset,'phase':phase,'status':'failed', **protocol_settings(),
             'error_type':type(error).__name__,'error':str(error),
             'operational_errors':sum(delta.get('errors',0) for delta in
                  [phase_call_delta(extraction_before,call_statistics(extraction_log)),
                   phase_call_delta(reader_before,call_statistics(reader_log))]),
             'openie_quality':summarize_audit(folder/'openie_audit.jsonl'),
             'calls':call_statistics(folder/'calls/extraction/llm_calls.jsonl'),
             'input_documents_not_skipped_or_truncated':True})
        raise
    finally:
        if engine is not None: engine.close()
        for llm in llms:
            llm.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/workspace/hg_reasoning_agent'))
    parser.add_argument('--phase', choices=['audit', 'preflight', 'benchmark', 'openie', 'encode-index', 'build', 'run'], default='audit')
    parser.add_argument('--dataset', choices=TEST_DATASETS, default='musique')
    parser.add_argument('--split', choices=SPLITS, default='dev-A')
    parser.add_argument('--profile')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--passages', type=int, default=100)
    parser.add_argument('--base-url', default='http://127.0.0.1:18080/v1')
    parser.add_argument('--model')
    parser.add_argument('--server-startup-seconds', type=float)
    parser.add_argument('--approved', action='store_true', help='Only supply after root R3 gates and explicit baseline GO')
    arguments = parser.parse_args()
    if arguments.phase == 'audit':
        result = preparation_audit(arguments.root)
    elif arguments.phase == 'preflight':
        from .hipporag_preflight import load_tokenizer, preflight_documents
        documents,_,sources=data(arguments.root,arguments.dataset)
        result=preflight_documents(documents,load_tokenizer(arguments.root))
        result['sources']=sources
        result['scope']='CPU token preflight only; no LLM/GPU requests'
        save(arguments.root/'reports'/('b2_preflight_'+arguments.dataset+'.json'),result)
    elif arguments.phase == 'benchmark':
        if not arguments.profile:
            parser.error('--profile is required for independent uncached benchmarks')
        result = benchmark(arguments.root, arguments.dataset, arguments.profile, arguments.workers or 4,
                           arguments.passages, arguments.base_url, arguments.model, arguments.server_startup_seconds)
    else:
        if not arguments.approved:
            parser.error('Formal B2 stages require explicit root GO via --approved')
        if arguments.phase == 'build':
            # Upstream synonym-KNN otherwise chooses CUDA automatically. Hide GPU
            # for this CPU-only graph stage; its upstream math remains unchanged.
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        result = execute(arguments.root, arguments.dataset, arguments.phase, arguments.base_url,
                         arguments.model, workers=arguments.workers or 8, split=arguments.split)
    print(json.dumps(result), flush=True)
