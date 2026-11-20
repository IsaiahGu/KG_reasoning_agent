#!/usr/bin/env python
"""Write configs/queue_v2.json: validation runs for the multi-derivation design (MuSiQue, train-S and dev-A only).

(1) fit the support model and build the two self-built plan memories (CPU);
(2) held-out train-S questions: plans from each self-built memory, and a second plan from the annotated memory;
(3) dev-A: the fixed-plan loop with records, the same three extra plans;
(4) annotated-memory plans on train-S positions 0-1500 (each question skips its own plan), for fitting and 2,000-question scores;
(5) the merge report. dev-B and the test subsets are not in this queue.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = 'envs/main/bin/python'
MEMORY = 'runs/plan_memory/musique_self_%s.json'
SUPPORT = 'configs/support/musique_chain_v1.json'


def chain(name, split, tag, extra, questions=500, est=16, deps=()):
    base = 'runs/chain/%s/musique/%s' % (split, tag)
    return {'name': name, 'cmd': [PYTHON, '-B', 'scripts/run_chain.py', '--dataset', 'musique', '--split', split, '--tag', tag,
                                  '--workers', '12', '--candidates', '1'] + list(extra),
            'check': 'chain', 'path': base + '/summary.json', 'progress': base + '/records.jsonl', 'questions': questions,
            'est_minutes': est, 'deps': list(deps)}


def cpu(name, command, path, deps=()):
    return {'name': name, 'cmd': [PYTHON, '-B'] + command, 'services': 'any', 'check': 'file', 'path': path, 'est_minutes': 2,
            'stall_minutes': 30, 'deps': list(deps)}


VAL = ['--offset', '1500', '--limit', '500']
KNN = ['--plan', 'knn']
jobs = [
    cpu('v2_fit_support', ['scripts/fit_support_model.py', '--out', SUPPORT], SUPPORT),
    cpu('v2_memory_answer', ['scripts/build_plan_memory.py', '--admit', 'answer', '--out', MEMORY % 'answer'], MEMORY % 'answer'),
    cpu('v2_memory_support', ['scripts/build_plan_memory.py', '--admit', 'support', '--out', MEMORY % 'support'], MEMORY % 'support'),
    chain('v2_val_selfmem_support', 'train-S', 'selfmem_support_val', VAL + KNN + ['--plan-cases', MEMORY % 'support'],
          deps=['v2_memory_support']),
    chain('v2_val_selfmem_answer', 'train-S', 'selfmem_answer_val', VAL + KNN + ['--plan-cases', MEMORY % 'answer'],
          deps=['v2_memory_answer']),
    chain('v2_val_knn_skip6', 'train-S', 'knn_skip6_val', VAL + KNN + ['--plan-pool-limit', '1500', '--plan-skip', '6']),
    chain('v2_devA_single', 'dev-A', 'single', []),
    chain('v2_devA_selfmem_support', 'dev-A', 'selfmem_support', KNN + ['--plan-cases', MEMORY % 'support'], deps=['v2_memory_support']),
    chain('v2_devA_selfmem_answer', 'dev-A', 'selfmem_answer', KNN + ['--plan-cases', MEMORY % 'answer'], deps=['v2_memory_answer']),
    chain('v2_devA_knn_skip6', 'dev-A', 'knn_skip6', KNN + ['--plan-skip', '6']),
    chain('v2_fit_knn_loo', 'train-S', 'knn_single_fit', ['--offset', '0', '--limit', '1500'] + KNN + ['--plan-loo'],
          questions=1500, est=50),
    {'name': 'v2_merge_report', 'cmd': [PYTHON, '-B', 'scripts/merge_report.py'], 'services': 'any', 'check': 'marker',
     'est_minutes': 2, 'stall_minutes': 30},
    # Round 2 of the self-built memory (added 2026-11-18): plan the fit questions again with the round-1 memory (each
    # question skips its own plan), admit the plans whose answer is right, and test the larger memory.
    chain('v3_fit_selfmem_answer', 'train-S', 'selfmem_answer_fit',
          ['--offset', '0', '--limit', '1500'] + KNN + ['--plan-cases', MEMORY % 'answer', '--plan-loo'], questions=1500, est=30),
    cpu('v3_memory_answer2', ['scripts/build_plan_memory.py', '--admit', 'answer', '--tags', 'selfmem_answer_fit', 'single_fit',
                              '--out', MEMORY % 'answer2'], MEMORY % 'answer2', deps=['v3_fit_selfmem_answer']),
    chain('v3_val_selfmem2', 'train-S', 'selfmem2_answer_val', VAL + KNN + ['--plan-cases', MEMORY % 'answer2'],
          deps=['v3_memory_answer2']),
    chain('v3_devA_selfmem2', 'dev-A', 'selfmem2_answer', KNN + ['--plan-cases', MEMORY % 'answer2'], deps=['v3_memory_answer2']),
    {'name': 'v3_merge_report', 'cmd': [PYTHON, '-B', 'scripts/merge_report.py', '--out', 'reports/merge_v2_report.md'],
     'services': 'any', 'check': 'marker', 'est_minutes': 2, 'stall_minutes': 30},
]


# The same method on 2Wiki and HotpotQA (added 2026-11-18, DESIGN_V2 gate V3). No decomposition annotation exists
# there: the plan memory is self-built (two rounds) and the support model is fitted on each dataset's own train-S.
def other(short, dataset):
    def run(name, split, tag, extra, questions=500, est=9, deps=()):
        base = 'runs/chain/%s/%s/%s' % (split, dataset, tag)
        return {'name': 'v4_%s_%s' % (short, name), 'cmd': [PYTHON, '-B', 'scripts/run_chain.py', '--dataset', dataset, '--split', split,
                                                           '--tag', tag, '--workers', '12', '--candidates', '1'] + list(extra),
                'check': 'chain', 'path': base + '/summary.json', 'progress': base + '/records.jsonl', 'questions': questions,
                'est_minutes': est, 'deps': ['v4_%s_%s' % (short, dep) for dep in deps]}

    def tool(name, command, path, deps=()):
        return {'name': 'v4_%s_%s' % (short, name), 'cmd': [PYTHON, '-B'] + command, 'services': 'any', 'check': 'file', 'path': path,
                'est_minutes': 2, 'stall_minutes': 30, 'deps': ['v4_%s_%s' % (short, dep) for dep in deps]}

    memory, support = 'runs/plan_memory/%s_self_answer%%s.json' % dataset, 'configs/support/%s_chain_v1.json' % dataset
    fit, val = ['--offset', '0', '--limit', '1500'], ['--offset', '1500', '--limit', '500']
    build = ['scripts/build_plan_memory.py', '--dataset', dataset, '--admit', 'answer']
    return [
        run('single_fit', 'train-S', 'single_fit', fit, questions=1500, est=25, deps=['embed']),
        run('a0_fit', 'train-S', 'a0_fit', fit + ['--mode', 'dense'], questions=1500, est=8, deps=['embed']),
        run('single_val', 'train-S', 'single_val', val, deps=['embed']),
        run('a0_val', 'train-S', 'a0', val + ['--mode', 'dense'], est=3, deps=['embed']),
        tool('fit_support', ['scripts/fit_support_model.py', '--dataset', dataset, '--tags', 'single_fit', '--out', support], support,
             deps=['single_fit', 'a0_fit']),
        tool('memory1', build + ['--tags', 'single_fit', '--out', memory % ''], memory % '', deps=['single_fit']),
        run('selfmem_fit', 'train-S', 'selfmem_answer_fit', fit + KNN + ['--plan-cases', memory % '', '--plan-loo'], questions=1500,
            est=25, deps=['memory1']),
        tool('memory2', build + ['--tags', 'selfmem_answer_fit', 'single_fit', '--out', memory % '2'], memory % '2', deps=['selfmem_fit']),
        run('val_selfmem', 'train-S', 'selfmem_answer_val', val + KNN + ['--plan-cases', memory % ''], deps=['memory1']),
        run('val_selfmem2', 'train-S', 'selfmem2_answer_val', val + KNN + ['--plan-cases', memory % '2'], deps=['memory2']),
        run('devA_single', 'dev-A', 'single', []),
        run('devA_selfmem', 'dev-A', 'selfmem_answer', KNN + ['--plan-cases', memory % ''], deps=['memory1']),
        run('devA_selfmem2', 'dev-A', 'selfmem2_answer', KNN + ['--plan-cases', memory % '2'], deps=['memory2']),
        {'name': 'v4_%s_report' % short, 'cmd': [PYTHON, '-B', 'scripts/merge_report.py', '--dataset', dataset, '--model', support,
                                                '--out', 'reports/merge_%s_report.md' % dataset],
         'services': 'any', 'check': 'marker', 'est_minutes': 2, 'stall_minutes': 30, 'deps': ['v4_%s_fit_support' % short]},
    ]


for short, dataset in (('2w', '2wikimultihopqa'), ('hp', 'hotpotqa')):
    # offline encoding of the train-S corpus needs the GPU to itself: services are stopped for this job
    jobs.append({'name': 'v4_%s_embed' % short, 'services': 'off', 'check': 'marker', 'est_minutes': 20, 'stall_minutes': 20,
                 'cmd': ['envs/embed/bin/python', '-m', 'hyperbind.prepare_split_embeddings', '--dataset', dataset, '--split', 'train-S']})
# Hypergraph module trial (added 2026-11-18): hop retrieval over the P1 passage hypergraph (index exists for the dev
# corpora only), as a new derivation on dev-A. Placed before the long v4 jobs so the result is known early.
def hyper(name, dataset, tag, extra=()):
    base = 'runs/chain/dev-A/%s/%s' % (dataset, tag)
    return {'name': name, 'cmd': [PYTHON, '-B', 'scripts/run_chain.py', '--dataset', dataset, '--split', 'dev-A', '--tag', tag,
                                  '--workers', '12', '--candidates', '1', '--retrieval', 'hyper'] + list(extra),
            'check': 'chain', 'path': base + '/summary.json', 'progress': base + '/records.jsonl', 'questions': 500,
            'est_minutes': 12, 'deps': []}


jobs += [hyper('h_mu_hyper_single', 'musique', 'hyper_single'),
         hyper('h_mu_hyper_selfmem2', 'musique', 'hyper_selfmem2', KNN + ['--plan-cases', MEMORY % 'answer2']),
         hyper('h_2w_hyper_single', '2wikimultihopqa', 'hyper_single'),
         hyper('h_hp_hyper_single', 'hotpotqa', 'hyper_single')]

# Test subsets (added 2026-11-18 on the user's instruction): every derivation named in configs/frozen_v2.json, 1,000
# questions each. The headline configuration was frozen in that file before these jobs were written.
def test(short, dataset, name, tag, extra, est=17, deps=()):
    base = 'runs/chain/test/%s/%s' % (dataset, tag)
    return {'name': 't_%s_%s' % (short, name), 'cmd': [PYTHON, '-B', 'scripts/run_chain.py', '--dataset', dataset, '--split', 'test',
                                                      '--tag', tag, '--workers', '12'] + list(extra),
            'check': 'chain', 'path': base + '/summary.json', 'progress': base + '/records.jsonl', 'questions': 1000,
            'est_minutes': est, 'deps': list(deps)}


def test_jobs(short, dataset):
    memory = 'runs/plan_memory/%s_self_answer%%s.json' % dataset
    first, second = ([], []) if short == 'mu' else (['v4_%s_memory1' % short], ['v4_%s_memory2' % short])
    one = ['--candidates', '1']
    return [test(short, dataset, 'a0', 'a0', ['--mode', 'dense'], est=5),
            test(short, dataset, 'single', 'single', one),
            test(short, dataset, 'selfmem', 'selfmem_answer', one + KNN + ['--plan-cases', memory % ''], deps=first),
            test(short, dataset, 'selfmem2', 'selfmem2_answer', one + KNN + ['--plan-cases', memory % '2'], deps=second),
            test(short, dataset, 'cand3', 'cand3', [])]


REPORT = {'name': 't_report', 'cmd': [PYTHON, '-B', 'scripts/test_report.py'], 'services': 'any', 'check': 'marker',
          'est_minutes': 2, 'stall_minutes': 30}
# Hypergraph derivation on test (added after it passed the dev-A rule in configs/frozen_v2.json): build the passage
# hypergraph of each test corpus (GPU free of vLLM; the script runs its own embedding service), then the hyper chain.
def index_job(short, dataset, deps=()):
    return {'name': 'x_%s_index' % short, 'services': 'off', 'check': 'file', 'path': 'runs/test_index/%s/metadata.json' % dataset,
            'progress': 'runs/test_index/%s/mentions.jsonl' % dataset, 'est_minutes': 60, 'stall_minutes': 45,
            'cmd': [PYTHON, '-B', 'scripts/build_test_graph_index.py', '--dataset', dataset], 'deps': list(deps)}


def hyper_job(short, dataset, est=20):
    return test(short, dataset, 'hyper', 'hyper_single', ['--candidates', '1', '--retrieval', 'hyper'], est=est,
                deps=['x_%s_index' % short])


# Order (changed 2026-11-18 05:55 UTC, job names and commands unchanged): one dataset after the other, and within a
# dataset the five derivations of the headline configuration first, so that each official headline row is complete as
# early as possible. Derivations that only ablations use run after the headline rows of all datasets.
ablations = []
for short, dataset in (('mu', 'musique'), ('2w', '2wikimultihopqa'), ('hp', 'hotpotqa')):
    if short != 'mu':
        jobs += other(short, dataset)
    runs = test_jobs(short, dataset)
    jobs += runs[:4] + [index_job(short, dataset), hyper_job(short, dataset)]
    ablations += runs[4:]
    if short == 'mu':
        ablations += [test('mu', 'musique', 'knn', 'knn_single', ['--candidates', '1'] + KNN),
                      test('mu', 'musique', 'cand3_scorer', 'cand3_s1500_bt',
                           ['--selector', 'scorer', '--scorer', 'configs/scorers/musique_l2_1500.json', '--backtrack'])]

# NQ and PopQA from the HippoRAG 2 suite (added 2026-11-18 after the user approved the download): test only.
# Transfer setting (configs/frozen_v2.json): the plan memories and the support model are HotpotQA's.
HOTPOT_MEMORY = 'runs/plan_memory/hotpotqa_self_answer%s.json'
EXTRA = (('nq', 'nq_rear'), ('pq', 'popqa'))
# Order (changed 2026-11-18 12:00 UTC, job names and commands unchanged): one dataset after the other, and within a
# dataset the hypergraph-retrieval chain right after the dense read and the fixed-example chain, so that the comparison
# "hypergraph retrieval against dense retrieval in the same loop" is the first thing known on each new dataset.
# The hypergraph module was removed on 2026-11-18 after the NQ run (configs/final.json): PopQA, which had not run yet,
# gets no index or hypergraph job. The jobs that did run stay in the queue file as the record of what was executed.
for short, dataset in EXTRA:
    embed, one, transfer = 'e_%s_embed' % short, ['--candidates', '1'], KNN + ['--plan-dataset', 'hotpotqa']
    hypergraph = [index_job(short, dataset, deps=[embed]), hyper_job(short, dataset, est=12)] if short == 'nq' else []
    jobs += [{'name': embed, 'cmd': [PYTHON, '-B', 'scripts/prepare_test_embeddings.py', '--dataset', dataset],
              'check': 'marker', 'est_minutes': 25, 'stall_minutes': 20},
             test(short, dataset, 'a0', 'a0', ['--mode', 'dense'], est=5, deps=[embed]),
             test(short, dataset, 'single', 'single', one, est=10, deps=[embed])] + hypergraph + [
             test(short, dataset, 'selfmem', 'selfmem_answer', one + transfer + ['--plan-cases', HOTPOT_MEMORY % ''], est=10,
                  deps=[embed, 'v4_hp_memory1']),
             test(short, dataset, 'selfmem2', 'selfmem2_answer', one + transfer + ['--plan-cases', HOTPOT_MEMORY % '2'], est=10,
                  deps=[embed, 'v4_hp_memory2'])]

# Supplementary runs for the paper (added 2026-11-18 15:45 UTC, after the final method was fixed in configs/final.json;
# none of them changes it). They are baselines and ablations that the method's own results are compared with.
#  1. Same reader and embedder on the test subsets: IRCoT (baseline B3) and the official HippoRAG 2 (baseline B2,
#     four phases: OpenIE with the LLM, offline encoding with the GPU free, graph build on CPU, answering).
#  2. Self-consistency: chain A1 sampled again at temperature 0.7 and 0.3 (same loop, no derivation diversity).
#  3. Design of the chain: what the final reader sees (notes only, passages only), and four passages per hop.
MULTIHOP = (('mu', 'musique'), ('2w', '2wikimultihopqa'), ('hp', 'hotpotqa'))
B2 = ['envs/embed/bin/python', '-u', '-m', 'hyperbind.baseline_hipporag', '--split', 'test', '--workers', '16', '--approved']
B2_MINUTES = {'mu': (125, 50, 215), '2w': (65, 25, 100), 'hp': (105, 45, 85)}  # OpenIE, encoding, answering (from the dev-A runs)
# The baseline's LLM phases run with the model server's default structured-output setting, with which its JSON-object
# requests were validated on 2026-11-16; under the compact setting of the chain agent they come back malformed. The
# queue therefore stops its own services for these jobs and scripts/with_services.py starts the server they need.
OWN_LLM = [PYTHON, '-B', 'scripts/with_services.py', '--llm', 'default-json']


def b2_job(short, dataset, phase, est, wrapper=(), **fields):
    folder = 'runs/b2_test/%s' % dataset
    return dict({'name': 'b_%s_hr_%s' % (short, phase.split('-')[0]),
                 'cmd': list(wrapper) + B2 + ['--phase', phase, '--dataset', dataset],
                 'check': 'file', 'path': '%s/%s_report.json' % (folder, phase), 'est_minutes': est, 'stall_minutes': 45,
                 'timeout_minutes': 360}, **fields)


for short, dataset in MULTIHOP:
    base = 'runs/chain/test/%s/b3_ircot' % dataset
    jobs.append({'name': 'b_%s_ircot' % short, 'cmd': [PYTHON, '-B', 'scripts/run_ircot_test.py', '--dataset', dataset],
                 'check': 'file', 'path': base + '/scored.json', 'progress': base + '/records.jsonl', 'est_minutes': 20})
def sampled(short, dataset, temperature, sample, **fields):
    tag = 'sc%02d_%d' % (round(10 * temperature), sample)  # sc07_1: temperature 0.7, sample 1 (the sample number is the seed)
    return test(short, dataset, tag, tag, ['--candidates', '1', '--temperature', str(temperature), '--seed', str(sample)], **fields)


# Gate: the unchanged extraction on 60 test passages must parse without a single error before hours are spent on it.
jobs.append({'name': 'b_hr_smoke', 'services': 'off', 'check': 'file', 'path': 'runs/b2_test/openie_smoke.json', 'est_minutes': 6,
             'max_attempts': 2, 'cmd': OWN_LLM + ['--', 'envs/embed/bin/python', '-B', 'scripts/smoke_hipporag_openie.py']})
for short, dataset in MULTIHOP:
    jobs.append(b2_job(short, dataset, 'openie', B2_MINUTES[short][0], wrapper=OWN_LLM + ['--'], services='off', deps=['b_hr_smoke'],
                       progress='runs/b2_test/%s/calls/extraction/llm_calls.jsonl' % dataset))
for short, dataset in MULTIHOP:
    jobs += [b2_job(short, dataset, 'encode-index', B2_MINUTES[short][1], services='off', deps=['b_%s_hr_openie' % short]),
             b2_job(short, dataset, 'build', 8, services='any', deps=['b_%s_hr_encode' % short])]
# The answering phase sends one request at a time, so the three datasets run side by side on the same model server.
jobs.append({'name': 'b_hr_run', 'services': 'off', 'check': 'file', 'path': 'runs/b2_test/run.done.json',
             'est_minutes': B2_MINUTES['mu'][2], 'stall_minutes': 90, 'timeout_minutes': 480,
             'deps': ['b_%s_hr_build' % short for short, _ in MULTIHOP],
             'cmd': OWN_LLM + ['--embed', '--', PYTHON, '-B', 'scripts/run_together.py', '--logs', 'runs/b2_test/logs',
                               '--done', 'runs/b2_test/run.done.json']
             + ['%s=%s@@runs/b2_test/%s/run_report.json' % (short, ' '.join(B2 + ['--phase', 'run', '--dataset', dataset]), dataset)
                for short, dataset in MULTIHOP]})
for short, dataset in MULTIHOP:
    jobs.append({'name': 'b_%s_hr_score' % short, 'cmd': [PYTHON, '-B', 'scripts/score_hipporag_test.py', '--dataset', dataset],
                 'services': 'any', 'check': 'file', 'path': 'runs/chain/test/%s/b2_hipporag2/scored.json' % dataset, 'est_minutes': 2,
                 'stall_minutes': 30, 'deps': ['b_hr_run']})
# Self-consistency at the usual temperature 0.7 (three samples) and at a milder 0.3 (two samples). With the greedy chain
# A1 as the first member, two samples match the final method's budget of three chains; the third shows a larger budget.
for temperature, samples in ((0.7, (1, 2, 3)), (0.3, (1, 2))):
    for short, dataset in MULTIHOP:
        jobs += [sampled(short, dataset, temperature, sample) for sample in samples]
for short, dataset in MULTIHOP:
    one = ['--candidates', '1']
    jobs += [test(short, dataset, 'ctx_notes', 'ctx_notes', one + ['--reader-context', 'notes'], est=8),
             test(short, dataset, 'ctx_passages', 'ctx_passages', one + ['--reader-context', 'passages'], est=8),
             test(short, dataset, 'khop4', 'khop4', one + ['--k-hop', '4'])]
jobs += ablations
# The same-reader HippoRAG 2 on the two single-hop sets as well (added 2026-11-19 03:50 UTC, after the multi-hop baseline
# had finished), so that every column of the main table has the same-reader baseline of the protocol's own paper. Same
# four phases and server setting as above; a second gate on NQ passages first. OpenIE minutes are MuSiQue's 119.5 scaled
# by the passage count, encoding minutes by analogy with 2Wiki and HotpotQA.
B2_SINGLE_MINUTES = {'nq': (100, 40, 150), 'pq': (90, 35, 150)}
jobs.append({'name': 'b_hr_smoke2', 'services': 'off', 'check': 'file', 'path': 'runs/b2_test/openie_smoke_nq_rear.json', 'est_minutes': 6,
             'max_attempts': 2, 'cmd': OWN_LLM + ['--', 'envs/embed/bin/python', '-B', 'scripts/smoke_hipporag_openie.py', '--dataset',
                                                  'nq_rear', '--out', 'runs/b2_test/openie_smoke_nq_rear.json']})
for short, dataset in EXTRA:
    jobs.append(b2_job(short, dataset, 'openie', B2_SINGLE_MINUTES[short][0], wrapper=OWN_LLM + ['--'], services='off',
                       deps=['b_hr_smoke2'], progress='runs/b2_test/%s/calls/extraction/llm_calls.jsonl' % dataset))
for short, dataset in EXTRA:
    jobs += [b2_job(short, dataset, 'encode-index', B2_SINGLE_MINUTES[short][1], services='off', deps=['b_%s_hr_openie' % short]),
             b2_job(short, dataset, 'build', 8, services='any', deps=['b_%s_hr_encode' % short])]
jobs.append({'name': 'b_hr_run2', 'services': 'off', 'check': 'file', 'path': 'runs/b2_test/run2.done.json',
             'est_minutes': B2_SINGLE_MINUTES['nq'][2], 'stall_minutes': 90, 'timeout_minutes': 480,
             'deps': ['b_%s_hr_build' % short for short, _ in EXTRA],
             'cmd': OWN_LLM + ['--embed', '--', PYTHON, '-B', 'scripts/run_together.py', '--logs', 'runs/b2_test/logs',
                               '--done', 'runs/b2_test/run2.done.json']
             + ['%s=%s@@runs/b2_test/%s/run_report.json' % (short, ' '.join(B2 + ['--phase', 'run', '--dataset', dataset]), dataset)
                for short, dataset in EXTRA]})
for short, dataset in EXTRA:
    jobs.append({'name': 'b_%s_hr_score' % short, 'cmd': [PYTHON, '-B', 'scripts/score_hipporag_test.py', '--dataset', dataset],
                 'services': 'any', 'check': 'file', 'path': 'runs/chain/test/%s/b2_hipporag2/scored.json' % dataset, 'est_minutes': 2,
                 'stall_minutes': 30, 'deps': ['b_hr_run2']})
for short, dataset in EXTRA:  # self-consistency on the two single-hop sets, last because it matters least
    jobs += [sampled(short, dataset, 0.7, sample, est=10, deps=['e_%s_embed' % short]) for sample in (1, 2)]
# Control for the merge rule (added 2026-11-19 13:40 UTC, after the self-consistency runs had shown that the merge rule and
# not the source of the derivations carries the gain): the reader itself picks among the distinct answers of the four
# derivations (scripts/run_llm_selector.py). One short call on the questions where the derivations disagree; no rerun.
for split, prefix, sets in (('test', 't', MULTIHOP + EXTRA), ('dev-A', 'd', MULTIHOP)):
    for short, dataset in sets:
        base = 'runs/chain/%s/%s/llm_select' % (split, dataset)
        jobs.append({'name': '%s_%s_llmsel' % (prefix, short), 'check': 'chain', 'path': base + '/summary.json',
                     'progress': base + '/records.jsonl', 'questions': 1000 if split == 'test' else 500, 'est_minutes': 4,
                     'cmd': [PYTHON, '-B', 'scripts/run_llm_selector.py', '--dataset', dataset, '--split', split]})
jobs.append(REPORT)
queue = {'defaults': {'services': 'on', 'max_attempts': 4, 'stall_minutes': 20, 'timeout_minutes': 240}, 'jobs': jobs}
path = ROOT / 'configs/queue_v2.json'
path.write_text(json.dumps(queue, indent=1) + '\n')
print('%d jobs, estimated %.1f hours in total, written to %s' % (len(jobs), sum(job['est_minutes'] for job in jobs) / 60, path))
