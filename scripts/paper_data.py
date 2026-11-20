#!/usr/bin/env python
"""The paper's data package: one file per planned table or figure. CPU only; no LLM is called and no run is changed.

Run after scripts/paper_results.py: this script reads paper/results.json and the finished runs. It writes paper/data/:
  README.md        what every file holds, which table or figure it feeds, and which runs were still missing
  T*.csv, F*.csv   main-paper tables and figures (long format: one row per dataset, split and configuration)
  A*.csv           appendix
  cases.jsonl      case studies picked by fixed rules, with every execution's steps and quotes
  prompts.md       every prompt of the method and of the added control
  config/          frozen configurations, fitted reliability models, hop scorer, implementation details
Development (dev-A) and test rows sit in the same files and carry a split column. F1 and EM are in percent; intervals are
95% bootstrap intervals over questions (10,000 resamples, the resamples of scripts/paper_results.py).
"""
import argparse
import csv
import dataclasses
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import paper_results as base  # noqa: E402
from hyperbind.data import load_test_samples  # noqa: E402
from hyperbind.support import FEATURES, merge  # noqa: E402
from hyperbind.textmatch import same_answer  # noqa: E402

NAMES = base.NAMES
MULTIHOP = ('musique', '2wikimultihopqa', 'hotpotqa')
T07, T03 = ['sc07_1', 'sc07_2', 'sc07_3'], ['sc03_1', 'sc03_2']
RIGHT = 0.5  # an answer counts as right from F1 0.5, as in the labels the reliability model was fitted on
HEAD = ['n', 'f1', 'f1_lo', 'f1_hi', 'em', 'llm_calls_per_question', 'tokens_per_question']
FILES = []
# Where each file is used, in the numbering of paper/OUTLINE.md (second draft). The T/F/A prefixes only name the files.
FEEDS = {'T1_main_results.csv': '表 1 主结果；图 1 首页小图', 'T2_merge_rules.csv': '表 2 合并规则',
         'T3_diversity_source.csv': '表 4 执行的来源', 'F2_f1_vs_executions.csv': '图 3 F1 随执行次数变化',
         'F3_calibration_bins.csv': '图 5 校准图', 'F3_calibration_summary.csv': '图 5 的图注，表 5 的校准数字',
         'F4_cost_accuracy.csv': '图 4 成本和准确率', 'T4a_reliability_features.csv': '表 5 可靠度模型：特征',
         'T4b_reliability_training_size.csv': '表 5 可靠度模型：训练量', 'T4c_reliability_transfer.csv': '表 5 可靠度模型：跨数据集',
         'T4d_reliability_weights.csv': '附录 E 可靠度模型的权重', 'T5_chain_variants.csv': '表 6 回路的设计选择和没有增益的变体',
         'T5b_hypergraph_retrieval.csv': '表 6 的补充：超图检索', 'T6a_breakdown_by_type.csv': '图 6 左：按题型或跳数',
         'T6b_error_categories.csv': '图 6 右：错误归类', 'T7_cost.csv': '表 3 成本', 'T7b_wallclock.csv': '附录 L 运行时间',
         'A1_dataset_statistics.csv': '附录 B 数据集统计', 'A3_agreement.csv': '附录 I 执行是否一致',
         'A4_pairwise_complementarity.csv': '附录 I 互补性热力图，支撑 5.4 节', 'A5_plan_memory_similarity.csv': '附录 J',
         'A6_risk_coverage.csv': '附录 K 选择性回答', 'A7_retrieval.csv': '附录 H 检索质量', 'cases.jsonl': '6.3 节案例，附录 M',
         'prompts.md': '附录 C 提示词', 'config/': '附录 D 实现细节，附录 F 登记的配置'}
# Published results under the same protocol (HippoRAG 2 test subsets), copied from RELATED_RESULTS.md section 2: EM, F1.
PUBLISHED = [
    ('NV-Embed-v2 dense retrieval', 'Llama-3.3-70B', 'arXiv 2502.14802', 'PopQA from memory of the paper, check before use',
     {'hotpotqa': (62.8, 75.3), '2wikimultihopqa': (57.5, 61.5), 'musique': (34.7, 45.7), 'nq_rear': ('', 61.9), 'popqa': ('', 55.7)}),
    ('HippoRAG 2', 'Llama-3.3-70B', 'arXiv 2502.14802', 'PopQA from memory of the paper, check before use',
     {'hotpotqa': (62.7, 75.5), '2wikimultihopqa': (65.0, 71.0), 'musique': (37.2, 48.6), 'nq_rear': ('', 63.3), 'popqa': ('', 56.2)}),
    ('GFM-RAG (single step)', 'GPT-4o-mini', 'arXiv 2502.01113', 'trained graph retriever; HotpotQA corpus of 9,221 passages',
     {'hotpotqa': (51.6, 66.9), '2wikimultihopqa': (69.8, 77.7), 'musique': (30.2, 40.4)}),
    ('IRCoT + GFM-RAG', 'GPT-4o-mini', 'arXiv 2502.01113', 'trained graph retriever; HotpotQA corpus of 9,221 passages',
     {'hotpotqa': (56.0, 71.8), '2wikimultihopqa': (72.5, 80.8), 'musique': (36.6, 49.2)}),
    ('HGRAG', 'Llama-3.3-70B', 'arXiv 2508.11247', '', {'hotpotqa': (63.9, 76.8), '2wikimultihopqa': (70.3, 78.3), 'musique': (42.2, 53.8)}),
    ('ConRAG', 'Gemma-4-31B', '2026-05', 'not checked against the paper table',
     {'hotpotqa': (61.5, 74.3), '2wikimultihopqa': (69.9, 78.2), 'musique': (40.4, 53.1)}),
    ('HyperProve', 'GPT-OSS-20B', '2026-09', '', {'hotpotqa': (63.9, 76.6), '2wikimultihopqa': (69.7, 80.0), 'musique': (42.8, 59.2)}),
    ('G3RAG (Geometric Gain Graph)', 'not stated', 'arXiv 2609.19622', 'F1 only',
     {'hotpotqa': ('', 78.00), '2wikimultihopqa': ('', 69.20), 'musique': ('', 52.28)}),
    ('HyperReCo', 'GPT-4o-mini', 'arXiv 2609.32327', 'trained hypergraph retriever; HotpotQA corpus of 9,221 passages',
     {'hotpotqa': (60.7, 74.0), '2wikimultihopqa': (72.2, 78.7), 'musique': (39.9, 51.5)}),
    ('SG2RAG', 'Llama-3.3-70B', 'arXiv 2610.15285', 'v1 withdrawn on 2026-11-17',
     {'hotpotqa': (57.70, 71.74), '2wikimultihopqa': (61.00, 69.08), 'musique': (32.10, 45.22)}),
    ('HyperAddress', 'Qwen3.5-27B', 'arXiv 2610.35412', 'reports the last 800 questions only',
     {'hotpotqa': (67.1, 80.1), '2wikimultihopqa': (66.8, 75.1), 'musique': (46.5, 59.1)}),
]


def write(out, name, feeds, note, header, rows):
    with (out / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    FILES.append((name, feeds, note, len(rows)))


def triple(value):
    return list(value) if value else ['', '', '']


def scores(result):
    """n, F1 with interval, EM, calls and tokens per question of one evaluated configuration."""
    value = base.summary(result)
    return [value['n']] + value['f1'] + [value['em'], value['calls'], value['tokens']]


def stored(value):
    """The same seven columns from a row of paper/results.json (calls and tokens when it has them)."""
    return [value['n']] + value['f1'] + [value['em'], value.get('calls', ''), value.get('tokens', '')]


def paired(left, right):
    """Paired F1 difference left minus right over the shared questions, with interval."""
    if left is None or right is None:
        return ['', '', '']
    shared = [key for key in left['keys'] if key in right['f1']]
    return base.interval([left['f1'][key] - right['f1'][key] for key in shared]) if shared else ['', '', '']


def answers_of(bench, names, key):
    return [bench.runs[name][1][key].get('reader_clean') or '' for name in names]


def main_results(out, results, benches, final):
    rows = []
    for (split, dataset), bench in benches.items():
        main = results['splits'][split][dataset]['main']
        runs = results.get('baseline_runs', {}).get(split, {}).get(dataset, {})
        cost = results.get('cost', {}).get(dataset, {}) if split == 'test' else {}
        full = bench.evaluate(final)

        def add(block, system, value, calls, tokens, minus_final):
            rows.append([split, NAMES[dataset], block, system, 'Qwen3.5-9B', value['n']] + value['f1'] + [value['em'], calls, tokens]
                        + triple(minus_final) + ['this work'])

        if 'a0' in runs:
            add('baseline, same reader', 'dense read: top 5 passages of NV-Embed-v2', runs['a0'], main['dense']['calls'],
                main['dense']['tokens'], runs['a0']['minus_final'])
        for tag, system, name in (('b3_ircot', 'IRCoT', 'IRCoT, same reader'),
                                  ('b2_hipporag2', 'HippoRAG 2 (official code)', 'HippoRAG 2, same reader')):
            if tag in runs:
                used = cost.get(name, {})
                add('baseline, same reader', system, runs[tag], used.get('calls_per_question', ''), used.get('tokens_per_question', ''),
                    runs[tag]['minus_final'])
        for key, system in (('fixed', 'chain A1 (one execution)'),
                            ('final', 'final: dense read + A1 + 2 memory plans, calibrated support'),
                            ('on_demand', 'final, expanded on demand')):
            add('ours', system, main[key], main[key]['calls'], main[key]['tokens'], main[key]['minus_final'])
        for label, use in (('A1 + 2 samples (T=0.7), calibrated support', ['fixed'] + T07[:2]),
                           ('A1 + 3 samples (T=0.7), calibrated support', ['fixed'] + T07)):
            if bench.has(use):
                result = bench.evaluate(use)
                value = base.summary(result)
                add('ours, samples instead of memory plans', label, value, value['calls'], value['tokens'], paired(result, full))
    for system, reader, source, note, cells in PUBLISHED:
        for dataset, (em, f1) in cells.items():
            rows.append(['test', NAMES[dataset], 'published, other reader', system, reader, 1000, f1, '', '', em, '', '', '', '', '',
                         source + ('; ' + note if note else '')])
    write(out, 'T1_main_results.csv', '表 1 主结果', '同读者基线、我们的各配置、已发表结果（读者不同，只作参照）。minus_final 是该系统减最终方法的配对差',
          ['split', 'dataset', 'block', 'system', 'reader'] + HEAD + ['minus_final', 'minus_final_lo', 'minus_final_hi', 'source'], rows)


def merge_rules(out, benches, final):
    sets = [('dense read + A1 + 2 memory plans (final)', final), ('A1 + 2 memory plans', ['fixed', 'memory1', 'memory2']),
            ('A1 + 2 samples T=0.7', ['fixed'] + T07[:2]), ('A1 + 3 samples T=0.7', ['fixed'] + T07),
            ('A1 + 2 samples T=0.3', ['fixed'] + T03)]
    rules = [('support', 'calibrated support: noisy-OR of reliabilities'), ('vote', 'majority vote'),
             ('reliable', 'most reliable execution, no pooling'), ('oracle', 'oracle: best execution (reads the gold answer)')]
    rows = []
    for (split, dataset), bench in benches.items():
        for label, use in sets:
            if not bench.has(use):
                continue
            support = bench.evaluate(use)
            options = [(name, bench.evaluate(use, rule=rule)) for rule, name in rules]
            if use == final:
                options.insert(3, ('LLM selector: the reader picks among the distinct answers', bench.selected(final)))
            rows += [[split, NAMES[dataset], label, len(use), name] + scores(result) + paired(support, result)
                     for name, result in options if result is not None]
    write(out, 'T2_merge_rules.csv', '表 2 合并规则', '同一组执行，用不同规则选答案。support_minus_rule 是校准支持度减该规则的配对差',
          ['split', 'dataset', 'executions', 'number_of_executions', 'rule'] + HEAD
          + ['support_minus_rule', 'support_minus_rule_lo', 'support_minus_rule_hi'], rows)


def diversity(out, benches, final):
    configurations = [('A1 alone', ['fixed']), ('A1 + 2 memory plans', ['fixed', 'memory1', 'memory2']),
                      ('A1 + 2 samples T=0.7', ['fixed'] + T07[:2]), ('A1 + 2 samples T=0.3', ['fixed'] + T03),
                      ('A1 + 3 samples T=0.7', ['fixed'] + T07),
                      ('dense read + A1 + 2 memory plans (final)', final), ('dense read + A1 + 2 samples T=0.7', ['dense', 'fixed'] + T07[:2]),
                      ('dense read + A1 + 2 samples T=0.3', ['dense', 'fixed'] + T03),
                      ('dense read + A1 + 3 samples T=0.7', ['dense', 'fixed'] + T07),
                      ('everything: dense read + A1 + 2 memory plans + 5 samples', final + T07 + T03)]
    rows = []
    for (split, dataset), bench in benches.items():
        full, fixed = bench.evaluate(final), bench.evaluate(['fixed'])
        for label, use in configurations:
            if bench.has(use):
                result = bench.evaluate(use)
                rows.append([split, NAMES[dataset], label, sum(name != 'dense' for name in use), 'dense' in use] + scores(result)
                            + paired(result, full) + paired(result, fixed))
    write(out, 'T3_diversity_source.csv', '表 3 多样性来源', '执行来自规划记忆还是来自重复采样；都按校准支持度合并。调用次数相同的行可以直接比',
          ['split', 'dataset', 'configuration', 'chain_executions', 'with_dense_read'] + HEAD
          + ['minus_final', 'minus_final_lo', 'minus_final_hi', 'minus_a1', 'minus_a1_lo', 'minus_a1_hi'], rows)


def executions_curve(out, benches):
    rows = []
    for (split, dataset), bench in benches.items():
        pools = [('memory plans', ['memory1', 'memory2']), ('samples T=0.7', T07), ('samples T=0.3', T03),
                 ('memory plans and samples together', ['memory1', 'memory2'] + T07 + T03)]
        for label, pool in pools:
            pool = [name for name in pool if name in bench.runs]
            if not pool or (label.endswith('together') and len(pool) <= 3):
                continue
            keys = [key for key in bench.ids if all(key in part for name in ['fixed'] + pool for part in bench.runs[name])]
            for count in range(len(pool) + 1):
                sets = [['fixed'] + list(extra) for extra in itertools.combinations(pool, count)]
                if len(sets) > 12:  # a fixed random dozen of the subsets is enough for the mean
                    sets = [sets[index] for index in np.random.default_rng(0).permutation(len(sets))[:12]]
                for rule in ('support', 'vote'):
                    found = [bench.evaluate(use, rule=rule) for use in sets]
                    f1 = np.mean([[result['f1'][key] for key in keys] for result in found], axis=0)
                    calls = float(np.mean([[result['calls'][key] for key in keys] for result in found]))
                    rows.append([split, NAMES[dataset], label, rule, count + 1, len(sets), len(keys)] + base.interval(f1) + [round(calls, 2)])
    write(out, 'F2_f1_vs_executions.csv', '图 2 F1 随执行次数变化', '第 1 次执行是贪心的 A1，其余从该来源里取；同一次数下对所有取法（最多 12 种）取平均。两条线：校准支持度、多数票',
          ['split', 'dataset', 'source_of_extra_executions', 'rule', 'chain_executions', 'subsets_averaged', 'n', 'f1', 'f1_lo', 'f1_hi',
           'llm_calls_per_question'], rows)


def bins(p, y, count=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    rows, error = [], 0.0
    for index in range(count):
        low, high = index / count, (index + 1) / count
        mask = (p >= low) & ((p < high) if index < count - 1 else (p <= 1.0))
        if mask.any():
            rows.append([round(low, 2), round(high, 2), int(mask.sum()), round(100 * float(p[mask].mean()), 2),
                         round(100 * float(y[mask].mean()), 2)])
            error += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return rows, round(100 * float(error), 2)


def calibration(out, benches, final):
    detail, overall = [], []
    for (split, dataset), bench in benches.items():
        chains = [name for name in final if name != 'dense']
        full = bench.evaluate(final)

        def single(names, keys):
            return ([bench.reliability(name, key) for name in names for key in keys],
                    [bench.runs[name][0][key]['f1'] >= RIGHT for name in names for key in keys])

        def pooled(result):
            keys = [key for key in result['keys'] if result['support'][key] is not None]
            return [result['support'][key] for key in keys], [result['f1'][key] >= RIGHT for key in keys]

        quantities = [('reliability of one chain execution (A1 and the memory plans)', *single(chains, full['keys'])),
                      ('support of the merged answer (final)', *pooled(full))]
        samples = [name for name in T07 if name in bench.runs]
        if samples:
            merged = bench.evaluate(['fixed'] + samples)
            quantities += [('reliability of one sampled chain execution (T=0.7)', *single(samples, merged['keys'])),
                           ('support of the merged answer (A1 + %d samples T=0.7)' % len(samples), *pooled(merged))]
        for label, p, y in quantities:
            table, error = bins(p, y)
            detail += [[split, NAMES[dataset], label] + row for row in table]
            overall.append([split, NAMES[dataset], NAMES[bench.source], label, len(p), error, base.auc(p, y),
                            round(float(np.mean((np.asarray(p) - np.asarray(y, float)) ** 2)), 4), round(100 * float(np.mean(y)), 2)])
    write(out, 'F3_calibration_bins.csv', '图 3 校准图（可靠度图）', '按预测值分 10 个等宽区间：区间内的平均预测值和答对比例（F1 至少 0.5 算对）',
          ['split', 'dataset', 'quantity', 'bin_low', 'bin_high', 'n', 'mean_predicted_percent', 'share_right_percent'], detail)
    write(out, 'F3_calibration_summary.csv', '图 3 的图注，表 4 的校准列', 'ECE（10 个区间）、AUC、Brier 分数、总体答对比例；reliability_model_from 是可靠度模型的训练数据集',
          ['split', 'dataset', 'reliability_model_from', 'quantity', 'n', 'ece_percent', 'auc', 'brier', 'share_right_percent'], overall)


def adaptive(bench, order, start, threshold):
    """Run the first `start` executions, then one more at a time while the best answer's support is below the threshold."""
    keys = [key for key in bench.ids if all(key in part for name in order for part in bench.runs[name])]
    f1, calls, tokens, used = [], [], [], []
    for key in keys:
        count = start
        while True:
            names = order[:count]
            index, support, _ = merge(answers_of(bench, names, key), [bench.reliability(name, key) for name in names])
            if support >= threshold or count == len(order):
                break
            count += 1
        usage = [bench.runs[name][1][key]['usage'] for name in names]
        f1.append(bench.runs[names[index]][0][key]['f1'])
        calls.append(sum(item['calls'] for item in usage))
        tokens.append(sum(item['prompt_tokens'] + item['completion_tokens'] for item in usage))
        used.append(count)
    return [len(keys)] + base.interval(f1) + [round(float(np.mean(calls)), 2), round(float(np.mean(tokens))), round(float(np.mean(used)), 2),
                                             round(100 * float(np.mean(np.array(used) > start)), 1)]


def cost_accuracy(out, results, benches, final):
    rows = []
    for (split, dataset), bench in benches.items():
        runs = results.get('baseline_runs', {}).get(split, {}).get(dataset, {})
        cost = results.get('cost', {}).get(dataset, {}) if split == 'test' else {}
        points = [('dense read', ['dense'], False), ('chain A1', ['fixed'], False), ('final', final, False),
                  ('final, expanded when dense read and A1 disagree', final, True), ('A1 + 1 sample T=0.7', ['fixed'] + T07[:1], False),
                  ('A1 + 2 samples T=0.7', ['fixed'] + T07[:2], False), ('A1 + 3 samples T=0.7', ['fixed'] + T07, False)]
        for label, use, lazy in points:
            if bench.has(use):
                value = base.summary(bench.evaluate(use, on_demand=lazy))
                rows.append([split, NAMES[dataset], 'point', label, ''] + stored(value) + ['', ''])
        for tag, label, name in (('b3_ircot', 'IRCoT, same reader', 'IRCoT, same reader'),
                                 ('b2_hipporag2', 'HippoRAG 2, same reader (indexing cost not included)', 'HippoRAG 2, same reader')):
            if tag in runs and name in cost:
                value = dict(runs[tag], calls=round(cost[name]['calls_per_question'], 2), tokens=round(cost[name]['tokens_per_question']))
                rows.append([split, NAMES[dataset], 'point', label, ''] + stored(value) + ['', ''])
        curves = [('adaptive: dense read and A1, then memory plans while support is below the threshold', final, 2),
                  ('adaptive: A1, then samples T=0.7 while support is below the threshold', ['fixed'] + T07, 1)]
        for label, order, start in curves:
            if bench.has(order):
                for threshold in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01):
                    value = adaptive(bench, order, start, threshold)
                    rows.append([split, NAMES[dataset], 'curve', label, threshold] + value[:4] + ['', value[4], value[5], value[6], value[7]])
    write(out, 'F4_cost_accuracy.csv', '图 4 成本和准确率', 'point 是固定配置；curve 是按支持度阈值决定要不要再执行一次（阈值 1.01 等于全部执行）',
          ['split', 'dataset', 'kind', 'system', 'support_threshold'] + HEAD + ['executions_used', 'questions_expanded_percent'], rows)


def reliability_model(out, results):
    features, training, transfer = [], [], []
    for split, data in results['splits'].items():
        for dataset, entry in data.items():
            source = NAMES[entry['support_model_from']]
            for label, value in entry['support_features'].items():
                if label != 'refit_weight_gap':
                    features.append([split, NAMES[dataset], source, label] + stored(value)[:5] + triple(value['minus_final']) + [value['auc']])
            for label, value in entry.get('reliability_training', {}).items():
                training.append([split, NAMES[dataset], source, label] + stored(value)[:5] + triple(value['minus_final']))
            for name, value in entry['transfer'].items():
                transfer.append([split, NAMES[dataset], NAMES[name], name == entry['support_model_from']] + stored(value)[:5]
                                + triple(value['minus_final']))
    head = ['n', 'f1', 'f1_lo', 'f1_hi', 'em', 'minus_final', 'minus_final_lo', 'minus_final_hi']
    write(out, 'T4a_reliability_features.csv', '表 4 可靠度模型：特征', '去掉一组特征后重新拟合，再合并四条推导。auc 是单条回路可靠度区分对错的 AUC',
          ['split', 'dataset', 'reliability_model_from', 'features'] + head + ['auc_of_chain_reliability'], features)
    write(out, 'T4b_reliability_training_size.csv', '表 4 可靠度模型：训练量（也可画成图）', '用 100 到 1,000 道训练题重新拟合，以及三个多跳数据集合起来拟合一个模型',
          ['split', 'dataset', 'reliability_model_from', 'training'] + head, training)
    write(out, 'T4c_reliability_transfer.csv', '表 4 可靠度模型：跨数据集', '用别的数据集上拟合的可靠度模型合并',
          ['split', 'dataset', 'reliability_model_fitted_on', 'used_in_final'] + head, transfer)
    group_of = {name: group for group, names in base.GROUPS.items() for name in names}
    rows = []
    for dataset in MULTIHOP:
        path = ROOT / 'configs/support' / (dataset + '_chain_v1.json')
        if path.exists():
            model = json.loads(path.read_text())
            rows += [[NAMES[dataset], name, group_of.get(name, 'intercept'), round(weight, 4), round(mean, 4), round(scale, 4)]
                     for name, weight, mean, scale in zip(model['names'], model['weights'], model['mean'], model['scale'])]
            rows.append([NAMES[dataset], 'dense_prior', 'reliability of the dense read (share right in training)', round(model['dense_prior'], 4), '', ''])
    write(out, 'T4d_reliability_weights.csv', '附录：可靠度模型的权重（可画成条形图）', '逻辑回归在标准化特征上的权重，以及标准化用的均值和标准差',
          ['fitted_on', 'feature', 'group', 'weight', 'mean', 'scale'], rows)


def variants(out, results):
    kinds = {'cand3': ('binding', 'three candidates per hop, first taken'),
             'cand3_s1500_bt': ('binding', 'three candidates per hop + learned hop scorer + backtracking'),
             'knn_single': ('planning', 'plan examples from gold decompositions of training questions'),
             'hyper_single': ('retrieval', 'hypergraph retrieval inside the same chain'),
             'no final reader': ('reader', 'no final reader: the last hop answers'),
             'ctx_notes': ('reader', 'final reader sees the hop notes only'),
             'ctx_passages': ('reader', 'final reader sees the passages only'),
             'khop4': ('retrieval', 'four passages per hop instead of eight')}
    rows, graph = [], []
    for split, data in results['splits'].items():
        for dataset, entry in data.items():
            runs = results.get('baseline_runs', {}).get(split, {}).get(dataset, {})
            found = [(tag, runs[tag], runs[tag]['minus_single']) for tag in ('cand3', 'cand3_s1500_bt', 'knn_single', 'hyper_single') if tag in runs]
            found += [(tag, value, value.get('minus_fixed')) for tag, value in entry.get('chain_design', {}).items()]
            for tag, value, minus_a1 in found:
                rows.append([split, NAMES[dataset], kinds[tag][0], kinds[tag][1]] + stored(value)[:5] + triple(minus_a1)
                            + triple(value.get('minus_final')))
            if 'hypergraph' in entry:
                value = entry['hypergraph']
                graph.append([split, NAMES[dataset]] + value['hyper_minus_fixed'] + value['preregistered_minus_final'] + [value['calls_added']]
                             + value['evidence_recall'] + value['evidence_all'] + [value['only_hyper_all'], value['only_dense_all'],
                                                                                    value['recall_differs'], value['same_answer']])
    write(out, 'T5_chain_variants.csv', '表 5 没有带来增益的设计，以及回路自身的设计选择', '每一行是把 A1 改动一处后的单次执行，和 A1 配对比较',
          ['split', 'dataset', 'part', 'variant', 'n', 'f1', 'f1_lo', 'f1_hi', 'em', 'minus_a1', 'minus_a1_lo', 'minus_a1_hi',
           'minus_final', 'minus_final_lo', 'minus_final_hi'], rows)
    write(out, 'T5b_hypergraph_retrieval.csv', '表 5 的补充：超图检索回路和稠密检索回路', '同一条回路，只换每一跳的检索；以及把它加进最终方法的效果',
          ['split', 'dataset', 'hyper_minus_a1', 'lo', 'hi', 'added_to_final', 'added_lo', 'added_hi', 'calls_added',
           'evidence_recall_dense', 'evidence_recall_hyper', 'all_evidence_dense', 'all_evidence_hyper', 'only_hyper_finds_all_percent',
           'only_dense_finds_all_percent', 'recall_differs_percent', 'same_final_answer_percent'], graph)


def errors(out, results, benches, final):
    rows = []
    for split, data in results['splits'].items():
        for dataset, entry in data.items():
            rows += [[split, NAMES[dataset], stratum, value['n'], value['dense'], value['fixed'], value['final'], value['oracle'],
                      value['evidence_all']] for stratum, value in entry['breakdown'].items()]
    write(out, 'T6a_breakdown_by_type.csv', '表 6 / 图 5 按题型或跳数', '各题型上的 F1，以及 A1 找齐全部证据段落的比例',
          ['split', 'dataset', 'type', 'n', 'dense_read_f1', 'chain_a1_f1', 'final_f1', 'oracle_of_four_f1', 'a1_all_evidence_found_percent'], rows)
    rows = []
    for (split, dataset), bench in benches.items():
        full, oracle = bench.evaluate(final), bench.evaluate(final, rule='oracle')
        records, scored = bench.runs['fixed'][1], bench.runs['fixed'][0]
        strata = ['all'] + sorted({str(records[key].get('stratum')) for key in full['keys']} - {'all'})
        for stratum in strata:
            keys = [key for key in full['keys'] if stratum == 'all' or str(records[key].get('stratum')) == stratum]
            groups = {'chain A1': {'right': 0, 'wrong: all evidence passages were in front of the reader': 0,
                                   'wrong: plan finished but evidence missing': 0, 'wrong: plan could not be finished': 0},
                      'final': {'right': 0, 'wrong: another execution had the right answer': 0, 'wrong: no execution was right': 0}}
            for key in keys:
                record = records[key]
                if scored[key]['f1'] >= RIGHT:
                    groups['chain A1']['right'] += 1
                elif record.get('evidence_all_supporting'):
                    groups['chain A1']['wrong: all evidence passages were in front of the reader'] += 1
                elif record.get('complete'):
                    groups['chain A1']['wrong: plan finished but evidence missing'] += 1
                else:
                    groups['chain A1']['wrong: plan could not be finished'] += 1
                if full['f1'][key] >= RIGHT:
                    groups['final']['right'] += 1
                elif oracle['f1'][key] >= RIGHT:
                    groups['final']['wrong: another execution had the right answer'] += 1
                else:
                    groups['final']['wrong: no execution was right'] += 1
            rows += [[split, NAMES[dataset], stratum, system, label, count, round(100 * count / max(1, len(keys)), 1)]
                     for system, parts in groups.items() for label, count in parts.items()]
    write(out, 'T6b_error_categories.csv', '表 6 错误归类', 'A1 的错误按证据是否找齐、规划是否走完分类；最终方法的错误按“有没有一次执行答对”分类（F1 至少 0.5 算对）',
          ['split', 'dataset', 'type', 'system', 'category', 'questions', 'percent'], rows)


def cost(out, results):
    rows = [[NAMES[dataset], system, value['index_tokens'], value['preparation_tokens'], round(value['calls_per_question'], 2),
             round(value['tokens_per_question']), value['index_tokens'] + 1000 * round(value['tokens_per_question'])]
            for dataset, systems in results.get('cost', {}).items() for system, value in systems.items()]
    write(out, 'T7_cost.csv', '表 7 成本', '测试集。建索引的 LLM 词元、一次性准备（拟合可靠度模型和填规划记忆的训练集运行）、每题调用次数和词元',
          ['dataset', 'system', 'index_tokens', 'preparation_tokens_once', 'llm_calls_per_question', 'tokens_per_question',
           'index_plus_1000_questions_tokens'], rows)
    path = ROOT / 'runs/queue/state.json'
    if path.exists():
        state = json.loads(path.read_text())
        state = state.get('jobs', state)
        rows = [[name, row.get('status'), round(row['seconds'] / 60, 1), row.get('finished')] for name, row in state.items()
                if isinstance(row, dict) and row.get('seconds')]
        write(out, 'T7b_wallclock.csv', '附录：运行时间', '队列里每个任务的墙钟时间（单张显卡；回路类任务 12 个并发；后跑的任务会命中调用缓存，时间偏短）',
              ['job', 'status', 'minutes', 'finished_utc'], rows)


def statistics(out, frozen, benches):
    rows = []
    for dataset in frozen['datasets']:
        samples = load_test_samples(ROOT, dataset)
        corpus = json.loads((ROOT / 'data/hipporag2' / (dataset + '_corpus.json')).read_text(encoding='utf-8'))
        rows += [[NAMES[dataset], 'test questions', len(samples)], [NAMES[dataset], 'corpus passages', len(corpus)],
                 [NAMES[dataset], 'gold passages per question (mean)', round(float(np.mean([len(item.gold_documents) for item in samples])), 2)],
                 [NAMES[dataset], 'question words (mean)', round(float(np.mean([len(item.question.split()) for item in samples])), 1)]]
        kinds = sorted({str(item.stratum) for item in samples})
        rows += [[NAMES[dataset], 'test questions of type ' + kind, sum(str(item.stratum) == kind for item in samples)] for kind in kinds]
        if ('dev-A', dataset) in benches:
            rows.append([NAMES[dataset], 'dev-A questions', len(benches[('dev-A', dataset)].ids)])
        source = frozen.get('transfer', {}).get(dataset, dataset)
        rows.append([NAMES[dataset], 'training questions behind the reliability model and plan memories', '1500 of ' + NAMES[source]])
    write(out, 'A1_dataset_statistics.csv', '附录：数据集统计（正文可放小表）', '测试子集、语料大小、题型分布', ['dataset', 'quantity', 'value'], rows)


def agreement(out, results, benches):
    rows = [[split, NAMES[dataset], name, value] for split, data in results['splits'].items() for dataset, entry in data.items()
            for name, value in entry.get('agreement', {}).items()]
    write(out, 'A3_agreement.csv', '附录：四条推导是否一致', '一致的比例，以及一致和不一致时的 F1', ['split', 'dataset', 'quantity', 'value'], rows)
    rows = []
    for (split, dataset), bench in benches.items():
        names = [name for name in ['dense', 'fixed', 'memory1', 'memory2'] + T07 + T03 if name in bench.runs]
        for left, right in itertools.combinations(names, 2):
            keys = [key for key in bench.ids if all(key in part for name in (left, right) for part in bench.runs[name])]
            a = np.array([bench.runs[left][0][key]['f1'] >= RIGHT for key in keys])
            b = np.array([bench.runs[right][0][key]['f1'] >= RIGHT for key in keys])
            same = np.mean([same_answer(*answers_of(bench, (left, right), key)) for key in keys])
            rows.append([split, NAMES[dataset], left, right, len(keys), round(100 * float(same), 1)] + [
                round(100 * float(part.mean()), 1) for part in (a & b, a & ~b, ~a & b, ~a & ~b)])
    write(out, 'A4_pairwise_complementarity.csv', '图：执行之间的互补性（可画成热力图），支撑“多样性来源”一节',
          '两次执行答案相同的比例，以及都对、只有一方对、都错的比例。fixed 是 A1，memory 是记忆规划，sc07/sc03 是温度 0.7/0.3 的采样',
          ['split', 'dataset', 'execution_a', 'execution_b', 'n', 'same_answer_percent', 'both_right_percent', 'only_a_right_percent',
           'only_b_right_percent', 'neither_right_percent'], rows)
    rows = [[split, NAMES[dataset], label, value['n'], value['a1'], value['memory2']] + triple(value['memory2_minus_a1'])
            for split, data in results['splits'].items() for dataset, entry in data.items()
            for label, value in entry.get('plan_memory_similarity', {}).items()]
    write(out, 'A5_plan_memory_similarity.csv', '附录：规划记忆的增益和训练题相似度', '按问题与记忆里最近训练题的余弦相似度分段',
          ['split', 'dataset', 'similarity_to_nearest_training_question', 'n', 'chain_a1_f1', 'memory_plan_2_f1', 'memory_minus_a1',
           'memory_minus_a1_lo', 'memory_minus_a1_hi'], rows)


def selective(out, benches, final):
    rows = []
    for (split, dataset), bench in benches.items():
        full, fixed = bench.evaluate(final), bench.evaluate(['fixed'])
        systems = [('final: support of the merged answer', full, {key: full['support'][key] or 0.0 for key in full['keys']}),
                   ('chain A1: reliability of the execution', fixed, {key: bench.reliability('fixed', key) for key in fixed['keys']})]
        if bench.has(['fixed'] + T07):
            merged = bench.evaluate(['fixed'] + T07)
            systems.append(('A1 + 3 samples T=0.7: support of the merged answer', merged, {key: merged['support'][key] or 0.0 for key in merged['keys']}))
        for label, result, confidence in systems:
            order = sorted(result['keys'], key=lambda key: -confidence[key])
            for coverage in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                kept = order[:max(1, round(coverage * len(order)))]
                rows.append([split, NAMES[dataset], label, coverage, len(kept), round(100 * float(np.mean([result['f1'][key] for key in kept])), 2),
                             round(100 * float(np.mean([result['em'][key] for key in kept])), 2),
                             round(float(min(confidence[key] for key in kept)), 4)])
    write(out, 'A6_risk_coverage.csv', '图：只回答最有把握的一部分题（选择性回答）', '按支持度从高到低保留一部分题，看保留部分的 F1 和 EM',
          ['split', 'dataset', 'confidence', 'coverage', 'questions_answered', 'f1', 'em', 'lowest_confidence_kept'], rows)


def retrieval(out, benches):
    rows = []
    labels = {'dense': 'dense read: top 5 passages of NV-Embed-v2', 'fixed': 'chain A1: passages given to the final reader',
              'memory1': 'memory plan 1', 'memory2': 'memory plan 2', 'hyper': 'hypergraph chain (removed)'}
    for (split, dataset), bench in benches.items():
        for name, label in labels.items():
            if name in bench.runs:
                records = [bench.runs[name][1][key] for key in bench.ids if key in bench.runs[name][1]]
                rows.append([split, NAMES[dataset], label, len(records), round(100 * float(np.mean([row.get('evidence_recall', 0) for row in records])), 1),
                             round(100 * float(np.mean([bool(row.get('evidence_all_supporting')) for row in records])), 1)])
        for tag, label in (('b3_ircot', 'IRCoT, same reader: top 5'), ('b2_hipporag2', 'HippoRAG 2, same reader: top 5')):
            path = ROOT / 'runs/chain' / split / dataset / tag / 'summary.json'
            if path.exists():
                value = json.loads(path.read_text())
                rows.append([split, NAMES[dataset], label, value.get('scored'), value.get('recall_at_5'), value.get('all_supporting_at_5', '')])
    write(out, 'A7_retrieval.csv', '表：检索质量（正文小表或附录）', '给最终阅读的 5 段里，标准证据段落的召回，以及全部找齐的比例',
          ['split', 'dataset', 'system', 'n', 'evidence_recall_percent', 'all_evidence_found_percent'], rows)


def cases(out, benches, final, per_kind=8):
    count = 0
    with (out / 'cases.jsonl').open('w', encoding='utf-8') as stream:
        for (split, dataset), bench in benches.items():
            if split != 'test':
                continue
            gold = {example.id: example for example in base.load_samples_of(bench)}
            full, vote, oracle = bench.evaluate(final), bench.evaluate(final, rule='vote'), bench.evaluate(final, rule='oracle')
            a1 = bench.runs['fixed'][0]
            by_support = sorted(full['keys'], key=lambda key: -(full['support'][key] or 0.0))
            kinds = [('support corrects A1', [key for key in by_support if full['f1'][key] >= RIGHT > a1[key]['f1']]),
                     ('support overrides a right A1', [key for key in by_support if a1[key]['f1'] >= RIGHT > full['f1'][key]]),
                     ('support right where the majority vote is wrong', [key for key in by_support if full['f1'][key] >= RIGHT > vote['f1'][key]]),
                     ('vote right where support is wrong', [key for key in by_support if vote['f1'][key] >= RIGHT > full['f1'][key]]),
                     ('no execution right', [key for key in sorted(full['keys']) if oracle['f1'][key] < RIGHT])]
            for kind, keys in kinds:
                for key in keys[:per_kind]:
                    line = {'dataset': NAMES[dataset], 'kind': kind, 'kinds_total': len(keys), 'id': key,
                            'question': bench.runs['fixed'][1][key]['question'], 'gold_answers': list(gold[key].answers) if key in gold else None,
                            'final': {'from': full['choice'][key], 'answer': answers_of(bench, [full['choice'][key]], key)[0],
                                      'support': full['support'][key], 'f1': full['f1'][key]},
                            'vote': {'from': vote['choice'][key], 'f1': vote['f1'][key]}, 'executions': []}
                    for name in final:
                        record = bench.runs[name][1][key]
                        steps = [{'question': step['question'], **{field: step['candidates'][step['chosen']].get(field)
                                                                   for field in ('answer', 'support', 'quote', 'quote_verbatim')}}
                                 for step in record.get('steps', []) if step.get('candidates') and step.get('chosen') is not None]
                        line['executions'].append({'name': name, 'answer': record.get('reader_clean') or '', 'f1': bench.runs[name][0][key]['f1'],
                                                   'reliability': round(bench.reliability(name, key), 4), 'complete': bool(record.get('complete')),
                                                   'plan': record.get('hops') or [], 'steps': steps})
                    stream.write(json.dumps(line, ensure_ascii=False) + '\n')
                    count += 1
    FILES.append(('cases.jsonl', '图：案例分析（正文 1 个，附录若干）',
                  '按固定规则挑的案例，每类每个数据集最多 %d 个，按支持度从高到低；含每次执行的规划、每一跳的问题、答案和引文' % per_kind, count))


def prompts(out):
    from hyperbind import chain_agent
    from hyperbind.baseline_dense import reader_messages
    import run_llm_selector as selector
    lines = ['# Prompts', '', 'Generated from the code by scripts/paper_data.py. The compare additions are used on every dataset except MuSiQue.', '']

    def block(title, text):
        lines.extend(['## ' + title, '', '```text', text, '```', ''])

    shots = lambda rows: '\n'.join('%s\n  -> %s' % (question, json.dumps({'hops': hops})) for question, hops in rows)
    block('Plan: system prompt', chain_agent.PLAN_SYS)
    block('Plan: addition for comparison questions', chain_agent.PLAN_SYS_COMPARE)
    block('Plan: fixed examples of chain A1', shots(chain_agent.FEW))
    block('Plan: additional fixed examples for comparison questions', shots(chain_agent.FEW_COMPARE))
    block('Plan with memory: the fixed examples are replaced by the nearest stored questions and their plans',
          'memory plan 1: the agent\'s own plans of training questions whose final answer was right\n'
          'memory plan 2: the same, built from the plans of round 1')
    block('Hop: system prompt', chain_agent.HOP_SYS)
    block('Hop: second, lenient try when the first finds nothing', chain_agent.LENIENT_SYS)
    block('Hop: user message', 'Question: <sub-question with earlier answers filled in>\n\nPassages:\n[1] <passage>\n\n[2] <passage>\n...')
    block('Hop with several candidates (variant, not in the final method)', chain_agent.CANDIDATE_SYS % 3)
    messages = reader_messages(ROOT, '<question>', ['<passage 1>', '<passage 2>', '<reasoning notes>'])
    block('Final reader: the reading prompt of HippoRAG 2, unchanged', '\n\n'.join('[%s]\n%s' % (message['role'], message['content']) for message in messages))
    block('Final reader: the reasoning notes appended after the passages',
          'Reasoning notes checked against the passages above\nStep 1: <sub-question> -> <answer>\n'
          'Step 2: <sub-question> -> <answer> (likely, not stated directly)')
    block('Control, LLM selector: system prompt', selector.SYSTEM)
    (out / 'prompts.md').write_text('\n'.join(lines), encoding='utf-8')
    FILES.append(('prompts.md', '附录：提示词', '方法用到的全部提示词，以及“让读者模型挑答案”对照的提示词', len(lines)))


def configuration(out):
    from hyperbind import chain_agent
    folder = out / 'config'
    folder.mkdir(parents=True, exist_ok=True)
    for path in [ROOT / 'configs/final.json', ROOT / 'configs/frozen_v2.json', ROOT / 'configs/splits.json'] + sorted(
            (ROOT / 'configs/support').glob('*.json')) + sorted((ROOT / 'configs/scorers').glob('*.json')):
        if path.exists():
            shutil.copy2(path, folder / path.name)
    details = {'reader_and_planner': 'Qwen3.5-9B served by vLLM, thinking off', 'embedder': 'NV-Embed-v2',
               'generation': {'temperature': 0, 'top_p': 1.0, 'seed': 42, 'max_tokens': {'plan': 400, 'hop': 300, 'final reader': 2048}},
               'chain_defaults': dataclasses.asdict(chain_agent.ChainConfig()),
               'chain_a1': 'run_chain.py --candidates 1 (one answer per hop, eight passages per hop, five passages for the final reader)',
               'reliability_features': FEATURES, 'feature_groups': base.GROUPS,
               'reliability_model': 'logistic regression, L2 1.0, on 1,500 training questions per dataset; label: F1 at least 0.5',
               'merge': 'answers grouped by token-level match; support = 1 - product of (1 - reliability) within a group',
               'on_demand': 'memory plans run only when the dense read and chain A1 give different answers',
               'intervals': '95% bootstrap over questions, 10,000 resamples, seed 0; differences are paired',
               'concurrency': '12 questions in parallel per run, one GPU'}
    (folder / 'implementation.json').write_text(json.dumps(details, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    FILES.append(('config/', '附录：实现细节和复现', '冻结的配置、拟合好的可靠度模型和逐跳打分器、实现细节 implementation.json', len(list(folder.iterdir()))))


def missing_runs(frozen):
    missing = []
    for dataset in frozen['datasets']:
        tags = ['a0', 'single', 'selfmem_answer', 'selfmem2_answer', 'b2_hipporag2', 'sc07_1', 'sc07_2', 'llm_select']
        if dataset in MULTIHOP:
            tags += ['b3_ircot', 'sc07_3', 'sc03_1', 'sc03_2', 'ctx_notes', 'ctx_passages', 'khop4', 'cand3', 'hyper_single']
        missing += ['test / %s / %s' % (NAMES[dataset], tag) for tag in tags
                    if not (ROOT / 'runs/chain/test' / dataset / tag / 'scored.json').exists()]
    missing += ['dev-A / %s / llm_select' % NAMES[dataset] for dataset in MULTIHOP
                if not (ROOT / 'runs/chain/dev-A' / dataset / 'llm_select/scored.json').exists()]
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'paper/data')
    arguments = parser.parse_args()
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((ROOT / 'paper/results.json').read_text(encoding='utf-8'))
    frozen = json.loads((ROOT / 'configs/frozen_v2.json').read_text())
    chosen = json.loads((ROOT / 'configs/final.json').read_text())
    final, transfer = chosen['final'], dict(frozen.get('transfer', {}), **chosen.get('transfer', {}))
    benches = {}
    for split in ('test', 'dev-A'):
        for dataset in frozen['datasets']:
            bench = base.Bench(frozen, transfer, dataset, split)
            if bench.ids and bench.has(final) and bench.model is not None and dataset in results['splits'].get(split, {}):
                benches[(split, dataset)] = bench
    main_results(out, results, benches, final)
    merge_rules(out, benches, final)
    diversity(out, benches, final)
    executions_curve(out, benches)
    calibration(out, benches, final)
    cost_accuracy(out, results, benches, final)
    reliability_model(out, results)
    variants(out, results)
    errors(out, results, benches, final)
    cost(out, results)
    statistics(out, frozen, benches)
    agreement(out, results, benches)
    selective(out, benches, final)
    retrieval(out, benches)
    cases(out, benches, final)
    prompts(out)
    configuration(out)
    missing = missing_runs(frozen)
    lines = ['# 论文数据包', '', '由 `scripts/paper_data.py` 生成，时间 %s UTC。先跑 `scripts/paper_results.py`，再跑本脚本。' % time.strftime(
        '%Y-%m-%d %H:%M', time.gmtime()), '',
        '- F1、EM 都是百分数；区间是按题目自助重采样的 95% 区间（10,000 次），差值都是配对差。',
        '- 每个文件都带 split 列：test 是测试集（HippoRAG 2 的 1,000 题子集），dev-A 是开发集。',
        '- 逐题的原始预测在 `../raw/<数据集>/<split>_predictions.jsonl`，所有数字的机器可读版本在 `../results.json`。',
        '- 图表编号以 `../OUTLINE.md` 为准；文件名开头的 T、F、A 只是文件标识。', '',
        '| 文件 | 用在哪里 | 内容 | 行数 |', '|---|---|---|---|']
    lines += ['| `%s` | %s | %s | %d |' % (name, FEEDS.get(name, feeds), note, count) for name, feeds, note, count in FILES]
    lines += ['', '## 生成时还缺的运行', '']
    lines += ['- ' + item for item in missing] if missing else ['没有缺的运行。']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('%d files written to %s; %d runs missing' % (len(FILES), out, len(missing)))


if __name__ == '__main__':
    main()
