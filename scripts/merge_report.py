#!/usr/bin/env python
"""Report: answers merged across several derivations by estimated support. CPU only; missing runs are skipped.

For each question the derivations' answers are grouped, each derivation gets a reliability from the support model
(configs/support/*.json, fitted on train-S positions 0-1500), and the answer with the largest noisy-OR support wins.
'只投票' gives every derivation the same reliability (plain self-consistency) as the ablation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hyperbind.support import SupportModel, merge  # noqa: E402
from hyperbind.textmatch import same_answer  # noqa: E402

GROUPS = [
    ('train-S 留出 500 题', 'train-S', 'single_val', 'a0',
     {'固定': 'single_val', '标注记忆': 'knn_single_val', '标注记忆2': 'knn_skip6_val', '自建记忆A': 'selfmem_answer_val',
      '自建记忆S': 'selfmem_support_val', '自建记忆A第2轮': 'selfmem2_answer_val', '多候选': 'cand3_val_s1500_bt',
      '多候选取第一个': 'cand3_val'}),
    ('dev-A', 'dev-A', 'single', 'a0',
     {'固定': 'single', '标注记忆': 'knn_single', '标注记忆2': 'knn_skip6', '自建记忆A': 'selfmem_answer',
      '自建记忆S': 'selfmem_support', '自建记忆A第2轮': 'selfmem2_answer', '多候选': 'cand3_s1500_bt',
      '多候选取第一个': 'cand3'}),
]
COMBOS = [['固定', '自建记忆A第2轮'], ['固定', '自建记忆A', '自建记忆A第2轮'], ['固定', '自建记忆A第2轮', '多候选'],
          ['固定', '自建记忆A', '多候选'], ['固定', '自建记忆A', '自建记忆A第2轮', '多候选取第一个'],
          ['固定', '自建记忆A第2轮', '多候选取第一个'], ['固定', '多候选取第一个'],
          ['固定', '自建记忆S'], ['固定', '自建记忆A'], ['固定', '自建记忆S', '自建记忆A'], ['固定', '标注记忆'],
          ['固定', '标注记忆', '标注记忆2'], ['固定', '标注记忆', '自建记忆S'], ['固定', '标注记忆', '多候选'],
          ['固定', '标注记忆', '标注记忆2', '自建记忆S', '自建记忆A', '多候选']]


def interval(values):
    values = np.asarray(values, float)
    generator = np.random.default_rng(0)
    means = values[generator.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return 100 * values.mean(), 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


def load(split, dataset, tag):
    folder = ROOT / 'runs/chain' / split / dataset / tag
    if not (folder / 'scored.json').exists() or not (folder / 'records.jsonl').exists():
        return None
    scored = {row['id']: row for row in json.loads((folder / 'scored.json').read_text())}
    records = {}
    for line in (folder / 'records.jsonl').open():
        record = json.loads(line)
        if 'error' not in record:
            records[record['id']] = record
    return scored, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='musique')
    parser.add_argument('--model', type=Path, default=ROOT / 'configs/support/musique_chain_v1.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/merge_v1_report.md')
    arguments = parser.parse_args()
    model = SupportModel.load(arguments.model)
    lines = ['# 多条推导按支持度合并：验证报告', '',
             '由 `scripts/merge_report.py` 生成。支持度模型只在 train-S 前 1,500 题上拟合；合并时不使用标准答案。'
             '“比参照”是和单条固定示例规划回路（A1）的配对差，区间是 1 万次重采样的 95% 区间。', '']
    for title, split, reference, dense_tag, tags in GROUPS:
        runs = {name: load(split, arguments.dataset, tag) for name, tag in tags.items()}
        runs = {name: run for name, run in runs.items() if run is not None}
        dense = load(split, arguments.dataset, dense_tag)
        base = runs.get('固定')
        if base is None:
            continue
        ids = sorted(set(base[0]) & set(base[1]))
        base_f1 = {key: base[0][key]['f1'] for key in ids}
        strata = sorted({str(base[0][key]['stratum']) for key in ids})
        lines += ['## %s %s' % (arguments.dataset, title), '', '可用的推导：' + '、'.join('%s（`%s`）' % (name, tags[name]) for name in runs) + '。', '',
                  '| 配置 | F1 | 比参照 | ' + ' | '.join('%s 跳' % s for s in strata) + ' | 每题调用 | 上限（事后最优） |',
                  '|---|---:|---|' + '---:|' * (len(strata) + 2)]

        def row(label, use, uniform=False, adaptive=False):
            use = [name for name in use if name == 'dense' or name in runs]
            scores, calls, oracle = {}, [], []
            for key in ids:
                names = list(use)
                if any(key not in part for name in names for part in (dense if name == 'dense' else runs[name])):
                    continue
                if adaptive:  # dense and the first plan; the rest only when these two disagree
                    first = [runs[names[1]][1][key]['reader_clean'] or '', dense[1][key]['reader_clean'] or '']
                    if same_answer(first[0], first[1]):
                        names = names[:2]
                answers = [(dense if name == 'dense' else runs[name])[1][key]['reader_clean'] or '' for name in names]
                rel = [0.5 if uniform else model.reliability((dense if name == 'dense' else runs[name])[1][key], dense=name == 'dense')
                       for name in names]
                index, _, _ = merge(answers, rel)
                f1 = [(dense if name == 'dense' else runs[name])[0][key]['f1'] for name in names]
                scores[key] = f1[index]
                oracle.append(max(f1))
                calls.append(sum((dense if name == 'dense' else runs[name])[1][key]['usage']['calls'] for name in names))
            keys = list(scores)
            diff = '%+.2f [%.2f, %.2f]' % interval([scores[key] - base_f1[key] for key in keys]) if len(use) > 1 or use != ['固定'] else '–'
            by = ['%.1f' % (100 * np.mean([scores[key] for key in keys if str(base[0][key]['stratum']) == s])) for s in strata]
            lines.append('| %s | %.2f | %s | %s | %.1f | %.2f |' % (label, 100 * np.mean(list(scores.values())), diff, ' | '.join(by),
                                                                 np.mean(calls), 100 * np.mean(oracle)))

        for name in runs:
            row('单条：' + name, [name])
        if dense is not None:
            row('单条：稠密直读', ['dense'])
        for combo in COMBOS:
            if all(name in runs for name in combo):
                label = ' + '.join(combo)
                row('合并：' + label, combo)
                row('合并（只投票）：' + label, combo, uniform=True)
                if dense is not None:
                    row('合并：' + label + ' + 稠密直读', combo + ['dense'])
                    row('按需展开：稠密直读 + ' + label, ['dense'] + combo, adaptive=True)
        lines.append('')
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
