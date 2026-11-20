#!/usr/bin/env python
"""Official results table from the derivation runs of one split (default: test). CPU only.

Which derivations form the headline configuration is read from configs/frozen_v2.json, which was written before any
test run. Every other configuration in that file is reported as an ablation. Missing runs are listed, not guessed.
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

NAMES = {'musique': 'MuSiQue', '2wikimultihopqa': '2Wiki', 'hotpotqa': 'HotpotQA', 'nq_rear': 'NQ', 'popqa': 'PopQA'}


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
    parser.add_argument('--split', default='test')
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/frozen_v2.json')
    parser.add_argument('--out', type=Path)
    arguments = parser.parse_args()
    frozen = json.loads(arguments.config.read_text())
    out = arguments.out or ROOT / ('reports/official_%s_results.md' % arguments.split.replace('-', '_'))
    lines = ['# 正式结果：%s' % arguments.split, '',
             '由 `scripts/test_report.py` 生成。配置来自 `%s`（写于 %s，早于任何测试集运行）。' % (
                 arguments.config.relative_to(ROOT), frozen['written']),
             'F1 和 EM 用上游（HippoRAG 2）的答案提取和计分规则；区间是 1 万次重采样的 95% 区间；配对差只算两组都有的题。', '']
    main_rows, missing = [], []
    for dataset in arguments.datasets or frozen.get('datasets', list(NAMES)):
        runs = {}
        for name, tag in frozen['derivations'].items():
            run = load(arguments.split, dataset, tag)
            if run is None:
                missing.append('%s/%s/%s' % (arguments.split, dataset, tag))
            else:
                runs[name] = run
        if 'fixed' not in runs:
            continue
        source = frozen.get('transfer', {}).get(dataset, dataset)  # datasets without training questions borrow the trained parts
        model_path = ROOT / frozen['support_model'].replace('<dataset>', source)
        model = SupportModel.load(model_path) if model_path.exists() else None
        base = runs['fixed']
        ids = sorted(set(base[0]) & set(base[1]))
        lines += ['## %s（%d 题）' % (NAMES.get(dataset, dataset), len(ids)), '']
        if source != dataset:
            lines += ['迁移设置：这个数据集没有训练题，可靠度模型和规划记忆都来自 %s 的训练题，没有在它上面拟合任何东西。' % NAMES.get(source, source), '']
        if model is None:
            lines += ['没有找到可靠度模型 `%s`，只列单条推导。' % model_path.relative_to(ROOT), '']
        lines += ['| 配置 | F1 [95% 区间] | EM | 每题调用 | 每题 token | 比单条回路 A1 | 比稠密直读 |', '|---|---|---:|---:|---:|---|---|']

        def evaluate(use, uniform=False, on_demand=False):
            if any(name not in runs for name in use) or (len(use) > 1 and model is None):
                return None
            keys = [key for key in ids if all(key in part for name in use for part in runs[name])]
            f1, em, calls, tokens = {}, {}, [], []
            for key in keys:
                names = list(use)
                if on_demand:  # the first two derivations; the rest only when their answers differ
                    pair = [runs[name][1][key]['reader_clean'] or '' for name in names[:2]]
                    if same_answer(pair[0], pair[1]):
                        names = names[:2]
                if len(names) == 1:
                    index = 0
                else:
                    answers = [runs[name][1][key]['reader_clean'] or '' for name in names]
                    reliabilities = [0.5 if uniform else model.reliability(runs[name][1][key], dense=name == 'dense') for name in names]
                    index, _, _ = merge(answers, reliabilities)
                row = runs[names[index]][0][key]
                f1[key], em[key] = row['f1'], row['em']
                usage = [runs[name][1][key]['usage'] for name in names]
                calls.append(sum(item['calls'] for item in usage))
                tokens.append(sum(item['prompt_tokens'] + item['completion_tokens'] for item in usage))
            return {'keys': keys, 'f1': f1, 'em': em, 'calls': float(np.mean(calls)), 'tokens': float(np.mean(tokens))}

        reference = {name: evaluate([name]) for name in ('fixed', 'dense')}

        def row(label, result):
            if result is None:
                return None
            cells = []
            for name in ('fixed', 'dense'):
                other = reference[name]
                shared = [key for key in result['keys'] if other is not None and key in other['f1']]
                cells.append('%+.2f [%.2f, %.2f]' % interval([result['f1'][key] - other['f1'][key] for key in shared])
                             if shared and result is not other else '–')
            mean, low, high = interval(list(result['f1'].values()))
            lines.append('| %s | %.2f [%.2f, %.2f] | %.2f | %.1f | %d | %s | %s |' % (
                label, mean, low, high, 100 * np.mean(list(result['em'].values())), result['calls'], result['tokens'], cells[0], cells[1]))
            return mean, 100 * np.mean(list(result['em'].values())), result['calls']

        summary = {'dense': row('稠密直读', reference['dense']), 'fixed': row('单条回路 A1', reference['fixed'])}
        for name in frozen['derivations']:
            if name not in ('dense', 'fixed') and name in runs:
                row('单条：%s' % frozen['labels'].get(name, name), evaluate([name]))
        values = {}
        for label, use in frozen['configurations'].items():
            result = evaluate(use)
            value = values[label] = row(label, result)
            if label == frozen['headline']:
                summary['headline'] = value
                summary['on_demand'] = row(label + '（按需展开）', evaluate(use, on_demand=True))
                row(label + '（只投票，消融）', evaluate(use, uniform=True))
        fallback = frozen.get('fallback')
        if summary.get('headline') is None and fallback and values.get(fallback) is not None:
            # a derivation of the headline is missing for this dataset: the main table shows the named fallback and says so
            summary['headline'], summary['fallback'] = values[fallback], True
            summary['on_demand'] = row(fallback + '（按需展开）', evaluate(frozen['configurations'][fallback], on_demand=True))
        lines.append('')
        main_rows.append((dataset, summary))
    table = ['## 主表', '', '%s。' % frozen['headline'], '',
             '| 数据集 | 稠密直读 F1 / EM | 单条回路 A1 F1 / EM | 正式配置 F1 / EM（调用） | 按需展开 F1 / EM（调用） |', '|---|---|---|---|---|']
    for dataset, summary in main_rows:
        cell = lambda value, calls=False: '–' if value is None else ('%.2f / %.2f' % value[:2] + ('（%.1f）' % value[2] if calls else ''))
        name = NAMES.get(dataset, dataset) + ('（超图回路没有结果，此行是去掉超图的配置）' if summary.get('fallback') else '')
        table.append('| %s | %s | %s | %s | %s |' % (name, cell(summary.get('dense')), cell(summary.get('fixed')),
                                                cell(summary.get('headline'), True), cell(summary.get('on_demand'), True)))
    lines[4:4] = table + ['']
    if missing:
        lines += ['## 还没有结果的运行', ''] + ['- `%s`' % name for name in missing] + ['']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
