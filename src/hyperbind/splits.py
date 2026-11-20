"""Create reproducible arena-A splits from original training records only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

from .data import (candidate_documents, dataset_name, get_gold_docs, load_raw_records,
                   normalize_record, official_record, record_id, record_stratum)


DEFAULT_SIZES = {'dev-A': 500, 'dev-B': 500, 'train-S': 2000}
TRAIN_PATHS = {
    'musique': 'data/raw/musique/extracted/data/musique_ans_v1.0_train.jsonl',
    '2wikimultihopqa': 'data/raw/2wikimultihop/extracted/train.json',
    'hotpotqa': 'data/raw/hotpotqa/distractor',
}
EXPECTED_TRAIN_ROWS = {'musique': 19938, '2wikimultihopqa': 167454, 'hotpotqa': 90447}
SOURCE_FILTERS = {'hotpotqa': {'level': 'hard'}}


def largest_remainder(total: int, proportions: Mapping[str, int]) -> dict[str, int]:
    """Hamilton allocation, with lexical stratum ordering for equal remainders."""
    if total < 0 or not proportions or any(value < 0 for value in proportions.values()):
        raise ValueError('Invalid target size or stratum weights')
    denominator = sum(proportions.values())
    if denominator <= 0:
        raise ValueError('Stratum weights must have positive total')
    exact = {key: Fraction(total * value, denominator) for key, value in proportions.items()}
    counts = {key: value.numerator // value.denominator for key, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    return dict(sorted(counts.items()))


def question_key(row: Mapping[str, Any]) -> str:
    value = row.get('question')
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Empty question: {record_id(row)}')
    return ' '.join(value.split()).casefold()


def _seed(seed: int, dataset: str, stratum: str) -> int:
    return int.from_bytes(hashlib.sha256(f'{seed}\0{dataset}\0{stratum}'.encode()).digest(), 'big')


def construct_splits(train_rows: Sequence[dict], test_rows: Sequence[dict], dataset: str,
                     seed: int = 42, sizes: Mapping[str, int] | None = None,
                     source_filters: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[dict, dict]:
    """No answer or supporting-passage label is consulted during selection."""
    dataset = dataset_name(dataset)
    sizes = dict(DEFAULT_SIZES if sizes is None else sizes)
    if not sizes or any(not isinstance(value, int) or value <= 0 for value in sizes.values()):
        raise ValueError('Every split size must be a positive integer')
    test_ids = {record_id(row) for row in test_rows}
    if len(test_ids) != len(test_rows):
        raise ValueError('Test question ids are not unique')
    test_questions = {question_key(row) for row in test_rows}
    proportions = Counter(record_stratum(row, dataset) for row in test_rows)
    allocations = {name: largest_remainder(size, proportions) for name, size in sizes.items()}
    filters = dict((SOURCE_FILTERS if source_filters is None else source_filters).get(dataset, {}))
    if dataset == 'hotpotqa' and filters != {'level': 'hard'}:
        raise ValueError('R1 requires HotpotQA level=hard before type stratification')
    excluded = Counter()
    groups = defaultdict(list)
    seen_ids, seen_questions = set(), set()
    raw_ids = [record_id(row) for row in train_rows]
    filtered_rows = [row for row in train_rows if all(row.get(key) == value for key, value in filters.items())]
    excluded['training_source_filter'] = len(train_rows) - len(filtered_rows)
    if not excluded['training_source_filter']:
        del excluded['training_source_filter']
    for row in sorted(filtered_rows, key=record_id):
        qid, text_key = record_id(row), question_key(row)
        if qid in test_ids:
            excluded['test_id'] += 1
        elif text_key in test_questions:
            excluded['test_question_text'] += 1
        elif qid in seen_ids:
            excluded['duplicate_training_id'] += 1
        elif text_key in seen_questions:
            excluded['duplicate_training_question_text'] += 1
        else:
            seen_ids.add(qid)
            seen_questions.add(text_key)
            groups[record_stratum(row, dataset)].append(row)
    result = {name: [] for name in sizes}
    for stratum in sorted(proportions):
        candidates = groups[stratum]
        required = sum(allocation[stratum] for allocation in allocations.values())
        if len(candidates) < required:
            raise ValueError(f'{dataset}/{stratum}: needs {required}, has {len(candidates)}')
        random.Random(_seed(seed, dataset, stratum)).shuffle(candidates)
        offset = 0
        for name in sizes:
            count = allocations[name][stratum]
            result[name].extend(candidates[offset:offset + count])
            offset += count
    # Randomise saved question order independently of type without changing membership.
    for name, rows in result.items():
        random.Random(_seed(seed, dataset, name)).shuffle(rows)
    id_sets = {name: {record_id(row) for row in rows} for name, rows in result.items()}
    question_sets = {name: {question_key(row) for row in rows} for name, rows in result.items()}
    overlaps = {}
    all_names = list(result) + ['test']
    id_sets['test'], question_sets['test'] = test_ids, test_questions
    for i, left in enumerate(all_names):
        for right in all_names[i + 1:]:
            key = left + '__' + right
            overlaps[key] = {'ids': len(id_sets[left] & id_sets[right]),
                             'question_texts': len(question_sets[left] & question_sets[right])}
            assert overlaps[key] == {'ids': 0, 'question_texts': 0}, (dataset, key, overlaps[key])
    report = {
        'dataset': dataset, 'seed': seed, 'sizes': sizes,
        'stratification': 'MuSiQue id hop prefix' if dataset == 'musique' else 'type',
        'test_strata': dict(sorted(proportions.items())), 'allocated_strata': allocations,
        'actual_strata': {name: dict(sorted(Counter(record_stratum(row, dataset) for row in rows).items()))
                          for name, rows in result.items()},
        'original_train_records': len(train_rows), 'original_train_unique_ids': len(set(raw_ids)),
        'training_source_filter': filters,
        'source_filtered_records': len(filtered_rows),
        'source_filtered_unique_ids': len({record_id(row) for row in filtered_rows}),
        'source_filtered_strata': dict(sorted(Counter(record_stratum(row, dataset) for row in filtered_rows).items())),
        'original_test_records': len(test_rows), 'raw_train_test_id_overlap': len(set(raw_ids) & test_ids),
        'excluded_records': dict(sorted(excluded.items())),
        'pairwise_overlap': overlaps,
        'selection_uses': ['source id', 'question text for duplicate exclusion', 'hop prefix or type']
                          + (['source difficulty metadata filter'] if filters else []),
        'labels_used_for_selection': False,
        'determinism': 'source rows sorted by id; per-dataset/stratum SHA256-derived Random(seed); Hamilton allocation with lexical ties',
    }
    if dataset == 'hotpotqa':
        report['level_statistics'] = {
            'original_train': dict(sorted(Counter(row.get('level', '<missing>') for row in train_rows).items())),
            'filtered_train': dict(sorted(Counter(row.get('level', '<missing>') for row in filtered_rows).items())),
            'test': dict(sorted(Counter(row.get('level', '<missing>') for row in test_rows).items())),
            'splits': {name: dict(sorted(Counter(row.get('level', '<missing>') for row in rows).items()))
                       for name, rows in result.items()},
        }
        if report['level_statistics']['test'] != {'hard': len(test_rows)}:
            raise ValueError('HotpotQA test subset is expected to contain hard questions only')
        for name, levels in report['level_statistics']['splits'].items():
            assert levels == {'hard': sizes[name]}, (name, levels)
    return result, report


def build_corpus(rows: Sequence[dict], dataset: str) -> list[dict[str, str]]:
    unique = {}
    for row in rows:
        documents = candidate_documents(row, dataset)
        contents = {doc.content for doc in documents}
        gold = get_gold_docs([row], dataset)[0]
        if not gold:
            raise ValueError(f'No supporting passage in candidate contexts: {record_id(row)}')
        official = official_record(row)
        if 'supporting_facts' in official:
            expected_titles = {item[0] for item in official['supporting_facts']}
            if not expected_titles.issubset({doc.title for doc in documents}):
                raise ValueError(f'Supporting title missing from candidate contexts: {record_id(row)}')
        if not set(gold).issubset(contents):
            raise ValueError(f'Supporting passage missing from candidate contexts: {record_id(row)}')
        for doc in documents:
            unique.setdefault(doc.key, doc.to_dict())
    return list(unique.values())


def write_json(path: Path, value: Any) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n')
    temporary.replace(path)
    with path.open('rb') as stream:
        sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256}


def _source_files(path: Path) -> list[Path]:
    return sorted(path.glob('*.parquet')) if path.is_dir() else [path]


def archive_existing_splits(root: Path, output_dir: str, destination: Path) -> dict:
    """Preserve the whole old split version before an accepted correction."""
    root = root.resolve()
    destination = (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    if not destination.is_relative_to(root):
        raise ValueError('Split history must remain inside the project root')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite existing split history: {destination}')
    sources = sorted((root / output_dir).rglob('*.json'))
    sources += [root / 'reports/p0_splits.json', root / 'runs/p0_splits/manifest.json']
    if not sources or any(not source.is_file() for source in sources):
        raise FileNotFoundError('The previous complete split version must exist before archival')
    destination.mkdir(parents=True)
    archived = []
    for source in sources:
        relative = source.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        with source.open('rb') as stream:
            source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        with target.open('rb') as stream:
            target_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if source_hash != target_hash:
            raise RuntimeError(f'Split history checksum mismatch: {relative}')
        archived.append({'source_path': str(relative), 'archived_path': str(target.relative_to(root)),
                         'bytes': source.stat().st_size, 'sha256': source_hash})
    manifest = {'status': 'obsolete', 'reason': 'R1: HotpotQA training difficulty did not match hard-only test subset',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'files': archived,
                'unaffected_datasets': ['musique', '2wikimultihopqa'],
                'note': 'Old complete split bundle retained for provenance; only HotpotQA is rebuilt.'}
    write_json(destination / 'obsolete.json', manifest)
    return manifest


def build_arena_a(root: Path, config: Mapping[str, Any], datasets: Sequence[str] | None = None) -> dict:
    selected_datasets = list(TRAIN_PATHS) if datasets is None else [dataset_name(x) for x in datasets]
    report_path = root / 'reports/p0_splits.json'
    # A correction to one dataset must retain the other dataset results and any
    # report fields added by a subsequent stage.
    report = json.loads(report_path.read_text(encoding='utf-8')) if datasets is not None and report_path.exists() else {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'plan_revision': '1.1',
        'scope': 'P0.1 data preparation only; no model, retrieval, QA or training calls',
        'config': dict(config), 'datasets': {},
        'output_format': 'Original source question records are preserved in JSON arrays; hyperbind.data normalises formats at read time',
        'corpus_policy': 'dev_corpus contains dev-A + dev-B candidate passages only; train-S_corpus is separate; test corpus unchanged',
        'question_duplicate_policy': 'In addition to source id exclusion, reject whitespace/case-normalised duplicate question text; no answer labels used',
    }
    report['config'] = dict(config)
    report['updated_utc'] = datetime.now(timezone.utc).isoformat()
    for dataset in selected_datasets:
        free = shutil.disk_usage(root).free
        minimum = int(config.get('minimum_free_gib', 15)) * 1024 ** 3
        if free < minimum:
            raise RuntimeError(f'Insufficient free disk before split preparation: {free} < {minimum}')
        source = root / TRAIN_PATHS[dataset]
        train = load_raw_records(source)
        expected = EXPECTED_TRAIN_ROWS[dataset]
        if len(train) != expected:
            raise ValueError(f'{dataset} train rows: {len(train)} != {expected}')
        test_path = root / 'data/hipporag2' / (dataset + '.json')
        test = load_raw_records(test_path)
        if len(test) != 1000:
            raise ValueError(f'{dataset} test rows: {len(test)} != 1000')
        if dataset == 'musique' and Counter(record_stratum(row, dataset) for row in test) != {'2': 518, '3': 316, '4': 166}:
            raise ValueError('MuSiQue test hop proportions differ from the accepted plan')
        partitions, dataset_report = construct_splits(train, test, dataset, int(config['seed']), config['sizes'],
                                                      config.get('training_source_filters', SOURCE_FILTERS))
        if dataset_report['original_train_unique_ids'] != expected:
            raise ValueError(f'{dataset} source train contains duplicate ids')
        if dataset_report['raw_train_test_id_overlap']:
            raise ValueError(f'{dataset} original train overlaps test ids')
        if dataset == 'hotpotqa':
            if dataset_report['source_filtered_records'] != 15661 or dataset_report['source_filtered_strata'] != {'bridge': 12451, 'comparison': 3210}:
                raise ValueError('HotpotQA hard training pool differs from the R1 review: expected 15661, bridge12451/comparison3210')
        destination = root / config.get('output_dir', 'data/splits/seed42') / dataset
        dataset_report['source'] = {'train': str(source), 'test': str(test_path)}
        dataset_report['source_files'] = []
        for file in _source_files(source) + [test_path]:
            with file.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            dataset_report['source_files'].append({'path': str(file), 'bytes': file.stat().st_size, 'sha256': digest})
        outputs = []
        for name, rows in partitions.items():
            # Exercise the unified schema on selected rows; preserve the source dicts on disk.
            for row in rows:
                normalize_record(row, dataset)
            outputs.append(write_json(destination / (name + '.json'), rows))
        dev_rows = partitions['dev-A'] + partitions['dev-B']
        dev_corpus = build_corpus(dev_rows, dataset)
        train_corpus = build_corpus(partitions['train-S'], dataset)
        outputs.append(write_json(destination / 'dev_corpus.json', dev_corpus))
        outputs.append(write_json(destination / 'train-S_corpus.json', train_corpus))
        dataset_report['corpora'] = {
            'dev': {'questions': len(dev_rows), 'documents': len(dev_corpus), 'deduplicate_by': ['title', 'text']},
            'train-S': {'questions': len(partitions['train-S']), 'documents': len(train_corpus), 'deduplicate_by': ['title', 'text']},
        }
        dataset_report['outputs'] = outputs
        report['datasets'][dataset] = dataset_report
        write_json(root / 'reports/p0_splits.json', report)
        print(json.dumps({'dataset': dataset, 'splits': {key: len(value) for key, value in partitions.items()},
                          'corpus_documents': dataset_report['corpora'], 'status': 'complete'}), flush=True)
        del train, test, partitions, dev_rows, dev_corpus, train_corpus
    write_json(root / 'reports/p0_splits.json', report)
    write_json(root / 'runs/p0_splits/manifest.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=list(TRAIN_PATHS))
    parser.add_argument('--archive-existing-to', type=Path,
                        help='Copy the current complete split version to an obsolete history directory before rebuilding')
    args = parser.parse_args()
    config_path = args.config or args.root / 'configs/splits.json'
    with config_path.open(encoding='utf-8') as stream:
        config = json.load(stream)
    if args.archive_existing_to is not None:
        archive_existing_splits(args.root, config.get('output_dir', 'data/splits/seed42'), args.archive_existing_to)
    build_arena_a(args.root, config, args.datasets)


if __name__ == '__main__':
    main()
