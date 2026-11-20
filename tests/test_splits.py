from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from hyperbind.data import record_id
from hyperbind.splits import (archive_existing_splits, build_arena_a, build_corpus,
                             construct_splits, largest_remainder)


def row(index, kind='bridge', question=None):
    return {'_id': f'q{index:04d}', 'question': question or f'Question {index}?', 'type': kind,
            'level': 'hard',
            'answer': 'hidden answer', 'context': [['A', ['First.', ' second.']], ['B', ['Other.']]],
            'supporting_facts': [['A', 0]]}


class StratifiedSplitTests(unittest.TestCase):
    def test_hotpot_filters_hard_before_type_sampling(self):
        train = [row(i, 'bridge' if i % 2 else 'comparison') for i in range(40)]
        for i, item in enumerate(train):
            item['level'] = 'hard' if i < 20 else ('easy' if i % 2 else 'medium')
        test = [row(1000, 'bridge'), row(1001, 'comparison')]
        partitions, report = construct_splits(train, test, 'hotpotqa', 42, {'dev-A': 4, 'dev-B': 4, 'train-S': 8})
        self.assertTrue(all(x['level'] == 'hard' for rows in partitions.values() for x in rows))
        self.assertEqual(report['original_train_records'], 40)
        self.assertEqual(report['source_filtered_records'], 20)
        self.assertEqual(report['source_filtered_strata'], {'bridge': 10, 'comparison': 10})
        self.assertEqual(report['level_statistics']['splits'], {'dev-A': {'hard': 4}, 'dev-B': {'hard': 4}, 'train-S': {'hard': 8}})
        self.assertEqual(report['excluded_records']['training_source_filter'], 20)
        with self.assertRaisesRegex(ValueError, 'level=hard'):
            construct_splits(train, test, 'hotpotqa', source_filters={})

    def test_missing_hard_rows_cannot_be_filled_with_easy(self):
        train = [row(i) for i in range(20)]
        for item in train:
            item['level'] = 'easy'
        with self.assertRaisesRegex(ValueError, 'needs 4, has 0'):
            construct_splits(train, [row(1000)], 'hotpotqa', sizes={'dev-A': 4})

    def test_partial_report_update_retains_unselected_dataset_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'reports').mkdir()
            original = {'created_utc': 'original', 'datasets': {'musique': {'keep': 'unchanged'}},
                        'future_field': {'preserve': True}}
            (root / 'reports/p0_splits.json').write_text(json.dumps(original))
            # Empty requested dataset list exercises report merge without touching any source data.
            result = build_arena_a(root, {'seed': 42, 'sizes': {'dev-A': 4}}, datasets=[])
            self.assertEqual(result['datasets'], original['datasets'])
            self.assertEqual(result['future_field'], original['future_field'])
            self.assertEqual(result['created_utc'], 'original')

    def test_history_keeps_checksums_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ['data/splits/seed42/hotpotqa/dev-A.json', 'reports/p0_splits.json', 'runs/p0_splits/manifest.json']
            for path in paths:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('[1]')
            destination = Path('runs/history/p0_splits_pre_R1')
            result = archive_existing_splits(root, 'data/splits/seed42', destination)
            self.assertEqual(result['status'], 'obsolete')
            self.assertEqual(len(result['files']), 3)
            self.assertEqual((root / destination / paths[0]).read_text(), '[1]')
            with self.assertRaises(FileExistsError):
                archive_existing_splits(root, 'data/splits/seed42', destination)

    def test_plan_musique_targets(self):
        weights = {'2': 518, '3': 316, '4': 166}
        self.assertEqual(largest_remainder(500, weights), {'2': 259, '3': 158, '4': 83})
        self.assertEqual(largest_remainder(2000, weights), {'2': 1036, '3': 632, '4': 332})

    def test_fractional_allocation_and_deterministic_ties(self):
        self.assertEqual(largest_remainder(5, {'b': 1, 'a': 1, 'c': 1}), {'a': 2, 'b': 2, 'c': 1})
        with self.assertRaises(ValueError):
            largest_remainder(5, {'a': 0})

    def test_disjoint_source_only_and_exact_strata(self):
        train = [row(i, 'bridge' if i < 80 else 'comparison') for i in range(100)]
        test = [row(1000 + i, 'bridge' if i < 3 else 'comparison') for i in range(4)]
        partitions, report = construct_splits(train, test, 'hotpotqa', 42,
                                               {'dev-A': 8, 'dev-B': 8, 'train-S': 12})
        self.assertEqual(Counter(x['type'] for x in partitions['dev-A']), {'bridge': 6, 'comparison': 2})
        self.assertEqual(Counter(x['type'] for x in partitions['train-S']), {'bridge': 9, 'comparison': 3})
        for overlap in report['pairwise_overlap'].values():
            self.assertEqual(overlap, {'ids': 0, 'question_texts': 0})
        original = {record_id(x) for x in train}
        self.assertTrue(all(record_id(x) in original for xs in partitions.values() for x in xs))
        self.assertFalse(report['labels_used_for_selection'])

    def test_membership_independent_of_input_order(self):
        train = [row(i) for i in range(50)]
        test = [row(1000)]
        first, _ = construct_splits(train, test, 'hotpotqa', 19, {'dev-A': 7, 'dev-B': 7})
        second, _ = construct_splits(list(reversed(train)), test, 'hotpotqa', 19, {'dev-A': 7, 'dev-B': 7})
        self.assertEqual(first, second)
        third, _ = construct_splits(train, test, 'hotpotqa', 20, {'dev-A': 7, 'dev-B': 7})
        self.assertNotEqual(first, third)

    def test_test_ids_and_same_question_with_different_id_excluded(self):
        test = [row(1000, question='What  is this?')]
        train = [row(i) for i in range(20)] + [row(1000), row(21, question=' what is THIS? '), row(22, question='Question 0?')]
        partitions, report = construct_splits(train, test, 'hotpotqa', 42, {'dev-A': 8, 'dev-B': 8})
        chosen = {record_id(x) for xs in partitions.values() for x in xs}
        self.assertFalse({'q1000', 'q0021', 'q0022'} & chosen)
        self.assertEqual(report['excluded_records'], {'duplicate_training_question_text': 1, 'test_id': 1, 'test_question_text': 1})

    def test_underfilled_stratum_is_an_error(self):
        train = [row(i, 'bridge') for i in range(20)] + [row(30, 'comparison')]
        test = [row(1000, 'bridge'), row(1001, 'comparison')]
        with self.assertRaisesRegex(ValueError, 'needs 2, has 1'):
            construct_splits(train, test, 'hotpotqa', 42, {'dev-A': 4})

    def test_missing_supporting_passage_cannot_silently_enter_corpus(self):
        malformed = row(1)
        malformed['context'] = [['A', ['First.']]]
        malformed['supporting_facts'] = [['Missing', 0]]
        with self.assertRaisesRegex(ValueError, 'No supporting passage'):
            build_corpus([malformed], 'hotpotqa')

    def test_train_corpus_does_not_normalise_whitespace(self):
        result = build_corpus([row(1), row(2)], 'hotpotqa')
        self.assertEqual(result, [{'title': 'A', 'text': 'First. second.'}, {'title': 'B', 'text': 'Other.'}])


if __name__ == '__main__':
    unittest.main()
