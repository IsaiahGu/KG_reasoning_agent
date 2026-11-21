import copy
import json
from pathlib import Path
import tempfile
import unittest

from supportchain.data import (Document, candidate_documents, deduplicate_documents, get_gold_docs,
                            load_samples, normalize_record, official_record, record_stratum)


class PassageFormattingTests(unittest.TestCase):
    def test_hotpot_keeps_sentence_boundary_whitespace_and_support_titles(self):
        row = {'_id': 'hotpot-1', 'question': 'Q?', 'answer': 'answer', 'type': 'bridge',
               'context': [['A', ['first.', ' second.']], ['B', ['wrong.']], ['A', ['first.', ' second.']]],
               'supporting_facts': [['A', 0], ['A', 1]]}
        self.assertEqual(get_gold_docs([row], 'hotpotqa'), [['A\nfirst. second.']])
        self.assertEqual(candidate_documents(row, 'hotpotqa')[0].text, 'first. second.')

    def test_2wiki_inserts_space_even_when_sentence_has_leading_space(self):
        row = {'_id': 'wiki-1', 'question': 'Q?', 'answer': 'answer', 'type': 'comparison',
               'context': [['A', ['first.', ' second.']]], 'supporting_facts': [['A', 1]]}
        self.assertEqual(get_gold_docs([row], '2wiki'), [['A\nfirst.  second.']])
        self.assertEqual(candidate_documents(row, '2wiki')[0].content, 'A\nfirst.  second.')

    def test_musique_support_default_and_text_fallback_match_upstream(self):
        row = {'id': '3hop__x', 'question': 'Q?', 'answer': 'answer',
               'answer_aliases': ['alias', 'answer'], 'paragraphs': [
                   {'title': 'same', 'paragraph_text': 'kept ', 'is_supporting': True},
                   {'title': 'other', 'text': 'not gold', 'is_supporting': False},
                   {'title': 'default', 'text': 'upstream default true'}],
               'question_decomposition': [{'id': 1, 'answer': 'bridge'}]}
        example = normalize_record(row, 'musique')
        self.assertEqual(example.answers, ('answer', 'alias'))
        self.assertEqual(example.stratum, '3')
        self.assertEqual(example.gold_documents, ('same\nkept ', 'default\nupstream default true'))
        self.assertEqual(example.metadata['question_decomposition'], row['question_decomposition'])
        self.assertIs(example.raw, row)

    def test_hf_parquet_structs_preserve_original_record(self):
        row = {'id': 'hf1', 'question': 'Q?', 'answer': 'A', 'type': 'bridge',
               'context': {'title': ['A', 'B'], 'sentences': [['one.', ' two.'], ['three.']]},
               'supporting_facts': {'title': ['A'], 'sent_id': [0]}}
        original = copy.deepcopy(row)
        self.assertEqual(get_gold_docs([row], 'hotpotqa'), [['A\none. two.']])
        normalized = normalize_record(row, 'hotpotqa')
        self.assertEqual(normalized.id, 'hf1')
        self.assertEqual(row, original)
        self.assertEqual(normalized.raw, original)
        self.assertEqual(official_record(row)['supporting_facts'], [('A', 0)])

    def test_unaligned_hf_columns_fail_instead_of_truncating(self):
        with self.assertRaises(ValueError):
            official_record({'context': {'title': ['A', 'B'], 'sentences': [['one']]}})

    def test_corpus_deduplicates_title_and_text_jointly(self):
        row = {'id': '2hop__x', 'question': 'Q?', 'answer': 'A', 'paragraphs': [
            {'title': 'A', 'text': 'same'}, {'title': 'A', 'text': 'same'},
            {'title': 'A', 'text': 'different'}, {'title': 'B', 'text': 'same'}]}
        result = deduplicate_documents([normalize_record(row, 'musique')])
        self.assertEqual([doc.key for doc in result], [('A', 'same'), ('A', 'different'), ('B', 'same')])

    def test_jsonl_reader_retains_all_source_fields(self):
        row = {'id': '4hop__q', 'question': 'Q?', 'answer': 'A', 'custom': {'x': [1]},
               'paragraphs': [{'title': 'A', 'paragraph_text': 'body'}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sample.jsonl'
            path.write_text(json.dumps(row) + '\n\n', encoding='utf-8')
            sample = load_samples(path, 'musique')[0]
        self.assertEqual(sample.raw, row)
        self.assertEqual(sample.contexts, (Document('A', 'body'),))

    def test_hops_require_the_source_prefix(self):
        self.assertEqual(record_stratum({'id': '3hop1__a_b_c'}, 'musique'), '3')
        self.assertEqual(record_stratum({'id': '3hop2__a_b_c'}, 'musique'), '3')
        self.assertEqual(record_stratum({'id': '4hop3__a_b_c_d'}, 'musique'), '4')
        with self.assertRaises(ValueError):
            record_stratum({'id': 'nohop', 'question_decomposition': [1, 2]}, 'musique')


if __name__ == '__main__':
    unittest.main()
