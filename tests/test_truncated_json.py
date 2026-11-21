"""Pure CPU recovery fixtures, including the quoted-comma truncation failure."""
import hashlib
import json
import unittest

from supportchain.truncated_json import recover_triple_prefix


class TriplePrefixTests(unittest.TestCase):
    def test_quoted_comma_and_incomplete_entity_never_create_a_value(self):
        first = ['John Wellborn Root, Jr.', 'joined', 'Holabird & Roche']
        raw = '{"triples":[' + json.dumps(first) + ',["John Wellborn Root, Jr.'
        result = recover_triple_prefix(raw)
        self.assertEqual(result.items, [first])
        self.assertTrue(result.details['discarded_incomplete_element'])
        self.assertEqual(result.details['invented_values'], 0)
        self.assertEqual(result.details['raw_response_sha256'], hashlib.sha256(raw.encode()).hexdigest())

    def test_repeated_values_and_order_are_preserved(self):
        first, second = ['A', 'r', 'B'], ['A', 's', 'C']
        raw = '{"triples":[' + ','.join(map(json.dumps, [first, first, second]))
        result = recover_triple_prefix(raw)
        self.assertEqual(result.items, [first, first, second])
        self.assertFalse(result.details['reordered_or_deduplicated'])

    def test_escaped_quotes_backslashes_and_brackets_do_not_change_boundaries(self):
        triple = ['A "quote", [x]', 'r\\s', 'B']
        raw = '{"triples":[' + json.dumps(triple) + ',["x\\"y'
        self.assertEqual(recover_triple_prefix(raw).items, [triple])

    def test_valid_unicode_escape_prefix_can_be_discarded(self):
        for suffix in ('["\\u', '["\\u1', '["\\u12', '["\\u123'):
            result = recover_triple_prefix('{"triples":[["A","r","B"],' + suffix)
            self.assertEqual(result.items, [['A', 'r', 'B']])

    def test_eof_at_each_valid_tail_state_preserves_only_completed_triplets(self):
        base = '{"triples":[["A","r","B"]'
        for suffix in ('', ' ', ',', ', ', ',[', ',[ ', ',["', ',["C"', ',["C",',
                       ',["C","s"', ',["C","s",', ',["C","s","D"', ']'):
            with self.subTest(suffix=suffix):
                self.assertEqual(recover_triple_prefix(base + suffix).items, [['A', 'r', 'B']])

    def test_complete_valid_root_is_not_a_fallback(self):
        for raw in ('{"triples":[["A","r","B"]]}', '{"triples":[["A","r","B"]]} garbage'):
            with self.assertRaises(ValueError): recover_triple_prefix(raw)

    def test_wrong_first_field_and_non_object_are_rejected(self):
        for raw in ('{"other":0,"triples":[["A","r","B"]', '[["A","r","B"]',
                    '```json\n{"triples":[["A","r","B"]', '{"triples":null', '{"triples":'):
            with self.assertRaises(ValueError): recover_triple_prefix(raw)

    def test_midstream_syntax_errors_do_not_become_successful_truncation(self):
        base = '{"triples":[["A","r","B"],'
        for suffix in ('["C",BAD', '["C" "s"', '["C","s","D",', '["C\\z',
                       '["C\\u12z', '["C\nx', 'null garbage', '["C",]', ',["C',
                       '["C","s"} ', '[] trailing'):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                recover_triple_prefix(base + suffix)

    def test_nonfinite_json_never_passes(self):
        for token in ('NaN', 'Infinity', '-Infinity'):
            with self.assertRaises(ValueError):
                recover_triple_prefix('{"triples":[["A","r","B"],' + token)

    def test_complete_invalid_values_are_preserved_for_native_filtering(self):
        result = recover_triple_prefix('{"triples":[["A","r","B"],["bad"],')
        self.assertEqual(result.items, [['A', 'r', 'B'], ['bad']])
        self.assertEqual(result.details['complete_json_values'], 2)
        self.assertEqual(result.details['complete_three_string_triples'], 1)

    def test_no_complete_valid_triple_is_rejected(self):
        for raw in ('{"triples":[', '{"triples":[["A","r"', '{"triples":[["bad"],',
                    '{"triples":[["","r","B"],', '{"triples":[null,'):
            with self.assertRaises(ValueError): recover_triple_prefix(raw)

    def test_extra_fields_or_trailing_comma_are_rejected(self):
        for raw in ('{"triples":[["A","r","B"]],"x":', '{"triples":[["A","r","B"],]'):
            with self.assertRaises(ValueError): recover_triple_prefix(raw)

    def test_every_eof_cut_retains_exactly_the_already_closed_triplets(self):
        items = [['A, Jr.', 'r', 'B'], ['C "quote"', 's', 'D\\E'], ['甲', 't', '乙']]
        parts = [json.dumps(item, ensure_ascii=True) for item in items]
        raw = '{"triples":[' + ',\n'.join(parts) + ']}'
        boundaries = []
        position = len('{"triples":[')
        for index, part in enumerate(parts):
            position += len(part)
            boundaries.append(position)
            if index < len(parts) - 1:
                position += 2
        for cutoff in range(boundaries[0], len(raw)):
            expected = [item for item, end in zip(items, boundaries) if end <= cutoff]
            with self.subTest(cutoff=cutoff):
                self.assertEqual(recover_triple_prefix(raw[:cutoff]).items, expected)


if __name__ == '__main__': unittest.main()
