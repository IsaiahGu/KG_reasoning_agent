"""Small CPU checks for the calibration's externally specified semantics."""
import os
from pathlib import Path
import unittest

import numpy as np

from hyperbind.calibrate_dense import rank
from hyperbind.hipporag_upstream import gold_function, query_instruction, recall_metric, score_normalizer

ROOT = Path(os.environ.get('HYPERBIND_ROOT', '/workspace/hg_reasoning_agent'))


class RetrievalSemantics(unittest.TestCase):
    def test_hotpot_sentence_separator_and_duplicate_support_titles(self):
        row = {'supporting_facts': [['Paris', 0], ['Paris', 1]],
               'context': [['Paris', ['First sentence.', ' Second sentence.']], ['Other', ['noise']]]}
        gold = gold_function(ROOT)
        self.assertEqual(gold([row], 'hotpotqa'), [['Paris\nFirst sentence. Second sentence.']])
        self.assertEqual(gold([row], '2wikimultihopqa'), [['Paris\nFirst sentence.  Second sentence.']])

    def test_musique_distractor_is_not_gold(self):
        row = {'paragraphs': [{'title': 'A', 'paragraph_text': 'support', 'is_supporting': True},
                              {'title': 'B', 'paragraph_text': 'noise', 'is_supporting': False}]}
        self.assertEqual(gold_function(ROOT)([row], 'musique'), [['A\nsupport']])

    def test_macro_recall_gives_partial_credit(self):
        aggregate, _ = recall_metric(ROOT).calculate_metric_scores([['a', 'b'], ['c']],
            [['a', 'x', 'y', 'z', 'w'], ['c', 'x', 'y', 'z', 'w']], k_list=[5])
        self.assertEqual(aggregate['Recall@5'], .75)

    def test_cosine_ranking_and_exact_instruction(self):
        corpus = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float16)
        queries = np.array([[.9, .1], [.1, .9]], dtype=np.float16)
        self.assertEqual(rank(corpus, queries, k=2, normalize=score_normalizer(ROOT)), [[0, 1], [1, 0]])
        self.assertEqual(query_instruction(ROOT), 'Given a question, retrieve relevant documents that best answer the question.')


if __name__ == '__main__':
    unittest.main()
