"""Answer support across derivations and the plan memory's neighbour selection (no GPU, no model)."""
import numpy as np

from hyperbind.chain_agent import PlanPool
from hyperbind.support import FEATURES, SupportModel, answer_groups, chain_features, merge


def record(answer='Paris', chain='Paris', complete=True, hops=('Where was X born?', 'What is the capital of #1?')):
    step = {'candidates': [{'features': {'explicit': 1.0, 'quote_verbatim': 1.0, 'answer_in_quote': 1.0, 'passage_rank': 0.0,
                                          'passage_score': 0.6, 'question_score': 0.4, 'has_next': 1.0, 'next_top1': 0.7}}],
            'chosen': 0}
    return {'question': 'What is the capital of the country where X was born?', 'hops': list(hops), 'steps': [step, step],
            'complete': complete, 'chain_answer': chain, 'reader_clean': answer, 'reader_marker': True, 'backtracked': False}


def test_features_have_the_declared_length_and_use_no_gold_field():
    row = record()
    row['gold'] = row['answers'] = 'should not matter'
    vector = chain_features(row)
    assert vector.shape == (len(FEATURES),) and vector[FEATURES.index('chain_reader_agree')] == 1.0
    assert chain_features(record(chain='Lyon'))[FEATURES.index('chain_reader_agree')] == 0.0
    assert chain_features({'question': 'q', 'reader_clean': ''}).shape == (len(FEATURES),)  # a direct read without steps


def test_answers_are_grouped_by_token_match_and_empty_answers_are_ignored():
    assert answer_groups(['Paris', 'paris.', '', 'Lyon']) == [[0, 1], [3]]


def test_two_weaker_derivations_that_agree_beat_one_stronger_derivation():
    index, support, distinct = merge(['Lyon', 'Paris', 'Paris'], [0.6, 0.5, 0.5])
    assert index in (1, 2) and abs(support - 0.75) < 1e-9 and distinct == 2
    assert merge(['Lyon', 'Paris'], [0.6, 0.5])[0] == 0  # without agreement the more reliable derivation wins
    assert merge(['', ''], [0.2, 0.9]) == (1, 0.0, 0)


def test_model_learns_a_separable_feature_and_survives_a_round_trip(tmp_path):
    generator = np.random.default_rng(0)
    X = generator.normal(size=(400, len(FEATURES)))
    X[:, 0] = 1.0
    y = (X[:, 3] > 0).astype(float)
    model = SupportModel.fit(X, y, dense_prior=0.4)
    assert np.mean((model.predict(X) > 0.5) == (y > 0.5)) > 0.95
    model.save(tmp_path / 'model.json')
    loaded = SupportModel.load(tmp_path / 'model.json')
    assert np.allclose(loaded.predict(X), model.predict(X)) and loaded.reliability({}, dense=True) == 0.4


def test_plan_memory_skips_the_own_question_and_the_nearest_examples():
    pool = PlanPool.__new__(PlanPool)
    pool.ids = ['a', 'b', 'c', 'd']
    pool.examples = [('qa', ['ha']), ('qb', ['hb']), ('qc', ['hc']), ('qd', ['hd'])]
    pool.vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert [example[0] for example in pool.nearest(query, 2)] == ['qb', 'qa']  # the nearest example comes last
    assert [example[0] for example in pool.nearest(query, 2, exclude_id='a')] == ['qc', 'qb']
    assert [example[0] for example in pool.nearest(query, 2, skip=2)] == ['qd', 'qc']
