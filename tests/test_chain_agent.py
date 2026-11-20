"""Offline tests of the repaired loop with a fake retriever and a fake LLM client (no model, no HTTP)."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hyperbind.binding_scorer import FEATURE_NAMES, BindingScorer, hop_groups, selection_report
from hyperbind.chain_agent import ChainAgent, ChainConfig, Hit, candidate_schema, musique_labeler
from hyperbind.llm import LocalLLMClient
from hyperbind.textmatch import clean_answer, contains, same_answer

ROOT = Path(__file__).resolve().parents[1]
DOCS = ['Green (album)\nGreen is an album by Steve Hillage.', 'Steve Hillage\nSteve Hillage is married to Miquette Giraudy.',
        'Green (band)\nGreen is a band from Chicago led by Jeff Lescher.', 'Filler one\nNothing here.', 'Filler two\nNothing here.',
        'Filler three\nNothing here.']


class FakeRetriever:
    dataset = 'musique'

    def __init__(self):
        self.documents = DOCS
        self.titles = [text.split('\n')[0] for text in DOCS]
        self.vectors = np.eye(len(DOCS), 4, dtype=np.float32)

    def search(self, text, k):
        order = {'Who is the performer of Green?': [0, 2, 3], 'Who is the spouse of Steve Hillage?': [1, 3, 4],
                 'Who is the spouse of Jeff Lescher?': [3, 4, 5]}.get(text, [0, 1, 2, 3, 4])
        hits = [Hit(index, 1.0 - 0.1 * rank, self.titles[index], self.documents[index]) for rank, index in enumerate(order)][:k]
        return hits, np.ones(4, dtype=np.float32)


class FakeClient:
    """Answers by stage. Hop 1 lists the band's leader first and the album's performer second."""

    def __init__(self):
        self.reader_prompts = []

    def complete(self, messages, schema=None, *, thinking=False, generation_overrides=None, metadata=None):
        stage, user = metadata['stage'], messages[-1]['content']
        usage = {'prompt_tokens': 10, 'completion_tokens': 5}
        if stage == 'PLAN':
            parsed = {'hops': ['Who is the performer of Green?', 'Who is the spouse of #1?']}
        elif stage == 'READ' and 'performer of Green' in user:
            parsed = {'candidates': [
                {'passage': 2, 'quote': 'Green is a band from Chicago led by Jeff Lescher.', 'support': 'implied', 'answer': 'Jeff Lescher'},
                {'passage': 1, 'quote': 'Green is an album by Steve Hillage.', 'support': 'explicit', 'answer': 'Steve Hillage'}]}
        elif stage == 'READ' and 'spouse of Steve Hillage' in user:
            parsed = {'candidates': [{'passage': 1, 'quote': 'Steve Hillage is married to Miquette Giraudy.',
                                      'support': 'explicit', 'answer': 'Miquette Giraudy'}]}
        elif stage == 'READ':
            parsed = {'candidates': []}
        else:
            self.reader_prompts.append(user)
            return SimpleNamespace(content='Thought: notes say so.\nAnswer: Miquette Giraudy', parsed=None, usage=usage, finish_reason='stop')
        return SimpleNamespace(content=json.dumps(parsed), parsed=parsed, usage=usage, finish_reason='stop')


EXAMPLE = SimpleNamespace(id='q1', stratum='2', question='Who is the spouse of the Green performer?', answers=('Miquette Giraudy',),
                          gold_documents=(DOCS[0], DOCS[1]),
                          raw={'question_decomposition': [{'answer': 'Steve Hillage'}, {'answer': 'Miquette Giraudy'}]})


def run(config, scorer=None):
    client = FakeClient()
    agent = ChainAgent(ROOT, FakeRetriever(), client, config, scorer=scorer, labeler=musique_labeler)
    return agent.run(EXAMPLE), client


def test_first_candidate_breaks_the_chain_but_the_reader_still_gets_evidence():
    record, client = run(ChainConfig())
    assert record['steps'][0]['chosen'] == 0 and not record['complete'] and record['chain_answer'] is None
    assert record['steps'][1]['candidates'] == [] and record['reader_answer'] == 'Miquette Giraudy'
    assert 'Green (band)' in client.reader_prompts[0] and 'Step 1' in client.reader_prompts[0]
    assert '(likely, not stated directly)' in client.reader_prompts[0]
    assert [candidate['label'] for candidate in record['steps'][0]['candidates']] == [0, 1]
    assert record['steps'][0]['prefix_correct'] is True and record['steps'][1]['prefix_correct'] is False


def test_backtracking_adopts_the_second_candidate_when_the_next_hop_has_no_answer():
    record, _ = run(ChainConfig(backtrack=True))
    assert record['complete'] and record['backtracked'] and record['chain_answer'] == 'Miquette Giraudy'
    assert record['steps'][0]['chosen'] == 1 and record['steps'][0]['backtracked_from'] == 0
    assert record['evidence_all_supporting']


def test_scorer_reorders_candidates_without_any_label():
    weights = [1.0 if name == 'explicit' else 0.0 for name in FEATURE_NAMES]
    scorer = BindingScorer(FEATURE_NAMES, weights, [0.0] * len(FEATURE_NAMES), [1.0] * len(FEATURE_NAMES))
    record, _ = run(ChainConfig(selector='scorer'), scorer)
    assert record['steps'][0]['order'] == [1, 0] and record['complete'] and record['usage']['calls'] == 4
    features = record['steps'][0]['candidates'][1]['features']
    assert features['has_next'] == 1.0 and features['next_title_match'] == 1.0 and features['answer_in_passage'] == 1.0


def test_fit_recovers_a_separating_feature_and_reports_selection():
    record, _ = run(ChainConfig())
    groups = hop_groups([record])
    assert len(groups) == 1 and groups[0]['labels'] == [0, 1]
    training = [(group['rows'], group['labels']) for group in groups] * 8
    scorer = BindingScorer.fit(training, FEATURE_NAMES, l2=0.1, steps=300)
    report = selection_report(groups, scorer)
    assert report['first_correct_percent'] == 0.0 and report['scorer_correct_percent'] == 100.0
    assert BindingScorer.from_dict(scorer.to_dict()).logits(groups[0]['rows']).argmax() == 1


def test_candidate_schema_keeps_evidence_before_answer():
    item = candidate_schema(8, 3)['properties']['candidates']['items']
    assert list(item['properties']) == ['passage', 'quote', 'support', 'answer']


def test_text_matching():
    assert contains('He was born in New York City.', 'new york city') and not contains('in 2012', '2')
    assert same_answer('Nashville, Tennessee', 'Tennessee') and not same_answer('England', 'English')
    assert clean_answer('Thought: Answer: x is wrong.\nAnswer: "Paris".\nWait, more text') == 'Paris'


def test_request_bytes_keep_declared_schema_order_and_order_is_part_of_the_cache_key(tmp_path, monkeypatch):
    import hyperbind.llm as llm
    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, *_):
            return b'{"ok": true}'

    monkeypatch.setattr(llm, 'urlopen', lambda request, timeout=None: sent.append(request.data) or Response())
    llm.http_transport('POST', 'http://127.0.0.1:1/v1/chat/completions', {'schema': {'properties': {'quote': 1, 'answer': 2}}}, {}, 1)
    assert sent[0].index(b'"quote"') < sent[0].index(b'"answer"')

    posts = []

    def transport(method, url, payload, headers, timeout):
        if method == 'GET':
            return {'data': [{'id': 'local'}]}
        posts.append(payload)
        return {'choices': [{'message': {'content': '{"quote":"q","answer":"a"}'}, 'finish_reason': 'stop'}], 'usage': {}}

    client = LocalLLMClient(base_url='http://127.0.0.1:18080/v1', cache_dir=tmp_path / 'cache', log_path=tmp_path / 'calls.jsonl',
                            transport=transport, retries=0)
    text = {'type': 'string'}
    ordered = {'type': 'object', 'properties': {'quote': text, 'answer': text}, 'required': ['quote', 'answer'], 'additionalProperties': False}
    swapped = {'type': 'object', 'properties': {'answer': text, 'quote': text}, 'required': ['quote', 'answer'], 'additionalProperties': False}
    messages = [{'role': 'user', 'content': 'x'}]
    assert not client.complete(messages, ordered).cache_hit and client.complete(messages, ordered).cache_hit
    assert not client.complete(messages, swapped).cache_hit and len(posts) == 2
    assert list(posts[0]['response_format']['json_schema']['schema']['properties']) == ['quote', 'answer']
