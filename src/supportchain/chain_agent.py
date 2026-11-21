"""The quote-bound chain: plan -> per-hop retrieve -> per-hop read with a quoted sentence -> final read.

Also the dense read (`ChainAgent.run_dense`) and the plan memory (`PlanPool`). The final reader uses HippoRAG 2's reading
prompt unchanged. Gold labels never enter a prompt or a selection: `labeler` only annotates the saved records of
training and development runs.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import threading

import numpy as np

from .baseline_dense import extract_answer, reader_messages
from .binding_scorer import candidate_features
from .calibrate_dense import fetch
from .data import load_corpus, load_samples, load_test_samples
from .llm import StructuredOutputError
from .textmatch import clean_answer, contains, same_answer, tokens

PLAN_SYS = (
    'You split a multi-hop question into the shortest sequence of single-hop questions. '
    'Each sub-question must be answerable from one Wikipedia paragraph and ask for exactly one entity, date or number. '
    'Refer to the answer of an earlier sub-question as #1, #2, and so on. '
    'Keep every name and constraint of the original question. Do not add facts from your own knowledge and do not answer. '
    'The answer of the last sub-question must be the answer of the original question. '
    'Use as few sub-questions as possible (usually 2 to 4) and never repeat the whole original question as a sub-question.')
PLAN_SYS_COMPARE = (
    ' Exception: for comparison or yes/no questions about two or more things, ask one sub-question for each fact that has to be compared '
    '(for example the birth date of each person) and do not ask the comparison itself; the comparison is made afterwards.')
FEW = [
    ('Who is the spouse of the Green performer?',
     ['Who is the performer of Green?', 'Who is the spouse of #1?']),
    ('When was the last time the team that Derek Jeter played for won the World Series?',
     ['Which team did Derek Jeter play for?', 'When was the last time #1 won the World Series?']),
    ('In what year was the university attended by the author of The Hobbit founded?',
     ['Who is the author of The Hobbit?', 'Which university did #1 attend?', 'In what year was #2 founded?']),
    ('What is the capital of the county that shares a border with the county where Lake Placid is located?',
     ['In which county is Lake Placid located?', 'Which county shares a border with #1?', 'What is the capital of #2?']),
    ('What is the population of the city where the headquarters of the company that makes the iPhone is located?',
     ['Which company makes the iPhone?', 'In which city is the headquarters of #1 located?', 'What is the population of #2?']),
    ('When did the country where Niels Bohr was born join the alliance whose headquarters are in the city where the Atomium is located?',
     ['In which country was Niels Bohr born?', 'In which city is the Atomium located?',
      'Which alliance has its headquarters in #2?', 'When did #1 join #3?']),
]
FEW_COMPARE = [
    ('Who was born first, Albert Einstein or Niels Bohr?',
     ['When was Albert Einstein born?', 'When was Niels Bohr born?']),
    ('Which film has the director who died later, Jaws or Psycho?',
     ['Who directed Jaws?', 'Who directed Psycho?', 'When did #1 die?', 'When did #2 die?']),
    ('Who is the paternal grandfather of Charles III?',
     ['Who is the father of Charles III?', 'Who is the father of #1?']),
    ('Are Lake Placid and Lake Tahoe both located in the same country?',
     ['In which country is Lake Placid located?', 'In which country is Lake Tahoe located?']),
]
PLAN_SCHEMA = {'type': 'object', 'properties': {'hops': {'type': 'array', 'minItems': 1, 'maxItems': 4,
               'items': {'type': 'string'}}}, 'required': ['hops'], 'additionalProperties': False}
# Single-answer hop prompts of the method (--candidates 1): the strict read, then the lenient read.
HOP_SYS = (
    'You are given one question and numbered Wikipedia passages. Answer using only the passages. '
    'Return JSON with four fields in this order. passage: the number of the passage that best answers the question, or 0 if every passage is unrelated. '
    'quote: one sentence copied exactly from that passage that supports the answer, or an empty string. '
    'support: "explicit" if the quote states the answer directly; "implied" if no passage states it directly but the passage strongly suggests it '
    '(for example a place where the person worked, lived or performed, or the larger region that contains a place); "none" if the passages are unrelated. '
    'answer: the shortest exact answer (a name, place, date or number, no explanation), or an empty string when support is none. '
    'If passages are about different things with similar names, choose the one that matches every detail of the question.')
LENIENT_SYS = (
    'You are given one question and numbered Wikipedia passages. The passages may not state the answer word for word. '
    'Pick the passage most closely related to the question and give the most likely answer that it suggests '
    '(for example a place where the person lived, worked or performed, or an event named in the passage). '
    'Return JSON with four fields in this order. passage: the number of that passage, or 0 only if every passage is about something else. '
    'quote: one sentence copied exactly from that passage. support: "explicit" if the quote states the answer directly, otherwise "implied"; "none" only with passage 0. '
    'answer: the shortest answer (a name, place, date or number, no explanation).')
# Variant with several candidates per hop (--candidates 3): one call returns up to b quoted candidates.
CANDIDATE_SYS = (
    'You are given one question and numbered Wikipedia passages. Answer using only the passages. '
    'Return JSON with one field, candidates: a list of up to %d different possible answers, the best one first. '
    'Each candidate has four fields in this order. passage: the number of the passage that supports it. '
    'quote: one sentence copied exactly from that passage. '
    'support: "explicit" if the quote states the answer directly, or "implied" if the passage only suggests it '
    '(for example a place where the person lived, worked or performed, or the larger region that contains a place). '
    'answer: the shortest exact answer (a name, place, date or number, no explanation). '
    'Add a second or third candidate only when it is a really different plausible answer, for example from a passage about another thing with a similar name, '
    'or another reading of the question. Prefer explicit candidates, but when nothing is stated directly give the most likely implied answer. '
    'Return an empty list only if every passage is unrelated to the question.')


def single_schema(count):
    return {'type': 'object', 'properties': {'passage': {'type': 'integer', 'enum': list(range(count + 1))},
            'quote': {'type': 'string'}, 'support': {'type': 'string', 'enum': ['explicit', 'implied', 'none']},
            'answer': {'type': 'string'}},
            'required': ['passage', 'quote', 'support', 'answer'], 'additionalProperties': False}


def candidate_schema(count, width):
    # Property order is the generation order: evidence first, answer last.
    item = {'type': 'object', 'properties': {'passage': {'type': 'integer', 'enum': list(range(1, count + 1))},
            'quote': {'type': 'string'}, 'support': {'type': 'string', 'enum': ['explicit', 'implied']},
            'answer': {'type': 'string'}},
            'required': ['passage', 'quote', 'support', 'answer'], 'additionalProperties': False}
    return {'type': 'object', 'properties': {'candidates': {'type': 'array', 'maxItems': width, 'items': item}},
            'required': ['candidates'], 'additionalProperties': False}


@dataclass(frozen=True)
class Hit:
    index: int
    score: float
    title: str
    content: str


class SplitRetriever:
    """Dense retrieval over one split's corpus with offline NV vectors; new texts go to the embedding service."""

    def __init__(self, root: Path, dataset: str, split: str, endpoint='http://127.0.0.1:18081'):
        self.root, self.dataset, self.split, self.endpoint = root, dataset, split, endpoint.rstrip('/')
        folder = root / 'data/splits/seed42' / dataset
        if split == 'train-S':
            vectors, corpus_path, query_file = root / 'runs/train_s/embeddings' / dataset, folder / 'train-S_corpus.json', 'queries.f16.npy'
        elif split in ('dev-A', 'dev-B'):
            vectors, corpus_path = root / 'runs/dev/embeddings' / dataset, folder / 'dev_corpus.json'
            query_file = 'queries_dev-A.f16.npy' if split == 'dev-A' else None
        elif split == 'test':
            # The HippoRAG 2 test subset. Its vectors come from calibrate_dense.py or prepare_test_embeddings.py: passages in
            # corpus order as title + newline + text, questions in file order, same encoder settings as the dev vectors.
            vectors, corpus_path, query_file = root / 'runs/test_vectors' / dataset, root / 'data/hipporag2' / (dataset + '_corpus.json'), 'queries.f16.npy'
        else:
            raise ValueError('Unknown split: ' + split)
        corpus = load_corpus(corpus_path)
        self.titles = [document.title for document in corpus]
        self.documents = [document.content for document in corpus]
        self.examples = load_test_samples(root, dataset) if split == 'test' else load_samples(folder / (split + '.json'), dataset)
        self._verify = None
        if split == 'test':
            progress = json.loads((vectors / 'passages.f16.progress.json').read_text())
            metadata = {'status': 'complete', 'passages_file': 'passages.f16.npy', 'settings': progress['settings'],
                        'query_ids': [example.id for example in self.examples]}
            if progress.get('completed') != len(corpus) or progress.get('total') != len(corpus):
                raise ValueError('Test passage vectors are incomplete')
        else:
            metadata = json.loads((vectors / 'metadata.json').read_text())
            if metadata['status'] != 'complete':
                raise ValueError('Passage vectors are not complete: ' + str(vectors))
            if metadata['passage_ids'] != [hashlib.sha256(content.encode('utf-8')).hexdigest() for content in self.documents]:
                raise ValueError('Passage vectors and corpus order differ')
        self.settings = metadata['settings']
        self.vectors = np.asarray(np.load(vectors / metadata['passages_file'], mmap_mode='r'), dtype=np.float32)
        if self.vectors.shape[0] != len(self.documents):
            raise ValueError('Passage vectors and corpus differ in length')
        self._known = {}
        if split == 'test':  # no metadata to prove the order: the first question is re-encoded once and compared
            stored = np.load(vectors / query_file, mmap_mode='r')
            if stored.shape[0] != len(self.examples):
                raise ValueError('Test question vectors and questions differ in length')
            self._verify = (self.examples[0].question, np.asarray(stored[0], dtype=np.float32))
        if query_file:
            stored = np.load(vectors / query_file, mmap_mode='r')
            by_id = {example.id: example for example in self.examples}
            for position, identifier in enumerate(metadata['query_ids']):
                if identifier in by_id:
                    self._known[by_id[identifier].question] = np.asarray(stored[position], dtype=np.float32)
        self._lock, self._checked = threading.Lock(), False

    def vector(self, text):
        known = self._known.get(text)
        if known is not None:
            return known
        with self._lock:  # the embedding HTTP service handles one request at a time
            known = self._known.get(text)
            if known is not None:
                return known
            if not self._checked:
                health = fetch(self.endpoint + '/health')
                for name in ('max_length', 'normalize', 'query_instruction', 'passage_instruction', 'dimension', 'model_dtype'):
                    if health['settings'][name] != self.settings[name]:
                        raise ValueError('Embedding service differs from the offline vectors: ' + name)
                if self._verify is not None:
                    response = fetch(self.endpoint + '/encode', {'texts': [self._verify[0]], 'kind': 'query'})
                    fresh = np.frombuffer(base64.b64decode(response['data']), dtype='<f2').astype(np.float32).reshape(-1)
                    cosine = float(fresh @ self._verify[1] / (np.linalg.norm(fresh) * np.linalg.norm(self._verify[1])))
                    if cosine < 0.995:
                        raise ValueError('Stored test question vectors do not match the embedding service (cosine %.3f)' % cosine)
                self._checked = True
            response = fetch(self.endpoint + '/encode', {'texts': [text], 'kind': 'query'})
            vector = np.frombuffer(base64.b64decode(response['data']), dtype='<f2').astype(np.float32).reshape(-1)
            if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all():
                raise ValueError('Invalid query vector')
            self._known[text] = vector
            return vector

    def search(self, text, k):
        vector = self.vector(text)
        scores = self.vectors @ vector
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top], kind='stable')]
        return [Hit(int(index), float(scores[index]), self.titles[index], self.documents[index]) for index in top], vector


@dataclass(frozen=True)
class ChainConfig:
    k_hop: int = 8          # passages shown to the hop reader per window
    budget: int = 5         # passages for the final reader
    candidates: int = 3     # 1 = the method's single-answer reads (strict, second window, lenient); 3 = variant
    selector: str = 'first'  # 'first' or 'scorer' (variant: learned hop scorer)
    lookahead: bool = True  # next-hop retrieval features for every candidate (embedding calls only)
    backtrack: bool = False  # on a hop without any candidate, try the previous hop's next candidate once
    final_reader: bool = True  # False: collect hop candidates only (training data), no answer is produced
    reader_context: str = 'both'  # 'both' = passages + reasoning notes, 'passages', or 'notes'
    plan_examples: str = 'fixed'  # 'fixed' = six hand-written examples, 'knn' = nearest training questions (plan memory)
    plan_neighbours: int = 6
    plan_skip: int = 0      # skip the nearest N plan examples: a second, different plan for the same question


class PlanPool:
    """Plan memory: solved train-S questions with their plans, used as nearest-neighbour few-shot examples for PLAN.

    Without `cases` the plans are the gold decompositions (MuSiQue only). With `cases` (a JSON list of
    {"id", "hops"}) they are the chain's own plans that were admitted to memory, so no decomposition annotation is
    used. No weight update either way. `limit` keeps only the first questions of train-S, so a held-out tail of
    train-S never sees its own plans.
    """

    def __init__(self, root: Path, dataset: str, limit=None, cases=None):
        folder = root / 'runs/train_s/embeddings' / dataset
        metadata = json.loads((folder / 'metadata.json').read_text())
        rows = json.loads((root / 'data/splits/seed42' / dataset / 'train-S.json').read_text())
        key = lambda row: str(row.get('id', row.get('_id')))  # 2Wiki rows carry `_id`
        if metadata['status'] != 'complete' or [key(row) for row in rows] != metadata['query_ids']:
            raise ValueError('train-S question vectors do not match train-S.json')
        rows = rows[:limit] if limit else rows
        if cases is None:
            plans = {key(row): [step['question'].strip() for step in row['question_decomposition']] for row in rows}
        else:
            plans = {str(case['id']): [hop.strip() for hop in case['hops']] for case in json.loads(Path(cases).read_text())}
        positions = [position for position, row in enumerate(rows) if key(row) in plans]
        if not positions:
            raise ValueError('The plan memory is empty')
        self.ids = [key(rows[position]) for position in positions]
        self.examples = [(rows[position]['question'], plans[key(rows[position])]) for position in positions]
        self.vectors = np.asarray(np.load(folder / metadata['queries_file'], mmap_mode='r')[positions], dtype=np.float32)

    def nearest(self, vector, count, exclude_id=None, skip=0):
        order = np.argsort(-(self.vectors @ vector), kind='stable')[:skip + count + 1]
        chosen = [int(index) for index in order if self.ids[int(index)] != exclude_id][skip:skip + count]
        return [self.examples[index] for index in reversed(chosen)]  # the nearest example comes last


class ChainAgent:
    def __init__(self, root: Path, retriever: SplitRetriever, client, config=ChainConfig(), scorer=None, labeler=None,
                 plan_pool=None):
        if config.selector not in ('first', 'scorer') or (config.selector == 'scorer' and scorer is None):
            raise ValueError('selector must be first, or scorer with a fitted scorer')
        if config.reader_context not in ('both', 'passages', 'notes') or config.plan_examples not in ('fixed', 'knn'):
            raise ValueError('Unknown reader context or plan example mode')
        if config.plan_examples == 'knn' and plan_pool is None:
            raise ValueError('Nearest-neighbour planning needs a PlanPool')
        self.root, self.retriever, self.client, self.config = root, retriever, client, config
        self.scorer, self.labeler, self.plan_pool = scorer, labeler, plan_pool
        self.compare = retriever.dataset != 'musique'

    def _call(self, messages, schema, max_tokens, stage, question_id, tally):
        try:
            result = self.client.complete(messages, schema, thinking=False, generation_overrides={'max_tokens': max_tokens},
                                          metadata={'phase': 'chain', 'stage': stage, 'question_id': question_id})
        except StructuredOutputError:
            tally['structured_failures'] += 1
            return None
        tally['calls'] += 1
        tally['prompt_tokens'] += int(result.usage.get('prompt_tokens', 0) or 0)
        tally['completion_tokens'] += int(result.usage.get('completion_tokens', 0) or 0)
        return result

    def plan(self, question, question_id, tally):
        messages = [{'role': 'system', 'content': PLAN_SYS + (PLAN_SYS_COMPARE if self.compare else '')}]
        if self.config.plan_examples == 'knn':
            shots = self.plan_pool.nearest(self.retriever.vector(question), self.config.plan_neighbours,
                                           exclude_id=question_id, skip=self.config.plan_skip)
        else:
            shots = FEW + (FEW_COMPARE if self.compare else [])
        for example, hops in shots:
            messages += [{'role': 'user', 'content': example}, {'role': 'assistant', 'content': json.dumps({'hops': hops})}]
        messages.append({'role': 'user', 'content': question})
        result = self._call(messages, PLAN_SCHEMA, 400, 'PLAN', question_id, tally)
        hops = [hop.strip() for hop in result.parsed['hops'] if hop.strip()] if result is not None else []
        valid = bool(hops) and all(1 <= int(ref) <= index for index, hop in enumerate(hops) for ref in re.findall(r'#(\d+)', hop))
        return (hops, True) if valid else ([question], False)

    def _listing(self, sub_question, hits):
        return 'Question: %s\n\nPassages:\n%s' % (sub_question, '\n\n'.join(
            '[%d] %s' % (number, hit.content) for number, hit in enumerate(hits, 1)))

    def _row(self, row, hits, offset):
        answer, number = (row.get('answer') or '').strip(), row.get('passage', 0)
        if not answer or row.get('support') == 'none' or not 1 <= number <= len(hits):
            return None
        hit, quote = hits[number - 1], row.get('quote') or ''
        return {'answer': answer, 'quote': quote, 'support': row['support'], 'passage_index': hit.index,
                'passage_rank': offset + number - 1, 'passage_score': hit.score,
                'quote_verbatim': bool(quote.strip()) and contains(hit.content, quote)}

    def read(self, sub_question, hits, question_id, tally):
        """Candidates for one hop, best first. Windows: passages 1..k, then k+1..2k when the first gives nothing."""
        k, width = self.config.k_hop, self.config.candidates
        windows = [(0, hits[:k]), (k, hits[k:2 * k])]
        if width == 1:
            for offset, window in windows:
                if not window:
                    continue
                result = self._call([{'role': 'system', 'content': HOP_SYS}, {'role': 'user', 'content': self._listing(sub_question, window)}],
                                    single_schema(len(window)), 300, 'READ', question_id, tally)
                row = self._row(result.parsed, window, offset) if result is not None else None
                if row is not None:
                    return [row]
            result = self._call([{'role': 'system', 'content': LENIENT_SYS}, {'role': 'user', 'content': self._listing(sub_question, windows[0][1])}],
                                single_schema(len(windows[0][1])), 300, 'READ_LENIENT', question_id, tally)
            row = self._row(result.parsed, windows[0][1], 0) if result is not None else None
            if row is not None and row['support'] != 'explicit':
                row['support'] = 'implied'
            return [row] if row is not None else []
        for offset, window in windows:
            if not window:
                continue
            result = self._call([{'role': 'system', 'content': CANDIDATE_SYS % width}, {'role': 'user', 'content': self._listing(sub_question, window)}],
                                candidate_schema(len(window), width), 700, 'READ', question_id, tally)
            found = []
            for raw in (result.parsed['candidates'] if result is not None else []):
                row = self._row(raw, window, offset)
                if row is not None and not any(tokens(row['answer']) == tokens(other['answer']) for other in found):
                    found.append(row)
            if found:
                return found
        return []

    def _annotate(self, hops, number, sub_question, hits, vector, candidates, answers, question_vector):
        """Features for every candidate of this hop. Lookahead uses the embedding service only."""
        following = None
        if self.config.lookahead and len(candidates) > 1:
            for later in range(number + 1, len(hops) + 1):
                refs = [int(ref) for ref in re.findall(r'#(\d+)', hops[later - 1])]
                if number in refs and all(ref == number or ref in answers for ref in refs):
                    following = hops[later - 1]
                    break
        bound = [answers[int(ref)] for ref in re.findall(r'#(\d+)', hops[number - 1]) if int(ref) in answers]
        top_score = hits[0].score if hits else 0.0
        window = hits[:self.config.k_hop]
        for position, candidate in enumerate(candidates):
            ahead = None
            if following is not None:
                filled = re.sub(r'#(\d+)', lambda match: candidate['answer'] if int(match.group(1)) == number
                                else answers[int(match.group(1))], following)
                next_hits, _ = self.retriever.search(filled, 3)
                ahead = {'scores': [hit.score for hit in next_hits], 'titles': [hit.title for hit in next_hits],
                         'contents': [hit.content for hit in next_hits]}
            index = candidate['passage_index']
            candidate['features'] = candidate_features(
                sub_question=sub_question, candidate=candidate, position=position, count=len(candidates),
                title=self.retriever.titles[index], content=self.retriever.documents[index], top_score=top_score,
                window_contents=[hit.content for hit in window], bound=bound, hop=number, hops=len(hops),
                question_score=float(self.retriever.vectors[index] @ question_vector), ahead=ahead)

    def _forward(self, hops, start, answers, question_id, question_vector, tally):
        steps, answers = [], dict(answers)
        for number in range(start, len(hops) + 1):
            template = hops[number - 1]
            if any(int(ref) not in answers for ref in re.findall(r'#(\d+)', template)):
                return steps, answers, False
            sub_question = re.sub(r'#(\d+)', lambda match: answers[int(match.group(1))], template)
            hits, vector = self.retriever.search(sub_question, 2 * self.config.k_hop)
            candidates = self.read(sub_question, hits, question_id, tally)
            self._annotate(hops, number, sub_question, hits, vector, candidates, answers, question_vector)
            order = list(range(len(candidates)))
            if self.config.selector == 'scorer' and len(candidates) > 1:
                logits = self.scorer.logits([candidate['features'] for candidate in candidates])
                order = sorted(order, key=lambda position: (-float(logits[position]), position))
                for position, candidate in enumerate(candidates):
                    candidate['logit'] = float(logits[position])
            steps.append({'hop': number, 'template': template, 'question': sub_question,
                          'retrieved': [hit.index for hit in hits[:self.config.k_hop]],
                          'candidates': candidates, 'order': order, 'chosen': order[0] if order else None})
            if not candidates:
                return steps, answers, False
            answers[number] = candidates[order[0]]['answer']
        return steps, answers, True

    def run(self, example):
        tally = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'structured_failures': 0}
        hops, plan_ok = self.plan(example.question, example.id, tally)
        base, question_vector = self.retriever.search(example.question, 10)
        steps, answers, complete = self._forward(hops, 1, {}, example.id, question_vector, tally)
        backtracked = False
        if not complete and self.config.backtrack and len(steps) > 1:
            for earlier in range(len(steps) - 1, 0, -1):  # nearest earlier hop first
                step = steps[earlier - 1]
                if len(step['order']) < 2:
                    continue
                alternative = step['order'][1]
                prefix = {number: answers[number] for number in range(1, earlier)}
                prefix[earlier] = step['candidates'][alternative]['answer']
                new_steps, new_answers, done = self._forward(hops, earlier + 1, prefix, example.id, question_vector, tally)
                if done:
                    steps = steps[:earlier - 1] + [{**step, 'chosen': alternative, 'backtracked_from': step['chosen']}] + new_steps
                    answers, complete, backtracked = new_answers, True, True
                    break
        complete = complete and plan_ok
        chosen = [(step, step['candidates'][step['chosen']]) for step in steps if step['chosen'] is not None]
        evidence, seen = [], set()
        for _, candidate in chosen:
            if candidate['passage_index'] not in seen:
                evidence.append(self.retriever.documents[candidate['passage_index']])
                seen.add(candidate['passage_index'])
        for hit in base:
            if len(evidence) >= self.config.budget:
                break
            if hit.index not in seen:
                evidence.append(hit.content)
                seen.add(hit.index)
        notes = None
        if chosen and plan_ok and self.config.reader_context != 'passages':
            notes = 'Reasoning notes checked against the passages above\n' + '\n'.join(
                'Step %d: %s -> %s%s' % (step['hop'], step['question'], candidate['answer'],
                                         '' if candidate['support'] == 'explicit' else ' (likely, not stated directly)')
                for step, candidate in chosen)
        # 'notes' drops the passages only when there are notes to read; otherwise the reader still gets passages.
        passages = [] if (self.config.reader_context == 'notes' and notes) else list(evidence)
        if notes:
            passages.append(notes)
        reader_answer, reader_clean, marker, finish = None, None, None, None
        if self.config.final_reader:
            result = self.client.complete(reader_messages(self.root, example.question, passages), thinking=False,
                                          generation_overrides={'max_tokens': 2048},
                                          metadata={'phase': 'chain', 'stage': 'ANSWER', 'question_id': example.id})
            tally['calls'] += 1
            tally['prompt_tokens'] += int(result.usage.get('prompt_tokens', 0) or 0)
            tally['completion_tokens'] += int(result.usage.get('completion_tokens', 0) or 0)
            (reader_answer, marker), reader_clean, finish = extract_answer(result.content), clean_answer(result.content), result.finish_reason
        gold = set(example.gold_documents)
        record = {'id': example.id, 'stratum': example.stratum, 'question': example.question, 'hops': hops, 'plan_ok': plan_ok,
                  'complete': complete, 'backtracked': backtracked, 'steps': steps,
                  'chain_answer': answers.get(len(hops)) if complete else None,
                  'reader_answer': reader_answer, 'reader_clean': reader_clean, 'reader_marker': marker,
                  'reader_finish': finish,
                  'evidence_recall': len(gold & set(evidence)) / len(gold) if gold else 0.0,
                  'evidence_all_supporting': bool(gold) and gold <= set(evidence), 'usage': tally}
        if self.labeler is not None:
            self.labeler(example, record)
        return record

    def run_dense(self, example):
        """Dense read: the top passages for the question and the same reader. No plan, no hops."""
        tally = {'calls': 1, 'prompt_tokens': 0, 'completion_tokens': 0, 'structured_failures': 0}
        hits, _ = self.retriever.search(example.question, self.config.budget)
        passages = [hit.content for hit in hits]
        result = self.client.complete(reader_messages(self.root, example.question, passages), thinking=False,
                                      generation_overrides={'max_tokens': 2048},
                                      metadata={'phase': 'chain', 'stage': 'ANSWER_A0', 'question_id': example.id})
        tally['prompt_tokens'] = int(result.usage.get('prompt_tokens', 0) or 0)
        tally['completion_tokens'] = int(result.usage.get('completion_tokens', 0) or 0)
        reader_answer, marker = extract_answer(result.content)
        gold = set(example.gold_documents)
        return {'id': example.id, 'stratum': example.stratum, 'question': example.question, 'hops': [], 'plan_ok': True,
                'complete': False, 'backtracked': False, 'steps': [], 'chain_answer': None, 'reader_answer': reader_answer,
                'reader_clean': clean_answer(result.content), 'reader_marker': marker, 'reader_finish': result.finish_reason,
                'evidence_recall': len(gold & set(passages)) / len(gold) if gold else 0.0,
                'evidence_all_supporting': bool(gold) and gold <= set(passages), 'usage': tally}


def musique_labeler(example, record):
    """Annotate candidates with gold hop labels from MuSiQue's decomposition. Used for train-S and dev diagnostics only."""
    gold = example.raw.get('question_decomposition') or []
    aligned = len(gold) == len(record['hops']) and record['plan_ok']
    record['aligned'] = aligned
    prefix_correct = True
    for step in record['steps']:
        last = step['hop'] == len(record['hops'])
        targets = [gold[step['hop'] - 1]['answer']] if aligned else []
        if aligned and last:
            targets += list(example.answers)
        for candidate in step['candidates']:
            candidate['label'] = int(any(same_answer(target, candidate['answer']) for target in targets)) if aligned else None
        step['prefix_correct'] = prefix_correct if aligned else None
        if aligned:
            picked = step['candidates'][step['chosen']] if step['chosen'] is not None else None
            prefix_correct = prefix_correct and picked is not None and picked['label'] == 1
