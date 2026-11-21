"""Lossless readers and the passage formatting used by HippoRAG 2.

The raw source record is retained on every example. Paragraph text is never
stripped or normalised: gold-document matching depends on exact strings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence


DATASET_ALIASES = {
    'musique': 'musique', '2wiki': '2wikimultihopqa',
    '2wikimultihopqa': '2wikimultihopqa', 'hotpot': 'hotpotqa', 'hotpotqa': 'hotpotqa',
    # simple-QA subsets of the HippoRAG 2 suite (test only)
    'nq': 'nq_rear', 'nq_rear': 'nq_rear', 'popqa': 'popqa',
}
SIMPLE_QA = ('nq_rear', 'popqa')  # no question type; NQ rows carry no id


def dataset_name(name: str) -> str:
    try:
        return DATASET_ALIASES[name.lower()]
    except KeyError as error:
        raise ValueError(f'Unsupported arena-A dataset: {name}') from error


@dataclass(frozen=True)
class Document:
    title: str
    text: str

    @property
    def content(self) -> str:
        return self.title + '\n' + self.text

    @property
    def key(self) -> tuple[str, str]:
        return self.title, self.text

    def to_dict(self) -> dict[str, str]:
        return {'title': self.title, 'text': self.text}


@dataclass(frozen=True)
class QAExample:
    id: str
    dataset: str
    question: str
    answers: tuple[str, ...]
    contexts: tuple[Document, ...]
    gold_documents: tuple[str, ...]
    stratum: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id, 'dataset': self.dataset, 'question': self.question,
            'answers': list(self.answers), 'contexts': [x.to_dict() for x in self.contexts],
            'gold_documents': list(self.gold_documents), 'stratum': self.stratum,
            'metadata': dict(self.metadata), 'raw': dict(self.raw),
        }


def record_id(row: Mapping[str, Any]) -> str:
    value = row.get('_id', row.get('id'))
    if value is None or not isinstance(value, (str, int)) or str(value) == '':
        raise ValueError('Every question must have a nonempty source id or _id')
    return str(value)


def record_stratum(row: Mapping[str, Any], dataset: str) -> str:
    dataset = dataset_name(dataset)
    if dataset == 'musique':
        # MuSiQue encodes topology in ids such as 3hop1__, 3hop2__, 4hop3__.
        # The leading digit is the hop count; the optional suffix is not a stratum.
        match = re.match(r'^([234])hop\d*__', record_id(row))
        if match is None:
            raise ValueError(f'MuSiQue id has no supported hop prefix: {record_id(row)}')
        return match.group(1)
    if dataset in SIMPLE_QA:
        return 'all'
    value = row.get('type')
    if not isinstance(value, str) or not value:
        raise ValueError(f'{dataset} question has no type: {record_id(row)}')
    return value


def _paired_columns(value: Any, first: str, second: str) -> list:
    """Turn HF parquet struct-of-lists into the official list-of-pairs shape."""
    if isinstance(value, Mapping):
        left, right = value[first], value[second]
        if len(left) != len(right):
            raise ValueError(f'Unaligned columns: {first}, {second}')
        return list(zip(left, right))
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'Expected paired fields, got {type(value).__name__}')
    return list(value)


def official_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return an equivalent HippoRAG record without mutating the source row."""
    result = dict(row)
    if 'context' in result:
        result['context'] = _paired_columns(result['context'], 'title', 'sentences')
    if 'supporting_facts' in result:
        result['supporting_facts'] = _paired_columns(result['supporting_facts'], 'title', 'sent_id')
    return result


def _join_context(sentences: Any, separator: str) -> str:
    if isinstance(sentences, str):
        return sentences
    if not isinstance(sentences, (list, tuple)) or not all(isinstance(x, str) for x in sentences):
        raise ValueError('Context sentences must be strings')
    return separator.join(sentences)


def candidate_documents(row: Mapping[str, Any], dataset: str) -> tuple[Document, ...]:
    dataset = dataset_name(dataset)
    value = official_record(row)
    if 'context' in value:
        separator = '' if dataset.startswith('hotpotqa') else ' '
        docs = [Document(item[0], _join_context(item[1], separator)) for item in value['context']]
    elif 'contexts' in value:
        docs = [Document(item['title'], item['text']) for item in value['contexts']]
    elif 'paragraphs' in value:
        docs = [Document(item['title'], item['text'] if 'text' in item else item['paragraph_text'])
                for item in value['paragraphs']]
    else:
        raise ValueError('Question has no candidate passages')
    if not all(isinstance(doc.title, str) and isinstance(doc.text, str) for doc in docs):
        raise ValueError('Passage title and text must be strings')
    return tuple(docs)


def get_gold_docs(samples: Iterable[Mapping[str, Any]], dataset: str) -> list[list[str]]:
    """Match external/HippoRAG/main.py:get_gold_docs exactly after HF decoding."""
    dataset = dataset_name(dataset)
    gold_docs = []
    for raw in samples:
        sample = official_record(raw)
        if 'supporting_facts' in sample:
            gold_titles = {item[0] for item in sample['supporting_facts']}
            supporting_contexts = [item for item in sample['context'] if item[0] in gold_titles]
            separator = '' if dataset.startswith('hotpotqa') else ' '
            gold_doc = [item[0] + '\n' + separator.join(item[1]) for item in supporting_contexts]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            if 'paragraphs' not in sample:
                raise ValueError('Each sample needs supporting_facts, contexts, or paragraphs')
            paragraphs = [item for item in sample['paragraphs'] if item.get('is_supporting', True)]
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text'])
                        for item in paragraphs]
        gold_docs.append(list(dict.fromkeys(gold_doc)))
    return gold_docs


def get_gold_answers(sample: Mapping[str, Any]) -> tuple[str, ...]:
    if 'answer' in sample or 'gold_ans' in sample:
        answer = sample.get('answer', sample.get('gold_ans'))
    elif 'reference' in sample:
        answer = sample['reference']
    elif 'obj' in sample:
        answer = [sample['obj'], sample['o_wiki_title']]
        for key in ('possible_answers', 'o_aliases'):
            value = sample.get(key, [])
            answer.extend(value if isinstance(value, list) else [value])
    else:
        raise ValueError('Question has no answer field')
    if answer is None:
        raise ValueError('Answer field must not be null')
    answers = [answer] if isinstance(answer, str) else list(answer)
    answers.extend(sample.get('answer_aliases', []))
    if not all(isinstance(value, str) for value in answers):
        raise ValueError('Answers and aliases must be strings')
    return tuple(dict.fromkeys(answers))


def normalize_record(row: Mapping[str, Any], dataset: str) -> QAExample:
    dataset = dataset_name(dataset)
    question = row.get('question')
    if not isinstance(question, str):
        raise ValueError('Question field must be a string')
    return QAExample(
        id=record_id(row), dataset=dataset, question=question, answers=get_gold_answers(row),
        contexts=candidate_documents(row, dataset), gold_documents=tuple(get_gold_docs([row], dataset)[0]),
        stratum=record_stratum(row, dataset),
        metadata={k: v for k, v in row.items() if k not in {
            '_id', 'id', 'question', 'answer', 'gold_ans', 'answer_aliases', 'context', 'contexts', 'paragraphs'}},
        raw=row,
    )


def iter_raw_records(path: Path | str) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*.parquet'))
        if not files:
            raise FileNotFoundError(f'No parquet shards found in {path}')
        for file in files:
            yield from iter_raw_records(file)
    elif path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == '.json':
        with path.open(encoding='utf-8') as stream:
            rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError(f'Expected JSON array: {path}')
        yield from rows
    elif path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
            yield from batch.to_pylist()
    else:
        raise ValueError(f'Unsupported dataset file: {path}')


def load_raw_records(path: Path | str) -> list[dict[str, Any]]:
    return list(iter_raw_records(path))


def load_samples(path: Path | str, dataset: str) -> list[QAExample]:
    return [normalize_record(row, dataset) for row in iter_raw_records(path)]


def load_test_samples(project_root: Path | str, dataset: str) -> list[QAExample]:
    dataset = dataset_name(dataset)
    path = Path(project_root) / 'data/hipporag2' / (dataset + '.json')
    if dataset in SIMPLE_QA:  # a row without an id gets its position in the file as id
        return [normalize_record(row if 'id' in row or '_id' in row else {'id': '%s_%04d' % (dataset, index), **row}, dataset)
                for index, row in enumerate(iter_raw_records(path))]
    return load_samples(path, dataset)


def load_corpus(path: Path | str) -> list[Document]:
    return [Document(row['title'], row['text']) for row in iter_raw_records(path)]


def deduplicate_documents(examples: Iterable[QAExample]) -> list[Document]:
    unique = {}
    for example in examples:
        for doc in example.contexts:
            unique.setdefault(doc.key, doc)
    return list(unique.values())
