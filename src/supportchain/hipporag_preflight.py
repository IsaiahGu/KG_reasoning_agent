"""OpenIE checks, native-first parsing and audited recovery of complete triples."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import threading

from hipporag.information_extraction.openie_openai import OpenIE, _extract_json_list_field
from hipporag.utils.llm_utils import fix_broken_generated_json
from hipporag.utils.misc_utils import filter_invalid_triples

from .truncated_json import recover_triple_prefix


class PromptLengthError(RuntimeError):
    pass


# Entity responses that upstream cannot parse stop the whole corpus there. Two shapes are let through, both recorded
# in the audit file: entities wrapped in objects with a type label (the names are kept), and a complete JSON object
# that has no entity list at all (native ner() already returns no entities for it). The second is allowed for at
# most this many passages of one corpus; beyond that, and for any other failure, the run stops as upstream does.
UNPARSED_NER_LIMIT = 5


def object_without_entity_list(response):
    try:
        payload = json.loads(response)
    except ValueError:
        return False
    return isinstance(payload, dict) and 'named_entities' not in payload


def labelled_entities(response):
    """Names from {"named_entities": [{"text": ..., "label": ...}, ...]}; None for any other shape."""
    try:
        payload = json.loads(response)
    except ValueError:
        return None
    items = payload.get('named_entities') if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        return None
    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and isinstance(item.get('text'), str) and item['text'].strip():
            names.append(item['text'])
        else:
            return None
    return names


def load_tokenizer(root):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(Path(root) / 'models/Qwen/Qwen3.5-9B'),
                                         local_files_only=True, trust_remote_code=False)


def prompt_tokens(tokenizer, messages):
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                             enable_thinking=False))


def summarize_audit(path):
    rows = [] if not Path(path).exists() else [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    total = len(rows)
    counts = Counter(row.get('error_kind') for row in rows if row.get('error_kind'))
    truncated = sum(row.get('finish_reason') == 'length' for row in rows)
    recovered = sum(bool(row.get('complete_prefix_recovery')) for row in rows)
    labelled = {row['chunk_id'] for row in rows if row.get('labelled_entity_recovery')}
    unparsed = {row['chunk_id'] for row in rows if row.get('unparsed_ner_kept_empty')}
    return {'attempts': total, 'operational_errors': counts['operational'], 'parse_errors': counts['parse'],
            'recovered_truncated_triples': recovered,
            'ner_passages_recovered_from_labelled_entities': sorted(labelled),
            'ner_passages_kept_with_no_entities': sorted(unparsed),
            'original_native_parse_failures': counts['parse'] + recovered,
            'truncated': truncated, 'truncated_ratio': truncated / total if total else None,
            'bad_or_truncated': sum(bool(row.get('error_kind')) or row.get('finish_reason') == 'length' for row in rows),
            'empty_outputs': sum(row.get('extracted_count') == 0 and not row.get('error_kind') for row in rows),
            'invalid_triples_dropped': sum(row.get('invalid_triples_dropped', 0) for row in rows),
            'parser': 'upstream OpenIE first; complete JSON-value prefix fallback only for length-truncated failed triples; '
                      'entity names taken from labelled entity objects; at most %d passages per corpus kept with no entities '
                      'when the entity answer has no entity list; original raw response and native filtering preserved' % UNPARSED_NER_LIMIT}


class AuditedOpenIE(OpenIE):
    """Delegate first, retaining complete values only after native length failure."""
    def __init__(self, *args, audit_path, tokenizer=None, context_limit=16384, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_lock = threading.Lock()
        self.tokenizer = tokenizer
        self.context_limit = context_limit
        self.unparsed_ner = set()

    def record(self, value):
        with self.audit_lock, self.audit_path.open('a') as output:
            output.write(json.dumps(value) + '\n')

    def check(self, chunk_key, stage, messages, maximum):
        count = None if self.tokenizer is None else prompt_tokens(self.tokenizer, messages)
        if count is not None and count + maximum > self.context_limit:
            self.record({'chunk_id': chunk_key, 'stage': stage, 'error_kind': 'operational',
                         'error_type': 'PromptLengthError', 'prompt_tokens': count,
                         'requested_completion_tokens': maximum, 'context_limit': self.context_limit})
            raise PromptLengthError(f'{stage} input+output budget {count}+{maximum} exceeds {self.context_limit}; no text was truncated or skipped')
        return count

    def observe(self, stage, result, count):
        error = result.metadata.get('error')
        # Native methods set response only after a successful backend return.
        # A metadata.error with no raw response is a transport/server error;
        # a nonempty raw response means native parsing failed.
        kind = ('parse' if result.response else 'operational') if error else None
        extracted = result.unique_entities if stage == 'NER' else result.triples
        record = {'chunk_id': result.chunk_id, 'stage': stage, 'error_kind': kind,
                  'error': error, 'finish_reason': result.metadata.get('finish_reason'),
                  'prompt_tokens_preflight': count, 'metadata': result.metadata,
                  'response_sha256': hashlib.sha256(result.response.encode()).hexdigest(),
                  'extracted_count': len(extracted)}
        recovery = result.metadata.get('complete_prefix_recovery')
        if recovery:
            record['complete_prefix_recovery'] = recovery
        for key in ('native_parse_error', 'labelled_entity_recovery', 'unparsed_ner_kept_empty'):
            if stage == 'NER' and result.metadata.get(key):
                record[key] = result.metadata[key]
        if stage == 'TRIPLES' and not error:
            raw = result.response if recovery else (fix_broken_generated_json(result.response) if result.metadata.get('finish_reason') == 'length' else result.response)
            # This is the same extraction helper used by the native method;
            # it is diagnostic only, and never changes its returned triples.
            try:
                before = recovery['complete_json_values'] if recovery else len(_extract_json_list_field(raw, 'triples'))
                record['invalid_triples_dropped'] = max(0, before - len(extracted))
            except ValueError:
                record['posthoc_audit_unavailable'] = True
        self.record(record)

    def ner(self, chunk_key, passage):
        messages = self.prompt_template_manager.render(name='ner', passage=passage)
        count = self.check(chunk_key, 'NER', messages, self.ner_max_tokens)
        result = super().ner(chunk_key, passage)
        if result.metadata.get('error') and result.response and result.metadata.get('finish_reason') != 'length':
            # Never retry the model, change its cache entry or invent an entity.
            names = labelled_entities(result.response)
            allowed = False
            if names is None and object_without_entity_list(result.response):
                with self.audit_lock:
                    self.unparsed_ner.add(chunk_key)
                    allowed = len(self.unparsed_ner) <= UNPARSED_NER_LIMIT
            if names is not None or allowed:
                result.metadata = dict(result.metadata)
                result.metadata['native_parse_error'] = result.metadata.pop('error')
                if names is not None:
                    result.unique_entities = list(dict.fromkeys(names))
                    result.metadata['labelled_entity_recovery'] = {'entities': len(result.unique_entities)}
                else:
                    result.metadata['unparsed_ner_kept_empty'] = True
        self.observe('NER', result, count)
        return result

    def triple_extraction(self, chunk_key, passage, named_entities):
        messages = self.prompt_template_manager.render(name='triple_extraction', passage=passage,
                                                        named_entity_json=json.dumps({'named_entities': named_entities}))
        count = self.check(chunk_key, 'TRIPLES', messages, self.triple_max_tokens)
        result = super().triple_extraction(chunk_key, passage, named_entities)
        if result.metadata.get('error') and result.metadata.get('finish_reason') == 'length':
            # Native rfind(',') can split a comma inside an entity string. Decode
            # complete values from the original raw text; never close a partial
            # string, invent a relation, retry a model, or alter its cache entry.
            try:
                prefix = recover_triple_prefix(result.response)
                triples = filter_invalid_triples(triples=prefix.items)
                if not triples:
                    raise ValueError('No native-valid complete triple before EOF')
                result.metadata = dict(result.metadata)
                result.metadata['native_parse_error'] = result.metadata.pop('error')
                result.metadata['complete_prefix_recovery'] = prefix.details
                result.triples = triples
            except (ValueError, TypeError) as error:
                result.metadata = {**result.metadata, 'prefix_recovery_rejected': str(error)}
        self.observe('TRIPLES', result, count)
        return result


def preflight_documents(documents, tokenizer, *, ner_max=2048, triple_max=4096, context_limit=16384):
    # Use the original prompt renderer, not a token-per-character approximation.
    from hipporag.prompts import PromptTemplateManager
    prompts = PromptTemplateManager()
    rows = []
    for index, passage in enumerate(documents):
        ner = prompt_tokens(tokenizer, prompts.render(name='ner', passage=passage))
        triple_lower_bound = prompt_tokens(tokenizer, prompts.render(name='triple_extraction', passage=passage,
                                                                      named_entity_json=json.dumps({'named_entities': []})))
        rows.append({'index': index, 'ner_prompt_tokens': ner, 'triple_prompt_tokens_with_empty_ner': triple_lower_bound,
                     'definite_overflow': ner + ner_max > context_limit or triple_lower_bound + triple_max > context_limit,
                     'possible_triple_overflow': triple_lower_bound + triple_max + ner_max > context_limit})
    return {'documents': len(documents), 'context_limit': context_limit, 'ner_completion_limit': ner_max,
            'triple_completion_limit': triple_max,
            'max_ner_prompt_tokens': max((row['ner_prompt_tokens'] for row in rows), default=0),
            'max_triple_lower_bound_tokens': max((row['triple_prompt_tokens_with_empty_ner'] for row in rows), default=0),
            'definite_overflow_documents': [row for row in rows if row['definite_overflow']],
            'possible_triple_overflow_documents': [row for row in rows if row['possible_triple_overflow']],
            'actual_triple_guard': 'AuditedOpenIE checks the exact prompt after NER before issuing each triple request',
            'passed_definite_bounds': not any(row['definite_overflow'] for row in rows)}
