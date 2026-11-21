"""Recover only fully decoded triples preceding an EOF-truncated JSON tail.

Call this only after native OpenIE parsing failed with finish_reason='length'.
It neither closes strings nor creates relationships. The caller must retain the
original response and run the original upstream filter_invalid_triples.
"""
from dataclasses import dataclass
import hashlib
import json


POLICY = 'complete_triple_prefix_v1'
WHITESPACE = ' \t\r\n'


def _nonfinite(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON object key')
        value[key] = item
    return value


DECODER = json.JSONDecoder(parse_constant=_nonfinite, object_pairs_hook=_unique_object)


def _space(text, position):
    while position < len(text) and text[position] in WHITESPACE:
        position += 1
    return position


def _incomplete_triple_prefix(text):
    """True only for a syntactically valid EOF prefix of [string,string,string]."""
    position = _space(text, 0)
    if position == len(text) or text[position] != '[':
        return False
    position += 1
    for field in range(3):
        position = _space(text, position)
        if position == len(text):
            return True
        if text[position] != '"':
            return False
        position += 1
        while True:
            if position == len(text):
                return True
            character = text[position]
            position += 1
            if character == '"':
                break
            if ord(character) < 0x20:
                return False
            if character == '\\':
                if position == len(text):
                    return True
                escape = text[position]
                position += 1
                if escape == 'u':
                    for _ in range(4):
                        if position == len(text):
                            return True
                        if text[position] not in '0123456789abcdefABCDEF':
                            return False
                        position += 1
                elif escape not in '"\\/bfnrt':
                    return False
        position = _space(text, position)
        if position == len(text):
            return True
        if field < 2:
            if text[position] != ',':
                return False
            position += 1
        else:
            # A fully closed triple would have been decoded already. Any
            # remaining token is either malformed or an excessive fourth field.
            return False
    return False


@dataclass(frozen=True)
class TriplePrefix:
    items: list
    details: dict


def recover_triple_prefix(raw: str) -> TriplePrefix:
    """Return complete values only, or reject non-EOF/malformed/complete input.

The root must contain exactly one first/only field named ``triples``. Complete
JSON values within that array are decoded independently. A final incomplete
item must itself be a valid prefix of a three-string triple; arbitrary malformed
suffixes are never silently discarded. Successful native responses should not
be passed here and complete roots are deliberately rejected.
"""
    if not isinstance(raw, str) or not raw:
        raise ValueError('No raw JSON response')
    position = _space(raw, 0)
    if position == len(raw) or raw[position] != '{':
        raise ValueError('Expected a root JSON object')
    position = _space(raw, position + 1)
    try:
        key, position = DECODER.raw_decode(raw, position)
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError('Incomplete or invalid root field') from error
    if key != 'triples':
        raise ValueError('The first root field must be triples')
    position = _space(raw, position)
    if position == len(raw) or raw[position] != ':':
        raise ValueError('Expected root field colon')
    position = _space(raw, position + 1)
    if position == len(raw) or raw[position] != '[':
        raise ValueError('The triples value must be an array')
    position += 1
    items, complete_end = [], position
    incomplete_item = False
    while True:
        position = _space(raw, position)
        if position == len(raw):
            break
        if raw[position] == ']':
            position = _space(raw, position + 1)
            if position == len(raw):
                break  # Only the final root brace is missing at EOF.
            if raw[position] == '}':
                raise ValueError('The root JSON object is already complete')
            raise ValueError('Unexpected token after the triples array')
        try:
            item, end = DECODER.raw_decode(raw, position)
        except (ValueError, json.JSONDecodeError) as error:
            if not _incomplete_triple_prefix(raw[position:]):
                raise ValueError('Malformed suffix is not an EOF-truncated triple') from error
            incomplete_item = True
            break
        items.append(item)
        complete_end = end
        position = _space(raw, end)
        if position == len(raw):
            break
        if raw[position] == ',':
            position += 1
            if _space(raw, position) < len(raw) and raw[_space(raw, position)] == ']':
                raise ValueError('Trailing comma before an array closing token')
            continue
        if raw[position] == ']':
            continue
        raise ValueError('Malformed token between complete triple values')
    valid_shapes = sum(isinstance(item, list) and len(item) == 3 and
                       all(isinstance(value, str) and value for value in item) for item in items)
    if not valid_shapes:
        raise ValueError('No complete three-string triple precedes EOF')
    suffix = raw[complete_end:]
    details = {'policy': POLICY, 'raw_response_sha256': hashlib.sha256(raw.encode()).hexdigest(),
               'raw_characters': len(raw), 'complete_json_values': len(items),
               'complete_three_string_triples': valid_shapes,
               'preserved_prefix_characters': complete_end,
               'discarded_suffix_characters': len(suffix),
               'discarded_suffix_sha256': hashlib.sha256(suffix.encode()).hexdigest(),
               'discarded_incomplete_element': incomplete_item,
               'invented_values': 0, 'reordered_or_deduplicated': False}
    return TriplePrefix(items, details)
