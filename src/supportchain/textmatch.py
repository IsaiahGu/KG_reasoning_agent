"""Small text-matching helpers shared by the chain agent, the binding scorer and the reports."""
import re
import unicodedata


def tokens(text):
    return re.sub(r'[\W_]+', ' ', unicodedata.normalize('NFKC', str(text)).casefold()).split()


def contains(haystack, needle):
    """Token-level contiguous containment, so that "2" does not match "2012"."""
    hay, pin = tokens(haystack), tokens(needle)
    if not pin or len(pin) > len(hay):
        return False
    return any(hay[start:start + len(pin)] == pin for start in range(len(hay) - len(pin) + 1))


def same_answer(left, right):
    return contains(left, right) or contains(right, left)


def clean_answer(response):
    """Second extraction rule, reported next to the upstream one for every system: last 'Answer:' segment, first line."""
    tail = response.split('Answer:')[-1] if 'Answer:' in response else response
    lines = [line.strip() for line in tail.strip().splitlines() if line.strip()]
    return lines[0].strip(' "\'').rstrip('.').strip(' "\'') if lines else ''
