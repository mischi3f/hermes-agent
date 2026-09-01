"""Bounded, complete-unit selection for Honcho automatic context injection."""

from __future__ import annotations

import heapq
import re
import unicodedata
from collections.abc import Iterator

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r".*?(?:[.!?](?=\s|$)|$)", re.DOTALL)
_TIMESTAMP_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "do",
    "for",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "was",
    "we",
    "what",
}
_QUERY_ALIASES = {
    "facility": {"complex", "location", "site", "venue"},
    "concern": {"issue", "problem", "unresolved"},
}

MAX_EXACT_KEYS = 4096
MAX_RANKED_CANDIDATES = 512
_MIN_COMPLETE_UNIT_CHARS = 2
_MIN_RETAINED_CANDIDATES = 16


def candidate_limit_for_budget(max_chars: int) -> int:
    """Bound retained candidates from output capacity, with a strict ceiling."""
    if max_chars <= 0:
        return 0
    capacity = max_chars // _MIN_COMPLETE_UNIT_CHARS + 8
    return min(MAX_RANKED_CANDIDATES, max(_MIN_RETAINED_CANDIDATES, capacity))


def _tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(
        token
        for token in _WORD_RE.findall(normalized)
        if token not in _STOPWORDS and not token.isdigit()
    )


def _iter_complete_units(text: str) -> Iterator[str]:
    """Yield normalized complete sentence/line units without retaining all input."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).replace("\r\n", "\n")
    for raw_line in normalized.splitlines() or [normalized]:
        line = " ".join(raw_line.split())
        if not line:
            continue
        yielded = False
        for match in _SENTENCE_RE.finditer(line):
            item = match.group(0).strip()
            if item and item[-1:] in ".!?":
                yielded = True
                yield item
        if not yielded and line.startswith(("- ", "* ", "• ")):
            yield line


def _canonical_tokens(text: str) -> frozenset[str]:
    without_timestamp = _TIMESTAMP_PREFIX_RE.sub("", text)
    return frozenset(_tokens(without_timestamp))


def _near_duplicate(
    candidate: frozenset[str],
    retained: list[frozenset[str]],
    stats: dict[str, int],
) -> bool:
    if not candidate:
        return False
    for previous in retained:
        stats["near_comparisons"] += 1
        union = candidate | previous
        if union and len(candidate & previous) / len(union) >= 0.85:
            return True
    return False


def select_complete_context(
    text: str,
    query: str,
    max_chars: int,
    *,
    _stats: dict[str, int] | None = None,
) -> str:
    """Select complete relevant units with input-independent dedup comparison bounds."""
    stats = _stats if _stats is not None else {}
    stats.update(
        {
            "input_units": 0,
            "exact_checks": 0,
            "exact_duplicate_rejections": 0,
            "exact_keys_retained": 0,
            "near_comparisons": 0,
            "near_signatures_retained": 0,
            "ranked_candidates_retained": 0,
        }
    )
    limit = candidate_limit_for_budget(max_chars)
    if limit == 0:
        return ""

    query_tokens = set(_tokens(query))
    for token in tuple(query_tokens):
        query_tokens.update(_QUERY_ALIASES.get(token, ()))

    exact_keys: set[frozenset[str]] = set()
    # Heap quality is (score, -index): lower score/later item is evicted first.
    heap: list[tuple[int, int, int, str, frozenset[str]]] = []

    for index, unit in enumerate(_iter_complete_units(text)):
        stats["input_units"] += 1
        if len(unit) > max_chars:
            continue
        canonical = _canonical_tokens(unit)
        stats["exact_checks"] += 1
        if canonical in exact_keys:
            stats["exact_duplicate_rejections"] += 1
            continue
        if len(exact_keys) < MAX_EXACT_KEYS:
            exact_keys.add(canonical)
        overlap = len(query_tokens & set(_tokens(unit)))
        candidate = (overlap, -index, index, unit, canonical)
        if len(heap) < limit:
            heapq.heappush(heap, candidate)
        elif candidate[:2] > heap[0][:2]:
            heapq.heapreplace(heap, candidate)

    stats["exact_keys_retained"] = len(exact_keys)
    stats["ranked_candidates_retained"] = len(heap)
    ranked = sorted(heap, key=lambda item: (-item[0], item[2]))

    near_signatures: list[frozenset[str]] = []
    selected: list[str] = []
    used = 0
    for _score, _negative_index, _index, unit, canonical in ranked:
        if _near_duplicate(canonical, near_signatures, stats):
            continue
        near_signatures.append(canonical)
        separator = 1 if selected else 0
        required = separator + len(unit)
        if required > max_chars - used:
            continue
        selected.append(unit)
        used += required

    stats["near_signatures_retained"] = len(near_signatures)
    return " ".join(selected)


def assemble_context_layers(
    *,
    base_context: str,
    dialectic_context: str,
    notice_context: str,
    max_chars: int,
) -> str:
    """Reserve budget for complete dialectic facts, then fit lower-priority layers."""
    if max_chars <= 0:
        return ""

    dialectic = select_complete_context(dialectic_context, "", max_chars)
    used = len(dialectic)

    notice_budget = max(0, max_chars - used - (2 if dialectic else 0))
    notice = select_complete_context(notice_context, "", notice_budget)
    used += len(notice) + (2 if notice and dialectic else 0)

    separators = 0
    if dialectic:
        separators += 2
    if notice:
        separators += 2
    base_budget = max(0, max_chars - used - separators)
    base = select_complete_context(base_context, "", base_budget)

    ordered = [part for part in (notice, base, dialectic) if part]
    result = "\n\n".join(ordered)
    if len(result) > max_chars:
        raise AssertionError("context assembly exceeded configured budget")
    return result
