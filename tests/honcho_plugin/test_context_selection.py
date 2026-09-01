from __future__ import annotations

import unicodedata
from unittest.mock import MagicMock

import pytest

from plugins.memory.honcho import context_selection as selection
from plugins.memory.honcho.context_selection import (
    assemble_context_layers,
    select_complete_context,
)
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSession, HonchoSessionManager
from plugins.memory.honcho import HonchoMemoryProvider


FACILITY = "Pinewood Athletic Complex is the possible future tournament location."
CONCERN = "The unresolved concern is insufficient shaded spectator seating."


def semantic_pinewood_match(text: str) -> bool:
    value = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    facility = "pinewood athletic complex" in value
    insufficiency = any(term in value for term in ("insufficient", "inadequate", "not enough"))
    shade = any(term in value for term in ("shade", "shaded"))
    spectators = any(term in value for term in ("spectator", "spectators", "seating", "seat"))
    return facility and insufficiency and shade and spectators


def test_final_sentence_fits_exactly_at_budget():
    text = "Alpha is complete."
    assert select_complete_context(text, "alpha", len(text)) == text


def test_final_sentence_exceeds_budget_by_one_is_omitted_not_sliced():
    first = "Alpha is complete."
    second = "Beta remains complete."
    text = f"{first} {second}"
    result = select_complete_context(text, "alpha beta", len(text) - 1)
    assert result in {first, second}
    assert "…" not in result
    assert result.endswith(".")


def test_multi_sentence_content_prefers_query_relevant_complete_facts():
    filler = "General background is available."
    text = f"{filler} {FACILITY} {CONCERN}"
    result = select_complete_context(text, "facility unresolved concern tournament", 180)
    assert FACILITY in result
    assert CONCERN in result
    assert result.endswith(".")


def test_oversized_low_relevance_item_does_not_crowd_out_relevant_fact():
    oversized = "Unrelated " + ("background " * 80) + "ends."
    result = select_complete_context(
        f"{oversized} {CONCERN}",
        "unresolved concern shaded seating",
        len(CONCERN) + 5,
    )
    assert result == CONCERN


def test_duplicate_and_near_duplicate_facts_consume_budget_once():
    duplicate = "The unresolved concern is insufficient shaded spectator seating."
    near_duplicate = "The unresolved concern was insufficient shaded spectator seating."
    result = select_complete_context(
        f"{duplicate} {duplicate} {near_duplicate} {FACILITY}",
        "facility unresolved concern",
        220,
    )
    assert result.count("insufficient shaded spectator seating") == 1
    assert FACILITY in result


def test_nfkc_matching_preserves_valid_unicode():
    decomposed = "Cafe\u0301 planning remains relevant."
    result = select_complete_context(decomposed, "Café planning", 80)
    assert unicodedata.normalize("NFC", result) == "Café planning remains relevant."


def test_unicode_emoji_is_not_split_or_malformed():
    text = "Spectator shade is limited ☀️. Parking is adequate."
    result = select_complete_context(text, "spectator shade", 35)
    assert result == "Spectator shade is limited ☀️."
    result.encode("utf-8")


def test_empty_dialectic_response_returns_empty():
    assert select_complete_context("", "query", 600) == ""
    assert select_complete_context("   ", "query", 600) == ""


def test_very_short_cap_returns_only_complete_item_that_fits():
    assert select_complete_context("One. Two.", "two", 4) == "Two."


def test_no_complete_sentence_can_fit_returns_empty():
    assert select_complete_context("A complete sentence.", "sentence", 3) == ""


def test_layer_assembly_preserves_dialectic_and_total_budget():
    base = "Background fact. " * 900
    dialectic = f"{FACILITY} {CONCERN}"
    result = assemble_context_layers(
        base_context=base,
        dialectic_context=dialectic,
        notice_context="",
        max_chars=12000,
    )
    assert len(result) <= 12000
    assert FACILITY in result
    assert CONCERN in result
    assert not result.endswith("…")


def test_pinewood_regression_keeps_facility_and_complete_concern_under_600():
    prefix = "General policy detail is less relevant. " * 25
    text = f"{prefix}{FACILITY} {CONCERN}"
    assert len(text) > 600
    result = select_complete_context(
        text,
        "What facility and unresolved concern did we identify?",
        600,
    )
    assert len(result) <= 600
    assert FACILITY in result
    assert CONCERN in result
    assert "insufficient shaded …" not in result
    assert result.endswith(".")


def test_dialectic_query_uses_complete_relevance_selection_at_configured_cap(monkeypatch):
    prefix = "General policy detail is less relevant. " * 25
    raw = f"{prefix}{FACILITY} {CONCERN}"
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        dialectic_max_chars=600,
    )
    manager = HonchoSessionManager(honcho=MagicMock(), config=config)
    manager._cache["synthetic"] = HonchoSession(
        key="synthetic",
        user_peer_id="synthetic-user",
        assistant_peer_id="synthetic-ai",
        honcho_session_id="synthetic-session",
    )
    monkeypatch.setattr(manager, "_authed_call", lambda _name, _call: raw)

    result = manager.dialectic_query(
        "synthetic",
        "What facility and unresolved concern did we identify?",
    )

    assert len(result) <= 600
    assert FACILITY in result
    assert CONCERN in result
    assert "insufficient shaded …" not in result


def test_production_depth_path_uses_original_query_with_unique_filler(monkeypatch):
    filler_words = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
        "mike november oscar papa quebec romeo sierra tango"
    ).split()
    filler = " ".join(
        f"General {word} policy detail remains less relevant."
        for word in filler_words
    )
    raw = f"{filler} {FACILITY} {CONCERN}"
    config = HonchoClientConfig(
        api_key="synthetic-test-key",
        enabled=True,
        dialectic_max_chars=600,
    )
    manager = HonchoSessionManager(honcho=MagicMock(), config=config)
    manager._cache["synthetic"] = HonchoSession(
        key="synthetic",
        user_peer_id="synthetic-user",
        assistant_peer_id="synthetic-ai",
        honcho_session_id="synthetic-session",
    )
    monkeypatch.setattr(manager, "_authed_call", lambda _name, _call: raw)
    provider = HonchoMemoryProvider()
    provider._manager = manager
    provider._session_key = "synthetic"
    provider._base_context_cache = ""
    provider._dialectic_depth = 1
    provider._query_rewrite_enabled = False

    result = provider._run_dialectic_depth(
        "What facility and unresolved concern did we identify?"
    )

    assert len(result) <= 600
    assert FACILITY in result
    assert CONCERN in result


def test_numbered_near_duplicates_do_not_crowd_out_pinewood():
    duplicates = " ".join(
        f"General policy detail item {index} remains less relevant."
        for index in range(1, 25)
    )
    raw = f"{duplicates} {FACILITY} {CONCERN}"
    result = select_complete_context(
        raw,
        "What facility and unresolved concern did we identify?",
        600,
    )

    assert len(result) <= 600
    assert FACILITY in result
    assert CONCERN in result
    assert result.count("General policy detail item") <= 1


@pytest.mark.parametrize("unit_count", [100, 500, 1000, 2000, 5000])
def test_unique_unit_comparison_and_retention_are_bounded(unit_count):
    noise = " ".join(
        f"Unique noise_token_{index} background remains irrelevant."
        for index in range(unit_count)
    )
    raw = f"{noise} {FACILITY} {CONCERN}"
    stats = {}
    result = select_complete_context(
        raw,
        "What facility and unresolved concern did we identify?",
        600,
        _stats=stats,
    )

    limit = selection.candidate_limit_for_budget(600)
    assert stats["near_comparisons"] <= limit * (limit - 1) // 2
    assert stats["ranked_candidates_retained"] <= limit
    assert stats["near_signatures_retained"] <= limit
    assert stats["exact_keys_retained"] <= selection.MAX_EXACT_KEYS
    assert len(result) <= 600
    assert FACILITY in result
    assert CONCERN in result


def test_comparison_growth_is_not_quadratic_when_units_double():
    comparisons = []
    for unit_count in (1000, 2000):
        raw = " ".join(
            f"Unique overlap_token_{index} policy context remains irrelevant."
            for index in range(unit_count)
        )
        stats = {}
        select_complete_context(raw, "facility concern", 600, _stats=stats)
        comparisons.append(stats["near_comparisons"])
    assert comparisons[1] <= comparisons[0] * 2


def test_exact_duplicates_use_hash_lookup_without_expanding_near_work():
    duplicate = "Repeated neutral policy statement."
    raw = " ".join([duplicate] * 5000 + [FACILITY, CONCERN])
    stats = {}
    result = select_complete_context(
        raw,
        "facility unresolved concern",
        600,
        _stats=stats,
    )
    assert stats["exact_duplicate_rejections"] >= 4999
    assert stats["exact_keys_retained"] <= 3
    assert stats["ranked_candidates_retained"] <= 3
    assert FACILITY in result and CONCERN in result


@pytest.mark.parametrize("pinewood_first", [True, False])
def test_pinewood_survives_thousands_of_noise_units_regardless_of_position(pinewood_first):
    noise = " ".join(
        f"Unique adversarial_token_{index} material overlaps policy context."
        for index in range(3000)
    )
    facts = f"{FACILITY} {CONCERN}"
    raw = f"{facts} {noise}" if pinewood_first else f"{noise} {facts}"
    stats = {}
    result = select_complete_context(
        raw,
        "What facility and unresolved concern did we identify?",
        600,
        _stats=stats,
    )
    assert FACILITY in result and CONCERN in result
    assert stats["ranked_candidates_retained"] <= selection.candidate_limit_for_budget(600)


def test_pathological_near_duplicate_buckets_remain_bounded():
    raw = " ".join(
        f"Shared alpha beta gamma delta variant_token_{index} remains relevant."
        for index in range(5000)
    )
    stats = {}
    result = select_complete_context(raw, "alpha beta gamma", 600, _stats=stats)
    limit = selection.candidate_limit_for_budget(600)
    assert stats["near_comparisons"] <= limit * (limit - 1) // 2
    assert stats["near_signatures_retained"] <= limit
    assert len(result) <= 600


def test_semantic_pinewood_acceptance_examples():
    passing = [
        "Pinewood Athletic Complex: insufficient shaded spectator seating.",
        "Pinewood Athletic Complex had insufficient shaded areas for spectators.",
        "Pinewood Athletic Complex has not enough shaded seating for spectators.",
        "Pinewood Athletic Complex has inadequate shade for spectator seating.",
    ]
    failing = [
        "Pinewood Athletic Complex has plenty of shaded spectator seating.",
        "Pinewood Athletic Complex has insufficient parking.",
        "Pinewood Athletic Complex spectator seating is excellent.",
        "Pinewood Athletic Complex only.",
    ]
    assert all(semantic_pinewood_match(item) for item in passing)
    assert not any(semantic_pinewood_match(item) for item in failing)
