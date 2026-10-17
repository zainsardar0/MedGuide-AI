"""Keyless smoke tests for the Medical Agentic Chatbot.

These run in CI without any API keys. Anything needing real keys
(Pinecone retrieval, LLM answers, Tavily) lives in evals/evaluate.py,
which Muhammad runs locally with his own keys.
"""

import json
import math
from pathlib import Path

import pytest

from app.agent import (
    _attach_sources,
    _sources_footer,
    close_graph,
    emergency_respond,
    get_graph,
    greeting_respond,
    guard,
    reject_offtopic,
    validate,
)
from app.config import settings
from app.guardrails import (
    EMERGENCY_FALLBACK,
    GREETING_RESPONSE,
    OFFTOPIC_REFUSAL,
    classify_intent,
    has_citations,
    has_diagnosis_assertion,
    has_emergency_guidance,
    is_emergency,
    is_greeting,
    is_refusal,
)
from app.main import _confidence_payload, app

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden.json"

# Expected pre-classifier bucket per golden case (before the LLM classifier).
EXPECTED_BUCKETS = {
    "emergency-chest-pain": "emergency",
    "emergency-stroke": "emergency",
    "guard-emergency": "emergency",
    "guard-greeting": "greeting",
}


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    """Deterministic without keys: classifier fails open, LLM calls fail."""
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(settings, "pinecone_api_key", "")
    monkeypatch.setattr(settings, "tavily_api_key", "")


@pytest.fixture()
def graph():
    g = get_graph()
    yield g
    close_graph()


def bucket(question):
    if is_emergency(question):
        return "emergency"
    if is_greeting(question):
        return "greeting"
    return "passthrough"


def test_input_gate_buckets():
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert len(cases) == 13
    for case in cases:
        expected = EXPECTED_BUCKETS.get(case["id"], "passthrough")
        assert bucket(case["question"]) == expected, case["id"]


def test_classifier_fails_open_without_keys():
    # A possibly-medical question must never be refused on classifier failure.
    assert classify_intent("Write me a funny poem about cats") == "medical"


def test_emergency_degraded_fallback():
    # No keys -> LLM down -> static fallback, which still carries guidance.
    answer = emergency_respond({"question": "I have crushing chest pain"})["answer"]
    assert answer == EMERGENCY_FALLBACK
    assert has_emergency_guidance(answer)


def test_static_responses():
    assert greeting_respond({})["answer"] == GREETING_RESPONSE
    refusal = reject_offtopic({})["answer"]
    assert refusal == OFFTOPIC_REFUSAL
    assert is_refusal(refusal)


def test_guard_node_buckets():
    assert guard({"question": "I have crushing chest pain", "has_image": False})["intent"] == "emergency"
    assert guard({"question": "Hi there!", "has_image": False})["intent"] == "greeting"
    # Vague caption + image must never be refused as off-topic.
    assert guard({"question": "what is this?", "has_image": True})["intent"] == "medical"


def test_output_validator_flags():
    kb = [{"page": 303, "text": "t", "score": 0.9}]
    # Diagnosis assertion: flagged; regeneration unavailable without keys,
    # so the original answer is kept and the failure recorded.
    bad = {"question": "q", "answer": "You definitely have migraines.", "kb": kb, "web": []}
    out = validate(bad)
    assert out["answer"] == bad["answer"]
    assert "diagnosis_assertion" in out["validation_flags"]
    assert "regeneration_failed" in out["validation_flags"]
    # Clean cited answer passes through untouched.
    good = {"question": "q", "answer": "Possible causes... [PDF p.303]", "kb": kb, "web": []}
    out2 = validate(good)
    assert out2["answer"] == good["answer"]
    assert out2["validation_flags"] == []
    # Missing citations are flagged (the footer normally prevents this).
    out3 = validate({"question": "q", "answer": "No citations here.", "kb": kb, "web": []})
    assert "missing_citations" in out3["validation_flags"]


def test_sources_footer():
    state = {
        "kb": [
            {"page": 303, "text": "t", "score": 0.9},
            {"page": 585, "text": "t", "score": 0.8},
            {"page": 303, "text": "dup", "score": 0.7},
        ],
        "web": [{"title": "Mayo Clinic", "url": "https://example.com/x", "content": "c"}],
    }
    footer = _sources_footer(state)
    assert "[PDF p.303]" in footer and "[PDF p.585]" in footer
    assert footer.count("[PDF p.303]") == 1  # pages deduped
    assert "https://example.com/x" in footer
    assert has_citations(footer)
    # Attached once, never duplicated, untouched when no evidence.
    body = _attach_sources("Migraines hurt.", state)
    assert body.startswith("Migraines hurt.") and "**Sources**" in body
    assert _attach_sources("x\n\n**Sources**\n- y", state) == "x\n\n**Sources**\n- y"
    assert _sources_footer({"kb": [], "web": []}) == ""
    assert _attach_sources("Hi.", {"kb": [], "web": []}) == "Hi."


def test_confidence_payload():
    confident = {"intent": "medical", "route": "answer",
                 "kb": [{"page": 1, "score": 0.85, "text": "t"}], "web": []}
    assert _confidence_payload(confident) == {
        "kb_top_score": 0.85, "route": "answer", "low_confidence": False, "cache_hit": False}

    web_grounded = {"intent": "medical", "route": "web",
                    "kb": [{"page": 1, "score": 0.5, "text": "t"}],
                    "web": [{"title": "x", "url": "u", "content": "c"}]}
    assert _confidence_payload(web_grounded)["low_confidence"] is False

    web_failed = {"intent": "medical", "route": "web",
                  "kb": [{"page": 1, "score": 0.5, "text": "t"}], "web": []}
    assert _confidence_payload(web_failed)["low_confidence"] is True

    no_evidence = {"intent": "medical", "route": "web", "kb": [], "web": []}
    r = _confidence_payload(no_evidence)
    assert r["low_confidence"] is True and r["kb_top_score"] is None

    # Guardrail fast-paths are not graded.
    for intent in ("emergency", "greeting", "off_topic"):
        assert _confidence_payload(
            {"intent": intent, "route": "", "kb": [], "web": []})["low_confidence"] is False


def test_full_graph_emergency_fast_path_no_keys(graph):
    state = graph.invoke(
        {"question": "I have crushing chest pain", "history": [],
         "image_description": "", "kb": [], "web": [], "answer": "", "route": "",
         "intent": "", "has_image": False, "validation_flags": []},
        config={"configurable": {"thread_id": "ci-emg"}},
    )
    assert state["intent"] == "emergency"
    assert state["kb"] == [] and state["web"] == []  # no retrieval spend
    assert has_emergency_guidance(state["answer"])
    assert not has_diagnosis_assertion(state["answer"])


def test_full_graph_greeting_no_keys(graph):
    state = graph.invoke(
        {"question": "Hi there!", "history": [], "image_description": "",
         "kb": [], "web": [], "answer": "", "route": "", "intent": "",
         "has_image": False, "validation_flags": []},
        config={"configurable": {"thread_id": "ci-greet"}},
    )
    assert state["intent"] == "greeting"
    assert state["answer"] == GREETING_RESPONSE


def test_stream_endpoint_registered():
    paths = [r.path for r in app.routes]
    assert "/api/chat/stream" in paths
    assert "/api/chat" in paths


# --- Step 10: semantic cache (all keyless via monkeypatched embeddings) ---

VEC_A = [1.0] + [0.0] * 9
VEC_ORTH = [0.0, 1.0] + [0.0] * 8


def _cache_module(monkeypatch, tmp_path, embed_fn):
    import app.semantic_cache as sc
    monkeypatch.setattr(sc, "_db_path", lambda: str(tmp_path / "cache.db"))
    monkeypatch.setattr(sc, "embed_query", embed_fn)
    return sc


def test_semantic_cache_roundtrip(monkeypatch, tmp_path):
    sc = _cache_module(monkeypatch, tmp_path, lambda t: VEC_A)
    kb = [{"page": 303, "text": "t", "score": 0.9}]
    sc.store_cached("What is a migraine?", "Migraines hurt. [PDF p.303]", kb=kb)
    hit = sc.lookup_cached("  WHAT is a migraine? ")  # normalization: case/space
    assert hit is not None
    answer, hit_kb, hit_web, score = hit
    assert answer == "Migraines hurt. [PDF p.303]"
    assert hit_kb == kb and hit_web == []
    assert score == pytest.approx(1.0)


def test_semantic_cache_miss_and_threshold(monkeypatch, tmp_path):
    sc = _cache_module(
        monkeypatch, tmp_path,
        lambda t: VEC_ORTH if "unrelated" in t else VEC_A,
    )
    sc.store_cached("What is a migraine?", "cached answer")
    # Orthogonal vector: cosine 0 -> miss.
    assert sc.lookup_cached("something completely unrelated") is None
    # Near-miss vector: cosine exactly 0.96 with VEC_A.
    near = [0.96, math.sqrt(1 - 0.96 ** 2)] + [0.0] * 8
    monkeypatch.setattr(sc, "embed_query", lambda t: near)
    assert sc.lookup_cached("what is a migraine?", threshold=0.95) is not None
    assert sc.lookup_cached("what is a migraine?", threshold=0.97) is None


def test_semantic_cache_ttl(monkeypatch, tmp_path):
    import sqlite3
    import time as time_mod

    sc = _cache_module(monkeypatch, tmp_path, lambda t: VEC_A)
    sc.store_cached("What is a migraine?", "cached answer")
    # Age the row beyond the TTL.
    conn = sqlite3.connect(str(tmp_path / "cache.db"))
    conn.execute(
        "UPDATE semantic_cache SET created_at = ?",
        (time_mod.time() - 31 * 86400,),
    )
    conn.commit()
    conn.close()
    assert sc.lookup_cached("what is a migraine?", ttl_days=30) is None


def test_cache_lookup_node_safety(monkeypatch):
    from app.agent import cache_lookup
    import app.semantic_cache as sc

    def boom(t):
        raise AssertionError("must not embed")

    # Image questions never touch the cache.
    monkeypatch.setattr(sc, "embed_query", boom)
    assert cache_lookup(
        {"question": "what is this?", "has_image": True, "intent": "medical"}
    ) == {"cache_hit": False}
    # Embedding failure degrades to a miss, never an exception.
    def down(t):
        raise RuntimeError("llm down")

    monkeypatch.setattr(sc, "embed_query", down)
    assert cache_lookup(
        {"question": "what is migraine?", "has_image": False, "intent": "medical"}
    ) == {"cache_hit": False}
    # Disabled cache never embeds either.
    monkeypatch.setattr(sc, "embed_query", boom)
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    try:
        assert cache_lookup(
            {"question": "q", "has_image": False, "intent": "medical"}
        ) == {"cache_hit": False}
    finally:
        monkeypatch.setattr(settings, "semantic_cache_enabled", True)


def test_full_graph_cache_hit_skips_retrieval(monkeypatch, tmp_path, graph):
    import app.semantic_cache as sc

    _cache_module(monkeypatch, tmp_path, lambda t: VEC_A)
    sc.store_cached("What is a migraine?", "Cached migraine answer. [PDF p.303]")
    state = graph.invoke(
        {"question": "What is a migraine?", "history": [],
         "image_description": "", "kb": [], "web": [], "answer": "", "route": "",
         "intent": "", "has_image": False, "validation_flags": [],
         "cache_hit": False},
        config={"configurable": {"thread_id": "ci-cache"}},
    )
    assert state["cache_hit"] is True
    assert state["answer"] == "Cached migraine answer. [PDF p.303]"
    assert state["kb"] == [] and state["web"] == []  # no retrieval spend
