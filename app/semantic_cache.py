"""SQLite-backed semantic cache for repeat medical questions .

A cache hit serves a past validated answer with no Pinecone, Tavily, or LLM
spend: ~300ms instead of several seconds. Safety rules, chosen for a medical
bot where a wrong cached answer could cause harm:

- only the `medical` intent is cached (never emergency/greeting/off-topic),
- only answers that passed output validation with zero flags are stored,
- only KB-grounded answers (route == 'answer') are stored; web-grounded
  answers stay fresh because guidance changes,
- questions asked with an image are never cached: a vague caption like
  "what is this?" must not match another user's image,
- a match needs cosine similarity >= threshold (default 0.95): paraphrases
  score ~0.97+, genuinely different questions do not,
- entries expire after SEMANTIC_CACHE_TTL_DAYS (default 30).

No new dependencies: embeddings reuse the KB embedding model, similarity is
plain Python, storage is SQLite (same pattern as the conversation memory).
"""

import json
import math
import os
import sqlite3
import time

from app.config import settings


def _db_path():
    return os.environ.get("SEMANTIC_CACHE_DB", "semantic_cache.db")


def _connect():
    path = _db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS semantic_cache ("
        "id INTEGER PRIMARY KEY, "
        "question TEXT NOT NULL, "
        "embedding TEXT NOT NULL, "
        "answer TEXT NOT NULL, "
        "kb_json TEXT NOT NULL DEFAULT '[]', "
        "web_json TEXT NOT NULL DEFAULT '[]', "
        "created_at REAL NOT NULL)"
    )
    return conn


def _normalize(question):
    return (question or "").strip().lower()


def _default_embed(text):
    # Lazy import: app.knowledge raises without API keys, but only when called.
    from app.knowledge import embed
    return embed([text])[0]


# Indirection so tests can monkeypatch embeddings without API keys.
embed_query = _default_embed


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def lookup_cached(question, threshold=None, ttl_days=None):
    """Return (answer, kb, web, similarity) for a close past question, else None.

    `kb`/`web` are the evidence lists the answer was built from, so the UI can
    still render source chips on a cache hit.
    """
    threshold = settings.semantic_cache_threshold if threshold is None else threshold
    ttl_days = settings.semantic_cache_ttl_days if ttl_days is None else ttl_days
    vector = embed_query(_normalize(question))
    cutoff = time.time() - ttl_days * 86400
    best = None  # (score, answer, kb, web)
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT answer, kb_json, web_json, embedding, created_at FROM semantic_cache"
        ).fetchall()
    finally:
        conn.close()
    for answer, kb_raw, web_raw, emb_raw, created_at in rows:
        if created_at < cutoff:
            continue
        score = _cosine(vector, json.loads(emb_raw))
        if score >= threshold and (best is None or score > best[0]):
            best = (score, answer, json.loads(kb_raw), json.loads(web_raw))
    if best is None:
        return None
    return best[1], best[2], best[3], best[0]


def store_cached(question, answer, kb=None, web=None):
    """Remember a validated answer. Prunes expired entries opportunistically."""
    question = _normalize(question)
    if not question or not answer:
        return
    vector = embed_query(question)
    now = time.time()
    cutoff = now - settings.semantic_cache_ttl_days * 86400
    conn = _connect()
    try:
        conn.execute("DELETE FROM semantic_cache WHERE created_at < ?", (cutoff,))
        conn.execute(
            "INSERT INTO semantic_cache (question, embedding, answer, kb_json, web_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (question, json.dumps(vector), answer,
             json.dumps(kb or []), json.dumps(web or []), now),
        )
        conn.commit()
    finally:
        conn.close()
