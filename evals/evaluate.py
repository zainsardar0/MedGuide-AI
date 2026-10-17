"""Evaluation suite for the Medical Agentic Chatbot.

Runs the golden question set (evals/golden.json) through the agent graph and checks:
- intent: guardrail input gate classifies emergency/greeting/off_topic/medical (FAIL-level)
- no_rag: guardrail fast-paths must skip retrieval + web search (FAIL-level)
- refusal: off-topic questions must get a polite refusal (WARN-level)
- route correctness: current-guideline questions must take the 'web' route (FAIL-level)
- citation presence: [PDF p.N] or source URLs when the KB was used (WARN-level)
- emergency triage: emergency questions must advise emergency services (FAIL-level)
- topic coverage: expected must-mention terms appear in the answer (WARN-level)
- no diagnosis assertions: scan for overconfident diagnostic language (WARN-level)

The citation/emergency/diagnosis/refusal checks are imported from app.guardrails,
so the eval and the production output validator test the same logic.

Severity: only intent/no_rag/route/emergency/crash failures FAIL the suite.
Citation, coverage, refusal and diagnosis gaps are reported as WARN signal.

Usage:
    python evals/evaluate.py            # run all cases (needs API keys in .env)
    python evals/evaluate.py --list     # list cases without running anything
    python evals/evaluate.py --case kb-migraine   # run a single case

Results are written to evals/results.json. Exit code is 1 if any case FAILs.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as `python evals/evaluate.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import get_graph, close_graph
from app.guardrails import (
    has_citations,
    has_emergency_guidance,
    has_diagnosis_assertion,
    is_refusal,
)

ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "golden.json"
RESULTS = ROOT / "results.json"

# Checks in this set FAIL the suite; everything else is WARN-level signal.
HARD_CHECKS = {"intent", "no_rag", "route", "emergency", "error"}


def load_cases():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def check_intent(state, expected):
    if not expected:
        return ("SKIP", "no expectation")
    actual = state.get("intent")
    return ("PASS", f"intent={actual}") if actual == expected else ("FAIL", f"expected {expected}, got {actual}")


def check_no_rag(state, expect):
    if not expect:
        return ("SKIP", "not required")
    used = bool(state.get("kb")) or bool(state.get("web"))
    return ("PASS", "no retrieval/web spend") if not used else ("FAIL", "guardrail path ran retrieval or web search")


def check_refusal(answer, expect):
    if not expect:
        return ("SKIP", "not required")
    return ("PASS", "polite refusal") if is_refusal(answer) else ("WARN", "expected a polite off-topic refusal")


def check_route(state, expected):
    if not expected:
        return ("SKIP", "no expectation")
    actual = state.get("route")
    return ("PASS", f"route={actual}") if actual == expected else ("FAIL", f"expected {expected}, got {actual}")


def check_citations(answer, kb, expect):
    if not expect:
        return ("SKIP", "not required")
    if not kb:
        return ("SKIP", "KB not used")
    found = has_citations(answer)
    return ("PASS", "citation(s) present") if found else ("WARN", "KB used but no [PDF p.N]/URL citations in answer")


def check_emergency(answer, expect):
    if not expect:
        return ("SKIP", "not required")
    hit = has_emergency_guidance(answer)
    return ("PASS", "emergency guidance present") if hit else ("FAIL", "no emergency guidance found in answer")


def check_coverage(answer, must_mention):
    if not must_mention:
        return ("SKIP", "no terms")
    low = answer.lower()
    missing = [t for t in must_mention if t.lower() not in low]
    cov = (len(must_mention) - len(missing)) / len(must_mention)
    return ("PASS" if not missing else "WARN", f"{cov:.0%} coverage, missing={missing}")


def check_no_diagnosis(answer):
    return ("WARN", "diagnosis assertion detected") if has_diagnosis_assertion(answer) else ("PASS", "clean")


def run_case(graph, case):
    thread_id = f"eval-{case['id']}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    state = graph.invoke(
        {
            "question": case["question"],
            "history": [],
            "image_description": "",
            "kb": [],
            "web": [],
            "answer": "",
            "route": "",
            "intent": "",
            "has_image": False,
            "validation_flags": [],
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = state.get("answer", "")
    kb = state.get("kb", [])
    results = {
        "intent": check_intent(state, case.get("expect_intent")),
        "no_rag": check_no_rag(state, case.get("expect_no_rag")),
        "refusal": check_refusal(answer, case.get("expect_refusal")),
        "route": check_route(state, case.get("expect_route")),
        "citations": check_citations(answer, kb, case.get("expect_citation")),
        "emergency": check_emergency(answer, case.get("expect_emergency_guidance")),
        "coverage": check_coverage(answer, case.get("must_mention", [])),
        "no_diagnosis": check_no_diagnosis(answer),
        "kb_top_score": kb[0]["score"] if kb else None,
        "tools": (["pinecone"] if kb else []) + (["tavily"] if state.get("route") == "web" else []),
        "answer_chars": len(answer),
        "validation_flags": state.get("validation_flags", []),
    }
    return state, results


def summarize(results):
    """FAIL only on hard checks; WARNs are signal, not failure."""
    for key, value in results.items():
        if isinstance(value, tuple) and key in HARD_CHECKS and value[0] == "FAIL":
            return "FAIL"
    return "PASS"


def main():
    ap = argparse.ArgumentParser(description="Evaluate the medical agent on the golden question set.")
    ap.add_argument("--list", action="store_true", help="list cases without running")
    ap.add_argument("--case", help="run a single case by id")
    args = ap.parse_args()

    cases = load_cases()
    if args.list:
        for c in cases:
            print(f"{c['id']:24} [{c.get('type', '-') :9}] {c['question']}")
        return

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            sys.exit(f"unknown case id: {args.case}")

    graph = get_graph()
    try:
        report = {"run_at": datetime.now(timezone.utc).isoformat(), "cases": []}
        failed = 0
        for case in cases:
            try:
                _, results = run_case(graph, case)
                verdict = summarize(results)
            except Exception as exc:  # noqa: BLE001 - eval harness must not crash on one bad case
                verdict = "FAIL"
                results = {"error": ("FAIL", repr(exc))}
            if verdict == "FAIL":
                failed += 1
            report["cases"].append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "verdict": verdict,
                    "checks": {
                        k: {"status": v[0], "detail": v[1]}
                        for k, v in results.items()
                        if isinstance(v, tuple)
                    },
                    "kb_top_score": results.get("kb_top_score"),
                    "tools": results.get("tools"),
                    "answer_chars": results.get("answer_chars"),
                    "validation_flags": results.get("validation_flags", []),
                }
            )
            checks_str = " ".join(f"{k}={v[0]}" for k, v in results.items() if isinstance(v, tuple))
            print(f"[{verdict:4}] {case['id']:24} {checks_str}")
        RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n{failed} failed / {len(report['cases'])} total — full report in evals/results.json")
        sys.exit(1 if failed else 0)
    finally:
        close_graph()


if __name__ == "__main__":
    main()
