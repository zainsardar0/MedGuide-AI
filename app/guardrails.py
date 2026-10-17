"""Input/output guardrails for the medical agent.

Design: lightweight and deterministic where safety is critical.
- Emergency detection is regex-based and never depends on an LLM's judgment.
- Greeting detection is an exact-phrase match (no false positives on
  "Hi, what causes migraines?").
- Medical vs off-topic uses one cheap LLM call, failing OPEN to 'medical'
  (a misclassified medical question must never be refused).
- Output validators are shared with the eval suite: evals/evaluate.py
  imports the has_* helpers from here, so production and eval test the
  same logic.
"""

import re

from langchain_openai import ChatOpenAI

from app.config import settings

# --- Input detection --------------------------------------------------------

# Acute red-flag symptoms. Deliberately sensitive: a false positive (info
# question taking the emergency fast-path) is safer than a false negative.
EMERGENCY_DETECTION_PATTERNS = [
    r"chest pain",
    r"heart attack",
    r"\bstroke\b",
    r"face.*droop|droop.*face|facial droop",
    r"slurred speech|speech.*slurr",
    r"can'?t breathe|trouble breathing|difficulty breathing",
    r"short of breath|shortness of breath",
    r"severe bleeding|bleeding heavily|hemorrhage",
    r"unconscious|passed out|fainted|blacked out",
    r"suicid|kill myself|end my life|self.?harm",
    r"overdose",
    r"anaphylaxis|throat.*clos|throat.*swell",
    r"\b911\b|emergency services|call an ambulance",
]

# Exact phrases only, matched against the whole stripped message.
GREETING_PHRASES = {
    "hi", "hey", "hello", "hi there", "hey there",
    "good morning", "good afternoon", "good evening",
    "salam", "salaam", "assalamualaikum",
    "thanks", "thank you", "bye", "goodbye", "good night",
}

_CLASSIFIER_PROMPT = (
    "Classify the user's message into exactly one category. Reply with only the category name, nothing else.\n"
    "- medical: a health, medical, or wellness question, a description of symptoms, or a request for health information\n"
    "- off_topic: anything not related to health (jokes, poems, homework, general knowledge, coding help, etc.)\n"
    'Message: "{question}"\n'
    "Category:"
)


def is_emergency(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in EMERGENCY_DETECTION_PATTERNS)


def is_greeting(text: str) -> bool:
    return text.strip().lower().rstrip("!. ").strip() in GREETING_PHRASES


def classify_intent(question: str) -> str:
    """Return 'medical' or 'off_topic'. Fails open to 'medical'."""
    if not settings.openai_api_key:
        return "medical"
    try:
        llm = ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0,
            api_key=settings.openai_api_key,
            timeout=20,
            max_retries=1,
        )
        out = llm.invoke(_CLASSIFIER_PROMPT.format(question=question[:500])).content.strip().lower()
        first = out.split()[0] if out.split() else ""
        if first in ("medical", "off_topic"):
            return first
        if first.startswith("off"):
            return "off_topic"
        return "medical"
    except Exception as exc:  # noqa: BLE001 - never block a question on classifier failure
        print(f"Intent classifier failed, failing open to medical: {exc!r}", flush=True)
        return "medical"


# --- Output validation ------------------------------------------------------

CITATION_RE = re.compile(r"\[PDF p\.\d+\]|https?://\S+")

DIAGNOSIS_PATTERNS = [
    r"you have been diagnosed",
    r"you (definitely|certainly) have",
    r"i (diagnose|confirm that you have)",
]

EMERGENCY_GUIDANCE_PATTERNS = [
    r"emergency",
    r"\b911\b",
    r"emergency services",
    r"call.*emergency",
    r"seek.*immediate",
    r"emergency room|emergency department",
]

REFUSAL_PATTERNS = [
    r"can't help with that",
    r"medical education assistant",
]


def has_citations(text: str) -> bool:
    return bool(CITATION_RE.search(text))


def has_emergency_guidance(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in EMERGENCY_GUIDANCE_PATTERNS)


def has_diagnosis_assertion(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in DIAGNOSIS_PATTERNS)


def is_refusal(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in REFUSAL_PATTERNS)


# --- Canned safe responses --------------------------------------------------

EMERGENCY_FALLBACK = (
    "This could be a medical emergency. Please call emergency services "
    "(911 or your local emergency number) or go to the nearest emergency room "
    "immediately \u2014 do not wait. I can answer health-education questions afterwards."
)

GREETING_RESPONSE = (
    "Hello! I'm MediAgent, a medical education assistant. Ask me a health question, "
    "describe symptoms, or upload an image and I'll explain what I find with cited sources. "
    "Note: I'm not a clinician \u2014 for emergencies, call emergency services immediately."
)

OFFTOPIC_REFUSAL = (
    "I'm a medical education assistant, so I can't help with that. If you have a "
    "health-related question \u2014 about symptoms, conditions, medications, or wellness \u2014 "
    "I'll do my best to answer with cited sources."
)

SAFE_FALLBACK = (
    "I wasn't able to generate a safe answer to that question. Please consult a "
    "qualified healthcare professional, or call emergency services if this is urgent."
)
