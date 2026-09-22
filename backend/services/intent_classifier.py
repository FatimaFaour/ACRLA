"""
Intent classification and educational strategy selection.
Every student message is classified before hitting the LLM.

`classify_intent`/`INTENT_PATTERNS` (regex-based) are consumed only by the
legacy rule-based fallback path in `chat_orchestrator.py`.

`select_strategy`/`Strategy`/`get_strategy_instruction`, despite living in
this "intent_classifier" module, are the mastery-aware strategy system the
ACTIVE simple-agent path actually uses (`chat_orchestrator.py`'s agent-context
builder calls `get_strategy_instruction(strategy)` to populate
`context["tutoring_strategy"]["instructions"]`). This is a different, unrelated
system from `services.strategy_selector.TutoringStrategy` (dataclass, has a
`.prompt_block` property, legacy-path only) -- do not assume the two are
interchangeable; a `Strategy` enum member has no `.prompt_block`.
"""

import re
from enum import Enum


class Intent(str, Enum):
    TUTORING = "tutoring"           # explain concept, give example, quiz me
    ANALYTICS = "analytics"         # what am I weak in, how am I doing
    NAVIGATION = "navigation"       # switch topic, go to chapter X
    PREFERENCE = "preference"       # harder questions, change mode
    ENGAGEMENT = "engagement"       # too difficult, I'm confused, I give up
    GREETING = "greeting"           # hi, hello, start session


class Strategy(str, Enum):
    WEAK = "weak"
    WEAK_CONFUSED = "weak_confused"
    MODERATE = "moderate"
    EXCELLENT = "excellent"
    EXAM = "exam"
    RECOVERY = "recovery"


# ── Intent classification ────────────────────────────────────────────────────

INTENT_PATTERNS = {
    Intent.ANALYTICS: [
        r"\bweak\b", r"\bweakest\b", r"\bstrong\b", r"\bstrongest\b",
        r"\bprogress\b", r"\bhow am i\b", r"\bwhat.*wrong\b",
        r"\banalytic\b", r"\bscores?\b", r"\bmastery\b", r"\blevels?\b",
        r"\bperform\b", r"\bperformance\b", r"\bready\b",
        r"\brest\b", r"\bother\b.*\b(course|concept|topic)s?\b", r"\ball\b.*\b(course|concept|topic|mastery)s?\b",
        r"\ball\b.*\btopics\b",
    ],
    Intent.NAVIGATION: [
        r"\bswitch.*(chapter|section)\b", r"\bgo to\b", r"\bchange.*chapter\b", r"\bnext\b",
        r"\bchapter\b", r"\bsection\b",
    ],
    Intent.PREFERENCE: [
        r"\bharder\b", r"\beasier\b", r"\bdifficulty\b", r"\bchange difficulty\b", r"\bdifficult.*question\b",
        r"\bchange mode\b", r"\bswitch (?:to )?(?:external|internal)\b",
        r"\bmore example\b", r"\bfewer\b",
    ],
    Intent.ENGAGEMENT: [
        r"\btoo hard\b", r"\btoo difficult\b", r"\bconfused\b", r"\bdon.t understand\b",
        r"\bgive up\b", r"\bstuck\b", r"\bfrustrat\b", r"\bhelp me\b",
        r"\bi.m lost\b", r"\bnot getting\b",
    ],
    Intent.GREETING: [
        r"^(hi|hello|hey|start|begin|good morning|good afternoon)\b",
    ],
    Intent.TUTORING: [
        r"\bsuggest\b",
        r"\byoutube\b",
        r"\bvideos?\b",
        r"\bwebsite\b",
        r"\bresource\b",
    ],
}


def classify_intent(message: str) -> Intent:
    """Lightweight deterministic intent classifier (no LLM call)."""
    msg = message.lower().strip()

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, msg):
                return intent

    return Intent.TUTORING  # default


# ── Strategy selector ────────────────────────────────────────────────────────

def select_strategy(
    mastery_level: float,
    recent_errors: int,
    is_confused: bool,
    is_exam_mode: bool,
    in_recovery: bool,
) -> Strategy:
    """
    Dynamically selects educational strategy based on student state.
    mastery_level: 0.0 (none) to 1.0 (expert)
    recent_errors: number of wrong answers in last 3 attempts
    """
    if in_recovery:
        return Strategy.RECOVERY

    if is_exam_mode:
        return Strategy.EXAM

    if mastery_level < 0.3:
        if is_confused or recent_errors >= 2:
            return Strategy.WEAK_CONFUSED
        return Strategy.WEAK

    if mastery_level < 0.65:
        return Strategy.MODERATE

    return Strategy.EXCELLENT


# ── Strategy instructions (injected into prompts) ────────────────────────────

STRATEGY_INSTRUCTIONS = {
    Strategy.WEAK: (
        "Use very simple language. Break every concept into small numbered steps. "
        "Avoid jargon. Check understanding after each step."
    ),
    Strategy.WEAK_CONFUSED: (
        "Start with a real-world analogy before explaining the concept. "
        "Use 'This is like...' framing. Keep sentences short."
    ),
    Strategy.MODERATE: (
        "Give medium-difficulty exercises. Explain reasoning behind each step. "
        "Offer one worked example then ask student to try one."
    ),
    Strategy.EXCELLENT: (
        "Challenge with multi-concept questions. Ask the student to explain "
        "concepts back to you. Introduce edge cases and counterexamples."
    ),
    Strategy.EXAM: (
        "Give concise, timed-style answers. No lengthy explanations. "
        "Focus on key facts and formula applications."
    ),
    Strategy.RECOVERY: (
        "Identify the prerequisite concept that is blocking progress. "
        "Teach that prerequisite first before returning to the current topic."
    ),
}


def get_strategy_instruction(strategy: Strategy) -> str:
    return STRATEGY_INSTRUCTIONS.get(strategy, "")
