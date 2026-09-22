"""Deterministic pedagogical state machine for personalized tutoring.

Reads/writes tutor state through `services.memory_manager.MemoryManager`
only, in `StudentLongTermMemory.learning_state` (no new DB columns -- see
`get_course_memory`/`update_course_memory`). This module never inspects
message text and never touches mastery -- it has no import of, and never
calls, `MemoryManager.set_mastery`/`update_mastery`. Every transition is
driven by (a) the semantic planner's `tutor_signal` field
(`agents.simple_planner`/`agents.agent_models` -- classified by meaning, the
same way `goal` already is, never keyword-matched) and (b) the deterministic
`AdaptivePolicy` outcome of an answer evaluation (`services.error_analyzer`).

    TutorState.EXPLAIN / EXAMPLE / GUIDED_PRACTICE / PROGRESS_CHECK_READY
        -- the four "resting" states, persisted between turns.
    TutorState.FEEDBACK / RETRY_OR_NEXT
        -- transient: computed and consumed within the single turn a
           practice answer is evaluated, never persisted as a standalone
           value (see `tools.tutor_state_tools.evaluate_practice_answer_tool`).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any


class TutorState(str, Enum):
    EXPLAIN = "EXPLAIN"
    EXAMPLE = "EXAMPLE"
    GUIDED_PRACTICE = "GUIDED_PRACTICE"
    FEEDBACK = "FEEDBACK"
    RETRY_OR_NEXT = "RETRY_OR_NEXT"
    PROGRESS_CHECK_READY = "PROGRESS_CHECK_READY"


PERSISTABLE_STATES = frozenset({
    TutorState.EXPLAIN, TutorState.EXAMPLE,
    TutorState.GUIDED_PRACTICE, TutorState.PROGRESS_CHECK_READY,
})

# correct+confident at this many completed rounds (this one included) moves
# to PROGRESS_CHECK_READY instead of escalating difficulty again.
ROUNDS_BEFORE_PROGRESS_CHECK = 3
# consecutive wrong answers (this one included) that send the student back
# to EXAMPLE instead of another guided-practice retry.
REPEATED_WRONG_THRESHOLD = 2

_DIFFICULTY_ORDER = ["easy", "medium", "hard"]


class RetryDecision:
    """Pure result of `AdaptivePolicy.decide` -- no I/O, no DB, no mastery."""

    def __init__(self, next_state: TutorState, difficulty: str, reason: str, reset_rounds: bool = False):
        self.next_state = next_state
        self.difficulty = difficulty
        self.reason = reason
        self.reset_rounds = reset_rounds


class AdaptivePolicy:
    """The 4-branch adaptive policy governing RETRY_OR_NEXT.

    - correct & confident -> escalate difficulty, next subskill/round (or
      PROGRESS_CHECK_READY once enough rounds are done).
    - correct & not confident -> same difficulty, another guided practice.
    - wrong (first time on this concept) -> hint, same difficulty, retry.
    - wrong (2+ consecutive on this concept) -> lower difficulty, back to
      EXAMPLE, reset the round count for this concept.
    """

    @staticmethod
    def decide(
        *, correct: bool, confident: bool, difficulty: str,
        consecutive_wrong: int, rounds_completed: int,
    ) -> RetryDecision:
        idx = _DIFFICULTY_ORDER.index(difficulty) if difficulty in _DIFFICULTY_ORDER else 1

        if correct and confident:
            if rounds_completed + 1 >= ROUNDS_BEFORE_PROGRESS_CHECK:
                return RetryDecision(TutorState.PROGRESS_CHECK_READY, difficulty, "correct_confident_ready_for_check")
            next_difficulty = _DIFFICULTY_ORDER[min(idx + 1, len(_DIFFICULTY_ORDER) - 1)]
            return RetryDecision(TutorState.GUIDED_PRACTICE, next_difficulty, "correct_confident_escalate")

        if correct and not confident:
            return RetryDecision(TutorState.GUIDED_PRACTICE, difficulty, "correct_weak_explanation_same_level")

        if consecutive_wrong + 1 >= REPEATED_WRONG_THRESHOLD:
            lowered = _DIFFICULTY_ORDER[max(idx - 1, 0)]
            return RetryDecision(TutorState.EXAMPLE, lowered, "repeated_wrong_reteach", reset_rounds=True)
        return RetryDecision(TutorState.GUIDED_PRACTICE, difficulty, "wrong_hint_retry")


_MAX_ASKED_VARIANTS_PER_DIFFICULTY = 20


def _clean_asked_variants(raw: dict[str, Any] | None) -> dict[str, list[int]]:
    cleaned: dict[str, list[int]] = {}
    for difficulty, indices in (raw or {}).items():
        if difficulty not in _DIFFICULTY_ORDER or not isinstance(indices, list):
            continue
        cleaned[difficulty] = [int(i) for i in indices if isinstance(i, (int, float))][-_MAX_ASKED_VARIANTS_PER_DIFFICULTY:]
    return cleaned


def load_tutor_state(memory, student_id: str, course_id: str, session_id: str) -> dict[str, Any]:
    """Read the session's current resting tutor state, or `{}` if none is
    active or it belongs to a different session (a new session starts fresh;
    error patterns, tracked separately, are not session-scoped)."""
    state = memory.get_tutor_state(student_id, course_id)
    if not state or state.get("session_id") != session_id:
        return {}
    raw_state = state.get("state")
    if raw_state not in {s.value for s in PERSISTABLE_STATES}:
        return {}
    return {
        "session_id": session_id,
        "concept": state.get("concept"),
        "state": raw_state,
        "difficulty": state.get("difficulty") or "medium",
        "rounds_completed": int(state.get("rounds_completed") or 0),
        "consecutive_wrong": int(state.get("consecutive_wrong") or 0),
        "current_question": state.get("current_question"),
        "asked_variants": _clean_asked_variants(state.get("asked_variants")),
    }


def save_tutor_state(
    memory, student_id: str, course_id: str, session_id: str, *,
    state: str, concept: str | None, difficulty: str,
    rounds_completed: int = 0, consecutive_wrong: int = 0,
    current_question: str | None = None, asked_variants: dict[str, Any] | None = None,
) -> None:
    """Write a new resting tutor state. `state` must be one of
    `PERSISTABLE_STATES`'s values -- FEEDBACK/RETRY_OR_NEXT are transient and
    never persisted (see module docstring)."""
    if state not in {s.value for s in PERSISTABLE_STATES}:
        raise ValueError(f"cannot persist transient tutor state: {state}")
    memory.set_tutor_state(student_id, course_id, {
        "session_id": session_id,
        "concept": concept,
        "state": state,
        "difficulty": difficulty if difficulty in _DIFFICULTY_ORDER else "medium",
        "rounds_completed": max(0, int(rounds_completed)),
        "consecutive_wrong": max(0, int(consecutive_wrong)),
        "current_question": current_question,
        "asked_variants": _clean_asked_variants(asked_variants),
        "updated_at": datetime.utcnow().isoformat(),
    })


def record_error_pattern(memory, student_id: str, course_id: str, concept: str, error_type: str) -> None:
    memory.record_tutor_error_pattern(student_id, course_id, concept, error_type)


def top_error_pattern(memory, student_id: str, course_id: str, concept: str) -> dict[str, Any] | None:
    """Most frequently recorded error type for this concept, if any."""
    patterns = memory.get_tutor_error_patterns(student_id, course_id, concept)
    if not patterns:
        return None
    return max(patterns, key=lambda entry: entry.get("count") or 0)
