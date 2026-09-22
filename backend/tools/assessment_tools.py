"""Quick Progress Check, launched from chat (goal=assessment).

`run_quick_progress_check_tool` is a short, session-scoped multi-turn flow:
select 3-5 concepts practiced this session, ask one question per concept at
the student's current difficulty (from the deterministic question bank --
never an LLM call), judge each answer, and once every concept has been
asked, update mastery and report a summary. Answer judging is deterministic
for MCQ (easy) questions -- a plain letter comparison, no LLM call -- and
only falls back to `services.error_analyzer.judge_answer` (one LLM call) for
open-ended medium/hard questions, where there is no fixed answer to compare
against. Combined with the fully template-based summary below and
`agents.simple_agent`'s planner-call skip while a check is pending, an
all-easy-difficulty Quick Progress Check costs a single LLM call in total
(the initial "start a check" classification) instead of one call per
question. State lives in
`StudentLongTermMemory.learning_state` via
`MemoryManager.get_quick_progress_check`/`set_quick_progress_check` (the same
storage mechanism `services.tutor_state_machine` uses for tutor state),
session-scoped so a stale/abandoned check from a different session is
treated as absent.

This is the ONLY chat-tool path allowed to call
`MemoryManager.set_mastery`/`update_mastery` -- ordinary tutoring/practice
(`tools.tutor_state_tools`) and mastery lookups (`tools.mastery_tools`) never
do (see their own module docstrings). It uses the same non-decreasing MVP
formula as `POST /assessment/submit` (`routers/api.py`):
`updated = max(previous, 0.7*previous + 0.3*assessment_score)` -- mastery
never decreases here either, so a wrong answer's delta is always `+0%`, not
negative, matching the existing MVP invariant (`non_decreasing_mvp`).
`routers/api.py`'s own Moodle-initial-mastery floor is intentionally omitted
here (that data is not available in an ordinary chat turn) -- a deliberate,
documented simplification of the same rule, not a different one.

After a completed check, the tutor state machine is reset (cleared) so the
next tutoring turn starts fresh at EXPLAIN (see services.tutor_state_machine).
"""

from __future__ import annotations

from typing import Any

from services.error_analyzer import judge_answer


MIN_CONCEPTS = 3
MAX_CONCEPTS = 5


def _select_assessment_concepts(context: dict[str, Any]) -> list[str]:
    """Concepts practiced this session, most-recent-first, deduped, capped at
    MAX_CONCEPTS -- padded from weak/available concepts only if fewer than
    MIN_CONCEPTS were actually practiced. Never invents a concept name."""
    seen: list[str] = []

    def add(concept: Any) -> None:
        if concept and concept not in seen:
            seen.append(concept)

    tutor_state = context.get("tutor_state") or {}
    add(tutor_state.get("concept"))
    add(context.get("current_concept"))
    for turn in reversed(context.get("recent_structured_turns") or []):
        for concept in (turn.get("resolved_entities") or {}).get("concepts") or []:
            add(concept)
        if len(seen) >= MAX_CONCEPTS:
            break

    if len(seen) < MIN_CONCEPTS:
        for concept in context.get("weak_concepts") or []:
            add(concept)
    if len(seen) < MIN_CONCEPTS:
        for concept in context.get("available_concepts") or []:
            add(concept)

    return seen[:MAX_CONCEPTS]


def _question_for_concept(concept: str, difficulty: str) -> str:
    from tools.tutor_state_tools import _question_bank_for_difficulty

    bank = _question_bank_for_difficulty(concept, difficulty)
    if not bank:
        return f"Explain {concept} in your own words, then give one short example of how it is used."
    entry = bank[0]
    if difficulty == "easy":
        question_text, opt_a, opt_b, opt_c, opt_d = entry
        return f"{question_text}\n{opt_a}\n{opt_b}\n{opt_c}\n{opt_d}"
    return entry


def run_quick_progress_check_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    memory = context.get("memory")
    student_id, course_id, session_id = context.get("student_id"), context.get("course_db_id"), context.get("session_id")
    if not memory or not student_id or not course_id or not session_id:
        return {"tool": "run_quick_progress_check", "success": False, "error": "missing_session_context"}

    state = memory.get_quick_progress_check(student_id, course_id)
    if state.get("session_id") != session_id or not state.get("concepts") or not state.get("current_question"):
        return _start(context, memory, student_id, course_id, session_id)
    return _continue(context, memory, student_id, course_id, session_id, state)


def _start(context, memory, student_id, course_id, session_id) -> dict[str, Any]:
    concepts = _select_assessment_concepts(context)
    if not concepts:
        return {
            "tool": "run_quick_progress_check", "success": False,
            "reply": "I don't have enough practiced concepts yet for a Quick Progress Check -- let's discuss a concept first.",
        }

    difficulty = context.get("difficulty") or "medium"
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    previous_mastery = {concept: memory.get_mastery(student_id, course_id, concept) for concept in concepts}
    question = _question_for_concept(concepts[0], difficulty)

    memory.set_quick_progress_check(student_id, course_id, {
        "session_id": session_id, "concepts": concepts, "index": 0, "difficulty": difficulty,
        "current_question": question, "results": [], "previous_mastery": previous_mastery,
    })
    total = len(concepts)
    reply = (
        f"Let's do a Quick Progress Check -- {total} question{'s' if total != 1 else ''}, one per concept.\n\n"
        f"Question 1 of {total} on {concepts[0]}:\n\n{question}"
    )
    return {"tool": "run_quick_progress_check", "success": True, "reply": reply, "phase": "started", "concepts": concepts}


_MCQ_CORRECT_LETTER = "B"  # the easy question bank's fixed convention -- see
# services.chat_orchestrator._evaluate_practice_answer's own identical
# "normalized[:1] == 'b'" rule for the same bank.


def _judge_answer(concept: str, question: str, answer: str, difficulty: str) -> tuple[bool, str | None, str]:
    """Judge one submitted answer. MCQ (easy) questions are judged by a plain
    deterministic letter comparison -- no LLM call -- since the easy bank's
    correct option is a fixed, known convention; open-ended (medium/hard)
    questions still need services.error_analyzer.judge_answer (there is no
    stored canonical answer to compare free text against)."""
    if difficulty == "easy":
        correct = answer[:1].strip().upper() == _MCQ_CORRECT_LETTER
        error_type = None if correct else "conceptual_misunderstanding"
        feedback_reason = "" if correct else f"The correct option was {_MCQ_CORRECT_LETTER}."
        return correct, error_type, feedback_reason
    judgment = judge_answer(concept, question, answer)
    return judgment.correct, judgment.error_type, judgment.feedback_reason


def _continue(context, memory, student_id, course_id, session_id, state) -> dict[str, Any]:
    concepts = list(state.get("concepts") or [])
    index = int(state.get("index") or 0)
    difficulty = state.get("difficulty") or "medium"
    question = state.get("current_question")
    if index >= len(concepts) or not question:
        return {"tool": "run_quick_progress_check", "success": False, "error": "corrupt_assessment_state"}

    concept = concepts[index]
    # The student's raw message this turn IS the assessment answer -- an
    # active Quick Progress Check always has exactly one pending question, so
    # there is no ambiguity to resolve here (see
    # agents.plan_compiler.compile_plan's session-state override, which
    # routes every turn to this tool while one is pending, regardless of
    # what goal the planner assigned this turn).
    answer = str(context.get("message") or "").strip()
    correct, _error_type, _feedback_reason = _judge_answer(concept, question, answer, difficulty)

    results = list(state.get("results") or [])
    results.append({"concept": concept, "correct": correct})
    next_index = index + 1

    if next_index < len(concepts):
        next_concept = concepts[next_index]
        next_question = _question_for_concept(next_concept, difficulty)
        memory.set_quick_progress_check(student_id, course_id, {
            **state, "index": next_index, "current_question": next_question, "results": results,
        })
        correct_so_far = sum(1 for r in results if r["correct"])
        reply = (
            f"{'Correct.' if correct else 'Not quite.'} ({correct_so_far}/{next_index} so far)\n\n"
            f"Question {next_index + 1} of {len(concepts)} on {next_concept}:\n\n{next_question}"
        )
        return {"tool": "run_quick_progress_check", "success": True, "reply": reply, "phase": "in_progress"}

    return _finish(context, memory, student_id, course_id, session_id, state, results)


def _score_tier_reply(correct: int, total: int, deltas: dict[str, float]) -> str:
    """Score-tiered summary text -- encouraging and actionable at every tier,
    never a bare "+0%" for a low/zero score."""
    percent = (correct / total * 100) if total else 0.0
    updated_concepts = {concept: delta for concept, delta in deltas.items() if delta > 0}
    concept_names = ", ".join(updated_concepts.keys()) or ", ".join(deltas.keys())

    if percent <= 0:
        return (
            f"You scored {correct}/{total}. Keep practicing — mastery updates when you answer "
            "correctly. Try reviewing the concept and practice more."
        )
    if percent < 50:
        return (
            f"You scored {correct}/{total}. Keep practicing — review {concept_names} and try "
            "again to build mastery."
        )
    if percent < 80:
        return f"Good effort! You scored {correct}/{total}. Mastery updated for {concept_names}. Keep going!"
    delta_text = ", ".join(f"{concept} +{delta:g}%" for concept, delta in updated_concepts.items())
    return f"Excellent! You scored {correct}/{total}. Mastery updated: {delta_text}. Great progress!"


def _finish(context, memory, student_id, course_id, session_id, state, results) -> dict[str, Any]:
    previous_mastery = state.get("previous_mastery") or {}
    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    deltas: dict[str, float] = {}
    for row in results:
        concept = row["concept"]
        previous_pct = round(float(previous_mastery.get(concept, 0.0)) * 100, 2)
        assessment_score_pct = 100.0 if row["correct"] else 0.0
        calculated_pct = round(0.7 * previous_pct + 0.3 * assessment_score_pct, 2)
        updated_pct = max(previous_pct, calculated_pct)
        memory.set_mastery(student_id, course_id, concept, updated_pct / 100)
        deltas[concept] = round(updated_pct - previous_pct, 2)

    memory.set_quick_progress_check(student_id, course_id, {})
    memory.set_tutor_state(student_id, course_id, {})  # reset to fresh EXPLAIN next tutoring turn

    reply = f"Quick Progress Check complete! {_score_tier_reply(correct, total, deltas)}"
    return {
        "tool": "run_quick_progress_check", "success": True, "reply": reply, "phase": "complete",
        "score": f"{correct}/{total}", "mastery_updates": deltas,
    }
