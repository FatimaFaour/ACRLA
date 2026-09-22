"""Agent-callable tools for the adaptive tutor state machine.

Three tools, registered into `agents.agent_tools.TOOL_REGISTRY`:
- `advance_tutor_state_tool` -- writes a new resting state (real side
  effect, same precedent as `tools.dialogue_tools.switch_focus_tool`).
- `generate_practice_question_tool` -- picks the next unused question
  variant for the active concept/difficulty; reuses the legacy question-bank
  CONTENT functions from `services.chat_orchestrator` (pure topic-string-in,
  text-out lookups) but NOT their volatile, process-local bookkeeping --
  variant rotation here is tracked durably in `tutor_state["asked_variants"]`.
- `evaluate_practice_answer_tool` -- judges the student's answer
  (`services.error_analyzer`), applies the adaptive policy
  (`services.tutor_state_machine.AdaptivePolicy`), records an error pattern
  when wrong, and phrases feedback.

None of these ever call `MemoryManager.set_mastery`/`update_mastery` -- chat
practice never updates mastery (see the module docstrings of
`services.tutor_state_machine`/`services.memory_manager`); only
`POST /assessment/submit` (Quick Progress Check) does.
"""

from __future__ import annotations

from typing import Any

from services.course_concepts import display_concept_name
from services.error_analyzer import judge_answer
from services.tutor_state_machine import AdaptivePolicy, TutorState, record_error_pattern, save_tutor_state, top_error_pattern


_ERROR_TYPE_LABELS = {
    "conceptual_misunderstanding": "a conceptual misunderstanding",
    "logic_error": "a logic error",
    "missing_base_case": "a missing base case",
    "algorithm_misuse": "a misuse of the algorithm/structure",
}

_PROGRESS_NOTE = "Mastery will update after you complete a Quick Progress Check."


def advance_tutor_state_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Write a new resting TutorState for the active session/concept.

    Two optional arguments (both unused by every pre-existing caller, so
    their absence keeps this tool's old behavior byte-for-byte):
    - `consecutive_wrong`: override the stored count instead of carrying
      the prior value forward -- lets plan_compiler track a "the student
      keeps needing help with THIS question" streak the same way
      AdaptivePolicy already tracks a wrong-answer streak, without this
      tool needing to know why the count changed.
    - `keep_question` (bool): when true, current_question is carried over
      from the prior state instead of being cleared -- used when "advancing"
      really just means "record a hint was given, the SAME question is
      still pending," not a real state transition.
    """
    memory = context.get("memory")
    student_id, course_id, session_id = context.get("student_id"), context.get("course_db_id"), context.get("session_id")
    if not memory or not student_id or not course_id or not session_id:
        return {"tool": "advance_tutor_state", "success": False, "error": "missing_session_context"}

    to_state = str(arguments.get("to") or "").strip()
    prior = context.get("tutor_state") or {}
    concept = arguments.get("concept") or prior.get("concept")
    if not concept:
        return {"tool": "advance_tutor_state", "success": False, "error": "missing_concept"}

    consecutive_wrong = arguments.get("consecutive_wrong")
    if consecutive_wrong is None:
        consecutive_wrong = prior.get("consecutive_wrong", 0)
    current_question = prior.get("current_question") if arguments.get("keep_question") else None

    save_tutor_state(
        memory, student_id, course_id, session_id,
        state=to_state, concept=concept,
        difficulty=arguments.get("difficulty") or prior.get("difficulty") or context.get("difficulty") or "medium",
        rounds_completed=prior.get("rounds_completed", 0),
        consecutive_wrong=consecutive_wrong,
        current_question=current_question,
        asked_variants=prior.get("asked_variants", {}),
    )
    context["tutor_state"] = {**prior, "state": to_state, "concept": concept, "consecutive_wrong": consecutive_wrong, "current_question": current_question}
    return {"tool": "advance_tutor_state", "success": True, "state": to_state, "concept": concept}


def _difficulty_label(difficulty: str) -> str:
    if difficulty == "medium":
        return "Moderate"
    return str(difficulty or "medium").capitalize()


def _question_bank_for_difficulty(concept: str, difficulty: str):
    from services.chat_orchestrator import _easy_questions_for_topic, _hard_questions_for_topic, _moderate_questions_for_topic

    if difficulty == "easy":
        return _easy_questions_for_topic(concept)
    if difficulty == "hard":
        return _hard_questions_for_topic(concept)
    return _moderate_questions_for_topic(concept)


def _pick_variant_index(asked: list[int], pool_size: int) -> int:
    if pool_size <= 0:
        return 0
    for index in range(pool_size):
        if index not in asked:
            return index
    return 0


def generate_practice_question_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Return the next guided-practice question for the active concept, at
    the tutor state's current difficulty, and persist it as the resting
    GUIDED_PRACTICE state."""
    from services.chat_orchestrator import _apply_adaptive_question_support

    memory = context.get("memory")
    student_id, course_id, session_id = context.get("student_id"), context.get("course_db_id"), context.get("session_id")
    tutor_state = context.get("tutor_state") or {}
    concept = arguments.get("concept") or tutor_state.get("concept")
    if not memory or not student_id or not course_id or not session_id or not concept:
        return {"tool": "generate_practice_question", "success": False, "error": "missing_session_context"}

    difficulty = tutor_state.get("difficulty") or context.get("difficulty") or "medium"
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"

    # `concept` is the canonical/backend identifier (a course's own raw
    # source-style label, e.g. "Chap: Linear Regression", is exactly what
    # mastery/RAG lookups are keyed by end to end -- never altered here).
    # `display_name` is ONLY for text the student actually reads: the
    # question bank lookup, the question's own wording, and the adaptive
    # wrapper text below all use it, so a practice question never repeats
    # a source-style label back at the student.
    display_name = display_concept_name(concept)
    bank = _question_bank_for_difficulty(display_name, difficulty)
    asked_variants = dict(tutor_state.get("asked_variants") or {})
    asked_for_difficulty = list(asked_variants.get(difficulty) or [])
    if len(asked_for_difficulty) >= len(bank):
        asked_for_difficulty = []
    variant_index = _pick_variant_index(asked_for_difficulty, len(bank))

    expected_answer_hint = None
    if difficulty == "easy" and bank:
        question_text, opt_a, opt_b, opt_c, opt_d = bank[variant_index]
        body = f"{question_text}\n{opt_a}\n{opt_b}\n{opt_c}\n{opt_d}"
        expected_answer_hint = opt_b
    elif bank:
        body = bank[variant_index]
    else:
        body = f"Explain {display_name} in your own words, then give one short example of how it is used."

    strategy = context.get("tutoring_strategy") or {}
    raw_question = f"{_difficulty_label(difficulty)} question on {display_name}:\n\n{body}"
    reply = _apply_adaptive_question_support(raw_question, display_name, difficulty, strategy.get("name") or "", strategy.get("reason") or "")

    asked_for_difficulty.append(variant_index)
    asked_variants[difficulty] = asked_for_difficulty
    save_tutor_state(
        memory, student_id, course_id, session_id,
        state=TutorState.GUIDED_PRACTICE.value, concept=concept, difficulty=difficulty,
        rounds_completed=tutor_state.get("rounds_completed", 0),
        consecutive_wrong=tutor_state.get("consecutive_wrong", 0),
        current_question=raw_question, asked_variants=asked_variants,
    )
    if expected_answer_hint:
        # Kept only for the immediately following evaluation turn (not part
        # of the documented tutor_state shape) -- MemoryManager.update_course_memory
        # merges arbitrary keys, so this rides along harmlessly.
        memory.update_course_memory(student_id, course_id, {"tutor_state": {
            **memory.get_tutor_state(student_id, course_id),
            "expected_answer_hint": expected_answer_hint,
        }})
    context["tutor_state"] = {**tutor_state, "state": TutorState.GUIDED_PRACTICE.value, "concept": concept, "difficulty": difficulty, "current_question": raw_question}
    return {"tool": "generate_practice_question", "success": True, "reply": reply, "concept": concept, "difficulty": difficulty}


def evaluate_practice_answer_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """Judge the student's answer, apply the adaptive policy, record an
    error pattern if wrong, persist the next resting state, and phrase
    feedback. Never calls MemoryManager.set_mastery/update_mastery."""
    memory = context.get("memory")
    student_id, course_id, session_id = context.get("student_id"), context.get("course_db_id"), context.get("session_id")
    tutor_state = context.get("tutor_state") or {}
    concept, question = tutor_state.get("concept"), tutor_state.get("current_question")
    answer = str(arguments.get("answer") or context.get("message") or "").strip()
    if not memory or not student_id or not course_id or not session_id or not concept or not question:
        return {"tool": "evaluate_practice_answer", "success": False, "error": "missing_session_context"}

    stored_state = memory.get_tutor_state(student_id, course_id)
    expected_hint = stored_state.get("expected_answer_hint") if stored_state.get("session_id") == session_id else None
    question_for_judge = question if not expected_hint else f"{question}\n\n[For grading only, not shown to the student: the correct option is {expected_hint}]"

    # RQ3 guided-practice grounding (evidence hierarchy): a deterministic
    # expected answer (EASY, `expected_hint` above) is the strongest
    # evidence and is used as-is, unchanged -- no retrieval needed. For an
    # open-ended (MODERATE/HARD) answer with no stored expected answer,
    # ground the SAME judge_answer call in real retrieved course material
    # instead of leaving it to judge from pretrained knowledge alone.
    # Reuses the existing RAG retrieval function (pipelines.rag_pipeline.
    # retrieve_context_for_scope) -- not a second RAG architecture -- and
    # the SAME institutional-privacy gateway (`for_external=True`)
    # tools.rag_tools.search_course_material_tool already applies, since
    # this call is just as external-LLM-bound as that one.
    course_evidence = None
    if not expected_hint:
        course_ids = context.get("retrieval_course_ids") or []
        if course_ids:
            from pipelines.rag_pipeline import retrieve_context_for_scope
            evidence_audit: dict[str, Any] = {}
            course_evidence, _sources = retrieve_context_for_scope(
                course_ids, question, selected_concept=concept,
                scope=context.get("remediation_level") or "chapter",
                canonical_courses=context.get("canonical_courses") or [],
                for_external=True, audit=evidence_audit,
            )
            if evidence_audit:
                from services.privacy_context import log_external_content_decision
                log_external_content_decision(course_id=course_id, audit=evidence_audit)

    judgment = judge_answer(concept, question_for_judge, answer, course_evidence=course_evidence)

    if not judgment.correct and judgment.error_type:
        record_error_pattern(memory, student_id, course_id, concept, judgment.error_type)

    difficulty = tutor_state.get("difficulty") or "medium"
    decision = AdaptivePolicy.decide(
        correct=judgment.correct, confident=judgment.confident, difficulty=difficulty,
        consecutive_wrong=tutor_state.get("consecutive_wrong", 0),
        rounds_completed=tutor_state.get("rounds_completed", 0),
    )
    rounds_completed = 0 if decision.reset_rounds else tutor_state.get("rounds_completed", 0) + 1
    # RQ1 fix: `decision.reset_rounds` is true exactly at a genuine
    # remediation reset boundary (AdaptivePolicy's repeated-wrong step-back
    # to EXAMPLE) -- reset the streak there too, the same signal
    # rounds_completed already uses, so a student who has just been
    # stepped back and re-taught starts their next attempt with a fresh
    # 2-strike allowance instead of one already-elevated wrong answer away
    # from an immediate second step-back. Normal first-wrong behavior
    # (reset_rounds=False) is unaffected -- still increments as before.
    consecutive_wrong = 0 if (judgment.correct or decision.reset_rounds) else tutor_state.get("consecutive_wrong", 0) + 1
    save_tutor_state(
        memory, student_id, course_id, session_id,
        state=decision.next_state.value, concept=concept, difficulty=decision.difficulty,
        rounds_completed=rounds_completed, consecutive_wrong=consecutive_wrong,
        current_question=None, asked_variants=tutor_state.get("asked_variants") or {},
    )
    context["tutor_state"] = {
        **tutor_state, "state": decision.next_state.value, "difficulty": decision.difficulty,
        "rounds_completed": rounds_completed, "consecutive_wrong": consecutive_wrong, "current_question": None,
    }

    reply = _phrase_feedback(judgment, decision, concept, difficulty)
    return {
        "tool": "evaluate_practice_answer", "success": True, "reply": reply,
        "correct": judgment.correct, "error_type": judgment.error_type,
        "next_state": decision.next_state.value,
    }


def _phrase_feedback(judgment, decision, concept: str, difficulty: str) -> str:
    from services.chat_orchestrator import _support_hint_for_topic

    if judgment.correct:
        base = f"Correct! {judgment.feedback_reason}".strip() if judgment.feedback_reason else "Correct!"
        if decision.next_state == TutorState.PROGRESS_CHECK_READY:
            return f"{base}\n\nYou've completed a few rounds on {concept} -- want to try a Quick Progress Check?\n\n{_PROGRESS_NOTE}"
        if decision.reason == "correct_confident_escalate":
            return f"{base} Let's raise the difficulty a bit.\n\n{_PROGRESS_NOTE}"
        return f"{base} Let's do one more at the same level to make sure it's solid.\n\n{_PROGRESS_NOTE}"

    reason_text = f" {judgment.feedback_reason}" if judgment.feedback_reason else ""
    if not judgment.confident:
        # RQ3: the judge itself flagged low confidence in this wrong
        # verdict (typically: no course evidence was available to verify
        # against, or the call was genuinely ambiguous) -- never state a
        # confidently incorrect pedagogical claim (a specific error-type
        # label) when the evaluator itself was not sure. This only changes
        # WORDING; the state transition (`decision`, computed above from
        # `correct`/consecutive_wrong only) is unaffected.
        opening = f"I'm not fully certain about this one.{reason_text}"
    else:
        label = _ERROR_TYPE_LABELS.get(judgment.error_type, "a gap in understanding")
        opening = f"Not quite -- this looks like {label}.{reason_text}"
    if decision.next_state == TutorState.EXAMPLE:
        return f"{opening}\n\nLet's go back to an example of {concept} and break it down again."
    hint = _support_hint_for_topic(concept, difficulty)
    return f"{opening}\n\nHint: {hint}\n\n{_PROGRESS_NOTE}"
