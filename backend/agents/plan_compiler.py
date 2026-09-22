"""Deterministic plan compiler: turns a `SemanticPlan` (goal + resolved
entities + analytics dimensions -- the only things that genuinely require
language understanding, see `agents.simple_planner`) into a `SimplePlan`
(concrete tool calls, dependency order, answer basis).

This is pure code, keyed only on the fixed `SemanticGoal` enum and the
entities/references the semantic planner already resolved -- never on
message wording. The same goal (with the same resolved entities/context)
always compiles to the same tool shape, so there is no LLM decision left
here to get wrong: `agents.simple_agent` no longer needs to validate that an
LLM-chosen tool set actually matches the goal (see
`agents.entity_validator.EVIDENCE_REQUIRING_GOALS`, which this compiler
satisfies by construction for every goal it lists) -- it only still grounds
the *entities* (real vocabulary, real tool names) the same way it always did.

Argument resolution for each compiled tool call is deliberately shallow: most
tools already resolve their own missing arguments from `context` (see
`tools.course_tools._concepts_from_arguments`, `tools.rag_tools`'s
is_followup/agent_selected_concept fallback chain) -- this module supplies
only the entities the semantic plan actually resolved and leaves everything
else (course/session scope, dependency side-effects) to those existing,
already-tested fallbacks and to `agents.simple_agent`'s topological execution
order (unchanged -- see `_topological_order`/`_execute_tools`).
"""

from __future__ import annotations

from typing import Any

from agents.agent_models import AnswerBasis, PlannerToolCall, SemanticPlan, SimplePlan


def compile_plan(semantic: SemanticPlan, context: dict[str, Any]) -> SimplePlan:
    """Map one `SemanticPlan` to a concrete `SimplePlan`, by goal, deterministically."""
    quick_progress_check = context.get("quick_progress_check") or {}
    if quick_progress_check.get("concepts") and quick_progress_check.get("current_question"):
        # A Quick Progress Check question is actively pending -- ANY message
        # this turn is this turn's answer attempt, regardless of what goal
        # the planner assigned (session state overrides classification here,
        # the same principle agents.simple_agent._pending_context_issue
        # already applies using prior-turn state). Once started, a Quick
        # Progress Check always has exactly one pending question until it
        # finishes, so there is no ambiguity to resolve the way
        # tutor_signal disambiguates an idle GUIDED_PRACTICE state.
        return SimplePlan(
            goal="assessment",
            resolved_entities=semantic.resolved_entities,
            tools=[_tool("run_quick_progress_check")],
            answer_basis="conversation",
            needs_clarification=False,
            clarification_question=None,
            confidence=semantic.confidence,
            analytics_request=semantic.analytics_request,
        )

    if semantic.needs_clarification:
        return SimplePlan(
            goal=semantic.goal,
            resolved_entities=semantic.resolved_entities,
            tools=[],
            answer_basis="conversation",
            needs_clarification=True,
            clarification_question=semantic.clarification_question,
            confidence=semantic.confidence,
            analytics_request=semantic.analytics_request,
        )

    concepts = list(semantic.resolved_entities.concepts)
    references = list(semantic.resolved_entities.references)
    tools, answer_basis = _compile_tools(semantic, context, concepts, references)

    return SimplePlan(
        goal=semantic.goal,
        resolved_entities=semantic.resolved_entities,
        tools=tools,
        answer_basis=answer_basis,
        needs_clarification=False,
        clarification_question=None,
        confidence=semantic.confidence,
        analytics_request=semantic.analytics_request,
    )


def _tool(name: str, arguments: dict[str, Any] | None = None, depends_on: str | None = None) -> PlannerToolCall:
    return PlannerToolCall(name=name, arguments=arguments or {}, depends_on=depends_on)


# Goals with a single, unconditional tool mapping -- the entire reason this
# table can be a flat dict is that none of these need to look at resolved
# entities/references to decide WHICH tool, only whether to call it at all
# (already decided by needs_clarification, handled in compile_plan above).
_FIXED_GOAL_TOOLS: dict[str, tuple[str, AnswerBasis]] = {
    "analytics_query": ("run_analytics_query", "analytics"),
    "personal_profile_query": ("get_student_learning_profile", "profile"),
    "study_recommendation": ("run_study_recommendation", "profile"),
    "source_provenance": ("get_source_provenance", "conversation"),
    "mastery_policy": ("get_mastery_policy", "conversation"),
    # Deterministic safety-guard backstop: chat_orchestrator's own regex-based
    # pre-agent gate already intercepts almost every real instance of a
    # mastery-modification request before the planner ever runs (see
    # services.chat_orchestrator._is_mastery_modification_request); this
    # covers whatever slips through classified purely by meaning.
    "mastery_modification_request": ("get_mastery_guard_response", "conversation"),
    "external_knowledge": ("answer_with_external_knowledge", "external"),
    # Launches (or continues -- see compile_plan's session-state override
    # above) the Quick Progress Check flow. Distinct from analytics_query:
    # analytics reports a stored mastery value, assessment launches a new
    # scored check that can change one (see tools.assessment_tools).
    "assessment": ("run_quick_progress_check", "conversation"),
}

# Goals that never need a tool -- answered (or clarified) directly by
# agents.response_generator / the clarification path.
_NO_TOOL_GOALS = {"casual_conversation", "clarification", "unclear"}


def _compile_tools(
    semantic: SemanticPlan, context: dict[str, Any], concepts: list[str], references: list[str],
) -> tuple[list[PlannerToolCall], AnswerBasis]:
    goal = semantic.goal

    fixed = _FIXED_GOAL_TOOLS.get(goal)
    if fixed:
        tool_name, answer_basis = fixed
        return [_tool(tool_name)], answer_basis

    if goal in _NO_TOOL_GOALS:
        return [], "conversation"

    if goal in ("concept_explanation", "personalized_tutoring"):
        tutor_result = _compile_tutor_state(semantic, context, concepts)
        if tutor_result is not None:
            return tutor_result
        # No active/startable tutor state (e.g. tutor_signal="new_topic", or
        # nothing to explain yet) -- fall through to the existing,
        # unmodified compilation below.

    if goal in ("concept_explanation", "concept_comparison"):
        # search_course_material resolves missing/invalid concepts itself
        # (query text, is_followup+current_concept, agent_selected_concept --
        # see tools.rag_tools) -- concepts=[] here is not an error state, it
        # is the same "let the tool auto-resolve" path the old prompt
        # documented for dependency-ordered calls.
        return [_tool("search_course_material", {"concepts": concepts})], "rag"

    if goal == "personalized_tutoring":
        return _compile_personalized_tutoring(concepts, references)

    if goal == "reference_followup":
        return _compile_reference_followup(semantic, references, concepts)

    if goal == "tutoring_methodology":
        # "How does tutoring/practice work" is this goal's own name and the
        # common case; scoring-formula and policy-band questions have their
        # own dedicated goals (tutoring_methodology's sibling
        # mastery_policy, and get_mastery_scoring_methodology reachable via
        # the self-evident tool list in the iterative agent) so this does
        # not need a message-wording-derived sub-topic classification.
        return [_tool("explain_methodology", {"topic": "tutoring"})], "conversation"

    if goal == "navigation":
        if concepts:
            return [_tool("switch_focus", {"concept": concepts[0]})], "conversation"
        return [_tool("get_course_structure")], "conversation"

    # Any goal not explicitly mapped above (should not happen -- SemanticGoal
    # is a closed enum and every value is covered) answers with no tool
    # rather than guessing one.
    return [], "conversation"


def _compile_tutor_state(
    semantic: SemanticPlan, context: dict[str, Any], concepts: list[str],
) -> tuple[list[PlannerToolCall], AnswerBasis] | None:
    """Deterministic tutor-state-machine branch: maps (current tutor_state,
    tutor_signal) to concrete tools. Returns None to fall through to the
    ordinary concept_explanation/personalized_tutoring compilation when
    there is nothing active to continue, or when the semantic planner
    classified this message as unrelated to the tutoring flow
    (tutor_signal == "new_topic") -- the stored tutor_state is left
    untouched either way, so it can be resumed later.

    `signal` must be exactly "continue" to advance a resting state --
    `None`/`needs_support` never do, so an ambiguous or clearly-unready
    message can never be silently treated the same as a real acknowledgement
    (the historical bug this guards against: "None" used to be treated as
    "continue", which is exactly backwards -- an uncertain signal should
    never be assumed to mean "the student is ready to move on").

    See services.tutor_state_machine for TutorState/AdaptivePolicy and
    tools.tutor_state_tools for what each compiled tool actually does.
    """
    tutor_state = context.get("tutor_state") or {}
    state = tutor_state.get("state")
    active_concept = tutor_state.get("concept")
    requested_concept = concepts[0] if concepts else None
    signal = semantic.tutor_signal

    if not state or (requested_concept and requested_concept != active_concept):
        # An explicit new/different concept always wins over stale tutor
        # context, regardless of signal -- see plan_compiler tests covering
        # "explain recursion"/"switch to graphs" during active tutoring.
        target = requested_concept or active_concept
        if not target:
            return None
        return [
            _tool("search_course_material", {"concepts": [target]}),
            _tool("advance_tutor_state", {"to": "EXPLAIN", "concept": target}),
        ], "rag"

    if signal == "new_topic":
        return None

    if state == "GUIDED_PRACTICE" and tutor_state.get("current_question") and signal == "needs_support":
        # A question is pending AND the planner classified this reply as
        # needs_support -- by MEANING ("I don't know how to answer", "can
        # you give me a hint?", "I'm stuck", ...; classified by the same
        # single planner call, never keyword-matched here) the student is
        # asking for help, not attempting the question. This must never be
        # graded: no evaluate_practice_answer/judge_answer call, so
        # rounds_completed (the practice denominator) is never touched.
        context["tutor_needs_support"] = True
        consecutive_confusion = tutor_state.get("consecutive_wrong", 0) + 1
        if consecutive_confusion >= 2:
            # Asked for help on THIS SAME question twice in a row -- step
            # back to EXAMPLE, the identical threshold/destination
            # AdaptivePolicy itself already uses for a second consecutive
            # WRONG answer, rather than repeating the same hint forever.
            # AdaptivePolicy.decide itself is not called (this was never a
            # graded attempt); this only reuses its resting-state/threshold
            # shape for a confusion streak instead of a wrong-answer streak.
            # RQ1 fix: explicitly reset the streak on this genuine
            # remediation-reset boundary (the same treatment AdaptivePolicy's
            # own repeated-wrong step-back now gives consecutive_wrong) --
            # otherwise the carried-forward elevated count would make the
            # very next confusion signal on the new question immediately
            # look like a second consecutive one.
            return [
                _tool("search_course_material", {"concepts": [active_concept]}),
                _tool("advance_tutor_state", {"to": "EXAMPLE", "concept": active_concept, "consecutive_wrong": 0}),
            ], "rag"
        # First request for help on this question -- record the streak
        # (so a SECOND one escalates) but keep the SAME question pending;
        # no tool result feeds response_generator's own text, so this is
        # still exactly one response call, not a new one.
        return [
            _tool("advance_tutor_state", {
                "to": "GUIDED_PRACTICE", "concept": active_concept,
                "consecutive_wrong": consecutive_confusion, "keep_question": True,
            }),
        ], "conversation"

    if state == "GUIDED_PRACTICE" and (signal == "practice_answer" or tutor_state.get("current_question")):
        # A question is actively pending and this reply is NOT needs_support
        # (excluded just above) -- treat it as this turn's answer attempt:
        # the pending question already fixes what this turn means, so there
        # is no ambiguity left for tutor_signal to resolve (the same
        # session-state-overrides-classification principle compile_plan's
        # own Quick Progress Check branch already uses). A non-answer that
        # wasn't classified needs_support is still judged like any other
        # attempt by services.error_analyzer/tools.tutor_state_tools -- it
        # comes back wrong/unsure, which already is the existing hint/retry
        # policy for that case, never a silently dropped turn.
        # `signal == "practice_answer"` is kept as its own alternative (not
        # just `current_question` alone) so a bounded replan on this same
        # turn -- which runs AFTER the tool already cleared
        # current_question -- still recompiles this same tool instead of
        # falling through to an unrelated goal-default compilation;
        # agents.simple_agent's own already-executed-this-turn dedupe is
        # what then safely skips the redundant second run.
        return [_tool("evaluate_practice_answer", {})], "conversation"

    if state == "GUIDED_PRACTICE" and signal == "needs_support" and not tutor_state.get("current_question"):
        # No question is actively pending (a round just finished --
        # evaluate_practice_answer_tool already cleared it), and the
        # student explicitly signaled they need help before continuing.
        # Step back to EXAMPLE -- the same place AdaptivePolicy itself
        # already steps back to on repeated wrong answers -- instead of
        # falling through to an unrelated goal-default (weakest-concept)
        # compilation, which is a generic-reply regression of exactly the
        # kind this fix exists to remove.
        context["tutor_needs_support"] = True
        # RQ1 fix: also a step-back to EXAMPLE (a genuine remediation reset
        # boundary) -- reset the streak here too, so it doesn't carry an
        # elevated count from an earlier wrong-answer run into the fresh
        # EXAMPLE/GUIDED_PRACTICE cycle that follows.
        return [
            _tool("search_course_material", {"concepts": [active_concept]}),
            _tool("advance_tutor_state", {"to": "EXAMPLE", "concept": active_concept, "consecutive_wrong": 0}),
        ], "rag"

    if state == "EXPLAIN":
        if signal == "continue":
            return [
                _tool("search_course_material", {"concepts": [active_concept]}),
                _tool("advance_tutor_state", {"to": "EXAMPLE", "concept": active_concept}),
            ], "rag"
        # needs_support, or no clear readiness signal yet -- stay on
        # EXPLAIN and re-ground; response_generator is told (via
        # tutor_needs_support) to explain differently/simpler rather than
        # repeat the same wording, and never to advance as if understanding
        # were confirmed.
        context["tutor_needs_support"] = True
        return [_tool("search_course_material", {"concepts": [active_concept]})], "rag"

    if state == "EXAMPLE":
        if signal == "continue":
            return [_tool("generate_practice_question", {"concept": active_concept})], "conversation"
        context["tutor_needs_support"] = True
        return [_tool("search_course_material", {"concepts": [active_concept]})], "rag"

    if state == "GUIDED_PRACTICE" and signal in (None, "continue"):
        # No question pending (a round just finished -- evaluate_practice_answer_tool
        # clears current_question and may keep the resting state at
        # GUIDED_PRACTICE, e.g. "wrong_hint_retry"/"correct_weak_explanation_same_level")
        # -- move forward with the next question.
        return [_tool("generate_practice_question", {"concept": active_concept})], "conversation"

    if state == "PROGRESS_CHECK_READY" and signal in (None, "continue"):
        return [_tool("generate_practice_question", {"concept": active_concept})], "conversation"

    return None


def _compile_personalized_tutoring(concepts: list[str], references: list[str]) -> tuple[list[PlannerToolCall], AnswerBasis]:
    if concepts:
        return [_tool("search_course_material", {"concepts": concepts})], "rag"
    if references:
        # "explain the weaker one" / "let's work on that again" -- continues
        # a previously discussed set rather than the whole course.
        return [
            _tool("select_lowest_mastery_among_previous_turn"),
            _tool("search_course_material", {"concepts": []}, depends_on="select_lowest_mastery_among_previous_turn"),
        ], "rag"
    # Nothing named at all ("help me study", "what should I work on") --
    # pick the weakest concept across the active scope, then explain it.
    # select_lowest_mastery_concept already reads mastery for every candidate
    # concept internally (tools.mastery_tools), so this is the same
    # "get mastery -> select weakest -> explain it" chain in one fewer call.
    return [
        _tool("select_lowest_mastery_concept"),
        _tool("search_course_material", {"concepts": []}, depends_on="select_lowest_mastery_concept"),
    ], "rag"


def _compile_reference_followup(
    semantic: SemanticPlan, references: list[str], concepts: list[str],
) -> tuple[list[PlannerToolCall], AnswerBasis]:
    if semantic.analytics_request is not None:
        return [_tool("run_analytics_query")], "analytics"
    if "previous_recommendation" in references:
        return [_tool("explain_recommendation")], "conversation"
    # Otherwise a continuing content question (explanation/comparison) --
    # search_course_material's own is_followup+current_concept/
    # agent_selected_concept fallback resolves the topic even with concepts=[].
    return [_tool("search_course_material", {"concepts": concepts})], "rag"
