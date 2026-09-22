"""Conversation agent: a true iterative tool-using tutoring agent.

    User message
    -> Load structured conversation state
    -> Agent Brain decides next action           [agents.agent_brain]
    -> Execute one allowed tool                  [agents.executor]
    -> Add structured observation
    -> Agent Brain reviews all observations
    -> Decide whether more evidence is needed
    -> Repeat until evidence is sufficient
    -> Generate one grounded final answer         [agents.response_generator]
    -> Save structured turn                       [chat_orchestrator, caller]

    MAX_AGENT_STEPS = 5
    for step in range(MAX_AGENT_STEPS):
        decision = agent_brain.decide(context)
        if decision.type == "tool":
            observations.append(executor.execute(decision.tool, decision.arguments, context))
            continue
        if decision.type == "clarification": return clarification_response
        if decision.type == "answer": return response_generator.generate(...)
        if decision.type == "fallback": return safe_fallback

`agents.agent_brain` combines goal understanding, reference resolution, tool
selection, and evidence-sufficiency judgment into one LLM call per step --
"do I have enough evidence" and "what should I do next" are the same
judgment, so they are not split across two calls the way an earlier version
of this module did (a separate planner + a separate reasoning-over-evidence
stage). Because the brain is re-consulted every step against the growing
`context["turn_observations"]` list, "internal_rag vs external_fallback" is
never a decision made up front: it falls out of which tool actually got
called this turn and whether `agents.evidence_validator` judged its result
reliable. The brain's own `answer_basis`/`evidence_status` are synthesis
guidance for `response_generator` -- they never override the deterministic
RAG evidence-reliability gate below, which is what actually controls whether
PDF sources may be shown.

If the step limit is reached without a terminal decision, this module does
not hallucinate: it answers only if the last evidence_status was sufficient,
otherwise asks one concise clarification question, otherwise falls back.

Logging: every step logs agent_step/agent_goal/decision_type/decision_reason/
tool_requested/tool_arguments/tool_result_summary/observation_count, and the
terminal outcome logs evidence_sufficient/evidence_reliable/evidence_coverage/
final_selected_pipeline/agent_success/legacy_fallback_reason. Only structured
summaries are logged -- never raw prompt text, and tool_result_summary
carries result *keys*, not full retrieved content (e.g. RAG chunk text).
"""

from __future__ import annotations

from typing import Any

from agents import agent_brain, decision_acceptance, entity_validator, evidence_validator, executor, response_generator
from agents.agent_json import model_dump, model_validate
from agents.debug_log import vprint
from agents.agent_models import AgentBrainOutput, AgentResult, AnalyticsRequest, ToolObservation


_SELECTION_TOOLS = {"select_lowest_mastery_concept", "select_lowest_mastery_among_previous_turn", "run_study_recommendation"}
_DETERMINISTIC_DATA_TOOLS = {
    "run_analytics_query", "get_mastery_for_concepts", "get_all_mastery", "select_lowest_mastery_concept",
    "select_lowest_mastery_among_previous_turn", "get_mastery_policy", "get_mastery_scoring_methodology",
    "get_student_learning_profile", "run_study_recommendation",
}
_CONTEXT_TOOLS = {
    "get_last_reference", "get_recent_conversation", "get_recent_structured_turns",
    "get_last_recommendation", "get_last_response_metadata",
}
MAX_AGENT_STEPS = 5


def run_conversation_agent(context: dict[str, Any]) -> AgentResult:
    """Run the agent-brain decide -> tool -> observe -> repeat loop for one
    turn, then fill in the per-turn LLM call/token metrics
    (agents.simple_agent's counterpart populates the same AgentResult fields)
    from the step trace -- see `_with_iterative_metrics`.

    Returns `agent_used=False` whenever the brain is low confidence, its
    output is malformed, or the turn cannot be safely answered -- the caller
    then falls back to the legacy orchestrator.
    """
    return _with_iterative_metrics(_run_conversation_agent_loop(context))


def _with_iterative_metrics(result: AgentResult) -> AgentResult:
    """Derive planner_call_count/response_call_count/token counts from the
    step trace already recorded above, without threading a new parameter
    through every one of this module's many AgentResult return points.
    `planner_call_count` is every step phase that represents one
    `agent_brain.decide()` call (successful or not); `response_call_count`/
    token counts come from the `response_generation_metrics` step appended at
    each of the two `response_generator.generate_final_answer` call sites,
    which is only actually `llm_called=True` when the deterministic
    agent_tools short-circuit did NOT fire -- so this never overcounts an
    LLM call that never happened."""
    planner_call_count = sum(1 for step in result.steps if step.get("phase") in ("decide", "decide_parse_error"))
    response_metrics_steps = [step for step in result.steps if step.get("phase") == "response_generation_metrics"]
    response_call_count = sum(1 for step in response_metrics_steps if step.get("llm_called"))
    response_prompt_tokens = sum(step.get("prompt_tokens", 0) for step in response_metrics_steps)
    response_completion_tokens = sum(step.get("completion_tokens", 0) for step in response_metrics_steps)
    total_tokens = response_prompt_tokens + response_completion_tokens
    data = model_dump(result)
    data.update({
        "agent_mode": "iterative",
        "planner_call_count": planner_call_count,
        "response_call_count": response_call_count,
        "total_llm_call_count": planner_call_count + response_call_count,
        "response_prompt_tokens": response_prompt_tokens,
        "response_completion_tokens": response_completion_tokens,
        "total_tokens": total_tokens,
    })
    return model_validate(AgentResult, data)


def _run_conversation_agent_loop(context: dict[str, Any]) -> AgentResult:
    steps: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    context["turn_observations"] = observations
    executed_all: list[str] = []
    rejected_all: list[str] = []
    requested_all: list[str] = []
    brain_output: AgentBrainOutput | None = None
    any_successful_decide = False

    for step_index in range(1, MAX_AGENT_STEPS + 1):
        brain_output, raw_output, parse_error, json_mode_fallback_used = agent_brain.decide(context)
        if parse_error:
            # agent_json.parse_json_object already made one safe repair
            # attempt. If the decision is still malformed, do not spend the
            # remaining agent steps blindly re-asking the same prompt against
            # the same observations.
            raw_output_len = len(raw_output or "")
            steps.append({
                "step": step_index, "phase": "decide_parse_error",
                "parse_error": str(parse_error)[:300],
                "raw_output_len": raw_output_len,
            })
            print(
                "[ACRLA] conversation_agent_decide_parse_error "
                f"agent_iteration={step_index} "
                f"parse_error={str(parse_error)[:300]!r} "
                f"raw_output_len={raw_output_len}"
            )
            reason = "decision_provider_error" if str(parse_error).startswith("provider_error:") else "decision_json_or_schema_invalid"
            _log_terminal(
                agent_step=step_index, agent_goal="unclear", agent_success=False,
                final_selected_pipeline=None, legacy_fallback_reason=reason,
            )
            return AgentResult(steps=steps, fallback_reason=reason)

        any_successful_decide = True

        # Downstream tools (e.g. search_course_material) read these instead of
        # re-deriving "is this a vague follow-up" or "is this a comparison"
        # from the message text themselves -- both come straight from the
        # agent brain's own semantic classification for this step.
        context["is_followup"] = bool(brain_output.resolved_entities.references)
        context["goal"] = brain_output.goal

        decision = brain_output.decision
        if decision.tool:
            requested_all.append(decision.tool)
        steps.append({
            "step": step_index,
            "phase": "decide",
            "decision_type": decision.type,
            "decision_reason": decision.reason,
            "tool_requested": decision.tool,
            "agent_goal": brain_output.goal,
            "evidence_sufficient": brain_output.evidence_status.sufficient,
            "confidence": brain_output.confidence,
        })
        print(
            "[ACRLA] conversation_agent_decide "
            f"agent_iteration={step_index} agent_goal={brain_output.goal} "
            f"agent_decision_type={decision.type} tool_requested={decision.tool} "
            f"agent_confidence={brain_output.confidence:.2f}"
        )
        # Full resolved_entities/tool_arguments/evidence-requirement dumps --
        # verbose-only (see agents.debug_log); the concise line above already
        # covers the always-on goal/decision-type/tool/confidence INFO fields.
        vprint(
            "[ACRLA] conversation_agent_decide_detail "
            f"agent_iteration={step_index} "
            f"agent_decision_reason={decision.reason!r} "
            f"resolved_entities={_redact_arguments({'concepts': brain_output.resolved_entities.concepts, 'courses': brain_output.resolved_entities.courses, 'references': brain_output.resolved_entities.references})} "
            f"tool_arguments={_redact_arguments(decision.arguments)} "
            f"evidence_sufficient={brain_output.evidence_status.sufficient} "
            f"missing_evidence={brain_output.evidence_status.missing} "
            f"observation_count={len(observations)}"
        )
        vprint(
            "[ACRLA] conversation_agent_evidence_state "
            f"agent_iteration={step_index} "
            f"agent_goal={brain_output.goal} "
            f"required_evidence={[model_dump(item) for item in brain_output.evidence_status.required_evidence]} "
            f"available_evidence={brain_output.evidence_status.available_evidence} "
            f"missing_evidence_types={brain_output.evidence_status.missing_evidence} "
            f"json_mode_fallback_used={json_mode_fallback_used}"
        )

        acceptance = decision_acceptance.evaluate_decision_acceptance(
            brain_output=brain_output, observations=observations, context=context,
            json_mode_fallback_used=json_mode_fallback_used,
        )
        print(
            "[ACRLA] conversation_agent_confidence_calibration "
            f"agent_iteration={step_index} "
            f"raw_confidence={brain_output.confidence:.2f} "
            f"calibrated_confidence={acceptance.calibrated_confidence:.2f} "
            f"decision_accepted={acceptance.accepted} "
            f"confidence_adjustments={acceptance.confidence_adjustments} "
            f"acceptance_reason={acceptance.acceptance_reason or acceptance.rejection_reason}"
        )
        steps.append({
            "step": step_index, "phase": "confidence_calibration",
            "raw_confidence": brain_output.confidence, "calibrated_confidence": acceptance.calibrated_confidence,
            "decision_accepted": acceptance.accepted, "confidence_adjustments": acceptance.confidence_adjustments,
            "acceptance_reason": acceptance.acceptance_reason or acceptance.rejection_reason,
        })

        if not acceptance.accepted:
            if decision.type == "tool" or acceptance.rejection_reason == "clarification_premature_tool_available":
                # Recoverable: this is exactly the kind of gap a replan can
                # fix (pick a different tool/argument, or actually try the
                # tool that exists instead of asking the student) -- surface
                # why and let the brain decide again next step, the same
                # synthetic-observation pattern the entity/analytics gates
                # already use, instead of aborting the whole turn.
                print(
                    "[ACRLA] conversation_agent_decision_rejected "
                    f"agent_iteration={step_index} "
                    f"decision_type={decision.type} "
                    f"rejection_reason={acceptance.rejection_reason}"
                )
                observations.append({
                    "tool": "decision_acceptance_check", "arguments": {}, "rejected": False,
                    "result": {
                        "accepted": False, "rejection_reason": acceptance.rejection_reason,
                        "note": (
                            "The previous proposed step was not accepted "
                            f"({acceptance.rejection_reason}). Do not repeat it -- choose a "
                            "different tool/argument that addresses the missing evidence, or "
                            "only ask for clarification if no allowed tool can obtain it."
                        ),
                    },
                })
                continue
            _log_terminal(
                agent_step=step_index, agent_goal=brain_output.goal, agent_success=False,
                final_selected_pipeline=None, legacy_fallback_reason=acceptance.rejection_reason or "confidence_below_0.80",
            )
            return AgentResult(
                goal=brain_output.goal,
                confidence=brain_output.confidence,
                tools_requested=requested_all,
                tools_executed=executed_all,
                resolved_concepts=brain_output.resolved_entities.concepts,
                analytics_request=context.get("analytics_request"),
                steps=steps,
                fallback_reason=acceptance.rejection_reason or "confidence_below_0.80",
            )

        if decision.type == "fallback":
            _log_terminal(
                agent_step=step_index, agent_goal=brain_output.goal, agent_success=False,
                final_selected_pipeline=None, legacy_fallback_reason="agent_requested_fallback",
            )
            return AgentResult(
                goal=brain_output.goal,
                confidence=brain_output.confidence,
                tools_requested=requested_all,
                tools_executed=executed_all,
                resolved_concepts=brain_output.resolved_entities.concepts,
                analytics_request=context.get("analytics_request"),
                steps=steps,
                fallback_reason="agent_requested_fallback",
            )

        if decision.type in ("answer", "clarification"):
            # The same deterministic gate applies whether the brain wants to
            # answer OR give up and ask the student a clarifying question --
            # both are "stop gathering evidence" decisions. A goal with a
            # real deterministic data source (e.g. personal_profile_query,
            # analytics_query) must not be allowed to fall back to asking the
            # student something ACRLA already has an authoritative tool for,
            # any more than it may answer without ever having called it.
            entity_status = entity_validator.validate_answer_readiness(context, brain_output, observations)
            if not entity_status.complete:
                # Deterministic re-check disagrees with the brain's own
                # evidence_status.sufficient=true: some entity this turn is
                # supposedly about has no real tool evidence behind it, the
                # goal's own evidence-type requirement was not met, or a tool
                # call failed.
                #
                # For the specific "broad analytics needs run_analytics_query"
                # gap, do not just surface it and hope the brain reacts
                # correctly next step -- in practice a smaller model would
                # keep re-calling get_all_mastery or trying to answer from it
                # again rather than reaching for run_analytics_query, and
                # loop until the step limit forced a clarification. Force the
                # correct tool call directly instead: this is a deterministic
                # control-flow correction keyed to the gap's own structural
                # reason (not message wording), the same "trust but verify"
                # pattern used for tool-name/concept grounding elsewhere.
                if (
                    any(gap.reason == entity_validator.BROAD_ANALYTICS_GAP_REASON for gap in entity_status.gaps)
                    and all(obs.get("tool") != "run_analytics_query" for obs in observations)
                ):
                    forced_result = _force_run_analytics_query(
                        context, brain_output, observations, requested_all, executed_all, rejected_all, steps, step_index,
                    )
                    if forced_result is not None:
                        return forced_result
                    # run_analytics_query itself failed -- fall through to the
                    # generic gap handling below so the failure is surfaced
                    # and the brain gets a normal chance to replan/report it.

                gap_dicts = [model_dump(gap) for gap in entity_status.gaps]
                steps.append({"step": step_index, "phase": "entity_completeness_gate", "entity_gaps": gap_dicts})
                print(
                    "[ACRLA] conversation_agent_entity_gate "
                    f"agent_iteration={step_index} "
                    f"entity_gaps={gap_dicts}"
                )
                observations.append(entity_validator.entity_gap_observation(entity_status))
                continue
            if decision.type == "clarification":
                return _clarification_result(context, brain_output, observations, requested_all, executed_all, steps, step_index)
            return _finalize_answer(context, brain_output, observations, requested_all, executed_all, steps, step_index)

        # decision.type == "tool": execute it, append the observation, loop again.
        # Re-consulting the brain next step can pick a *different* tool once
        # this result is seen -- e.g. after run_study_recommendation resolves
        # a concept, the next step's decision becomes search_course_material
        # for that concept.
        tool_arguments = decision.arguments
        if decision.tool == "run_analytics_query":
            persisted_request = _get_or_create_analytics_request(context, brain_output)
            if persisted_request is not None:
                tool_arguments = dict(tool_arguments)
                tool_arguments["analytics_request"] = persisted_request
        raw_observation = executor.execute(decision.tool, tool_arguments, context)
        observations.append(raw_observation)
        rejected = bool(raw_observation.get("rejected"))
        result = raw_observation.get("result") or {}
        observation = ToolObservation(
            step=step_index,
            tool=decision.tool,
            arguments=tool_arguments,
            success=not rejected and not result.get("error"),
            result=result,
            summary=_summarize_one_result(result, rejected),
        )
        if rejected:
            rejected_all.append(decision.tool)
        else:
            executed_all.append(decision.tool)
        steps.append({
            "step": step_index,
            "phase": "tool_execution",
            "tool_requested": decision.tool,
            "tool_result_summary": observation.summary,
            "tool_success": observation.success,
        })
        print(
            "[ACRLA] conversation_agent_tools "
            f"agent_iteration={step_index} tool_requested={observation.tool} "
            f"tool_success={observation.success}"
        )
        vprint(
            "[ACRLA] conversation_agent_tools_detail "
            f"agent_iteration={step_index} "
            f"tool_arguments={_redact_arguments(observation.arguments)} "
            f"tool_result_summary={observation.summary!r} "
            f"observation_count={len(observations)}"
        )
    else:
        # Step limit reached without a terminal decision. Do not hallucinate:
        # answer only if the last step's evidence was judged sufficient AND
        # the deterministic entity-completeness re-check agrees, otherwise
        # ask one concise clarification, otherwise fall back.
        return _handle_step_limit_exhausted(
            context, brain_output, observations, requested_all, executed_all, steps,
            any_successful_decide=any_successful_decide,
        )


def _get_or_create_analytics_request(context: dict[str, Any], brain_output: AgentBrainOutput) -> dict[str, Any] | None:
    """Create the structured analytics request once, then always reuse it.

    The FIRST time this is called this turn (context["analytics_request"] not
    set yet), the brain's own semantic classification for this step
    (brain_output.analytics_request) becomes the canonical request. Every
    later call this turn -- even if the brain's own output drifts on a later
    step -- reuses the persisted one, so run_analytics_query is never handed
    a differently-reinterpreted operation mid-turn. A prior turn's analytics
    request (from structured conversation memory) is honored the same way if
    already present on `context` before this turn started, so a follow-up
    refines it instead of starting over.
    """
    existing = context.get("analytics_request")
    if existing is not None:
        return existing
    if brain_output.analytics_request is not None:
        created = model_dump(brain_output.analytics_request)
        context["analytics_request"] = created
        print(f"[ACRLA] conversation_agent_analytics_request_created analytics_request={created}")
        return created
    return None


def _force_run_analytics_query(
    context: dict[str, Any],
    brain_output: AgentBrainOutput,
    observations: list[dict[str, Any]],
    requested_all: list[str],
    executed_all: list[str],
    rejected_all: list[str],
    steps: list[dict[str, Any]],
    step_index: int,
) -> AgentResult | None:
    """Deterministically run run_analytics_query for a broad analytics gap.

    Uses the persisted structured `analytics_request` if one was already
    created this turn (see `_get_or_create_analytics_request`) so the
    precise operation is preserved instead of being re-inferred; falls back
    to the original user message (not a paraphrase, not get_all_mastery
    rows) when no structured request exists yet, so
    tools.analytics_tools.plan_analytics_query can still infer the operation
    from the same source text a fresh top-level question would use.

    Returns the finished AgentResult if run_analytics_query succeeded
    (finalizing immediately from its own deterministic formatted result,
    without spending another decide() call on whether to answer now) or None
    if it failed, so the caller falls through to normal gap handling.
    """
    requested_all.append("run_analytics_query")
    tool_arguments: dict[str, Any] = {"query": context.get("message", "")}
    persisted_request = _get_or_create_analytics_request(context, brain_output)
    if persisted_request is not None:
        tool_arguments["analytics_request"] = persisted_request
    raw_observation = executor.execute("run_analytics_query", tool_arguments, context)
    observations.append(raw_observation)
    rejected = bool(raw_observation.get("rejected"))
    result = raw_observation.get("result") or {}
    success = not rejected and not result.get("error")
    summary = _summarize_one_result(result, rejected)
    steps.append({
        "step": step_index,
        "phase": "forced_analytics_tool",
        "tool_requested": "run_analytics_query",
        "tool_result_summary": summary,
        "tool_success": success,
    })
    print(
        "[ACRLA] conversation_agent_forced_analytics_query "
        f"agent_iteration={step_index} "
        f"tool_success={success} "
        f"tool_result_summary={summary!r}"
    )
    if rejected:
        # Not expected in practice -- run_analytics_query is a real,
        # allowlisted tool -- but handled the same way the normal
        # tool-execution branch above handles a rejected call, for
        # consistency of the executed/rejected bookkeeping.
        rejected_all.append("run_analytics_query")
        return None
    if success:
        executed_all.append("run_analytics_query")
        return _finalize_answer(context, brain_output, observations, requested_all, executed_all, steps, step_index)
    executed_all.append("run_analytics_query")
    return None


def _finalize_answer(
    context: dict[str, Any],
    brain_output: AgentBrainOutput,
    observations: list[dict[str, Any]],
    requested_all: list[str],
    executed_all: list[str],
    steps: list[dict[str, Any]],
    step_index: int,
) -> AgentResult:
    """Select the pipeline (deterministic, evidence-gated) and generate the reply."""
    resolved_concepts = brain_output.resolved_entities.concepts
    recommendation, recommendation_reason = _recommendation_from_tool_results(observations)

    # Evidence-reliability gate: this decides whether PDF sources may be shown.
    # It is deterministic and mechanical (which tools ran + their own embedded
    # evidence verdict) on purpose -- the agent brain's answer_basis/
    # evidence_status above are synthesis guidance, never a safety override.
    if "search_course_material" in executed_all:
        search_result = (_last_observation(observations, "search_course_material") or {}).get("result") or {}
        evidence = search_result.get("evidence")
        retrieved_sources = search_result.get("sources") or []
        if not isinstance(evidence, dict):
            evidence = {"reliable": False, "coverage": "none", "supported_concepts": [], "reason": "search_tool_returned_no_evidence", "confidence": 0.0}
        fallback_reason_if_unreliable = None if evidence["reliable"] else f"evidence_not_reliable:{evidence['reason']}"
        steps.append({"step": len(steps) + 1, "phase": "observation", "evidence_status": evidence, "replan_reason": fallback_reason_if_unreliable})
        print(
            "[ACRLA] conversation_agent_evidence "
            f"evidence_reliable={evidence['reliable']} "
            f"evidence_coverage={evidence['coverage']} "
            f"replan_reason={fallback_reason_if_unreliable or 'none'}"
        )
        if evidence["reliable"] and not retrieved_sources:
            _log_terminal(
                agent_step=len(steps), agent_goal=brain_output.goal, agent_success=False,
                final_selected_pipeline=None, legacy_fallback_reason="rag_reliable_but_sources_missing",
            )
            return AgentResult(
                goal=brain_output.goal,
                confidence=brain_output.confidence,
                tools_requested=requested_all,
                tools_executed=executed_all,
                resolved_concepts=resolved_concepts,
                evidence_reliable=evidence["reliable"],
                evidence_coverage=evidence["coverage"],
                evidence_supported_concepts=evidence["supported_concepts"],
                evidence_reason=evidence["reason"],
                evidence_confidence=evidence["confidence"],
                recommendation=recommendation,
                recommendation_reason=recommendation_reason,
                analytics_request=context.get("analytics_request"),
                steps=steps,
                fallback_reason="rag_reliable_but_sources_missing",
            )
        selected_pipeline = "internal_rag" if evidence["reliable"] else "external_fallback"
        sources = retrieved_sources if evidence["reliable"] else []
        final_knowledge_strategy = "internal_candidate" if evidence["reliable"] else "external"
        agent_fallback_reason = None if evidence["reliable"] else fallback_reason_if_unreliable
    elif set(executed_all) & _DETERMINISTIC_DATA_TOOLS:
        selected_pipeline = "agent_tools"
        sources = []
        final_knowledge_strategy = "analytics"
        evidence = evidence_validator.empty_evidence("analytics_no_rag")
        agent_fallback_reason = None
    elif set(executed_all) & _CONTEXT_TOOLS:
        selected_pipeline = "context_metadata"
        sources = []
        final_knowledge_strategy = "context_only"
        evidence = evidence_validator.empty_evidence("context_only_no_rag")
        agent_fallback_reason = None
    else:
        selected_pipeline = "external_fallback"
        sources = []
        final_knowledge_strategy = "external"
        evidence = evidence_validator.empty_evidence("no_tool_executed_external")
        agent_fallback_reason = None

    steps.append({"step": len(steps) + 1, "phase": "answer", "final_goal": brain_output.goal, "selected_pipeline": selected_pipeline})

    relevant_observations = _filter_relevant(observations, brain_output.evidence_status.relevant_observations)
    token_usage: dict[str, Any] = {}
    reply = response_generator.generate_final_answer(
        context, brain_output, relevant_observations, selected_pipeline, evidence, sources=sources, token_usage=token_usage,
    )
    steps.append({"step": len(steps) + 1, "phase": "response_generation_metrics", **token_usage})
    if not reply:
        _log_terminal(
            agent_step=len(steps), agent_goal=brain_output.goal, agent_success=False,
            final_selected_pipeline=selected_pipeline, legacy_fallback_reason="empty_final_answer",
        )
        return AgentResult(
            goal=brain_output.goal,
            knowledge_strategy=final_knowledge_strategy,
            confidence=brain_output.confidence,
            tools_requested=requested_all,
            tools_executed=executed_all,
            resolved_concepts=resolved_concepts,
            evidence_reliable=evidence.get("reliable"),
            evidence_coverage=evidence.get("coverage"),
            evidence_supported_concepts=evidence.get("supported_concepts") or [],
            evidence_reason=evidence.get("reason"),
            evidence_confidence=evidence.get("confidence"),
            recommendation=recommendation,
            recommendation_reason=recommendation_reason,
            analytics_request=context.get("analytics_request"),
            steps=steps,
            fallback_reason="empty_final_answer",
        )

    _log_terminal(
        agent_step=len(steps), agent_goal=brain_output.goal, agent_success=True,
        final_selected_pipeline=selected_pipeline, legacy_fallback_reason=agent_fallback_reason,
        answer_basis=brain_output.answer_basis,
    )
    concepts_out = evidence.get("supported_concepts") or resolved_concepts
    analytics_operation, analytics_items = _analytics_reference_from_observations(observations)
    return AgentResult(
        agent_used=True,
        reply=reply,
        goal=brain_output.goal,
        knowledge_strategy=final_knowledge_strategy,
        confidence=brain_output.confidence,
        tools_requested=requested_all,
        tools_executed=executed_all,
        resolved_concepts=resolved_concepts,
        selected_pipeline=selected_pipeline,
        sources=sources,
        concepts=concepts_out,
        evidence_reliable=evidence.get("reliable"),
        evidence_coverage=evidence.get("coverage"),
        evidence_supported_concepts=evidence.get("supported_concepts") or [],
        evidence_reason=evidence.get("reason"),
        evidence_confidence=evidence.get("confidence"),
        recommendation=recommendation,
        recommendation_reason=recommendation_reason,
        analytics_operation=analytics_operation,
        analytics_items=analytics_items,
        analytics_request=context.get("analytics_request"),
        reasoning_basis=brain_output.answer_basis,
        reasoning_summary=brain_output.decision.reason,
        steps=steps,
        fallback_reason=agent_fallback_reason,
    )


def _clarification_result(
    context: dict[str, Any],
    brain_output: AgentBrainOutput,
    observations: list[dict[str, Any]],
    requested_all: list[str],
    executed_all: list[str],
    steps: list[dict[str, Any]],
    step_index: int,
) -> AgentResult:
    token_usage: dict[str, Any] = {}
    reply = response_generator.generate_final_answer(
        context, brain_output, observations, "clarification",
        evidence_validator.empty_evidence("clarification_requested"), sources=[], token_usage=token_usage,
    )
    steps.append({"step": step_index, "phase": "response_generation_metrics", **token_usage})
    if not reply:
        _log_terminal(
            agent_step=step_index, agent_goal=brain_output.goal, agent_success=False,
            final_selected_pipeline=None, legacy_fallback_reason="empty_final_answer",
        )
        return AgentResult(
            goal=brain_output.goal, confidence=brain_output.confidence,
            analytics_request=context.get("analytics_request"), steps=steps, fallback_reason="empty_final_answer",
        )
    _log_terminal(
        agent_step=step_index, agent_goal=brain_output.goal, agent_success=True,
        final_selected_pipeline="clarification", legacy_fallback_reason=None,
        answer_basis=brain_output.answer_basis,
    )
    return AgentResult(
        agent_used=True,
        reply=reply,
        goal=brain_output.goal,
        knowledge_strategy="clarification",
        confidence=brain_output.confidence,
        tools_requested=requested_all,
        tools_executed=executed_all,
        resolved_concepts=brain_output.resolved_entities.concepts,
        selected_pipeline="clarification",
        analytics_request=context.get("analytics_request"),
        steps=steps,
    )


def _handle_step_limit_exhausted(
    context: dict[str, Any],
    brain_output: AgentBrainOutput | None,
    observations: list[dict[str, Any]],
    requested_all: list[str],
    executed_all: list[str],
    steps: list[dict[str, Any]],
    *,
    any_successful_decide: bool = True,
) -> AgentResult:
    """MAX_AGENT_STEPS reached without a terminal decision. Never hallucinate:
    answer from sufficient evidence if we have it AND the deterministic
    entity-completeness re-check agrees, else ask one concise clarification
    (naming the specific unresolved entity if that is why), else defer to the
    legacy fallback."""
    print(f"[ACRLA] conversation_agent_result agent_iteration={MAX_AGENT_STEPS} step_limit_reached=True")
    if brain_output is None or not any_successful_decide:
        # Every step failed to produce a parseable decision -- distinguish
        # that from a "the brain kept asking for more tools" exhaustion so
        # the fallback reason is honest about what actually happened. Note:
        # agent_brain.decide() always returns a (blank, on failure)
        # AgentBrainOutput rather than None, so `any_successful_decide` is
        # the real signal here; the None check is defensive only.
        _log_terminal(
            agent_step=MAX_AGENT_STEPS, agent_goal="unclear", agent_success=False,
            final_selected_pipeline=None, legacy_fallback_reason="decision_parse_failed_repeatedly",
            step_limit_reached=True,
        )
        return AgentResult(steps=steps, fallback_reason="decision_parse_failed_repeatedly")

    if brain_output.evidence_status.sufficient and observations:
        entity_status = entity_validator.validate_answer_readiness(context, brain_output, observations)
        if entity_status.complete:
            return _finalize_answer(context, brain_output, observations, requested_all, executed_all, steps, MAX_AGENT_STEPS)
        # No steps left to replan with. Do not answer with a gap -- ask a
        # concise clarification that names the specific unresolved entity
        # (or reports it was not found) instead of guessing.
        gap_dicts = [model_dump(gap) for gap in entity_status.gaps]
        steps.append({"step": MAX_AGENT_STEPS, "phase": "entity_completeness_gate", "entity_gaps": gap_dicts})
        print(f"[ACRLA] conversation_agent_entity_gate agent_iteration={MAX_AGENT_STEPS} entity_gaps={gap_dicts}")
        brain_output = _with_entity_gap_missing(brain_output, entity_status)

    if observations:
        return _clarification_result(context, brain_output, observations, requested_all, executed_all, steps, MAX_AGENT_STEPS)

    _log_terminal(
        agent_step=MAX_AGENT_STEPS, agent_goal=brain_output.goal, agent_success=False,
        final_selected_pipeline=None, legacy_fallback_reason="max_agent_steps_exceeded",
        step_limit_reached=True,
    )
    return AgentResult(
        goal=brain_output.goal,
        confidence=brain_output.confidence,
        tools_requested=requested_all,
        tools_executed=executed_all,
        resolved_concepts=brain_output.resolved_entities.concepts,
        analytics_request=context.get("analytics_request"),
        steps=steps,
        fallback_reason="max_agent_steps_exceeded",
    )


def _with_entity_gap_missing(brain_output: AgentBrainOutput, entity_status) -> AgentBrainOutput:
    """Fold entity-completeness gap reasons into evidence_status.missing so
    the clarification reply (agents.response_generator) can name the specific
    unresolved entity instead of asking a generic clarifying question."""
    data = model_dump(brain_output)
    data["evidence_status"]["sufficient"] = False
    existing_missing = data["evidence_status"].get("missing") or []
    data["evidence_status"]["missing"] = list(dict.fromkeys(
        existing_missing + entity_validator.entity_gap_descriptions(entity_status)
    ))
    return model_validate(AgentBrainOutput, data)


def _filter_relevant(observations: list[dict[str, Any]], relevant_tool_names: list[str] | None) -> list[dict[str, Any]]:
    """Drop observations the agent brain flagged irrelevant/ignored before they
    reach response_generator, so a stale/unrelated tool result cannot leak
    into the final answer's synthesis. Falls back to using everything if the
    brain didn't name any relevant tool."""
    if not relevant_tool_names:
        return observations
    relevant = {name for name in relevant_tool_names if name}
    filtered = [obs for obs in observations if obs.get("tool") in relevant]
    return filtered or observations


def _log_terminal(
    *, agent_step: int, agent_goal: str, agent_success: bool,
    final_selected_pipeline: str | None, legacy_fallback_reason: str | None,
    answer_basis: str | None = None, step_limit_reached: bool = False,
) -> None:
    print(
        "[ACRLA] conversation_agent_result "
        f"agent_iteration={agent_step} "
        f"agent_goal={agent_goal} "
        f"agent_success={agent_success} "
        f"answer_basis={answer_basis} "
        f"selected_pipeline={final_selected_pipeline} "
        f"fallback_reason={legacy_fallback_reason or 'none'} "
        f"step_limit_reached={step_limit_reached}"
    )


def _redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Log argument shape/short values only -- never a full free-text blob."""
    redacted: dict[str, Any] = {}
    for key, value in (arguments or {}).items():
        if isinstance(value, str) and len(value) > 80:
            redacted[key] = value[:80] + "...(truncated)"
        else:
            redacted[key] = value
    return redacted


def _summarize_one_result(result: dict[str, Any], rejected: bool) -> str:
    if rejected:
        return "rejected: tool not allowed"
    # A truthy check, not "error" in result: a tool result may legitimately
    # carry an "error" key set to None/"" when there was no error (e.g.
    # tools.analytics_tools.run_analytics_query_tool always includes
    # "error": result.get("error")) -- key *presence* alone would wrongly
    # summarize a successful result as "error: None".
    if result.get("error"):
        return f"error: {result['error']}"
    return "returned keys: " + ", ".join(sorted(result.keys())) if result else "empty result"


def _last_observation(observations: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    for obs in reversed(observations):
        if obs.get("tool") == tool_name:
            return obs
    return None


def _recommendation_from_tool_results(tool_results: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    """Build a recommendation + reason when a recommendation tool ran this turn."""
    for item in tool_results:
        tool_name = item.get("tool")
        if tool_name not in _SELECTION_TOOLS:
            continue
        result = item.get("result") or {}
        if tool_name == "run_study_recommendation":
            concept = result.get("recommended_concept")
            mastery = result.get("current_mastery")
            candidates = result.get("candidates") or []
        else:
            selected = result.get("selected_concept") or {}
            concept = selected.get("concept")
            mastery = selected.get("current_mastery")
            candidates = result.get("candidates") or []
        if not concept:
            continue
        other_names = [c.get("concept") for c in candidates if c.get("concept") and c.get("concept") != concept]
        reason = f"{concept} has lower current mastery"
        if isinstance(mastery, (int, float)):
            reason += f" ({mastery:.0%})"
        if other_names:
            reason += f" than {', '.join(other_names)}"
        reason += ", so it's a good place to start."
        return {"concept": concept, "current_mastery": mastery, "candidates": candidates}, reason
    return None, None


def _analytics_reference_from_observations(observations: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract authoritative analytics rows for structured follow-up memory.

    The response text is not a safe source for later questions like "which one
    is strongest?" or "from what course?". The tool result is. This helper
    carries forward the rows that were actually used, so later turns can answer
    deterministically when the LLM planner is unavailable or rate-limited.
    """
    for observation in reversed(observations or []):
        result = observation.get("result") or {}
        if observation.get("tool") == "run_analytics_query":
            items = result.get("items") if isinstance(result.get("items"), list) else []
            plan = result.get("plan") if isinstance(result.get("plan"), dict) else {}
            if items:
                return plan.get("operation"), items
        if observation.get("tool") == "get_all_mastery":
            rows = result.get("mastery") if isinstance(result.get("mastery"), list) else []
            if rows:
                return "list", rows
        if observation.get("tool") == "get_mastery_for_concepts":
            rows = result.get("mastery") if isinstance(result.get("mastery"), list) else []
            if rows:
                return "get_value", rows
    return None, []
