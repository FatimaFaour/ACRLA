"""Confidence calibration: replaces a single fixed confidence threshold with
an acceptance decision that also weighs structural signals -- schema
validity, goal clarity, entity resolution, tool-allowlist/precondition
validity, whether the decision actually addresses a known missing-evidence
item, whether it conflicts with observations, and whether it followed a
JSON-mode recovery.

Rationale: a recovered decision can carry a lower raw confidence (e.g. 0.70)
purely because the model second-guesses itself after a retry, even though
every structural signal says the decision is sound -- a valid, read-only tool
call that directly addresses a real gap. Rejecting that outright (a flat
`confidence < 0.80`) throws away a perfectly safe next step. A low-confidence
final answer is a different risk profile -- nothing left to verify it against
before it reaches the student -- so it stays held to the original bar.
"""

from __future__ import annotations

from typing import Any

from agents.agent_models import AgentBrainOutput, DecisionAcceptance
from agents.agent_tools import TOOL_REGISTRY
from agents.tool_capabilities import tool_provides

# Below this, nothing is accepted regardless of type -- a decision this
# unconfident is not "calibrated up", it is simply not usable.
_FLOOR_CONFIDENCE = 0.35
# A read-only, allowlisted tool call that addresses a known missing-evidence
# item may be accepted below the original flat bar (unlike a final answer).
_TOOL_ACCEPTANCE_THRESHOLD = 0.55
# A final answer keeps the original bar -- nothing left to verify it against
# once it reaches the student.
_ANSWER_ACCEPTANCE_THRESHOLD = 0.80
_CLARIFICATION_ACCEPTANCE_THRESHOLD = 0.50


def evaluate_decision_acceptance(
    *,
    brain_output: AgentBrainOutput,
    observations: list[dict[str, Any]],
    context: dict[str, Any],
    json_mode_fallback_used: bool = False,
) -> DecisionAcceptance:
    """Calibrate `brain_output.confidence` against structural signals and
    decide whether this step's decision may be acted on.

    `context`/`observations` are read-only here (never mutated) -- this is a
    pure evaluation, the same "trust but verify" pattern
    `agent_brain._ground_decision` already applies to tool names and concept
    names, just for the accept/reject question instead of grounding values.
    """
    decision = brain_output.decision
    adjustments: list[str] = []
    calibrated = brain_output.confidence

    if brain_output.goal == "unclear":
        calibrated -= 0.15
        adjustments.append("goal_unclear:-0.15")

    if brain_output.resolved_entities.unresolved:
        calibrated -= 0.10
        adjustments.append("unresolved_entities_present:-0.10")

    if json_mode_fallback_used:
        calibrated -= 0.05
        adjustments.append("json_mode_recovery:-0.05")

    missing_types = set(brain_output.evidence_status.missing_evidence or [])

    tool_valid = True
    tool_addresses_missing_evidence = False
    tool_is_duplicate = False
    if decision.type == "tool":
        tool_valid = bool(decision.tool) and decision.tool in TOOL_REGISTRY
        if not tool_valid:
            calibrated -= 0.50
            adjustments.append("tool_not_allowlisted:-0.50")
        else:
            provided = set(tool_provides(decision.tool))
            tool_addresses_missing_evidence = bool(missing_types & provided) or not missing_types
            if tool_addresses_missing_evidence:
                calibrated += 0.10
                adjustments.append("tool_addresses_missing_evidence:+0.10")
            else:
                calibrated -= 0.10
                adjustments.append("tool_does_not_address_missing_evidence:-0.10")
            tool_is_duplicate = _is_duplicate_successful_call(observations, decision.tool, decision.arguments)
            if tool_is_duplicate:
                calibrated -= 0.20
                adjustments.append("duplicate_tool_call:-0.20")

    calibrated = max(0.0, min(1.0, calibrated))

    if calibrated < _FLOOR_CONFIDENCE:
        return DecisionAcceptance(
            accepted=False, calibrated_confidence=calibrated,
            rejection_reason="confidence_below_floor", confidence_adjustments=adjustments,
        )

    if decision.type == "tool":
        if not tool_valid:
            reason = "tool_not_allowlisted"
        elif tool_is_duplicate:
            reason = "duplicate_tool_call_already_succeeded"
        elif calibrated < _TOOL_ACCEPTANCE_THRESHOLD:
            reason = "tool_call_confidence_too_low"
        else:
            reason = "tool_call_accepted_read_only_and_addresses_gap"
        accepted = tool_valid and not tool_is_duplicate and calibrated >= _TOOL_ACCEPTANCE_THRESHOLD
        return DecisionAcceptance(
            accepted=accepted, calibrated_confidence=calibrated,
            acceptance_reason=reason if accepted else None,
            rejection_reason=None if accepted else reason,
            confidence_adjustments=adjustments,
        )

    if decision.type == "answer":
        accepted = calibrated >= _ANSWER_ACCEPTANCE_THRESHOLD
        reason = "answer_accepted_sufficient_confidence" if accepted else "answer_rejected_insufficient_confidence"
        return DecisionAcceptance(
            accepted=accepted, calibrated_confidence=calibrated,
            acceptance_reason=reason if accepted else None,
            rejection_reason=None if accepted else reason,
            confidence_adjustments=adjustments,
        )

    if decision.type == "clarification":
        # A clarification is only justified once no allowed tool could
        # obtain the missing evidence -- otherwise the brain is giving up
        # prematurely instead of trying the tool that is actually available.
        untried_tool_exists = _untried_tool_covers_missing_evidence(missing_types, observations)
        if untried_tool_exists:
            adjustments.append("untried_tool_available_for_missing_evidence:reject")
            return DecisionAcceptance(
                accepted=False, calibrated_confidence=calibrated,
                rejection_reason="clarification_premature_tool_available",
                confidence_adjustments=adjustments,
            )
        accepted = calibrated >= _CLARIFICATION_ACCEPTANCE_THRESHOLD
        reason = "clarification_accepted_no_tool_can_resolve_gap" if accepted else "clarification_rejected_insufficient_confidence"
        return DecisionAcceptance(
            accepted=accepted, calibrated_confidence=calibrated,
            acceptance_reason=reason if accepted else None,
            rejection_reason=None if accepted else reason,
            confidence_adjustments=adjustments,
        )

    # decision.type == "fallback": the brain has already said "I cannot
    # safely handle this" -- that is itself a terminal, safe decision, always
    # accepted (this is what routes to the legacy orchestrator).
    return DecisionAcceptance(
        accepted=True, calibrated_confidence=calibrated,
        acceptance_reason="fallback_always_accepted", confidence_adjustments=adjustments,
    )


def _is_duplicate_successful_call(observations: list[dict[str, Any]], tool: str, arguments: dict[str, Any]) -> bool:
    """True if this exact (tool, arguments) pair already produced a
    successful observation this turn -- exact-match only, so a legitimate
    re-call with different arguments (e.g. search_course_material for a
    different concept) is never blocked, only a genuinely wasted repeat.

    Compares only the LLM's own proposed arguments, ignoring
    "analytics_request" -- that key is injected by
    agents.conversation_agent from the persisted structured request, not
    something the LLM itself varies, so it must not defeat this comparison.
    """
    comparable = {key: value for key, value in (arguments or {}).items() if key != "analytics_request"}
    for obs in observations:
        if obs.get("tool") != tool or obs.get("rejected"):
            continue
        stored_arguments = obs.get("arguments") or {}
        stored_comparable = {key: value for key, value in stored_arguments.items() if key != "analytics_request"}
        if stored_comparable != comparable:
            continue
        result = obs.get("result")
        if isinstance(result, dict) and not result.get("error") and result.get("success") is not False:
            return True
    return False


def _untried_tool_covers_missing_evidence(missing_types: set[str], observations: list[dict[str, Any]]) -> bool:
    if not missing_types:
        return False
    from agents.tool_capabilities import tools_providing

    attempted_tools = {obs.get("tool") for obs in observations if not obs.get("rejected")}
    for missing_type in missing_types:
        candidates = tools_providing(missing_type)
        if any(candidate not in attempted_tools for candidate in candidates):
            return True
    return False
