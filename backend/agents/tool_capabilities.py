"""Tool capability metadata: what evidence each allowlisted tool can provide.

This is the structural link between "the planner has a missing-evidence
item of type X" and "which tool could resolve it" -- the agent brain's
prompt renders this metadata so it can map missing evidence to a tool
generically, and `agents.decision_acceptance`/`agents.entity_validator` use
it deterministically to check whether a chosen tool actually addresses a
known gap, instead of either side relying on memorized example questions.

Every tool in `agents.agent_tools.TOOL_REGISTRY` has an entry here. Adding a
new tool means adding one entry -- no other code needs to change to make the
new tool "visible" to evidence-gap resolution.
"""

from __future__ import annotations

from typing import Any

# entity_type values are free-form semantic labels (not a fixed enum, see
# agents.agent_models.EvidenceRequirement) shared between this file, the
# agent brain prompt, and agents.decision_acceptance.
#
# authoritative=True means the tool's own data is the real, deterministic
# source of truth for what it provides (never invented/approximated).
# answer_with_external_knowledge is the one deliberate exception: it
# generates a general-knowledge answer, so it does not "provide" a stored
# fact -- it is not authoritative over ACRLA's own student/course data.
TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "get_recent_conversation": {"provides": ["conversation_history"], "authoritative": True, "scope": "session"},
    "get_last_reference": {"provides": ["conversation_history", "previous_reference"], "authoritative": True, "scope": "session"},
    "get_recent_structured_turns": {"provides": ["conversation_history"], "authoritative": True, "scope": "session"},
    "get_last_recommendation": {"provides": ["previous_recommendation"], "authoritative": True, "scope": "session"},
    "get_last_response_metadata": {"provides": ["previous_answer_metadata"], "authoritative": True, "scope": "session"},
    "get_current_scope": {"provides": ["remediation_scope"], "authoritative": True, "scope": "session"},
    "get_current_course": {"provides": ["current_course"], "authoritative": True, "scope": "session"},
    "get_available_concepts": {"provides": ["available_concepts"], "authoritative": True, "scope": "course"},
    "get_mastery_for_concepts": {"provides": ["student_mastery_rows"], "authoritative": True, "scope": "student"},
    "get_all_mastery": {"provides": ["student_mastery_rows"], "authoritative": True, "scope": "student"},
    "get_course_for_concepts": {"provides": ["course_membership"], "authoritative": True, "scope": "course"},
    "get_mastery_policy": {"provides": ["mastery_policy_explanation"], "authoritative": True, "scope": "global"},
    "get_mastery_scoring_methodology": {"provides": ["scoring_methodology_explanation"], "authoritative": True, "scope": "global"},
    "select_lowest_mastery_concept": {"provides": ["study_recommendation"], "authoritative": True, "scope": "student"},
    "select_lowest_mastery_among_previous_turn": {"provides": ["study_recommendation"], "authoritative": True, "scope": "student"},
    "get_student_learning_profile": {
        "provides": ["student_name", "preferences", "current_course", "tutoring_strategy", "student_mastery_rows"],
        "authoritative": True, "scope": "student",
    },
    "run_study_recommendation": {"provides": ["study_recommendation"], "authoritative": True, "scope": "student"},
    "search_course_material": {
        "provides": ["course_material"], "authoritative": True, "scope": "course",
        "note": "authoritative only when the tool's own embedded evidence.reliable is true",
    },
    "get_tutoring_strategy": {"provides": ["tutoring_strategy"], "authoritative": True, "scope": "student"},
    "run_analytics_query": {
        "provides": ["filtered_mastery_results", "rankings", "course_averages", "overall_mastery", "student_mastery_rows"],
        "authoritative": True, "scope": "student",
    },
    "answer_with_external_knowledge": {"provides": ["external_knowledge_answer"], "authoritative": False, "scope": "global"},
    "get_source_provenance": {"provides": ["previous_answer_metadata"], "authoritative": True, "scope": "session"},
    "explain_methodology": {"provides": ["tutoring_methodology_explanation"], "authoritative": True, "scope": "global"},
    "explain_recommendation": {"provides": ["previous_recommendation_reason"], "authoritative": True, "scope": "session"},
    "get_mastery_guard_response": {"provides": ["mastery_guard_response"], "authoritative": True, "scope": "global"},
    "get_course_structure": {"provides": ["course_structure"], "authoritative": True, "scope": "course"},
    "get_preferences": {"provides": ["preferences"], "authoritative": True, "scope": "student"},
    "switch_focus": {"provides": ["focus_switch_confirmation"], "authoritative": True, "scope": "session"},
}


def tools_providing(evidence_type: str) -> list[str]:
    """Every allowlisted tool whose `provides` list includes this evidence
    type, authoritative tools first. Returns [] for an evidence type nothing
    in the registry can supply (e.g. something only a human/teacher knows)."""
    matches = [
        name for name, capability in TOOL_CAPABILITIES.items()
        if evidence_type in (capability.get("provides") or [])
    ]
    return sorted(matches, key=lambda name: not TOOL_CAPABILITIES[name].get("authoritative", False))


def tool_provides(tool_name: str) -> list[str]:
    """The evidence types one tool provides, or [] if the tool is unknown."""
    return list((TOOL_CAPABILITIES.get(tool_name) or {}).get("provides") or [])


def is_authoritative(tool_name: str) -> bool:
    return bool((TOOL_CAPABILITIES.get(tool_name) or {}).get("authoritative", False))


def render_capabilities_for_prompt() -> str:
    """Compact, deterministic text rendering of the capability table for the
    agent brain's own prompt -- so the LLM maps missing evidence to a tool
    using this metadata, not memorized example questions."""
    lines = []
    for tool_name, capability in TOOL_CAPABILITIES.items():
        provides = ", ".join(capability.get("provides") or [])
        authoritative = "authoritative" if capability.get("authoritative") else "non-authoritative"
        lines.append(f"- {tool_name}: provides [{provides}] ({authoritative}, scope={capability.get('scope')})")
    return "\n".join(lines)


# NOTE: this module used to also export render_capabilities_grouped_for_prompt/
# render_capabilities_compact_for_prompt (terser TOOL_CAPABILITIES renderings)
# for agents.simple_planner's prompt. That planner no longer asks the LLM to
# select tools at all -- see agents.plan_compiler, which maps goal ->
# tool(s) deterministically in code -- so neither renderer has a caller left.
# Removed rather than left as dead code; render_capabilities_for_prompt below
# (agent_brain's own iterative-mode prompt, which still asks the LLM to pick
# a tool per step) is unaffected.
