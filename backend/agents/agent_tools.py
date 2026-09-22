"""Tool registry + executor for the ACRLA conversational agent.

This module is now a thin aggregator: the actual tool implementations live in
`backend/tools/*` (split by domain -- rag, analytics, mastery, course, memory,
tutoring, external, profile). Keeping this module's public surface
(`TOOL_REGISTRY`, `execute_agent_tools`, `MAX_AGENT_TOOL_CALLS`) stable means
`agents.conversation_agent` (and anything else importing from here) does not
need to change as the tool implementations move/evolve.

The agent planner may request these tools by name, but this module enforces an
allowlist and a maximum number of calls. Almost every tool returns structured
data only and never generates natural language or updates mastery; the one
deliberate exception is `answer_with_external_knowledge`, whose entire job is
producing a controlled, non-Moodle-grounded general-knowledge answer via the
existing external fallback prompt (see `tools.external_tools`).
"""

from __future__ import annotations

from typing import Any, Callable

from tools.rag_tools import search_course_material_tool
from tools.course_tools import (
    get_current_scope_tool,
    get_current_course_tool,
    get_available_concepts_tool,
    get_course_for_concepts_tool,
)
from tools.mastery_tools import (
    get_mastery_for_concepts_tool,
    get_all_mastery_tool,
    get_mastery_policy_tool,
    get_mastery_scoring_methodology_tool,
    select_lowest_mastery_concept_tool,
    select_lowest_mastery_among_previous_turn_tool,
)
from tools.profile_tools import get_student_learning_profile_tool, run_study_recommendation_tool
from tools.memory_tools import (
    get_recent_conversation_tool,
    get_last_reference_tool,
    get_recent_structured_turns_tool,
    get_last_recommendation_tool,
    get_last_response_metadata_tool,
)
from tools.tutoring_tools import get_tutoring_strategy_tool
from tools.analytics_tools import run_analytics_query_tool
from tools.external_tools import answer_with_external_knowledge_tool
from tools.dialogue_tools import (
    get_source_provenance_tool,
    explain_methodology_tool,
    explain_recommendation_tool,
    get_mastery_guard_response_tool,
    get_course_structure_tool,
    get_preferences_tool,
    switch_focus_tool,
)
from tools.tutor_state_tools import (
    advance_tutor_state_tool,
    generate_practice_question_tool,
    evaluate_practice_answer_tool,
)
from tools.assessment_tools import run_quick_progress_check_tool


MAX_AGENT_TOOL_CALLS = 4


TOOL_REGISTRY: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    "get_recent_conversation": get_recent_conversation_tool,
    "get_last_reference": get_last_reference_tool,
    "get_recent_structured_turns": get_recent_structured_turns_tool,
    "get_last_recommendation": get_last_recommendation_tool,
    "get_last_response_metadata": get_last_response_metadata_tool,
    "get_current_scope": get_current_scope_tool,
    "get_current_course": get_current_course_tool,
    "get_available_concepts": get_available_concepts_tool,
    "get_mastery_for_concepts": get_mastery_for_concepts_tool,
    "get_all_mastery": get_all_mastery_tool,
    "get_course_for_concepts": get_course_for_concepts_tool,
    "get_mastery_policy": get_mastery_policy_tool,
    "get_mastery_scoring_methodology": get_mastery_scoring_methodology_tool,
    "select_lowest_mastery_concept": select_lowest_mastery_concept_tool,
    "select_lowest_mastery_among_previous_turn": select_lowest_mastery_among_previous_turn_tool,
    "get_student_learning_profile": get_student_learning_profile_tool,
    "run_study_recommendation": run_study_recommendation_tool,
    "search_course_material": search_course_material_tool,
    "get_tutoring_strategy": get_tutoring_strategy_tool,
    "run_analytics_query": run_analytics_query_tool,
    "answer_with_external_knowledge": answer_with_external_knowledge_tool,
    "get_source_provenance": get_source_provenance_tool,
    "explain_methodology": explain_methodology_tool,
    "explain_recommendation": explain_recommendation_tool,
    "get_mastery_guard_response": get_mastery_guard_response_tool,
    "get_course_structure": get_course_structure_tool,
    "get_preferences": get_preferences_tool,
    "switch_focus": switch_focus_tool,
    "advance_tutor_state": advance_tutor_state_tool,
    "generate_practice_question": generate_practice_question_tool,
    "evaluate_practice_answer": evaluate_practice_answer_tool,
    "run_quick_progress_check": run_quick_progress_check_tool,
}


def execute_agent_tools(actions: list[dict[str, Any]], context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Execute up to four allowlisted tools and report rejected requests."""
    results: list[dict[str, Any]] = []
    executed: list[str] = []
    rejected: list[str] = []
    for action in actions[:MAX_AGENT_TOOL_CALLS]:
        tool_name = str(action.get("tool") or "").strip()
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        tool = TOOL_REGISTRY.get(tool_name)
        if not tool:
            rejected.append(tool_name or "unknown")
            continue
        try:
            output = tool(context, arguments)
        except Exception as exc:
            output = {"error": str(exc)}
        results.append({"tool": tool_name, "arguments": arguments, "result": output})
        executed.append(tool_name)
        if tool_name in {"select_lowest_mastery_concept", "select_lowest_mastery_among_previous_turn"}:
            selected = (output or {}).get("selected_concept") or {}
            if selected.get("concept"):
                context["agent_selected_concept"] = selected["concept"]
        elif tool_name == "run_study_recommendation":
            recommended = (output or {}).get("recommended_concept")
            if recommended:
                context["agent_selected_concept"] = recommended
    return results, executed, rejected
