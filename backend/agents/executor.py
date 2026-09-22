"""Executor: runs exactly one named tool call for the agent loop.

Allowlist validation and the 4-call-per-turn cap are enforced by
`agents.agent_tools.execute_agent_tools` (the tool registry itself); this
module's job is the single-call primitive the loop in
`agents.conversation_agent` uses every step: run one tool by name, return one
structured observation, and clearly mark whether it actually executed or was
rejected (unknown/disallowed tool name) so the loop can log and reason about
the difference.
"""

from __future__ import annotations

from typing import Any

from agents.agent_tools import execute_agent_tools


def execute(tool: str, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Run one tool call and return a single observation.

    Returns a dict shaped like:
        {"tool": tool, "arguments": arguments, "result": {...}, "rejected": bool}

    `rejected=True` means the tool name is not in the allowlist -- the call
    never ran. A tool that ran but raised internally still comes back with
    `rejected=False` and an `"error"` key inside `result` (see
    `agents.agent_tools.execute_agent_tools`), since that is a genuine
    observation the planner should be allowed to see and react to.
    """
    results, executed, rejected = execute_agent_tools([{"tool": tool, "arguments": arguments}], context)
    if executed:
        observation = dict(results[0])
        observation["rejected"] = False
        return observation
    return {
        "tool": tool,
        "arguments": arguments,
        "result": {"error": f"tool_not_allowed:{tool}"},
        "rejected": True,
    }
