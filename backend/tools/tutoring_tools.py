"""Tutoring strategy lookup tool."""

from __future__ import annotations

from typing import Any


def get_tutoring_strategy_tool(context: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "difficulty": context.get("difficulty"),
        "tutoring_strategy": context.get("tutoring_strategy"),
    }
