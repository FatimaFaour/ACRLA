"""Verbose debug logging gate.

Large/bulk diagnostic dumps -- full canonical course records, complete
concept manifests, full analytics item arrays, repeated scope-concept dumps,
full tool arguments/results -- print only when `ACRLA_DEBUG_VERBOSE=true`.

Concise INFO logs (goal, selected pipeline, tools executed, planner/response
call counts, prompt/completion tokens, fallback reason, provider error
category/status code, latency) always print regardless of this flag -- they
are the caller's own always-on `print(...)` lines, not routed through here.

Never logs API keys, full prompts, student-sensitive data, or complete RAG
chunks even when verbose is enabled -- this only controls whether an
already-redacted/structural dump appears, not what may be dumped.
"""

from __future__ import annotations


def verbose_enabled() -> bool:
    from config import get_settings

    return bool(get_settings().acrla_debug_verbose)


def vprint(message: str) -> None:
    """Print `message` only when ACRLA_DEBUG_VERBOSE is enabled."""
    if verbose_enabled():
        print(message)
