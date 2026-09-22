"""Shared text-normalization helpers used across tool and agent modules.

Extracted so `_normalize_key`/`_routing_tokens`/`_singular_token` have one
definition instead of being copy-pasted in agent_tools.py, conversation_agent.py,
and chat_orchestrator.py.
"""

from __future__ import annotations

import re


def normalize_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def singular_token(token: str) -> str:
    token = str(token or "").strip().lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


ROUTING_STOPWORDS = {
    "a", "an", "and", "are", "about", "can", "could", "do", "does", "explain",
    "for", "from", "give", "how", "i", "in", "into", "is", "it", "me", "my",
    "of", "on", "please", "tell", "the", "this", "to", "using", "what", "when",
    "where", "which", "who", "why", "with", "you", "your", "compare", "versus",
    "vs", "between",
}


def routing_tokens(text: str | None) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in ROUTING_STOPWORDS
    }
