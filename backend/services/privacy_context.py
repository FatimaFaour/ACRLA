"""Centralized privacy/data-minimization boundary between ACRLA's local
student context and any external LLM call.

Research Question 2: "How can ACRLA preserve student data privacy while
combining local data processing, retrieval-augmented generation and an
external generative AI model?" The answer this module implements: every
prompt-building function that constructs text destined for an external LLM
(services.llm_factory.get_llm/get_json_llm -- whichever provider is
currently configured) builds its student-facing context through
`build_llm_safe_student_context(context)` instead of hand-picking fields
itself, so ONE explicit allow-list is enforced everywhere, rather than
being an incidental, per-call-site convention. The ACRLA privacy audit
found exactly that inconsistency: agents.response_generator's own
`student_profile` dict deliberately minimized identity, while
tools.external_tools/pipelines.hybrid_pipeline's `build_external_fallback_
prompt` embedded the student's real name a few call frames away -- the
same student, the same turn, two different outcomes. This module exists so
there is only ever one place that decision gets made.

ALLOW-LIST, NOT BLOCK-LIST: a new field added to `context` anywhere else
in the codebase is invisible here until someone deliberately adds it to
`build_llm_safe_student_context` below -- the safe default is "not sent,"
never "sent unless blocked." `context["student_name"]`/`context["email"]`/
`context["student_id"]`/Moodle ids are never read by this module at all,
by construction, not filtered out afterward.

Scope of what this module does NOT do (see the task report for the full
list; each is deliberately separate, ongoing work):
- No NLP/regex redaction of free-text the student typed themselves
  (`context["message"]`) -- if a student voluntarily types their own name
  into a chat message, that text still reaches the LLM verbatim, same as
  any other course-related answer needs the student's own words to work.
- No authentication/access control, retention/consent, or transport
  encryption -- this module only governs what a LOCAL, already-authorized
  turn's context is allowed to put in an OUTBOUND LLM prompt.
- Does not touch mastery UPDATE logic, the tutor state machine, RAG
  retrieval, or Ollama/ChromaDB in any way -- it only reads (never writes)
  a handful of already-computed context fields.
"""

from __future__ import annotations

import time
from typing import Any


def build_llm_safe_student_context(
    context: dict[str, Any], *, include_mastery: bool = False,
) -> dict[str, Any]:
    """Return the minimal, explicitly allow-listed subset of an agent
    `context` dict that an external LLM prompt may use.

    Never returns (because they are never read from `context` at all):
    student_name/username, email, moodle_user_id, student_id/db id,
    session_id, any other student's data, a raw profile/learning_state
    object, or full grade/assessment history.

    Always includes (pedagogically necessary, never identity-bearing):
    course name, current/resolved concepts, available concepts,
    remediation level + its textual scope instruction, difficulty, and the
    tutoring strategy's name + phrasing instructions (never its internal
    "reason", which -- like agents.response_generator's own pre-existing
    `student_profile` already established -- only explains a routing
    decision and never changes how an answer should read).

    `include_mastery=True` additionally carries weak-concept NAMES (never
    a raw mastery percentage/row) -- opt in per call site for a turn that
    genuinely benefits from knowing which concepts are weak (e.g. the
    external-knowledge fallback's existing redirect-to-weak-areas
    behavior), never a blanket default. This still never includes
    anything identity-bearing; a call site that needs an actual mastery
    NUMBER (e.g. analytics output) reads it from its own tool result, not
    from this helper -- mastery values already flow through
    agents.response_generator's `tool_results` field exactly when an
    analytics/mastery tool ran that turn, unaffected by this function.
    """
    current_course = context.get("current_course") or {}
    tutoring_strategy = context.get("tutoring_strategy") or {}
    safe_context: dict[str, Any] = {
        "course_name": current_course.get("name"),
        "current_concept": context.get("current_concept"),
        "available_concepts": list(context.get("available_concepts") or []),
        "resolved_concepts": list(context.get("resolved_concepts") or []),
        "remediation_level": context.get("remediation_level"),
        "scope_rules": context.get("scope_rules"),
        "difficulty": context.get("difficulty"),
        "tutoring_strategy": {
            "name": tutoring_strategy.get("name"),
            "instructions": tutoring_strategy.get("instructions"),
        },
    }
    if include_mastery:
        safe_context["weak_concepts"] = list(context.get("weak_concepts") or [])
    return safe_context


# Field names this module will NEVER read from `context`, kept here as a
# single source of truth for the structural regression tests (scratchpad
# test_privacy_context.py) that assert this function's own source never
# references them -- not used by the function itself (the allow-list above
# already guarantees this by construction; this tuple only documents it).
FORBIDDEN_IDENTITY_FIELDS = (
    "student_name", "username", "email", "moodle_user_id", "student_id", "db_id",
)


# ===========================================================================
# STEP 4: institutional (course-document) content sensitivity.
#
# RQ2 now has two categories to protect at the same local/external boundary:
#   A. Student data      -- build_llm_safe_student_context, above.
#   B. Institutional data -- filter_course_chunks_for_external, below.
# Both live in this one module deliberately (the task's own "reuse/extend,
# do not create multiple unrelated privacy systems" instruction) and share
# the same posture: explicit, deterministic, allow-list-shaped, and never
# delegated to the external LLM itself.
# ===========================================================================

# Three institution-assigned sensitivity levels for retrieved course content.
# PUBLIC: the institution explicitly allows relevant retrieved content
#   externally. INTERNAL: usable locally; only the minimum retrieved
#   excerpt(s) actually needed for this turn may leave the local
#   environment. RESTRICTED: raw retrieved text must never reach an
#   external LLM prompt at all.
SENSITIVITY_LEVELS = ("PUBLIC", "INTERNAL", "RESTRICTED")

# Safe default for a chunk with no (or an unrecognized) sensitivity value --
# e.g. a document ingested before this feature existed. Deliberately NOT
# "PUBLIC": a document nobody has classified yet has not been explicitly
# cleared for external use, and INTERNAL is exactly the existing "usable
# locally, minimized externally" behavior every other retrieved chunk
# already gets from this same module's student-data half -- the same
# minimize-by-default posture, applied to institutional content instead of
# identity fields. Never RESTRICTED-by-default either: that would silently
# break every course's tutoring on this feature's rollout day for material
# an institution never asked to protect that strictly -- an explicit
# RESTRICTED classification is opt-in, not assumed.
DEFAULT_SENSITIVITY = "INTERNAL"

# Overall size budget (characters) for course material handed to an
# EXTERNAL LLM in one retrieval call -- configurable in one place rather
# than an implicit per-chunk `[:900]`/`[:3000]` slice scattered through the
# RAG pipeline. Applies to PUBLIC and INTERNAL content alike once
# `for_external=True` is requested; RESTRICTED content is excluded before
# this budget is ever considered. Chosen to comfortably fit a small number
# of the pipeline's own already-small per-chunk excerpts (900 chars each,
# see pipelines.rag_pipeline.retrieve_context) without materially changing
# what a normal turn could already retrieve.
MAX_EXTERNAL_RAG_CHARS = 4000


def normalize_sensitivity(raw: Any) -> str:
    """Coerce any stored/ingested sensitivity value to one of
    SENSITIVITY_LEVELS, defaulting to DEFAULT_SENSITIVITY for anything
    missing, blank, or unrecognized. This is the single place that
    decides what an unclassified document means -- both at ingestion time
    (pipelines.rag_pipeline.ingest_documents stamps this onto every chunk)
    and at read time (pipelines.rag_pipeline.retrieve_context reads it back
    off whatever Chroma returns), so a legacy chunk stored before this
    field existed is treated identically to one explicitly marked INTERNAL."""
    value = str(raw or "").strip().upper()
    return value if value in SENSITIVITY_LEVELS else DEFAULT_SENSITIVITY


def filter_course_chunks_for_external(
    chunks: list[dict[str, Any]], *, max_chars: int = MAX_EXTERNAL_RAG_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic institutional-content privacy gateway -- never an LLM
    decision (no chat-model/completion call happens here or is ever needed
    for this to work).

    `chunks`: already relevance-ranked (nearest-first), each a dict with at
    least `"text"` and `"sensitivity"` (a raw/possibly-missing value --
    normalized here, not by the caller). Never re-ranks or reorders them.

    Behavior:
    - RESTRICTED chunks are dropped entirely and never appear in the
      returned list -- not truncated, not summarized, simply excluded.
    - PUBLIC/INTERNAL chunks are kept, in order, until `max_chars` would be
      exceeded -- whole chunks only (never truncated mid-text); at least
      one chunk is always kept if any PUBLIC/INTERNAL chunk exists, even if
      it alone exceeds `max_chars`, so a single relevant excerpt is never
      reduced to nothing over a budget that exists to trim the LONG TAIL of
      extra chunks, not to block the single most relevant one.

    Returns (allowed_chunks, audit). `audit` contains ONLY counts/levels/
    sizes -- never chunk text, source paths, or any student field -- so it
    is always safe to log as-is (see log_external_content_decision).
    """
    allowed: list[dict[str, Any]] = []
    blocked_restricted = 0
    levels_seen: set[str] = set()
    for chunk in chunks:
        level = normalize_sensitivity(chunk.get("sensitivity"))
        levels_seen.add(level)
        if level == "RESTRICTED":
            blocked_restricted += 1
            continue
        allowed.append(chunk)

    kept: list[dict[str, Any]] = []
    used_chars = 0
    for chunk in allowed:
        text = str(chunk.get("text") or "")
        if kept and used_chars + len(text) > max_chars:
            break
        kept.append(chunk)
        used_chars += len(text)

    total = len(chunks)
    audit = {
        "chunks_considered": total,
        "chunks_allowed": len(kept),
        "chunks_blocked_restricted": blocked_restricted,
        "chunks_dropped_for_size": len(allowed) - len(kept),
        "sensitivity_levels_seen": sorted(levels_seen),
        "external_chars_sent": used_chars,
        "all_candidates_restricted": total > 0 and blocked_restricted == total,
    }
    return kept, audit


def merge_external_content_audit(total: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    """Accumulate one filter_course_chunks_for_external `audit` dict into a
    running total across multiple retrieval calls in the same turn (e.g.
    one call per requested concept) -- pure counter arithmetic, no content.
    Returns `total` (mutated in place) for convenient chaining."""
    if not part:
        return total
    for key in ("chunks_considered", "chunks_allowed", "chunks_blocked_restricted", "chunks_dropped_for_size", "external_chars_sent"):
        total[key] = total.get(key, 0) + part.get(key, 0)
    total["sensitivity_levels_seen"] = sorted(set(total.get("sensitivity_levels_seen") or []) | set(part.get("sensitivity_levels_seen") or []))
    # True only once every sub-call that actually had candidates was fully
    # restricted -- a turn spanning two concepts where only one is
    # restricted must NOT report the whole turn as blocked.
    considered = total.get("chunks_considered", 0)
    blocked = total.get("chunks_blocked_restricted", 0)
    total["all_candidates_restricted"] = considered > 0 and blocked == considered
    return total


def log_external_content_decision(*, course_id: Any, audit: dict[str, Any]) -> None:
    """Print one privacy-safe audit line for an institutional-content
    decision made for an external LLM call -- timestamp, course id,
    sensitivity levels involved, chunk counts, and character count only.

    Never logs: raw chunk/document text, source file paths, student name/
    email/Moodle id/db id, API keys, or the assembled external prompt. This
    is intentionally the ONLY thing STEP 4 logs about a turn's institutional
    content -- enough for an institution to answer "what categories and
    quantity of information left our local environment", without storing a
    second copy of the content itself anywhere.
    """
    print(
        "[ACRLA] external_content_privacy_decision "
        f"timestamp={time.time():.3f} "
        f"course_id={course_id if course_id is not None else 'unknown'} "
        f"sensitivity_levels_seen={audit.get('sensitivity_levels_seen')} "
        f"chunks_considered={audit.get('chunks_considered', 0)} "
        f"chunks_allowed_external={audit.get('chunks_allowed', 0)} "
        f"chunks_blocked_restricted={audit.get('chunks_blocked_restricted', 0)} "
        f"chunks_dropped_for_size={audit.get('chunks_dropped_for_size', 0)} "
        f"external_chars_sent={audit.get('external_chars_sent', 0)} "
        f"all_candidates_restricted={audit.get('all_candidates_restricted', False)}"
    )
