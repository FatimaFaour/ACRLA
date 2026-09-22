"""Deterministic validity gate for LLM-generated assessment (QPC)
questions -- runs once, at generation time, before a dynamically-generated
question variant is cached in `AssessmentQuestionVariant` and served to any
student.

RQ3 finding this addresses: `routers/api.py:_llm_dynamic_variants` only
checked that Gemini's output PARSED into the right JSON shape (exactly 4
options, `correct` in A-D) -- never whether the question was actually about
the right concept, grounded in the retrieved course material, unambiguous,
or correctly answered. Once persisted, a bad question was served to every
future student indefinitely (see
`evaluation/rq3_accuracy/technical_audit.md`, section G, and
`evaluation/rq3_accuracy/qpc_question_validity.md`).

Deliberately deterministic, not an LLM judge -- see
`evaluation/rq3_accuracy/qpc_question_validity.md` for the full
justification. In short: QPC questions are generated once per concept and
cached forever (a low-frequency, easily auditable operation, unlike a
per-turn tutoring call), and this project's established discipline around
this exact mastery-writing pipeline favors not adding another LLM call to
it without first showing a deterministic check cannot do the job. These
checks catch the clearest failure modes (wrong concept, no grounding in the
supplied material, duplicate/ambiguous options, and a "correct" answer the
material supports LESS than a distractor does) at zero added cost, latency,
or nondeterminism. They cannot catch a subtly-wrong but plausible-sounding,
well-grounded-vocabulary claim -- documented as a remaining limitation, not
silently assumed away.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from tools.text_utils import routing_tokens

# Both thresholds are intentionally low bars -- a genuine, on-topic question
# generated "from this course material" (see _llm_dynamic_variants' own
# prompt) should clear them easily; they exist to catch content with
# essentially no real connection to the concept/material, not to police
# phrasing style.
MIN_GROUNDING_OVERLAP_TOKENS = 2


class ValidationResult(NamedTuple):
    passed: bool
    reason: str


def _option_text(option: str) -> str:
    """Strip a leading 'A. '/'B) '/'C - ' marker, if present."""
    text = str(option or "").strip()
    if len(text) >= 2 and text[0] in "ABCD" and text[1] in ".)-:":
        return text[2:].strip()
    return text


def validate_generated_question(
    item: dict[str, Any], *, concept: str, course_context: str,
) -> ValidationResult:
    """One candidate question, one verdict. Structural checks (non-empty
    prompt, exactly 4 options, `correct` in A-D) are NOT duplicated here --
    `_llm_dynamic_variants` already enforces that shape before a candidate
    ever reaches this function; this only adds checks that function does
    not perform. Still defensively re-verified below (cheap, no assumption
    an un-vetted `item` can't arrive some other way in the future) but not
    treated as this function's main contribution.
    """
    prompt_text = str(item.get("prompt") or "").strip()
    options = item.get("options") or []
    correct_letter = str(item.get("correct") or "").strip().upper()[:1]

    if not prompt_text or len(options) != 4 or correct_letter not in {"A", "B", "C", "D"}:
        return ValidationResult(False, "structurally_invalid")

    option_texts = [_option_text(opt) for opt in options]
    if any(not text for text in option_texts):
        return ValidationResult(False, "empty_option_text")

    # (4) exactly one defensible correct answer / (6) distractors must be
    # genuinely distinct: two options saying the same thing make the
    # "correct" one indefensible -- a student could correctly pick either.
    normalized_options = [text.strip().lower() for text in option_texts]
    if len(set(normalized_options)) < len(normalized_options):
        return ValidationResult(False, "duplicate_or_ambiguous_options")

    # (1) evaluates the intended concept: the question must at least share
    # vocabulary with the concept name/sub_concept it claims to be about --
    # the same "at least one shared routing token" bar
    # agents.evidence_validator._deterministic_evidence_check already uses
    # for concept-tag matching, reused here rather than reinvented.
    concept_tokens = routing_tokens(concept)
    question_tokens = routing_tokens(prompt_text + " " + str(item.get("sub_concept") or ""))
    if concept_tokens and not (concept_tokens & question_tokens):
        return ValidationResult(False, "concept_mismatch")

    # (2) supported by the relevant course material / (8) does not require
    # knowledge outside the provided context: the question STEM must share
    # real vocabulary with the retrieved material -- a question generated
    # from essentially no grounding must not be trusted merely because
    # Gemini produced well-formed JSON.
    context_tokens = routing_tokens(course_context)
    if not context_tokens:
        # No course context was available to generate FROM at all -- there
        # is nothing to verify against. (In practice _llm_dynamic_variants
        # already refuses to call the LLM when context is empty, so this is
        # defense-in-depth, not the common path.)
        return ValidationResult(False, "no_course_context_to_ground_against")

    # Concept-name tokens are excluded from the overlap counts below --
    # otherwise a question could satisfy "grounded in the material" merely
    # by repeating the concept's own name (which the material naturally
    # also contains), without borrowing any ADDITIONAL real content from
    # it. This keeps this check meaningfully distinct from the concept-
    # match check above rather than re-testing the same thing.
    prompt_overlap = (question_tokens - concept_tokens) & context_tokens
    if len(prompt_overlap) < MIN_GROUNDING_OVERLAP_TOKENS:
        return ValidationResult(False, "insufficient_grounding_in_course_material")

    # (5) the indicated correct answer is actually correct, as far as this
    # is deterministically verifiable: the option Gemini marked correct
    # must be supported by the material AT LEAST as well as every
    # distractor. If a WRONG option overlaps the retrieved material MORE
    # than the marked-correct one does, that is a concrete, checkable
    # signal the answer key itself may be wrong or mislabeled -- not proof
    # the question is merely hard. Same concept-name exclusion as above.
    correct_index = "ABCD".index(correct_letter)
    option_overlaps = [len((routing_tokens(text) - concept_tokens) & context_tokens) for text in option_texts]
    correct_overlap = option_overlaps[correct_index]
    if any(overlap > correct_overlap for i, overlap in enumerate(option_overlaps) if i != correct_index):
        return ValidationResult(False, "indicated_answer_less_grounded_than_a_distractor")

    return ValidationResult(True, "passed_deterministic_checks")


def validate_generated_questions(
    items: list[dict[str, Any]], *, concept: str, course_context: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter a batch of candidate generated questions.

    Returns (accepted, rejected_log_entries). `rejected_log_entries` are
    privacy-safe summaries (question/sub-concept identifiers and the
    rejection reason only -- no student data, no full option text) meant to
    be logged for RQ3 research observability (see
    `routers.api._stored_or_generated_variants`'s own log line), never
    stored as a reusable question.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in items:
        result = validate_generated_question(item, concept=concept, course_context=course_context)
        if result.passed:
            accepted.append(item)
        else:
            rejected.append({
                "question_id": item.get("question_id"),
                "sub_concept": item.get("sub_concept"),
                "reason": result.reason,
            })
    return accepted, rejected
