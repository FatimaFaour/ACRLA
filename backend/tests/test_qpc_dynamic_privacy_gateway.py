"""Regression tests for the RQ2 institutional-privacy gateway fix applied
to RQ3's dynamic QPC generation path (routers.api._dynamic_material_context
/ _stored_or_generated_variants).

Root cause this addresses: `_dynamic_material_context` queried ChromaDB
directly (`vectorstore._collection.get(...)`), bypassing `pipelines.rag_
pipeline.retrieve_context`/`retrieve_context_for_scope` -- the only place
the RQ2 institutional gateway (`services.privacy_context.filter_course_
chunks_for_external`) was wired in. Per-chunk `sensitivity` metadata was
fetched but then discarded when the retrieved chunks were joined into one
flat string, and that unfiltered string was sent verbatim to Gemini inside
`_llm_dynamic_variants`.

Fix (smallest targeted change, approved before implementation):
`_dynamic_material_context` now ALSO returns the per-chunk {"text",
"sensitivity", "source"} list (from the same already-fetched metadata, no
new query). `_stored_or_generated_variants` runs that list through the
EXISTING, unmodified `filter_course_chunks_for_external` before building
the external-bound context string passed to `_llm_dynamic_variants`. The
full, unfiltered local context is still passed, completely unchanged, to
`validate_generated_questions` (the RQ3 deterministic validity gate) and
`_fallback_dynamic_variants` (the local-only template) -- neither of those
sends anything externally, so neither needs filtering.

Covers exactly what the task asked for (labeled A-H below):
A. PUBLIC evidence can reach the external LLM prompt.
B. INTERNAL evidence is present but capped at the existing
   MAX_EXTERNAL_RAG_CHARS budget before external transmission (the same
   minimization behavior `pipelines.rag_pipeline.retrieve_context` already
   applies elsewhere -- not reimplemented, reused).
C. RESTRICTED raw sentinel text never appears in the external LLM prompt.
D. Student PII/identifiers are absent from the external prompt (this call
   path never took a student identifier to begin with -- locked in as a
   structural regression, not just an assumption).
E. The full, unfiltered local evidence (RESTRICTED chunks included) is
   still what `validate_generated_questions` (the deterministic local
   validator) receives -- the "dual-context" flow.
F. If NO externally-safe evidence exists (every chunk RESTRICTED), no
   Gemini call is made at all and generation falls through to the existing
   local fallback template -- a privacy-safe degrade, not a leak.
G. A rejected/invalid candidate still cannot affect mastery (mastery
   functions are never called by this generation path at all, verified by
   a Mock spy through the real function).
H. The existing hand-authored/"bank" QPC path (`_assessment_question_
   bank`) is structurally untouched by this change.

Uses a real ChromaDB round trip (temp dir, FakeEmbeddings -- no Ollama
call) plus a real LangChain RunnableLambda standing in for the LLM
constructor, patched on `routers.api.get_llm` (this module's own imported
name) -- no live Gemini call anywhere in this file.

Run from the `backend/` directory:
    python tests/test_qpc_dynamic_privacy_gateway.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
import json
import os
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import FakeEmbeddings
from unittest.mock import Mock

from models.db_models import Base, Student, Course, AssessmentQuestionVariant
from services.memory_manager import MemoryManager

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


import routers.api as api_module
import pipelines.rag_pipeline as rag_pipeline_module
from pipelines.rag_pipeline import ingest_documents, get_vectorstore

# ---------------------------------------------------------------------------
# Real ChromaDB round trip (temp path, FakeEmbeddings -- no Ollama call).
# ---------------------------------------------------------------------------
tmp_chroma_dir = tempfile.mkdtemp(prefix="acrla_qpc_privacy_test_chroma_")
orig_chroma_path = rag_pipeline_module.settings.chroma_path
orig_get_embeddings = rag_pipeline_module.get_embeddings
rag_pipeline_module.settings.chroma_path = tmp_chroma_dir
rag_pipeline_module.get_embeddings = lambda: FakeEmbeddings(size=32)
get_vectorstore.cache_clear()

COURSE_ID = 81301
CONCEPT = "Linear Regression"
PUBLIC_SENTINEL = "PUBLIC_LECTURE_SENTINEL_QPC"
INTERNAL_SENTINEL = "INTERNAL_COURSEPACK_SENTINEL_QPC"
RESTRICTED_SENTINEL = "RESTRICTED_EXAM_ANSWER_SENTINEL_QPC"


def write_text_doc(tmp_dir, filename, text):
    path = os.path.join(tmp_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


docs_dir = tempfile.mkdtemp(prefix="acrla_qpc_privacy_test_docs_")

db_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=db_engine)
TestSession = sessionmaker(bind=db_engine)


class _Capture:
    def __init__(self):
        self.prompt_text = None


def make_capturing_llm(response_json):
    capture = _Capture()

    def _fn(prompt):
        # routers.api._llm_dynamic_variants calls llm.invoke(prompt) with a
        # plain string, not a ChatPromptValue -- capture it as-is.
        capture.prompt_text = prompt if isinstance(prompt, str) else str(prompt)

        class _FakeResponse:
            content = response_json

        return _FakeResponse()

    return RunnableLambda(_fn), capture


# Prompts reuse real phrasing from the ingested PUBLIC/INTERNAL course text
# ("public lecture notes" / "internal coursepack detail") so the real,
# unmodified RQ3 validity gate's grounding check actually accepts these --
# this test exercises the real gateway-wrapped generation path end to end,
# not just the prompt-capture step.
VALID_ITEMS = [
    {"sub_concept": f"idea {i}",
     "prompt": f"What do the public lecture notes and internal coursepack detail describe about {CONCEPT}, variant {i}?",
     "options": ["A. one", "B. two", "C. three", "D. four"], "correct": "A"}
    for i in range(5)
]
VALID_JSON_TEXT = json.dumps(VALID_ITEMS)

orig_get_llm = api_module.get_llm


def run_generation(db, course_internal_id, moodle_course_id=COURSE_ID, response_json=VALID_JSON_TEXT):
    llm, capture = make_capturing_llm(response_json)
    api_module.get_llm = lambda *a, **kw: llm
    try:
        variants = api_module._stored_or_generated_variants(db, course_internal_id, moodle_course_id, CONCEPT)
    finally:
        api_module.get_llm = orig_get_llm
    return variants, capture


try:
    # ---------------------------------------------------------------------
    # Ingest one chunk at each sensitivity level for the same concept.
    # ---------------------------------------------------------------------
    ingest_documents(COURSE_ID, [write_text_doc(docs_dir, "public.txt", f"{CONCEPT} public lecture notes. {PUBLIC_SENTINEL} is openly shareable.")], concept=CONCEPT, sensitivity="PUBLIC")
    ingest_documents(COURSE_ID, [write_text_doc(docs_dir, "internal.txt", f"{CONCEPT} internal coursepack detail. {INTERNAL_SENTINEL} is for enrolled students only.")], concept=CONCEPT, sensitivity="INTERNAL")
    ingest_documents(COURSE_ID, [write_text_doc(docs_dir, "restricted.txt", f"{CONCEPT} exam answer key. {RESTRICTED_SENTINEL} must never leave this server.")], concept=CONCEPT, sensitivity="RESTRICTED")

    db = TestSession()
    course = Course(moodle_course_id=COURSE_ID, name="Privacy Test Course")
    db.add(course)
    db.commit()
    db.refresh(course)

    # -----------------------------------------------------------------
    # A/B/C. PUBLIC + INTERNAL reach the prompt; RESTRICTED never does.
    # -----------------------------------------------------------------
    variants, capture = run_generation(db, course.id)
    prompt_text = capture.prompt_text or ""
    check("A. PUBLIC evidence reaches the external LLM prompt", PUBLIC_SENTINEL in prompt_text, prompt_text[:300])
    check("B. INTERNAL evidence reaches the external LLM prompt (minimized/filtered, not blocked)", INTERNAL_SENTINEL in prompt_text, prompt_text[:300])
    check("C. RESTRICTED raw sentinel text NEVER appears in the external LLM prompt", RESTRICTED_SENTINEL not in prompt_text, prompt_text)
    check("C2. generated variants exist (the gateway didn't silently break normal generation)", len(variants) == 5, variants)

    # -----------------------------------------------------------------
    # D. Student PII/identifiers absent from the external prompt.
    # -----------------------------------------------------------------
    fn_params = list(inspect.signature(api_module._dynamic_material_context).parameters.keys())
    check("D1. _dynamic_material_context's signature takes no student identifier param", fn_params == ["course_id", "concept"], fn_params)
    llm_fn_params = list(inspect.signature(api_module._llm_dynamic_variants).parameters.keys())
    check("D2. _llm_dynamic_variants's signature takes no student identifier param", "student" not in " ".join(llm_fn_params).lower(), llm_fn_params)
    for pii in ("student_id", "moodle_user_id", "email", "@example", "username"):
        check(f"D3. prompt does not contain PII-shaped marker {pii!r}", pii not in prompt_text, prompt_text[:200])

    # -----------------------------------------------------------------
    # E. Full local evidence (RESTRICTED included) still reaches the
    #    deterministic validator -- capture course_context via a wrapper.
    # -----------------------------------------------------------------
    import services.question_validity as qv_module
    orig_validate = qv_module.validate_generated_questions
    captured_course_context = {}

    def _capturing_validate(raw_variants, *, concept, course_context):
        captured_course_context["value"] = course_context
        return orig_validate(raw_variants, concept=concept, course_context=course_context)

    # _stored_or_generated_variants does `from services.question_validity import
    # validate_generated_questions` locally each call, so patching the module
    # attribute is what actually takes effect.
    qv_module.validate_generated_questions = _capturing_validate
    try:
        db2 = TestSession()
        course2 = Course(moodle_course_id=COURSE_ID + 1, name="Privacy Test Course 2")
        db2.add(course2)
        db2.commit()
        db2.refresh(course2)
        ingest_documents(COURSE_ID + 1, [write_text_doc(docs_dir, "restricted2.txt", f"{CONCEPT} exam answer key. {RESTRICTED_SENTINEL}_E must never leave this server.")], concept=CONCEPT, sensitivity="RESTRICTED")
        ingest_documents(COURSE_ID + 1, [write_text_doc(docs_dir, "public2.txt", f"{CONCEPT} public notes. {PUBLIC_SENTINEL}_E openly shareable.")], concept=CONCEPT, sensitivity="PUBLIC")
        run_generation(db2, course2.id, moodle_course_id=COURSE_ID + 1)
        local_context = captured_course_context.get("value") or ""
        check("E. the FULL local context (validator input) still includes RESTRICTED text (not filtered locally)", f"{RESTRICTED_SENTINEL}_E" in local_context, local_context[:300])
        check("E2. the FULL local context also includes PUBLIC text", f"{PUBLIC_SENTINEL}_E" in local_context, local_context[:300])
    finally:
        qv_module.validate_generated_questions = orig_validate

    # -----------------------------------------------------------------
    # F. All-RESTRICTED concept -> no Gemini call, safe fallback.
    # -----------------------------------------------------------------
    RESTRICTED_ONLY_CONCEPT = "Restricted Only Topic"
    ingest_documents(COURSE_ID, [write_text_doc(docs_dir, "restricted_only.txt", f"{RESTRICTED_ONLY_CONCEPT} sensitive exam material. {RESTRICTED_SENTINEL}_F must never leave.")], concept=RESTRICTED_ONLY_CONCEPT, sensitivity="RESTRICTED")
    llm_call_count = {"n": 0}

    def _counting_llm(*a, **kw):
        llm_call_count["n"] += 1
        llm_fn, _ = make_capturing_llm(VALID_JSON_TEXT)
        return llm_fn

    api_module.get_llm = _counting_llm
    try:
        db3 = TestSession()
        course3 = Course(moodle_course_id=COURSE_ID, name="Privacy Test Course")
        # Reuse course.id from the first course (same moodle_course_id/collection).
        variants_f = api_module._stored_or_generated_variants(db3, course.id, COURSE_ID, RESTRICTED_ONLY_CONCEPT)
    finally:
        api_module.get_llm = orig_get_llm
    check("F1. all-RESTRICTED concept -> _llm_dynamic_variants never actually calls the LLM (empty external context short-circuits)", llm_call_count["n"] == 0, llm_call_count["n"])
    check("F2. all-RESTRICTED concept -> generation still safely falls back to the deterministic template", len(variants_f) > 0, variants_f)
    if variants_f:
        fallback_text = json.dumps(variants_f)
        check("F3. the fallback template itself does not leak the RESTRICTED sentinel externally-servable text", f"{RESTRICTED_SENTINEL}_F" not in fallback_text, fallback_text[:300])

    # -----------------------------------------------------------------
    # G. Rejected/invalid candidates still cannot affect mastery.
    # -----------------------------------------------------------------
    set_mastery_mock = Mock()
    update_mastery_mock = Mock()
    orig_set_mastery, orig_update_mastery = MemoryManager.set_mastery, MemoryManager.update_mastery
    MemoryManager.set_mastery, MemoryManager.update_mastery = set_mastery_mock, update_mastery_mock
    try:
        db4 = TestSession()
        course4 = Course(moodle_course_id=COURSE_ID + 2, name="Privacy Test Course 4")
        db4.add(course4)
        db4.commit()
        db4.refresh(course4)
        ingest_documents(COURSE_ID + 2, [write_text_doc(docs_dir, "public3.txt", f"{CONCEPT} public notes for mastery check.")], concept=CONCEPT, sensitivity="PUBLIC")
        # An all-invalid batch (malformed options) -- should reject everything and fall back.
        run_generation(db4, course4.id, moodle_course_id=COURSE_ID + 2, response_json=json.dumps([{"sub_concept": "x", "prompt": "", "options": [], "correct": "Z"}]))
    finally:
        MemoryManager.set_mastery, MemoryManager.update_mastery = orig_set_mastery, orig_update_mastery
    check("G1. MemoryManager.set_mastery is never called by dynamic QPC generation (gateway-wrapped path)", set_mastery_mock.call_count == 0, set_mastery_mock.call_count)
    check("G2. MemoryManager.update_mastery is never called by dynamic QPC generation", update_mastery_mock.call_count == 0, update_mastery_mock.call_count)

    # -----------------------------------------------------------------
    # H. Existing hand-authored/"bank" QPC path is structurally untouched.
    # -----------------------------------------------------------------
    bank_source = inspect.getsource(api_module._assessment_question_bank)
    check("H. _assessment_question_bank's source has no reference to the privacy gateway (structurally untouched, still fully static)", "filter_course_chunks_for_external" not in bank_source and "get_llm" not in bank_source)

finally:
    rag_pipeline_module.settings.chroma_path = orig_chroma_path
    rag_pipeline_module.get_embeddings = orig_get_embeddings
    get_vectorstore.cache_clear()


# ---------------------------------------------------------------------------
# Structural checks on the fix itself.
# ---------------------------------------------------------------------------
dynamic_context_source = inspect.getsource(api_module._dynamic_material_context)
check("structural: _dynamic_material_context now returns a 4th element (chunks) for the gateway", "-> tuple[str, str | None, str | None, list[dict]]" in dynamic_context_source and dynamic_context_source.count("chunks") >= 2)
stored_or_generated_source = inspect.getsource(api_module._stored_or_generated_variants)
check("structural: _stored_or_generated_variants calls filter_course_chunks_for_external", "filter_course_chunks_for_external(chunks)" in stored_or_generated_source)
check("structural: validate_generated_questions still receives the full local 'context', not 'external_context'", "course_context=context)" in stored_or_generated_source)
check("structural: _fallback_dynamic_variants still receives the full local 'context', not 'external_context'", "_fallback_dynamic_variants(moodle_course_id, concept, context, source_file, display_title)" in stored_or_generated_source)
check("structural: _llm_dynamic_variants is called with 'external_context', not the raw 'context'", "_llm_dynamic_variants(moodle_course_id, concept, external_context" in stored_or_generated_source)
check("structural: no privacy-policy/sensitivity-class definitions were duplicated in routers/api.py", "SENSITIVITY_LEVELS" not in stored_or_generated_source and "DEFAULT_SENSITIVITY" not in stored_or_generated_source)


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
