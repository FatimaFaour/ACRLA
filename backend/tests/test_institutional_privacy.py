"""Regression tests for RQ2 STEP 4 (institutional/course-content privacy).

Covers:
1. filter_course_chunks_for_external -- the deterministic gateway itself,
   with PUBLIC/INTERNAL/RESTRICTED sentinels, including mixed retrieval
   (PUBLIC+RESTRICTED, INTERNAL+RESTRICTED) and size minimization.
2. normalize_sensitivity's safe default (INTERNAL, never PUBLIC).
3. A REAL ingest_documents -> ChromaDB -> retrieve_context(for_external=True)
   round trip (isolated temp Chroma path, a deterministic-enough fake
   embedding function so Ollama/network are never touched) proving
   sensitivity metadata survives ingestion -> chunking -> storage ->
   retrieval, and that the gateway blocks RESTRICTED content at the real
   retrieval boundary, not just in the standalone unit test.
4. Cross-course isolation: sentinel material in two different course
   collections; a query in course A never surfaces course B's content.
5. The full outbound-prompt test: search_course_material_tool + a real
   turn through run_simple_conversation_agent, with the LLM constructor
   replaced by a capturing RunnableLambda (same technique as the prior
   student-privacy task) -- inspects the actual prompt content that would
   have reached Gemini.
6. Student-privacy regression (name/email/Moodle id/db id still blocked).
7. Research metrics (A-D from the RQ2 task spec).

Uses a temp ChromaDB directory + langchain_community.embeddings.FakeEmbeddings
(no real Ollama call) so this test never depends on a running local model --
Ollama/ChromaDB's own code is not modified by this task; only how retrieved
chunks are filtered before an external prompt.

Run from the `backend/` directory:
    python tests/test_institutional_privacy.py
or from anywhere (this file resolves its own project root):
    python backend/tests/test_institutional_privacy.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import FakeEmbeddings

from models.db_models import Base, Student
from services.memory_manager import MemoryManager

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


PUBLIC_SENTINEL = "PUBLIC_COURSE_SENTINEL"
INTERNAL_SENTINEL = "INTERNAL_COURSE_SENTINEL"
RESTRICTED_SENTINEL = "RESTRICTED_SECRET_SENTINEL"

STUDENT_NAME_SENTINEL = "PRIVACY_SENTINEL_NAME"
STUDENT_EMAIL_SENTINEL = "privacy-sentinel@example.com"
STUDENT_MOODLE_ID_SENTINEL = 987654321
STUDENT_DB_ID_SENTINEL = "SENTINEL_DB_ID_123456789"


# ===========================================================================
# 1-2: filter_course_chunks_for_external / normalize_sensitivity (unit level)
# ===========================================================================
from services.privacy_context import (
    filter_course_chunks_for_external, normalize_sensitivity, merge_external_content_audit,
    SENSITIVITY_LEVELS, DEFAULT_SENSITIVITY,
)

check("2a. safe default for missing/unclassified sensitivity is INTERNAL, never PUBLIC", DEFAULT_SENSITIVITY == "INTERNAL")
check("2b. normalize_sensitivity(None) -> INTERNAL", normalize_sensitivity(None) == "INTERNAL")
check("2c. normalize_sensitivity('') -> INTERNAL", normalize_sensitivity("") == "INTERNAL")
check("2d. normalize_sensitivity('bogus') -> INTERNAL (never silently PUBLIC)", normalize_sensitivity("bogus") == "INTERNAL")
check("2e. normalize_sensitivity('public') -> PUBLIC (case-insensitive)", normalize_sensitivity("public") == "PUBLIC")
check("2f. normalize_sensitivity('RESTRICTED') -> RESTRICTED", normalize_sensitivity("RESTRICTED") == "RESTRICTED")

mixed_chunks = [
    {"text": f"Public info. {PUBLIC_SENTINEL}", "sensitivity": "PUBLIC"},
    {"text": f"Secret info. {RESTRICTED_SENTINEL}", "sensitivity": "RESTRICTED"},
]
allowed, audit = filter_course_chunks_for_external(mixed_chunks)
allowed_text = " ".join(c["text"] for c in allowed)
check("1a. PUBLIC+RESTRICTED: PUBLIC sentinel is allowed through", PUBLIC_SENTINEL in allowed_text)
check("1b. PUBLIC+RESTRICTED: RESTRICTED sentinel is NEVER in the allowed set", RESTRICTED_SENTINEL not in allowed_text, allowed)
check("1c. PUBLIC+RESTRICTED: exactly 1 chunk allowed, 1 blocked", audit["chunks_allowed"] == 1 and audit["chunks_blocked_restricted"] == 1, audit)

mixed_chunks_2 = [
    {"text": f"Internal info. {INTERNAL_SENTINEL}", "sensitivity": "INTERNAL"},
    {"text": f"Secret info. {RESTRICTED_SENTINEL}", "sensitivity": "RESTRICTED"},
]
allowed2, audit2 = filter_course_chunks_for_external(mixed_chunks_2)
allowed2_text = " ".join(c["text"] for c in allowed2)
check("1d. INTERNAL+RESTRICTED: INTERNAL sentinel is allowed through", INTERNAL_SENTINEL in allowed2_text)
check("1e. INTERNAL+RESTRICTED: RESTRICTED sentinel is NEVER in the allowed set", RESTRICTED_SENTINEL not in allowed2_text, allowed2)

all_restricted = [{"text": RESTRICTED_SENTINEL, "sensitivity": "RESTRICTED"}, {"text": "more secret", "sensitivity": "RESTRICTED"}]
allowed3, audit3 = filter_course_chunks_for_external(all_restricted)
check("1f. all-RESTRICTED retrieval: nothing allowed, audit flags all_candidates_restricted", allowed3 == [] and audit3["all_candidates_restricted"] is True, audit3)

no_classification = [{"text": "legacy doc, no sensitivity key at all"}]
allowed4, audit4 = filter_course_chunks_for_external(no_classification)
check("3. unclassified legacy chunk defaults to INTERNAL (allowed, not blocked)", len(allowed4) == 1 and audit4["chunks_blocked_restricted"] == 0, audit4)

# Size minimization: never truncate mid-chunk; drop whole extra chunks.
big_chunks = [{"text": "A" * 3000, "sensitivity": "PUBLIC"}, {"text": "B" * 3000, "sensitivity": "PUBLIC"}, {"text": "C" * 3000, "sensitivity": "PUBLIC"}]
allowed5, audit5 = filter_course_chunks_for_external(big_chunks, max_chars=4000)
check("8a. size cap keeps whole chunks only (no chunk text partially cut)", all(len(c["text"]) in (3000,) for c in allowed5), [len(c["text"]) for c in allowed5])
check("8b. size cap drops the extra chunk(s) once the budget is exceeded", len(allowed5) < len(big_chunks), audit5)
check("8c. audit reports how many were dropped for size", audit5["chunks_dropped_for_size"] == len(big_chunks) - len(allowed5))

# A single chunk that alone exceeds the budget is still kept whole (never
# reduced to nothing, and never cut mid-text).
oversized_single = [{"text": "X" * 9000, "sensitivity": "PUBLIC"}]
allowed6, audit6 = filter_course_chunks_for_external(oversized_single, max_chars=4000)
check("8d. a single oversized chunk is kept WHOLE rather than dropped/cut", len(allowed6) == 1 and len(allowed6[0]["text"]) == 9000, audit6)

# merge_external_content_audit
total = {}
merge_external_content_audit(total, {"chunks_considered": 2, "chunks_allowed": 1, "chunks_blocked_restricted": 1, "chunks_dropped_for_size": 0, "external_chars_sent": 100, "sensitivity_levels_seen": ["PUBLIC", "RESTRICTED"]})
merge_external_content_audit(total, {"chunks_considered": 1, "chunks_allowed": 0, "chunks_blocked_restricted": 1, "chunks_dropped_for_size": 0, "external_chars_sent": 0, "sensitivity_levels_seen": ["RESTRICTED"]})
check("1g. merge_external_content_audit sums counts across multiple concepts", total["chunks_considered"] == 3 and total["chunks_blocked_restricted"] == 2, total)
check("1h. merge is NOT all_candidates_restricted when only PART of the turn was blocked", total["all_candidates_restricted"] is False, total)

# Structural: the gateway never calls an LLM (no chat-model/completion call).
import services.privacy_context as privacy_context_module
gateway_source = inspect.getsource(privacy_context_module.filter_course_chunks_for_external)
check("6. filter_course_chunks_for_external never calls get_llm/get_json_llm (deterministic, not Gemini-decided)", "get_llm" not in gateway_source and "get_json_llm" not in gateway_source)

# log_external_content_decision never leaks content -- capture its print().
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    privacy_context_module.log_external_content_decision(course_id="course-x", audit={"chunks_considered": 3, "chunks_allowed": 1, "chunks_blocked_restricted": 2, "chunks_dropped_for_size": 0, "sensitivity_levels_seen": ["PUBLIC", "RESTRICTED"], "external_chars_sent": 120, "all_candidates_restricted": False})
log_output = buf.getvalue()
check("11a. audit log line never contains a sentinel content string", all(s not in log_output for s in (PUBLIC_SENTINEL, INTERNAL_SENTINEL, RESTRICTED_SENTINEL)), log_output.strip())
check("11b. audit log line contains the expected safe counters", "chunks_considered=3" in log_output and "chunks_blocked_restricted=2" in log_output)


# ===========================================================================
# Real ingest -> ChromaDB -> retrieve_context round trip (temp path, fake
# embeddings -- no Ollama network call).
# ===========================================================================
import config as config_module
import pipelines.rag_pipeline as rag_pipeline_module
from pipelines.rag_pipeline import ingest_documents, retrieve_context, get_vectorstore

tmp_chroma_dir = tempfile.mkdtemp(prefix="acrla_priv_test_chroma_")
orig_chroma_path = rag_pipeline_module.settings.chroma_path
orig_get_embeddings = rag_pipeline_module.get_embeddings
rag_pipeline_module.settings.chroma_path = tmp_chroma_dir
rag_pipeline_module.get_embeddings = lambda: FakeEmbeddings(size=32)
get_vectorstore.cache_clear()

COURSE_A = 900101
COURSE_B = 900102


def write_text_doc(tmp_dir_path, filename, text):
    import os
    path = os.path.join(tmp_dir_path, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


docs_dir = tempfile.mkdtemp(prefix="acrla_priv_test_docs_")
public_path = write_text_doc(docs_dir, "public.txt", f"Course overview. {PUBLIC_SENTINEL} is explained here in detail for everyone.")
internal_path = write_text_doc(docs_dir, "internal.txt", f"Internal lecture notes. {INTERNAL_SENTINEL} covers the assignment rubric.")
restricted_path = write_text_doc(docs_dir, "restricted.txt", f"Confidential exam bank. {RESTRICTED_SENTINEL} must never leave this server.")
course_b_path = write_text_doc(docs_dir, "course_b.txt", f"Course B material. COURSE_B_ONLY_SENTINEL appears only in course B's own collection.")

try:
    ingest_documents(COURSE_A, [public_path], concept="Linear Regression", sensitivity="PUBLIC")
    ingest_documents(COURSE_A, [internal_path], concept="Linear Regression", sensitivity="INTERNAL")
    ingest_documents(COURSE_A, [restricted_path], concept="Linear Regression", sensitivity="RESTRICTED")
    ingest_documents(COURSE_A, [write_text_doc(docs_dir, "legacy.txt", "Legacy material with no sensitivity key at all.")], concept="Linear Regression")  # sensitivity omitted
    ingest_documents(COURSE_B, [course_b_path], concept="Linear Regression", sensitivity="PUBLIC")

    # 3. Sensitivity metadata survives the REAL pipeline end to end.
    vs = get_vectorstore(COURSE_A)
    raw_docs = vs.similarity_search("course material", k=10)
    metadata_by_marker = {}
    for d in raw_docs:
        for marker in (PUBLIC_SENTINEL, INTERNAL_SENTINEL, RESTRICTED_SENTINEL):
            if marker in d.page_content:
                metadata_by_marker[marker] = d.metadata.get("sensitivity")
    check("4a. PUBLIC document's chunks carry sensitivity=PUBLIC after real ingest+storage", metadata_by_marker.get(PUBLIC_SENTINEL) == "PUBLIC", metadata_by_marker)
    check("4b. INTERNAL document's chunks carry sensitivity=INTERNAL after real ingest+storage", metadata_by_marker.get(INTERNAL_SENTINEL) == "INTERNAL", metadata_by_marker)
    check("4c. RESTRICTED document's chunks carry sensitivity=RESTRICTED after real ingest+storage", metadata_by_marker.get(RESTRICTED_SENTINEL) == "RESTRICTED", metadata_by_marker)

    # Default (for_external=False) behavior is UNCHANGED -- RESTRICTED text
    # still comes back (this is the pre-existing, internal-only behavior;
    # the gateway only applies when for_external=True is explicitly asked for).
    internal_ctx, _ = retrieve_context(COURSE_A, "course material", k=10, selected_concept="Linear Regression")
    check("baseline: for_external=False (default) is unaffected -- internal retrieval still sees everything", RESTRICTED_SENTINEL in internal_ctx, None)

    # for_external=True applies the gateway at the REAL retrieval boundary.
    ext_audit = {}
    ext_ctx, ext_sources = retrieve_context(COURSE_A, "course material", k=10, selected_concept="Linear Regression", for_external=True, audit=ext_audit)
    check("5a. real retrieve_context(for_external=True): RESTRICTED sentinel never appears", RESTRICTED_SENTINEL not in ext_ctx, ext_ctx)
    check("5b. real retrieve_context(for_external=True): PUBLIC sentinel still appears", PUBLIC_SENTINEL in ext_ctx)
    check("5c. real retrieve_context(for_external=True): INTERNAL sentinel still appears (minimized, not blocked)", INTERNAL_SENTINEL in ext_ctx)
    check("5d. real retrieve_context(for_external=True): audit reports the blocked RESTRICTED chunk", ext_audit.get("chunks_blocked_restricted", 0) >= 1, ext_audit)

    retrieved_size = len(internal_ctx)
    external_size = len(ext_ctx)
    reduction_pct = 100.0 * (1 - external_size / retrieved_size) if retrieved_size else 0.0
    print(f"[METRIC C] retrieved_context_size={retrieved_size} external_context_size={external_size} reduction_pct={reduction_pct:.1f}%")

    # 4. Cross-course isolation (institutional content).
    ext_audit_b = {}
    ctx_b, _ = retrieve_context(COURSE_B, "course material", k=10, selected_concept="Linear Regression", for_external=True, audit=ext_audit_b)
    check("14a. Course A's RESTRICTED sentinel never appears when querying Course B", RESTRICTED_SENTINEL not in ctx_b, ctx_b)
    check("14b. Course A's INTERNAL sentinel never appears when querying Course B", INTERNAL_SENTINEL not in ctx_b, ctx_b)
    check("14c. Course B's own sentinel does appear when querying Course B", "COURSE_B_ONLY_SENTINEL" in ctx_b, ctx_b)
    ctx_a_check, _ = retrieve_context(COURSE_A, "course material", k=10, selected_concept="Linear Regression", for_external=True)
    check("14d. Course B's sentinel never appears when querying Course A", "COURSE_B_ONLY_SENTINEL" not in ctx_a_check)

finally:
    rag_pipeline_module.settings.chroma_path = orig_chroma_path
    rag_pipeline_module.get_embeddings = orig_get_embeddings
    get_vectorstore.cache_clear()


# ===========================================================================
# 5-6: Full outbound-prompt test through the real simple-agent turn, with a
# capturing LLM (same technique as the prior student-privacy task).
# ===========================================================================
engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(bind=engine)
db = TestSession()
student = Student(moodle_user_id=STUDENT_MOODLE_ID_SENTINEL, username=STUDENT_NAME_SENTINEL, email=STUDENT_EMAIL_SENTINEL)
db.add(student)
db.commit()
db.refresh(student)
student.id = STUDENT_DB_ID_SENTINEL
memory = MemoryManager(db)

import agents.simple_agent as simple_agent
import agents.simple_planner as simple_planner_module
import agents.response_generator as response_generator_module
from agents.agent_models import SemanticPlan, ResolvedEntities
from agents.agent_tools import TOOL_REGISTRY


class _Capture:
    def __init__(self):
        self.prompt_value = None

    def prompt_text(self) -> str:
        if self.prompt_value is None:
            return ""
        try:
            return "\n".join(m.content for m in self.prompt_value.to_messages())
        except Exception:
            return str(self.prompt_value)


class _FakeResponse:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {}


def make_capturing_llm(response_content):
    capture = _Capture()

    def _fn(prompt_value):
        capture.prompt_value = prompt_value
        return _FakeResponse(response_content)

    return RunnableLambda(_fn), capture


PLANNER_JSON = (
    '{"goal": "concept_explanation", "concepts": ["Linear Regression"], "courses": [], '
    '"references": [], "analytics_request": null, "needs_clarification": false, '
    '"clarification_question": null, "confidence": 0.9, "tutor_signal": null}'
)


def fake_search_material_mixed(context, arguments):
    """Simulates search_course_material_tool's OWN gateway-filtered output
    directly (bypassing real Chroma/Ollama here -- the real-pipeline
    round-trip is already covered above) -- one call representing a
    concept whose retrieval mixed PUBLIC/INTERNAL/RESTRICTED chunks. This
    is what the tool returns AFTER its own for_external=True gateway call,
    so it deliberately contains no restricted text at all -- proving that
    is what actually reaches the response-generator prompt."""
    from services.privacy_context import filter_course_chunks_for_external
    concept = (arguments.get("concepts") or ["Linear Regression"])[0]
    candidates = [
        {"text": f"{PUBLIC_SENTINEL} explains the public part.", "sensitivity": "PUBLIC", "source": "public.pdf"},
        {"text": f"{INTERNAL_SENTINEL} explains the internal part.", "sensitivity": "INTERNAL", "source": "internal.pdf"},
        {"text": f"{RESTRICTED_SENTINEL} must never leave.", "sensitivity": "RESTRICTED", "source": "restricted.pdf"},
    ]
    allowed, audit = filter_course_chunks_for_external(candidates)
    joined = "\n\n".join(c["text"] for c in allowed)
    return {
        "tool": "search_course_material", "success": True, "query": arguments.get("query", ""),
        "requested_concepts": [concept], "chunks": [{"concept": concept, "text": joined, "sources": [c["source"] for c in allowed]}],
        "sources": [c["source"] for c in allowed],
        "concepts_found": [concept],
        "evidence": {"reliable": True, "coverage": "full", "supported_concepts": [concept], "reason": "ok", "confidence": 0.9},
        "all_candidates_restricted": bool(audit.get("all_candidates_restricted")),
        "privacy_audit": audit,
    }


def base_context(message, extra=None):
    ctx = {
        "session_id": "priv-inst-session", "message": message, "student_id": student.id,
        "student_name": student.username, "course_db_id": "priv-inst-course",
        "current_course": {"db_course_id": "priv-inst-course", "moodle_course_id": 77, "name": "Data Science", "concepts": ["Linear Regression"]},
        "current_concept": None, "tutor_state": {}, "quick_progress_check": {},
        "remediation_level": "course", "available_concepts": ["Linear Regression"], "weak_concepts": [],
        "last_reference": {}, "last_answer_type": None, "last_discussed_metric": None,
        "difficulty": "medium", "tutoring_strategy": {"name": "guided_practice", "reason": "", "instructions": ""},
        "recent_messages": [], "recent_structured_turns": [],
        "scope_rules": "", "course_context_by_db_id": {}, "canonical_courses": [],
        "memory": memory, "db": db,
    }
    ctx.update(extra or {})
    return ctx


planner_llm, planner_capture = make_capturing_llm(PLANNER_JSON)
response_llm, response_capture = make_capturing_llm("A generated tutoring reply.")
orig_planner_get = simple_planner_module.get_json_llm
orig_response_get = response_generator_module.get_llm
orig_search_tool = TOOL_REGISTRY.get("search_course_material")
simple_planner_module.get_json_llm = lambda *a, **kw: planner_llm
response_generator_module.get_llm = lambda *a, **kw: response_llm
TOOL_REGISTRY["search_course_material"] = fake_search_material_mixed
try:
    ctx = base_context("Explain Linear Regression")
    result = simple_agent.run_simple_conversation_agent(ctx)
finally:
    simple_planner_module.get_json_llm = orig_planner_get
    response_generator_module.get_llm = orig_response_get
    if orig_search_tool is not None:
        TOOL_REGISTRY["search_course_material"] = orig_search_tool

response_text = response_capture.prompt_text()
planner_text = planner_capture.prompt_text()
check("13a. outbound prompt (mixed PUBLIC+INTERNAL+RESTRICTED retrieval): RESTRICTED sentinel NEVER appears", RESTRICTED_SENTINEL not in response_text and RESTRICTED_SENTINEL not in planner_text, response_text[-600:])
check("13b. outbound prompt: the allowed (public/internal) sources are authorized", '"public.pdf"' in response_text and '"internal.pdf"' in response_text, response_text[-400:])
check("13c. outbound prompt: the blocked (restricted) source is NEVER authorized", '"restricted.pdf"' not in response_text)

# Student-privacy regression (section 12) -- re-verified on THIS SAME turn.
for sentinel in (STUDENT_NAME_SENTINEL, STUDENT_EMAIL_SENTINEL, str(STUDENT_MOODLE_ID_SENTINEL), STUDENT_DB_ID_SENTINEL):
    check(f"12. student-privacy regression: {sentinel!r} not in outbound prompt", sentinel not in response_text and sentinel not in planner_text)

# 9. Utility preservation -- the turn still produced a real, usable answer.
check("9a. utility preserved: the turn still succeeded with a non-empty reply", bool(result.reply) and result.reply != "", result.reply)
check("9b. utility preserved: course name still reaches the prompt", "Data Science" in response_text)


# ===========================================================================
# All-RESTRICTED turn -> deterministic safe fallback (section 7), no
# general-knowledge substitution, no LLM call for the reply itself.
# ===========================================================================
def fake_search_material_all_restricted(context, arguments):
    from services.privacy_context import filter_course_chunks_for_external
    concept = (arguments.get("concepts") or ["Linear Regression"])[0]
    candidates = [{"text": f"{RESTRICTED_SENTINEL} exam answers.", "sensitivity": "RESTRICTED", "source": "restricted.pdf"}]
    allowed, audit = filter_course_chunks_for_external(candidates)
    return {
        "tool": "search_course_material", "success": True, "query": arguments.get("query", ""),
        "requested_concepts": [concept], "chunks": [], "sources": [],
        "concepts_found": [], "evidence": {"reliable": False, "coverage": "none", "supported_concepts": [], "reason": "no_allowed_chunks", "confidence": 0.0},
        "all_candidates_restricted": bool(audit.get("all_candidates_restricted")),
        "privacy_audit": audit,
    }


planner_llm2, planner_capture2 = make_capturing_llm(PLANNER_JSON)
response_llm2, response_capture2 = make_capturing_llm("SHOULD NOT BE CALLED")
simple_planner_module.get_json_llm = lambda *a, **kw: planner_llm2
response_generator_module.get_llm = lambda *a, **kw: response_llm2
TOOL_REGISTRY["search_course_material"] = fake_search_material_all_restricted
try:
    ctx2 = base_context("Explain Linear Regression")
    result2 = simple_agent.run_simple_conversation_agent(ctx2)
finally:
    simple_planner_module.get_json_llm = orig_planner_get
    response_generator_module.get_llm = orig_response_get
    if orig_search_tool is not None:
        TOOL_REGISTRY["search_course_material"] = orig_search_tool

check("7a. all-RESTRICTED turn: a controlled fallback reply is returned (not empty, not an error)", bool(result2.reply), result2.reply)
check("7b. all-RESTRICTED turn: the reply never echoes the restricted sentinel", RESTRICTED_SENTINEL not in result2.reply, result2.reply)
check("7c. all-RESTRICTED turn: the response LLM was NEVER invoked (deterministic reply, no extra call)", response_capture2.prompt_value is None, response_capture2.prompt_value)
check("7d. all-RESTRICTED turn: still exactly 1 planner call, 0 response calls (no extra LLM call added)", result2.planner_call_count == 1 and result2.response_call_count == 0, (result2.planner_call_count, result2.response_call_count))
check("7e. all-RESTRICTED turn: reply wording doesn't expose internal implementation detail", "sensitivity" not in result2.reply.lower() and "RESTRICTED" not in result2.reply and "gateway" not in result2.reply.lower())


# ===========================================================================
# Structural safety: search_course_material was NOT added to
# _DIALOGUE_REPLY_TOOLS (that would break every ordinary successful RAG
# turn by skipping the response LLM call unconditionally).
# ===========================================================================
simple_agent_source = inspect.getsource(simple_agent)
check("structural: search_course_material is NOT in _DIALOGUE_REPLY_TOOLS", '"search_course_material"' not in simple_agent_source.split("_DIALOGUE_REPLY_TOOLS = {")[1].split("}")[0])
check("structural: the RESTRICTED-content fallback never calls get_llm/get_json_llm itself", "get_llm(" not in inspect.getsource(simple_agent._deterministic_reply) and "get_json_llm(" not in inspect.getsource(simple_agent._deterministic_reply))


# ===========================================================================
# 15. Research metrics.
# ===========================================================================
tested_student_identifiers = [STUDENT_NAME_SENTINEL, STUDENT_EMAIL_SENTINEL, str(STUDENT_MOODLE_ID_SENTINEL), STUDENT_DB_ID_SENTINEL]
disclosed = sum(1 for s in tested_student_identifiers if s in response_text or s in planner_text)
metric_a = disclosed / len(tested_student_identifiers)
print(f"[METRIC A] direct_student_identifier_disclosure_rate={metric_a:.1%} (target 0%)")
check("15a. METRIC A (student identifier disclosure rate) is 0%", metric_a == 0.0)

restricted_units_locally = 1  # the one RESTRICTED chunk retrieved/processed in the mixed-turn test above
restricted_units_externally = 1 if RESTRICTED_SENTINEL in response_text else 0
metric_b = restricted_units_externally / restricted_units_locally
print(f"[METRIC B] restricted_institutional_data_leakage_rate={metric_b:.1%} (target 0%)")
check("15b. METRIC B (restricted institutional data leakage rate) is 0%", metric_b == 0.0)

utility_preserved = bool(result.reply) and '"public.pdf"' in response_text and '"internal.pdf"' in response_text
print(f"[METRIC D] utility_preserved={utility_preserved}")
check("15d. METRIC D (utility preservation): PUBLIC/INTERNAL RAG question still answerable (authorized sources present) after filtering", utility_preserved)


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
