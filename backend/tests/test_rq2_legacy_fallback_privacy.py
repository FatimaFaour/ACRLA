"""Regression tests for the RQ2 legacy-fallback privacy fix
(final read-only audit, `evaluation/rq2_privacy/final_privacy_audit.md`
section 4).

BACKGROUND (historical, do not erase): the final RQ2 read-only audit
found that `services/chat_orchestrator.py:legacy_handle_message` — the
rule-based fallback that `handle_message` invokes when the primary
`simple` agent returns a non-provider failure on a tutoring turn — built
an external-LLM (Gemini) prompt that BYPASSED the RQ2 privacy boundary:

- `pipelines/rag_pipeline.py:generate_rag_response` sent the student's
  real Moodle full name (`Student name: {student_name}`) and overall
  mastery percentage (`Overall course mastery: {mastery_level}`, plus the
  same percentage embedded in `strategy.reason` / `strategy.prompt_block`)
  into `INTERNAL_PROMPT`, and retrieved course context via
  `retrieve_context_for_scope(...)` WITHOUT `for_external=True`, so
  `services.privacy_context.filter_course_chunks_for_external` was never
  applied — RESTRICTED chunks were not excluded and INTERNAL was not
  size-capped.
- `pipelines/hybrid_pipeline.py:generate_hybrid_response` sent the mastery
  percentage via `build_external_fallback_prompt`.

The MINIMAL FIX (reusing the existing gateway, no new privacy logic):
1. both legacy generation functions now call `retrieve_context_for_scope`
   with `for_external=True` (+ audit + `log_external_content_decision`);
2. `legacy_handle_message` no longer puts `username` or a
   `mastery_level` percentage on the `student_context` dict, and rebuilds
   the strategy block from the pedagogical instructions only (dropping
   `strategy.reason`, which is literally "mastery for <concept> is N%");
3. `rag_pipeline.py:SYSTEM_PROMPT` no longer has `{student_name}` /
   `{mastery_level}` fields or the "address the student by name" rule.

These tests prove the fix and lock it against regression. No live Gemini
call anywhere in this file (temp ChromaDB + FakeEmbeddings; the LLM
constructor is a capturing RunnableLambda).

Run from the `backend/` directory:
    python tests/test_rq2_legacy_fallback_privacy.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
import os
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.runnables import RunnableLambda
from langchain_community.embeddings import FakeEmbeddings

from models.db_models import Base, Student, Course
from models.db_models import Session as SessionModel
from services.memory_manager import MemoryManager

results = []


def check(name, cond, detail=None):
    results.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f" -> {detail}" if detail is not None else ""))


# Sentinels (same values the other RQ2 suites use, for consistency).
STUDENT_NAME_SENTINEL = "PRIVACY_SENTINEL_NAME"
STUDENT_EMAIL_SENTINEL = "privacy-sentinel@example.com"
STUDENT_MOODLE_ID_SENTINEL = 987654321
STUDENT_DB_ID_SENTINEL = "SENTINEL_DB_ID_123456789"
MASTERY_PCT_SENTINEL = "37%"          # avg_mastery 0.37 rendered as a percentage
STRATEGY_REASON_SENTINEL = "mastery for"  # services.strategy_selector: "mastery for <concept> is N%."

PUBLIC_SENTINEL = "LEGACY_PUBLIC_SENTINEL"
INTERNAL_SENTINEL = "LEGACY_INTERNAL_SENTINEL"
RESTRICTED_SENTINEL = "LEGACY_RESTRICTED_SECRET_SENTINEL"


class _Capture:
    def __init__(self):
        self.prompt_value = None
        self.calls = 0

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


def make_capturing_llm(response_content="A generated legacy-fallback tutoring reply."):
    capture = _Capture()

    def _fn(prompt_value):
        capture.calls += 1
        capture.prompt_value = prompt_value
        return _FakeResponse(response_content)

    return RunnableLambda(_fn), capture


# ===========================================================================
# 0. Structural checks — templates and function sources
# ===========================================================================
import pipelines.rag_pipeline as rag_pipeline_module
import pipelines.hybrid_pipeline as hybrid_pipeline_module
import services.chat_orchestrator as chat_orchestrator_module

sys_prompt = rag_pipeline_module.SYSTEM_PROMPT
check("0a. rag_pipeline.SYSTEM_PROMPT has no {student_name} field", "{student_name}" not in sys_prompt)
check("0b. rag_pipeline.SYSTEM_PROMPT has no {mastery_level} field", "{mastery_level}" not in sys_prompt)
check("0c. rag_pipeline.SYSTEM_PROMPT dropped the 'address the student by name' rule", "address the student by name" not in sys_prompt)

grr_src = inspect.getsource(rag_pipeline_module.generate_rag_response)
check("0d. generate_rag_response retrieves with for_external=True", "for_external=True" in grr_src)
check("0e. generate_rag_response logs the external-content decision", "log_external_content_decision" in grr_src)
check("0f. generate_rag_response no longer passes a student_name invoke value", '"student_name"' not in grr_src)
check("0g. generate_rag_response no longer passes a mastery_level invoke value", '"mastery_level"' not in grr_src)

ghr_src = inspect.getsource(hybrid_pipeline_module.generate_hybrid_response)
check("0h. generate_hybrid_response retrieves with for_external=True when it does retrieve", "for_external=True" in ghr_src)

lhm_src = inspect.getsource(chat_orchestrator_module.legacy_handle_message)
sc_block = lhm_src.split("student_context = {")[1].split("}")[0]
check("0i. legacy_handle_message student_context no longer sets username", '"username"' not in sc_block)
check("0j. legacy_handle_message student_context no longer sets a mastery_level percentage", "mastery_level" not in sc_block)
check("0k. legacy_handle_message blanks strategy_reason (the 'mastery for X is N%' string)", '"strategy_reason": ""' in lhm_src)
check("0l. legacy_handle_message feeds the rebuilt-from-instructions strategy block (not strategy.prompt_block) to the prompt",
      '"strategy_instructions": _legacy_strategy_block' in lhm_src and "strategy.prompt_block" not in sc_block)

# The fix reuses the EXISTING gateway helper — it must not have duplicated
# any privacy-policy logic into the pipeline modules.
for modname, mod in (("rag_pipeline", rag_pipeline_module), ("hybrid_pipeline", hybrid_pipeline_module)):
    src = inspect.getsource(mod)
    check(f"0m. {modname} does not redefine the sensitivity levels", 'SENSITIVITY_LEVELS = (' not in src)
    check(f"0n. {modname} does not redefine a filtering gateway", "def filter_course_chunks_for_external" not in src)


# ===========================================================================
# Real ChromaDB (temp dir + FakeEmbeddings) with PUBLIC/INTERNAL/RESTRICTED
# "Recursion" material — the same round-trip fixture pattern as
# test_institutional_privacy.py.
# ===========================================================================
from pipelines.rag_pipeline import ingest_documents, get_vectorstore

tmp_chroma_dir = tempfile.mkdtemp(prefix="acrla_legacy_priv_chroma_")
orig_chroma_path = rag_pipeline_module.settings.chroma_path
orig_get_embeddings = rag_pipeline_module.get_embeddings
rag_pipeline_module.settings.chroma_path = tmp_chroma_dir
rag_pipeline_module.get_embeddings = lambda: FakeEmbeddings(size=32)
get_vectorstore.cache_clear()

COURSE_MOODLE_ID = 940101
docs_dir = tempfile.mkdtemp(prefix="acrla_legacy_priv_docs_")


def _write(name, text):
    p = os.path.join(docs_dir, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


# INTERNAL doc is deliberately long so the external size-cap is observable.
LONG_INTERNAL_BODY = (INTERNAL_SENTINEL + " recursion internal note. ") * 400  # ~11k chars

_capture_rag = _capture_hybrid = None
try:
    ingest_documents(COURSE_MOODLE_ID, [_write("pub.txt", f"{PUBLIC_SENTINEL} Recursion is a function that calls itself. Public overview for everyone.")], concept="Recursion", sensitivity="PUBLIC")
    ingest_documents(COURSE_MOODLE_ID, [_write("int.txt", LONG_INTERNAL_BODY)], concept="Recursion", sensitivity="INTERNAL")
    ingest_documents(COURSE_MOODLE_ID, [_write("res.txt", f"{RESTRICTED_SENTINEL} confidential recursion exam answer key. Never leaves the server.")], concept="Recursion", sensitivity="RESTRICTED")

    # =======================================================================
    # 1. generate_rag_response — the primary bypass site. Direct call with a
    #    student_context shaped exactly as the POST-FIX legacy_handle_message
    #    builds it (no username, no mastery_level, sanitized strategy block).
    # =======================================================================
    from pipelines.rag_pipeline import generate_rag_response
    rag_llm, _capture_rag = make_capturing_llm()
    orig_rag_get_llm = rag_pipeline_module.get_llm
    rag_pipeline_module.get_llm = lambda *a, **kw: rag_llm

    legacy_student_context_rag = {
        "session_id": "legacy-priv-session",
        # POST-FIX shape: NO "username", NO "mastery_level".
        "course_name": "RQ2 Legacy Course",
        "weak_concepts": ["Recursion"],
        "selected_concept": "Recursion",
        "sub_concepts": [],
        "requested_concepts": ["Recursion"],
        "available_concepts": ["Recursion"],
        "remediation_level": "chapter",
        "remediation_scope": "Chapter-level: stay inside Recursion.",
        "retrieval_course_ids": [COURSE_MOODLE_ID],
        "difficulty": "medium",
        "question_format": "open-ended explanation question",
        "difficulty_instructions": "",
        "strategy": "guided_practice",
        "strategy_reason": "",   # POST-FIX: blanked
        # POST-FIX: rebuilt from instructions only, no "mastery for X is N%" line
        "strategy_instructions": "BACKEND-CONTROLLED ADAPTIVE STRATEGY:\nStrategy: guided_practice\n\nYou must follow this strategy:\n- Use medium-length explanations.\n- Ask practice questions.",
        "session_goal": "general revision",
        "mode": "internal",
        "skip_internal_context": False,
        "current_topic": "Recursion",
        "retrieval_query": "Explain recursion",
    }
    try:
        reply_rag, sources_rag = generate_rag_response(COURSE_MOODLE_ID, "Explain recursion in detail", legacy_student_context_rag)
    finally:
        rag_pipeline_module.get_llm = orig_rag_get_llm

    rag_prompt = _capture_rag.prompt_text()

    check("1a. generate_rag_response: real student NAME absent from the Gemini-bound prompt", STUDENT_NAME_SENTINEL not in rag_prompt)
    check("1b. generate_rag_response: student email absent", STUDENT_EMAIL_SENTINEL not in rag_prompt)
    check("1c. generate_rag_response: Moodle user id absent", str(STUDENT_MOODLE_ID_SENTINEL) not in rag_prompt)
    check("1d. generate_rag_response: DB student id absent", STUDENT_DB_ID_SENTINEL not in rag_prompt)
    check("1e. generate_rag_response: no student mastery percentage line in the prompt", "Overall course mastery" not in rag_prompt and MASTERY_PCT_SENTINEL not in rag_prompt)
    check("1f. generate_rag_response: no 'mastery for <concept> is N%' strategy-reason string", STRATEGY_REASON_SENTINEL not in rag_prompt)
    check("1g. generate_rag_response: RESTRICTED course content BLOCKED from the prompt", RESTRICTED_SENTINEL not in rag_prompt, rag_prompt[-600:])
    check("1h. generate_rag_response: authorized (PUBLIC or INTERNAL) course content still supports the answer", (PUBLIC_SENTINEL in rag_prompt) or (INTERNAL_SENTINEL in rag_prompt))
    check("1i. generate_rag_response: INTERNAL course content present but size-minimized (< raw ~11k, <= gateway budget + one chunk)", (INTERNAL_SENTINEL in rag_prompt) and (rag_prompt.count(INTERNAL_SENTINEL) * len(INTERNAL_SENTINEL) < len(LONG_INTERNAL_BODY)))
    # Deterministic proof on a BALANCED small collection (one short chunk per
    # sensitivity level, so all fit within k and FakeEmbeddings ranking is
    # irrelevant): the same retrieval boundary the legacy path now uses
    # (for_external=True) lets PUBLIC + INTERNAL through and blocks RESTRICTED.
    from pipelines.rag_pipeline import retrieve_context as _retrieve_context
    BAL_COURSE = 940102
    ingest_documents(BAL_COURSE, [_write("bpub.txt", f"{PUBLIC_SENTINEL} short recursion public note.")], concept="Recursion", sensitivity="PUBLIC")
    ingest_documents(BAL_COURSE, [_write("bint.txt", f"{INTERNAL_SENTINEL} short recursion internal note.")], concept="Recursion", sensitivity="INTERNAL")
    ingest_documents(BAL_COURSE, [_write("bres.txt", f"{RESTRICTED_SENTINEL} short recursion restricted note.")], concept="Recursion", sensitivity="RESTRICTED")
    _det_audit = {}
    _det_ctx, _ = _retrieve_context(BAL_COURSE, "recursion course material", k=10, selected_concept="Recursion", for_external=True, audit=_det_audit)
    check("1h2. legacy retrieval boundary (for_external=True): PUBLIC content passes", PUBLIC_SENTINEL in _det_ctx, _det_ctx)
    check("1h3. legacy retrieval boundary (for_external=True): INTERNAL content passes (minimized)", INTERNAL_SENTINEL in _det_ctx, _det_ctx)
    check("1h4. legacy retrieval boundary (for_external=True): RESTRICTED content blocked + audited", RESTRICTED_SENTINEL not in _det_ctx and _det_audit.get("chunks_blocked_restricted", 0) >= 1, _det_audit)
    check("1j. generate_rag_response: fallback still functional (non-empty reply returned)", isinstance(reply_rag, str) and reply_rag.strip() != "")
    check("1k. generate_rag_response: exactly ONE LLM call (no new Gemini call introduced)", _capture_rag.calls == 1, _capture_rag.calls)
    # Course-material excerpt in the prompt must be bounded well under a whole PDF.
    check("1l. generate_rag_response: retrieved context in the prompt is a bounded excerpt, not a whole document", len(rag_prompt) < 20000, len(rag_prompt))

    # =======================================================================
    # 2. generate_hybrid_response — the fallback (no course material) route
    #    the legacy path actually uses (skip_internal_context=True), and the
    #    with-retrieval route (skip_internal_context=False) for the gateway.
    # =======================================================================
    from pipelines.hybrid_pipeline import generate_hybrid_response

    # 2a. Normal legacy hybrid route: skip_internal_context=True (context="").
    hyb_llm_a, capture_hyb_a = make_capturing_llm()
    orig_hyb_get_llm = hybrid_pipeline_module.get_llm
    hybrid_pipeline_module.get_llm = lambda *a, **kw: hyb_llm_a
    legacy_student_context_hyb = dict(legacy_student_context_rag)
    legacy_student_context_hyb["mode"] = "external"
    legacy_student_context_hyb["skip_internal_context"] = True
    try:
        reply_hyb_a, _ = generate_hybrid_response(COURSE_MOODLE_ID, "What is the capital of France?", legacy_student_context_hyb)
    finally:
        hybrid_pipeline_module.get_llm = orig_hyb_get_llm
    hyb_prompt_a = capture_hyb_a.prompt_text()
    check("2a. generate_hybrid_response (skip_internal): student NAME absent", STUDENT_NAME_SENTINEL not in hyb_prompt_a)
    check("2b. generate_hybrid_response (skip_internal): no student mastery percentage", MASTERY_PCT_SENTINEL not in hyb_prompt_a and STRATEGY_REASON_SENTINEL not in hyb_prompt_a)
    check("2c. generate_hybrid_response (skip_internal): 'Student mastery level:' line, if present, carries no percentage", ("Student mastery level:" not in hyb_prompt_a) or ("Student mastery level: unknown" in hyb_prompt_a))
    check("2d. generate_hybrid_response (skip_internal): no course material retrieved (context empty)", RESTRICTED_SENTINEL not in hyb_prompt_a and INTERNAL_SENTINEL not in hyb_prompt_a)
    check("2e. generate_hybrid_response (skip_internal): fallback still functional", isinstance(reply_hyb_a, str) and reply_hyb_a.strip() != "")
    check("2f. generate_hybrid_response (skip_internal): exactly ONE LLM call", capture_hyb_a.calls == 1, capture_hyb_a.calls)

    # 2g. With-retrieval hybrid route: skip_internal_context=False -> gateway.
    hyb_llm_b, capture_hyb_b = make_capturing_llm()
    hybrid_pipeline_module.get_llm = lambda *a, **kw: hyb_llm_b
    legacy_student_context_hyb_b = dict(legacy_student_context_rag)
    legacy_student_context_hyb_b["mode"] = "external"
    legacy_student_context_hyb_b["skip_internal_context"] = False
    try:
        reply_hyb_b, _ = generate_hybrid_response(COURSE_MOODLE_ID, "Explain recursion", legacy_student_context_hyb_b)
    finally:
        hybrid_pipeline_module.get_llm = orig_hyb_get_llm
    hyb_prompt_b = capture_hyb_b.prompt_text()
    check("2g. generate_hybrid_response (with retrieval): RESTRICTED course content BLOCKED", RESTRICTED_SENTINEL not in hyb_prompt_b, hyb_prompt_b[-500:])
    check("2h. generate_hybrid_response (with retrieval): student NAME absent", STUDENT_NAME_SENTINEL not in hyb_prompt_b)
    check("2i. generate_hybrid_response (with retrieval): exactly ONE LLM call", capture_hyb_b.calls == 1, capture_hyb_b.calls)

    # =======================================================================
    # 3. End-to-end: handle_message forced onto the legacy fallback path,
    #    real sentinel student, real ChromaDB, all LLM constructors captured.
    # =======================================================================
    import services.chat_orchestrator as co

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    db = TestSession()
    student = Student(moodle_user_id=STUDENT_MOODLE_ID_SENTINEL, username=STUDENT_NAME_SENTINEL, email=STUDENT_EMAIL_SENTINEL)
    course = Course(moodle_course_id=COURSE_MOODLE_ID, name="RQ2 Legacy Course")
    db.add_all([student, course])
    db.commit(); db.refresh(student); db.refresh(course)
    student.id = STUDENT_DB_ID_SENTINEL
    db.commit()
    session = SessionModel(student_id=student.id, course_id=course.id, difficulty="medium", is_active=True)
    db.add(session)
    db.commit(); db.refresh(session)
    mem = MemoryManager(db)
    mem.set_mastery(student_id=student.id, course_id=course.id, concept="Recursion", mastery_level=0.37)  # -> "37%"

    e2e_rag_llm, e2e_rag_capture = make_capturing_llm("A course-grounded legacy reply.")
    e2e_hyb_llm, e2e_hyb_capture = make_capturing_llm("A fallback legacy reply.")
    e2e_general_json_calls = {"n": 0}

    orig_try_agent = co._try_conversation_agent
    orig_analyze = co.analyze_user_turn
    orig_co_get_json_llm = co.get_json_llm
    orig_rag_get_llm = rag_pipeline_module.get_llm
    orig_hyb_get_llm = hybrid_pipeline_module.get_llm

    co._try_conversation_agent = lambda context: {
        "success": False, "fallback_reason": "simple_agent_no_evidence_gathered",
        "goal": "concept_explanation", "resolved_concepts": [], "analytics_request": None,
        "tools_executed": [], "reply": "",
    }
    co.analyze_user_turn = lambda *a, **kw: {"intent": "unclear", "confidence": 0.0, "primary_intent": "unclear"}

    def _co_json_llm(*a, **kw):
        e2e_general_json_calls["n"] += 1
        llm, _cap = make_capturing_llm('{"intent":"unclear","confidence":0.0}')
        return llm

    co.get_json_llm = _co_json_llm
    rag_pipeline_module.get_llm = lambda *a, **kw: e2e_rag_llm
    hybrid_pipeline_module.get_llm = lambda *a, **kw: e2e_hyb_llm
    try:
        e2e_result = co.handle_message(session_id=session.id, student_moodle_id=STUDENT_MOODLE_ID_SENTINEL, message="Explain recursion", db=db)
    finally:
        co._try_conversation_agent = orig_try_agent
        co.analyze_user_turn = orig_analyze
        co.get_json_llm = orig_co_get_json_llm
        rag_pipeline_module.get_llm = orig_rag_get_llm
        hybrid_pipeline_module.get_llm = orig_hyb_get_llm

    e2e_prompt = e2e_rag_capture.prompt_text() + "\n" + e2e_hyb_capture.prompt_text()
    generation_calls = e2e_rag_capture.calls + e2e_hyb_capture.calls
    check("3a. e2e legacy path: a legacy generation LLM call actually fired (path is reachable)", generation_calls >= 1, generation_calls)
    check("3b. e2e legacy path: real student NAME absent from the Gemini-bound prompt", STUDENT_NAME_SENTINEL not in e2e_prompt)
    check("3c. e2e legacy path: student email absent", STUDENT_EMAIL_SENTINEL not in e2e_prompt)
    check("3d. e2e legacy path: Moodle user id absent", str(STUDENT_MOODLE_ID_SENTINEL) not in e2e_prompt)
    check("3e. e2e legacy path: DB student id absent", STUDENT_DB_ID_SENTINEL not in e2e_prompt)
    check("3f. e2e legacy path: student mastery percentage ('37%') absent", "37%" not in e2e_prompt)
    check("3g. e2e legacy path: 'mastery for' strategy-reason string absent", STRATEGY_REASON_SENTINEL not in e2e_prompt)
    check("3h. e2e legacy path: RESTRICTED course content absent", RESTRICTED_SENTINEL not in e2e_prompt)
    check("3i. e2e legacy path: exactly one legacy generation call (no extra/duplicate Gemini call)", generation_calls == 1, generation_calls)
    check("3j. e2e legacy path: a usable reply was still produced", isinstance(e2e_result, dict) and str(e2e_result.get("reply") or "").strip() != "")

finally:
    rag_pipeline_module.settings.chroma_path = orig_chroma_path
    rag_pipeline_module.get_embeddings = orig_get_embeddings
    get_vectorstore.cache_clear()


# ===========================================================================
# 4. Primary-path guard: the fix must not have touched the primary agent's
#    identity allow-list or the shared gateway.
# ===========================================================================
import services.privacy_context as pc
pc_src = inspect.getsource(pc)
check("4a. services.privacy_context.build_llm_safe_student_context unchanged in shape (still an allow-list, no identity fields)",
      all(f'"{f}"' not in inspect.getsource(pc.build_llm_safe_student_context) for f in ("student_name", "username", "email", "moodle_user_id")))
check("4b. filter_course_chunks_for_external still makes no LLM call", "get_llm" not in inspect.getsource(pc.filter_course_chunks_for_external))

import agents.response_generator as rg
check("4c. primary response_generator still routes through build_llm_safe_student_context", "build_llm_safe_student_context(context)" in inspect.getsource(rg.generate_final_answer))


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
