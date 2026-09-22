"""Regression tests for RQ3 guided-practice answer evaluation grounding
(services.error_analyzer.judge_answer's new `course_evidence` parameter,
its wiring into tools.tutor_state_tools.evaluate_practice_answer_tool, and
the low-confidence feedback-wording change).

Root cause this addresses: judge_answer received concept + question +
student answer, but for MODERATE/HARD (open-ended) practice questions --
which have no stored expected answer -- it never received any course
material either, so a correctness judgment rested entirely on the model's
pretrained knowledge of the concept name. EASY questions already had a
deterministic expected-answer hint injected (unchanged by this step).

Covers exactly what the task asked for (labeled A-M below):
A. EASY deterministic correct answer -> correct (hint reaches the judge,
   no RAG retrieval attempted)
B. EASY deterministic wrong answer -> incorrect
C. open-ended correct answer, grounded in retrieved course evidence -> correct
D. open-ended wrong answer, contradicted by course evidence -> incorrect
E. alternative wording with the same correct meaning -> still accepted
   (no exact-string matching was introduced by this change)
F. insufficient evidence -> safe behavior (hedged feedback wording, state
   transition unaffected by confidence)
G. the evaluator receives the actual relevant course/reference evidence
H. feedback text matches the correctness decision in all three cases
   (correct / confidently wrong / not-confidently wrong)
I. an incorrect answer still follows the existing adaptive support path
   (AdaptivePolicy, unmodified)
J. a correct answer still follows the existing positive path
K. "I don't know" / support-request handling is unchanged (structural --
   still never reaches evaluate_practice_answer_tool at all)
L. guided practice never updates mastery (Mock call-count assertion,
   across the NEW retrieval branch too)
M. QPC's own mastery-update formula is untouched by this step

Also verifies the RQ2 privacy angle: the new retrieval call applies the
SAME institutional-privacy gateway (`for_external=True`) search_course_
material_tool already uses, with a real ChromaDB round trip proving a
RESTRICTED-classified chunk never reaches the judge's prompt.

Uses controlled fixtures throughout -- no live Gemini calls. Follows this
project's established convention: standalone script, hand-rolled check()
assertions, and patching each module's own locally-imported name.

Run from the `backend/` directory:
    python tests/test_guided_practice_evaluation.py
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


CONCEPT = "Linear Regression"
COURSE_CONTEXT = (
    "Linear regression models the relationship between a numeric target variable "
    "and one or more predictor variables by fitting a straight line that minimizes "
    "squared error between predicted and actual values."
)

import tools.tutor_state_tools as tst
import services.error_analyzer as error_analyzer_module
from services.tutor_state_machine import TutorState, save_tutor_state, load_tutor_state


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


def make_capturing_llm(response_json):
    capture = _Capture()

    def _fn(prompt_value):
        capture.prompt_value = prompt_value
        return _FakeResponse(response_json)

    return RunnableLambda(_fn), capture


def judgment_json(correct, confident=True, error_type=None, reason=""):
    import json
    return json.dumps({"correct": correct, "confident": confident, "error_type": error_type, "feedback_reason": reason})


engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
TestSession = sessionmaker(bind=engine)
db = TestSession()
student = Student(moodle_user_id=77001, username="guided_practice_student")
db.add(student)
db.commit()
db.refresh(student)
memory = MemoryManager(db)

student_id, course_id, session_id = student.id, "gp-course", "gp-session"

orig_get_json_llm = error_analyzer_module.get_json_llm


def base_context(message, tutor_state, extra=None):
    ctx = {
        "session_id": session_id, "message": message, "student_id": student_id,
        "course_db_id": course_id, "tutor_state": tutor_state, "memory": memory, "db": db,
        "difficulty": tutor_state.get("difficulty", "medium"),
        "remediation_level": "course", "canonical_courses": [], "retrieval_course_ids": [],
    }
    ctx.update(extra or {})
    return ctx


def run_evaluation(answer, tutor_state, response_json, extra_context=None):
    llm, capture = make_capturing_llm(response_json)
    error_analyzer_module.get_json_llm = lambda *a, **kw: llm
    try:
        ctx = base_context(answer, tutor_state, extra=extra_context)
        ctx["tutor_state"] = tutor_state
        result = tst.evaluate_practice_answer_tool(ctx, {"answer": answer})
    finally:
        error_analyzer_module.get_json_llm = orig_get_json_llm
    return result, capture


# ---------------------------------------------------------------------------
# A & B. EASY deterministic answer -- hint reaches the judge, no RAG
# retrieval is attempted (retrieval_course_ids present but should be
# ignored since a deterministic hint already exists).
# ---------------------------------------------------------------------------
easy_state = {
    "state": "GUIDED_PRACTICE", "concept": CONCEPT, "difficulty": "easy",
    "rounds_completed": 0, "consecutive_wrong": 0,
    "current_question": "Easy question on Linear Regression:\n\nWhich statement best describes it?",
    "asked_variants": {},
}
save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="easy",
                  rounds_completed=0, consecutive_wrong=0, current_question=easy_state["current_question"], asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": "B. It predicts a number from input variables."}})

result_a, capture_a = run_evaluation("B", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(True, True))
check("A. EASY correct answer with deterministic hint -> judged correct", result_a.get("correct") is True, result_a)
check("A2. the deterministic expected-answer hint reached the judge prompt", "correct option is B. It predicts a number from input variables." in capture_a.prompt_text(), capture_a.prompt_text())
# Note: "COURSE EVIDENCE" also appears as static INSTRUCTIONAL text in the
# judge's system prompt regardless of this call -- check for the actual
# human-template block marker (only present when evidence was supplied),
# not the bare phrase.
check("A3. no RAG retrieval/course-evidence block was needed or sent for EASY (deterministic hint takes priority)", "COURSE EVIDENCE (judge primarily" not in capture_a.prompt_text())

save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="easy",
                  rounds_completed=0, consecutive_wrong=0, current_question=easy_state["current_question"], asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": "B. It predicts a number from input variables."}})
result_b, capture_b = run_evaluation("D", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(False, True, "conceptual_misunderstanding", "Picked an unrelated option."))
check("B. EASY wrong answer with deterministic hint -> judged incorrect", result_b.get("correct") is False, result_b)


# ---------------------------------------------------------------------------
# C, D, G. Open-ended (MODERATE/HARD) answer -- grounded via real retrieval.
# Real ChromaDB round trip (temp path, FakeEmbeddings -- no Ollama call).
# ---------------------------------------------------------------------------
import pipelines.rag_pipeline as rag_pipeline_module
from pipelines.rag_pipeline import ingest_documents, get_vectorstore

tmp_chroma_dir = tempfile.mkdtemp(prefix="acrla_gp_test_chroma_")
orig_chroma_path = rag_pipeline_module.settings.chroma_path
orig_get_embeddings = rag_pipeline_module.get_embeddings
rag_pipeline_module.settings.chroma_path = tmp_chroma_dir
rag_pipeline_module.get_embeddings = lambda: FakeEmbeddings(size=32)
get_vectorstore.cache_clear()

GP_COURSE_ID = 77101
RESTRICTED_SENTINEL = "RESTRICTED_EXAM_ANSWER_SENTINEL"
PUBLIC_SENTINEL = "PUBLIC_LECTURE_SENTINEL"


def write_text_doc(tmp_dir, filename, text):
    import os
    path = os.path.join(tmp_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


docs_dir = tempfile.mkdtemp(prefix="acrla_gp_test_docs_")
try:
    ingest_documents(GP_COURSE_ID, [write_text_doc(docs_dir, "public.txt", f"{COURSE_CONTEXT} {PUBLIC_SENTINEL} is public lecture content.")], concept=CONCEPT, sensitivity="PUBLIC")
    ingest_documents(GP_COURSE_ID, [write_text_doc(docs_dir, "restricted.txt", f"Linear regression exam answer key: {RESTRICTED_SENTINEL} must never leave this server.")], concept=CONCEPT, sensitivity="RESTRICTED")

    open_state = {
        "state": "GUIDED_PRACTICE", "concept": CONCEPT, "difficulty": "medium",
        "rounds_completed": 0, "consecutive_wrong": 0,
        "current_question": "Moderate question on Linear Regression:\n\nExplain what linear regression predicts.",
        "asked_variants": {},
    }
    save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                      rounds_completed=0, consecutive_wrong=0, current_question=open_state["current_question"], asked_variants={})
    # No expected_answer_hint set for this session -- open-ended path.
    memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})

    # C. correct, grounded answer.
    result_c, capture_c = run_evaluation(
        "It predicts a numeric target value from predictor variables.",
        load_tutor_state(memory, student_id, course_id, session_id),
        judgment_json(True, True, None, "Matches the course material."),
        extra_context={"retrieval_course_ids": [GP_COURSE_ID], "canonical_courses": []},
    )
    check("C. open-ended answer grounded in retrieved evidence -> judged correct", result_c.get("correct") is True, result_c)
    check("G. the judge prompt actually contains the retrieved COURSE EVIDENCE block", "COURSE EVIDENCE (judge primarily" in capture_c.prompt_text() and PUBLIC_SENTINEL in capture_c.prompt_text(), capture_c.prompt_text()[-600:])
    check("privacy: RESTRICTED content is NEVER included in the evidence sent to the judge", RESTRICTED_SENTINEL not in capture_c.prompt_text(), capture_c.prompt_text())

    save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                      rounds_completed=0, consecutive_wrong=0, current_question=open_state["current_question"], asked_variants={})
    memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})

    # D. wrong, grounded answer (contradicted by evidence).
    result_d, capture_d = run_evaluation(
        "It sorts a list of numbers from smallest to largest.",
        load_tutor_state(memory, student_id, course_id, session_id),
        judgment_json(False, True, "conceptual_misunderstanding", "Describes sorting, not regression."),
        extra_context={"retrieval_course_ids": [GP_COURSE_ID], "canonical_courses": []},
    )
    check("D. open-ended answer contradicted by retrieved evidence -> judged incorrect", result_d.get("correct") is False, result_d)

    # E. alternative wording, same meaning -- still accepted (no exact-
    # string matching layer was introduced by this change).
    save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                      rounds_completed=0, consecutive_wrong=0, current_question=open_state["current_question"], asked_variants={})
    memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
    result_e, capture_e = run_evaluation(
        "Given some input factors, it estimates a continuous numeric outcome.",  # different wording, same meaning
        load_tutor_state(memory, student_id, course_id, session_id),
        judgment_json(True, True, None, "Same idea, different wording -- still correct."),
        extra_context={"retrieval_course_ids": [GP_COURSE_ID], "canonical_courses": []},
    )
    check("E. alternative wording conveying the same correct meaning is still accepted", result_e.get("correct") is True, result_e)

finally:
    rag_pipeline_module.settings.chroma_path = orig_chroma_path
    rag_pipeline_module.get_embeddings = orig_get_embeddings
    get_vectorstore.cache_clear()


# ---------------------------------------------------------------------------
# F & H. Insufficient evidence -> safe, hedged behavior; feedback matches
# the decision in every branch.
# ---------------------------------------------------------------------------
no_evidence_state = {
    "state": "GUIDED_PRACTICE", "concept": CONCEPT, "difficulty": "medium",
    "rounds_completed": 0, "consecutive_wrong": 0,
    "current_question": "Moderate question on Linear Regression:\n\nExplain what linear regression predicts.",
    "asked_variants": {},
}
save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                  rounds_completed=0, consecutive_wrong=0, current_question=no_evidence_state["current_question"], asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
# retrieval_course_ids intentionally empty -- no course material available at all.
result_f, capture_f = run_evaluation(
    "I think it predicts something numeric, not totally sure.",
    load_tutor_state(memory, student_id, course_id, session_id),
    judgment_json(False, False, None, "Vague, and I have no course material to verify against."),
    extra_context={"retrieval_course_ids": [], "canonical_courses": []},
)
check("F1. insufficient evidence: no COURSE EVIDENCE block was sent (none was available)", "COURSE EVIDENCE (judge primarily" not in capture_f.prompt_text())
check("F2. insufficient evidence: the low-confidence wrong verdict produces HEDGED feedback, not a confident wrong claim", "not fully certain" in result_f.get("reply", "").lower(), result_f.get("reply"))
check("F3. insufficient evidence: feedback never claims a specific error type when the judge wasn't confident", "conceptual misunderstanding" not in result_f.get("reply", "").lower() and "logic error" not in result_f.get("reply", "").lower(), result_f.get("reply"))
check("H1. feedback correctly matches an incorrect decision (never says 'Correct!')", "Correct!" not in result_f.get("reply", ""), result_f.get("reply"))

# H (correct branch) and H (confidently wrong branch), for completeness.
save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                  rounds_completed=0, consecutive_wrong=0, current_question=no_evidence_state["current_question"], asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
result_h_correct, _ = run_evaluation("A correct, confident answer.", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(True, True), extra_context={"retrieval_course_ids": [], "canonical_courses": []})
check("H2. feedback correctly matches a correct decision ('Correct!' present)", "Correct!" in result_h_correct.get("reply", ""), result_h_correct.get("reply"))

save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                  rounds_completed=0, consecutive_wrong=0, current_question=no_evidence_state["current_question"], asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
result_h_wrong_confident, _ = run_evaluation("A confidently wrong answer.", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(False, True, "logic_error", "Clear mistake."), extra_context={"retrieval_course_ids": [], "canonical_courses": []})
check("H3. a confidently-wrong verdict still uses the normal (non-hedged) wording", "Not quite" in result_h_wrong_confident.get("reply", "") and "not fully certain" not in result_h_wrong_confident.get("reply", "").lower(), result_h_wrong_confident.get("reply"))


# ---------------------------------------------------------------------------
# I & J. Adaptivity is preserved -- AdaptivePolicy itself is untouched;
# confirm the existing transitions still fire the same way through the
# real evaluate_practice_answer_tool.
# ---------------------------------------------------------------------------
# I: first wrong answer -> same difficulty, retry (not stepped back yet).
save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                  rounds_completed=0, consecutive_wrong=0, current_question="Q1", asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
result_i1, _ = run_evaluation("wrong", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(False, True, "logic_error"), extra_context={"retrieval_course_ids": [], "canonical_courses": []})
check("I1. first wrong answer -> stays in GUIDED_PRACTICE (hint/retry), not stepped back yet", result_i1.get("next_state") == "GUIDED_PRACTICE", result_i1)

# Second consecutive wrong -> steps back to EXAMPLE (REPEATED_WRONG_THRESHOLD = 2).
save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                  rounds_completed=0, consecutive_wrong=1, current_question="Q2", asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
result_i2, _ = run_evaluation("wrong again", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(False, True, "logic_error"), extra_context={"retrieval_course_ids": [], "canonical_courses": []})
check("I2. second consecutive wrong answer -> steps back to EXAMPLE (existing AdaptivePolicy threshold, unmodified)", result_i2.get("next_state") == "EXAMPLE", result_i2)

# J: correct + confident -> escalates (stays GUIDED_PRACTICE at a higher difficulty, or PROGRESS_CHECK_READY).
save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                  rounds_completed=0, consecutive_wrong=0, current_question="Q3", asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
result_j1, _ = run_evaluation("correct and confident", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(True, True), extra_context={"retrieval_course_ids": [], "canonical_courses": []})
check("J1. correct + confident answer -> follows the existing positive/escalate path", result_j1.get("next_state") in ("GUIDED_PRACTICE", "PROGRESS_CHECK_READY"), result_j1)

# correct + NOT confident -> same difficulty, another round (not escalated).
save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                  rounds_completed=0, consecutive_wrong=0, current_question="Q4", asked_variants={})
memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
result_j2, _ = run_evaluation("correct but shaky", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(True, False), extra_context={"retrieval_course_ids": [], "canonical_courses": []})
after_state_j2 = load_tutor_state(memory, student_id, course_id, session_id)
check("J2. correct + not-confident answer -> stays at the SAME difficulty (not escalated)", after_state_j2.get("difficulty") == "medium", after_state_j2)


# ---------------------------------------------------------------------------
# K. "I don't know" / support-request handling is unchanged -- structural:
# the compiler still routes a pending-question + needs_support turn away
# from evaluate_practice_answer entirely (Task K's own fix, untouched here).
# ---------------------------------------------------------------------------
import agents.plan_compiler as plan_compiler_module
compiler_source = inspect.getsource(plan_compiler_module)
check("K. needs_support with a pending question still never reaches evaluate_practice_answer (untouched by this step)", 'signal == "needs_support"' in compiler_source and 'return [], "conversation"' in compiler_source)


# ---------------------------------------------------------------------------
# L. Guided practice never updates mastery -- Mock call-count assertion
# across the NEW retrieval branch too (the highest-risk place for a
# regression, since it's new code touching the same function).
# ---------------------------------------------------------------------------
from unittest.mock import Mock
set_mastery_mock = Mock()
update_mastery_mock = Mock()
orig_set_mastery, orig_update_mastery = MemoryManager.set_mastery, MemoryManager.update_mastery
MemoryManager.set_mastery, MemoryManager.update_mastery = set_mastery_mock, update_mastery_mock
try:
    save_tutor_state(memory, student_id, course_id, session_id, state="GUIDED_PRACTICE", concept=CONCEPT, difficulty="medium",
                      rounds_completed=0, consecutive_wrong=0, current_question="Q5", asked_variants={})
    memory.update_course_memory(student_id, course_id, {"tutor_state": {**memory.get_tutor_state(student_id, course_id), "expected_answer_hint": None}})
    run_evaluation("some answer", load_tutor_state(memory, student_id, course_id, session_id), judgment_json(True, True), extra_context={"retrieval_course_ids": [77101], "canonical_courses": []})
finally:
    MemoryManager.set_mastery, MemoryManager.update_mastery = orig_set_mastery, orig_update_mastery
check("L1. MemoryManager.set_mastery is NEVER called by guided-practice evaluation", set_mastery_mock.call_count == 0, set_mastery_mock.call_count)
check("L2. MemoryManager.update_mastery is NEVER called by guided-practice evaluation", update_mastery_mock.call_count == 0, update_mastery_mock.call_count)


# ---------------------------------------------------------------------------
# M. QPC's own mastery-update formula is untouched by this step.
# ---------------------------------------------------------------------------
import routers.api as api_module
submit_source = inspect.getsource(api_module.submit_assessment)
check("M. QPC's mastery formula (0.7*previous + 0.3*assessment_score) is present and unmodified", "0.7 * previous_mastery" in submit_source and "0.3 * assessment_score" in submit_source, None)


# ---------------------------------------------------------------------------
# Structural: judge_answer's new parameter is backward compatible (default
# None), and the grounding wiring calls retrieve_context_for_scope with
# for_external=True (the RQ2 institutional-privacy gateway).
# ---------------------------------------------------------------------------
judge_answer_sig = inspect.signature(error_analyzer_module.judge_answer)
check("structural: judge_answer's course_evidence parameter defaults to None (backward compatible)", judge_answer_sig.parameters["course_evidence"].default is None)
eval_tool_source = inspect.getsource(tst.evaluate_practice_answer_tool)
check("structural: evaluate_practice_answer_tool retrieves with for_external=True (RQ2 gateway applied)", "for_external=True" in eval_tool_source)
check("structural: no extra LLM call was introduced (still exactly one judge_answer call)", eval_tool_source.count("judge_answer(") == 1)


print()
passed = sum(1 for _, ok in results if ok)
print(f"{passed}/{len(results)} passed")
assert passed == len(results)
