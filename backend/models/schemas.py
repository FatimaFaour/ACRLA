"""
ACRLA API schema definitions.

Purpose:
    Defines the request and response contracts shared by the FastAPI backend,
    standalone frontend, and Moodle local plugin.

Role in ACRLA:
    These models are the public boundary for sessions, chat responses, Moodle
    sync, document ingestion, analytics, and Quick Progress Check assessments.

Main responsibilities:
    - validate inbound Moodle/student/course payloads
    - keep chat response metadata stable for the UI
    - expose assessment question and mastery update fields
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


# ── Inbound payload from Moodle plugin ──────────────────────────────────────

class MoodlePayload(BaseModel):
    """Moodle/student/course context used to start or resume ACRLA."""
    student_id: int
    username: str
    email: Optional[str] = None
    course_id: int
    course_name: str
    chapter_id: Optional[int] = None
    chapter_name: Optional[str] = None
    scores: dict = {}           # {"chapter_2": 0.42, "chapter_4": 0.63}
    weak_concepts: list[str] = []
    learning_preferences: dict = {}
    selected_mode: str = "internal"   # internal | external


# ── Session management ───────────────────────────────────────────────────────

class SessionStartRequest(BaseModel):
    moodle_payload: MoodlePayload
    difficulty: Optional[str] = "medium"
    learning_goal: Optional[str] = None


class SessionStartResponse(BaseModel):
    session_id: str
    greeting: str
    weak_concepts: list[str]
    available_concepts: list[str] = []
    suggested_difficulty: str
    difficulty: Optional[str] = None
    mode: str
    learning_mode: Optional[str] = None


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    student_id: int


class ChatResponse(BaseModel):
    """Chat reply plus routing metadata used by the frontend.

    `selected_pipeline` and `source_mode` determine whether PDF source chips
    should be shown. `mastery_update` remains for compatibility, but chat
    responses should not change mastery in the MVP.

    `tutor_state` and `proactive_bootstrap` are structured adaptive-tutor
    presentation metadata (see services.tutor_state_machine /
    services.remediation_bootstrap) -- the frontend uses these to drive UI
    (a subtle current-stage label, a one-time "today's focus" card) instead
    of parsing the reply text. `proactive_bootstrap` is non-null ONLY on the
    single turn a proactive remediation session actually started.
    """
    reply: str
    intent: str
    strategy: str
    mastery_update: Optional[dict] = None
    difficulty: Optional[str] = None
    mode: Optional[str] = None
    learning_mode: Optional[str] = None
    selected_pipeline: Optional[str] = None
    source_mode: Optional[str] = None
    retrieved_sources: list[str] = []
    sources: list[str] = []
    tutor_state: Optional[str] = None
    proactive_bootstrap: Optional[dict] = None


# ── Document ingestion ───────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    course_id: int
    chunks_created: int
    files_processed: list[str]


# ── Analytics ────────────────────────────────────────────────────────────────

class StudentAnalytics(BaseModel):
    student_id: int
    weak_concepts: list[dict]
    mastery_by_concept: dict
    session_count: int
    total_messages: int


# Moodle sync

class MoodleSyncRequest(BaseModel):
    student_id: int
    course_id: int
    student_name: str
    course_name: str
    learning_mode: Optional[Literal["internal", "external"]] = None
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None
    mastery: dict[str, float] = Field(default_factory=dict)


class MoodleSyncResponse(BaseModel):
    status: str
    student_id: int
    course_id: int
    student_name: str
    weak_concepts: list[str]
    strongest_concepts: list[str]
    mastery: dict[str, float]


# Quick remediation assessment

class AssessmentStartRequest(BaseModel):
    session_id: str
    student_id: int


class AssessmentQuestion(BaseModel):
    id: str
    question_id: Optional[str] = None
    variant_id: Optional[str] = None
    sub_concept: Optional[str] = None
    prompt: str
    options: list[str]
    concepts_used: list[str] = Field(default_factory=list)
    course_ids_used: list[int] = Field(default_factory=list)
    question_type: str = "single_concept"


class AssessmentStartResponse(BaseModel):
    """Assessment payload shown in the Quick Progress Check modal."""
    assessment_id: str
    level_type: str
    course_id: int
    concept: Optional[str] = None
    moodle_initial_mastery: float = 0.0
    current_acrla_mastery: float = 0.0
    questions: list[AssessmentQuestion]


class AssessmentSubmitRequest(BaseModel):
    assessment_id: str
    session_id: str
    student_id: int
    answers: dict[str, str]


class AssessmentSubmitResponse(BaseModel):
    """Result of scoring a Quick Progress Check and persisting mastery."""
    status: str
    moodle_initial_mastery: float
    current_acrla_mastery: float
    previous_mastery: float
    assessment_score: float
    calculated_mastery: float
    updated_mastery: float
    mastery_delta: float
    level_type: str
    course_id: int
    concept: Optional[str] = None
    correct: int
    total: int
