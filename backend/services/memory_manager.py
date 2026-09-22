"""
Three-layer memory system:
  Layer 1 — Conversation buffer (in-memory, current session window)
  Layer 2 — Session memory (DB: goals, topics, performance this session)
  Layer 3 — Long-term educational memory (DB: mastery history across sessions)
"""

import re
from datetime import datetime
from sqlalchemy.orm import Session as DBSession

from models.db_models import (
    Student, Session, ConversationMessage, MasteryRecord,
    StudentPreference, SessionAnalytics, StudentLongTermMemory,
)
from services.course_concepts import canonicalize_concept


# ==========================================================
# File Purpose
# ==========================================================
# Central persistence facade for ACRLA. This module hides the difference
# between short-lived conversation buffers, active session state, long-term
# Moodle/student profile memory, and mastery records.


def _mastery_concept(raw: str | None) -> str | None:
    concept = canonicalize_concept(raw)
    if concept:
        return concept
    text = str(raw or "").replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None
    return " ".join(part[:1].upper() + part[1:] for part in text.split())


# ── Layer 1: In-memory conversation buffer ───────────────────────────────────

class ConversationBuffer:
    """Short-term coherence within the active session window (last N messages)."""

    def __init__(self, max_messages: int = 20):
        self.messages: list[dict] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def to_langchain_format(self) -> list[dict]:
        return self.messages.copy()

    def last_n(self, n: int) -> list[dict]:
        return self.messages[-n:]


# Global buffer store (session_id → buffer)
_buffers: dict[str, ConversationBuffer] = {}
_current_topics: dict[str, str] = {}


def get_buffer(session_id: str) -> ConversationBuffer:
    if session_id not in _buffers:
        _buffers[session_id] = ConversationBuffer()
    return _buffers[session_id]


def clear_buffer(session_id: str):
    _buffers.pop(session_id, None)
    _current_topics.pop(session_id, None)


# ── Layer 2 & 3: Database-backed memory ─────────────────────────────────────

class MemoryManager:

    def __init__(self, db: DBSession):
        self.db = db

    # ── Student ──────────────────────────────────────────────────────────────

    def get_or_create_student(self, moodle_user_id: int, username: str, email: str = None) -> Student:
        student = self.db.query(Student).filter_by(moodle_user_id=moodle_user_id).first()
        if not student:
            student = Student(
                moodle_user_id=moodle_user_id,
                username=username,
                email=email,
            )
            self.db.add(student)
            self.db.commit()
            self.db.refresh(student)
        return student

    def get_or_create_long_term_memory(self, student_id: str) -> StudentLongTermMemory:
        memory = self.db.query(StudentLongTermMemory).filter_by(student_id=student_id).first()
        if not memory:
            memory = StudentLongTermMemory(student_id=student_id, profile={}, learning_state={})
            self.db.add(memory)
            self.db.commit()
            self.db.refresh(memory)
        return memory

    def set_profile_name(self, student_id: str, name: str):
        memory = self.get_or_create_long_term_memory(student_id)
        profile = dict(memory.profile or {})
        profile["name"] = name
        memory.profile = profile
        memory.updated_at = datetime.utcnow()
        student = self.db.query(Student).filter_by(id=student_id).first()
        if student:
            student.username = name
        self.db.commit()

    def get_profile_name(self, student_id: str) -> str | None:
        memory = self.get_or_create_long_term_memory(student_id)
        return (memory.profile or {}).get("name")

    def set_profile_preferences(self, student_id: str, difficulty: str = None, learning_mode: str = None):
        memory = self.get_or_create_long_term_memory(student_id)
        profile = dict(memory.profile or {})
        if difficulty:
            profile["difficulty"] = difficulty
        if learning_mode:
            profile["learning_mode"] = learning_mode
        memory.profile = profile
        memory.updated_at = datetime.utcnow()
        self.db.commit()

    def get_profile_preferences(self, student_id: str) -> dict:
        memory = self.get_or_create_long_term_memory(student_id)
        profile = dict(memory.profile or {})
        return {
            "name": profile.get("name"),
            "difficulty": profile.get("difficulty"),
            "learning_mode": profile.get("learning_mode"),
        }

    def get_course_memory(self, student_id: str, course_id: str) -> dict:
        """Return persisted learning state for one student/course pair."""
        memory = self.get_or_create_long_term_memory(student_id)
        state = dict(memory.learning_state or {})
        courses = dict(state.get("courses") or {})
        return dict(courses.get(str(course_id)) or {})

    def update_course_memory(self, student_id: str, course_id: str, updates: dict):
        """Merge course-scoped learning state into long-term memory.

        Callers that switch Moodle courses should explicitly overwrite scope
        keys because this method intentionally preserves unrelated course state.

        This merge behavior is useful for durable profile data, recent
        questions, and summaries, but it means launch code must reset
        remediation_concepts/selected_concept when the student opens a
        different Moodle course.
        """
        memory = self.get_or_create_long_term_memory(student_id)
        state = dict(memory.learning_state or {})
        courses = dict(state.get("courses") or {})
        course_memory = dict(courses.get(str(course_id)) or {})
        course_memory.update(updates)
        courses[str(course_id)] = course_memory
        state["courses"] = courses
        memory.learning_state = state
        memory.updated_at = datetime.utcnow()
        self.db.commit()
        return course_memory

    def record_asked_question(self, student_id: str, course_id: str, concept: str, question_text: str):
        concept = _mastery_concept(concept)
        if not concept or not question_text:
            return
        course_memory = self.get_course_memory(student_id, course_id)
        recent = list(course_memory.get("recent_questions") or [])
        item = {"concept": concept, "question": question_text}
        recent = [q for q in recent if q.get("question") != question_text]
        recent.append(item)
        recent = recent[-40:]
        self.update_course_memory(student_id, course_id, {
            "last_concept": concept,
            "last_question": question_text,
            "recent_questions": recent,
            "last_activity": f"practiced {concept}",
            "next_recommended_action": f"continue with {concept}",
        })

    def record_answer_history(
        self,
        student_id: str,
        course_id: str,
        concept: str,
        question_text: str,
        correct: bool,
    ):
        concept = _mastery_concept(concept)
        if not concept:
            return
        course_memory = self.get_course_memory(student_id, course_id)
        answers = list(course_memory.get("answer_history") or [])
        answers.append({
            "concept": concept,
            "question": question_text,
            "correct": bool(correct),
            "timestamp": datetime.utcnow().isoformat(),
        })
        answers = answers[-40:]
        self.update_course_memory(student_id, course_id, {
            "last_concept": concept,
            "last_activity": f"answered {'correctly' if correct else 'incorrectly'} about {concept}",
            "answer_history": answers,
            "next_recommended_action": f"continue with {concept}",
        })

    def get_recent_question_texts(self, student_id: str, course_id: str, concept: str = None) -> list[str]:
        concept = canonicalize_concept(concept) if concept else None
        recent = self.get_course_memory(student_id, course_id).get("recent_questions") or []
        return [
            item.get("question", "")
            for item in recent
            if item.get("question") and (not concept or item.get("concept") == concept)
        ]

    # ── Session ──────────────────────────────────────────────────────────────

    def start_session(
        self,
        student_id: str,
        course_id: str,
        chapter_id: str = None,
        mode: str = "internal",
        difficulty: str = "medium",
        learning_goal: str = None,
    ) -> Session:
        """Create a new active learning session.

        Existing active sessions for the same student are closed first so the
        chat and assessment flows have one canonical session id.
        """
        # Close any open sessions for this student
        open_sessions = self.db.query(Session).filter_by(
            student_id=student_id, is_active=True
        ).all()
        for s in open_sessions:
            s.is_active = False
            s.ended_at = datetime.utcnow()

        session = Session(
            student_id=student_id,
            course_id=course_id,
            chapter_id=chapter_id,
            mode=mode,
            difficulty=difficulty,
            learning_goal=learning_goal,
        )
        self.db.add(session)
        self.db.flush()

        # Create analytics record
        analytics = SessionAnalytics(session_id=session.id)
        self.db.add(analytics)

        self.db.commit()
        self.db.refresh(session)
        return session

    def end_session(self, session_id: str):
        session = self.db.query(Session).filter_by(id=session_id).first()
        if session:
            session.is_active = False
            session.ended_at = datetime.utcnow()
            self.db.commit()
        clear_buffer(session_id)

    def get_active_session(self, session_id: str) -> Session | None:
        return self.db.query(Session).filter_by(id=session_id, is_active=True).first()

    def set_current_topic(self, session_id: str, topic: str | None):
        """Store or clear the in-memory topic focus for the active session."""
        topic = canonicalize_concept(topic)
        if topic:
            _current_topics[session_id] = topic
        else:
            _current_topics.pop(session_id, None)

    def get_current_topic(self, session_id: str) -> str | None:
        return _current_topics.get(session_id)

    # ── Messages ─────────────────────────────────────────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: str = None,
        strategy: str = None,
    ):
        msg = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            strategy=strategy,
        )
        self.db.add(msg)

        # Update analytics
        analytics = self.db.query(SessionAnalytics).filter_by(session_id=session_id).first()
        if analytics:
            analytics.total_messages += 1

        self.db.commit()

        # Also add to in-memory buffer
        get_buffer(session_id).add(role, content)

    def get_session_history(self, session_id: str, limit: int = 50) -> list[ConversationMessage]:
        return (
            self.db.query(ConversationMessage)
            .filter_by(session_id=session_id)
            .order_by(ConversationMessage.timestamp.asc())
            .limit(limit)
            .all()
        )

    # ── Mastery ───────────────────────────────────────────────────────────────

    def _mastery_records_for_concept(self, student_id: str, course_id: str, concept: str) -> list[MasteryRecord]:
        concept = _mastery_concept(concept)
        if not concept:
            return []
        records = self.db.query(MasteryRecord).filter_by(
            student_id=student_id, course_id=course_id
        ).all()
        return [record for record in records if _mastery_concept(record.concept) == concept]

    def _latest_mastery_record(self, records: list[MasteryRecord]) -> MasteryRecord | None:
        if not records:
            return None
        return max(records, key=lambda record: record.last_updated or datetime.min)

    def get_mastery(self, student_id: str, course_id: str, concept: str) -> float:
        concept = _mastery_concept(concept)
        if not concept:
            return 0.0

        record = self._latest_mastery_record(
            self._mastery_records_for_concept(student_id, course_id, concept)
        )
        return record.mastery_level if record else 0.0

    def update_mastery(
        self,
        student_id: str,
        course_id: str,
        concept: str,
        delta: float,
        correct: bool = True,
    ):
        concept = _mastery_concept(concept)
        if not concept:
            return None

        records = self._mastery_records_for_concept(student_id, course_id, concept)
        record = self._latest_mastery_record(records)

        if not record:
            record = MasteryRecord(
                student_id=student_id,
                course_id=course_id,
                concept=concept,
            )
            self.db.add(record)
            records = [record]

        current_mastery = record.mastery_level or 0.0
        current_attempts = record.attempts or 0
        current_correct = record.correct or 0

        new_mastery = max(0.0, min(1.0, current_mastery + delta))
        now = datetime.utcnow()
        for item in records:
            item.concept = concept
            item.mastery_level = new_mastery
            item.attempts = current_attempts + 1
            item.correct = current_correct + 1 if correct else current_correct
            item.last_updated = now
        self.db.commit()
        return record

    def set_mastery(
        self,
        student_id: str,
        course_id: str,
        concept: str,
        mastery_level: float,
    ):
        """Persist the canonical mastery value for one course concept.

        Assessment submission and Moodle sync use this to keep database-backed
        mastery aligned with long-term memory. The caller decides whether the
        value represents initial Moodle mastery or current ACRLA mastery.
        """
        concept = _mastery_concept(concept)
        if not concept:
            return None

        records = self._mastery_records_for_concept(student_id, course_id, concept)
        record = self._latest_mastery_record(records)

        if not record:
            record = MasteryRecord(
                student_id=student_id,
                course_id=course_id,
                concept=concept,
            )
            self.db.add(record)
            records = [record]

        new_mastery = max(0.0, min(1.0, mastery_level))
        now = datetime.utcnow()
        for item in records:
            item.concept = concept
            item.mastery_level = new_mastery
            item.attempts = item.attempts or 0
            item.correct = item.correct or 0
            item.last_updated = now
        self.db.commit()
        return record

    def get_all_mastery(self, student_id: str, course_id: str) -> list[MasteryRecord]:
        records = self.db.query(MasteryRecord).filter_by(
            student_id=student_id, course_id=course_id
        ).all()
        canonical_records: dict[str, MasteryRecord] = {}
        for record in records:
            canonical = _mastery_concept(record.concept)
            if not canonical:
                continue
            record.concept = canonical
            existing = canonical_records.get(canonical)
            if existing is None or (record.last_updated or datetime.min) > (existing.last_updated or datetime.min):
                canonical_records[canonical] = record
        return list(canonical_records.values())

    def get_weak_concepts(self, student_id: str, course_id: str, threshold: float = 0.4) -> list[str]:
        records = sorted(self.get_all_mastery(student_id, course_id), key=lambda r: r.mastery_level or 0.0)
        return [r.concept for r in records if (r.mastery_level or 0.0) < threshold]

    def get_average_mastery(self, student_id: str, course_id: str) -> float:
        records = self.get_all_mastery(student_id, course_id)
        if not records:
            return 0.0
        return sum(r.mastery_level for r in records) / len(records)

    # ── Tutor state machine (services.tutor_state_machine) ────────────────────
    # Session-scoped resting pedagogical state + durable per-concept error
    # patterns, stored inside the same learning_state JSON as course memory
    # (see get_course_memory/update_course_memory above) -- never mastery.

    def get_tutor_state(self, student_id: str, course_id: str) -> dict:
        return dict(self.get_course_memory(student_id, course_id).get("tutor_state") or {})

    def set_tutor_state(self, student_id: str, course_id: str, state: dict) -> None:
        self.update_course_memory(student_id, course_id, {"tutor_state": state})

    def get_tutor_error_patterns(self, student_id: str, course_id: str, concept: str) -> list[dict]:
        concept = _mastery_concept(concept) or concept
        patterns = self.get_course_memory(student_id, course_id).get("tutor_error_patterns") or {}
        return list(patterns.get(concept) or [])

    def record_tutor_error_pattern(self, student_id: str, course_id: str, concept: str, error_type: str) -> None:
        concept = _mastery_concept(concept) or concept
        if not concept:
            return
        course_memory = self.get_course_memory(student_id, course_id)
        patterns = dict(course_memory.get("tutor_error_patterns") or {})
        entries = list(patterns.get(concept) or [])
        existing = next((e for e in entries if e.get("error_type") == error_type), None)
        now = datetime.utcnow().isoformat()
        if existing:
            existing["count"] = int(existing.get("count") or 0) + 1
            existing["last_seen"] = now
        else:
            entries.append({"error_type": error_type, "count": 1, "last_seen": now})
        patterns[concept] = entries
        self.update_course_memory(student_id, course_id, {"tutor_error_patterns": patterns})

    # ── Quick Progress Check (tools.assessment_tools) ──────────────────────────
    # Session-scoped chat-driven assessment state, stored in the same
    # learning_state JSON as tutor state -- see get_course_memory/
    # update_course_memory above. This is the only chat-tool flow allowed to
    # call set_mastery/update_mastery (via tools.assessment_tools, never
    # tools.tutor_state_tools/tools.mastery_tools).

    def get_quick_progress_check(self, student_id: str, course_id: str) -> dict:
        return dict(self.get_course_memory(student_id, course_id).get("quick_progress_check") or {})

    def set_quick_progress_check(self, student_id: str, course_id: str, state: dict) -> None:
        self.update_course_memory(student_id, course_id, {"quick_progress_check": state})

    # ── Preferences ──────────────────────────────────────────────────────────

    def get_preferences(self, student_id: str) -> StudentPreference | None:
        return self.db.query(StudentPreference).filter_by(student_id=student_id).first()

    def save_preferences(self, student_id: str, mode: str = None, difficulty: str = None):
        prefs = self.db.query(StudentPreference).filter_by(student_id=student_id).first()
        if not prefs:
            prefs = StudentPreference(student_id=student_id)
            self.db.add(prefs)
        if mode:
            prefs.preferred_mode = mode
        if difficulty:
            prefs.preferred_difficulty = difficulty
        prefs.updated_at = datetime.utcnow()
        self.db.commit()
