from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_uuid():
    return str(uuid.uuid4())


class Student(Base):
    __tablename__ = "students"

    id = Column(String, primary_key=True, default=generate_uuid)
    moodle_user_id = Column(Integer, unique=True, nullable=False)
    username = Column(String(100), nullable=False)
    email = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="student")
    masteries = relationship("MasteryRecord", back_populates="student")
    preferences = relationship("StudentPreference", back_populates="student", uselist=False)
    long_term_memory = relationship("StudentLongTermMemory", back_populates="student", uselist=False)


class Course(Base):
    __tablename__ = "courses"

    id = Column(String, primary_key=True, default=generate_uuid)
    moodle_course_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    sessions = relationship("Session", back_populates="course")
    chapters = relationship("Chapter", back_populates="course")


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(String, primary_key=True, default=generate_uuid)
    course_id = Column(String, ForeignKey("courses.id"), nullable=False)
    moodle_chapter_id = Column(Integer, nullable=False)
    name = Column(String(200), nullable=False)
    concepts = Column(JSON, default=list)  # list of concept strings

    course = relationship("Course", back_populates="chapters")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    student_id = Column(String, ForeignKey("students.id"), nullable=False)
    course_id = Column(String, ForeignKey("courses.id"), nullable=False)
    chapter_id = Column(String, nullable=True)
    mode = Column(String(20), default="internal")  # internal | external
    difficulty = Column(String(20), default="medium")  # easy | medium | hard
    learning_goal = Column(String(200))
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

    student = relationship("Student", back_populates="sessions")
    course = relationship("Course", back_populates="sessions")
    messages = relationship("ConversationMessage", back_populates="session")
    analytics = relationship("SessionAnalytics", back_populates="session", uselist=False)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    intent = Column(String(50))  # tutoring | analytics | navigation | engagement | preference
    strategy = Column(String(50))  # weak | moderate | excellent | exam | recovery
    timestamp = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")


class MasteryRecord(Base):
    __tablename__ = "mastery_records"

    id = Column(String, primary_key=True, default=generate_uuid)
    student_id = Column(String, ForeignKey("students.id"), nullable=False)
    course_id = Column(String, nullable=False)
    concept = Column(String(200), nullable=False)
    mastery_level = Column(Float, default=0.0)  # 0.0 to 1.0
    attempts = Column(Integer, default=0)
    correct = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="masteries")


class AssessmentRecord(Base):
    __tablename__ = "assessment_records"

    id = Column(String, primary_key=True, default=generate_uuid)
    student_id = Column(String, ForeignKey("students.id"), nullable=False)
    course_id = Column(String, nullable=False)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=True)
    concept = Column(String(300), nullable=True)
    level_type = Column(String(20), default="chapter")
    previous_mastery = Column(Float, default=0.0)  # percent, e.g. 25.0
    assessment_score = Column(Float, default=0.0)  # percent, e.g. 80.0
    updated_mastery = Column(Float, default=0.0)  # percent, e.g. 41.5
    details = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)


class AssessmentQuestionVariant(Base):
    __tablename__ = "assessment_question_variants"

    id = Column(String, primary_key=True, default=generate_uuid)
    course_id = Column(String, nullable=False)
    moodle_course_id = Column(Integer, nullable=True)
    concept = Column(String(300), nullable=False)
    sub_concept = Column(String(200), nullable=True)
    question_id = Column(String(200), nullable=False)
    variant_id = Column(String(240), unique=True, nullable=False)
    difficulty = Column(String(20), default="easy")
    prompt = Column(Text, nullable=False)
    options = Column(JSON, default=list)
    correct = Column(String(5), default="A")
    source_file = Column(String(300), nullable=True)
    display_title = Column(String(300), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class StudentPreference(Base):
    __tablename__ = "student_preferences"

    id = Column(String, primary_key=True, default=generate_uuid)
    student_id = Column(String, ForeignKey("students.id"), unique=True, nullable=False)
    preferred_mode = Column(String(20), default="internal")
    preferred_difficulty = Column(String(20), default="medium")
    language = Column(String(10), default="en")
    updated_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="preferences")


class StudentLongTermMemory(Base):
    __tablename__ = "student_long_term_memory"

    id = Column(String, primary_key=True, default=generate_uuid)
    student_id = Column(String, ForeignKey("students.id"), unique=True, nullable=False)
    profile = Column(JSON, default=dict)
    learning_state = Column(JSON, default=dict)
    updated_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="long_term_memory")


class SessionAnalytics(Base):
    __tablename__ = "session_analytics"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.id"), unique=True, nullable=False)
    total_messages = Column(Integer, default=0)
    correct_answers = Column(Integer, default=0)
    hints_requested = Column(Integer, default=0)
    difficulty_changes = Column(Integer, default=0)
    weak_concepts_addressed = Column(JSON, default=list)
    mastery_gained = Column(Float, default=0.0)

    session = relationship("Session", back_populates="analytics")
