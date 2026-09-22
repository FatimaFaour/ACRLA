"""
Continuous Adaptive Tutoring Engine — phi3:mini optimized.
Shorter, more direct prompts work better with small local models.
"""

from langchain.prompts import ChatPromptTemplate
from services.llm_factory import get_llm
from services.intent_classifier import Strategy, get_strategy_instruction


def _llm(temperature: float = 0.4):
    return get_llm(temperature=temperature, max_tokens=400)


# ── Question generator ───────────────────────────────────────────────────────

QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Generate ONE {difficulty} {question_type} question about: {concept}
Strategy: {strategy_instruction}

Reply in exactly this format:
QUESTION: <question text>
EXPECTED: <key answer points>
HINT: <subtle hint without giving away the answer>"""),
    ("human", "Generate the question now."),
])


def generate_question(concept: str, difficulty: str = "medium",
                      question_type: str = "open", strategy: Strategy = Strategy.MODERATE) -> dict:
    chain = QUESTION_PROMPT | _llm()
    result = chain.invoke({
        "concept": concept,
        "difficulty": difficulty,
        "question_type": question_type,
        "strategy_instruction": get_strategy_instruction(strategy),
    })
    return {
        "question": _extract(result.content, "QUESTION:"),
        "expected": _extract(result.content, "EXPECTED:"),
        "hint": _extract(result.content, "HINT:"),
    }


def _extract(text: str, label: str) -> str:
    for line in text.split("\n"):
        if line.startswith(label):
            return line[len(label):].strip()
    return ""


# ── Answer evaluator ─────────────────────────────────────────────────────────

EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Evaluate this student answer. Reply ONLY in JSON, no markdown:
{{"score": 0.0-1.0, "correct": true/false, "feedback": "one sentence", "mastery_delta": -0.05 to 0.15}}"""),
    ("human", "Concept: {concept}\nQuestion: {question}\nExpected: {expected}\nStudent answer: {student_answer}"),
])


def evaluate_answer(concept: str, question: str, expected: str, student_answer: str) -> dict:
    import json, re
    chain = EVAL_PROMPT | get_llm(temperature=0.1, max_tokens=150)
    result = chain.invoke({
        "concept": concept, "question": question,
        "expected": expected, "student_answer": student_answer,
    })
    try:
        # Strip markdown fences if phi3 adds them
        clean = re.sub(r"```[a-z]*|```", "", result.content).strip()
        data = json.loads(clean)
        return {
            "score": float(data.get("score", 0.5)),
            "correct": bool(data.get("correct", False)),
            "feedback": data.get("feedback", ""),
            "mastery_delta": float(data.get("mastery_delta", 0.0)),
        }
    except Exception:
        return {"score": 0.5, "correct": False, "feedback": result.content, "mastery_delta": 0.0}


# ── Hint generator ───────────────────────────────────────────────────────────

HINT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Give a level-{hint_level} hint (1=subtle, 3=direct) for this question. "
               "Do NOT reveal the full answer. Strategy: {strategy_instruction}"),
    ("human", "Question: {question}"),
])


def generate_hint(question: str, hint_level: int = 1, strategy: Strategy = Strategy.MODERATE) -> str:
    chain = HINT_PROMPT | get_llm(temperature=0.5, max_tokens=150)
    result = chain.invoke({
        "question": question,
        "hint_level": hint_level,
        "strategy_instruction": get_strategy_instruction(strategy),
    })
    return result.content


# ── Difficulty adapter ───────────────────────────────────────────────────────

def adapt_difficulty(current_difficulty: str, recent_scores: list[float]) -> str:
    if len(recent_scores) < 2:
        return current_difficulty
    avg = sum(recent_scores) / len(recent_scores)
    if avg >= 0.8 and current_difficulty != "hard":
        return "hard" if current_difficulty == "medium" else "medium"
    elif avg <= 0.4 and current_difficulty != "easy":
        return "easy" if current_difficulty == "medium" else "medium"
    return current_difficulty


# ── Session greeting ─────────────────────────────────────────────────────────

GREETING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are ACRLA, a friendly CS tutor in Moodle. Write a warm 2-sentence greeting. Be encouraging."),
    ("human", "Student: {username}\nChapter: {chapter_name}\nScore: {score}%\n"
              "Weak concepts: {weak_concepts}\nPrevious sessions: {session_count}"),
])


def generate_greeting(username: str, chapter_name: str, score: float,
                      weak_concepts: list[str], session_count: int) -> str:
    chapter = chapter_name or "your CS course"
    if weak_concepts:
        focus = ", ".join(weak_concepts[:3])
        return (
            f"Hi {username}! Welcome back to {chapter}. "
            f"We'll focus on {focus} today and build it up step by step."
        )
    if session_count > 1:
        return (
            f"Hi {username}! Welcome back to {chapter}. "
            "Ask me anything from your course and we'll keep strengthening your understanding."
        )
    return (
        f"Hi {username}! I'm ACRLA, your learning assistant for {chapter}. "
        "Ask me a question or tell me what feels unclear, and we'll work through it together."
    )
