"""
ACRLA Evaluation — Scenario-based tests.
Simulates weak student, moderate student, and exam-prep scenarios.
Run: python tests/evaluation_scenarios.py --backend http://localhost:8000
"""

import requests
import json
import argparse
import sys
from dataclasses import dataclass


@dataclass
class TestResult:
    scenario: str
    passed: bool
    details: str


def start_session(backend: str, student_id: int, course_id: int, weak_concepts: list) -> str:
    res = requests.post(f"{backend}/api/v1/session/start", json={
        "moodle_payload": {
            "student_id": student_id,
            "username": f"Test Student {student_id}",
            "email": f"student{student_id}@test.com",
            "course_id": course_id,
            "course_name": "Test Course",
            "scores": {"chapter_1": 0.4},
            "weak_concepts": weak_concepts,
            "learning_preferences": {},
            "selected_mode": "internal",
        }
    })
    res.raise_for_status()
    return res.json()["session_id"]


def chat(backend: str, session_id: str, student_id: int, message: str) -> dict:
    res = requests.post(f"{backend}/api/v1/chat", json={
        "session_id": session_id,
        "message": message,
        "student_id": student_id,
    })
    res.raise_for_status()
    return res.json()


def run_scenario_weak_student(backend: str) -> TestResult:
    """Weak student should get simplified, scaffolded responses."""
    try:
        sid = start_session(backend, 101, 1, ["recursion", "pointers", "memory management"])
        resp = chat(backend, sid, 101, "I don't understand recursion at all")

        passed = (
            resp["intent"] in ["tutoring", "engagement"] and
            resp["strategy"] in ["weak", "weak_confused"] and
            len(resp["reply"]) > 50
        )
        return TestResult(
            scenario="Weak student — confused about recursion",
            passed=passed,
            details=f"Intent={resp['intent']}, Strategy={resp['strategy']}, Reply length={len(resp['reply'])}"
        )
    except Exception as e:
        return TestResult("Weak student", False, str(e))


def run_scenario_intent_routing(backend: str) -> TestResult:
    """Analytics intent should return mastery data, not tutoring content."""
    try:
        sid = start_session(backend, 102, 1, ["sorting algorithms"])
        resp = chat(backend, sid, 102, "What am I weak in? Show me my progress")

        passed = resp["intent"] == "analytics"
        return TestResult(
            scenario="Analytics intent routing",
            passed=passed,
            details=f"Intent={resp['intent']} (expected: analytics)"
        )
    except Exception as e:
        return TestResult("Analytics routing", False, str(e))


def run_scenario_mode_switch(backend: str) -> TestResult:
    """Student requesting harder questions should trigger preference update."""
    try:
        sid = start_session(backend, 103, 1, [])
        resp = chat(backend, sid, 103, "Give me harder questions please")

        passed = (
            resp["intent"] == "preference" and
            resp["difficulty"] == "hard"
        )
        return TestResult(
            scenario="Preference update — harder questions",
            passed=passed,
            details=f"Intent={resp['intent']}, Difficulty={resp['difficulty']}"
        )
    except Exception as e:
        return TestResult("Preference update", False, str(e))


def run_scenario_navigation(backend: str) -> TestResult:
    """Navigation intent should be detected correctly."""
    try:
        sid = start_session(backend, 104, 1, [])
        resp = chat(backend, sid, 104, "Switch to the chapter on binary trees")

        passed = resp["intent"] == "navigation"
        return TestResult(
            scenario="Navigation intent detection",
            passed=passed,
            details=f"Intent={resp['intent']} (expected: navigation)"
        )
    except Exception as e:
        return TestResult("Navigation", False, str(e))


def run_scenario_engagement_recovery(backend: str) -> TestResult:
    """Frustrated student should get recovery/engagement response."""
    try:
        sid = start_session(backend, 105, 1, ["binary trees"])
        resp = chat(backend, sid, 105, "This is too difficult, I give up")

        passed = resp["intent"] == "engagement"
        return TestResult(
            scenario="Engagement recovery — frustrated student",
            passed=passed,
            details=f"Intent={resp['intent']}, Reply snippet: {resp['reply'][:80]}..."
        )
    except Exception as e:
        return TestResult("Engagement recovery", False, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  ACRLA Evaluation Suite")
    print(f"  Backend: {args.backend}")
    print(f"{'='*60}\n")

    # Check health
    try:
        r = requests.get(f"{args.backend}/health", timeout=5)
        r.raise_for_status()
        print("✓ Backend is healthy\n")
    except Exception as e:
        print(f"✗ Backend not reachable: {e}")
        sys.exit(1)

    scenarios = [
        run_scenario_weak_student,
        run_scenario_intent_routing,
        run_scenario_mode_switch,
        run_scenario_navigation,
        run_scenario_engagement_recovery,
    ]

    results = []
    for fn in scenarios:
        result = fn(args.backend)
        results.append(result)
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} | {result.scenario}")
        print(f"       {result.details}\n")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"{'='*60}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'='*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
