# ACRLA Demo Testing Guide

Use this checklist to manually validate the main ACRLA demo flows end-to-end through the UI.

## Quick Demo Checklist

- Backend starts without syntax/import errors.
- Moodle dashboard shows overall and course ACRLA mastery controls.
- Moodle course page shows inline chapter mastery controls.
- Floating ACRLA button opens the overview panel.
- Course/chapter mastery clicks open scoped remediation in the same panel.
- Internal RAG answers show PDF/display-title sources.
- External fallback answers show no PDF sources.
- Quick Progress Check updates current ACRLA mastery.
- Closing/reopening ACRLA keeps the updated mastery.

## Backend Startup

```powershell
cd backend
python -m py_compile routers\api.py services\chat_orchestrator.py pipelines\rag_pipeline.py pipelines\hybrid_pipeline.py services\memory_manager.py models\schemas.py
uvicorn main:app --reload --port 8000
```

Open:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`

## Automatic Routing

Automatic routing is visible through the frontend route badge and console logs.
The student should not manually choose Internal or External mode.

### Internal RAG

Ask:

```text
Explain recursion
```

Expected:

- `selected_pipeline = internal_rag`
- badge shows course-material routing
- source includes the recursion PDF/display title

Ask:

```text
what are pointers
```

Expected:

- `selected_pipeline = internal_rag`
- source includes the pointers/memory PDF/display title

### External Fallback

Ask:

```text
Explain quantum computing
```

Expected:

- `selected_pipeline = external_fallback`
- no PDF source is displayed
- answer does not reuse the previous course topic

## Course Scope Isolation

### Computer Science

Open Introduction to Computer Science remediation.

Expected concepts:

- Recursion
- Sorting Algorithms
- Pointers and Memory Management
- Binary Trees and BSTs

No Data Science or Mathematics concepts should appear.

Useful log checks:

- `clicked_course_id` is the CS Moodle id.
- `scope_concepts` contains only CS concepts.
- `active_tutoring_concept` is one CS concept.

### Data Science

Open Data Science remediation.

Expected concepts depend on synced materials. For the demo:

- Linear Regression

No Computer Science concepts should appear.

Useful log checks:

- `clicked_course_id` is the Data Science Moodle id.
- `scope_concepts` contains only Data Science material concepts.
- no `Recursion`, `Pointers`, `Sorting`, or `Binary Trees` appears.

### Mathematics

Open Mathematics remediation.

Expected concepts:

- Logic
- Sets
- Graphs
- Relations Functions

No Computer Science concepts should appear.

## Quick Progress Check

1. Open a chapter/course/overall remediation launch.
2. Click Check my progress.
3. Submit the three-question assessment.

Expected:

- assessment scope matches the launch level
- mastery update is shown after submit
- current ACRLA mastery is persisted
- Moodle buttons refresh to the current ACRLA mastery

Important:

- Starting an assessment does not update mastery.
- Chat answers do not update mastery.
- Only assessment submission persists the updated current ACRLA mastery.

## Mastery Rules

Chat should not update mastery.

Test:

1. Ask for an explanation.
2. Ask for a practice question.
3. Answer in chat.
4. Recheck mastery.

Expected:

- no mastery increase from chat alone
- mastery changes only after Quick Progress Check submit

## Analytics Responses

Ask:

```text
What is my overall mastery level?
```

Expected:

- answer aggregates all available/enrolled courses
- no PDF source is displayed

Ask:

```text
What are my weakest concepts?
```

Expected:

- answer uses persisted mastery/profile data
- no PDF source is displayed
