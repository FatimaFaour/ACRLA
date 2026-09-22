# ACRLA

Adaptive Conversational Remediation and Learning Assistant

## Overview

ACRLA is an adaptive tutoring assistant embedded in Moodle courses. It
combines a FastAPI backend, an LLM-orchestrated conversation agent,
retrieval-augmented generation (RAG) over synced course PDFs, persistent
student memory, mastery analytics, a Moodle side-panel plugin, and Quick
Progress Check assessments. New Moodle courses do not require code changes:
courses and concepts are discovered dynamically from Moodle sync data,
synced PDF/resource metadata, and ChromaDB document metadata.

## Main Features

- Moodle-integrated adaptive remediation via a local Moodle plugin (floating
  launcher + persistent side panel).
- Chapter, course, and overall remediation launch levels, each with
  independently tracked mastery.
- A structured six-state adaptive tutoring loop (explain, example, guided
  practice, feedback, retry-or-advance, progress-check-ready) with
  per-concept error-pattern tracking, instead of a free-form reply every turn.
- Deterministic, mastery-aware adaptive concept selection (proactive
  remediation bootstrap, weakest-concept selection for course/overall
  launches).
- Retrieval-augmented generation (RAG) over Moodle-synced course PDFs via
  ChromaDB, with automatic internal-vs-external routing (no manual mode
  switch): reliable course evidence -> internal RAG with cited sources;
  otherwise -> a controlled external-knowledge fallback with no fabricated
  sources.
- A conversation agent (two interchangeable architectures, `simple` /
  `iterative`) that plans, calls deterministic tools, and validates evidence
  before answering, instead of routing on keyword/phrase matching.
- Quick Progress Check assessments (chat-triggered or UI-triggered), the
  only flow that updates mastery.
- Mastery tracking with independent chapter/course/overall values and a
  non-decreasing update formula.
- A privacy-aware boundary between local student/course context and any
  external LLM call: student identity is minimized by an explicit allow-list,
  and institution-assigned content sensitivity (PUBLIC/INTERNAL/RESTRICTED)
  gates what course material may leave the local environment.
- Multi-provider LLM support (Gemini by default; Groq, Ollama, OpenAI, and
  Cerebras are also supported) via a single provider factory.

## Architecture

```text
Moodle (plugin) / standalone frontend
        |
        v
   FastAPI backend (routers/)
        |
        v
Conversation orchestration & adaptive tutoring (services/, agents/, tools/)
        |
        +--> Internal RAG (ChromaDB retrieval over synced course PDFs)
        |
        +--> External LLM fallback (privacy-filtered context)
        |
        v
PostgreSQL (students, courses, sessions, mastery, long-term memory, assessments)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full request flow,
the conversation agent/routing layer, and the privacy boundary, and
[docs/tutoring_flow.md](docs/tutoring_flow.md) for the adaptive tutor state
machine, mastery rules, and Quick Progress Check flow.

## Repository Structure

```text
ACRLA/
|-- backend/                 FastAPI backend
|   |-- agents/              Agent brain, planner, executor, entity/evidence
|   |                        validators, response generator, policies
|   |-- tools/                Deterministic tool implementations, by domain
|   |                         (rag, analytics, mastery, course, memory,
|   |                         profile, tutoring, external, dialogue,
|   |                         tutor_state, assessment, text_utils)
|   |-- main.py              App entry point
|   |-- config.py            Environment/settings
|   |-- models/              SQLAlchemy and Pydantic models
|   |-- pipelines/           RAG and hybrid LLM pipelines
|   |-- routers/             API endpoints
|   |-- services/            Orchestration (chat_orchestrator.py), memory,
|   |                        strategy, concepts, tutor state machine,
|   |                        error analyzer, privacy boundary, remediation
|   |                        bootstrap
|   `-- tests/                Backend regression tests (see Testing below)
|-- course_docs/             Local/Moodle-synced course PDFs (gitignored;
|                             not published, see Configuration)
|-- frontend/                Standalone HTML chat UI
|-- moodle_plugin/           Moodle local plugin source (`local/acrla`)
|-- scripts/                 Utility scripts (document ingestion)
|-- tests/                   Root-level test/smoke-test scripts
|-- docs/                    Architecture and tutoring-flow documentation
|-- docker-compose.yml       Postgres + backend + Moodle demo stack
`-- README.md
```

Note: this public repository excludes generated research/evaluation
artifacts (see Research Context below) and local runtime data (`.env`,
`chroma_db/`, logs, `course_docs/`) via `.gitignore`. These remain on the
original development machine but are not part of the published history.

## Requirements

- Python 3.11 (see `backend/Dockerfile`)
- PostgreSQL 15+ (or the bundled `docker-compose.yml` Postgres service)
- ChromaDB (installed via `requirements.txt`; runs embedded, no separate server)
- Ollama, running locally, for the `nomic-embed-text` embedding model used
  by RAG ingestion/retrieval
- An LLM provider API key (Gemini by default; Groq/OpenAI/Cerebras are also
  supported) -- or a local Ollama chat model if running fully offline
- Moodle 4.3+ (the demo stack uses `bitnamilegacy/moodle:4.3`) for the
  Moodle plugin integration
- Docker + Docker Compose (optional, for the bundled Postgres/backend/Moodle stack)

## Installation

### Backend (local, without Docker)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Start PostgreSQL and Ollama yourself (or point `DATABASE_URL`/
`OLLAMA_BASE_URL` at existing instances), then see Configuration and
Running the Backend below.

### Full stack (Docker Compose)

```powershell
docker-compose up -d
```

This starts PostgreSQL, the backend, and a Moodle instance with the ACRLA
plugin volume-mounted at `/local/acrla`. Ollama is not included in
`docker-compose.yml` and must be run separately for embeddings.

## Configuration

Copy the example environment file and fill in real values (never commit the
result):

```powershell
cd backend
copy .env.example .env
```

Edit `backend/.env` and set at minimum `LLM_PROVIDER` and the matching
provider API key (e.g. `GEMINI_API_KEY`), plus `DATABASE_URL` if not using
the Docker Compose defaults. See the inline comments in
`backend/.env.example` for every supported variable, including the
conversation-agent mode switch (`ACRLA_AGENT_MODE`) and quota-saving/debug
flags.

## Running the Backend

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

Open:

- Health: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`
- Standalone chat UI: `http://localhost:8000`

Tables are created on startup from SQLAlchemy metadata; no separate
migration step is required.

To ingest course PDFs directly (outside of Moodle sync):

```powershell
python scripts/ingest_documents.py --course_id 1 --path ./course_docs/moodle_course_1
```

## Moodle Integration

1. Copy `moodle_plugin/acrla/` to Moodle's `/local/acrla`.
2. Visit Moodle Site Administration notifications to complete plugin installation.
3. Configure the backend URL in the plugin settings (default `http://localhost:8000`).

The plugin (`moodle_plugin/acrla/`) adds a floating ACRLA launcher, a
persistent right-side panel, dashboard/course-page mastery controls, and a
material sync button. It does not connect directly to Moodle's database --
all data flows through the ACRLA backend's API endpoints
(`/api/v1/moodle/sync`, `/api/v1/moodle/launch`,
`/api/v1/moodle/materials/sync`). See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full session/launch flow.

## Main API Endpoints

- `GET /health`
- `POST /api/v1/session/start`
- `POST /api/v1/chat`
- `GET /api/v1/mastery/student/{student_id}`
- `POST /api/v1/ingest/{course_id}`
- `POST /api/v1/moodle/sync`
- `GET /api/v1/moodle/launch`
- `POST /api/v1/moodle/materials/sync`
- `POST /api/v1/assessment/start`
- `POST /api/v1/assessment/submit`
- `GET /api/v1/rag/debug/{course_id}`
- `GET /api/v1/rag/collection/{course_id}`
- `POST /api/v1/rag/reset/{course_id}`
- `POST /api/v1/rag/rebuild/{course_id}`

## Testing

Backend regression tests live in `backend/tests/` and verify specific,
narrow claims about production code behavior (privacy boundary,
RAG-grounding fixes, QPC question validity, guided-practice evaluation
grounding, tutor-state/routing fixes). They mock the LLM constructor with a
real LangChain `RunnableLambda` so prompt composition runs exactly as in
production, and make no live LLM-provider calls. Each file is a standalone
script (not pytest-collected); see `backend/tests/README.md` for the full
per-file description and running convention.

```powershell
cd backend
$env:PYTHONPATH = "."
python tests/test_privacy_context.py
python tests/test_institutional_privacy.py
python tests/test_compact_value_depth_fix.py
python tests/test_qpc_question_validity.py
python tests/test_guided_practice_evaluation.py
python tests/test_qpc_dynamic_generation_robustness.py
python tests/test_qpc_dynamic_privacy_gateway.py
python tests/test_chapter_scope_concept_lock_fix.py
python tests/test_tutor_continuation_classification_fix.py
python tests/test_remediation_bootstrap_message_priority_fix.py
```

Exit code `0` means every check in that file passed.

`tests/evaluation_scenarios.py` (repository root) is an optional, manual
black-box smoke test: it starts a session and sends a handful of scripted
messages to a **running** backend over HTTP to check routing/intent
behavior (weak-student phrasing, analytics intent, preference updates,
navigation, engagement recovery). Unlike `backend/tests/`, it requires a
live backend with a configured LLM provider key -- it is not part of the
mocked regression suite and is not run in CI-style checks.

```powershell
python tests/evaluation_scenarios.py --backend http://localhost:8000
```

See also [docs/DEMO_TESTING.md](docs/DEMO_TESTING.md) for a manual
end-to-end checklist covering routing, mastery, and course-scope isolation
through the actual UI.

## Privacy / Security Notes

`backend/services/privacy_context.py` is the single, centralized boundary
between local student/course context and any external LLM call:

- Student identity (name, email, Moodle id, database id, session id) is
  never read from the context passed to an external-LLM prompt builder --
  an explicit allow-list, not a block-list, so a new context field is
  invisible to external prompts until someone deliberately adds it.
- Retrieved course-document chunks carry an institution-assigned
  `PUBLIC`/`INTERNAL`/`RESTRICTED` sensitivity level. `RESTRICTED` content
  is excluded entirely from any external-bound prompt; `PUBLIC`/`INTERNAL`
  content is size-capped.
  Every such decision is logged as a privacy-safe audit line (counts and
  sensitivity levels only -- never chunk text, file paths, or student
  identifiers).

This module governs outbound LLM prompt construction only. It is not an
authentication, retention/consent, or transport-encryption control, and it
does not redact free text a student chooses to type themselves. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#privacy-boundary) for the full
design.

## Research Context

ACRLA was developed as a research prototype exploring adaptive conversational
remediation for Moodle-based courses, combining retrieval-augmented tutoring
with mastery-driven, privacy-aware use of external LLMs. This repository
contains the software implementation; thesis evaluation data and results are
maintained separately and are not part of this public repository.
