# ACRLA — Quick Start Guide

## Prerequisites
- Python 3.11+
- PostgreSQL 15+ (or Docker)
- An LLM provider API key (Gemini by default; Groq/OpenAI/Cerebras are also
  supported, or use Ollama locally with no key -- see `backend/.env.example`)

---

## Option A: Run with Docker (recommended, easiest)

```bash
# 1. Clone/unzip the project
cd acrla

# 2. Set your LLM provider key
cp backend/.env.example backend/.env
# Edit backend/.env and set LLM_PROVIDER (default "gemini") and the matching
# API key, e.g. GEMINI_API_KEY=your-gemini-key-here

# 3. Start everything (Postgres + Backend + Moodle)
docker-compose up

# Wait ~2 minutes for Moodle to initialize on first run.
# Backend: http://localhost:8000
# Moodle:  http://localhost:8080  (admin / admin123)
# API docs: http://localhost:8000/docs
```

---

## Option B: Run backend only (no Docker)

### Step 1 — PostgreSQL
```bash
# Mac
brew install postgresql@15 && brew services start postgresql@15
createdb acrla

# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo -u postgres createdb acrla
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'password';"

# Windows — download from https://www.postgresql.org/download/windows/
```

### Step 2 — Python environment
```bash
cd acrla/backend
python3 -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3 — Environment variables
```bash
cp .env.example .env
```
Edit `.env`:
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key-here
DATABASE_URL=postgresql://postgres:password@localhost:5432/acrla
CHROMA_PATH=./chroma_db
SECRET_KEY=any-random-string-here
```

### Step 4 — Start the backend
```bash
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 5 — Verify it works
Open http://localhost:8000/docs — you'll see the interactive API.

Or run:
```bash
curl http://localhost:8000/health
# {"status":"healthy"}
```

---

## Step 6 — Upload course documents (PDFs)

```bash
# From the acrla/ root
python scripts/ingest_documents.py --course_id 1 --path ./your-course-pdfs/
```

Or via the API at http://localhost:8000/docs → POST /ingest/{course_id}

---

## Step 7 — Test the chatbot (no Moodle needed)

Open `frontend/index.html` in your browser.
- Backend URL: http://localhost:8000
- Set any Student ID and Course ID
- Click "Start Session"

---

## Step 8 — Install Moodle plugin (optional)

1. Copy `moodle_plugin/acrla/` into your Moodle's `/local/` folder
2. Go to: Site Admin → Notifications → install
3. Go to: Site Admin → Plugins → Local plugins → ACRLA
4. Set Backend URL to: `http://localhost:8000`

---

## Run evaluation tests

```bash
# With backend running:
cd acrla
python tests/evaluation_scenarios.py --backend http://localhost:8000
```

Expected output: 5/5 scenarios pass (requires course docs uploaded for tutoring scenarios).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `psycopg2` install fails | `pip install psycopg2-binary` (not psycopg2) |
| ChromaDB error on start | Delete `./chroma_db/` folder and restart |
| LLM provider auth error (401/403) | Check `LLM_PROVIDER` and the matching API key in `.env` |
| Port 8000 in use | `uvicorn main:app --port 8001` |
| Moodle plugin not showing | Check Moodle version ≥ 4.0, clear Moodle cache |
