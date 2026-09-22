# ACRLA Windows Setup — Step by Step

## What you need
- Windows 10 or 11
- Python 3.11+ → https://www.python.org/downloads/
- PostgreSQL 15 → https://www.postgresql.org/download/windows/
- Ollama → https://ollama.com/download

---

## Step 1 — Install Ollama
1. Go to https://ollama.com/download
2. Download the Windows installer and run it
3. Ollama installs as a background service — you'll see it in your system tray
4. Open Command Prompt and run:
```
ollama pull phi3:mini
ollama pull nomic-embed-text
```
phi3:mini is ~2.3GB, nomic-embed-text is ~270MB. Download once, use forever.

---

## Step 2 — Install PostgreSQL
1. Download from https://www.postgresql.org/download/windows/
2. Run installer. When asked for a password, use: `password`
3. Keep default port: 5432
4. After install, open pgAdmin or Command Prompt and run:
```
psql -U postgres -c "CREATE DATABASE acrla;"
```
Or use pgAdmin: right-click Databases → Create → Database → name it `acrla`

---

## Step 3 — Install Python dependencies
Open Command Prompt in the `acrla` folder:
```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

---

## Step 4 — Start ACRLA
Double-click `start.bat`

It will:
- Check Ollama is running
- Verify phi3:mini is downloaded
- Start the FastAPI backend on http://localhost:8000

---

## Step 5 — Load your course documents

Run `ingest_cs_docs.bat` from a command prompt, passing a course ID and the
path to a PDF/TXT/MD file or a folder of them:

```
ingest_cs_docs.bat 1 C:\path\to\your\course\pdfs
```

Or double-click it and enter the course ID and path when prompted. This
loads the given file(s) into ChromaDB so ACRLA can answer questions grounded
in that material. The example questions below assume you've ingested course
material covering recursion, sorting, pointers, and binary trees (e.g. the
sample chapters under `course_docs/`, if you have them) -- substitute your
own topics/questions if you ingested different material.

---

## Step 6 — Open the chatbot
Open `frontend\index.html` in your browser (Chrome or Edge).

Settings:
- Backend URL: http://localhost:8000
- Student ID: 1
- Course ID: 1
- Course Name: Introduction to Computer Science
- Weak Concepts: recursion, pointers

Click **Start Session** and start chatting!

---

## Example questions to test ACRLA

**Recursion:**
- "What is the base case in recursion?"
- "Explain how the call stack works with factorial"
- "What is the difference between recursion and iteration?"

**Sorting:**
- "What is the time complexity of merge sort?"
- "When should I use quick sort vs merge sort?"
- "Explain bubble sort step by step"

**Pointers:**
- "What is a dangling pointer?"
- "What is the difference between stack and heap?"
- "How do I avoid memory leaks in C?"

**Binary Trees:**
- "How does BST search work?"
- "What are the three tree traversal methods?"
- "Why do we need balanced BSTs?"

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ollama: command not found` | Restart Command Prompt after installing Ollama |
| Backend crashes on start | Make sure PostgreSQL is running and `acrla` database exists |
| ChromaDB error | Delete the `backend/chroma_db/` folder and re-run ingest |
| Slow responses | Normal for CPU — phi3:mini takes 5-15 seconds per response on CPU |
| Port 8000 in use | Edit start.bat and change `--port 8000` to `--port 8001`, update frontend too |
