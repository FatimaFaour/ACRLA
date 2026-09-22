@echo off
title ACRLA - Ingest Course Documents
color 0B

echo.
echo  ================================================
echo   ACRLA - Ingest Course Documents into RAG
echo  ================================================
echo.

set COURSE_ID=%~1
set DOCS_PATH=%~2

if "%COURSE_ID%"=="" (
    set /p COURSE_ID="Enter the Moodle course ID to ingest into: "
)
if "%DOCS_PATH%"=="" (
    set /p DOCS_PATH="Enter the path to a course document file or folder (PDF/TXT/MD): "
)

if not exist "%DOCS_PATH%" (
    echo.
    echo  Error: "%DOCS_PATH%" does not exist.
    echo  Usage: ingest_cs_docs.bat [course_id] [path-to-file-or-folder]
    exit /b 1
)

call backend\venv\Scripts\activate.bat

echo.
echo Ingesting documents from "%DOCS_PATH%" for course %COURSE_ID%...
python scripts\ingest_documents.py --course_id %COURSE_ID% --path "%DOCS_PATH%"

echo.
echo  ================================================
echo   Done.
echo  ================================================
echo.
pause
