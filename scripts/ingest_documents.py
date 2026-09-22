"""
CLI script to ingest course documents into ChromaDB.
Usage: python scripts/ingest_documents.py --course_id 1 --path ./docs/
"""

import argparse
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from pipelines.rag_pipeline import ingest_documents
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Ingest course documents into ACRLA ChromaDB")
    parser.add_argument("--course_id", type=int, required=True, help="Moodle course ID")
    parser.add_argument("--path", type=str, required=True, help="Path to file or directory")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: {target} does not exist")
        sys.exit(1)

    # Collect files
    if target.is_dir():
        file_paths = [
            str(f) for f in target.rglob("*")
            if f.suffix.lower() in (".pdf", ".txt", ".md")
        ]
    else:
        file_paths = [str(target)]

    if not file_paths:
        print("No PDF/TXT/MD files found.")
        sys.exit(1)

    print(f"Ingesting {len(file_paths)} file(s) for course {args.course_id}...")
    result = ingest_documents(args.course_id, file_paths)

    print(f"\nDone!")
    print(f"  Files processed : {len(result['files_processed'])}")
    print(f"  Chunks created  : {result['chunks_created']}")
    for f in result["files_processed"]:
        print(f"  ✓ {f}")


if __name__ == "__main__":
    main()
