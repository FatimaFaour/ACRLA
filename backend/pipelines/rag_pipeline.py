"""
Internal RAG pipeline.
Phase 1 — Knowledge preparation: chunk → embed → store in ChromaDB
Phase 2 — Student response: retrieve → assemble context → LLM response
"""

from pathlib import Path
from typing import Optional
from functools import lru_cache
import json
import re

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate

from config import get_settings
from services.llm_factory import get_llm, get_embeddings
from services.course_concepts import ALLOWED_CONCEPTS, canonicalize_concept, concept_source_file, sub_concepts_for
from tools.text_utils import normalize_key

settings = get_settings()


# ==========================================================
# File Purpose
# ==========================================================
# Implements ACRLA's internal course-material RAG path. This file owns PDF
# ingestion, ChromaDB collection access, course/chapter source filtering,
# retrieval diagnostics, and prompt construction for course-grounded answers.


# ==========================================================
# Vectorstore Access
# ==========================================================


def _collection_name(course_id: int) -> str:
    return f"course_{course_id}"


@lru_cache(maxsize=32)
def get_vectorstore(course_id: int) -> Chroma:
    """Return the Chroma collection for one Moodle/ACRLA course.

    Already cached for the process lifetime (one `Chroma` object per
    `course_id`, via `@lru_cache`) -- investigated as a possible source of
    repeated `ClientStartEvent`/`ClientCreateCollectionEvent` chromadb
    telemetry noise per request, but that noise is fully explained by
    `retrieve_context_for_scope` previously calling this once per COURSE in
    the allowed scope even when a resolved concept only lives in one of
    them (see its own docstring and `_concept_owning_course_ids`) -- fewer
    distinct `course_id`s touched per turn means fewer first-time
    constructions here, with no change needed to this caching itself.
    Invalidated by `reset_course_collection`'s `get_vectorstore.cache_clear()`
    when material is resynced/deleted, so a stale cached client is never
    reused after a collection changes.
    """
    return Chroma(
        collection_name=_collection_name(course_id),
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path,
    )


# ── PDF loader ────────────────────────────────────────────────────────────────

def _retrieval_source_name(doc: Document) -> str:
    return doc.metadata.get("source_file") or doc.metadata.get("source") or ""


# Moodle teachers may name a resource differently from the uploaded PDF file.
# Filtering can still use source_file/document_origin, but the user-facing UI
# should prefer display_title so citations match what the teacher sees.
def _display_source_name(doc: Document) -> str:
    origin = doc.metadata.get("document_origin")
    if origin and str(origin).strip().lower() in {"moodle", "bundled", "unknown"}:
        origin = ""
    return doc.metadata.get("display_title") or origin or _retrieval_source_name(doc)


def _normalize_source_label(label: str | None) -> str:
    text = str(label or "").strip()
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    return text


def _dedupe_source_labels(labels: list[str] | set[str]) -> list[str]:
    deduped: dict[str, str] = {}
    for label in labels:
        normalized = _normalize_source_label(label)
        if normalized:
            deduped.setdefault(normalized.lower(), normalized)
    return sorted(deduped.values())


def _log_retrieval(
    query: str,
    concept: str | None,
    collection_name: str,
    docs: list[Document],
    count_before: int | str,
) -> None:
    """Print the exact retrieval decision used for debugging/demo validation.

    These logs make RAG behavior explainable: query, concept filter, collection,
    retrieved chunks, raw files, and display labels are all visible.
    """
    files = sorted({_retrieval_source_name(doc) for doc in docs if _retrieval_source_name(doc)})
    display_sources = _dedupe_source_labels({_display_source_name(doc) for doc in docs if _display_source_name(doc)})
    print(
        "[RAG] retrieval "
        f"query={query!r} "
        f"concept={concept or 'none'} "
        f"collection={collection_name} "
        f"collection_count={count_before} "
        f"retrieved_chunk_count={len(docs)} "
        f"retrieved_files={files} "
        f"display_sources={display_sources}"
    )


def reset_course_collection(course_id: int) -> dict:
    """Delete a persisted Chroma course collection.

    Used by diagnostics/rebuild tooling when source PDFs have changed and old
    chunks must not remain visible to retrieval.
    """
    collection_name = _collection_name(course_id)
    get_vectorstore.cache_clear()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path,
    )
    try:
        deleted_chunks = vectorstore._collection.count()
    except Exception:
        deleted_chunks = 0
    try:
        vectorstore._client.delete_collection(collection_name)
    except Exception as exc:
        message = str(exc).lower()
        if "does not exist" not in message and "not found" not in message:
            raise
    get_vectorstore.cache_clear()
    print(f"[RAG] reset collection={collection_name} deleted_chunks={deleted_chunks}")
    return {"collection": collection_name, "deleted_chunks": deleted_chunks}


def rebuild_course_collection(course_id: int, document_dir: str | None = None) -> dict:
    """Recreate a Chroma course collection from documents currently on disk.

    This supports the demo scenario where Moodle PDFs are resynced and stale
    chunks must be removed before validating source relevance.
    """
    reset = reset_course_collection(course_id)
    docs_root = Path(document_dir) if document_dir else _default_course_docs_dir(course_id)
    if not docs_root.exists():
        return {
            "course_id": course_id,
            "collection": reset["collection"],
            "document_dir": str(docs_root),
            "files_ingested": [],
            "chunks_created": 0,
            "deleted_chunks": reset["deleted_chunks"],
        }

    file_paths = _document_paths(docs_root)
    processed: list[str] = []
    chunks_created = 0
    manifest = _load_material_manifest(docs_root)
    for path in file_paths:
        filename = Path(path).name
        metadata = manifest.get(filename, {})
        result = ingest_documents(
            course_id,
            [path],
            concept=metadata.get("concept") or _infer_concept_from_path(path),
            display_title=metadata.get("display_title"),
            original_file_name=metadata.get("original_file_name") or filename,
            # RQ2 institutional privacy: a teacher/institution classifies a
            # document by adding "sensitivity": "PUBLIC"|"INTERNAL"|
            # "RESTRICTED" next to that file's entry in the course's own
            # materials_manifest.json (the SAME existing, already
            # teacher/institution-editable file _load_material_manifest
            # already reads concept/display_title from -- no new config
            # surface introduced). Absent/unrecognized -> normalize_sensitivity
            # defaults to INTERNAL, never PUBLIC.
            sensitivity=metadata.get("sensitivity"),
        )
        processed.extend(result.get("files_processed", []))
        chunks_created += int(result.get("chunks_created", 0))
    print(
        "[RAG] rebuild "
        f"collection={reset['collection']} "
        f"document_dir={docs_root} "
        f"files={processed} "
        f"chunks_created={chunks_created}"
    )
    return {
        "course_id": course_id,
        "collection": reset["collection"],
        "document_dir": str(docs_root),
        "files_ingested": processed,
        "chunks_created": chunks_created,
        "deleted_chunks": reset["deleted_chunks"],
    }


def _default_course_docs_dir(course_id: int) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    moodle_dir = project_root / "course_docs" / f"moodle_course_{course_id}"
    return moodle_dir if moodle_dir.exists() else project_root / "course_docs"


def _document_paths(docs_root: Path) -> list[str]:
    project_docs = Path(__file__).resolve().parents[2] / "course_docs"
    iterator = docs_root.glob("*") if docs_root == project_docs else docs_root.rglob("*")
    return [
        str(path)
        for path in sorted(iterator)
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
    ]


def _load_material_manifest(docs_root: Path) -> dict:
    manifest_path = docs_root / "materials_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            data = json.load(manifest_file)
    except Exception as exc:
        print(f"[RAG] Failed to read material manifest {manifest_path}: {exc}")
        return {}
    files = data.get("files", data if isinstance(data, dict) else {})
    return files if isinstance(files, dict) else {}


def _infer_concept_from_path(path: str) -> str | None:
    return canonicalize_concept(Path(path).stem.replace("_", " "))


def _document_origin(path: str) -> str:
    return Path(path).name


def _load_pdf(path: str) -> list[Document]:
    """Load one PDF into LangChain Documents with page/source metadata."""
    docs = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page": i + 1, "source": Path(path).name}
                ))
    return docs


# ── Phase 1: Ingest documents ────────────────────────────────────────────────

def ingest_documents(
    course_id: int,
    file_paths: list[str],
    concept: str | None = None,
    display_title: str | None = None,
    original_file_name: str | None = None,
    sensitivity: str | None = None,
) -> dict:
    """Ingest PDFs into the course-specific Chroma collection.

    Existing chunks from the same source file are deleted first so Moodle PDF
    resyncs replace stale material instead of duplicating it.

    `sensitivity` (RQ2 institutional-privacy step): the institution's
    PUBLIC/INTERNAL/RESTRICTED classification for this document, normalized
    via services.privacy_context.normalize_sensitivity and stamped onto
    EVERY chunk's metadata (so it survives ingestion -> chunking -> ChromaDB
    storage -> retrieval unchanged, the same way concept/source_file
    already do). Missing/unrecognized values normalize to INTERNAL, the
    documented safe default -- never silently PUBLIC.
    """
    from services.privacy_context import normalize_sensitivity
    normalized_sensitivity = normalize_sensitivity(sensitivity)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )

    all_docs: list[Document] = []
    processed = []
    vectorstore = get_vectorstore(course_id)

    for path in file_paths:
        ext = Path(path).suffix.lower()
        source_file = Path(path).name
        try:
            try:
                vectorstore._collection.delete(where={"source_file": source_file})
            except Exception as delete_exc:
                print(f"[RAG] Could not clear old chunks for {source_file}: {delete_exc}")
            if ext == ".pdf":
                raw_docs = _load_pdf(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                raw_docs = [Document(page_content=text, metadata={"source": source_file})]

            print(f"[RAG] Loaded {len(raw_docs)} pages from {source_file}")
            chunks = splitter.split_documents(raw_docs)
            for chunk in chunks:
                chunk.metadata["course_id"] = str(course_id)
                chunk.metadata["source_file"] = source_file
                chunk.metadata["document_origin"] = original_file_name or _document_origin(path)
                chunk.metadata["sensitivity"] = normalized_sensitivity
                if display_title:
                    chunk.metadata["display_title"] = display_title
                if concept:
                    chunk.metadata["concept"] = concept
            all_docs.extend(chunks)
            processed.append(source_file)
            print(f"[RAG] Created {len(chunks)} chunks")
        except Exception as e:
            print(f"[RAG] Failed to load {path}: {e}")

    if all_docs:
        vectorstore.add_documents(all_docs)
        print(f"[RAG] Stored {len(all_docs)} total chunks in ChromaDB")

    return {"course_id": course_id, "chunks_created": len(all_docs), "files_processed": processed}


# ── Phase 2: Response generation ─────────────────────────────────────────────

# ==========================================================
# Internal RAG Prompt
# ==========================================================
# The prompt receives backend-controlled scope, difficulty, strategy, current
# topic, and retrieved course context. The LLM writes the answer, but the
# backend decides what material and concepts it is allowed to see.

SYSTEM_PROMPT = """You are ACRLA, an adaptive course tutor embedded in Moodle.

Weak concepts: {weak_concepts}
Current difficulty: {difficulty}
Required question format for this difficulty: {question_format}
Teaching strategy: {strategy}
Session goal: {session_goal}
Selected response route: {mode}
Allowed course concepts: {allowed_concepts}
Focus topic: {selected_concept}
Requested concepts: {requested_concepts}
Target sub-concepts for this topic: {sub_concepts}
Remediation scope:
{remediation_scope}
The student is currently learning about: {current_topic}. Stay focused on this topic unless the student clearly asks to switch.

{difficulty_instructions}

{strategy_instructions}

AUTOMATIC ROUTING RULE:
- ACRLA automatically searches course materials first. This response was routed to the course-grounded path, so use only the retrieved course context below.
- Do not claim the student selected an internal or external mode.
- If the student asks whether you use internal or external mode, explain that ACRLA automatically checks course materials first and falls back externally only when course context is not relevant.
Course scope guardrail:
- This is a course tutoring assistant for these allowed concepts only: {allowed_concepts}.
- Course-grounded answers must stay inside the retrieved course materials and the allowed concepts.
- Do not answer non-CS or unrelated questions, even if some retrieved text appears loosely related.
- Do not force analogies from unrelated topics back into the course.
- If the student asks about something clearly outside the course, reply: "That's outside the scope of this course. Let's get back to {current_topic} - want to continue there?"

Course materials:
{context}

Conversation so far:
{history}

Rules:
- Base answers on the course materials above
- Only mention course concepts from the allowed course concepts list
- If Requested concepts has more than one concept, every requested concept must appear in the answer. Do not explain only the first concept.
- For multi-concept requests, generate synthesis questions or explanations that connect all requested concepts together.
- Multi-concept questions must use a concrete mini-scenario with actual data, code, graph/set notation, or algorithm context.
- Avoid vague phrases like "best connects", "frame the condition", "organize related items", or "include X in the final reasoning".
- When useful, diagnose or practice one of the target sub-concepts instead of changing chapters
- Refuse or redirect unrelated non-course questions instead of answering them
- Do not mention backend state, orchestration fields, or unresolved focus-topic values
- Never invent concept names, mastery scores, chapters, or source files
- Match the student's level and strategy
- After every explanation, suggest ONE practice question like: "Want to try a question on this?"
- If the student says yes or asks to be tested, generate a question at {difficulty} level
- Easy questions must be multiple choice only
- Moderate questions must be open-ended explanation questions only
- Hard questions must be code-writing or algorithm-design questions only
- Show mastery progress like: "Your understanding of pointers is improving!"
- Keep answers focused and under 300 words"""

INTERNAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def retrieve_context(
    course_id: int,
    query: str,
    k: Optional[int] = None,
    selected_concept: str | None = None,
    *,
    for_external: bool = False,
    audit: dict | None = None,
) -> tuple[str, list[str]]:
    """Retrieve text chunks and displayable source labels for one course.

    When a selected concept is provided, retrieval is filtered by concept/source
    metadata so chapter remediation does not cite unrelated course material.

    `for_external` (RQ2 institutional-privacy step, default False = exactly
    the pre-existing behavior for every caller that doesn't pass it): when
    True, retrieved chunks are passed through
    services.privacy_context.filter_course_chunks_for_external before being
    joined into `context` -- RESTRICTED chunks are dropped entirely, and the
    remaining PUBLIC/INTERNAL chunks are capped to a configurable total
    character budget (whole chunks only, never mid-text truncation). This
    is the deterministic gateway between local retrieval and an external
    LLM prompt; it is never itself an LLM call. `audit` (optional, filled
    in place) receives the gateway's own privacy-safe counts (see
    filter_course_chunks_for_external's own docstring for exactly what it
    contains -- never chunk text).
    """
    k = k or min(settings.max_retrieval_chunks, 2)
    source_file = concept_source_file(selected_concept)
    concept = canonicalize_concept(selected_concept) or (
        re.sub(r"\s+", " ", str(selected_concept).strip()) if selected_concept else None
    )
    collection_name = _collection_name(course_id)
    count_before: int | str = "unknown"
    docs: list[Document] = []
    try:
        vectorstore = get_vectorstore(course_id)
        count_before = vectorstore._collection.count()
        if count_before == 0:
            _log_retrieval(query, concept, collection_name, [], count_before)
            return "", []
        # Embed the query ONCE and reuse the same vector for every filtered
        # similarity_search below -- `similarity_search(query, ...)` embeds
        # `query` fresh internally on every call (confirmed by reading
        # langchain_community.vectorstores.chroma.Chroma.similarity_search_with_score),
        # so up to three calls for the same query/course (concept filter,
        # source_file filter, source fallback) were re-embedding identical
        # text three times. similarity_search_by_vector(embedding, ...) takes
        # the same path after that point (empirically verified to return
        # identical documents given the same embedding), so ranking/results
        # are unchanged -- only the redundant embedding calls are removed.
        embed_fn = getattr(vectorstore, "_embedding_function", None)
        query_embedding = embed_fn.embed_query(query) if embed_fn is not None else None
        if concept:
            if query_embedding is not None:
                docs.extend(vectorstore.similarity_search_by_vector(query_embedding, k=k, filter={"concept": concept}))
            else:
                docs.extend(vectorstore.similarity_search(query, k=k, filter={"concept": concept}))
            source_docs = []
            if source_file:
                if query_embedding is not None:
                    source_docs = vectorstore.similarity_search_by_vector(query_embedding, k=k, filter={"source_file": source_file})
                    if not source_docs:
                        source_docs = vectorstore.similarity_search_by_vector(query_embedding, k=k, filter={"source": source_file})
                else:
                    source_docs = vectorstore.similarity_search(query, k=k, filter={"source_file": source_file})
                    if not source_docs:
                        source_docs = vectorstore.similarity_search(query, k=k, filter={"source": source_file})
            seen_sources = {
                (_retrieval_source_name(doc), doc.metadata.get("page"), doc.page_content[:80])
                for doc in docs
            }
            for doc in source_docs:
                key = (_retrieval_source_name(doc), doc.metadata.get("page"), doc.page_content[:80])
                if key not in seen_sources:
                    docs.append(doc)
                    seen_sources.add(key)
        elif query_embedding is not None:
            docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)
        else:
            docs = vectorstore.similarity_search(query, k=k)
    except Exception as exc:
        print(
            "[RAG] retrieval_error "
            f"query={query!r} "
            f"concept={concept or 'none'} "
            f"collection={collection_name} "
            f"error={exc}"
        )
        return "", []

    _log_retrieval(query, concept, collection_name, docs, count_before)

    if for_external:
        # RQ2 institutional-privacy gateway -- applied here, where per-chunk
        # `sensitivity` metadata is still available, before it collapses
        # into one joined string. Deterministic; never an LLM call.
        from services.privacy_context import filter_course_chunks_for_external
        candidate_chunks = [
            {"text": d.page_content[:900], "sensitivity": d.metadata.get("sensitivity"), "source": _display_source_name(d) or "doc"}
            for d in docs
        ]
        allowed_chunks, gateway_audit = filter_course_chunks_for_external(candidate_chunks)
        if audit is not None:
            audit.update(gateway_audit)
        context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in allowed_chunks)
        sources = sorted({c["source"] for c in allowed_chunks if c["source"] and c["source"] != "doc"})
        return context, _dedupe_source_labels(sources)

    context = "\n\n---\n\n".join(
        f"[{_display_source_name(d) or 'doc'}]\n{d.page_content[:900]}" for d in docs
    )
    sources = sorted({
        _display_source_name(d)
        for d in docs
        if _display_source_name(d)
    })
    return context, _dedupe_source_labels(sources)


def _concept_owning_course_ids(concept: str, canonical_courses: list[dict] | None) -> list[int] | None:
    """Moodle course IDs the concept/material manifest says teach `concept`,
    or None if ownership cannot be determined from the manifest at all (no
    manifest given, or no course's concept list contains it) -- callers must
    treat None as "unresolvable" and fall back to searching every allowed
    course, never narrowing to nothing just because a concept happens to be
    missing from the manifest. `canonical_courses` is the same
    `context["canonical_courses"]` shape `tools.mastery_tools` already uses
    for the identical "which real course teaches this concept" question
    (db_course_id/moodle_course_id/name/concepts per course) -- built at
    request time from the live DB + materials manifest, never a hardcoded
    concept list, so this stays correct for any course/concept set.
    """
    if not canonical_courses or not concept:
        return None
    key = normalize_key(concept)
    owning: list[int] = []
    for course in canonical_courses:
        moodle_id = course.get("moodle_course_id")
        if moodle_id is None:
            continue
        if any(normalize_key(c) == key for c in course.get("concepts") or []):
            if moodle_id not in owning:
                owning.append(moodle_id)
    return owning or None


def retrieve_context_for_scope(
    course_ids: list[int] | tuple[int, ...],
    query: str,
    k: Optional[int] = None,
    selected_concept: str | None = None,
    requested_concepts: list[str] | tuple[str, ...] | None = None,
    scope: str = "chapter",
    canonical_courses: list[dict] | None = None,
    *,
    for_external: bool = False,
    audit: dict | None = None,
) -> tuple[str, list[str]]:
    """Retrieve context across the allowed remediation scope.

    Chapter scope filters to one concept, course scope stays inside the selected
    course, and overall scope may span the provided list of Moodle course IDs.

    `canonical_courses` (optional, the concept/material manifest -- see
    `_concept_owning_course_ids`) lets a resolved concept prune the course
    fan-out to only the course(s) that actually teach it, instead of
    searching every allowed course's Chroma collection for a concept that
    (per the manifest) only ever lives in one of them. This never widens
    the search beyond `course_ids` (the caller's own authorized scope) --
    only narrows it -- and falls back to the original "search every allowed
    course" behavior whenever ownership can't be resolved (no manifest
    passed, concept missing from every course's list, or the owning
    course(s) happen to fall outside the allowed `course_ids` for this
    scope), so a chapter/course-scoped turn or an unresolvable concept
    behaves exactly as before.

    `for_external`/`audit` (RQ2 institutional-privacy step, both default to
    the pre-existing behavior when omitted): forwarded unchanged to each
    underlying `retrieve_context` call -- see that function's own docstring.
    `audit`, if given a dict, ends up holding the SUM across every
    concept/course call this scope made (services.privacy_context.
    merge_external_content_audit), so a multi-concept comparison's
    institutional-privacy accounting reflects the whole turn, not just one
    concept.
    """
    ids = [int(course_id) for course_id in course_ids if course_id is not None]
    if not ids:
        return "", []
    allowed_id_set = set(ids)

    scope = str(scope or "chapter").lower()
    requested: list[str] = []
    for concept in requested_concepts or []:
        canonical = canonicalize_concept(concept) or re.sub(r"\s+", " ", str(concept).strip())
        if canonical and canonical not in requested:
            requested.append(canonical)
    if scope == "chapter" and not requested:
        canonical = canonicalize_concept(selected_concept) or (
            re.sub(r"\s+", " ", str(selected_concept).strip()) if selected_concept else None
        )
        if canonical:
            requested = [canonical]

    # Precomputed once per concept (not once per course-x-concept pair) --
    # None means "search every allowed course for this concept" (unresolved
    # ownership, the safe fallback); a list means "only search these
    # allowed courses for this concept".
    pruned_course_ids_by_concept: dict[str, list[int] | None] = {}
    for concept in requested:
        owning = _concept_owning_course_ids(concept, canonical_courses)
        if owning is None:
            pruned_course_ids_by_concept[concept] = None
            continue
        pruned = [cid for cid in owning if cid in allowed_id_set]
        # Empty after intersecting with the allowed scope means the manifest's
        # answer doesn't apply within this turn's authorized courses (e.g. a
        # chapter-scoped turn asking about a concept owned by a different
        # course) -- never expand beyond `ids`, but also never search zero
        # courses for a resolvable concept; fall back to the full allowed set.
        pruned_course_ids_by_concept[concept] = pruned or None

    all_contexts: list[str] = []
    all_sources: set[str] = set()
    for course_id in ids:
        if requested:
            for concept in requested:
                pruned = pruned_course_ids_by_concept.get(concept)
                if pruned is not None and course_id not in pruned:
                    continue
                call_audit = {} if for_external and audit is not None else None
                context, sources = retrieve_context(
                    course_id, query, k=k, selected_concept=concept,
                    for_external=for_external, audit=call_audit,
                )
                if audit is not None and call_audit:
                    from services.privacy_context import merge_external_content_audit
                    merge_external_content_audit(audit, call_audit)
                if context:
                    all_contexts.append(context)
                all_sources.update(sources)
        else:
            call_audit = {} if for_external and audit is not None else None
            context, sources = retrieve_context(
                course_id, query, k=k, selected_concept=None,
                for_external=for_external, audit=call_audit,
            )
            if audit is not None and call_audit:
                from services.privacy_context import merge_external_content_audit
                merge_external_content_audit(audit, call_audit)
            if context:
                all_contexts.append(context)
            all_sources.update(sources)

    return "\n\n=== COURSE BOUNDARY ===\n\n".join(all_contexts), _dedupe_source_labels(all_sources)


def retrieve_scored_context_for_scope(
    course_ids: list[int] | tuple[int, ...],
    query: str,
    k: Optional[int] = None,
    requested_concepts: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """Raw-message retrieval probe with scores for automatic routing.

    Unlike retrieve_context_for_scope, this never substitutes the previous
    selected concept. It only filters by concepts explicitly found in the
    current user message.

    Returned scores are treated by the orchestrator as distance-like values:
    lower is better. This function does not decide routing; it only exposes the
    evidence used by the relevance gate.
    """
    ids = [int(course_id) for course_id in course_ids if course_id is not None]
    if not ids:
        return []

    k = k or min(max(settings.max_retrieval_chunks, 3), 4)
    requested: list[str] = []
    for concept in requested_concepts or []:
        canonical = canonicalize_concept(concept) or re.sub(r"\s+", " ", str(concept).strip())
        if canonical and canonical not in requested:
            requested.append(canonical)

    results: list[dict] = []
    for course_id in ids:
        collection_name = _collection_name(course_id)
        try:
            vectorstore = get_vectorstore(course_id)
            count_before = vectorstore._collection.count()
            if count_before == 0:
                _log_retrieval(query, None, collection_name, [], count_before)
                continue
            pairs = []
            if requested:
                for concept in requested:
                    pairs.extend(vectorstore.similarity_search_with_score(query, k=k, filter={"concept": concept}))
            else:
                pairs = vectorstore.similarity_search_with_score(query, k=k)
            docs = [doc for doc, _score in pairs]
            _log_retrieval(query, ", ".join(requested) if requested else None, collection_name, docs, count_before)
            for doc, score in pairs:
                results.append({
                    "course_id": course_id,
                    "score": float(score),
                    "score_kind": "distance_lower_is_better",
                    "source": _display_source_name(doc),
                    "source_file": _retrieval_source_name(doc),
                    "concept": doc.metadata.get("concept"),
                    "text": doc.page_content or "",
                })
        except Exception as exc:
            print(
                "[RAG] scored_retrieval_error "
                f"query={query!r} "
                f"collection={collection_name} "
                f"error={exc}"
            )
    return sorted(results, key=lambda item: item["score"])


def generate_rag_response(course_id: int, question: str, student_context: dict) -> tuple[str, list[str]]:
    """Generate a course-grounded answer from retrieved internal material.

    The orchestrator supplies the selected concept, remediation level, routing
    scope, difficulty, strategy, and conversation history. This function should
    not decide mastery updates; it only returns response text and sources.
    """
    selected_concept = student_context.get("selected_concept")
    retrieval_query = student_context.get("retrieval_query") or question
    remediation_level = student_context.get("remediation_level", "chapter")
    retrieval_course_ids = student_context.get("retrieval_course_ids") or [course_id]
    # RQ2: this legacy fallback's answer is written by an external LLM
    # (currently Gemini), exactly like the primary agent path -- so its
    # retrieval must pass through the SAME institutional-privacy gateway
    # (services.privacy_context.filter_course_chunks_for_external, applied
    # inside retrieve_context when for_external=True): RESTRICTED chunks
    # excluded, PUBLIC/INTERNAL size-capped. Reuses the existing helper; no
    # new privacy logic here.
    external_audit: dict = {}
    context, sources = retrieve_context_for_scope(
        retrieval_course_ids,
        retrieval_query,
        selected_concept=selected_concept,
        requested_concepts=student_context.get("requested_concepts"),
        scope=remediation_level,
        for_external=True,
        audit=external_audit,
    )
    if external_audit:
        from services.privacy_context import log_external_content_decision
        log_external_content_decision(course_id=course_id, audit=external_audit)
    mode = student_context.get("mode", "internal")
    if mode == "internal" and not context:
        target = selected_concept if remediation_level == "chapter" and selected_concept else "this remediation scope"
        return (
            f"I don't have course material for {target} available in ACRLA yet.\n\n"
            "Please ask the teacher to sync or upload the relevant Moodle PDF for this concept, "
            "then I can explain it using the course materials.",
            [],
        )

    # Build conversation history string
    from services.memory_manager import get_buffer
    session_id = student_context.get("session_id", "")
    buffer = get_buffer(session_id)
    history_msgs = buffer.last_n(6)  # last 6 messages
    history = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'ACRLA'}: {m['content'][:120]}"
        for m in history_msgs
    ) if history_msgs else "This is the start of the conversation."

    route_rule = (
        "ACRLA no longer uses a manually selected mode. This answer was routed to course-grounded retrieval because the course materials were relevant."
    )

    chain = INTERNAL_PROMPT | get_llm(temperature=0.3, max_tokens=256)

    response = chain.invoke({
        # RQ2: no system-added student identity (real Moodle full name) and
        # no student mastery percentage in an external-LLM prompt -- the
        # SYSTEM_PROMPT no longer has {student_name}/{mastery_level} fields,
        # matching the primary agent path's build_llm_safe_student_context
        # allow-list. weak-concept NAMES only (same as the primary path's
        # external-knowledge tool).
        "weak_concepts": ", ".join(student_context.get("weak_concepts", [])) or "none yet",
        "difficulty": student_context.get("difficulty", "medium"),
        "question_format": student_context.get("question_format", "open-ended explanation question"),
        "difficulty_instructions": student_context.get("difficulty_instructions", ""),
        "strategy": student_context.get("strategy", "moderate"),
        "strategy_instructions": student_context.get("strategy_instructions", ""),
        "remediation_scope": student_context.get("remediation_scope", ""),
        "session_goal": student_context.get("session_goal", "general revision"),
        "mode": mode,
        "allowed_concepts": ", ".join(student_context.get("available_concepts") or ALLOWED_CONCEPTS),
        "selected_concept": selected_concept or "the student's current course question",
        "requested_concepts": ", ".join(student_context.get("requested_concepts") or []) or "none",
        "sub_concepts": ", ".join(student_context.get("sub_concepts") or sub_concepts_for(selected_concept)) or "none",
        "current_topic": student_context.get("current_topic") or selected_concept or "not set yet",
        "context": context or "Retrieved course context is unavailable for this turn.",
        "history": history,
        "question": f"[AUTOMATIC ROUTE: COURSE_GROUNDED] {route_rule}\n\nStudent question: {question}",
    })

    return response.content, sources
