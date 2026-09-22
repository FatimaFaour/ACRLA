import re


COURSE_CONCEPTS = {
    "Recursion": {
        "chapter": 1,
        "file": "chapter1_recursion.pdf",
        "aliases": [
            "chapter 1",
            "chapter1",
            "chapter_1",
            "recursion",
            "recursive",
            "base case",
            "recursive case",
        ],
    },
    "Sorting Algorithms": {
        "chapter": 2,
        "file": "chapter2_sorting.pdf",
        "aliases": [
            "chapter 2",
            "chapter2",
            "chapter_2",
            "sorting",
            "sorting algorithms",
            "sort",
            "quick sort",
            "quicksort",
            "merge sort",
            "bubble sort",
            "selection sort",
            "pivot",
            "partition",
        ],
    },
    "Pointers and Memory Management": {
        "chapter": 3,
        "file": "chapter3_pointers_memory.pdf",
        "aliases": [
            "chapter 3",
            "chapter3",
            "chapter_3",
            "pointers",
            "pointer",
            "memory",
            "memory management",
            "pointers and memory",
            "pointers and memory management",
            "address",
            "dereference",
            "dereferencing",
            "null pointer",
        ],
    },
    "Binary Trees and BSTs": {
        "chapter": 4,
        "file": "chapter4_binary_trees.pdf",
        "aliases": [
            "chapter 4",
            "chapter4",
            "chapter_4",
            "binary tree",
            "binary trees",
            "binary trees and bsts",
            "bst",
            "bsts",
            "binary search tree",
            "binary search trees",
            "tree traversal",
            "inorder",
            "preorder",
            "postorder",
            "leaf node",
            "root node",
        ],
    },
    "Logic": {
        "chapter": 1,
        "file": "chapter1_logic.pdf",
        "aliases": [
            "logic",
            "propositional logic",
            "predicate logic",
            "truth table",
            "truth tables",
            "proposition",
            "predicate",
        ],
    },
    "Sets": {
        "chapter": 2,
        "file": "chapter2_sets.pdf",
        "aliases": [
            "set",
            "sets",
            "set theory",
            "union",
            "intersection",
            "subset",
            "membership",
        ],
    },
    "Graphs": {
        "chapter": 3,
        "file": "chapter3_graphs.pdf",
        "aliases": [
            "graph",
            "graphs",
            "vertices",
            "vertex",
            "edges",
            "edge",
            "path",
            "cycle",
        ],
    },
}

SUB_CONCEPTS = {
    "Recursion": [
        "base case",
        "recursive case",
        "call stack",
        "problem decomposition",
        "termination",
    ],
    "Sorting Algorithms": [
        "comparison",
        "partitioning",
        "pivot selection",
        "time complexity",
        "stable vs unstable sorting",
    ],
    "Pointers and Memory Management": [
        "pointer basics",
        "dereferencing",
        "memory allocation",
        "null pointers",
        "pointer arithmetic",
        "memory leaks",
        "dangling pointers",
    ],
    "Binary Trees and BSTs": [
        "tree traversal",
        "insertion",
        "deletion",
        "search",
        "BST property",
        "root and leaf nodes",
    ],
    "Logic": [
        "propositions",
        "truth tables",
        "logical equivalence",
        "predicates",
        "inference rules",
    ],
    "Sets": [
        "membership",
        "subsets",
        "union",
        "intersection",
        "complements",
    ],
    "Graphs": [
        "vertices and edges",
        "paths",
        "cycles",
        "degree",
        "connected graphs",
    ],
}

ALLOWED_CONCEPTS = tuple(COURSE_CONCEPTS.keys())
CS_CONCEPTS = (
    "Recursion",
    "Sorting Algorithms",
    "Pointers and Memory Management",
    "Binary Trees and BSTs",
)
MATH_CONCEPTS = (
    "Logic",
    "Sets",
    "Graphs",
)
PLACEHOLDER_TOPICS = {
    "",
    "current topic",
    "the current topic",
    "topic",
    "that topic",
    "this topic",
    "unknown",
    "none",
    "n/a",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def canonicalize_concept(raw: str | None) -> str | None:
    if raw is None:
        return None

    normalized = _normalize(raw)
    if normalized in PLACEHOLDER_TOPICS:
        return None

    for concept in ALLOWED_CONCEPTS:
        if normalized == _normalize(concept):
            return concept

    for concept, meta in COURSE_CONCEPTS.items():
        file_stem = meta["file"].removesuffix(".pdf")
        aliases = [meta["file"], file_stem, *meta["aliases"]]
        for alias in aliases:
            alias_norm = _normalize(alias)
            if normalized == alias_norm or alias_norm in normalized:
                return concept

    return None


def concepts_in_text(raw: str | None, allowed: set[str] | tuple[str, ...] | list[str] | None = None) -> list[str]:
    if raw is None:
        return []

    normalized = _normalize(raw)
    if not normalized:
        return []

    allowed_set = set(allowed or ALLOWED_CONCEPTS)
    matches: list[str] = []
    for concept, meta in COURSE_CONCEPTS.items():
        if concept not in allowed_set:
            continue
        terms = [concept, meta["file"], meta["file"].removesuffix(".pdf"), *meta["aliases"]]
        for term in terms:
            term_norm = _normalize(term)
            if not term_norm:
                continue
            if normalized == term_norm or re.search(rf"\b{re.escape(term_norm)}s?\b", normalized):
                matches.append(concept)
                break

    return matches


def is_allowed_concept(raw: str | None) -> bool:
    return canonicalize_concept(raw) is not None


def concept_source_file(raw: str | None) -> str | None:
    concept = canonicalize_concept(raw)
    if not concept:
        return None
    return COURSE_CONCEPTS[concept]["file"]


# A Moodle course's own chapter/activity naming convention (whatever a
# teacher typed) can prepend a source-style label like "Chap: " or
# "Chapter 3:" directly onto the concept name the platform sends -- e.g. a
# Data Science course's launch context naming its concept literally
# "Chap: Linear Regression". That raw string IS the real mastery/analytics
# lookup key end to end (Moodle's own scores dict, MemoryManager, RAG
# concept metadata) and must never be altered for matching/storage -- only
# how it reads to the STUDENT needs cleaning up. This is the single,
# central place that strips such a prefix for display; every call site that
# puts a concept name in front of a student (practice questions, feedback
# text, the proactive-bootstrap focus card) should render through this
# rather than reimplementing its own string surgery.
_SOURCE_LABEL_PREFIX_RE = re.compile(r"^\s*chap(?:ter)?\.?\s*\d*\s*[:.\-]\s*", re.IGNORECASE)


def display_concept_name(raw: str | None) -> str:
    """Clean, student-facing form of a concept identifier -- e.g.
    "Chap: Linear Regression" -> "Linear Regression". Presentation only:
    never use this value as a mastery/RAG/canonicalize_concept lookup key,
    only ever for text a student reads. Returns the trimmed original
    string unchanged when there is no such prefix to strip, and an empty
    string only for empty/None input."""
    text = str(raw or "").strip()
    if not text:
        return text
    cleaned = _SOURCE_LABEL_PREFIX_RE.sub("", text).strip()
    return cleaned or text


def resolve_concept_identity(
    raw_candidate: str | None,
    available: "set[str] | list[str] | tuple[str, ...]",
) -> str | None:
    """Resolve a candidate concept name (e.g. a semantic planner's natural,
    clean-form proposal like "Linear Regression") to the exact stored
    identity in `available` (e.g. the raw ingestion-time label
    "Chap: Linear Regression"), tolerating only the same superficial
    display-prefix difference `display_concept_name` already strips for
    student-facing text (RQ1.B root-cause fix -- see
    rq1b_root_cause_analysis.md, Root cause 2).

    Deterministic, exact-match-after-normalization only: this never does a
    fuzzy/token-overlap match, so two genuinely different concepts (e.g.
    "Logistic Regression" vs. the ingested "Chap: Linear Regression") are
    never conflated just because they share a word -- only a candidate that
    is IDENTICAL to a real available concept once both are lowercased,
    punctuation-normalized, and stripped of a leading "Chap:"/"Chapter N:"
    label resolves here. Every course's canonical vocabulary already goes
    through `COURSE_CONCEPTS`/`canonicalize_concept` and needs no help from
    this function (their names never carry the prefix); this exists for
    courses (e.g. a course whose only material came through generic
    ingestion, never registered in `COURSE_CONCEPTS`) whose raw storage key
    itself carries the prefix.

    Scoped entirely to the `available` list passed in -- never resolves a
    concept that is not already present in `available`, so this adds no new
    way to cross a course boundary; the caller (as today) is solely
    responsible for scoping `available` to one course/session.

    Returns the exact string from `available` on a match, or None.
    """
    if not raw_candidate:
        return None
    if raw_candidate in available:
        return raw_candidate
    target = _normalize(display_concept_name(raw_candidate))
    if not target:
        return None
    for candidate in available:
        if _normalize(display_concept_name(candidate)) == target:
            return candidate
    return None


def sub_concepts_for(raw: str | None) -> list[str]:
    concept = canonicalize_concept(raw)
    if not concept:
        return []
    return list(SUB_CONCEPTS.get(concept, []))


def concepts_for_course(course_id: int | None = None, course_name: str | None = None) -> tuple[str, ...]:
    return tuple()


def course_structure() -> list[dict]:
    return [
        {
            "chapter": meta["chapter"],
            "concept": concept,
            "source_file": meta["file"],
        }
        for concept, meta in sorted(COURSE_CONCEPTS.items(), key=lambda item: item[1]["chapter"])
    ]


def course_structure_answer() -> str:
    chapters = course_structure()
    lines = "\n".join(
        f"{item['chapter']}. {item['concept']}"
        for item in chapters
    )
    return f"The course contains {len(chapters)} chapters:\n\n{lines}"


def all_source_files() -> list[str]:
    return [item["source_file"] for item in course_structure()]
