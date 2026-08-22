"""Variable chunking: content-aware separators, length-adaptive chunk size.

Token counts are approximated as chars/4 — close enough for sizing chunks,
and it keeps this module dependency-free and offline-testable.
"""
import re

# per-profile: separators tried in order, target chunk size in tokens
PROFILES = {
    "markdown": {"separators": ["\n## ", "\n### ", "\n\n", "\n", " "], "target": 400},
    "code": {"separators": ["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " "], "target": 300},
    "prose": {"separators": ["\n\n", "\n", ". ", " "], "target": 450},
}
OVERLAP_FRACTION = 0.12
_CODE_HINT = re.compile(r"(^|\n)\s*(def |class |import |function |const |#include|=>)", re.M)


def _tokens(text):
    return len(text) // 4


def detect_profile(text, source=""):
    s = source.lower()
    if s.endswith((".md", ".mdx", ".markdown")) or text.lstrip().startswith("#"):
        return "markdown"
    if s.endswith((".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cpp")):
        return "code"
    if len(_CODE_HINT.findall(text)) >= 5:
        return "code"
    return "prose"


def adaptive_target(base_target, doc_tokens):
    """Scale chunk size with document length to bound total chunk count."""
    if doc_tokens <= base_target:
        return doc_tokens or 1  # whole doc fits in one chunk
    if doc_tokens > 50_000:  # very long docs: bigger chunks, fewer vectors
        return base_target * 2
    return base_target


def _split(text, separators, target):
    """Recursively split text until every piece is <= target tokens."""
    if _tokens(text) <= target or not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split(text, rest, target)
    # re-attach separator to keep headings/structure with their section
    pieces = [parts[0]] + [sep + p for p in parts[1:]]
    out = []
    for p in pieces:
        out.extend(_split(p, rest, target) if _tokens(p) > target else [p])
    return out


def chunk_document(text, source=""):
    """Split a document into chunks. Returns list of {text, ordinal, profile}."""
    text = text.strip()
    if not text:
        return []
    profile = detect_profile(text, source)
    cfg = PROFILES[profile]
    target = adaptive_target(cfg["target"], _tokens(text))
    pieces = _split(text, cfg["separators"], target)

    # greedily merge small pieces up to target, with token overlap between chunks
    chunks, current = [], ""
    for p in pieces:
        if current and _tokens(current) + _tokens(p) > target:
            chunks.append(current)
            overlap_chars = int(len(current) * OVERLAP_FRACTION)
            current = current[-overlap_chars:] if overlap_chars else ""
        current += p
    if current.strip():
        chunks.append(current)
    return [
        {"text": c.strip(), "ordinal": i, "profile": profile}
        for i, c in enumerate(chunks) if c.strip()
    ]
