"""Deterministic citation verification.

Specialists are told to copy each hit's `citation` verbatim, but a model
can still invent one (the held-out eval caught flash-lite citing a
`docs/install_and_requirements.md` that doesn't exist). So the search tools
record every (source_file, header_path) they actually return during a
request, and the final answer is checked against that record: any cited
source that wasn't retrieved is replaced with a visible marker.

The record lives in a ContextVar set per request by `cli.run_question`.
ADK runs sync tools inline on the event loop, and the specialist's
AgentTool sub-run is awaited inside the router's tool call, so every tool
call in a request sees the same set — and concurrent server requests
don't see each other's.
"""

from __future__ import annotations

import re
from contextvars import ContextVar

UNVERIFIED_MARKER = "(citation removed: not in retrieved sources)"

_retrieved: ContextVar[set[tuple[str, str]] | None] = ContextVar("retrieved_citations", default=None)

# Locates the file part of a citation; the full span is found by walking
# out to the enclosing parentheses, because headers can contain their own
# parentheses, e.g. "Game engine (`convex/engine`) > Engine state management".
_FILE_RE = re.compile(r"([\w./-]+\.md) > ")


def start_request() -> set[tuple[str, str]]:
    """Begin recording for the current request; returns the live record."""
    record: set[tuple[str, str]] = set()
    _retrieved.set(record)
    return record


def record_hits(hits: list[dict]) -> None:
    record = _retrieved.get()
    if record is None:
        return  # called outside a request (tests, direct tool use)
    for h in hits:
        record.add((h["source_file"], h["header_path"]))


def _enclosing_parens(text: str, pos: int) -> tuple[int, int] | None:
    depth = 0
    start = pos
    while start > 0:
        start -= 1
        if text[start] == ")":
            depth += 1
        elif text[start] == "(":
            if depth == 0:
                break
            depth -= 1
    else:
        return None
    depth = 0
    for end in range(start + 1, len(text)):
        if text[end] == "(":
            depth += 1
        elif text[end] == ")":
            if depth == 0:
                return start, end + 1
            depth -= 1
    return None


def _is_verified(source_file: str, header: str, record: set[tuple[str, str]]) -> bool:
    # A citation may shorten the header path ("README.md > Ares") but must
    # point at a file and section that were actually retrieved.
    return any(
        f == source_file and (h == header or h.startswith(header + " > "))
        for f, h in record
    )


# Junk between one citation's header and the next citation: separators,
# backticks, and label prefixes like "devdesk_ares:" or "Citations:".
_TRAILING_JUNK = re.compile(r"(?:[\s,;`]|\b[A-Za-z_][\w-]*:)+$")
_LEADING_JUNK = re.compile(r"^(?:[\s,;`]|\b[A-Za-z_][\w-]*:)+")


def _citations_in(inner: str) -> list[tuple[int, int, str, str]]:
    """(start, end, source_file, header) for each citation in a parenthesis.

    A header runs to the next file reference rather than to the next comma,
    because real headers contain commas ("Architecture, Traceability and
    Validation Evidence")."""
    matches = list(_FILE_RE.finditer(inner))
    found = []
    for i, m in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(inner)
        raw = inner[m.end() : stop]
        header = _TRAILING_JUNK.sub("", raw).strip()
        found.append((m.start(), m.end() + len(header), m.group(1), header))
    return found


def verify_answer(answer: str, record: set[tuple[str, str]]) -> tuple[str, list[str]]:
    """Return (answer with unverified citations removed, the removed citations).

    Handles the formats models actually produce inside one parenthesis:
    several citations separated by ";" or ",", backtick-wrapped, and with
    label prefixes ("verisim: ...", "Citations: ..."). Only parenthesised
    citations are checked — the format the specialists are told to use.
    """
    spans: list[tuple[int, int]] = []
    for m in _FILE_RE.finditer(answer):
        span = _enclosing_parens(answer, m.start())
        if span is not None and span not in spans:
            spans.append(span)

    removed: list[str] = []
    for start, end in sorted(spans, reverse=True):
        inner = answer[start + 1 : end - 1]
        cites = _citations_in(inner)
        bad = [c for c in cites if not _is_verified(c[2], c[3], record)]
        if not bad:
            continue
        removed[:0] = [f"{f} > {h}" for _, _, f, h in bad]
        if len(bad) == len(cites):
            replacement = UNVERIFIED_MARKER
        else:
            kept = inner
            for c_start, c_end, _, _ in reversed(bad):
                kept = kept[:c_start] + kept[c_end:]
            kept = _LEADING_JUNK.sub("", _TRAILING_JUNK.sub("", kept))
            kept = re.sub(r"(?:\s*[,;]\s*){2,}", "; ", kept)
            replacement = f"({kept})"
        answer = answer[:start] + replacement + answer[end:]
    return answer, removed
