"""Markdown-header-aware chunking.

Splits on header boundaries first so a chunk stays a whole semantic unit
where possible, then hard-splits any section still too long. Token counts
always use the MiniLM tokenizer, even when the active embedder is Gemini:
MiniLM has the stricter 256-token hard cutoff of the two backends, so
sizing every chunk against it keeps both backends safe from silent
truncation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from devdesk.config import CHUNK_OVERLAP_TOKENS, MAX_CHUNK_TOKENS

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    text: str
    header_path: str
    index: int


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(header_path, section_text)], preserving document order."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    header_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            path = " > ".join(h for _, h in header_stack) or "(root)"
            sections.append((path, body))

    for line in lines:
        match = _HEADER_RE.match(line)
        if match:
            flush()
            current_lines = []
            level = len(match.group(1))
            title = match.group(2).strip()
            header_stack = [h for h in header_stack if h[0] < level]
            header_stack.append((level, title))
        else:
            current_lines.append(line)
    flush()
    return sections


def _pack_words(words: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack words into windows <= max_tokens, with token overlap."""
    if not words:
        return []
    windows: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = start + 1
        candidate = words[start]
        while end < n:
            next_candidate = candidate + " " + words[end]
            if count_tokens(next_candidate) > max_tokens:
                break
            candidate = next_candidate
            end += 1
        windows.append(candidate)
        if end >= n:
            break
        # Step back from `end` until the tail is <= overlap_tokens, to seed
        # the next window's overlap.
        back = end - 1
        while back > start and count_tokens(" ".join(words[back:end])) < overlap_tokens:
            back -= 1
        start = max(back, start + 1)
    return windows


def chunk_markdown(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for header_path, section_text in _split_sections(text):
        if count_tokens(section_text) <= max_tokens:
            chunks.append(Chunk(text=section_text, header_path=header_path, index=index))
            index += 1
            continue
        for window in _pack_words(section_text.split(), max_tokens, overlap_tokens):
            chunks.append(Chunk(text=window, header_path=header_path, index=index))
            index += 1
    return chunks
