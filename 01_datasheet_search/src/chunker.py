"""Page- and section-aware text chunking components."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PageText:
    """Text extracted from one 1-based PDF page."""

    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """A searchable document unit and its Chroma-compatible metadata."""

    text: str
    metadata: dict[str, str | int]


NEWLINE = chr(10)

_HEADING_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+\S+")


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    words = line.split()
    if len(words) > 14 or line.endswith(('.', ',', ';', ':')):
        return False
    if _HEADING_NUMBER.match(line):
        return True
    # 데이터시트의 핀 표는 DP3 · GND · VDD33 처럼 짧은 대문자 토큰이 줄마다 온다.
    # 대문자 비율만 보면 표의 모든 줄이 제목이 되어 표가 낱줄로 쪼개진다.
    if len(words) < 2:
        return False
    letters = [character for character in line if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters
        else 0.0
    )
    return uppercase_ratio >= 0.75


def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        target_end = min(start + chunk_size, len(text))
        end = target_end
        if target_end < len(text):
            candidates = [
                text.rfind("\n", start, target_end),
                text.rfind(". ", start, target_end),
                text.rfind(" ", start, target_end),
            ]
            boundary = max(candidates)
            if boundary > start + chunk_size // 2:
                end = min(
                    boundary + (2 if text[boundary : boundary + 2] == ". " else 1),
                    target_end,
                )

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _merge_small_chunks(
    chunks: list[Chunk], chunk_size: int, min_chunk_size: int
) -> list[Chunk]:
    """같은 페이지의 연속된 작은 청크를 붙인다.

    핀 표처럼 한 줄이 한 항목인 구간은 낱줄로 쪼개면 검색해도 의미가 없다.
    표의 인접 행을 한 청크로 묶어 헤더와 단위가 같이 남게 한다.
    chunk_size 는 상한으로 지킨다.
    """
    if min_chunk_size <= 0:
        return chunks
    target = min(min_chunk_size, chunk_size)
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged:
            last = merged[-1]
            same_page = last.metadata.get("page") == chunk.metadata.get("page")
            combined = len(last.text) + 1 + len(chunk.text)
            if same_page and len(last.text) < target and combined <= chunk_size:
                merged[-1] = Chunk(
                    text=last.text + NEWLINE + chunk.text,
                    metadata=last.metadata,
                )
                continue
        merged.append(chunk)
    return merged


def create_chunks(
    pages: Iterable[PageText],
    base_metadata: Mapping[str, str],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int = 0,
) -> list[Chunk]:
    """Create page-bounded chunks while carrying the most recent heading."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    output: list[Chunk] = []
    current_section = "Unknown"
    for page in pages:
        normalized = _normalize_text(page.text)
        if not normalized:
            continue

        sections: list[tuple[str, list[str]]] = []
        body: list[str] = []
        section = current_section
        for line in normalized.splitlines():
            if _looks_like_heading(line):
                if body:
                    sections.append((section, body))
                    body = []
                section = line
                current_section = line
            body.append(line)
        if body:
            sections.append((section, body))

        for section_name, lines in sections:
            section_text = "\n".join(lines)
            for part in _split_with_overlap(section_text, chunk_size, overlap):
                metadata: dict[str, str | int] = dict(base_metadata)
                metadata.update({"page": page.page, "section": section_name})
                output.append(Chunk(text=part, metadata=metadata))
    return _merge_small_chunks(output, chunk_size, min_chunk_size)
