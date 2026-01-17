"""Markdown 구조 보존 분할 모듈

Markdown 헤더를 인식하고 섹션별로 분할합니다.
각 청크에 헤더 경로 메타데이터를 추가합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.chunking.chunk import Chunk
from rag.chunking.splitter import split_text


# Markdown 헤더 패턴 (# ~ ######)
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Section:
    """Markdown 섹션"""
    level: int          # 헤더 레벨 (1-6)
    title: str          # 헤더 제목
    content: str        # 섹션 본문
    start_pos: int      # 원본에서 시작 위치
    end_pos: int        # 원본에서 끝 위치


def _parse_sections(text: str) -> list[Section]:
    """Markdown 텍스트를 섹션으로 파싱
    
    Args:
        text: Markdown 텍스트
        
    Returns:
        Section 리스트
    """
    sections: list[Section] = []
    
    # 모든 헤더 찾기
    headers = list(HEADER_PATTERN.finditer(text))
    
    if not headers:
        # 헤더가 없으면 전체를 하나의 섹션으로
        return [Section(
            level=0,
            title="",
            content=text,
            start_pos=0,
            end_pos=len(text),
        )]
    
    # 첫 헤더 이전 내용
    if headers[0].start() > 0:
        content = text[:headers[0].start()].strip()
        if content:
            sections.append(Section(
                level=0,
                title="",
                content=content,
                start_pos=0,
                end_pos=headers[0].start(),
            ))
    
    # 각 헤더별 섹션 생성
    for i, match in enumerate(headers):
        level = len(match.group(1))
        title = match.group(2).strip()
        
        # 다음 헤더까지 또는 끝까지
        start_pos = match.start()
        if i + 1 < len(headers):
            end_pos = headers[i + 1].start()
        else:
            end_pos = len(text)
        
        content = text[start_pos:end_pos].strip()
        
        sections.append(Section(
            level=level,
            title=title,
            content=content,
            start_pos=start_pos,
            end_pos=end_pos,
        ))
    
    return sections


def _build_header_path(sections: list[Section], current_index: int) -> str:
    """현재 섹션의 헤더 경로 생성
    
    Args:
        sections: 모든 섹션 리스트
        current_index: 현재 섹션 인덱스
        
    Returns:
        헤더 경로 (예: "# Title > ## Section > ### Subsection")
    """
    if current_index >= len(sections):
        return ""
    
    current = sections[current_index]
    if current.level == 0:
        return ""
    
    path_parts: list[str] = []
    target_level = current.level
    
    # 현재 섹션부터 역순으로 상위 헤더 찾기
    for i in range(current_index, -1, -1):
        section = sections[i]
        
        if section.level > 0 and section.level <= target_level:
            prefix = "#" * section.level
            path_parts.append(f"{prefix} {section.title}")
            target_level = section.level - 1
        
        if target_level == 0:
            break
    
    path_parts.reverse()
    return " > ".join(path_parts)


def split_markdown(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    source: str = "",
) -> list[Chunk]:
    """Markdown 구조를 보존하며 청크 분할
    
    Args:
        text: Markdown 텍스트
        chunk_size: 목표 청크 크기 (자)
        chunk_overlap: 오버랩 크기
        source: 원본 문서 경로 (메타데이터용)
        
    Returns:
        Chunk 리스트
    """
    if not text or not text.strip():
        return []
    
    sections = _parse_sections(text)
    chunks: list[Chunk] = []
    chunk_index = 0
    
    for section_idx, section in enumerate(sections):
        header_path = _build_header_path(sections, section_idx)
        
        # 섹션이 chunk_size보다 크면 추가 분할
        if len(section.content) > chunk_size:
            sub_chunks = split_text(
                section.content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source=source,
            )
            
            for sub_chunk in sub_chunks:
                # 메타데이터 업데이트
                sub_chunk.metadata["chunk_index"] = chunk_index
                sub_chunk.metadata["start_char"] = section.start_pos + sub_chunk.metadata["start_char"]
                sub_chunk.metadata["end_char"] = section.start_pos + sub_chunk.metadata["end_char"]
                sub_chunk.metadata["header_path"] = header_path
                
                chunks.append(sub_chunk)
                chunk_index += 1
        else:
            # 섹션 전체를 하나의 청크로
            if section.content.strip():
                chunk = Chunk.create(
                    content=section.content,
                    source=source,
                    chunk_index=chunk_index,
                    start_char=section.start_pos,
                    end_char=section.end_pos,
                    header_path=header_path,
                )
                chunks.append(chunk)
                chunk_index += 1
    
    return chunks
