"""기본 텍스트 분할 모듈

문자 수 기반으로 텍스트를 청크로 분할합니다.
문장/문단 경계를 최대한 보존합니다.
"""

from __future__ import annotations

import re

from rag.chunking.chunk import Chunk


# 분할 우선순위별 구분자 패턴
SEPARATORS = [
    "\n\n",    # 문단 경계
    "\n",      # 줄바꿈
    ". ",      # 문장 경계 (마침표)
    "? ",      # 문장 경계 (물음표)
    "! ",      # 문장 경계 (느낌표)
    "。",      # 한국어/일본어 마침표
    " ",       # 단어 경계
]


def _find_split_point(text: str, max_size: int) -> int:
    """최적의 분할 지점 찾기
    
    우선순위에 따라 구분자를 찾아 분할 지점 결정.
    
    Args:
        text: 분할할 텍스트
        max_size: 최대 크기
        
    Returns:
        분할 지점 인덱스
    """
    if len(text) <= max_size:
        return len(text)
    
    # 각 구분자에 대해 max_size 이하에서 가장 마지막 위치 찾기
    for sep in SEPARATORS:
        # max_size 범위 내에서 구분자의 마지막 위치 찾기
        search_area = text[:max_size]
        pos = search_area.rfind(sep)
        
        if pos > 0:
            # 구분자 포함하여 분할 (문장 부호는 앞 청크에 포함)
            return pos + len(sep)
    
    # 구분자를 찾지 못하면 강제 분할
    return max_size


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    source: str = "",
) -> list[Chunk]:
    """텍스트를 청크로 분할
    
    Args:
        text: 분할할 텍스트
        chunk_size: 목표 청크 크기 (자)
        chunk_overlap: 오버랩 크기
        source: 원본 문서 경로 (메타데이터용)
        
    Returns:
        Chunk 리스트
    """
    if not text or not text.strip():
        return []
    
    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # 남은 텍스트
        remaining = text[start:]
        
        # 분할 지점 찾기
        split_point = _find_split_point(remaining, chunk_size)
        
        # 청크 내용 추출
        chunk_content = remaining[:split_point].strip()
        
        if chunk_content:  # 빈 청크 제외
            chunk = Chunk.create(
                content=chunk_content,
                source=source,
                chunk_index=chunk_index,
                start_char=start,
                end_char=start + split_point,
            )
            chunks.append(chunk)
            chunk_index += 1
        
        # 다음 시작점 (오버랩 적용)
        if start + split_point >= len(text):
            break
        
        # 오버랩을 적용하되, 최소한 1자는 진행
        next_start = start + split_point - chunk_overlap
        start = max(next_start, start + 1)
    
    return chunks
