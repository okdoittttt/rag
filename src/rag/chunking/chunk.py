"""청크 데이터 모델

문서를 분할한 청크의 본문과 메타데이터를 담는 데이터 클래스입니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """청크 데이터 클래스
    
    Attributes:
        content: 청크 본문 텍스트
        metadata: 청크 메타데이터
            - source: 원본 문서 경로
            - chunk_index: 청크 순서 (0부터)
            - start_char: 원본에서 시작 위치
            - end_char: 원본에서 끝 위치
            - header_path: Markdown 헤더 경로 (예: "# Title > ## Section")
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        content: str,
        source: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        header_path: str = "",
    ) -> Chunk:
        """청크 생성 헬퍼
        
        Args:
            content: 청크 본문
            source: 원본 문서 경로
            chunk_index: 청크 순서
            start_char: 원본에서 시작 위치
            end_char: 원본에서 끝 위치
            header_path: Markdown 헤더 경로
            
        Returns:
            Chunk 인스턴스
        """
        return cls(
            content=content,
            metadata={
                "source": source,
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
                "header_path": header_path,
            }
        )
    
    def __len__(self) -> int:
        """청크 본문 길이 반환"""
        return len(self.content)
    
    def __repr__(self) -> str:
        index = self.metadata.get("chunk_index", "?")
        length = len(self.content)
        return f"Chunk(index={index}, length={length})"
