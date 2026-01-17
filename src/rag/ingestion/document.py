"""문서 데이터 모델

문서의 본문과 메타데이터를 담는 데이터 클래스입니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """문서 데이터 클래스
    
    Attributes:
        content: 문서 본문 텍스트
        metadata: 문서 메타데이터
            - source: 원본 파일 경로 (절대 경로)
            - filename: 파일명
            - extension: 파일 확장자 (.txt, .md 등)
            - size_bytes: 파일 크기 (바이트)
            - created_at: 문서 객체 생성 시간
            - language: 감지된 언어 (ko/en/unknown)
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: Path, content: str) -> Document:
        """파일 정보로부터 Document 생성
        
        Args:
            path: 파일 경로
            content: 파일 본문
            
        Returns:
            Document 인스턴스
        """
        stat = path.stat()
        
        return cls(
            content=content,
            metadata={
                "source": str(path.resolve()),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "created_at": datetime.now().isoformat(),
                "language": "unknown",  # 나중에 감지
            }
        )
    
    def __len__(self) -> int:
        """문서 본문 길이 반환"""
        return len(self.content)
    
    def __repr__(self) -> str:
        filename = self.metadata.get("filename", "unknown")
        length = len(self.content)
        return f"Document(filename={filename!r}, length={length})"
