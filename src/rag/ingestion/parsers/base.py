"""문서 파서 기본 클래스

파일 타입별 파싱 로직을 추상화합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class DocumentParser(ABC):
    """문서 파서 추상 클래스"""
    
    # 지원하는 파일 확장자 (하위 클래스에서 정의)
    extensions: list[str] = []
    
    @abstractmethod
    def parse(self, path: Path) -> str:
        """파일을 파싱하여 텍스트 반환
        
        Args:
            path: 파일 경로
            
        Returns:
            추출된 텍스트
        """
        ...
    
    def can_parse(self, path: Path) -> bool:
        """이 파서가 해당 파일을 파싱할 수 있는지 확인"""
        return path.suffix.lower() in self.extensions
