"""문서 파서 모듈

파일 타입별 파싱 기능을 제공합니다.
"""

from pathlib import Path
from typing import Optional

from rag.ingestion.parsers.base import DocumentParser
from rag.ingestion.parsers.text import TextParser
from rag.ingestion.parsers.pdf import PDFParser


# 등록된 파서 목록
_PARSERS: list[DocumentParser] = [
    TextParser(),
    PDFParser(),
]


def get_parser(path: Path) -> Optional[DocumentParser]:
    """파일에 맞는 파서 반환
    
    Args:
        path: 파일 경로
        
    Returns:
        적합한 파서 또는 None
    """
    for parser in _PARSERS:
        if parser.can_parse(path):
            return parser
    return None


def get_supported_extensions() -> list[str]:
    """지원되는 모든 확장자 반환"""
    extensions: list[str] = []
    for parser in _PARSERS:
        extensions.extend(parser.extensions)
    return extensions


__all__ = [
    "DocumentParser",
    "TextParser",
    "PDFParser",
    "get_parser",
    "get_supported_extensions",
]
