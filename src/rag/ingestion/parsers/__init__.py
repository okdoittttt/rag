"""문서 파서 모듈

파일 타입별 파싱 기능을 제공합니다.

새 파일 형식을 추가하려면 ``DocumentParser`` 를 상속한 파서를 만들고 아래
``_PARSERS`` 리스트에만 등록하면 됩니다. 로더(``DocumentLoader``), 설정
(``IngestionConfig.supported_extensions``), API(``/config/supported-extensions``)가
모두 이 레지스트리를 단일 소스로 참조하므로 다른 곳을 고칠 필요가 없습니다.
"""

from pathlib import Path
from typing import Optional

from rag.ingestion.parsers.base import DocumentParser
from rag.ingestion.parsers.text import TextParser
from rag.ingestion.parsers.pdf import PDFParser
from rag.ingestion.parsers.docx import DocxParser
from rag.ingestion.parsers.xlsx import XlsxParser
from rag.ingestion.parsers.pptx import PptxParser
from rag.ingestion.parsers.csv import CsvParser
from rag.ingestion.parsers.hwpx import HwpxParser


# 등록된 파서 목록 (단일 소스 — 새 파서는 여기에만 추가).
# 앞쪽 파서가 우선 매칭되므로 확장자가 겹치지 않도록 유지한다.
_PARSERS: list[DocumentParser] = [
    TextParser(),
    PDFParser(),
    DocxParser(),
    XlsxParser(),
    PptxParser(),
    CsvParser(),
    HwpxParser(),
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
    """등록된 모든 파서가 지원하는 확장자 목록을 반환한다.

    Returns:
        ``.txt`` 처럼 점(.)을 포함한 소문자 확장자 리스트. 등록 순서를 유지하며
        중복은 제거한다.
    """
    extensions: list[str] = []
    for parser in _PARSERS:
        for ext in parser.extensions:
            if ext not in extensions:
                extensions.append(ext)
    return extensions


__all__ = [
    "DocumentParser",
    "TextParser",
    "PDFParser",
    "DocxParser",
    "XlsxParser",
    "PptxParser",
    "CsvParser",
    "HwpxParser",
    "get_parser",
    "get_supported_extensions",
]
