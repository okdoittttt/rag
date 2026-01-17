"""Ingestion 모듈

문서 수집, 정규화, 언어 감지 기능을 제공합니다.
"""

from rag.ingestion.document import Document
from rag.ingestion.loader import load_documents, load_file
from rag.ingestion.normalizer import normalize_text
from rag.ingestion.language import detect_language


__all__ = [
    "Document",
    "load_documents",
    "load_file",
    "normalize_text",
    "detect_language",
]
