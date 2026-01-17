"""Chunking 모듈

문서를 청크로 분할하는 기능을 제공합니다.
"""

from rag.chunking.chunk import Chunk
from rag.chunking.splitter import split_text
from rag.chunking.markdown import split_markdown
from rag.config import get_config
from rag.ingestion.document import Document


def chunk_document(doc: Document) -> list[Chunk]:
    """문서를 청크로 분할
    
    확장자에 따라 적절한 분할 전략을 선택합니다.
    - .md: Markdown 구조 보존 분할
    - 기타: 기본 텍스트 분할
    
    Args:
        doc: 분할할 Document
        
    Returns:
        Chunk 리스트
    """
    config = get_config()
    chunk_size = config.chunking.chunk_size
    chunk_overlap = config.chunking.chunk_overlap
    source = doc.metadata.get("source", "")
    extension = doc.metadata.get("extension", "")
    
    if extension == ".md":
        return split_markdown(
            doc.content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=source,
        )
    else:
        return split_text(
            doc.content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source=source,
        )


__all__ = [
    "Chunk",
    "split_text",
    "split_markdown",
    "chunk_document",
]
