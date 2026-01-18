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


def chunk_text(
    text: str,
    filename: str = "uploaded",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """텍스트를 청크로 분할 (API용 헬퍼)
    
    Args:
        text: 분할할 텍스트
        filename: 파일명 (메타데이터용)
        chunk_size: 청크 크기 (None이면 config 사용)
        chunk_overlap: 오버랩 크기 (None이면 config 사용)
        
    Returns:
        Chunk 리스트
    """
    config = get_config()
    size = chunk_size or config.chunking.chunk_size
    overlap = chunk_overlap or config.chunking.chunk_overlap
    
    return split_text(
        text,
        chunk_size=size,
        chunk_overlap=overlap,
        source=filename,
    )


__all__ = [
    "Chunk",
    "split_text",
    "split_markdown",
    "chunk_document",
    "chunk_text",
]
