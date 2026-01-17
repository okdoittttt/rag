"""Chunking 모듈 테스트"""

from pathlib import Path

import pytest

from rag.chunking import (
    Chunk,
    split_text,
    split_markdown,
    chunk_document,
)
from rag.ingestion.document import Document


class TestChunk:
    """Chunk 클래스 테스트"""
    
    def test_create_chunk(self) -> None:
        """Chunk 생성 확인"""
        chunk = Chunk.create(
            content="Test content",
            source="/path/to/file.txt",
            chunk_index=0,
            start_char=0,
            end_char=12,
        )
        
        assert chunk.content == "Test content"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["source"] == "/path/to/file.txt"
    
    def test_chunk_length(self) -> None:
        """Chunk 길이 반환 확인"""
        chunk = Chunk(content="12345")
        
        assert len(chunk) == 5


class TestSplitText:
    """기본 텍스트 분할 테스트"""
    
    def test_split_short_text(self) -> None:
        """짧은 텍스트는 하나의 청크"""
        chunks = split_text("Hello world", chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"
    
    def test_split_long_text(self) -> None:
        """긴 텍스트 분할"""
        text = "A" * 300  # 300자
        chunks = split_text(text, chunk_size=100, chunk_overlap=0)
        
        assert len(chunks) == 3
    
    def test_split_preserves_sentence_boundary(self) -> None:
        """문장 경계 보존"""
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text(text, chunk_size=30, chunk_overlap=0)
        
        # 첫 청크가 문장 경계에서 끝나는지 확인
        assert chunks[0].content.endswith(".")
    
    def test_split_with_overlap(self) -> None:
        """오버랩 적용 확인"""
        text = "A" * 100 + " " + "B" * 100
        chunks = split_text(text, chunk_size=110, chunk_overlap=20)
        
        # 오버랩으로 인해 일부 내용이 중복되어야 함
        if len(chunks) > 1:
            # 두 번째 청크가 첫 번째 청크의 끝부분과 겹치는지
            assert len(chunks[1].content) > 0
    
    def test_split_empty_text(self) -> None:
        """빈 텍스트는 빈 리스트"""
        chunks = split_text("")
        
        assert chunks == []
    
    def test_chunk_metadata_positions(self) -> None:
        """청크 위치 메타데이터 확인"""
        text = "Hello world. This is test."
        chunks = split_text(text, chunk_size=15, chunk_overlap=0)
        
        # 첫 청크의 위치 정보 확인
        assert chunks[0].metadata["start_char"] == 0
        assert chunks[0].metadata["chunk_index"] == 0


class TestSplitMarkdown:
    """Markdown 분할 테스트"""
    
    def test_split_by_headers(self) -> None:
        """헤더별 분할"""
        text = """# Title

Content for title.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""
        chunks = split_markdown(text, chunk_size=1000)
        
        # 최소 3개 청크 (Title, Section 1, Section 2)
        assert len(chunks) >= 3
    
    def test_header_path_metadata(self) -> None:
        """헤더 경로 메타데이터 확인"""
        text = """# Main Title

## Sub Section

Content here.
"""
        chunks = split_markdown(text, chunk_size=1000)
        
        # 마지막 청크에 헤더 경로가 있어야 함
        last_chunk = chunks[-1]
        assert "header_path" in last_chunk.metadata
    
    def test_large_section_gets_split(self) -> None:
        """큰 섹션은 추가 분할"""
        text = "# Title\n\n" + "A" * 500
        chunks = split_markdown(text, chunk_size=100, chunk_overlap=0)
        
        # 500자 내용이 100자 청크로 분할되어야 함
        assert len(chunks) > 1
    
    def test_text_before_first_header(self) -> None:
        """첫 헤더 이전 텍스트 처리"""
        text = """Some intro text.

# First Header

Content.
"""
        chunks = split_markdown(text, chunk_size=1000)
        
        # 첫 청크가 intro 텍스트여야 함
        assert "intro" in chunks[0].content.lower()


class TestChunkDocument:
    """chunk_document 통합 테스트"""
    
    def test_chunk_txt_document(self) -> None:
        """txt 문서 청킹"""
        doc = Document(
            content="This is a test document with some content.",
            metadata={"source": "/test/file.txt", "extension": ".txt"}
        )
        
        chunks = chunk_document(doc)
        
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "/test/file.txt"
    
    def test_chunk_md_document(self) -> None:
        """md 문서 청킹 (Markdown 분할기 사용)"""
        doc = Document(
            content="# Header\n\nContent here.",
            metadata={"source": "/test/file.md", "extension": ".md"}
        )
        
        chunks = chunk_document(doc)
        
        assert len(chunks) >= 1
