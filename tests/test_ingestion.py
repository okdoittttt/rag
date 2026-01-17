"""Ingestion 모듈 테스트"""

import tempfile
from pathlib import Path

import pytest

from rag.ingestion import (
    Document,
    load_documents,
    load_file,
    normalize_text,
    detect_language,
)


class TestDocument:
    """Document 클래스 테스트"""
    
    def test_from_file_creates_document(self, tmp_path: Path) -> None:
        """파일에서 Document 생성 확인"""
        # 테스트 파일 생성
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        doc = Document.from_file(test_file, "Hello, World!")
        
        assert doc.content == "Hello, World!"
        assert doc.metadata["filename"] == "test.txt"
        assert doc.metadata["extension"] == ".txt"
    
    def test_document_length(self) -> None:
        """Document 길이 반환 확인"""
        doc = Document(content="12345")
        
        assert len(doc) == 5
    
    def test_document_repr(self) -> None:
        """Document repr 확인"""
        doc = Document(
            content="test content",
            metadata={"filename": "test.md"}
        )
        
        assert "test.md" in repr(doc)


class TestNormalizer:
    """정규화 테스트"""
    
    def test_normalize_consecutive_spaces(self) -> None:
        """연속 공백 정규화 확인"""
        text = "hello   world"
        result = normalize_text(text)
        
        assert "   " not in result
        assert "hello world" in result
    
    def test_normalize_consecutive_newlines(self) -> None:
        """연속 개행 정규화 확인"""
        text = "hello\n\n\n\nworld"
        result = normalize_text(text)
        
        assert "\n\n\n" not in result
    
    def test_normalize_tabs(self) -> None:
        """탭 → 공백 변환 확인"""
        text = "hello\tworld"
        result = normalize_text(text)
        
        assert "\t" not in result
    
    def test_normalize_strips_edges(self) -> None:
        """앞뒤 공백 제거 확인"""
        text = "   hello world   "
        result = normalize_text(text)
        
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestLanguageDetection:
    """언어 감지 테스트"""
    
    def test_detect_korean(self) -> None:
        """한국어 감지 확인"""
        text = "안녕하세요. 한국어 테스트입니다."
        
        assert detect_language(text) == "ko"
    
    def test_detect_english(self) -> None:
        """영어 감지 확인"""
        text = "Hello, this is an English text for testing."
        
        assert detect_language(text) == "en"
    
    def test_detect_mixed_korean_dominant(self) -> None:
        """한국어가 우세한 혼합 텍스트"""
        text = "안녕하세요 Hello 테스트입니다"
        
        assert detect_language(text) == "ko"
    
    def test_detect_empty_returns_unknown(self) -> None:
        """빈 텍스트는 unknown 반환"""
        assert detect_language("") == "unknown"
        assert detect_language("   ") == "unknown"


class TestLoader:
    """파일 로더 테스트"""
    
    def test_load_single_txt_file(self, tmp_path: Path) -> None:
        """단일 txt 파일 로딩"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        doc = load_file(test_file)
        
        assert doc is not None
        assert doc.content == "Test content"
        assert doc.metadata["extension"] == ".txt"
    
    def test_load_single_md_file(self, tmp_path: Path) -> None:
        """단일 md 파일 로딩"""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Heading\n\nContent here")
        
        doc = load_file(test_file)
        
        assert doc is not None
        assert "# Heading" in doc.content
        assert doc.metadata["extension"] == ".md"
    
    def test_load_directory(self, tmp_path: Path) -> None:
        """디렉토리 로딩"""
        # 파일 생성
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.md").write_text("Content 2")
        (tmp_path / "file3.py").write_text("# Not supported")
        
        docs = load_documents(tmp_path)
        
        assert len(docs) == 2  # .py는 제외
    
    def test_load_directory_recursive(self, tmp_path: Path) -> None:
        """재귀 디렉토리 로딩"""
        # 중첩 디렉토리 생성
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        (tmp_path / "root.txt").write_text("Root")
        (subdir / "nested.txt").write_text("Nested")
        
        docs = load_documents(tmp_path, recursive=True)
        
        assert len(docs) == 2
    
    def test_load_nonexistent_returns_empty(self) -> None:
        """존재하지 않는 경로는 빈 리스트 반환"""
        docs = load_documents("/nonexistent/path")
        
        assert docs == []
