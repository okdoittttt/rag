"""설정 모듈 테스트"""

import os
from pathlib import Path

import pytest

from rag.config import (
    Config,
    load_config,
    get_config,
    reload_config,
)


class TestConfig:
    """Config 테스트"""
    
    def test_default_config_loads(self) -> None:
        """기본 설정 파일 로딩 확인"""
        config = load_config()
        
        assert config.project.name == "terminal-rag"
        assert config.project.version == "0.1.0"
    
    def test_config_has_all_sections(self) -> None:
        """모든 설정 섹션 존재 확인"""
        config = load_config()
        
        assert config.project is not None
        assert config.ingestion is not None
        assert config.chunking is not None
        assert config.embedding is not None
        assert config.retrieval is not None
        assert config.generation is not None
        assert config.logging is not None
    
    def test_chunking_defaults(self) -> None:
        """청킹 기본값 확인"""
        config = load_config()
        
        assert config.chunking.chunk_size == 1000
        assert config.chunking.chunk_overlap == 150
        assert config.chunking.preserve_structure is True
    
    def test_retrieval_defaults(self) -> None:
        """검색 기본값 확인"""
        config = load_config()
        
        assert config.retrieval.top_k == 5
        assert config.retrieval.score_threshold == 0.7
        assert config.retrieval.search_type == "vector"
    
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """환경변수 오버라이드 확인"""
        monkeypatch.setenv("RAG_LOGGING_LEVEL", "DEBUG")
        
        config = reload_config()
        
        assert config.logging.level == "DEBUG"
        
        # 정리
        reload_config()
    
    def test_singleton_caching(self) -> None:
        """싱글톤 캐싱 확인"""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_reload_clears_cache(self) -> None:
        """reload_config이 캐시를 초기화하는지 확인"""
        config1 = get_config()
        config2 = reload_config()
        
        # 새로운 인스턴스가 생성되어야 함
        assert config1 is not config2
