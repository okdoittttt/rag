"""Retrieval 모듈 테스트"""

from pathlib import Path

import numpy as np
import pytest

from rag.chunking.chunk import Chunk
from rag.embedding.embedder import Embedder
from rag.embedding import VectorStore
from rag.retrieval.bm25 import BM25Searcher
from rag.retrieval.searcher import HybridSearcher
from rag.retrieval.tokenizer import tokenize_query


class MockEmbedder(Embedder):
    """테스트용 Mock Embedder"""
    def __init__(self):
        # 4차원 더미 임베딩
        pass
        
    def embed(self, texts: list[str]) -> np.ndarray:
        # 텍스트 길이에 따라 다른 벡터 생성 (구분을 위해)
        return np.array([
            [len(t) * 0.1] * 4 for t in texts
        ], dtype=np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        return np.array([
            [len(query) * 0.1] * 4
        ], dtype=np.float32)


@pytest.fixture
def sample_chunks():
    return [
        Chunk(content="사과는 과일이다", metadata={"chunk_index": 0}),
        Chunk(content="바나나는 노랗다", metadata={"chunk_index": 1}),
        Chunk(content="하늘은 파랗다", metadata={"chunk_index": 2}),
    ]


class TestTokenizer:
    """토크나이저 테스트"""
    
    def test_tokenize_korean(self):
        """한국어 토크나이징 (Kiwi 설치 시)"""
        # Kiwi가 설치되어 있다고 가정하거나, mock 처리 필요
        # 여기서는 단순 실행 여부만 확인
        tokens = tokenize_query("사과는 맛있다")
        assert isinstance(tokens, list)
    
    def test_tokenize_english(self):
        """영어 토크나이징"""
        tokens = tokenize_query("Apple is delicious", language="en")
        assert "apple" in tokens


class TestBM25Searcher:
    """BM25 검색 테스트"""
    
    def test_index_and_search(self, sample_chunks):
        searcher = BM25Searcher()
        searcher.index(sample_chunks)
        
        # "사과" 검색
        results = searcher.search("사과")
        assert len(results) > 0
        assert results[0][0].content == "사과는 과일이다"
        
    def test_save_and_load(self, sample_chunks, tmp_path: Path):
        searcher = BM25Searcher()
        searcher.index(sample_chunks)
        
        save_path = tmp_path / "bm25_index"
        searcher.save(save_path)
        
        new_searcher = BM25Searcher()
        new_searcher.load(save_path)
        
        results = new_searcher.search("바나나")
        assert len(results) > 0
        assert results[0][0].content == "바나나는 노랗다"


class TestHybridSearcher:
    """하이브리드 검색 테스트"""
    
    @pytest.fixture
    def hybrid_searcher(self):
        embedder = MockEmbedder()
        store = VectorStore(dimension=4)
        return HybridSearcher(embedder, store)
    
    def test_index_and_search(self, hybrid_searcher, sample_chunks):
        hybrid_searcher.index(sample_chunks)
        
        # 검색 실행 (에러 없이 동작하는지 확인)
        results = hybrid_searcher.search("사과", top_k=2)
        
        assert len(results) <= 2
        
    def test_alpha_weight(self, hybrid_searcher, sample_chunks):
        hybrid_searcher.index(sample_chunks)
        
        # alpha=1.0 (Vector only)
        vec_results = hybrid_searcher.search("사과", alpha=1.0)
        
        # alpha=0.0 (BM25 only)
        bm25_results = hybrid_searcher.search("사과", alpha=0.0)
        
        # alpha=0.5 (Hybrid)
        hybrid_results = hybrid_searcher.search("사과", alpha=0.5)
        
        assert isinstance(vec_results, list)
        assert isinstance(bm25_results, list)
        assert isinstance(hybrid_results, list)

    def test_save_and_load(self, hybrid_searcher, sample_chunks, tmp_path: Path):
        hybrid_searcher.index(sample_chunks)
        
        save_path = tmp_path / "hybrid_index"
        hybrid_searcher.save(save_path)
        
        # 로드 테스트
        new_searcher = HybridSearcher(MockEmbedder(), VectorStore(dimension=4))
        new_searcher.load(save_path)
        
        results = new_searcher.search("하늘")
        assert len(results) > 0

    def test_rrf_fusion_basic(self, hybrid_searcher, sample_chunks):
        """RRF 융합 방식 기본 동작 테스트"""
        hybrid_searcher.index(sample_chunks)
        
        # RRF 방식으로 검색
        results = hybrid_searcher.search("사과", top_k=2, fusion_type="rrf")
        
        assert isinstance(results, list)
        assert len(results) <= 2
        # RRF 점수는 0보다 커야 함
        if results:
            assert results[0][1] > 0

    def test_rrf_vs_weighted(self, hybrid_searcher, sample_chunks):
        """RRF와 Weighted 방식 결과 비교 (둘 다 동작해야 함)"""
        hybrid_searcher.index(sample_chunks)
        
        rrf_results = hybrid_searcher.search("과일", fusion_type="rrf")
        weighted_results = hybrid_searcher.search("과일", fusion_type="weighted")
        
        assert isinstance(rrf_results, list)
        assert isinstance(weighted_results, list)

    def test_rrf_k_parameter(self, hybrid_searcher, sample_chunks):
        """RRF k 파라미터 테스트"""
        hybrid_searcher.index(sample_chunks)
        
        # 다른 k 값으로 검색
        results_k60 = hybrid_searcher.search("사과", fusion_type="rrf", rrf_k=60)
        results_k10 = hybrid_searcher.search("사과", fusion_type="rrf", rrf_k=10)
        
        assert isinstance(results_k60, list)
        assert isinstance(results_k10, list)
