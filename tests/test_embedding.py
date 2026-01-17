"""Embedding 모듈 테스트"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from rag.chunking.chunk import Chunk
from rag.embedding.embedder import Embedder
from rag.embedding.store import VectorStore


# 테스트용 임베딩 모델 (매우 작음)
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def embedder():
    """모듈 레벨에서 한 번만 모델 로딩"""
    return Embedder(model_name=TEST_MODEL)


class TestEmbedder:
    """Embedder 테스트"""
    
    def test_embed_single_text(self, embedder):
        """단일 텍스트 임베딩"""
        texts = ["Hello world"]
        embeddings = embedder.embed(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 1
        assert embeddings.shape[1] == 384  # MiniLM 차원
    
    def test_embed_multiple_texts(self, embedder):
        """다중 텍스트 임베딩"""
        texts = ["Hello", "World", "Python"]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 3
        assert embeddings.shape[1] == 384
    
    def test_embed_empty_list(self, embedder):
        """빈 리스트 처리"""
        embeddings = embedder.embed([])
        
        assert len(embeddings) == 0
    
    def test_embed_query(self, embedder):
        """쿼리 임베딩"""
        embedding = embedder.embed_query("search query")
        
        assert len(embedding) == 1
        assert embedding.shape[1] == 384


class TestVectorStore:
    """VectorStore 테스트"""
    
    @pytest.fixture
    def store(self):
        """384차원 저장소"""
        return VectorStore(dimension=384)
    
    @pytest.fixture
    def sample_data(self):
        """검색 테스트용 데이터"""
        chunks = [
            Chunk(content="Apple is a fruit", metadata={"id": 1}),
            Chunk(content="Banana is yellow", metadata={"id": 2}),
            Chunk(content="Sky is blue", metadata={"id": 3}),
        ]
        
        # 임의의 직교 벡터 생성 (검색 결과 명확성을 위해)
        # 실제로는 embedder를 써야 하지만, 여기선 store 로직만 테스트
        embeddings = np.array([
            [1.0, 0.0, 0.0] + [0.0] * 381,  # Apple
            [0.5, 0.5, 0.0] + [0.0] * 381,  # Banana (Apple과 약간 유사)
            [0.0, 0.0, 1.0] + [0.0] * 381,  # Sky
        ], dtype=np.float32)
        
        return chunks, embeddings
    
    def test_add_and_search(self, store, sample_data):
        """추가 및 검색 확인"""
        chunks, embeddings = sample_data
        store.add(chunks, embeddings)
        
        assert store.total_chunks == 3
        
        # Apple 검색 (첫 번째 벡터와 동일한 쿼리)
        query = np.array([[1.0, 0.0, 0.0] + [0.0] * 381], dtype=np.float32)
        results = store.search(query, top_k=1)
        
        assert len(results) == 1
        chunk, score = results[0]
        assert chunk.content == "Apple is a fruit"
        assert score > 0.99  # 자기 자신과 유사도 1.0
        
    def test_search_top_k(self, store, sample_data):
        """top_k 검색 확인"""
        chunks, embeddings = sample_data
        store.add(chunks, embeddings)
        
        # 모든 벡터 검색
        query = np.array([[1.0, 0.0, 0.0] + [0.0] * 381], dtype=np.float32)
        results = store.search(query, top_k=3)
        
        assert len(results) == 3
        # 점수 내림차순 정렬 확인
        assert results[0][1] >= results[1][1]
        
    def test_dimension_mismatch_error(self, store):
        """차원 불일치 에러"""
        chunks = [Chunk(content="test")]
        # 384차원인데 2차원 벡터 입력
        embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
        
        with pytest.raises(ValueError, match="dimension"):
            store.add(chunks, embeddings)
            
    def test_save_and_load(self, store, sample_data, tmp_path: Path):
        """저장 및 로드 확인"""
        chunks, embeddings = sample_data
        store.add(chunks, embeddings)
        
        save_dir = tmp_path / "index_data"
        store.save(save_dir)
        
        # 파일 생성 확인
        assert (save_dir / "faiss.index").exists()
        assert (save_dir / "chunks.pkl").exists()
        assert (save_dir / "meta.json").exists()
        
        # 새로운 저장소로 로드
        new_store = VectorStore(dimension=384)
        new_store.load(save_dir)
        
        assert new_store.total_chunks == 3
        assert new_store.chunks[0].content == "Apple is a fruit"
