"""Embedding 모듈 테스트"""

import numpy as np
import pytest

from rag.embedding.embedder import Embedder


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
