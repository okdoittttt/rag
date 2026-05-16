"""Embedding 모듈 테스트"""

from pathlib import Path

import numpy as np
import pytest

from rag.chunking.chunk import Chunk
from rag.embedding.embedder import Embedder
from rag.embedding.faiss_store import FAISSStore


# 테스트용 임베딩 모델 (매우 작음)
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _unit_vec(values: list[float], dim: int = 4) -> np.ndarray:
    """단위 정규화된 임베딩 벡터 생성 (코사인 유사도용)"""
    v = np.array(values + [0.0] * (dim - len(values)), dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _make_chunk(content: str, source: str, chunk_index: int, user_id: str | None = None) -> Chunk:
    meta = {"source": source, "chunk_index": chunk_index}
    if user_id is not None:
        meta["user_id"] = user_id
    return Chunk(content=content, metadata=meta)


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


class TestFAISSStore:
    """FAISSStore 직접 테스트 (Embedder 의존 없음)"""

    def test_add_and_search(self):
        store = FAISSStore(dimension=4)
        chunks = [
            _make_chunk("apple", "a.txt", 0),
            _make_chunk("banana", "a.txt", 1),
        ]
        embs = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([0, 1, 0])])
        store.add(chunks, embs)
        assert store.total_chunks == 2

        results = store.search(_unit_vec([1, 0, 0]).reshape(1, -1), top_k=1)
        assert len(results) == 1
        assert results[0][0].content == "apple"

    def test_user_id_filter(self):
        store = FAISSStore(dimension=4)
        chunks = [
            _make_chunk("u1 apple", "a.txt", 0, user_id="u1"),
            _make_chunk("u2 apple", "b.txt", 0, user_id="u2"),
        ]
        embs = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([1, 0, 0])])
        store.add(chunks, embs)

        results = store.search(_unit_vec([1, 0, 0]).reshape(1, -1), top_k=5, user_id="u1")
        assert len(results) == 1
        assert results[0][0].metadata["user_id"] == "u1"

    def test_delete_by_source(self):
        store = FAISSStore(dimension=4)
        chunks = [
            _make_chunk("c1", "a.txt", 0, user_id="u1"),
            _make_chunk("c2", "a.txt", 1, user_id="u1"),
            _make_chunk("c3", "b.txt", 0, user_id="u1"),
        ]
        embs = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([0, 1, 0]), _unit_vec([0, 0, 1])])
        store.add(chunks, embs)
        assert store.total_chunks == 3

        deleted = store.delete_by_source("a.txt", user_id="u1")
        assert deleted == 2
        assert store.total_chunks == 1

        # 남은 청크는 b.txt
        results = store.search(_unit_vec([0, 0, 1]).reshape(1, -1), top_k=5)
        assert len(results) == 1
        assert results[0][0].metadata["source"] == "b.txt"

    def test_delete_isolates_users(self):
        """같은 source라도 다른 user_id의 청크는 보존"""
        store = FAISSStore(dimension=4)
        chunks = [
            _make_chunk("u1 doc", "report.pdf", 0, user_id="u1"),
            _make_chunk("u2 doc", "report.pdf", 0, user_id="u2"),
        ]
        embs = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([0, 1, 0])])
        store.add(chunks, embs)

        deleted = store.delete_by_source("report.pdf", user_id="u1")
        assert deleted == 1
        assert store.total_chunks == 1
        # 남은 청크는 u2의 것
        results = store.search(_unit_vec([0, 1, 0]).reshape(1, -1), top_k=5)
        assert results[0][0].metadata["user_id"] == "u2"

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        store = FAISSStore(dimension=4)
        chunks = [_make_chunk("hello", "a.txt", 0, user_id="u1")]
        store.add(chunks, _unit_vec([1, 0, 0]).reshape(1, -1))
        store.save(tmp_path)

        new_store = FAISSStore(dimension=4)
        new_store.load(tmp_path)
        assert new_store.total_chunks == 1

        # 로드 후에도 user_id 필터 동작
        results = new_store.search(_unit_vec([1, 0, 0]).reshape(1, -1), top_k=5, user_id="u1")
        assert len(results) == 1

    def test_reindex_does_not_accumulate(self):
        """delete_by_source 후 같은 청크 재추가 시 총 개수가 유지되어야 함"""
        store = FAISSStore(dimension=4)
        chunks = [_make_chunk("doc", "a.txt", 0, user_id="u1")]
        emb = _unit_vec([1, 0, 0]).reshape(1, -1)

        store.add(chunks, emb)
        store.delete_by_source("a.txt", user_id="u1")
        store.add(chunks, emb)
        assert store.total_chunks == 1
