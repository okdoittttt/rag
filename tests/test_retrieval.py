"""Retrieval 모듈 테스트"""

from pathlib import Path

import numpy as np
import pytest

from rag.chunking.chunk import Chunk
from rag.embedding.embedder import Embedder
from rag.embedding.base import VectorStoreBase
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


class InMemoryVectorStore(VectorStoreBase):
    """테스트용 인메모리 벡터 저장소 (FAISS 대체)"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not self.chunks or self.embeddings is None:
            return []
        scores = (query_embedding @ self.embeddings.T)[0]
        order = np.argsort(scores)[::-1]
        results: list[tuple[Chunk, float]] = []
        for i in order:
            if scores[i] <= 0:
                continue
            chunk = self.chunks[i]
            if user_id is not None and chunk.metadata.get("user_id") != user_id:
                continue
            results.append((chunk, float(scores[i])))
            if len(results) >= top_k:
                break
        return results

    def delete_by_source(self, source: str, user_id: str | None = None) -> int:
        keep_indices: list[int] = []
        deleted = 0
        for i, c in enumerate(self.chunks):
            if c.metadata.get("source") != source:
                keep_indices.append(i)
                continue
            chunk_user = c.metadata.get("user_id")
            matches = (chunk_user == user_id) if user_id is not None else (chunk_user in (None, ""))
            if matches:
                deleted += 1
            else:
                keep_indices.append(i)

        if deleted == 0:
            return 0

        self.chunks = [self.chunks[i] for i in keep_indices]
        if self.embeddings is not None and keep_indices:
            self.embeddings = self.embeddings[keep_indices]
        elif not keep_indices:
            self.embeddings = None
        return deleted

    def save(self, path: str | Path) -> None:
        pass

    def load(self, path: str | Path) -> None:
        pass

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks = []
        self.embeddings = None


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
        store = InMemoryVectorStore(dimension=4)
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
        new_searcher = HybridSearcher(MockEmbedder(), InMemoryVectorStore(dimension=4))
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


@pytest.fixture
def multi_user_chunks():
    """사용자 두 명, 문서 두 개로 구성된 청크 묶음"""
    return [
        Chunk(content="사과는 과일이다", metadata={"source": "a.txt", "chunk_index": 0, "user_id": "u1"}),
        Chunk(content="바나나는 노랗다", metadata={"source": "a.txt", "chunk_index": 1, "user_id": "u1"}),
        Chunk(content="하늘은 파랗다", metadata={"source": "b.txt", "chunk_index": 0, "user_id": "u2"}),
        Chunk(content="우주는 검다", metadata={"source": "b.txt", "chunk_index": 1, "user_id": "u2"}),
    ]


class TestBM25UserFilter:
    """BM25 user_id 격리 회귀 테스트"""

    def test_search_filters_by_user_id(self, multi_user_chunks):
        searcher = BM25Searcher()
        searcher.index(multi_user_chunks)

        u1_results = searcher.search("사과", top_k=10, user_id="u1")
        for chunk, _ in u1_results:
            assert chunk.metadata["user_id"] == "u1"

        u2_results = searcher.search("하늘", top_k=10, user_id="u2")
        for chunk, _ in u2_results:
            assert chunk.metadata["user_id"] == "u2"

    def test_no_filter_returns_all_matches(self, multi_user_chunks):
        searcher = BM25Searcher()
        searcher.index(multi_user_chunks)
        results = searcher.search("사과", top_k=10)
        # user_id 미지정이면 양쪽 사용자 모두 노출
        assert any(c.metadata["user_id"] == "u1" for c, _ in results)


class TestBM25Cumulative:
    """index()는 누적이어야 함 (vector_store.add()와 일관)"""

    def test_index_is_additive(self):
        searcher = BM25Searcher()
        searcher.index([Chunk(content="첫 번째 문서", metadata={"source": "a.txt", "chunk_index": 0})])
        searcher.index([Chunk(content="두 번째 문서", metadata={"source": "b.txt", "chunk_index": 0})])
        assert len(searcher.chunks) == 2


class TestBM25Delete:
    """BM25 delete_by_source 테스트"""

    def test_delete_removes_only_matching(self, multi_user_chunks):
        searcher = BM25Searcher()
        searcher.index(multi_user_chunks)
        deleted = searcher.delete_by_source("a.txt", user_id="u1")
        assert deleted == 2
        # u2의 b.txt 청크는 남아있어야 함 (BM25 IDF는 작은 코퍼스에서 0이 될 수 있으므로
        # 검색이 아닌 내부 청크 리스트로 검증)
        remaining_sources = {c.metadata["source"] for c in searcher.chunks}
        assert remaining_sources == {"b.txt"}
        remaining_users = {c.metadata["user_id"] for c in searcher.chunks}
        assert remaining_users == {"u2"}

    def test_delete_returns_zero_when_no_match(self, multi_user_chunks):
        searcher = BM25Searcher()
        searcher.index(multi_user_chunks)
        deleted = searcher.delete_by_source("nonexistent.txt", user_id="u1")
        assert deleted == 0


class TestHybridUserIsolation:
    """HybridSearcher 멀티 문서/사용자 격리 회귀 테스트"""

    def _make(self):
        embedder = MockEmbedder()
        store = InMemoryVectorStore(dimension=4)
        return HybridSearcher(embedder, store)

    def test_weighted_isolates_by_user(self, multi_user_chunks):
        searcher = self._make()
        searcher.index(multi_user_chunks)
        results = searcher.search("사과", top_k=10, user_id="u1", fusion_type="weighted")
        for chunk, _ in results:
            assert chunk.metadata["user_id"] == "u1"

    def test_rrf_isolates_by_user(self, multi_user_chunks):
        searcher = self._make()
        searcher.index(multi_user_chunks)
        results = searcher.search("사과", top_k=10, user_id="u1", fusion_type="rrf")
        for chunk, _ in results:
            assert chunk.metadata["user_id"] == "u1"

    def test_weighted_bm25_mapping_multi_doc(self):
        """chunk_index 매핑 버그 회귀 방지

        멀티 문서에서 chunk_index가 0으로 충돌해도 (source, chunk_index)로
        올바른 BM25 점수가 매핑되어야 한다.
        """
        searcher = self._make()
        chunks = [
            Chunk(content="고유한 키워드 알파", metadata={"source": "a.txt", "chunk_index": 0}),
            Chunk(content="다른 내용 베타", metadata={"source": "b.txt", "chunk_index": 0}),
            Chunk(content="또 다른 내용 감마", metadata={"source": "c.txt", "chunk_index": 0}),
        ]
        searcher.index(chunks)
        # "알파" 검색 시 a.txt:0 청크가 가장 높은 점수를 받아야 함
        results = searcher.search("알파", top_k=3, fusion_type="weighted", alpha=0.0)
        assert results
        # alpha=0이면 BM25 점수만 사용; 정렬 1위가 "알파" 포함 청크여야 함
        top_content = results[0][0].content
        assert "알파" in top_content


class TestHybridDelete:
    """HybridSearcher.delete_by_source 통합 테스트"""

    def test_delete_both_stores(self, multi_user_chunks):
        embedder = MockEmbedder()
        store = InMemoryVectorStore(dimension=4)
        searcher = HybridSearcher(embedder, store)
        searcher.index(multi_user_chunks)

        assert store.total_chunks == 4
        deleted = searcher.delete_by_source("a.txt", user_id="u1")
        assert deleted == 2
        assert store.total_chunks == 2
        # 남은 청크는 모두 u2/b.txt
        for c in store.chunks:
            assert c.metadata["user_id"] == "u2"
            assert c.metadata["source"] == "b.txt"

    def test_reindex_does_not_accumulate(self, multi_user_chunks):
        """같은 파일을 두 번 인덱싱해도 청크가 누적되지 않아야 함"""
        embedder = MockEmbedder()
        store = InMemoryVectorStore(dimension=4)
        searcher = HybridSearcher(embedder, store)

        u1_chunks = [c for c in multi_user_chunks if c.metadata["user_id"] == "u1"]
        searcher.index(u1_chunks)
        assert store.total_chunks == 2

        # 같은 source/user_id로 다시 인덱싱 → 이전 청크 삭제 후 추가
        searcher.delete_by_source("a.txt", user_id="u1")
        searcher.index(u1_chunks)
        assert store.total_chunks == 2
