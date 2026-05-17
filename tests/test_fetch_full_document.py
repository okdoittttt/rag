"""문서 전체 조회 경로 테스트.

``FAISSStore.get_all_by_source`` 와 ``HybridSearcher.fetch_full_document``
가 source/user_id 필터링, chunk_index 정렬, limit 컷오프를 올바르게
수행하는지 검증한다. Qdrant 는 네트워크 의존이라 별도 통합 테스트로
분리한다.
"""

from __future__ import annotations

import numpy as np

from rag.chunking.chunk import Chunk
from rag.embedding.faiss_store import FAISSStore
from rag.retrieval.searcher import HybridSearcher


def _make_chunks(spec: list[tuple[str, int, str | None]]) -> list[Chunk]:
    """간단한 청크 빌더.

    Args:
        spec: ``(source, chunk_index, user_id)`` 튜플 리스트.

    Returns:
        ``Chunk`` 리스트.
    """
    chunks: list[Chunk] = []
    for src, idx, uid in spec:
        c = Chunk.create(
            content=f"{src}-{idx}",
            source=src,
            chunk_index=idx,
            start_char=0,
            end_char=10,
        )
        if uid is not None:
            c.metadata["user_id"] = uid
        chunks.append(c)
    return chunks


def _make_store(chunks: list[Chunk], dim: int = 4) -> FAISSStore:
    """결정적인 임베딩으로 채워진 FAISS 저장소."""
    store = FAISSStore(dimension=dim)
    embeddings = np.ones((len(chunks), dim), dtype=np.float32)
    store.add(chunks, embeddings)
    return store


class TestFaissGetAllBySource:
    """``FAISSStore.get_all_by_source`` 정렬/필터 검증."""

    def test_returns_sorted_by_chunk_index(self) -> None:
        chunks = _make_chunks(
            [
                ("a.md", 2, None),
                ("a.md", 0, None),
                ("a.md", 1, None),
                ("b.md", 0, None),
            ]
        )
        store = _make_store(chunks)
        out = store.get_all_by_source("a.md")
        assert [c.metadata["chunk_index"] for c in out] == [0, 1, 2]
        assert all(c.metadata["source"] == "a.md" for c in out)

    def test_user_id_isolation(self) -> None:
        chunks = _make_chunks(
            [
                ("a.md", 0, ""),       # anonymous
                ("a.md", 1, "alice"),  # alice
                ("a.md", 2, "bob"),    # bob
            ]
        )
        store = _make_store(chunks)

        anon = store.get_all_by_source("a.md", user_id=None)
        assert [c.metadata["chunk_index"] for c in anon] == [0]

        alice = store.get_all_by_source("a.md", user_id="alice")
        assert [c.metadata["chunk_index"] for c in alice] == [1]

    def test_limit_truncates(self) -> None:
        chunks = _make_chunks([("a.md", i, None) for i in range(10)])
        store = _make_store(chunks)
        out = store.get_all_by_source("a.md", limit=3)
        assert [c.metadata["chunk_index"] for c in out] == [0, 1, 2]

    def test_no_match_returns_empty(self) -> None:
        chunks = _make_chunks([("a.md", 0, None)])
        store = _make_store(chunks)
        assert store.get_all_by_source("missing.md") == []


class TestHybridSearcherFetchFullDocument:
    """``HybridSearcher.fetch_full_document`` 가 저장소 호출을 위임하는지 검증."""

    def test_delegates_to_vector_store(self) -> None:
        chunks = _make_chunks(
            [
                ("a.md", 1, None),
                ("a.md", 0, None),
                ("b.md", 0, None),
            ]
        )
        store = _make_store(chunks)
        # Embedder 는 사용되지 않으므로 None 전달이 가능하지만, 타입 보존을
        # 위해 가벼운 더미를 둔다.
        searcher = HybridSearcher.__new__(HybridSearcher)
        searcher.embedder = None  # type: ignore[assignment]
        searcher.vector_store = store
        searcher.bm25 = None  # type: ignore[assignment]

        out = searcher.fetch_full_document("a.md", max_chunks=10)
        assert [c.metadata["chunk_index"] for c in out] == [0, 1]
