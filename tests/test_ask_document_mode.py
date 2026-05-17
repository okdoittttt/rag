"""문서 모드(``doc_mode`` + ``source_filter``) 검색 분기 테스트.

`_search_documents` 가 다음 조건에 따라 분기되는지 검증한다:

1. ``doc_mode=True`` + ``source_filter`` + 요약 의도 → ``fetch_full_document``
   로 우회되어 문서 전체 청크가 LLM 에 투입된다.
2. ``doc_mode=True`` + ``source_filter`` + 일반 사실 질의 → 기존 hybrid
   검색 경로 유지.
3. ``doc_mode=False`` → 기존 경로 (회귀 방지).
4. ``summarize_override=True/False`` 는 휴리스틱을 무시하고 강제.

실제 Qdrant/임베딩은 사용하지 않고 ``HybridSearcher`` 를 MagicMock 으로
치환하며, 인덱스 존재 검사는 ``tmp_path`` 와 ``RAG_PROJECT_INDEX_PATH``
환경변수로 우회한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from api.schemas import AskRequest
from rag.chunking.chunk import Chunk


def _chunk(idx: int, source: str, content: str = "x") -> Chunk:
    """간단한 청크 헬퍼."""
    return Chunk.create(
        content=content,
        source=source,
        chunk_index=idx,
        start_char=0,
        end_char=len(content),
    )


@pytest.fixture
def ask_module(monkeypatch, tmp_path):
    """index_path 존재 검사를 우회하고 싱글톤을 초기화한 ask 모듈.

    - ``RAG_PROJECT_INDEX_PATH`` 를 ``tmp_path`` 로 지정해 인덱스가 있는
      것처럼 위장한다.
    - ``rag.config.get_config`` 캐시를 비워 환경변수 변경이 반영되도록 한다.
    - 모듈 로딩 후 싱글톤(`_searcher`, `_reranker`)을 정리한다.
    """
    monkeypatch.setenv("RAG_PROJECT_INDEX_PATH", str(tmp_path))

    from rag import config as config_module

    config_module.get_config.cache_clear()

    from api.routes import ask as ask_module  # noqa: WPS433 — late import

    ask_module._searcher = None
    ask_module._reranker = None
    yield ask_module
    ask_module._searcher = None
    ask_module._reranker = None
    config_module.get_config.cache_clear()


def _install_searcher(monkeypatch, ask_module, *, search_chunks, full_doc_chunks):
    """``get_searcher`` 가 반환할 MagicMock 검색기를 주입한다.

    Args:
        monkeypatch: pytest fixture.
        ask_module: ask 라우터 모듈.
        search_chunks: ``searcher.search`` 가 반환할 (chunk, score) 리스트.
        full_doc_chunks: ``fetch_full_document`` 가 반환할 청크 리스트.

    Returns:
        주입한 ``MagicMock`` 인스턴스.
    """
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = search_chunks
    fake_searcher.fetch_full_document.return_value = full_doc_chunks
    monkeypatch.setattr(ask_module, "get_searcher", lambda: fake_searcher)
    return fake_searcher


class TestSummaryModeRouting:
    """문서 요약 모드로의 분기 검증."""

    def test_summary_intent_triggers_fetch_full_document(self, ask_module, monkeypatch):
        full_doc = [_chunk(i, "doc.md") for i in range(7)]
        fake = _install_searcher(
            monkeypatch, ask_module, search_chunks=[], full_doc_chunks=full_doc,
        )
        req = AskRequest(
            query="이 문서를 요약해줘",
            source_filter="doc.md",
            doc_mode=True,
        )

        chunks, scored = ask_module._search_documents(req)

        fake.fetch_full_document.assert_called_once()
        kwargs = fake.fetch_full_document.call_args.kwargs
        assert kwargs["source"] == "doc.md"
        assert kwargs["user_id"] is None
        assert kwargs["max_chunks"] >= 1
        assert [c.metadata["chunk_index"] for c in chunks] == list(range(7))
        # 점수는 1.0 으로 채워져야 한다(검색이 아닌 조회).
        assert all(s == 1.0 for _, s in scored)
        # 검색 메서드는 호출되지 않아야 한다.
        fake.search.assert_not_called()

    def test_explicit_override_true_bypasses_heuristic(self, ask_module, monkeypatch):
        full_doc = [_chunk(0, "doc.md")]
        fake = _install_searcher(
            monkeypatch, ask_module, search_chunks=[], full_doc_chunks=full_doc,
        )
        req = AskRequest(
            # 휴리스틱으론 음성으로 분류되는 단순 사실 질의
            query="이 함수의 인자는?",
            source_filter="doc.md",
            doc_mode=True,
            summarize_override=True,
        )

        ask_module._search_documents(req)
        fake.fetch_full_document.assert_called_once()
        fake.search.assert_not_called()

    def test_explicit_override_false_forces_search(self, ask_module, monkeypatch):
        fake = _install_searcher(
            monkeypatch,
            ask_module,
            search_chunks=[(_chunk(0, "doc.md"), 0.9)],
            full_doc_chunks=[_chunk(0, "doc.md")],
        )
        req = AskRequest(
            # 휴리스틱으론 양성. override 가 False 이므로 강제로 일반 경로.
            query="이 문서를 요약해줘",
            source_filter="doc.md",
            doc_mode=True,
            summarize_override=False,
        )

        ask_module._search_documents(req)
        fake.search.assert_called_once()
        fake.fetch_full_document.assert_not_called()


class TestRegularSearchPath:
    """기본 hybrid 검색 경로가 유지되는지 검증."""

    def test_doc_mode_without_summary_intent_uses_search(self, ask_module, monkeypatch):
        fake = _install_searcher(
            monkeypatch,
            ask_module,
            search_chunks=[(_chunk(3, "doc.md"), 0.8)],
            full_doc_chunks=[],
        )
        req = AskRequest(
            query="BM25 점수는 어떻게 계산되나요?",
            source_filter="doc.md",
            doc_mode=True,
        )

        chunks, scored = ask_module._search_documents(req)
        fake.search.assert_called_once()
        fake.fetch_full_document.assert_not_called()
        assert chunks[0].metadata["chunk_index"] == 3
        assert scored[0][1] == 0.8

    def test_doc_mode_false_uses_search(self, ask_module, monkeypatch):
        fake = _install_searcher(
            monkeypatch,
            ask_module,
            search_chunks=[(_chunk(0, "doc.md"), 0.5)],
            full_doc_chunks=[],
        )
        # 요약 의도가 있어도 doc_mode=False 면 일반 경로.
        req = AskRequest(
            query="이 문서를 요약해줘",
            source_filter="doc.md",
            doc_mode=False,
        )

        ask_module._search_documents(req)
        fake.search.assert_called_once()
        fake.fetch_full_document.assert_not_called()

    def test_doc_mode_without_source_filter_uses_search(self, ask_module, monkeypatch):
        fake = _install_searcher(
            monkeypatch,
            ask_module,
            search_chunks=[(_chunk(0, "any.md"), 0.5)],
            full_doc_chunks=[],
        )
        # source_filter 미지정이면 어느 문서를 풀로 가져올지 모호 → 일반 경로.
        req = AskRequest(
            query="이 문서를 요약해줘",
            source_filter=None,
            doc_mode=True,
        )

        ask_module._search_documents(req)
        fake.search.assert_called_once()
        fake.fetch_full_document.assert_not_called()


class TestDedupeKey:
    """``_search_documents`` 의 dedupe 키가 ``(source, chunk_index)`` 튜플인지 검증.

    동일 ``chunk_index`` 를 가진 두 청크가 서로 다른 ``source`` 에서 나오면
    둘 다 결과에 살아남아야 한다(과거에는 chunk_index 단독 키로 충돌 제거).
    """

    def test_same_index_different_source_not_deduped(self, ask_module, monkeypatch):
        same_idx_a = (_chunk(1, "a.md", "from-a"), 0.9)
        same_idx_b = (_chunk(1, "b.md", "from-b"), 0.8)
        _install_searcher(
            monkeypatch,
            ask_module,
            search_chunks=[same_idx_a, same_idx_b],
            full_doc_chunks=[],
        )
        req = AskRequest(query="hybrid 검색?", top_k=5)

        chunks, _ = ask_module._search_documents(req)
        sources = sorted(c.metadata["source"] for c in chunks)
        assert sources == ["a.md", "b.md"]
