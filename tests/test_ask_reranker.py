"""``/ask`` 라우트의 reranker 분기 및 폴백 동작 테스트.

플랜 03 의 핵심 행동 계약을 회귀 방지하기 위해 다음을 검증한다:

1. ``rerank=True`` 일 때 1차 검색의 후보 수가 ``max(top_k*5, 20)`` 로 보강된다.
2. ``rerank=False`` 일 때는 ``top_k*2`` 가 유지된다.
3. ``get_reranker_instance`` 가 ``None`` 을 반환하면 reranker 없이 검색 결과가
   그대로 사용된다(예외 없이 폴백).
4. ``get_reranker_instance`` 가 성공적으로 인스턴스를 반환하면 그 ``rerank``
   메서드가 호출되어 결과가 대체된다.
5. 모델 로드 실패 시 ``get_reranker_instance`` 가 음성 캐시(``None``)를 유지하고
   재시도하지 않는다.

실제 임베딩/벡터 스토어/Cross-Encoder 는 사용하지 않으며 monkeypatch 로
모두 대체한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from api.schemas import AskRequest
from rag.chunking.chunk import Chunk


def _chunk(idx: int, source: str = "doc.md") -> Chunk:
    return Chunk.create(
        content=f"{source}-{idx}",
        source=source,
        chunk_index=idx,
        start_char=0,
        end_char=10,
    )


@pytest.fixture
def ask_module(monkeypatch, tmp_path):
    """싱글톤이 정리된 ask 라우터 모듈."""
    monkeypatch.setenv("RAG_PROJECT_INDEX_PATH", str(tmp_path))

    from rag import config as config_module

    config_module.get_config.cache_clear()

    from api.routes import ask as ask_module  # noqa: WPS433 — late import

    ask_module._searcher = None
    ask_module._reranker = None
    ask_module._reranker_load_failed = False
    yield ask_module
    ask_module._searcher = None
    ask_module._reranker = None
    ask_module._reranker_load_failed = False
    config_module.get_config.cache_clear()


def _install_searcher(monkeypatch, ask_module, search_chunks):
    """``get_searcher`` 가 반환할 fake searcher 를 주입."""
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = search_chunks
    fake_searcher.fetch_full_document.return_value = []
    monkeypatch.setattr(ask_module, "get_searcher", lambda: fake_searcher)
    return fake_searcher


class TestCandidatePoolSizing:
    """1차 검색 후보 수가 reranker 사용 여부에 따라 다르게 전달되는지 검증."""

    def test_rerank_true_uses_at_least_20_candidates(self, ask_module, monkeypatch):
        """``rerank=True`` 면 top_k 가 작아도 최소 20 개를 가져온다."""
        fake = _install_searcher(
            monkeypatch,
            ask_module,
            search_chunks=[(_chunk(i), 1.0 - i * 0.01) for i in range(30)],
        )
        # rerank 활성. 인스턴스는 None 폴백시켜 호출 영향을 배제.
        monkeypatch.setattr(ask_module, "get_reranker_instance", lambda: None)

        req = AskRequest(query="q", top_k=2, rerank=True)
        ask_module._search_documents(req)

        kwargs = fake.search.call_args.kwargs
        # top_k * 5 = 10 이지만 최소 20 이 적용되어야 한다.
        assert kwargs["top_k"] == 20

    def test_rerank_true_scales_with_top_k(self, ask_module, monkeypatch):
        """top_k 가 크면 ``top_k*5`` 가 적용된다(최소값 미달 시에만 20 적용)."""
        fake = _install_searcher(
            monkeypatch,
            ask_module,
            search_chunks=[(_chunk(i), 1.0 - i * 0.01) for i in range(50)],
        )
        monkeypatch.setattr(ask_module, "get_reranker_instance", lambda: None)

        req = AskRequest(query="q", top_k=10, rerank=True)
        ask_module._search_documents(req)
        kwargs = fake.search.call_args.kwargs
        assert kwargs["top_k"] == 50

    def test_rerank_false_uses_double_top_k(self, ask_module, monkeypatch):
        """``rerank=False`` 면 ``top_k*2`` 가 유지된다(회귀 방지)."""
        fake = _install_searcher(
            monkeypatch,
            ask_module,
            search_chunks=[(_chunk(i), 1.0 - i * 0.01) for i in range(10)],
        )
        req = AskRequest(query="q", top_k=5, rerank=False)
        ask_module._search_documents(req)
        kwargs = fake.search.call_args.kwargs
        assert kwargs["top_k"] == 10


class TestRerankerFallback:
    """Reranker 로드 실패 시 폴백 동작."""

    def test_none_reranker_does_not_raise(self, ask_module, monkeypatch):
        """``get_reranker_instance`` 가 None 을 반환하면 검색 결과를 그대로 사용."""
        scored = [(_chunk(i), 1.0 - i * 0.1) for i in range(3)]
        _install_searcher(monkeypatch, ask_module, search_chunks=scored)
        monkeypatch.setattr(ask_module, "get_reranker_instance", lambda: None)

        req = AskRequest(query="q", top_k=3, rerank=True)
        chunks, unique = ask_module._search_documents(req)

        # reranker 가 없어도 예외 없이 검색 결과가 그대로 전달된다.
        assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]
        assert unique == scored

    def test_active_reranker_replaces_results(self, ask_module, monkeypatch):
        """Reranker 가 정상 동작하면 그 결과가 최종 응답에 반영된다."""
        scored = [(_chunk(i), 0.5) for i in range(3)]
        _install_searcher(monkeypatch, ask_module, search_chunks=scored)

        # Reranker 가 점수를 뒤집어 반환한다고 가정.
        fake_reranker = MagicMock()
        fake_reranker.rerank.return_value = [(scored[2][0], 0.9), (scored[0][0], 0.7)]
        monkeypatch.setattr(ask_module, "get_reranker_instance", lambda: fake_reranker)

        req = AskRequest(query="q", top_k=2, rerank=True)
        chunks, unique = ask_module._search_documents(req)

        fake_reranker.rerank.assert_called_once()
        assert [c.metadata["chunk_index"] for c in chunks] == [2, 0]
        assert unique[0][1] == 0.9


class TestGetRerankerInstanceLoadFailure:
    """``get_reranker_instance`` 의 음성 캐시 및 폴백."""

    def test_load_failure_returns_none_and_caches(self, ask_module, monkeypatch):
        """Reranker 생성자가 예외를 던지면 None 반환 + 재시도 안 함."""
        # CrossEncoder 자체가 아닌, 모듈 내 ``Reranker`` 클래스를 직접 패치한다.
        attempts = {"count": 0}

        def _boom(*args, **kwargs):
            attempts["count"] += 1
            raise RuntimeError("model download failed")

        monkeypatch.setattr(ask_module, "Reranker", _boom)

        first = ask_module.get_reranker_instance()
        second = ask_module.get_reranker_instance()

        assert first is None
        assert second is None
        # 음성 캐시로 인해 한 번만 시도되어야 한다.
        assert attempts["count"] == 1
        assert ask_module._reranker_load_failed is True
