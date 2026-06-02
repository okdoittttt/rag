"""Reranker 단위 테스트.

CrossEncoder 모델 로드는 비용이 크기 때문에 실제 모델은 사용하지 않고
``CrossEncoder.predict`` 를 mock 으로 치환한다. 정렬, 빈 입력, ``top_k``
초과 입력 등 ``Reranker.rerank`` 의 행위 계약을 검증한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.chunking.chunk import Chunk
from rag.retrieval.reranker import Reranker


def _make_chunk(idx: int, text: str = "content") -> Chunk:
    """간단한 청크 헬퍼."""
    return Chunk.create(
        content=f"{text}-{idx}",
        source=f"doc{idx}.md",
        chunk_index=idx,
        start_char=0,
        end_char=len(text),
    )


@pytest.fixture
def reranker_with_mock_model():
    """모델 로드를 우회한 ``Reranker`` 인스턴스를 yield 한다.

    ``CrossEncoder`` 의 모든 행위는 ``MagicMock`` 으로 대체한다. 테스트에서
    ``predict`` 의 반환값을 케이스별로 지정한다.
    """
    with patch("rag.retrieval.reranker.CrossEncoder") as cross_encoder_cls:
        mock_model = MagicMock()
        cross_encoder_cls.return_value = mock_model
        reranker = Reranker(model_name="fake-model", batch_size=8)
        yield reranker, mock_model


class TestRerankerBehavior:
    """``Reranker.rerank`` 의 핵심 계약 검증."""

    def test_sorts_by_predicted_score_descending(self, reranker_with_mock_model):
        """CrossEncoder 점수 기준 내림차순으로 정렬되어야 한다."""
        reranker, mock_model = reranker_with_mock_model
        chunks = [
            (_make_chunk(0), 0.1),  # 초기 점수는 무시되어야 한다.
            (_make_chunk(1), 0.9),
            (_make_chunk(2), 0.5),
        ]
        # 입력 순서: idx=0,1,2. predict 가 reverse 순으로 점수를 매김.
        mock_model.predict.return_value = np.array([0.2, 0.9, 0.5])

        out = reranker.rerank("q", chunks, top_k=3)
        assert [c.metadata["chunk_index"] for c, _ in out] == [1, 2, 0]
        # 점수도 재정렬된 결과여야 한다.
        assert [pytest.approx(s) for _, s in out] == [
            pytest.approx(0.9),
            pytest.approx(0.5),
            pytest.approx(0.2),
        ]

    def test_returns_empty_on_empty_input(self, reranker_with_mock_model):
        """입력이 비면 모델 호출 없이 빈 리스트를 반환."""
        reranker, mock_model = reranker_with_mock_model
        assert reranker.rerank("q", [], top_k=5) == []
        mock_model.predict.assert_not_called()

    def test_top_k_truncates_results(self, reranker_with_mock_model):
        """``top_k`` 가 입력 길이보다 작으면 상위만 반환."""
        reranker, mock_model = reranker_with_mock_model
        chunks = [(_make_chunk(i), 0.0) for i in range(5)]
        mock_model.predict.return_value = np.array([0.1, 0.5, 0.9, 0.2, 0.7])

        out = reranker.rerank("q", chunks, top_k=2)
        assert len(out) == 2
        # 가장 큰 두 점수(0.9, 0.7)에 대응하는 chunk_index 는 2, 4.
        assert [c.metadata["chunk_index"] for c, _ in out] == [2, 4]

    def test_top_k_greater_than_input_returns_all(self, reranker_with_mock_model):
        """``top_k`` 가 입력보다 크면 모두 반환."""
        reranker, mock_model = reranker_with_mock_model
        chunks = [(_make_chunk(0), 0.0), (_make_chunk(1), 0.0)]
        mock_model.predict.return_value = np.array([0.3, 0.7])

        out = reranker.rerank("q", chunks, top_k=10)
        assert len(out) == 2

    def test_passes_batch_size_to_predict(self, reranker_with_mock_model):
        """``predict`` 호출 시 인스턴스의 ``batch_size`` 가 전달되어야 한다."""
        reranker, mock_model = reranker_with_mock_model
        chunks = [(_make_chunk(0), 0.0)]
        mock_model.predict.return_value = np.array([0.5])

        reranker.rerank("q", chunks, top_k=1)
        _, kwargs = mock_model.predict.call_args
        assert kwargs.get("batch_size") == 8


class TestRerankerInit:
    """초기화 시 ``CrossEncoder`` 에 device 가 전달되는지 확인."""

    def test_passes_device_to_cross_encoder(self):
        with patch("rag.retrieval.reranker.CrossEncoder") as cross_encoder_cls:
            cross_encoder_cls.return_value = MagicMock()
            Reranker(model_name="fake-model", device="cuda")
            args, kwargs = cross_encoder_cls.call_args
            assert kwargs.get("device") == "cuda"
            assert args[0] == "fake-model"
