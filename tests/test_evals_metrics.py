"""evals.metrics 단위 테스트.

LLM/검색기를 거치지 않고 결정적으로 동작하는 메트릭들이라 외부 mock
없이 청크 fixture만으로 검증한다.
"""

from __future__ import annotations

import pytest

from evals.metrics import (
    compute_search_metrics,
    mrr,
    precision_at_k,
    recall_at_k,
    source_recall,
)
from rag.chunking.chunk import Chunk


def _chunk(idx: int, source: str = "a.md") -> Chunk:
    """테스트용 청크 헬퍼.

    Args:
        idx: ``chunk_index`` 값.
        source: ``source`` 메타데이터.

    Returns:
        간단한 본문/메타데이터를 갖는 ``Chunk``.
    """
    return Chunk.create(
        content=f"chunk-{idx}",
        source=source,
        chunk_index=idx,
        start_char=0,
        end_char=10,
    )


class TestRecallAtK:
    """``recall_at_k`` 의 경계와 정확도 검증."""

    def test_returns_zero_when_expected_empty(self) -> None:
        retrieved = [_chunk(1), _chunk(2)]
        assert recall_at_k(retrieved, [], k=5) == 0.0

    def test_full_recall_when_all_expected_in_top_k(self) -> None:
        retrieved = [_chunk(3), _chunk(1), _chunk(5)]
        assert recall_at_k(retrieved, [1, 3], k=3) == 1.0

    def test_partial_recall(self) -> None:
        retrieved = [_chunk(3), _chunk(1), _chunk(5)]
        # expected={1, 4} 중 1만 hit → 0.5
        assert recall_at_k(retrieved, [1, 4], k=3) == 0.5

    def test_truncates_to_top_k(self) -> None:
        retrieved = [_chunk(9), _chunk(1)]
        # k=1 이므로 chunk 1은 후보에 들어오지 않음
        assert recall_at_k(retrieved, [1], k=1) == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        retrieved = [_chunk(1)]
        assert recall_at_k(retrieved, [1], k=10) == 1.0

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k([_chunk(1)], [1], k=0)


class TestPrecisionAtK:
    """``precision_at_k`` 의 동작 검증."""

    def test_all_relevant(self) -> None:
        retrieved = [_chunk(1), _chunk(2)]
        assert precision_at_k(retrieved, [1, 2], k=2) == 1.0

    def test_partial(self) -> None:
        retrieved = [_chunk(1), _chunk(99)]
        assert precision_at_k(retrieved, [1], k=2) == 0.5

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], [1], k=5) == 0.0


class TestMRR:
    """``mrr`` 검증."""

    def test_first_hit_rank_one(self) -> None:
        retrieved = [_chunk(7), _chunk(2)]
        assert mrr(retrieved, [7]) == 1.0

    def test_second_position(self) -> None:
        retrieved = [_chunk(99), _chunk(7)]
        assert mrr(retrieved, [7]) == pytest.approx(0.5)

    def test_not_found_returns_zero(self) -> None:
        retrieved = [_chunk(1), _chunk(2)]
        assert mrr(retrieved, [99]) == 0.0

    def test_empty_expected_returns_zero(self) -> None:
        assert mrr([_chunk(1)], []) == 0.0


class TestSourceRecall:
    """``source_recall`` 검증."""

    def test_basic(self) -> None:
        retrieved = [_chunk(0, "a.md"), _chunk(1, "b.md")]
        assert source_recall(retrieved, ["a.md", "b.md"]) == 1.0

    def test_partial(self) -> None:
        retrieved = [_chunk(0, "a.md")]
        assert source_recall(retrieved, ["a.md", "b.md"]) == 0.5

    def test_empty_expected(self) -> None:
        assert source_recall([_chunk(0)], []) == 0.0


class TestComputeSearchMetrics:
    """``compute_search_metrics`` 통합 동작 검증."""

    def test_all_keys_present(self) -> None:
        retrieved = [_chunk(1, "a.md"), _chunk(2, "b.md")]
        metrics = compute_search_metrics(
            retrieved=retrieved,
            expected_chunk_ids=[1],
            expected_sources=["a.md"],
            k_list=(1, 3),
        )
        assert "recall_at_1" in metrics
        assert "recall_at_3" in metrics
        assert "precision_at_1" in metrics
        assert "precision_at_3" in metrics
        assert "mrr" in metrics
        assert "source_recall" in metrics
        assert metrics["recall_at_1"] == 1.0
        assert metrics["source_recall"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_none_expected_returns_zeros(self) -> None:
        retrieved = [_chunk(1)]
        metrics = compute_search_metrics(
            retrieved=retrieved,
            expected_chunk_ids=None,
            expected_sources=None,
            k_list=(1,),
        )
        assert metrics["recall_at_1"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["source_recall"] == 0.0
