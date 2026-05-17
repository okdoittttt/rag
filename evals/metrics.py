"""검색 단계 전용 메트릭.

LLM 호출 없이 ``expected_chunk_ids``/``expected_sources`` 와 실제 검색
결과를 비교한다. RAGAS 메트릭이 LLM-as-judge로 느리고 비용이 큰
반면, 본 모듈의 메트릭은 결정적이며 ms 단위로 산출되므로 회귀 감지와
스윕 실험에 적합하다.
"""

from __future__ import annotations

from typing import Iterable

from rag.chunking.chunk import Chunk


def _expected_id_set(expected_ids: Iterable[int | str]) -> set[int | str]:
    """기대 청크 ID를 비교 가능한 집합으로 정규화한다.

    Args:
        expected_ids: 기대 ``chunk_index`` 리스트(또는 호환 iterable).

    Returns:
        중복 제거된 ID 집합. ``None``은 제거된다.
    """
    return {eid for eid in expected_ids if eid is not None}


def recall_at_k(
    retrieved: list[Chunk],
    expected_ids: Iterable[int | str],
    k: int,
) -> float:
    """Recall@k 계산.

    검색 결과 상위 ``k``개 중에서 기대 청크가 몇 개나 포함됐는지를
    기대 청크 수로 나눈 값이다.

    Args:
        retrieved: 검색된 청크 리스트(전체, 상위 ``k`` 슬라이스 전).
        expected_ids: 정답 ``chunk_index`` 리스트.
        k: 평가할 상위 결과 수. 1 이상.

    Returns:
        ``0.0``~``1.0`` 사이 recall 값. 기대 청크가 없으면 ``0.0``.

    Raises:
        ValueError: ``k`` 가 1 미만일 때.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    expected = _expected_id_set(expected_ids)
    if not expected:
        return 0.0
    top_ids = {c.metadata.get("chunk_index") for c in retrieved[:k]}
    hits = len(expected & top_ids)
    return hits / len(expected)


def precision_at_k(
    retrieved: list[Chunk],
    expected_ids: Iterable[int | str],
    k: int,
) -> float:
    """Precision@k 계산.

    상위 ``k``개 검색 결과 중 기대 청크에 해당하는 비율.

    Args:
        retrieved: 검색된 청크 리스트.
        expected_ids: 정답 ``chunk_index`` 리스트.
        k: 평가할 상위 결과 수. 1 이상.

    Returns:
        ``0.0``~``1.0`` 사이 precision 값.

    Raises:
        ValueError: ``k`` 가 1 미만일 때.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    expected = _expected_id_set(expected_ids)
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for c in top if c.metadata.get("chunk_index") in expected)
    return hits / len(top)


def mrr(retrieved: list[Chunk], expected_ids: Iterable[int | str]) -> float:
    """Mean Reciprocal Rank (단일 케이스).

    검색 결과를 순회하며 첫 정답 청크가 등장한 순위의 역수를 반환한다.

    Args:
        retrieved: 검색된 청크 리스트(상위부터 정렬됨).
        expected_ids: 정답 ``chunk_index`` 리스트.

    Returns:
        ``0.0`` ~ ``1.0`` 사이 reciprocal rank. 미발견 시 ``0.0``.
    """
    expected = _expected_id_set(expected_ids)
    if not expected:
        return 0.0
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.metadata.get("chunk_index") in expected:
            return 1.0 / rank
    return 0.0


def source_recall(
    retrieved: list[Chunk],
    expected_sources: Iterable[str],
) -> float:
    """문서(source) 단위 recall.

    청크 인덱스가 청킹 전략 변경으로 흔들리기 쉬우므로, 더 안정적인
    문서 단위 정답 비교를 함께 제공한다.

    Args:
        retrieved: 검색된 청크 리스트.
        expected_sources: 기대 ``source`` 파일명 리스트.

    Returns:
        기대 문서 중 검색 결과에 포함된 비율 (``0.0``~``1.0``).
    """
    expected = {s for s in expected_sources if s}
    if not expected:
        return 0.0
    found_sources = {c.metadata.get("source") for c in retrieved}
    hits = len(expected & found_sources)
    return hits / len(expected)


def compute_search_metrics(
    retrieved: list[Chunk],
    expected_chunk_ids: Iterable[int | str] | None,
    expected_sources: Iterable[str] | None,
    k_list: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    """단일 케이스에 대한 검색 단계 메트릭을 한 번에 계산한다.

    Args:
        retrieved: 검색된 청크 리스트(상위부터 정렬됨).
        expected_chunk_ids: 정답 ``chunk_index`` 리스트. ``None`` 또는 빈
            리스트면 청크 단위 지표는 ``0.0``으로 채워진다.
        expected_sources: 정답 ``source`` 리스트. ``None`` 가능.
        k_list: Recall/Precision@k를 계산할 k 후보들.

    Returns:
        ``recall_at_1``, ``recall_at_3``, ..., ``mrr``, ``source_recall``
        키를 갖는 dict. RAGAS 결과와 병합하기 쉬운 평탄 구조.
    """
    chunk_ids = list(expected_chunk_ids or [])
    sources = list(expected_sources or [])

    metrics: dict[str, float] = {}
    for k in k_list:
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved, chunk_ids, k)
        metrics[f"precision_at_{k}"] = precision_at_k(retrieved, chunk_ids, k)
    metrics["mrr"] = mrr(retrieved, chunk_ids)
    metrics["source_recall"] = source_recall(retrieved, sources)
    return metrics
