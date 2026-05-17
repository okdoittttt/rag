"""rag.retrieval.intent 휴리스틱 검증.

LLM 호출 없이 패턴 매칭만으로 동작하므로 외부 mock 없이 결정적 테스트.
양성/음성 케이스를 카테고리별로 균형 있게 둔다.
"""

from __future__ import annotations

import pytest

from rag.retrieval.intent import is_summarization_intent


@pytest.mark.parametrize(
    "query",
    [
        "해당 문서를 요약해서 알려줘.",
        "이 문서를 요약해줘",
        "내용을 좀 정리해서 알려줘",
        "전체 내용이 뭐야?",
        "전체적인 흐름을 설명해줘",
        "핵심 내용만 추려줘",
        "주요 포인트를 알려줘",
        "이 문서 무슨 내용이야?",
        "어떤 내용인지 알려줄래",
        "한 줄로 요약해줘",
        "한 문장으로 정리해줘",
        "이 문서의 개요",
        "summarize this document",
        "Summary please",
        "tl;dr",
        "TLDR",
        "Give me an overview",
        "Outline of the doc",
    ],
)
def test_positive_cases(query: str) -> None:
    """요약/조망 의도를 가진 한국어/영어 질의를 양성 판정해야 한다."""
    assert is_summarization_intent(query), f"should be positive: {query!r}"


@pytest.mark.parametrize(
    "query",
    [
        "BM25는 어떤 검색을 수행하나요?",
        "이 함수의 인자는?",
        "reranker_model 기본값이 뭐야?",
        "임베딩 차원이 384인가?",
        "Qdrant 포트를 어떻게 바꿔?",
        "정리 안 된 곳이 어디야?",  # "정리" 만 단독으로는 양성 아님
        "전체 라는 단어가 들어간 다른 의미",
        "",
        "   ",
    ],
)
def test_negative_cases(query: str) -> None:
    """단순 사실 질의는 음성 판정해야 한다."""
    assert not is_summarization_intent(query), f"should be negative: {query!r}"


def test_none_safe() -> None:
    """``None`` 입력에도 예외 없이 ``False`` 반환."""
    assert is_summarization_intent(None) is False  # type: ignore[arg-type]
