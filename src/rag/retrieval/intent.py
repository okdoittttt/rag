"""질의 의도 감지 모듈.

요약/전체 조망 의도를 가진 질의를 패턴 매칭으로 식별한다. LLM 호출
없이 ms 단위로 결정되므로, 검색 단계 분기(``_search_documents``)에서
부담 없이 호출할 수 있다. 복잡한 분류가 필요해지면 추후 LLM-based
classifier로 확장 가능하다.

문서 모드(``doc_mode=True`` + ``source_filter``)에서 요약 의도가
감지되면 라우터는 top-k 검색 대신 문서 전체 청크를 LLM 에 투입한다.
이로써 다음과 같은 회귀 버그를 해소한다:

- 사용자가 "해당 문서를 요약해서 알려줘" 라고 질의 → top_k=5 청크만
  전달 → 헤더·날짜만 답변되는 문제.
"""

from __future__ import annotations

import re

# 한국어 요약 의도 패턴.
# 각 패턴은 ``re.search`` 로 평가되므로 부분 일치도 양성.
# 너무 모호한 표현("이거 뭐야?")은 의도적으로 포함하지 않는다 — 모호한
# 케이스는 사용자가 ``summarize_override=True`` 로 명시 강제할 수 있다.
SUMMARIZATION_PATTERNS: tuple[str, ...] = (
    # "요약해줘", "요약을 알려줘", "요약 좀 보여줘" 등
    r"요약(?:해|을|좀|해서)?\s*(?:알려|줘|해|보여|설명|부탁)",
    # "정리해줘", "정리 좀 알려줘" 등
    r"정리(?:해|를|좀|해서)?\s*(?:알려|줘|해|보여|설명|부탁)",
    # "전체 내용", "전체적인 흐름"
    r"전체(?:적인?)?\s*(?:내용|개요|흐름|구조)",
    # "핵심 내용", "주요 포인트", "요점"
    r"(?:핵심|주요)\s*(?:내용|포인트|요점|사항)",
    # "무슨/어떤 내용"
    r"(?:무슨|어떤)\s*내용",
    # "한 줄/한 문장 요약"
    r"한\s*(?:줄|문장|단락)\s*(?:로|으로)?\s*(?:요약|정리)",
    # "개요", "개관"
    r"\b(?:개요|개관)\b",
    # 영어 변형. 문장 어느 위치에서도 매칭(예: "Give me an overview").
    r"\b(?:summarize|summary|tl;?dr|overview|outline)\b",
)

# 컴파일 캐시 — 모듈 로드 시 한 번만 컴파일.
_SUMMARIZATION_RE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in SUMMARIZATION_PATTERNS
)


def is_summarization_intent(query: str) -> bool:
    """질의가 문서 요약/조망 의도를 가지는지 판정한다.

    Args:
        query: 사용자 질의 문자열. 빈 문자열/``None`` 유사 입력은 ``False``.

    Returns:
        요약 의도로 판정되면 ``True``. 패턴 미일치 시 ``False``.

    Example:
        >>> is_summarization_intent("이 문서를 요약해줘.")
        True
        >>> is_summarization_intent("BM25 점수는 어떻게 계산되나요?")
        False
    """
    if not query:
        return False
    q = query.strip()
    if not q:
        return False
    return any(p.search(q) for p in _SUMMARIZATION_RE)
