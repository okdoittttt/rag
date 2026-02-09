"""토크나이저 모듈

한국어 형태소 분석기(Kiwi)와 공백 토크나이저를 제공합니다.
텍스트 언어를 자동 감지하여 적절한 토크나이저를 선택합니다.
"""

from __future__ import annotations

from typing import List

from rag.ingestion.language import detect_language

try:
    from kiwipiepy import Kiwi
    _kiwi = Kiwi()
    HAS_KIWI = True
except ImportError:
    HAS_KIWI = False


def _tokenize_korean(text: str) -> List[str]:
    """한국어 형태소 분석 토크나이징"""
    if not HAS_KIWI:
        return _tokenize_english(text)

    tokens = []
    for token in _kiwi.tokenize(text):
        # N: 명사, V: 동사, VA: 형용사, XR: 어근, SL: 외국어
        if token.tag.startswith(('N', 'V', 'VA', 'XR', 'SL')):
            tokens.append(token.form.lower())
    return tokens


def _tokenize_english(text: str) -> List[str]:
    """영어 공백 기반 토크나이징"""
    return text.lower().split()


def tokenize_query(query: str, language: str | None = None) -> List[str]:
    """쿼리 토크나이징 (언어 자동 감지)

    Args:
        query: 검색 쿼리
        language: 언어 코드 ("ko", "en", None=자동 감지)

    Returns:
        토큰 리스트
    """
    if language is None:
        language = detect_language(query)

    if language == "ko" and HAS_KIWI:
        return _tokenize_korean(query)

    return _tokenize_english(query)


def tokenize_content(content: str) -> List[str]:
    """문서 본문 토크나이징 (인덱싱용, 언어 자동 감지)"""
    language = detect_language(content)
    return tokenize_query(content, language=language)
