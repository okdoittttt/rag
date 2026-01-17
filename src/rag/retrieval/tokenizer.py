"""토크나이저 모듈

한국어 형태소 분석기(Kiwi)와 공백 토크나이저를 제공합니다.
"""

from __future__ import annotations

from typing import List

try:
    from kiwipiepy import Kiwi
    _kiwi = Kiwi()
    HAS_KIWI = True
except ImportError:
    HAS_KIWI = False


def tokenize_query(query: str, language: str = "ko") -> List[str]:
    """쿼리 토크나이징
    
    Args:
        query: 검색 쿼리
        language: 언어 코드 ("ko", "en")
        
    Returns:
        토큰 리스트
    """
    if language == "ko" and HAS_KIWI:
        # 명사, 동사, 형용사, 어근 등 실질 형태소만 추출
        tokens = []
        for token in _kiwi.tokenize(query):
            if token.tag.startswith(('N', 'V', 'VA', 'XR', 'SL')):
                tokens.append(token.form)
        return tokens
    
    # 기본 공백 분리 (영어 등)
    return query.lower().split()


def tokenize_content(content: str) -> List[str]:
    """문서 본문 토크나이징 (인덱싱용)"""
    # 쿼리 분석과 동일한 로직 사용
    return tokenize_query(content)
