"""텍스트 정규화 모듈

문서 본문의 공백, 개행, 유니코드 등을 정규화합니다.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """공백 정규화
    
    - 연속 공백 → 단일 공백
    - 탭 → 공백 4개
    - 줄 끝 공백 제거
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        정규화된 텍스트
    """
    # 탭을 공백 4개로 변환
    text = text.replace("\t", "    ")
    
    # 줄 끝 공백 제거 (각 줄별로)
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)
    
    # 연속 공백을 단일 공백으로 (줄바꿈은 유지)
    text = re.sub(r"[^\S\n]+", " ", text)
    
    return text


def normalize_newlines(text: str, max_consecutive: int = 2) -> str:
    """개행 정규화
    
    연속 개행이 max_consecutive보다 많으면 max_consecutive개로 줄임
    
    Args:
        text: 정규화할 텍스트
        max_consecutive: 최대 연속 개행 수 (기본: 2)
        
    Returns:
        정규화된 텍스트
    """
    # 3개 이상 연속 개행을 max_consecutive개로
    pattern = r"\n{" + str(max_consecutive + 1) + r",}"
    replacement = "\n" * max_consecutive
    
    return re.sub(pattern, replacement, text)


def normalize_unicode(text: str) -> str:
    """유니코드 정규화 (NFKC)
    
    - 호환 문자를 표준 형태로 변환
    - 전각 문자를 반각으로 변환
    - 합성 문자를 분해 후 재결합
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        정규화된 텍스트
    """
    return unicodedata.normalize("NFKC", text)


def normalize_text(text: str) -> str:
    """전체 텍스트 정규화
    
    모든 정규화를 순서대로 적용:
    1. 유니코드 정규화
    2. 공백 정규화
    3. 개행 정규화
    4. 앞뒤 공백 제거
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        정규화된 텍스트
    """
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    text = normalize_newlines(text)
    text = text.strip()
    
    return text
