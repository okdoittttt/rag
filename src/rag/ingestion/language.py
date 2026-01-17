"""언어 감지 모듈

텍스트의 언어를 간단히 감지합니다 (한국어/영어).
외부 라이브러리 없이 문자 비율 기반으로 판단합니다.
"""

from __future__ import annotations

import re


# 한글 유니코드 범위
KOREAN_PATTERN = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")

# ASCII 알파벳 패턴
ASCII_ALPHA_PATTERN = re.compile(r"[a-zA-Z]")


def count_korean_chars(text: str) -> int:
    """한글 문자 수 반환"""
    return len(KOREAN_PATTERN.findall(text))


def count_ascii_alpha_chars(text: str) -> int:
    """ASCII 알파벳 문자 수 반환"""
    return len(ASCII_ALPHA_PATTERN.findall(text))


def detect_language(text: str, threshold: float = 0.3) -> str:
    """텍스트 언어 감지
    
    한글/영어 문자 비율을 기반으로 언어를 감지합니다.
    
    Args:
        text: 분석할 텍스트
        threshold: 언어 판단 임계값 (기본: 0.3)
            해당 언어 문자 비율이 이 값 이상이면 해당 언어로 판단
            
    Returns:
        "ko" (한국어), "en" (영어), 또는 "unknown"
    """
    if not text or not text.strip():
        return "unknown"
    
    # 공백, 숫자, 특수문자 제외한 문자만 계산
    korean_count = count_korean_chars(text)
    ascii_count = count_ascii_alpha_chars(text)
    
    total_alpha = korean_count + ascii_count
    
    if total_alpha == 0:
        return "unknown"
    
    korean_ratio = korean_count / total_alpha
    ascii_ratio = ascii_count / total_alpha
    
    # 한글이 threshold 이상이면 한국어
    if korean_ratio >= threshold:
        return "ko"
    
    # 영어가 threshold 이상이면 영어
    if ascii_ratio >= threshold:
        return "en"
    
    return "unknown"
