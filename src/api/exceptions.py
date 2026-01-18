"""API 예외 및 에러 핸들러

커스텀 예외 클래스와 표준 에러 응답 스키마를 정의합니다.
"""

from typing import Any

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """표준 에러 응답"""
    error: str
    message: str
    detail: Any | None = None


# === 커스텀 예외 클래스 ===

class RAGException(Exception):
    """RAG API 기본 예외"""
    def __init__(self, message: str = "An error occurred"):
        self.message = message
        super().__init__(self.message)


class IndexNotFoundError(RAGException):
    """인덱스가 없을 때 발생"""
    def __init__(self, message: str = "인덱스가 없습니다. 먼저 문서를 인덱싱하세요."):
        super().__init__(message)


class LLMError(RAGException):
    """LLM 호출 실패 시 발생"""
    def __init__(self, message: str = "LLM 호출에 실패했습니다."):
        super().__init__(message)


class SearchError(RAGException):
    """검색 실패 시 발생"""
    def __init__(self, message: str = "검색 중 오류가 발생했습니다."):
        super().__init__(message)
