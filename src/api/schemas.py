"""API 스키마 정의

Pydantic 모델을 사용한 Request/Response 스키마
"""

from pydantic import BaseModel, Field


# === Ask Endpoint ===

class AskRequest(BaseModel):
    """질문-답변 요청"""
    query: str = Field(..., description="질문 내용")
    top_k: int = Field(default=5, ge=1, le=20, description="참조할 청크 개수")
    rerank: bool = Field(default=False, description="Reranker로 결과 재정렬")
    expand: bool = Field(default=False, description="Query Rewriting으로 검색 확장")
    provider: str | None = Field(default=None, description="LLM Provider (gemini/ollama)")
    user_id: str | None = Field(default=None, description="사용자 ID (격리된 검색용)")
    
    # LLM 설정 (클라이언트 오버라이드)
    api_key: str | None = Field(default=None, description="API Key (Optional)")
    model_name: str | None = Field(default=None, description="Model Name (Optional)")
    base_url: str | None = Field(default=None, description="Base URL (Ollama only)")


class ChunkReference(BaseModel):
    """청크 참조 정보"""
    content: str = Field(..., description="청크 내용")
    source: str = Field(..., description="출처 파일명")
    score: float = Field(..., description="관련도 점수")


class AskResponse(BaseModel):
    """질문-답변 응답"""
    answer: str = Field(..., description="생성된 답변")
    references: list[ChunkReference] = Field(default_factory=list, description="참조된 청크 목록")


# === Search Endpoint ===

class SearchRequest(BaseModel):
    """검색 요청"""
    query: str = Field(..., description="검색어")
    top_k: int = Field(default=5, ge=1, le=20, description="반환할 결과 개수")
    user_id: str | None = Field(default=None, description="사용자 ID (격리된 검색용)")


class SearchResponse(BaseModel):
    """검색 응답"""
    results: list[ChunkReference] = Field(default_factory=list, description="검색 결과")


# === Index Endpoint ===

class IndexRequest(BaseModel):
    """인덱싱 요청"""
    content: str = Field(..., description="인덱싱할 텍스트")
    filename: str = Field(default="uploaded", description="파일명 (메타데이터용)")
    user_id: str | None = Field(default=None, description="사용자 ID (격리된 인덱싱용)")


class IndexResponse(BaseModel):
    """인덱싱 응답"""
    message: str = Field(..., description="결과 메시지")
    chunk_count: int = Field(..., description="생성된 청크 수")
