"""Search 라우터

검색 전용 엔드포인트 (LLM 호출 없이 관련 문서만 반환)
"""

from pathlib import Path

from fastapi import APIRouter

from api.schemas import SearchRequest, SearchResponse, ChunkReference
from api.exceptions import IndexNotFoundError
from api.routes.ask import get_searcher
from rag.config import get_config


router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """관련 문서 검색"""
    config = get_config()
    index_path = Path(config.project.index_path)
    
    if not index_path.exists():
        raise IndexNotFoundError()
    
    searcher = get_searcher()
    results = searcher.search(request.query, top_k=request.top_k)
    
    references = [
        ChunkReference(
            content=chunk.content[:500],
            source=chunk.metadata.get("filename", "unknown"),
            score=score,
        )
        for chunk, score in results
    ]
    
    return SearchResponse(results=references)
