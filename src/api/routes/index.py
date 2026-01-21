"""Index 라우터

문서 인덱싱 엔드포인트
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import IndexRequest, IndexResponse
from api.routes.ask import get_searcher
from rag.config import get_config
from rag.chunking import chunk_text
from rag.ingestion.loader import load_file


router = APIRouter()


@router.post("/index", response_model=IndexResponse)
async def index_document(request: IndexRequest):
    """텍스트를 청킹하여 인덱스에 추가"""
    config = get_config()
    
    # 텍스트 추출 (파일 경로 또는 직접 내용)
    content = ""
    if request.file_path:
        try:
            content = load_file(request.file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif request.content:
        content = request.content
    else:
        raise HTTPException(status_code=400, detail="Either content or file_path must be provided")

    if not content:
         raise HTTPException(status_code=400, detail="Empty content")

    # 텍스트 청킹
    chunks = chunk_text(
        text=content,
        filename=request.filename,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
    
    if not chunks:
        return IndexResponse(message="No chunks created", chunk_count=0)
    
    # user_id를 각 청크 메타데이터에 추가
    if request.user_id:
        for chunk in chunks:
            chunk.metadata["user_id"] = request.user_id
    
    # 인덱스에 추가
    searcher = get_searcher()
    searcher.index(chunks)
    
    # 인덱스 저장
    index_path = Path(config.project.index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    searcher.save(index_path)
    
    return IndexResponse(
        message=f"Successfully indexed {len(chunks)} chunks for user {request.user_id or 'anonymous'}",
        chunk_count=len(chunks),
    )
