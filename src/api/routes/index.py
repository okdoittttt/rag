"""Index 라우터

문서 인덱싱 엔드포인트
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import IndexRequest, IndexResponse
from api.routes.ask import get_searcher, get_searcher_lock
from rag.config import get_config
from rag.chunking import chunk_document
from rag.ingestion.document import Document
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
            content = load_file(request.file_path).content
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    elif request.content:
        content = request.content
    else:
        raise HTTPException(status_code=400, detail="Either content or file_path must be provided")

    if not content:
         raise HTTPException(status_code=400, detail="Empty content")

    # 파일명에서 확장자 추출하여 파일 타입별 청킹 전략 적용
    filename = request.filename or "uploaded.txt"
    extension = Path(filename).suffix.lower()

    doc = Document(
        content=content,
        metadata={
            "source": filename,
            "filename": filename,
            "extension": extension,
        },
    )
    chunks = chunk_document(doc)
    
    if not chunks:
        return IndexResponse(message="No chunks created", chunk_count=0)
    
    # user_id를 각 청크 메타데이터에 추가
    if request.user_id:
        for chunk in chunks:
            chunk.metadata["user_id"] = request.user_id

    searcher = get_searcher()
    index_path = Path(config.project.index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    # 동시 인덱싱/검색으로부터 BM25/FAISS in-memory 상태 보호
    with get_searcher_lock():
        # 동일 (source, user_id) 기존 청크 제거 후 재인덱싱 (중복 누적 방지)
        replaced = searcher.delete_by_source(filename, user_id=request.user_id)
        searcher.index(chunks)
        searcher.save(index_path)

    user_label = request.user_id or "anonymous"
    base_msg = f"Successfully indexed {len(chunks)} chunks for user {user_label}"
    if replaced:
        base_msg += f" (replaced {replaced} previous chunks)"

    return IndexResponse(
        message=base_msg,
        chunk_count=len(chunks),
    )
