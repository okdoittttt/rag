"""Ask 라우터

질문-답변 엔드포인트 (일반 및 스트리밍)
"""

import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import AskRequest, AskResponse, ChunkReference
from api.exceptions import IndexNotFoundError
from rag.config import get_config
from rag.embedding import Embedder, get_vector_store
from rag.generation import build_prompt, get_llm
from rag.retrieval import HybridSearcher
from rag.retrieval.reranker import Reranker
from rag.retrieval.query_rewriter import QueryRewriter


router = APIRouter()

# 싱글톤 인스턴스들 (지연 로딩)
_searcher: HybridSearcher | None = None
_reranker: Reranker | None = None


def get_searcher() -> HybridSearcher:
    """HybridSearcher 싱글톤"""
    global _searcher
    if _searcher is None:
        config = get_config()
        embedder = Embedder()
        store = get_vector_store(config)
        _searcher = HybridSearcher(embedder, store)
        
        index_path = Path(config.project.index_path)
        if index_path.exists():
            _searcher.load(index_path)
    return _searcher


def get_reranker_instance() -> Reranker:
    """Reranker 싱글톤"""
    global _reranker
    if _reranker is None:
        config = get_config()
        _reranker = Reranker(model_name=config.retrieval.reranker_model)
    return _reranker


def _search_documents(request: AskRequest) -> tuple[list, list]:
    """검색 로직 공통 함수
    
    Returns:
        (chunks, unique_results) 튜플
    """
    config = get_config()
    index_path = Path(config.project.index_path)
    
    if not index_path.exists():
        raise IndexNotFoundError()
    
    searcher = get_searcher()
    
    # Query Rewriting (선택적)
    if request.expand:
        llm = get_llm(request.provider)
        rewriter = QueryRewriter(llm)
        queries = rewriter.rewrite(request.query)
    else:
        queries = [request.query]
    
    # 검색 (user_id 및 source_filter 적용)
    search_top_k = request.top_k * 3 if request.rerank else request.top_k * 2

    all_results = []
    for q in queries:
        # 검색 수행
        results = searcher.search(
            query=q,
            top_k=search_top_k,
            user_id=request.user_id,
            source_filter=request.source_filter,
            fusion_type="weighted",
            alpha=0.7,  # 벡터 검색 가중치 70%
        )
        all_results.extend(results)
    
    # 중복 제거
    seen = set()
    unique_results = []
    for chunk, score in all_results:
        chunk_id = chunk.metadata.get("chunk_index", id(chunk))
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_results.append((chunk, score))
    
    unique_results.sort(key=lambda x: x[1], reverse=True)
    
    # Reranking (선택적)
    if request.rerank and unique_results:
        reranker = get_reranker_instance()
        unique_results = reranker.rerank(request.query, unique_results, top_k=request.top_k)
    
    chunks = [r[0] for r in unique_results[:request.top_k]]
    
    return chunks, unique_results[:request.top_k]


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """질문에 대한 답변 생성"""
    chunks, unique_results = _search_documents(request)
    
    if not chunks:
        return AskResponse(answer="관련 문서를 찾을 수 없습니다.", references=[])
    
    # LLM 호출
    prompt = build_prompt(request.query, chunks)
    llm = get_llm(
        provider=request.provider,
        api_key=request.api_key,
        model_name=request.model_name,
        base_url=request.base_url,
    )
    answer = llm.generate(prompt)
    
    # 참조 정보 구성
    references = [
        ChunkReference(
            content=chunk.content[:500],
            source=chunk.metadata.get("source", "unknown"),
            score=score,
        )
        for chunk, score in unique_results
    ]
    
    return AskResponse(answer=answer, references=references)


@router.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """스트리밍 방식으로 답변 생성 (SSE)"""
    chunks, unique_results = _search_documents(request)
    
    if not chunks:
        async def empty_response():
            yield f"data: {json.dumps({'text': '관련 문서를 찾을 수 없습니다.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty_response(), media_type="text/event-stream")
    
    # 참조 정보 (스트림 시작 시 전송)
    references = [
        {
            "content": chunk.content[:500],
            "source": chunk.metadata.get("source", "unknown"),
            "score": score,
        }
        for chunk, score in unique_results
    ]
    
    prompt = build_prompt(request.query, chunks)
    llm = get_llm(
        provider=request.provider,
        api_key=request.api_key,
        model_name=request.model_name,
        base_url=request.base_url,
    )
    
    async def generate():
        # 먼저 참조 정보 전송
        yield f"data: {json.dumps({'references': references})}\n\n"
        
        # 스트리밍 응답 생성
        for chunk_text in llm.generate_stream(prompt):
            yield f"data: {json.dumps({'text': chunk_text})}\n\n"
        
        # 완료 신호
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
