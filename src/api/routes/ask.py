"""Ask 라우터

질문-답변 엔드포인트
"""

from pathlib import Path

from fastapi import APIRouter

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


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """질문에 대한 답변 생성"""
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
    
    # 검색
    search_top_k = request.top_k * 3 if request.rerank else request.top_k * 2
    
    all_results = []
    for q in queries:
        results = searcher.search(q, top_k=search_top_k)
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
    
    if not chunks:
        return AskResponse(answer="관련 문서를 찾을 수 없습니다.", references=[])
    
    # LLM 호출
    prompt = build_prompt(request.query, chunks)
    llm = get_llm(request.provider)
    answer = llm.generate(prompt)
    
    # 참조 정보 구성
    references = [
        ChunkReference(
            content=chunk.content[:300],
            source=chunk.metadata.get("filename", "unknown"),
            score=score,
        )
        for chunk, score in unique_results[:request.top_k]
    ]
    
    return AskResponse(answer=answer, references=references)
