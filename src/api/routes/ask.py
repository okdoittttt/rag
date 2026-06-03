"""Ask 라우터

질문-답변 엔드포인트 (일반 및 스트리밍)
"""

import json
import threading
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas import AskRequest, AskResponse, ChunkReference
from api.exceptions import IndexNotFoundError
from rag.config import get_config
from rag.embedding import Embedder, get_vector_store
from rag.generation import COT_DELIMITER, build_prompt, get_llm
from rag.logger import get_logger
from rag.retrieval import HybridSearcher
from rag.retrieval.intent import is_summarization_intent
from rag.retrieval.reranker import Reranker
from rag.retrieval.query_rewriter import QueryRewriter


logger = get_logger(__name__)


router = APIRouter()

# 싱글톤 인스턴스들 (지연 로딩)
_searcher: HybridSearcher | None = None
_reranker: Reranker | None = None
# reranker 로드를 한 번 실패하면 재시도하지 않도록 음성 캐시한다.
_reranker_load_failed: bool = False
# BM25/FAISS in-memory 상태 보호용. Qdrant도 같은 락을 통과시켜 일관 보장.
_searcher_lock = threading.RLock()


def get_searcher() -> HybridSearcher:
    """HybridSearcher 싱글톤"""
    global _searcher
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                config = get_config()
                embedder = Embedder()
                store = get_vector_store(config)
                _searcher = HybridSearcher(embedder, store)

                index_path = Path(config.project.index_path)
                if index_path.exists():
                    _searcher.load(index_path)
    return _searcher


def get_searcher_lock() -> threading.RLock:
    """다른 라우터(예: /index)가 mutate 시 공유 락을 잡도록"""
    return _searcher_lock


def get_reranker_instance() -> Reranker | None:
    """Reranker 싱글톤.

    모델 로드 실패 시 ``None`` 을 반환해 호출부가 reranker 없이 진행할 수 있게
    한다. 실패한 로드를 재시도하지 않도록 ``_reranker_load_failed`` 플래그로
    음성 캐시한다.

    Returns:
        성공 시 ``Reranker`` 인스턴스. 실패 시 ``None``.
    """
    global _reranker, _reranker_load_failed
    if _reranker is not None:
        return _reranker
    if _reranker_load_failed:
        return None
    with _searcher_lock:
        if _reranker is not None:
            return _reranker
        if _reranker_load_failed:
            return None
        config = get_config()
        try:
            _reranker = Reranker(
                model_name=config.retrieval.reranker_model,
                device=config.retrieval.reranker_device,
                batch_size=config.retrieval.reranker_batch_size,
            )
        except Exception as exc:  # 모델 다운로드/디바이스 이슈 등
            logger.error(
                "reranker_load_failed",
                model=config.retrieval.reranker_model,
                error=str(exc),
            )
            _reranker_load_failed = True
            return None
    return _reranker


def _should_use_summary_mode(request: AskRequest) -> bool:
    """문서 요약 모드를 사용해야 하는지 결정한다.

    ``doc_mode`` 와 ``source_filter`` 가 함께 지정된 경우에만 활성을 고려한다.
    ``summarize_override`` 가 명시되면 그 값을 그대로 따르고, 아니면
    ``is_summarization_intent`` 휴리스틱으로 판정한다.

    Args:
        request: 사용자 요청.

    Returns:
        요약 모드를 사용해야 하면 ``True``.
    """
    if not (request.doc_mode and request.source_filter):
        return False
    if request.summarize_override is not None:
        return request.summarize_override
    return is_summarization_intent(request.query)


def _search_documents(request: AskRequest) -> tuple[list, list]:
    """검색 로직 공통 함수.

    ``doc_mode`` + ``source_filter`` 가 켜져 있고 요약 의도가 감지되거나
    명시되면 검색을 건너뛰고 해당 문서의 모든 청크를 ``chunk_index``
    오름차순으로 반환한다. 그 외엔 기존 hybrid 검색 흐름을 사용한다.

    Returns:
        ``(chunks, scored_results)`` 튜플. 요약 모드의 ``scored_results`` 는
        ``(chunk, 1.0)`` 형태로 채워져 참조 표시에 사용된다.
    """
    config = get_config()
    index_path = Path(config.project.index_path)

    if not index_path.exists():
        raise IndexNotFoundError()

    searcher = get_searcher()

    # 1) 문서 요약 모드 분기 — 검색을 우회한다.
    if _should_use_summary_mode(request):
        chunks = searcher.fetch_full_document(
            source=request.source_filter,
            user_id=request.user_id,
            max_chunks=config.retrieval.summary_max_chunks,
        )
        scored = [(c, 1.0) for c in chunks]
        logger.info(
            "doc_mode_summary_routing",
            source=request.source_filter,
            user_id=request.user_id,
            chunk_count=len(chunks),
            summarize_override=request.summarize_override,
        )
        return chunks, scored

    # 2) 일반 hybrid 검색 흐름.
    # Query Rewriting (선택적)
    if request.expand:
        llm = get_llm(request.provider)
        rewriter = QueryRewriter(llm)
        queries = rewriter.rewrite(request.query)
    else:
        queries = [request.query]

    # 검색 (user_id 및 source_filter 적용).
    # Reranker 사용 시 1차 후보를 넓게 가져와야 의미가 있다. 표준은 top_k*5~10
    # 수준이며, 너무 적으면 cross-encoder가 보정할 여지가 사라진다. 최소 20개
    # 후보를 보장해 top_k 가 1~2 인 짧은 질의에서도 reranking 이 동작하도록 한다.
    if request.rerank:
        search_top_k = max(request.top_k * 5, 20)
    else:
        search_top_k = request.top_k * 2

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

    # 중복 제거: dedupe key 는 (source, chunk_index) — chunk_index 단독이면
    # 멀티 문서 환경에서 서로 다른 문서의 동일 인덱스가 충돌한다.
    seen: set = set()
    unique_results = []
    for chunk, score in all_results:
        dedupe_key = (
            chunk.metadata.get("source"),
            chunk.metadata.get("chunk_index", id(chunk)),
        )
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            unique_results.append((chunk, score))

    unique_results.sort(key=lambda x: x[1], reverse=True)

    # Reranking (선택적). 모델 로드 실패 시 `None` 폴백으로 reranker 없이 진행.
    if request.rerank and unique_results:
        reranker = get_reranker_instance()
        if reranker is not None:
            unique_results = reranker.rerank(
                request.query, unique_results, top_k=request.top_k,
            )
        else:
            logger.warning(
                "reranker_unavailable_fallback",
                query_preview=request.query[:50],
                candidate_count=len(unique_results),
            )

    chunks = [r[0] for r in unique_results[:request.top_k]]

    return chunks, unique_results[:request.top_k]


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """질문에 대한 답변 생성"""
    chunks, unique_results = _search_documents(request)
    
    if not chunks:
        return AskResponse(answer="관련 문서를 찾을 수 없습니다.", references=[])
    
    # LLM 호출
    prompt_mode = "summary" if _should_use_summary_mode(request) else "default"
    prompt = build_prompt(request.query, chunks, mode=prompt_mode)
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


def _sse(payload: dict) -> str:
    """딕셔너리를 SSE ``data:`` 라인으로 직렬화한다.

    Args:
        payload: 클라이언트로 전송할 이벤트 딕셔너리.

    Returns:
        ``data: {json}\\n\\n`` 형식의 SSE 문자열. 한국어 보존을 위해
        ``ensure_ascii=False`` 로 직렬화한다.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """질문에 답하며 진행 단계와 답변을 SSE로 스트리밍한다.

    검색 → 분석 → 추론(CoT) → 답변 순으로 진행하며 각 단계를
    ``text/event-stream`` 으로 전송한다. LLM 은 한 번 호출로
    ``추론 + COT_DELIMITER + 답변`` 을 생성하고, 이 함수가 구분자를 경계로
    토큰을 추론(``phase="reasoning"``)과 최종 답변(``text``)으로 라벨링한다.
    구분자가 청크 경계에 걸쳐 분할되는 경우를 대비해 마지막 ``len(구분자)-1``
    글자를 다음 청크까지 보류(holdback)한다.

    이벤트 종류:
        - ``{"phase": "searching"}``: 문서 검색 시작.
        - ``{"phase": "analyzing"}``: 프롬프트 구성/LLM 준비.
        - ``{"references": [...]}``: 참조 청크 목록.
        - ``{"phase": "reasoning_start"}``: 추론 스트리밍 시작.
        - ``{"phase": "reasoning", "text": "..."}``: 추론 토큰.
        - ``{"text": "..."}``: 최종 답변 토큰.
        - ``{"phase": "error", "detail": "..."}``: 오류.
        - ``[DONE]``: 스트림 종료 신호.

    Args:
        request: ``query`` 와 검색/LLM 옵션을 담은 질문 요청.

    Returns:
        SSE(``text/event-stream``) ``StreamingResponse``.
    """

    def event_stream():
        try:
            # 1. 검색
            yield _sse({"phase": "searching"})
            chunks, unique_results = _search_documents(request)

            if not chunks:
                yield _sse({"text": "관련 문서를 찾을 수 없습니다."})
                yield "data: [DONE]\n\n"
                return

            # 2. 분석 (프롬프트 구성 + LLM 준비)
            yield _sse({"phase": "analyzing"})
            prompt_mode = "summary" if _should_use_summary_mode(request) else "default"
            prompt = build_prompt(request.query, chunks, mode=prompt_mode, cot=True)
            llm = get_llm(
                provider=request.provider,
                api_key=request.api_key,
                model_name=request.model_name,
                base_url=request.base_url,
            )

            # 3. 참조 정보 전송
            references = [
                {
                    "content": chunk.content[:500],
                    "source": chunk.metadata.get("source", "unknown"),
                    "score": score,
                }
                for chunk, score in unique_results
            ]
            yield _sse({"references": references})

            # 4. CoT 스트리밍: 추론 → 구분자 → 답변
            yield _sse({"phase": "reasoning_start"})
            delimiter = COT_DELIMITER
            hold = len(delimiter) - 1  # 구분자가 청크 경계에 걸칠 때를 대비한 보류 길이
            buffer = ""  # 추론/답변 판정 전 보류 중인 텍스트
            in_answer = False

            for chunk_text in llm.generate_stream(prompt):
                if in_answer:
                    if chunk_text:
                        yield _sse({"text": chunk_text})
                    continue

                buffer += chunk_text
                idx = buffer.find(delimiter)
                if idx != -1:
                    # 구분자 발견: 앞부분=추론, 뒷부분=답변
                    reasoning_part = buffer[:idx]
                    if reasoning_part:
                        yield _sse({"phase": "reasoning", "text": reasoning_part})
                    answer_part = buffer[idx + len(delimiter):].lstrip("\n")
                    if answer_part:
                        yield _sse({"text": answer_part})
                    buffer = ""
                    in_answer = True
                elif len(buffer) > hold:
                    # 구분자 미발견: 꼬리 hold 글자만 보류하고 나머지는 추론으로 흘린다.
                    emit = buffer[: len(buffer) - hold]
                    buffer = buffer[len(buffer) - hold:]
                    if emit:
                        yield _sse({"phase": "reasoning", "text": emit})

            # 5. 루프 종료 후 잔여 버퍼 처리
            if buffer:
                if in_answer:
                    yield _sse({"text": buffer})
                else:
                    # 구분자가 끝내 안 나옴(모델 형식 미준수). 남은 꼬리도 추론으로 마무리.
                    # 답변(text)이 하나도 없으면 프론트가 추론을 답변으로 승격 처리한다.
                    yield _sse({"phase": "reasoning", "text": buffer})

            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001 - 스트림 소비자에게 오류 전달
            logger.error("ask_stream_failed", error=str(exc))
            yield _sse({"phase": "error", "detail": "답변 생성 중 오류가 발생했습니다."})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
