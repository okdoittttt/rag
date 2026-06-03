"""Index 라우터

문서 인덱싱 엔드포인트
"""

import json
import os
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    IndexDeleteRequest,
    IndexDeleteResponse,
    IndexRequest,
    IndexResponse,
)
from api.routes.ask import get_searcher, get_searcher_lock
from rag.config import get_config
from rag.chunking import chunk_document
from rag.ingestion.document import Document
from rag.ingestion.loader import load_file
from rag.logger import get_logger


router = APIRouter()

logger = get_logger(__name__)


def _allowed_upload_bases() -> list[Path]:
    """``file_path`` 검증에 사용할 허용 베이스 디렉터리 목록을 반환한다.

    UI가 파일을 저장하는 ``UPLOAD_DIR`` 환경변수(설정된 경우)와 함께
    Docker(``/app/data/uploads``) 및 로컬 저장소(``<repo>/data/uploads``)
    기본값을 항상 포함한다. 호출자는 반환 리스트를 기준으로 정규화된 입력
    경로가 어느 하나의 베이스 하위에 속하는지 검사한다.

    Returns:
        허용 베이스 ``Path`` 리스트. 중복은 제거되지 않으나 검사 로직상
        순서·중복은 결과에 영향을 주지 않는다.
    """
    bases: list[Path] = []
    env_dir = os.environ.get("UPLOAD_DIR", "").strip()
    if env_dir:
        bases.append(Path(env_dir))
    # 컨테이너 환경 (compose.yaml의 ./data:/app/data 볼륨 마운트 기준)
    bases.append(Path("/app/data/uploads"))
    # 로컬 개발 환경 (<repo>/data/uploads)
    repo_root = Path(__file__).resolve().parents[3]
    bases.append(repo_root / "data" / "uploads")
    return bases


def _resolve_safe_upload_path(raw_path: str) -> Path:
    """입력 경로를 정규화하고 허용 베이스 하위인지 검증한다.

    경로 순회(``..``, 심볼릭 링크 등)를 모두 풀어낸 절대 경로가
    ``_allowed_upload_bases()`` 중 한 곳에도 속하지 않으면
    ``HTTPException(400)``을 발생시켜 임의 파일 접근을 차단한다.

    Args:
        raw_path: 클라이언트가 전달한 ``file_path`` 문자열.

    Returns:
        검증을 통과한 절대 ``Path``. 이후 ``load_file``에 전달해 안전하게
        파싱할 수 있다.

    Raises:
        HTTPException: 입력 경로가 허용 베이스 외부를 가리킬 때 400.
    """
    candidate = Path(raw_path).resolve(strict=False)
    for base in _allowed_upload_bases():
        try:
            base_resolved = base.resolve(strict=False)
        except OSError:
            continue
        if not base_resolved.exists():
            continue
        try:
            if candidate.is_relative_to(base_resolved):
                return candidate
        except ValueError:
            continue

    logger.warning(
        "rejected_unsafe_file_path",
        path_suffix=Path(raw_path).name,
    )
    raise HTTPException(
        status_code=400,
        detail="허용되지 않은 file_path 입니다.",
    )


@router.post("/index", response_model=IndexResponse)
async def index_document(request: IndexRequest):
    """텍스트를 청킹하여 인덱스에 추가"""
    config = get_config()

    # 텍스트 추출 (파일 경로 또는 직접 내용)
    content = ""
    if request.file_path:
        safe_path = _resolve_safe_upload_path(request.file_path)
        try:
            content = load_file(safe_path).content
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


def _sse(payload: dict) -> str:
    """딕셔너리를 SSE ``data:`` 라인으로 직렬화한다.

    Args:
        payload: 클라이언트로 전송할 진행 이벤트 딕셔너리.

    Returns:
        ``data: {json}\\n\\n`` 형식의 SSE 문자열.
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/index/stream")
async def index_document_stream(request: IndexRequest):
    """문서를 인덱싱하며 진행 상황을 SSE로 스트리밍한다.

    파싱 → 청킹 → 임베딩(청크 배치 단위 진행률) → 인덱싱 단계를 순서대로
    수행하면서 각 단계를 ``text/event-stream``으로 전송한다. 임베딩은
    ``embedding.batch_size`` 단위로 나누어 호출해 ``current``/``total`` 진행률을
    보고하며, 그 결과를 ``searcher.index(embeddings=...)``에 전달해 재임베딩을
    생략한다.

    이벤트 종류:
        - ``{"phase": "parsing"}``: 파일 파싱 시작.
        - ``{"phase": "chunking", "total": N}``: 청킹 완료, 총 청크 수.
        - ``{"phase": "preparing"}``: 임베딩 모델 로딩 구간(첫 호출 시).
        - ``{"phase": "embedding", "current": M, "total": N}``: 임베딩 진행률.
        - ``{"phase": "indexing"}``: BM25/벡터 저장소 반영 단계.
        - ``{"phase": "done", "chunk_count": N, "replaced": R}``: 완료.
        - ``{"phase": "error", "detail": "..."}``: 오류.
        - ``[DONE]``: 스트림 종료 신호.

    Args:
        request: ``file_path`` 또는 ``content``, ``filename``, ``user_id`` 를 담은
            인덱싱 요청.

    Returns:
        SSE(``text/event-stream``) ``StreamingResponse``.
    """
    config = get_config()

    def event_stream():
        try:
            # 1. 파싱
            yield _sse({"phase": "parsing"})
            content = ""
            if request.file_path:
                safe_path = _resolve_safe_upload_path(request.file_path)
                content = load_file(safe_path).content
            elif request.content:
                content = request.content
            else:
                yield _sse({"phase": "error", "detail": "Either content or file_path must be provided"})
                return

            if not content:
                yield _sse({"phase": "error", "detail": "Empty content"})
                return

            # 2. 청킹
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
                yield _sse({"phase": "done", "chunk_count": 0})
                yield "data: [DONE]\n\n"
                return

            if request.user_id:
                for chunk in chunks:
                    chunk.metadata["user_id"] = request.user_id

            total = len(chunks)
            yield _sse({"phase": "chunking", "total": total})

            # 3. 임베딩 (배치 단위 진행률 보고)
            yield _sse({"phase": "preparing"})
            searcher = get_searcher()  # 첫 호출 시 임베딩 모델 로딩
            batch_size = config.embedding.batch_size
            contents = [c.content for c in chunks]
            embedding_batches: list[np.ndarray] = []
            for start in range(0, total, batch_size):
                batch = contents[start:start + batch_size]
                embedding_batches.append(searcher.embedder.embed(batch))
                done_n = min(start + batch_size, total)
                yield _sse({"phase": "embedding", "current": done_n, "total": total})

            embeddings = np.vstack(embedding_batches) if embedding_batches else None

            # 4. 인덱싱 (in-memory 상태 보호 락)
            yield _sse({"phase": "indexing"})
            index_path = Path(config.project.index_path)
            index_path.mkdir(parents=True, exist_ok=True)
            with get_searcher_lock():
                replaced = searcher.delete_by_source(filename, user_id=request.user_id)
                searcher.index(chunks, embeddings=embeddings)
                searcher.save(index_path)

            # 5. 완료
            yield _sse({"phase": "done", "chunk_count": total, "replaced": replaced})
            yield "data: [DONE]\n\n"
        except HTTPException as exc:
            yield _sse({"phase": "error", "detail": str(exc.detail)})
        except Exception as exc:  # noqa: BLE001 - 스트림 소비자에게 오류 전달
            logger.error("index_stream_failed", error=str(exc))
            yield _sse({"phase": "error", "detail": "인덱싱 중 오류가 발생했습니다."})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.delete("/index/by-source", response_model=IndexDeleteResponse)
async def delete_index_by_source(request: IndexDeleteRequest):
    """``source``(파일명) + ``user_id`` 단위로 인덱스 청크를 삭제한다.

    UI에서 문서를 삭제할 때 호출되어 Qdrant·BM25 in-memory 상태를 일관되게
    정리한다. 인덱싱 흐름과 동일한 ``_searcher_lock`` 안에서 처리하여 동시
    검색과의 경쟁 조건을 방지한다.

    Args:
        request: 삭제 대상 ``filename`` 과 ``user_id`` 를 담은 요청 객체.

    Returns:
        삭제된 청크 수와 결과 메시지를 담은 ``IndexDeleteResponse``.

    Raises:
        HTTPException: 내부 처리 중 예기치 못한 오류 발생 시 500.
    """
    config = get_config()
    searcher = get_searcher()
    index_path = Path(config.project.index_path)

    try:
        with get_searcher_lock():
            deleted = searcher.delete_by_source(
                request.filename, user_id=request.user_id
            )
            if deleted > 0:
                searcher.save(index_path)
    except Exception as e:
        logger.error(
            "index_delete_failed",
            filename=request.filename,
            user_id=request.user_id or "anonymous",
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=str(e))

    user_label = request.user_id or "anonymous"
    return IndexDeleteResponse(
        deleted_count=deleted,
        message=f"Deleted {deleted} chunks for {request.filename} (user={user_label})",
    )
