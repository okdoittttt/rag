"""FastAPI 앱 진입점

Terminal RAG REST API 서버
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import ask, search, index, config
from api.routes.ask import get_reranker_instance, get_searcher
from api.exceptions import RAGException
from api.dependencies import APIKeyMiddleware
from rag.config import get_config
from rag.logger import get_logger


_logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 부팅 시 검색기·Reranker 를 사전 로드해 첫 요청 지연을 제거한다.

    - 검색기 싱글톤은 임베딩 모델/벡터 스토어 핸들을 준비한다.
    - Reranker 는 ``use_reranker=True`` 일 때만 미리 로드한다. 로드 실패 시
      ``get_reranker_instance`` 가 ``None`` 을 반환해 폴백되므로 부팅을
      막지는 않는다.
    """

    try:
        get_searcher()
    except Exception as exc:  # 인덱스 미생성 등은 정상 흐름이므로 경고만.
        _logger.warning("searcher_warmup_failed", error=str(exc))

    if get_config().retrieval.use_reranker:
        # 결과(None 포함)를 기록만 한다. 음성 캐시는 함수 내부에서 처리.
        reranker = get_reranker_instance()
        if reranker is None:
            _logger.warning("reranker_warmup_skipped_due_to_load_failure")
        else:
            _logger.info("reranker_warmed_up")

    yield


app = FastAPI(
    title="Terminal RAG API",
    description="RAG 기반 문서 Q&A API. 문서를 인덱싱하고 질문에 답변합니다.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "QA", "description": "질문-답변 (일반 및 스트리밍)"},
        {"name": "Search", "description": "문서 검색"},
        {"name": "Index", "description": "문서 인덱싱"},
    ]
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key 인증 미들웨어 (CORS 미들웨어 이후에 추가)
app.add_middleware(APIKeyMiddleware)

# 라우터 등록
app.include_router(ask.router, tags=["QA"])
app.include_router(search.router, tags=["Search"])
app.include_router(index.router, tags=["Index"])
app.include_router(config.router, prefix="/config", tags=["Config"])


# === 글로벌 예외 핸들러 ===

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """RAG 예외 핸들러"""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """요청 검증 에러 핸들러"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "message": "요청 데이터가 올바르지 않습니다.",
            "detail": exc.errors(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "서버 내부 오류가 발생했습니다.",
        }
    )


# === 기본 엔드포인트 ===

@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok"}


@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "Terminal RAG API",
        "version": "0.1.0",
        "docs": "/docs",
    }
