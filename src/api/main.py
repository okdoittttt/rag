"""FastAPI 앱 진입점

Terminal RAG REST API 서버
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import ask, search, index
from api.exceptions import RAGException


app = FastAPI(
    title="Terminal RAG API",
    description="Document Q&A with RAG pipeline",
    version="0.1.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(ask.router, tags=["QA"])
app.include_router(search.router, tags=["Search"])
app.include_router(index.router, tags=["Index"])


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
