"""FastAPI 앱 진입점

Terminal RAG REST API 서버
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.config import get_config


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
