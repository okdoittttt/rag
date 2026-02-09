"""API 인증 의존성

API Key 기반 서비스 간 인증 미들웨어를 제공합니다.
Next.js → FastAPI 호출 시 X-API-Key 헤더를 검증합니다.
"""

import os

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# API Key 인증이 면제되는 경로
PUBLIC_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}


def _get_api_key() -> str | None:
    """환경 변수에서 API Key를 가져옵니다."""
    return os.environ.get("API_KEY")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key 검증 미들웨어

    X-API-Key 헤더가 환경 변수 API_KEY와 일치하는지 확인합니다.
    API_KEY가 설정되지 않은 경우 미들웨어를 건너뜁니다 (개발 환경 호환).
    """

    async def dispatch(self, request: Request, call_next):
        api_key = _get_api_key()

        # API_KEY가 설정되지 않은 경우 미들웨어 비활성화 (개발 편의)
        if not api_key:
            return await call_next(request)

        # 공개 경로는 인증 면제
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # OPTIONS 요청 (CORS preflight)은 면제
        if request.method == "OPTIONS":
            return await call_next(request)

        # X-API-Key 헤더 검증
        request_api_key = request.headers.get("X-API-Key")
        if not request_api_key or request_api_key != api_key:
            raise HTTPException(
                status_code=401,
                detail="유효하지 않은 API Key입니다.",
            )

        return await call_next(request)
