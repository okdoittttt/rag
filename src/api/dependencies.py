"""API 인증 의존성.

API Key 기반 서비스 간 인증 미들웨어를 제공한다.
Next.js → FastAPI 호출 시 ``X-API-Key`` 헤더를 검증한다.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from rag.logger import get_logger


logger = get_logger(__name__)


PUBLIC_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json"}

_PRODUCTION_ENV_VALUES = {"production", "prod"}


def _is_production() -> bool:
    """현재 실행 환경이 production인지 판단한다.

    ``APP_ENV`` 또는 ``ENV`` 환경변수 중 하나라도 ``production``/``prod``
    (대소문자 무시)이면 production으로 본다.

    Returns:
        production이면 ``True``, 아니면 ``False``.
    """
    for var in ("APP_ENV", "ENV"):
        value = os.environ.get(var, "").strip().lower()
        if value in _PRODUCTION_ENV_VALUES:
            return True
    return False


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key 검증 미들웨어.

    ``X-API-Key`` 헤더가 환경변수 ``API_KEY``와 일치하는지 검증한다.
    환경변수는 미들웨어 생성 시 한 번만 읽어 인스턴스에 캐시한다.

    동작 정책:
        - production 환경(``APP_ENV``/``ENV``가 ``production``)에서 ``API_KEY``가
          비어 있으면 ``RuntimeError``를 던져 앱이 무인증 상태로 기동되지
          못하게 한다 (fail-fast).
        - production이 아닌 환경에서 ``API_KEY``가 비어 있으면 미들웨어를
          비활성화한다 (개발 편의, fail-open).
        - 키 비교는 ``hmac.compare_digest``를 사용해 상수 시간 비교를 보장한다.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._api_key = os.environ.get("API_KEY", "").strip()

        if not self._api_key:
            if _is_production():
                raise RuntimeError(
                    "API_KEY가 설정되지 않았습니다. production 환경에서는 "
                    "반드시 설정해야 합니다."
                )
            logger.warning(
                "api_key_not_set_dev_mode",
                detail="개발 환경: API Key 미들웨어가 비활성화됩니다.",
            )

    async def dispatch(self, request: Request, call_next):
        """요청별 인증 처리.

        Args:
            request: 들어온 HTTP 요청.
            call_next: 다음 미들웨어/엔드포인트 핸들러.

        Returns:
            인증 통과 시 다음 핸들러의 응답. ``X-API-Key`` 가 없거나 일치하지
            않으면 401 ``JSONResponse``.

        Note:
            ``BaseHTTPMiddleware`` 는 라우팅 바깥에 위치하므로 여기서
            ``HTTPException`` 을 ``raise`` 해도 FastAPI 의 예외 핸들러가 401 로
            변환하지 못한다(상위 ``ServerErrorMiddleware`` 까지 전파됨). 따라서
            ``JSONResponse`` 를 직접 반환해 401 을 보장한다.
        """
        if not self._api_key:
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if request.method == "OPTIONS":
            return await call_next(request)

        request_api_key = request.headers.get("X-API-Key", "")
        if not request_api_key or not hmac.compare_digest(
            request_api_key, self._api_key
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "유효하지 않은 API Key입니다."},
            )

        return await call_next(request)
