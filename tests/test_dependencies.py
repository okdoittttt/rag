"""``APIKeyMiddleware`` 동작 테스트.

다음 동작을 검증한다.
    1. ``API_KEY`` 가 비어 있을 때 dev 환경에서는 미들웨어 비활성, production
       환경(``APP_ENV=production``)에서는 ``RuntimeError`` 로 fail-fast.
    2. 키가 설정된 상태에서 ``X-API-Key`` 가 일치/불일치/누락/길이 다름에
       따라 200 또는 401을 정확히 반환.
    3. ``PUBLIC_PATHS`` 와 ``OPTIONS`` 메서드는 인증을 면제.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _build_app(monkeypatch, *, api_key: str | None, env: str | None = None):
    """환경변수를 셋업한 뒤 미들웨어를 새로 import 하여 앱을 구성한다.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        api_key: ``API_KEY`` 환경변수 값. ``None`` 이면 제거.
        env: ``APP_ENV`` 환경변수 값. ``None`` 이면 ``APP_ENV``/``ENV`` 둘 다 제거.

    Returns:
        ``protected`` 와 ``health`` 엔드포인트가 마운트된 ``FastAPI`` 앱.
    """
    if api_key is None:
        monkeypatch.delenv("API_KEY", raising=False)
    else:
        monkeypatch.setenv("API_KEY", api_key)

    if env is None:
        monkeypatch.delenv("APP_ENV", raising=False)
        monkeypatch.delenv("ENV", raising=False)
    else:
        monkeypatch.setenv("APP_ENV", env)

    # 환경변수를 반영한 새 미들웨어 인스턴스를 만들기 위해 모듈 reload
    import api.dependencies as dependencies_module

    importlib.reload(dependencies_module)

    app = FastAPI()
    app.add_middleware(dependencies_module.APIKeyMiddleware)

    @app.get("/protected")
    def _protected():
        return {"ok": True}

    @app.get("/health")
    def _health():
        return {"status": "ok"}

    return app


def test_dev_without_api_key_passes_through(monkeypatch):
    """dev 환경에서 ``API_KEY`` 미설정이면 헤더 없이도 통과한다."""
    app = _build_app(monkeypatch, api_key=None, env=None)
    client = TestClient(app)

    resp = client.get("/protected")
    assert resp.status_code == 200


def test_dev_with_matching_key_returns_200(monkeypatch):
    """키 일치 시 200 응답."""
    app = _build_app(monkeypatch, api_key="secret", env=None)
    client = TestClient(app)

    resp = client.get("/protected", headers={"X-API-Key": "secret"})
    assert resp.status_code == 200


def test_dev_with_wrong_key_returns_401(monkeypatch):
    """키 불일치 시 401."""
    app = _build_app(monkeypatch, api_key="secret", env=None)
    client = TestClient(app)

    resp = client.get("/protected", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_dev_with_missing_header_returns_401(monkeypatch):
    """헤더 누락 시 401."""
    app = _build_app(monkeypatch, api_key="secret", env=None)
    client = TestClient(app)

    resp = client.get("/protected")
    assert resp.status_code == 401


def test_production_without_api_key_fails_fast(monkeypatch):
    """production 환경에서 키 미설정 시 ``RuntimeError`` 로 앱 구성 자체가 실패한다."""
    with pytest.raises(RuntimeError):
        _build_app(monkeypatch, api_key=None, env="production")


def test_public_path_bypasses_auth(monkeypatch):
    """``PUBLIC_PATHS`` 는 키 설정 여부와 무관하게 인증 없이 통과한다."""
    app = _build_app(monkeypatch, api_key="secret", env=None)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200


def test_options_request_bypasses_auth(monkeypatch):
    """OPTIONS(프리플라이트) 요청은 인증 면제."""
    app = _build_app(monkeypatch, api_key="secret", env=None)
    client = TestClient(app)

    resp = client.options("/protected")
    # FastAPI 기본 OPTIONS 응답 (405 또는 200 등) 이 무엇이든 401은 아니어야 한다.
    assert resp.status_code != 401


def test_short_wrong_key_returns_401(monkeypatch):
    """길이가 다른 잘못된 키도 안전하게 401 처리. ``compare_digest`` 가 길이 차이로
    예외를 던지지 않는지 확인."""
    app = _build_app(monkeypatch, api_key="secret", env=None)
    client = TestClient(app)

    resp = client.get("/protected", headers={"X-API-Key": "s"})
    assert resp.status_code == 401
