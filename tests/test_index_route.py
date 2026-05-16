"""``/index`` 라우터의 Path Traversal 차단 테스트.

``request.file_path`` 가 화이트리스트(`UPLOAD_DIR` 또는 기본 베이스) 외부를
가리킬 때 400을 반환하는지, 정상 경로는 통과되는지 확인한다.
실제 Qdrant/embedding 의존성은 monkeypatch 로 제거하고 라우트 자체의
입력 검증 로직만 검증한다.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """``/index`` 라우터만 마운트한 격리된 FastAPI TestClient.

    실제 ``HybridSearcher`` / ``Embedder`` / ``Qdrant`` 의존성을 제거하기
    위해 ``get_searcher`` 와 ``chunk_document`` 를 가벼운 mock 으로 치환한다.
    ``UPLOAD_DIR`` 환경변수를 ``tmp_path`` 로 설정하여 화이트리스트의
    "정상 베이스"를 테스트 디렉터리로 고정한다.
    """
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    # 라우터 모듈 import 전에 환경변수가 설정되어 있어야 한다.
    from api.routes import index as index_module

    fake_searcher = MagicMock()
    fake_searcher.delete_by_source.return_value = 0
    fake_searcher.index.return_value = None
    fake_searcher.save.return_value = None
    monkeypatch.setattr(index_module, "get_searcher", lambda: fake_searcher)
    monkeypatch.setattr(index_module, "get_searcher_lock", lambda: _NullLock())

    app = FastAPI()
    app.include_router(index_module.router)
    return TestClient(app)


class _NullLock:
    """``with`` 구문 호환용 더미 락. 테스트 단일 스레드라 동기화 불필요."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_index_accepts_path_inside_upload_dir(client, tmp_path):
    """``UPLOAD_DIR`` 하위 경로는 200으로 통과해야 한다."""
    target = tmp_path / "ok.txt"
    target.write_text("hello world\n안녕하세요", encoding="utf-8")

    resp = client.post(
        "/index",
        json={"file_path": str(target), "filename": "ok.txt", "user_id": "u1"},
    )
    assert resp.status_code == 200, resp.text


def test_index_rejects_absolute_path_outside_base(client):
    """베이스 밖 절대 경로(`/etc/passwd`)는 400 으로 거부되어야 한다."""
    resp = client.post(
        "/index",
        json={"file_path": "/etc/passwd", "filename": "p.txt"},
    )
    assert resp.status_code == 400
    assert "허용되지 않은" in resp.json()["detail"]


def test_index_rejects_relative_traversal(client, tmp_path):
    """``..`` 를 이용해 베이스 밖을 노리는 경로는 400 이어야 한다."""
    traversal = str(tmp_path / ".." / ".." / ".." / "etc" / "passwd")
    resp = client.post(
        "/index",
        json={"file_path": traversal, "filename": "p.txt"},
    )
    assert resp.status_code == 400


def test_index_rejects_symlink_pointing_outside(client, tmp_path):
    """베이스 안에 있어 보이지만 심볼릭 링크가 외부를 가리키면 400."""
    outside_dir = tmp_path.parent / "symlink_target_for_index_route"
    outside_dir.mkdir(exist_ok=True)
    outside = outside_dir / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("심볼릭 링크 생성을 지원하지 않는 환경")

    resp = client.post(
        "/index",
        json={"file_path": str(link), "filename": "link.txt"},
    )
    assert resp.status_code == 400


def test_index_accepts_content_without_file_path(client):
    """``content`` 만 전달되는 기존 흐름은 변경되지 않는다."""
    resp = client.post(
        "/index",
        json={"content": "그냥 본문\n두번째 줄", "filename": "inline.txt"},
    )
    assert resp.status_code == 200


def test_index_missing_file_inside_base_returns_400(client, tmp_path):
    """베이스 내부지만 실제 존재하지 않는 파일은 ``load_file`` 단에서 400."""
    missing = tmp_path / "not_exist.txt"
    resp = client.post(
        "/index",
        json={"file_path": str(missing), "filename": "not_exist.txt"},
    )
    assert resp.status_code == 400


def test_delete_index_by_source_removes_chunks(monkeypatch, tmp_path):
    """매칭되는 청크가 있으면 ``delete_by_source`` 호출 후 ``save`` 까지 수행한다."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    from api.routes import index as index_module

    fake_searcher = MagicMock()
    fake_searcher.delete_by_source.return_value = 3
    monkeypatch.setattr(index_module, "get_searcher", lambda: fake_searcher)
    monkeypatch.setattr(index_module, "get_searcher_lock", lambda: _NullLock())

    app = FastAPI()
    app.include_router(index_module.router)
    test_client = TestClient(app)

    resp = test_client.request(
        "DELETE",
        "/index/by-source",
        json={"filename": "doc.txt", "user_id": "u1"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_count"] == 3
    fake_searcher.delete_by_source.assert_called_once_with("doc.txt", user_id="u1")
    fake_searcher.save.assert_called_once()


def test_delete_index_by_source_no_match_returns_zero(monkeypatch, tmp_path):
    """매칭되는 청크가 없으면 ``deleted_count`` 가 0이고 ``save`` 호출도 없다."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

    from api.routes import index as index_module

    fake_searcher = MagicMock()
    fake_searcher.delete_by_source.return_value = 0
    monkeypatch.setattr(index_module, "get_searcher", lambda: fake_searcher)
    monkeypatch.setattr(index_module, "get_searcher_lock", lambda: _NullLock())

    app = FastAPI()
    app.include_router(index_module.router)
    test_client = TestClient(app)

    resp = test_client.request(
        "DELETE",
        "/index/by-source",
        json={"filename": "ghost.txt", "user_id": "u1"},
    )

    assert resp.status_code == 200
    assert resp.json()["deleted_count"] == 0
    fake_searcher.delete_by_source.assert_called_once_with("ghost.txt", user_id="u1")
    fake_searcher.save.assert_not_called()
