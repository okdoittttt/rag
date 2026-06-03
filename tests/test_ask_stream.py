"""``/ask/stream`` SSE 엔드포인트의 CoT 추론–답변 분리 로직 테스트.

LLM 이 ``추론 + COT_DELIMITER + 답변`` 을 한 번에 생성한다고 가정하고,
``ask_stream`` 이 구분자를 경계로 토큰을 추론(``phase="reasoning"``)과
최종 답변(``text``)으로 정확히 라벨링하는지 검증한다. 구분자가 청크 경계에
걸쳐 분할되는 경우(holdback)와 모델이 형식을 무시한 경우(폴백)도 다룬다.

실제 검색/LLM 의존성은 ``_search_documents`` 와 ``get_llm`` 을 monkeypatch 하여
제거하고, 스트림 파싱 로직만 격리해 검증한다.
"""

from __future__ import annotations

import json
from typing import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.schemas import AskRequest
from rag.chunking.chunk import Chunk
from rag.generation.prompt import COT_DELIMITER


def _chunk(idx: int, source: str, content: str = "본문") -> Chunk:
    """간단한 청크 헬퍼."""
    return Chunk.create(
        content=content,
        source=source,
        chunk_index=idx,
        start_char=0,
        end_char=len(content),
    )


class _FakeLLM:
    """``generate_stream`` 이 미리 정한 청크들을 순서대로 yield 하는 가짜 LLM."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def generate(self, prompt: str) -> str:
        return "".join(self._chunks)

    def generate_stream(self, prompt: str) -> Iterator[str]:
        for c in self._chunks:
            yield c


@pytest.fixture
def make_client(monkeypatch):
    """스트림 청크와 검색 결과를 주입해 ``/ask/stream`` TestClient 를 만드는 팩토리.

    Returns:
        ``_make(stream_chunks, search_result=None)`` 형태의 팩토리 함수.
        ``search_result`` 미지정 시 단일 청크 검색 결과를 기본으로 사용한다.
    """

    def _make(stream_chunks: list[str], search_result=None) -> TestClient:
        from api.routes import ask as ask_module

        if search_result is None:
            c = _chunk(0, "doc.md", "근거 본문")
            search_result = ([c], [(c, 0.9)])

        monkeypatch.setattr(ask_module, "_search_documents", lambda req: search_result)
        monkeypatch.setattr(ask_module, "get_llm", lambda **kwargs: _FakeLLM(stream_chunks))

        app = FastAPI()
        app.include_router(ask_module.router)
        return TestClient(app)

    return _make


def _parse_sse(raw: str) -> list[dict | str]:
    """SSE 본문의 ``data:`` 라인을 파싱한다.

    Args:
        raw: ``text/event-stream`` 응답 본문 전체.

    Returns:
        각 이벤트 페이로드 리스트. JSON 이벤트는 ``dict``, 종료 신호는
        문자열 ``"[DONE]"`` 로 반환한다.
    """
    events: list[dict | str] = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def _reasoning_text(events: list[dict | str]) -> str:
    """이벤트 목록에서 추론 토큰(``phase="reasoning"``)을 이어 붙인다."""
    return "".join(
        e["text"] for e in events
        if isinstance(e, dict) and e.get("phase") == "reasoning" and "text" in e
    )


def _answer_text(events: list[dict | str]) -> str:
    """이벤트 목록에서 최종 답변 토큰(``phase`` 없는 ``text``)을 이어 붙인다."""
    return "".join(
        e["text"] for e in events
        if isinstance(e, dict) and "text" in e and "phase" not in e
    )


def _post_stream(client: TestClient, query: str = "질문입니다") -> list[dict | str]:
    """``/ask/stream`` 을 호출하고 파싱된 이벤트 목록을 반환한다."""
    resp = client.post("/ask/stream", json=AskRequest(query=query).model_dump())
    assert resp.status_code == 200, resp.text
    return _parse_sse(resp.text)


class TestAskStream:
    """``/ask/stream`` 이벤트 순서 및 CoT 분리 검증."""

    def test_stream_emits_phase_order(self, make_client):
        """searching → analyzing → reasoning_start 순서 후 [DONE] 으로 끝난다."""
        client = make_client(["추론입니다.", f"{COT_DELIMITER}\n", "답변입니다."])
        events = _post_stream(client)

        assert events[-1] == "[DONE]"
        # 텍스트 없는 진행 단계(phase-only) 이벤트만 추출
        phase_only = [
            e["phase"] for e in events
            if isinstance(e, dict) and "phase" in e and "text" not in e
        ]
        assert phase_only == ["searching", "analyzing", "reasoning_start"]

    def test_stream_splits_reasoning_and_answer(self, make_client):
        """구분자 앞은 추론, 뒤는 최종 답변으로 분리된다."""
        client = make_client(["추론입니다.", f"{COT_DELIMITER}\n", "실제 답변입니다."])
        events = _post_stream(client)

        assert _reasoning_text(events).strip() == "추론입니다."
        assert _answer_text(events).strip() == "실제 답변입니다."

    def test_stream_delimiter_across_chunks(self, make_client):
        """구분자가 두 청크에 걸쳐 분할돼도 추론/답변을 정확히 나눈다."""
        # COT_DELIMITER 가 "===최종" + "답변===" 로 쪼개져 도착하도록 구성한다.
        head, tail = COT_DELIMITER[:5], COT_DELIMITER[5:]
        client = make_client(
            ["이것은 추론 과정입니다.", head, f"{tail}\n최종 답변 본문입니다."]
        )
        events = _post_stream(client)

        assert _reasoning_text(events).strip() == "이것은 추론 과정입니다."
        assert _answer_text(events).strip() == "최종 답변 본문입니다."

    def test_stream_no_delimiter_emits_only_reasoning(self, make_client):
        """구분자가 끝내 없으면 전체를 추론으로 흘리고 답변 토큰은 내지 않는다.

        (프론트엔드가 답변이 비고 추론만 있을 때 추론을 답변으로 승격 처리한다.)
        """
        client = make_client(["구분자 없이 그냥 답해버린 모델의 출력입니다."])
        events = _post_stream(client)

        assert events[-1] == "[DONE]"
        assert _answer_text(events) == ""
        assert _reasoning_text(events).strip() == "구분자 없이 그냥 답해버린 모델의 출력입니다."

    def test_stream_no_chunks_returns_message(self, make_client):
        """검색 결과가 0건이면 안내 문구와 [DONE] 만 내보낸다."""
        client = make_client(["사용되지 않음"], search_result=([], []))
        events = _post_stream(client)

        assert events[-1] == "[DONE]"
        assert _answer_text(events) == "관련 문서를 찾을 수 없습니다."
        # 분석/추론 단계로 진입하지 않아야 한다.
        phases = [e.get("phase") for e in events if isinstance(e, dict)]
        assert "analyzing" not in phases
        assert "reasoning_start" not in phases
