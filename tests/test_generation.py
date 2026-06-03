"""Generation 모듈 테스트"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.chunking.chunk import Chunk
from rag.generation import GeminiLLM, build_prompt
from rag.generation.prompt import SUMMARY_MODE_INSTRUCTION


def _fake_gemini_response(text: str, prompt_feedback=None):
    """``candidates→content→parts`` 구조의 가짜 Gemini 응답/청크를 만든다.

    ``_extract_gemini_text`` 가 parts 의 ``text`` 를 읽으므로, 신규 SDK 응답을
    흉내 낸 ``SimpleNamespace`` 트리를 반환한다. ``text`` 가 비면 parts 도 비운다.
    """
    parts = [SimpleNamespace(text=text, thought=None)] if text else []
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content, finish_reason=1)
    return SimpleNamespace(candidates=[candidate], prompt_feedback=prompt_feedback)


class TestPrompt:
    """프롬프트 생성 테스트"""
    
    def test_build_prompt_format(self):
        """컨텍스트와 질문이 올바르게 결합되는지 확인"""
        chunks = [
            Chunk(content="Python is great.", metadata={"filename": "doc1.txt", "chunk_index": 0}),
            Chunk(content="Rust is safe.", metadata={"filename": "doc2.md", "chunk_index": 1}),
        ]
        query = "Which language is safe?"
        
        prompt = build_prompt(query, chunks)
        
        # 필수 요소 포함 확인
        assert "[Context]" in prompt
        assert "[Chunk 1]" in prompt
        assert "doc1.txt" in prompt
        assert "Python is great" in prompt
        assert "[Chunk 2]" in prompt
        assert "[Question]" in prompt
        assert query in prompt
        assert "[Answer]" in prompt

    def test_build_prompt_summary_mode_prepends_instruction(self):
        """``mode='summary'`` 일 때 요약 전용 지침이 prepend 되어야 한다."""
        chunks = [
            Chunk(content="A", metadata={"filename": "d.md", "chunk_index": 0}),
        ]
        default_prompt = build_prompt("요약해줘", chunks, mode="default")
        summary_prompt = build_prompt("요약해줘", chunks, mode="summary")

        assert SUMMARY_MODE_INSTRUCTION.strip().split("\n", 1)[0] not in default_prompt
        assert SUMMARY_MODE_INSTRUCTION.strip().split("\n", 1)[0] in summary_prompt
        # 컨텍스트와 질문 구조는 동일하게 유지되어야 한다.
        assert "[Context]" in summary_prompt
        assert "[Question]" in summary_prompt


class TestGeminiLLM:
    """Gemini LLM 테스트 (``google-genai`` SDK)"""

    @pytest.fixture
    def mock_genai(self):
        """``rag.generation.llm.genai`` 를 mock 하고 mock client 를 함께 노출한다.

        ``types``/``errors`` 는 실제 SDK 객체를 그대로 사용한다(설정 구성 검증 겸용).
        """
        with patch("rag.generation.llm.genai") as mock:
            client = MagicMock()
            mock.Client.return_value = client
            yield mock, client

    def test_init_raises_error_without_api_key(self):
        """API 키가 없으면 에러 발생"""
        # 환경변수 제거
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiLLM(api_key=None)

    def test_init_with_api_key(self, mock_genai):
        """API 키가 있으면 Client 가 키와 함께 생성된다."""
        mock, client = mock_genai
        llm = GeminiLLM(api_key="fake_key")

        mock.Client.assert_called_once_with(api_key="fake_key")
        assert llm.client is client

    def test_generate_calls_api(self, mock_genai):
        """API 호출 및 parts 에서 텍스트 추출 확인"""
        mock, client = mock_genai
        client.models.generate_content.return_value = _fake_gemini_response(
            "This is a generated answer."
        )

        llm = GeminiLLM(api_key="fake_key")
        response = llm.generate("Test prompt")

        # contents 인자로 프롬프트가 전달됐는지 확인
        _, kwargs = client.models.generate_content.call_args
        assert kwargs["contents"] == "Test prompt"
        assert response == "This is a generated answer."

    def test_generate_handles_empty_response(self, mock_genai):
        """빈 응답(안전 차단 등) 처리 확인"""
        mock, client = mock_genai
        client.models.generate_content.return_value = _fake_gemini_response(
            "", prompt_feedback="Blocked due to safety"
        )

        llm = GeminiLLM(api_key="fake_key")
        response = llm.generate("Unsafe prompt")

        assert "죄송합니다" in response
        assert "안전 정책" in response

    def test_generate_stream_yields_chunk_text(self, mock_genai):
        """스트리밍 청크의 텍스트를 순서대로 흘린다."""
        mock, client = mock_genai
        client.models.generate_content_stream.return_value = [
            _fake_gemini_response("안녕"),
            _fake_gemini_response("하세요"),
        ]

        llm = GeminiLLM(api_key="fake_key")
        out = "".join(llm.generate_stream("Test prompt"))

        assert out == "안녕하세요"
        _, kwargs = client.models.generate_content_stream.call_args
        assert kwargs["contents"] == "Test prompt"

    def test_generate_stream_empty_yields_fallback(self, mock_genai):
        """빈 스트림이면 안내 폴백 문구를 흘린다(깜깜이 방지)."""
        mock, client = mock_genai
        client.models.generate_content_stream.return_value = [
            _fake_gemini_response(""),  # part 없는 청크만
        ]

        llm = GeminiLLM(api_key="fake_key")
        out = "".join(llm.generate_stream("Test prompt"))

        assert "빈 응답" in out
