"""Generation 모듈 테스트"""

import os
from unittest.mock import MagicMock, patch

import pytest

from rag.chunking.chunk import Chunk
from rag.generation import GeminiLLM, build_prompt


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


class TestGeminiLLM:
    """Gemini LLM 테스트"""
    
    @pytest.fixture
    def mock_genai(self):
        with patch("rag.generation.llm.genai") as mock:
            yield mock
            
    def test_init_raises_error_without_api_key(self):
        """API 키가 없으면 에러 발생"""
        # 환경변수 제거
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiLLM(api_key=None)
                
    def test_init_with_api_key(self, mock_genai):
        """API 키가 있으면 정상 초기화"""
        llm = GeminiLLM(api_key="fake_key")
        
        # 설정 호출 확인
        mock_genai.configure.assert_called_with(api_key="fake_key")
        # 모델 생성 확인
        mock_genai.GenerativeModel.assert_called_once()
        assert llm.model is not None

    def test_generate_calls_api(self, mock_genai):
        """API 호출 및 응답 반환 확인"""
        # Mock 모델 및 응답 설정
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a generated answer."
        mock_model.generate_content.return_value = mock_response
        
        mock_genai.GenerativeModel.return_value = mock_model
        
        llm = GeminiLLM(api_key="fake_key")
        response = llm.generate("Test prompt")
        
        # API 호출 확인
        mock_model.generate_content.assert_called_with("Test prompt")
        assert response == "This is a generated answer."

    def test_generate_handles_empty_response(self, mock_genai):
        """빈 응답(안전 차단 등) 처리 확인"""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""  # 빈 텍스트
        mock_response.prompt_feedback = "Blocked due to safety"
        mock_model.generate_content.return_value = mock_response
        
        mock_genai.GenerativeModel.return_value = mock_model
        
        llm = GeminiLLM(api_key="fake_key")
        response = llm.generate("Unsafe prompt")
        
        assert "죄송합니다" in response
        assert "안전 정책" in response
