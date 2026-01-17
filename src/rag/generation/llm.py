"""LLM 인터페이스 및 구현체

Google Gemini API를 사용하는 LLM 구현체를 제공합니다.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import google.generativeai as genai
from google.api_core import exceptions

from rag.config import get_config
from rag.logger import get_logger


logger = get_logger(__name__)


class LLM(ABC):
    """LLM 추상 기본 클래스"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """프롬프트에 대한 응답 생성
        
        Args:
            prompt: 입력 프롬프트
            
        Returns:
            생성된 텍스트 응답
        """
        pass


class GeminiLLM(LLM):
    """Google Gemini API 구현체"""
    
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        """
        Args:
            model_name: 모델명 (기본값: 설정 파일 또는 gemini-1.5-flash)
            api_key: API 키 (기본값: 환경변수 GOOGLE_API_KEY)
        """
        config = get_config()
        self.model_name = model_name or config.generation.model
        
        # API 키 로드 (인자 > 환경변수)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API Key not found. Please set GOOGLE_API_KEY environment variable."
            )
            
        # Gemini 설정
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("llm_initialized", model=self.model_name)
        except Exception as e:
            logger.error("llm_init_failed", error=str(e))
            raise
            
    def generate(self, prompt: str) -> str:
        """프롬프트에 대한 응답 생성"""
        try:
            logger.debug("generating_response", prompt_length=len(prompt))
            
            # 응답 생성
            response = self.model.generate_content(prompt)
            
            # 안전 설정 등으로 인해 응답이 차단된 경우 처리
            if not response.text:
                logger.warning("empty_response_from_llm", feedback=response.prompt_feedback)
                return "죄송합니다. 안전 정책 등의 이유로 답변을 생성할 수 없습니다."
                
            return response.text
            
        except exceptions.GoogleAPIError as e:
            logger.error("gemini_api_error", error=str(e))
            return f"API 호출 중 오류가 발생했습니다: {str(e)}"
        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
