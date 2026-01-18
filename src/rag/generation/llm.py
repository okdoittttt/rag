"""LLM 인터페이스 및 구현체

Google Gemini API 및 Ollama를 사용하는 LLM 구현체를 제공합니다.
스트리밍 응답도 지원합니다.
"""

from __future__ import annotations

import os
import json
from abc import ABC, abstractmethod
from typing import Iterator, Optional

import requests
import google.generativeai as genai
from google.api_core import exceptions
from dotenv import load_dotenv

from rag.config import get_config, GenerationConfig
from rag.logger import get_logger

load_dotenv()


logger = get_logger(__name__)


class LLM(ABC):
    """LLM 추상 기본 클래스"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """프롬프트에 대한 응답 생성"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str) -> Iterator[str]:
        """스트리밍 응답 생성"""
        pass


class GeminiLLM(LLM):
    """Google Gemini API 구현체"""
    
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        config = get_config()
        self.model_name = model_name or config.generation.model
        
        # API 키 로드
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API Key not found. Please set GOOGLE_API_KEY environment variable."
            )
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("gemini_initialized", model=self.model_name)
        except Exception as e:
            logger.error("gemini_init_failed", error=str(e))
            raise
            
    def generate(self, prompt: str) -> str:
        try:
            logger.debug("gemini_generating", prompt_length=len(prompt))
            response = self.model.generate_content(prompt)
            
            if not response.text:
                logger.warning("gemini_empty_response", feedback=response.prompt_feedback)
                return "죄송합니다. 답변을 생성할 수 없습니다."
                
            return response.text
            
        except exceptions.GoogleAPIError as e:
            logger.error("gemini_api_error", error=str(e))
            return f"API 오류: {str(e)}"
        except Exception as e:
            logger.error("gemini_generation_failed", error=str(e))
            return f"오류 발생: {str(e)}"

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Gemini 스트리밍 응답 생성"""
        try:
            logger.debug("gemini_streaming", prompt_length=len(prompt))
            response = self.model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except exceptions.GoogleAPIError as e:
            logger.error("gemini_stream_error", error=str(e))
            yield f"API 오류: {str(e)}"
        except Exception as e:
            logger.error("gemini_stream_failed", error=str(e))
            yield f"오류 발생: {str(e)}"


class OllamaLLM(LLM):
    """Ollama API 구현체"""
    
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        config = get_config()
        self.base_url = base_url or config.generation.ollama.base_url
        self.model = model or config.generation.ollama.model
        
        logger.info("ollama_initialized", base_url=self.base_url, model=self.model)
        
    def generate(self, prompt: str) -> str:
        try:
            logger.debug("ollama_generating", model=self.model, prompt_length=len(prompt))
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error("ollama_api_error", error=str(e))
            return f"Ollama 연결 오류: {str(e)}"
        except Exception as e:
            logger.error("ollama_generation_failed", error=str(e))
            return f"오류 발생: {str(e)}"

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Ollama 스트리밍 응답 생성"""
        try:
            logger.debug("ollama_streaming", model=self.model, prompt_length=len(prompt))
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    text = data.get("response", "")
                    if text:
                        yield text
                        
        except requests.exceptions.RequestException as e:
            logger.error("ollama_stream_error", error=str(e))
            yield f"Ollama 연결 오류: {str(e)}"
        except Exception as e:
            logger.error("ollama_stream_failed", error=str(e))
            yield f"오류 발생: {str(e)}"


def get_llm(provider: str | None = None) -> LLM:
    """설정된 Provider에 맞는 LLM 인스턴스 반환
    
    Args:
        provider: 'gemini' 또는 'ollama'. None이면 config 설정을 따름.
    """
    config = get_config()
    target_provider = provider or config.generation.provider
    
    if target_provider == "ollama":
        return OllamaLLM()
    else:
        return GeminiLLM()
