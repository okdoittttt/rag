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


def _extract_gemini_text(response) -> str:
    """Gemini 응답/스트리밍 청크에서 텍스트를 안전하게 추출한다.

    ``response.text`` 빠른 접근자는 유효한 ``Part`` 가 하나도 없는 응답/청크
    (예: ``finish_reason`` 만 담은 종료 청크, thinking 계열 모델의 사고 전용
    청크, 안전 정책으로 비어 있는 후보)에 접근하면 ``ValueError`` 를 던진다.
    스트리밍 중 그런 청크가 섞이면 전체 스트림이 끊기므로, 후보의
    ``content.parts`` 를 직접 순회하며 텍스트 Part 만 모아 반환한다.

    Args:
        response: ``generate_content`` 의 응답 객체 또는 스트리밍 청크.

    Returns:
        추출된 텍스트. 유효한 텍스트 Part 가 없으면 빈 문자열.
    """
    try:
        candidates = getattr(response, "candidates", None) or []
        texts: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)
        return "".join(texts)
    except Exception:  # noqa: BLE001 - 추출 실패는 빈 문자열로 처리
        return ""


def _chunk_finish_reason(chunk) -> object | None:
    """스트리밍 청크/응답에서 첫 후보의 ``finish_reason`` 을 안전하게 읽는다.

    Args:
        chunk: ``generate_content`` 의 응답 객체 또는 스트리밍 청크.

    Returns:
        ``finish_reason`` 값(없으면 ``None``).
    """
    try:
        for candidate in (getattr(chunk, "candidates", None) or []):
            reason = getattr(candidate, "finish_reason", None)
            if reason is not None:
                return reason
    except Exception:  # noqa: BLE001
        pass
    return None


def _describe_gemini_parts(chunk) -> list[str]:
    """진단용: 청크의 각 Part 를 ``thought=.., len=..`` 요약 문자열로 나열한다.

    빈 응답 원인(사고 전용 Part 인지, Part 자체가 없는지)을 로그로 식별하기 위함이다.
    """
    out: list[str] = []
    try:
        for candidate in (getattr(chunk, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            for part in (getattr(content, "parts", None) or []):
                text = getattr(part, "text", None) or ""
                out.append(f"thought={getattr(part, 'thought', None)},len={len(text)}")
    except Exception:  # noqa: BLE001
        pass
    return out


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

            text = _extract_gemini_text(response)
            if not text:
                logger.warning("gemini_empty_response", feedback=getattr(response, "prompt_feedback", None))
                # prompt_feedback이 존재하면 안전 정책 차단으로 간주
                if getattr(response, "prompt_feedback", None):
                    return (
                        "죄송합니다. 안전 정책에 의해 답변을 생성할 수 없습니다."
                    )
                return "죄송합니다. 답변을 생성할 수 없습니다."

            return text
            
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

            yielded = False
            last_finish_reason: object | None = None
            for chunk in response:
                last_finish_reason = _chunk_finish_reason(chunk) or last_finish_reason
                # chunk.text 빠른 접근자는 Part 가 없는 청크에서 예외를 던지므로
                # 직접 추출한다(part 없는 종료 청크는 빈 문자열 → 건너뜀).
                text = _extract_gemini_text(chunk)
                if not text:
                    logger.debug(
                        "gemini_stream_chunk_no_text",
                        finish_reason=str(_chunk_finish_reason(chunk)),
                        parts=_describe_gemini_parts(chunk),
                    )
                    continue
                yielded = True
                yield text

            if not yielded:
                # 빈 응답: UI 가 깜깜이가 되지 않도록 안내 문구를 흘리고, 원인 진단을
                # 위해 finish_reason 을 경고로 남긴다(MAX_TOKENS=2, SAFETY=3 등).
                logger.warning(
                    "gemini_stream_empty",
                    model=self.model_name,
                    finish_reason=str(last_finish_reason),
                )
                yield "죄송합니다. 모델이 빈 응답을 반환했습니다. (서버 로그의 finish_reason 을 확인해 주세요.)"

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


def get_llm(
    provider: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    base_url: str | None = None,
) -> LLM:
    """설정된 Provider에 맞는 LLM 인스턴스 반환
    
    Args:
        provider: 'gemini' 또는 'ollama'. None이면 config 설정을 따름.
        api_key: Gemini API Key (Optional Override)
        model_name: Model Name (Optional Override)
        base_url: Ollama Base URL (Optional Override)
    """
    config = get_config()
    target_provider = provider or config.generation.provider
    
    if target_provider == "ollama":
        return OllamaLLM(base_url=base_url, model=model_name)
    else:
        return GeminiLLM(api_key=api_key, model_name=model_name)
