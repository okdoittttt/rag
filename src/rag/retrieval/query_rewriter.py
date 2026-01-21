"""Query Rewriter 모듈

LLM을 사용하여 사용자 질문을 검색에 최적화된 여러 변형으로 재작성합니다.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from rag.generation.llm import LLM, get_llm
from rag.logger import get_logger


logger = get_logger(__name__)


REWRITE_PROMPT = """당신은 검색 쿼리 최적화 및 다국어 번역 전문가입니다.
사용자의 질문을 검색에 최적화된 3~5개의 변형으로 재작성하세요.

**핵심 규칙:**
1. **[필수] 교차 언어 검색 지원**: 
   - 사용자가 **한국어**로 질문했지만 예상되는 답변 문서가 **영어**일 가능성이 높다면(예: 기술 용어, 연구 논문, 원명이 영어인 개념 등), **반드시 영어로 번역된 쿼리를 포함**하세요.
   - 예: "어텐션 메커니즘" -> "Attention mechanism", "Self-attention mechanism"

2. 검색 최적화:
   - 동의어, 유의어를 포함하여 다양한 관점에서 질문을 생성하세요.
   - 약어는 풀어서 작성하고, 모호한 표현은 구체화하세요.

**원본 질문:** {query}

**JSON 배열 형식으로만 응답하세요:**
["영어 번역 쿼리 (필요시)", "구체화된 질문", "다양한 표현의 질문"]
"""


class QueryRewriter:
    """Multi-Query Rewriter
    
    LLM을 사용하여 원본 질문을 여러 검색 쿼리로 확장합니다.
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """QueryRewriter 초기화
        
        Args:
            llm: LLM 인스턴스 (None이면 config에 따라 자동 생성)
        """
        self.llm = llm or get_llm()
    
    def rewrite(self, query: str, num_queries: int = 3) -> list[str]:
        """질문을 여러 변형으로 재작성
        
        Args:
            query: 원본 질문
            num_queries: 생성할 변형 수 (기본: 3)
            
        Returns:
            재작성된 질문 리스트 (원본 포함)
        """
        try:
            prompt = REWRITE_PROMPT.format(query=query)
            response = self.llm.generate(prompt)
            
            # JSON 파싱
            rewritten = self._parse_response(response)
            
            if rewritten:
                logger.debug(
                    "query_rewritten",
                    original=query,
                    rewritten_count=len(rewritten),
                )
                # 원본 질문 + 재작성된 질문들
                return [query] + rewritten[:num_queries]
            
        except Exception as e:
            logger.warning("query_rewrite_failed", error=str(e))
        
        # 실패 시 원본만 반환
        return [query]
    
    def _parse_response(self, response: str) -> list[str]:
        """LLM 응답에서 JSON 배열 파싱
        
        Args:
            response: LLM 응답 문자열
            
        Returns:
            파싱된 질문 리스트
        """
        # JSON 배열 추출 시도
        try:
            # 코드블록 내 JSON 처리
            if "```" in response:
                match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
            
            # 직접 JSON 파싱
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
                
        except json.JSONDecodeError:
            pass
        
        return []
