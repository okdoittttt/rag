"""Query Rewriter 모듈

LLM을 사용하여 사용자 질문을 검색에 최적화된 여러 변형으로 재작성합니다.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from rag.generation.llm import GeminiLLM
from rag.logger import get_logger


logger = get_logger(__name__)


REWRITE_PROMPT = """당신은 검색 쿼리 최적화 전문가입니다.
사용자의 질문을 검색에 최적화된 3가지 변형으로 재작성하세요.

**규칙:**
1. 원본 질문의 의도를 유지하면서 다른 관점/표현으로 변형
2. 약어가 있으면 풀어서 작성
3. 모호한 표현은 구체화
4. 각 변형은 독립적이고 완전한 질문이어야 함

**원본 질문:** {query}

**JSON 형식으로만 응답하세요:**
["변형1", "변형2", "변형3"]
"""


class QueryRewriter:
    """Multi-Query Rewriter
    
    LLM을 사용하여 원본 질문을 여러 검색 쿼리로 확장합니다.
    """
    
    def __init__(self, llm: Optional[GeminiLLM] = None):
        """QueryRewriter 초기화
        
        Args:
            llm: LLM 인스턴스 (None이면 새로 생성)
        """
        self.llm = llm or GeminiLLM()
    
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
