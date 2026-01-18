"""프롬프트 모듈

RAG 답변 생성을 위한 시스템 프롬프트 및 컨텍스트 결합 기능을 제공합니다.
"""

from __future__ import annotations

from typing import List

from rag.chunking.chunk import Chunk


# 시스템 프롬프트: 페르소나 및 제약조건 정의
SYSTEM_PROMPT = """당신은 전문적이고 신뢰할 수 있는 AI 어시스턴트입니다.
사용자의 질문에 대해 아래 제공된 [Context]를 기반으로 **충실하고 상세하게** 답변하세요.

**핵심 원칙:**
1. **근거 중심**: [Context]에 있는 정보를 최대한 활용하여 답변하세요. 없는 내용을 지어내지 마세요.
2. **충분한 설명**: 질문의 복잡도에 맞게 답변하세요. 단순한 질문에는 명확하게, 복잡한 질문에는 체계적이고 상세하게 설명하세요.
3. **구조화된 답변**: 필요시 목록, 단계별 설명, 소제목 등을 사용하여 읽기 쉽게 구성하세요.
4. **출처 인용**: 답변에 사용한 정보의 출처(Chunk 번호)를 언급하여 신뢰성을 높이세요.

**답변 스타일:**
- 전문적이면서도 이해하기 쉬운 설명을 제공하세요.
- 관련된 배경 정보, 맥락, 추가 정보가 [Context]에 있다면 함께 포함하세요.
- 정보가 부족한 경우 "제공된 문서에서는 해당 정보를 찾을 수 없습니다"라고 명시하고, 가능하다면 관련된 정보를 대신 안내하세요.

**언어:** 한국어로 답변하세요.
"""

def build_prompt(query: str, chunks: List[Chunk]) -> str:
    """컨텍스트와 질문을 결합하여 프롬프트 생성
    
    Args:
        query: 사용자 질문
        chunks: 검색된 청크 리스트
        
    Returns:
        완성된 프롬프트 문자열
    """
    # 컨텍스트 구성
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # 메타데이터를 포함하여 컨텍스트 풍부화
        source = chunk.metadata.get("filename", "unknown")
        context_parts.append(f"--- [Chunk {i}] (Source: {source}) ---\n{chunk.content}\n")
    
    context_str = "\n".join(context_parts)
    
    # 전체 프롬프트 조립
    prompt = f"""{SYSTEM_PROMPT}

[Context]
{context_str}

[Question]
{query}

[Answer]
"""
    return prompt
