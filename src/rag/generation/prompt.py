"""프롬프트 모듈

RAG 답변 생성을 위한 시스템 프롬프트 및 컨텍스트 결합 기능을 제공합니다.
"""

from __future__ import annotations

from typing import List

from rag.chunking.chunk import Chunk


# 시스템 프롬프트: 페르소나 및 제약조건 정의
SYSTEM_PROMPT = """당신은 정확하고 신뢰할 수 있는 AI 어시스턴트입니다.
사용자의 질문에 대해 아래 제공된 [Context]만을 기반으로 답변하세요.

**지침:**
1. [Context]에 없는 내용은 절대 지어내거나 추측하지 마세요.
2. 정보가 부족하면 "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 정중하게 답하세요.
3. 답변은 한국어로 작성하며, 간결하고 명확하게 서술하세요.
4. 답변의 각 주장이나 사실 끝에는 반드시 관련 청크의 번호를 인용하세요. (예: [1], [2])
5. 인용은 답변의 신뢰성을 높이는 데 필수적입니다.
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
