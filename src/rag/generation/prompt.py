"""프롬프트 모듈

RAG 답변 생성을 위한 시스템 프롬프트 및 컨텍스트 결합 기능을 제공합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal

from rag.chunking.chunk import Chunk


PromptMode = Literal["default", "summary"]


# 시스템 프롬프트: 페르소나 및 제약조건 정의
# 기본 시스템 프롬프트 (파일이 없을 경우 사용)
DEFAULT_SYSTEM_PROMPT = """당신은 전문적이고 신뢰할 수 있는 AI 어시스턴트입니다.
사용자의 질문에 대해 아래 제공된 [Context]를 기반으로 **충실하고 상세하게** 답변하세요.

**핵심 원칙:**
1. **근거 중심**: [Context]에 있는 정보를 최대한 활용하여 답변하세요. 없는 내용을 지어내지 마세요.
2. **충분한 설명**: 질문의 복잡도에 맞게 답변하세요. 단순한 질문에는 명확하게, 복잡한 질문에는 체계적이고 상세하게 설명하세요.
3. **구조화된 답변**: 필요시 목록, 단계별 설명, 소제목 등을 사용하여 읽기 쉽게 구성하세요.

**답변 스타일:**
- 전문적이면서도 이해하기 쉬운 설명을 제공하세요.
- 관련된 배경 정보, 맥락, 추가 정보가 [Context]에 있다면 함께 포함하세요.
- 정보가 부족한 경우 "제공된 문서에서는 해당 정보를 찾을 수 없습니다"라고 명시하고, 가능하다면 관련된 정보를 대신 안내하세요.

**언어:** 한국어로 답변하세요.

**주의사항:** 답변 본문에 [Chunk 1], (Source: doc.txt) 등과 같은 출처 표기를 절대 포함하지 마세요.
"""

# 요약 모드 전용 추가 지침. 기본 시스템 프롬프트 뒤에 prepend 되어 LLM 에게
# 전체 컨텍스트(= 문서 전체 청크)를 빠짐없이 종합하라는 신호를 준다.
SUMMARY_MODE_INSTRUCTION = """**요약 모드 안내:** 아래 [Context]는 단일 문서의 청크들이 chunk_index 오름차순으로 **전체** 포함된 상태입니다.
- 일부 청크의 헤더/날짜/메타데이터만 보고 답하지 말고, **문서 전반을 종합**하여 한국어로 요약하세요.
- 도입–본론–결론 또는 주제별 소제목 구조를 사용해 빠짐없이, 그러나 중복 없이 정리하세요.
- 청크 사이의 자연스러운 연결을 복원하고, 모순 없이 통합된 한 편의 요약을 작성하세요.
"""

def get_system_prompt() -> str:
    """시스템 프롬프트를 로드합니다.
    
    설정 파일(data/system_prompt.txt)이 있으면 해당 내용을,
    없으면 기본 상수(DEFAULT_SYSTEM_PROMPT)를 반환합니다.
    """
    try:
        # Docker에서는 /app/data, 로컬에서는 프로젝트 루트의 data/ 사용
        docker_path = Path("/app/data")
        if docker_path.exists():
            data_path = docker_path
        else:
            # 로컬 개발: 프로젝트 루트 기준 data/ 폴더
            data_path = Path(__file__).parent.parent.parent.parent / "data"
        
        prompt_path = data_path / "system_prompt.txt"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Failed to load system prompt file: {e}")
        
    return DEFAULT_SYSTEM_PROMPT

def build_prompt(
    query: str,
    chunks: List[Chunk],
    mode: PromptMode = "default",
) -> str:
    """컨텍스트와 질문을 결합하여 프롬프트를 생성한다.

    Args:
        query: 사용자 질문.
        chunks: 검색된 청크 리스트(요약 모드에서는 문서 전체 청크).
        mode: 프롬프트 모드. ``"summary"`` 면 요약 전용 시스템 지침이
            기본 시스템 프롬프트 앞에 prepend 된다.

    Returns:
        완성된 프롬프트 문자열.
    """
    # 컨텍스트 구성
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        # 메타데이터를 포함하여 컨텍스트 풍부화
        source = chunk.metadata.get("filename", "unknown")
        context_parts.append(f"--- [Chunk {i}] (Source: {source}) ---\n{chunk.content}\n")

    context_str = "\n".join(context_parts)

    # 동적 시스템 프롬프트 로드
    system_prompt = get_system_prompt()
    if mode == "summary":
        system_prompt = f"{SUMMARY_MODE_INSTRUCTION}\n{system_prompt}"

    # 전체 프롬프트 조립
    prompt = f"""{system_prompt}

[Context]
{context_str}

[Question]
{query}

[Answer]
"""
    return prompt
