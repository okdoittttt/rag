"""PDF 파서

pypdf를 사용하여 PDF 파일에서 텍스트를 추출합니다.
PDF 특유의 레이아웃 문제를 해결하기 위한 후처리를 포함합니다.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from rag.ingestion.parsers.base import DocumentParser
from rag.logger import get_logger


logger = get_logger(__name__)


def normalize_pdf_text(text: str) -> str:
    """PDF 텍스트 정규화
    
    PDF 추출 시 발생하는 일반적인 문제들을 해결합니다:
    - 하이픈으로 연결된 단어 병합
    - 불필요한 줄바꿈 제거
    - 문단 구분 유지
    
    Args:
        text: PDF에서 추출된 원본 텍스트
        
    Returns:
        정규화된 텍스트
    """
    if not text:
        return ""
    
    # 1. 하이픈으로 끊긴 단어 병합 (예: "atten-\ntion" → "attention")
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # 2. 연속된 2개 이상 줄바꿈 → 문단 구분자로 임시 표시
    text = re.sub(r'\n\s*\n+', '\n\n<<PARA>>\n\n', text)
    
    # 3. 단일 줄바꿈 → 공백 (PDF에서 줄바꿈은 레이아웃용)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 4. 문단 구분자 복원
    text = text.replace('<<PARA>>', '\n\n')
    
    # 5. 연속 공백 정리
    text = re.sub(r' +', ' ', text)
    
    # 6. 불필요한 공백 줄 정리
    text = re.sub(r'\n +', '\n', text)
    text = re.sub(r' +\n', '\n', text)
    
    # 7. 3개 이상 연속 줄바꿈 → 2개로
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


class PDFParser(DocumentParser):
    """PDF 파일 파서"""
    
    extensions = [".pdf"]
    
    def parse(self, path: Path) -> str:
        """PDF 파일에서 텍스트 추출
        
        Args:
            path: PDF 파일 경로
            
        Returns:
            추출 및 정규화된 텍스트
        """
        try:
            reader = PdfReader(path)
            text_parts: list[str] = []
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # 페이지별로 정규화 적용
                    normalized = normalize_pdf_text(page_text)
                    if normalized:
                        text_parts.append(normalized)
            
            # 페이지 간 구분 (문단 구분)
            full_text = "\n\n".join(text_parts)
            
            logger.debug(
                "pdf_parsed",
                path=str(path),
                pages=len(reader.pages),
                text_length=len(full_text),
            )
            
            return full_text
            
        except Exception as e:
            logger.warning("pdf_parse_failed", path=str(path), error=str(e))
            return ""
