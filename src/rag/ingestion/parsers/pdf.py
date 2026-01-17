"""PDF 파서

pypdf를 사용하여 PDF 파일에서 텍스트를 추출합니다.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag.ingestion.parsers.base import DocumentParser
from rag.logger import get_logger


logger = get_logger(__name__)


class PDFParser(DocumentParser):
    """PDF 파일 파서"""
    
    extensions = [".pdf"]
    
    def parse(self, path: Path) -> str:
        """PDF 파일에서 텍스트 추출
        
        Args:
            path: PDF 파일 경로
            
        Returns:
            추출된 텍스트 (페이지별 개행 구분)
        """
        try:
            reader = PdfReader(path)
            text_parts: list[str] = []
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    
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
