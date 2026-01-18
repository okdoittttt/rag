"""텍스트/마크다운 파서

.txt, .md 파일을 파싱합니다.
"""

from __future__ import annotations

from pathlib import Path

from rag.ingestion.parsers.base import DocumentParser
from rag.logger import get_logger


logger = get_logger(__name__)


class TextParser(DocumentParser):
    """텍스트 파일 파서"""
    
    extensions = [".txt", ".md"]
    
    def parse(self, path: Path, encoding: str = "utf-8") -> str:
        """텍스트 파일 파싱"""
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # UTF-8 실패 시 latin-1 시도
            try:
                return path.read_text(encoding="latin-1")
            except Exception as e:
                logger.warning("text_parse_failed", path=str(path), error=str(e))
                return ""
        except Exception as e:
            logger.warning("text_parse_failed", path=str(path), error=str(e))
            return ""
