"""문서 로더
    
파일 확장자에 따라 적절한 파서를 선택하여 텍스트를 추출합니다.
"""

from __future__ import annotations

from pathlib import Path

from rag.ingestion.document import Document
from rag.ingestion.parsers.base import DocumentParser
from rag.ingestion.parsers.text import TextParser
from rag.ingestion.parsers.pdf import PDFParser
from rag.logger import get_logger


logger = get_logger(__name__)


class DocumentLoader:
    """문서 로더"""
    
    def __init__(self):
        self.parsers: list[DocumentParser] = [
            TextParser(),
            PDFParser(),
        ]
        
    def get_parser(self, path: Path) -> DocumentParser | None:
        """파일에 적합한 파서 반환"""
        for parser in self.parsers:
            if parser.can_parse(path):
                return parser
        return None
        
    def load(self, path: Path | str) -> str:
        """파일에서 텍스트 추출
        
        Args:
            path: 파일 경로
            
        Returns:
            추출된 텍스트
            
        Raises:
            ValueError: 지원하지 않는 파일 형식인 경우
            FileNotFoundError: 파일이 없는 경우
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
            
        parser = self.get_parser(file_path)
        if not parser:
            raise ValueError(f"No parser found for file: {path}")
            
        try:
            return parser.parse(file_path)
        except Exception as e:
            logger.error("load_failed", path=str(path), error=str(e))
            raise e


def load_file(path: Path | str) -> str:
    """헬퍼 함수: 파일 로드"""
    loader = DocumentLoader()
    return loader.load(path)


def load_documents(path: Path | str) -> list[Document]:
    """디렉터리 또는 파일에서 문서를 로드 (재귀)
    
    Args:
        path: 파일 또는 디렉터리 경로
        
    Returns:
        Document 객체 리스트
    """
    root_path = Path(path)
    loader = DocumentLoader()
    documents: list[Document] = []
    
    if not root_path.exists():
        logger.warning("path_not_found", path=str(path))
        return []
    
    # 1. 단일 파일인 경우
    if root_path.is_file():
        try:
            content = loader.load(root_path)
            doc = Document.from_file(root_path, content)
            documents.append(doc)
        except (ValueError, FileNotFoundError):
            pass  # 지원하지 않는 파일은 무시
        except Exception:
            pass
        return documents
    
    # 2. 디렉터리인 경우 (재귀 탐색)
    # 숨김 파일/폴더 제외
    for file_path in root_path.rglob("*"):
        if file_path.name.startswith(".") or not file_path.is_file():
            continue
            
        try:
            content = loader.load(file_path)
            doc = Document.from_file(file_path, content)
            documents.append(doc)
        except ValueError:
            continue  # 지원하지 않는 파일 스킵
        except Exception as e:
            logger.warning("load_doc_failed", path=str(file_path), error=str(e))
            continue
            
    logger.info("documents_loaded", count=len(documents), path=str(path))
    return documents
