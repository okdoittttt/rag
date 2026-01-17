"""파일 로더 모듈

파일 또는 디렉토리에서 문서를 로딩합니다.
"""

from __future__ import annotations

from pathlib import Path

from rag.config import get_config
from rag.logger import get_logger, setup_logging
from rag.ingestion.document import Document
from rag.ingestion.normalizer import normalize_text
from rag.ingestion.language import detect_language


logger = get_logger(__name__)


def _is_supported_file(path: Path, extensions: list[str]) -> bool:
    """지원되는 파일 확장자인지 확인"""
    return path.is_file() and path.suffix.lower() in extensions


def _read_file_content(path: Path, encoding: str = "utf-8") -> str | None:
    """파일 내용 읽기
    
    Args:
        path: 파일 경로
        encoding: 인코딩 (기본: utf-8)
        
    Returns:
        파일 내용 또는 None (읽기 실패 시)
    """
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        # UTF-8 실패 시 latin-1 시도
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            logger.warning("file_read_fallback_failed", path=str(path), error=str(e))
            return None
    except Exception as e:
        logger.warning("file_read_failed", path=str(path), error=str(e))
        return None


def load_file(path: Path, normalize: bool = True) -> Document | None:
    """단일 파일 로딩
    
    Args:
        path: 파일 경로
        normalize: 정규화 여부 (기본: True)
        
    Returns:
        Document 또는 None (로딩 실패 시)
    """
    config = get_config()
    
    content = _read_file_content(path, config.ingestion.encoding)
    if content is None:
        return None
    
    # 정규화
    if normalize:
        content = normalize_text(content)
    
    # Document 생성
    doc = Document.from_file(path, content)
    
    # 언어 감지
    doc.metadata["language"] = detect_language(content)
    
    logger.debug(
        "file_loaded",
        filename=doc.metadata["filename"],
        length=len(doc),
        language=doc.metadata["language"],
    )
    
    return doc


def load_documents(
    path: str | Path,
    recursive: bool = True,
    normalize: bool = True,
) -> list[Document]:
    """파일 또는 디렉토리에서 문서 로딩
    
    Args:
        path: 파일 또는 디렉토리 경로
        recursive: 디렉토리 재귀 탐색 여부 (기본: True)
        normalize: 정규화 여부 (기본: True)
        
    Returns:
        Document 리스트
    """
    path = Path(path)
    config = get_config()
    extensions = config.ingestion.supported_extensions
    
    documents: list[Document] = []
    
    if path.is_file():
        # 단일 파일
        if _is_supported_file(path, extensions):
            doc = load_file(path, normalize)
            if doc:
                documents.append(doc)
    elif path.is_dir():
        # 디렉토리
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if _is_supported_file(file_path, extensions):
                doc = load_file(file_path, normalize)
                if doc:
                    documents.append(doc)
    else:
        logger.warning("path_not_found", path=str(path))
    
    logger.info(
        "documents_loaded",
        path=str(path),
        count=len(documents),
        recursive=recursive,
    )
    
    return documents
