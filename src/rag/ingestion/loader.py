"""문서 로더
    
파일 확장자에 따라 적절한 파서를 선택하여 텍스트를 추출합니다.
"""

from __future__ import annotations

from pathlib import Path

from rag.ingestion.document import Document
from rag.ingestion.parsers import _PARSERS, get_parser as _registry_get_parser
from rag.ingestion.parsers.base import DocumentParser
from rag.logger import get_logger


logger = get_logger(__name__)


class DocumentLoader:
    """문서 로더"""

    def __init__(self):
        # 파서 레지스트리(단일 소스)를 그대로 재사용한다.
        self.parsers: list[DocumentParser] = _PARSERS

    def get_parser(self, path: Path) -> DocumentParser | None:
        """파일에 적합한 파서 반환"""
        return _registry_get_parser(path)
        
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


def load_file(path: Path | str) -> Document:
    """파일을 로드하여 ``Document`` 객체로 반환한다.

    내부적으로 ``DocumentLoader``로 텍스트를 추출하고 ``Document.from_file``로
    메타데이터를 채워 반환한다.

    Args:
        path: 로드할 파일 경로.

    Returns:
        본문과 메타데이터를 포함한 ``Document`` 인스턴스.

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때 발생.
        ValueError: 지원하지 않는 파일 형식일 때 발생.
    """
    file_path = Path(path)
    loader = DocumentLoader()
    content = loader.load(file_path)
    return Document.from_file(file_path, content)


def load_documents(path: Path | str, recursive: bool = True) -> list[Document]:
    """디렉터리 또는 파일에서 문서들을 로드한다.

    경로가 단일 파일이면 해당 파일 하나를 ``Document``로 감싸 반환하고,
    디렉터리이면 지원 가능한 모든 파일을 순회하며 ``Document`` 리스트를
    생성한다. ``recursive``가 ``True``이면 하위 디렉터리까지 재귀 탐색한다.
    숨김 파일/폴더(``.``로 시작)는 항상 제외한다.

    Args:
        path: 파일 또는 디렉터리 경로.
        recursive: 디렉터리일 때 하위 디렉터리까지 재귀 탐색할지 여부.
            기본값은 ``True``.

    Returns:
        로드된 ``Document`` 객체 리스트. 경로가 존재하지 않거나 지원되는
        파일이 없으면 빈 리스트를 반환한다.
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

    # 2. 디렉터리인 경우 (재귀 여부에 따라 탐색 범위 결정)
    file_iter = root_path.rglob("*") if recursive else root_path.glob("*")
    for file_path in file_iter:
        # 숨김 파일/폴더 제외
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
