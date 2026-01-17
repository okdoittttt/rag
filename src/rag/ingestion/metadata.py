"""인덱스 메타데이터 관리

인덱싱된 문서의 해시와 경로를 추적합니다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag.logger import get_logger


logger = get_logger(__name__)


class IndexMetadata:
    """인덱스 메타데이터 관리자
    
    인덱싱된 문서의 해시를 저장하여 중복 인덱싱을 방지합니다.
    """
    
    FILENAME = "index_meta.json"
    
    def __init__(self, index_path: Path):
        self.index_path = Path(index_path)
        self.meta_file = self.index_path / self.FILENAME
        self._data: dict[str, Any] = {
            "indexed_files": {},  # {file_path: {"hash": ..., "chunks_count": ...}}
            "version": "1.0",
        }
        self._load()
    
    def _load(self) -> None:
        """메타데이터 로드"""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.debug("index_metadata_loaded", count=len(self._data.get("indexed_files", {})))
            except Exception as e:
                logger.warning("index_metadata_load_failed", error=str(e))
    
    def save(self) -> None:
        """메타데이터 저장"""
        self.index_path.mkdir(parents=True, exist_ok=True)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        logger.debug("index_metadata_saved", count=len(self._data.get("indexed_files", {})))
    
    def clear(self) -> None:
        """메타데이터 초기화"""
        self._data = {"indexed_files": {}, "version": "1.0"}
        if self.meta_file.exists():
            self.meta_file.unlink()
        logger.debug("index_metadata_cleared")
    
    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """파일 내용의 해시 계산 (SHA256)"""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def is_indexed(self, file_path: Path) -> bool:
        """파일이 이미 인덱싱되었고 변경되지 않았는지 확인"""
        str_path = str(file_path.absolute())
        
        if str_path not in self._data["indexed_files"]:
            return False
        
        stored_hash = self._data["indexed_files"][str_path].get("hash")
        current_hash = self.compute_file_hash(file_path)
        
        if stored_hash != current_hash:
            logger.debug("file_changed", path=str_path)
            return False
        
        return True
    
    def mark_indexed(self, file_path: Path, chunks_count: int) -> None:
        """파일을 인덱싱됨으로 표시"""
        str_path = str(file_path.absolute())
        self._data["indexed_files"][str_path] = {
            "hash": self.compute_file_hash(file_path),
            "chunks_count": chunks_count,
        }
    
    def get_indexed_count(self) -> int:
        """인덱싱된 파일 수 반환"""
        return len(self._data.get("indexed_files", {}))
    
    def remove_indexed(self, file_path: Path) -> None:
        """파일을 인덱스 목록에서 제거"""
        str_path = str(file_path.absolute())
        if str_path in self._data["indexed_files"]:
            del self._data["indexed_files"][str_path]
