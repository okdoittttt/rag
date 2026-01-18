"""벡터 저장소 기본 프로토콜

VectorStore 인터페이스를 정의합니다.
FAISS, Qdrant 등 다양한 백엔드를 지원하기 위한 추상화 레이어입니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag.chunking.chunk import Chunk


class VectorStoreBase(ABC):
    """벡터 저장소 추상 기본 클래스"""
    
    @abstractmethod
    def add(self, chunks: list["Chunk"], embeddings: np.ndarray) -> None:
        """청크와 임베딩을 저장소에 추가
        
        Args:
            chunks: 청크 리스트
            embeddings: 임베딩 벡터 (numpy array, shape=[N, dim])
        """
        ...
    
    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple["Chunk", float]]:
        """유사한 청크 검색
        
        Args:
            query_embedding: 쿼리 벡터 (numpy array, shape=[1, dim])
            top_k: 반환할 청크 수
            
        Returns:
            (청크, 점수) 튜플 리스트
        """
        ...
    
    @abstractmethod
    def save(self, path: str) -> None:
        """인덱스와 메타데이터 저장"""
        ...
    
    @abstractmethod
    def load(self, path: str) -> None:
        """인덱스와 메타데이터 로드"""
        ...
    
    @property
    @abstractmethod
    def total_chunks(self) -> int:
        """저장된 총 청크 수"""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """저장소 초기화"""
        ...
