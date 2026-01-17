"""벡터 저장소 모듈

FAISS를 사용하여 임베딩 벡터를 저장하고 검색합니다.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from rag.chunking.chunk import Chunk
from rag.logger import get_logger


logger = get_logger(__name__)


class VectorStore:
    """FAISS 기반 벡터 저장소"""
    
    def __init__(self, dimension: int):
        """
        Args:
            dimension: 벡터 차원
        """
        self.dimension = dimension
        # 코사인 유사도(정규화된 벡터의 내적)를 위한 IndexFlatIP 사용
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []
        
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """청크와 임베딩 추가
        
        Args:
            chunks: 청크 리스트
            embeddings: 임베딩 벡터 (numpy array, shape=[N, dim])
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) and embeddings count ({len(embeddings)}) must match"
            )
        
        if len(chunks) == 0:
            return
            
        # 차원 확인
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})"
            )
            
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        
        logger.info(
            "chunks_added_to_index",
            count=len(chunks),
            total_chunks=len(self.chunks),
        )
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """유사한 청크 검색
        
        Args:
            query_embedding: 쿼리 벡터 (numpy array, shape=[1, dim])
            top_k: 반환할 청크 수
            
        Returns:
            (청크, 점수) 튜플 리스트
        """
        if len(self.chunks) == 0:
            return []
            
        # 검색
        scores, indices = self.index.search(query_embedding, top_k)
        
        results: list[tuple[Chunk, float]] = []
        
        # 1차원이 아닌 2차원 배열로 반환됨 (쿼리가 하나여도)
        query_scores = scores[0]
        query_indices = indices[0]
        
        for score, idx in zip(query_scores, query_indices):
            if idx == -1:  # 결과 부족시 -1 반환됨
                continue
            
            chunk = self.chunks[idx]
            results.append((chunk, float(score)))
            
        return results
    
    def save(self, path: str | Path) -> None:
        """인덱스와 메타데이터 저장
        
        Args:
            path: 저장할 디렉토리 경로 (파일이 아님)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # FAISS 인덱스 저장
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # 청크 데이터(메타데이터) 저장
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
            
        # 설정 저장 (차원 정보)
        meta_path = path / "meta.json"
        import json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"dimension": self.dimension}, f)
            
        logger.info("index_saved", path=str(path), total_chunks=len(self.chunks))
        
    def load(self, path: str | Path) -> None:
        """인덱스와 메타데이터 로드
        
        Args:
            path: 저장된 디렉토리 경로
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index path not found: {path}")
            
        # FAISS 인덱스 로드
        index_path = path / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        
        # 청크 데이터 로드
        chunks_path = path / "chunks.pkl"
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
            
        # 차원 정보 확인 (일치하지 않으면 경고)
        meta_path = path / "meta.json"
        if meta_path.exists():
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if meta.get("dimension") != self.dimension:
                    logger.warning(
                        "index_dimension_mismatch",
                        expected=self.dimension,
                        loaded=meta.get("dimension"),
                    )
                    # 필요한 경우 self.dimension = meta["dimension"] 업데이트 고려
            
        logger.info("index_loaded", path=str(path), total_chunks=len(self.chunks))

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
