"""하이브리드 검색 모듈

BM25(키워드)와 Vector(의미) 검색 결과를 결합합니다.
Weighted Sum 방식을 사용하여 점수를 합산합합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from rag.chunking.chunk import Chunk
from rag.config import get_config
from rag.embedding import Embedder, VectorStore
from rag.logger import get_logger
from rag.retrieval.bm25 import BM25Searcher


logger = get_logger(__name__)


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """점수 Min-Max 정규화"""
    if len(scores) == 0:
        return scores
    
    min_val = np.min(scores)
    max_val = np.max(scores)
    
    if max_val == min_val:
        return np.zeros_like(scores)
        
    return (scores - min_val) / (max_val - min_val)


class HybridSearcher:
    """하이브리드 검색기"""
    
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = BM25Searcher()
        
    def index(self, chunks: List[Chunk], embeddings: np.ndarray | None = None) -> None:
        """인덱스 구축
        
        Args:
            chunks: 청크 리스트
            embeddings: 임베딩 벡터 (이미 생성된 경우)
        """
        # BM25 인덱싱
        self.bm25.index(chunks)
        
        # 벡터 인덱싱 (임베딩이 없으면 생성)
        if embeddings is None:
            contents = [c.content for c in chunks]
            embeddings = self.embedder.embed(contents)
            
        self.vector_store.add(chunks, embeddings)
        
    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,  # 0: BM25 only, 1: Vector only
    ) -> List[Tuple[Chunk, float]]:
        """하이브리드 검색
        
        Score = alpha * Vector_Score + (1 - alpha) * BM25_Score
        (각 점수는 정규화됨)
        
        Args:
            query: 검색 쿼리
            top_k: 반환 개수
            alpha: 벡터 검색 가중치 (0.0 ~ 1.0)
            
        Returns:
            (청크, 최종 점수) 튜플 리스트
        """
        # 0. BM25 점수 계산
        bm25_scores = self.bm25.get_full_scores(query)
        if len(bm25_scores) == 0:
            return []
            
        # 1. 벡터 검색 및 점수 계산
        # FAISS는 top_k만 반환하므로, 전체 점수를 얻기 위해선
        # 사실 모든 벡터와의 유사도를 계산해야 정확한 하이브리드가 가능함.
        # 하지만 성능상, 여기서는 편의상 검색된 청크들만 대상으로 재정렬하거나,
        # VectorStore가 전체 점수를 반환하도록 기능을 확장해야 함.
        #
        # 현실적인 대안 (RRF 또는 상위 k개 풀링):
        # 여기서는 VectorStore에서 top_k * 2 개를 가져오고, 
        # 그 청크들에 대해서만 BM25 점수를 합산하는 방식(Reranking)을 사용.
        
        # 벡터 쿼리
        query_vec = self.embedder.embed_query(query)
        
        # 1차 검색 (후보군 추출): 벡터 기준 top_k * 3
        candidates = self.vector_store.search(query_vec, top_k=top_k * 3)
        
        if not candidates:
            # 벡터 실패 시 BM25만 반환
            return self.bm25.search(query, top_k)
            
        # 2. 점수 결합
        hybrid_results = []
        
        # 벡터 점수 정규화용
        vec_scores = np.array([score for _, score in candidates])
        norm_vec_scores = min_max_normalize(vec_scores)
        
        # 해당 후보들에 대한 BM25 점수 가져오기
        # BM25Searcher의 chunks와 VectorStore의 chunks 인덱스가 동일하다고 가정
        # (index 메서드에서 함께 추가했으므로)
        
        for i, (chunk, _) in enumerate(candidates):
            # 원본 인덱스 찾기 (VectorStore가 인덱스를 반환하지 않아 chunk 객체로 매핑 필요)
            # 여기서는 편의상 chunk.metadata['chunk_index'] 활용
            chunk_idx = chunk.metadata.get("chunk_index")
            current_bm25_score = 0.0
            
            if chunk_idx is not None and chunk_idx < len(bm25_scores):
                current_bm25_score = bm25_scores[chunk_idx]
                
            hybrid_results.append({
                "chunk": chunk,
                "vec_score": norm_vec_scores[i],
                "bm25_score": current_bm25_score
            })
            
        # BM25 점수 정규화
        cand_bm25_scores = np.array([r["bm25_score"] for r in hybrid_results])
        norm_bm25_scores = min_max_normalize(cand_bm25_scores)
        
        # 최종 점수 계산 및 정렬
        final_results = []
        for i, res in enumerate(hybrid_results):
            final_score = (alpha * res["vec_score"]) + ((1 - alpha) * norm_bm25_scores[i])
            final_results.append((res["chunk"], final_score))
            
        # 최종 정렬
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results[:top_k]
    
    def save(self, path: Path) -> None:
        """인덱스 저장"""
        self.vector_store.save(path)
        self.bm25.save(path)
        
    def load(self, path: Path) -> None:
        """인덱스 로드"""
        self.vector_store.load(path)
        self.bm25.load(path)
