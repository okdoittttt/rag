"""BM25 검색 모듈

rank_bm25를 사용한 키워드 검색 기능을 제공합니다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking.chunk import Chunk
from rag.logger import get_logger
from rag.retrieval.tokenizer import tokenize_content, tokenize_query


logger = get_logger(__name__)


class BM25Searcher:
    """BM25 검색기"""
    
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Chunk] = []
        
    def index(self, chunks: List[Chunk]) -> None:
        """청크 인덱싱
        
        Args:
            chunks: 인덱싱할 청크 리스트
        """
        self.chunks = chunks
        
        # 코퍼스 토크나이징
        tokenized_corpus = [
            tokenize_content(chunk.content)
            for chunk in chunks
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("bm25_indexed", count=len(chunks))
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """키워드 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 개수
            
        Returns:
            (청크, 점수) 튜플 리스트. 점수는 정규화되지 않음.
        """
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = tokenize_query(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 점수 내림차순 정렬
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            score = scores[idx]
            if score > 0:  # 관련성 있는 것만
                results.append((self.chunks[idx], float(score)))
                
        return results
    
    def get_full_scores(self, query: str) -> np.ndarray:
        """전체 문서에 대한 점수 반환 (Hybrid 검색용)"""
        if not self.bm25:
            return np.array([])
            
        tokenized_query = tokenize_query(query)
        return np.array(self.bm25.get_scores(tokenized_query))

    def save(self, path: Path) -> None:
        """인덱스 저장"""
        if not self.bm25:
            return
            
        path.mkdir(parents=True, exist_ok=True)
        # BM25 객체와 청크 저장
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunks": self.chunks
            }, f)
            
    def load(self, path: Path) -> None:
        """인덱스 로드"""
        try:
            with open(path / "bm25.pkl", "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
        except FileNotFoundError:
            logger.warning("bm25_index_not_found", path=str(path))
