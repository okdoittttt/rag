"""Reranker 모듈

Cross-Encoder를 사용하여 검색 결과를 재정렬합니다.
1차 검색(Bi-Encoder) 결과를 받아 2차 정밀 점수를 계산합니다.
"""

from __future__ import annotations

from typing import Optional

from sentence_transformers import CrossEncoder

from rag.chunking.chunk import Chunk
from rag.logger import get_logger


logger = get_logger(__name__)


class Reranker:
    """Cross-Encoder 기반 Reranker
    
    질문과 문서를 함께 입력하여 관련성 점수를 계산합니다.
    Bi-Encoder보다 정확하지만 느리므로 후보군에만 적용합니다.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[str] = None,
    ):
        """Reranker 초기화
        
        Args:
            model_name: Cross-Encoder 모델 이름
            device: 실행 디바이스 (None이면 자동 선택)
        """
        self.model_name = model_name
        
        logger.info("loading_reranker", model=model_name)
        
        self.model = CrossEncoder(
            model_name,
            max_length=512,
            device=device,
        )
        
        logger.info("reranker_loaded", model=model_name)
    
    def rerank(
        self,
        query: str,
        chunks: list[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """검색 결과 재정렬
        
        Args:
            query: 사용자 질문
            chunks: (Chunk, score) 튜플 리스트 (1차 검색 결과)
            top_k: 반환할 상위 결과 수
            
        Returns:
            재정렬된 (Chunk, rerank_score) 리스트
        """
        if not chunks:
            return []
        
        # 질문-문서 쌍 생성
        pairs = [(query, chunk.content) for chunk, _ in chunks]
        
        # Cross-Encoder 점수 계산
        scores = self.model.predict(pairs)
        
        # 청크와 새 점수 매핑
        reranked = [
            (chunk, float(score))
            for (chunk, _), score in zip(chunks, scores)
        ]
        
        # 점수 기준 내림차순 정렬
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(
            "reranked",
            query_preview=query[:50],
            input_count=len(chunks),
            output_count=min(top_k, len(reranked)),
        )
        
        return reranked[:top_k]
