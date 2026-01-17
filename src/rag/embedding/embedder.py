"""임베딩 생성 모듈

SentenceTransformers를 사용하여 텍스트를 벡터로 변환합니다.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import get_config
from rag.logger import get_logger


logger = get_logger(__name__)


class Embedder:
    """텍스트 임베딩 생성기"""
    
    def __init__(self, model_name: str | None = None):
        """
        Args:
            model_name: 사용할 모델명. None이면 설정 파일값 사용.
        """
        config = get_config()
        self.model_name = model_name or config.embedding.model
        
        logger.info("loading_embedding_model", model=self.model_name)
        self.model = SentenceTransformer(self.model_name)
        
        # CPU 사용 시 intra-op parallelism 제한 (선택사항)
        # torch.set_num_threads(4)
        
    def embed(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트를 벡터로 변환
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            임베딩 벡터 (numpy array, shape=[N, dim])
        """
        if not texts:
            return np.array([])
        
        config = get_config()
        batch_size = config.embedding.batch_size
        
        logger.debug("embedding_texts", count=len(texts), batch_size=batch_size)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 코사인 유사도를 위해 정규화
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """쿼리를 벡터로 변환
        
        Args:
            query: 검색 쿼리
            
        Returns:
            임베딩 벡터 (numpy array, shape=[1, dim])
        """
        return self.embed([query])
