"""Embedding 모듈

임베딩 생성 및 벡터 저장소 기능을 제공합니다.
"""

from typing import TYPE_CHECKING

from rag.embedding.base import VectorStoreBase
from rag.embedding.embedder import Embedder

if TYPE_CHECKING:
    from rag.config import Config


def get_vector_store(config: "Config") -> VectorStoreBase:
    """설정에 따라 적절한 벡터 저장소 반환

    Args:
        config: 전체 설정 객체

    Returns:
        VectorStoreBase 구현체 (QdrantStore)
    """
    store_type = config.embedding.store_type
    dimension = config.embedding.dimension

    if store_type == "qdrant":
        from rag.embedding.qdrant_store import QdrantStore
        qdrant_config = config.embedding.qdrant
        return QdrantStore(
            dimension=dimension,
            host=qdrant_config.host,
            port=qdrant_config.port,
            collection_name=qdrant_config.collection,
        )
    else:
        raise ValueError(f"지원하지 않는 store_type: {store_type}. 'qdrant'를 사용하세요.")


# 하위 호환성을 위한 별칭
VectorStore = VectorStoreBase


__all__ = [
    "Embedder",
    "VectorStore",
    "VectorStoreBase",
    "get_vector_store",
]
