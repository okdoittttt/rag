"""Qdrant 기반 벡터 저장소

Qdrant 서버에 연결하여 벡터를 저장하고 검색합니다.
영속성과 확장성을 제공합니다.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from rag.chunking.chunk import Chunk
from rag.embedding.base import VectorStoreBase
from rag.logger import get_logger


logger = get_logger(__name__)


class QdrantStore(VectorStoreBase):
    """Qdrant 기반 벡터 저장소"""
    
    def __init__(
        self,
        dimension: int,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "terminal-rag",
        **kwargs: Any,
    ):
        """
        Args:
            dimension: 벡터 차원
            host: Qdrant 서버 호스트
            port: Qdrant 서버 포트
            collection_name: 컬렉션 이름
        """
        self.dimension = dimension
        self.host = host
        self.port = port
        self.collection_name = collection_name
        
        # Qdrant 클라이언트 초기화 (타임아웃 증가)
        self.client = QdrantClient(host=host, port=port, timeout=60)
        
        # 컬렉션 생성 (없으면)
        self._ensure_collection()
        
        logger.info(
            "qdrant_store_initialized",
            host=host,
            port=port,
            collection=collection_name,
        )
    
    def _ensure_collection(self) -> None:
        """컬렉션이 없으면 생성"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info("collection_exists", collection=self.collection_name)
        except (UnexpectedResponse, Exception):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("collection_created", collection=self.collection_name)
    
    def add(self, chunks: list[Chunk], embeddings: np.ndarray, batch_size: int = 50) -> None:
        """청크와 임베딩 추가 (배치 업로드)
        
        Args:
            chunks: 청크 리스트
            embeddings: 임베딩 배열
            batch_size: 한 번에 업로드할 포인트 수 (기본: 50)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) and embeddings count ({len(embeddings)}) must match"
            )
        
        if len(chunks) == 0:
            return
        
        # 포인트 생성
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            payload = {
                "content": chunk.content,
                "source": chunk.metadata.get("source", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0),
                "start_char": chunk.metadata.get("start_char", 0),
                "end_char": chunk.metadata.get("end_char", 0),
                "header_path": chunk.metadata.get("header_path", ""),
                "user_id": chunk.metadata.get("user_id", ""),  # 사용자 ID 추가
                "metadata": chunk.metadata,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )
        
        # 배치 업로드 (타임아웃 방지)
        total_batches = (len(points) + batch_size - 1) // batch_size
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )
            batch_num = (i // batch_size) + 1
            logger.debug(
                "batch_uploaded",
                batch=batch_num,
                total_batches=total_batches,
                points_in_batch=len(batch),
            )
        
        logger.info(
            "chunks_added_to_qdrant",
            count=len(chunks),
            collection=self.collection_name,
            batches=total_batches,
        )
    
    def search(
        self, query_embedding: np.ndarray, top_k: int = 5, user_id: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """유사한 청크 검색
        
        Args:
            query_embedding: 쿼리 벡터
            top_k: 반환할 결과 수
            user_id: 사용자 ID (None이면 필터 없음)
        """
        # 쿼리 벡터 준비
        query_vector = query_embedding[0].tolist() if query_embedding.ndim == 2 else query_embedding.tolist()
        
        # 사용자 ID 필터 설정
        query_filter = None
        if user_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    )
                ]
            )
        
        # 검색 (query_points API 사용 - qdrant-client 1.10+)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        
        # 결과 변환
        output: list[tuple[Chunk, float]] = []
        for hit in results.points:
            payload = hit.payload or {}
            # 저장된 metadata 또는 payload에서 재구성
            metadata = payload.get("metadata", {})
            if not metadata:
                metadata = {
                    "source": payload.get("source", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "start_char": payload.get("start_char", 0),
                    "end_char": payload.get("end_char", 0),
                    "header_path": payload.get("header_path", ""),
                }
            chunk = Chunk(
                content=payload.get("content", ""),
                metadata=metadata,
            )
            output.append((chunk, hit.score))
        
        return output
    
    def save(self, path: str | Path) -> None:
        """Qdrant는 자동 영속화되므로 별도 저장 불필요"""
        logger.info("qdrant_auto_persisted", collection=self.collection_name)
    
    def load(self, path: str | Path) -> None:
        """Qdrant는 서버가 데이터를 관리하므로 별도 로드 불필요"""
        self._ensure_collection()
        logger.info("qdrant_collection_ready", collection=self.collection_name)
    
    @property
    def total_chunks(self) -> int:
        """저장된 총 청크 수"""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0
    
    def clear(self) -> None:
        """컬렉션 초기화 (삭제 후 재생성)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info("collection_deleted", collection=self.collection_name)
        except Exception:
            pass
        self._ensure_collection()
