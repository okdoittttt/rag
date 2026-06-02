"""FAISS 기반 벡터 저장소

로컬 파일 시스템에 인덱스를 저장하는 FAISS 구현체입니다.
임베딩 사본을 유지하여 source 단위 삭제 시 인덱스를 재구축합니다.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from rag.chunking.chunk import Chunk
from rag.embedding.base import VectorStoreBase
from rag.logger import get_logger


logger = get_logger(__name__)


INDEX_VERSION = 2


class FAISSStore(VectorStoreBase):
    """FAISS 기반 벡터 저장소"""

    def __init__(self, dimension: int):
        """
        Args:
            dimension: 벡터 차원
        """
        self.dimension = dimension
        # 코사인 유사도(정규화된 벡터의 내적)를 위한 IndexFlatIP
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []
        # 삭제 시 인덱스 재구축을 위해 임베딩 사본 유지
        self.embeddings: np.ndarray | None = None

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

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})"
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        # faiss SWIG stub은 add(n, x) 시그니처지만 런타임은 monkey-patch된 add(x).
        # stub과 런타임 시그니처가 달라 정상 호출에도 타입 체커가 오류를 내므로 ignore.
        self.index.add(embeddings)  # type: ignore[call-arg]
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = embeddings.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        logger.info(
            "chunks_added_to_index",
            count=len(chunks),
            total_chunks=len(self.chunks),
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        """유사한 청크 검색

        Args:
            query_embedding: 쿼리 벡터 (numpy array, shape=[1, dim])
            top_k: 반환할 청크 수
            user_id: 사용자 ID 필터. None이면 필터 없음.

        Returns:
            (청크, 점수) 튜플 리스트
        """
        if len(self.chunks) == 0:
            return []

        # user_id 필터를 위한 over-fetch (1/5 휴리스틱)
        if user_id is not None:
            fetch_k = min(top_k * 5, len(self.chunks))
        else:
            fetch_k = min(top_k, len(self.chunks))

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        # faiss SWIG stub은 search(n, x, k, distances, labels)지만 런타임은 search(x, k).
        scores, indices = self.index.search(query_embedding, fetch_k)  # type: ignore[call-arg]

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[int(idx)]
            if user_id is not None and chunk.metadata.get("user_id") != user_id:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        if user_id is not None and len(results) < top_k:
            logger.debug(
                "faiss_user_filter_short_result",
                requested=top_k,
                returned=len(results),
                fetched=fetch_k,
            )

        return results

    def delete_by_source(self, source: str, user_id: str | None = None) -> int:
        """source(+user_id) 매칭 청크 삭제 후 인덱스 재구축

        IndexFlatIP는 효율적 삭제를 지원하지 않으므로 매칭 청크 제외 후
        남은 임베딩으로 인덱스를 재구축한다.
        """
        if not self.chunks:
            return 0

        keep_indices: list[int] = []
        deleted = 0
        for i, chunk in enumerate(self.chunks):
            if chunk.metadata.get("source") != source:
                keep_indices.append(i)
                continue
            chunk_user = chunk.metadata.get("user_id")
            if user_id is None:
                matches = chunk_user in (None, "")
            else:
                matches = chunk_user == user_id
            if matches:
                deleted += 1
            else:
                keep_indices.append(i)

        if deleted == 0:
            return 0

        if not keep_indices:
            self.chunks = []
            self.embeddings = None
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            assert self.embeddings is not None
            self.chunks = [self.chunks[i] for i in keep_indices]
            self.embeddings = self.embeddings[keep_indices]
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(np.ascontiguousarray(self.embeddings, dtype=np.float32))  # type: ignore[call-arg]

        logger.info(
            "chunks_deleted_from_index",
            source=source,
            user_id=user_id,
            count=deleted,
            remaining=len(self.chunks),
        )
        return deleted

    def get_all_by_source(
        self,
        source: str,
        user_id: str | None = None,
        limit: int = 1000,
    ) -> list[Chunk]:
        """source(+user_id) 매칭 청크를 chunk_index 오름차순으로 반환.

        FAISS 는 메타데이터 기반 필터를 자체 제공하지 않으므로 ``self.chunks``
        선형 스캔으로 수집한다. 단일 문서 청크 수 가정(수백~수천)에서는
        충분히 빠르다.

        Args:
            source: chunk.metadata["source"].
            user_id: 사용자 ID. ``None`` 이면 ``user_id`` 가 ``None`` 또는 빈
                문자열인 청크만 반환한다(``delete_by_source`` 와 동일 정책).
            limit: 반환할 최대 청크 수.

        Returns:
            ``chunk_index`` 오름차순으로 정렬된 ``Chunk`` 리스트.
        """
        matched: list[Chunk] = []
        for chunk in self.chunks:
            if chunk.metadata.get("source") != source:
                continue
            chunk_user = chunk.metadata.get("user_id")
            if user_id is None:
                if chunk_user not in (None, ""):
                    continue
            else:
                if chunk_user != user_id:
                    continue
            matched.append(chunk)

        matched.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        return matched[:limit]

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

        # 임베딩 사본 저장 (delete_by_source 지원용)
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)

        meta_path = path / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "store_type": "faiss",
                    "version": INDEX_VERSION,
                },
                f,
            )

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

        # 임베딩 사본 로드 (v1 인덱스는 없으므로 인덱스에서 재구성)
        emb_path = path / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
        elif self.index.ntotal > 0:
            # v1 마이그레이션: 인덱스에서 벡터 재구성
            try:
                self.embeddings = np.vstack(
                    [self.index.reconstruct(i) for i in range(self.index.ntotal)]
                )
                logger.info("embeddings_reconstructed_from_v1_index", count=self.index.ntotal)
            except Exception as e:
                logger.warning("embedding_reconstruction_failed", error=str(e))
                self.embeddings = None
        else:
            self.embeddings = None

        # 차원 정보 확인
        meta_path = path / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if meta.get("dimension") != self.dimension:
                    logger.warning(
                        "index_dimension_mismatch",
                        expected=self.dimension,
                        loaded=meta.get("dimension"),
                    )

        logger.info("index_loaded", path=str(path), total_chunks=len(self.chunks))

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        """저장소 초기화"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.embeddings = None
        logger.info("index_cleared")


# 하위 호환성을 위한 별칭
VectorStore = FAISSStore
