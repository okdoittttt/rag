"""하이브리드 검색 모듈

BM25(키워드)와 Vector(의미) 검색 결과를 결합합니다.
RRF(Reciprocal Rank Fusion) 또는 Weighted Sum 방식을 지원합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Literal, Tuple

import numpy as np

from rag.chunking.chunk import Chunk
from rag.config import get_config
from rag.embedding import Embedder, VectorStoreBase
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


def rrf_score(rank: int, k: int = 60) -> float:
    """RRF 점수 계산
    
    Args:
        rank: 순위 (1부터 시작)
        k: 상수 (기본값 60, 논문 권장값)
        
    Returns:
        RRF 점수: 1 / (k + rank)
    """
    return 1.0 / (k + rank)


class HybridSearcher:
    """하이브리드 검색기"""
    
    def __init__(self, embedder: Embedder, vector_store: VectorStoreBase):
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
        alpha: float = 0.7,  # 기본적으로 벡터 검색 비중 높임
        fusion_type: Literal["rrf", "weighted"] = "weighted",  # 기본값을 weighted로 변경
        rrf_k: int = 60,
        user_id: str | None = None,  # 사용자 ID 필터
        source_filter: str | None = None,  # 문서 필터
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """하이브리드 검색

        Args:
            query: 검색 쿼리
            top_k: 반환 개수
            alpha: 벡터 검색 가중치 (0.0~1.0, weighted 방식에서만 사용)
            fusion_type: 융합 방식 ("rrf" 또는 "weighted")
            rrf_k: RRF 상수 (기본값 60)
            user_id: 사용자 ID (이 사용자의 문서만 검색)
            source_filter: 특정 문서로 검색 제한 (파일명)

        Returns:
            (청크, 최종 점수) 튜플 리스트
        """
        if fusion_type == "rrf":
            return self._search_rrf(query, top_k, rrf_k, user_id=user_id, source_filter=source_filter, **kwargs)
        else:
            return self._search_weighted(query, top_k, alpha, user_id=user_id, source_filter=source_filter, **kwargs)

    def _search_rrf(
        self,
        query: str,
        top_k: int,
        rrf_k: int = 60,
        user_id: str | None = None,
        source_filter: str | None = None,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """RRF 방식 하이브리드 검색

        두 검색 결과의 순위만 사용하여 점수 스케일에 무관한 통합을 수행합니다.
        """
        # 1. BM25 검색 (순위용) - user_id 필터 적용
        bm25_results = self.bm25.search(query, top_k=top_k * 2, user_id=user_id)

        # 2. 벡터 검색 (순위용) - user_id 필터 적용
        query_vec = self.embedder.embed_query(query)
        vec_results = self.vector_store.search(query_vec, top_k=top_k * 2, user_id=user_id)

        if not bm25_results and not vec_results:
            return []

        # 3. RRF 점수 계산
        # dedupe key: (source, chunk_index) — 멀티 문서 환경에서 chunk_index만으로는 충돌
        rrf_scores: dict[Any, tuple[Chunk, float]] = {}

        def _dedupe_key(chunk: Chunk) -> Any:
            return (
                chunk.metadata.get("source"),
                chunk.metadata.get("chunk_index", id(chunk)),
            )

        # BM25 결과에 RRF 점수 부여
        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            if source_filter and chunk.metadata.get("source") != source_filter:
                continue
            key = _dedupe_key(chunk)
            score = rrf_score(rank, rrf_k)
            if key in rrf_scores:
                rrf_scores[key] = (chunk, rrf_scores[key][1] + score)
            else:
                rrf_scores[key] = (chunk, score)

        # 벡터 결과에 RRF 점수 부여
        for rank, (chunk, _) in enumerate(vec_results, start=1):
            if source_filter and chunk.metadata.get("source") != source_filter:
                continue
            key = _dedupe_key(chunk)
            score = rrf_score(rank, rrf_k)
            if key in rrf_scores:
                rrf_scores[key] = (chunk, rrf_scores[key][1] + score)
            else:
                rrf_scores[key] = (chunk, score)

        # 4. 점수순 정렬
        final_results = list(rrf_scores.values())
        final_results.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            "rrf_search_completed",
            query_preview=query[:30],
            bm25_count=len(bm25_results),
            vec_count=len(vec_results),
            merged_count=len(final_results),
        )

        return final_results[:top_k]

    def _search_weighted(
        self,
        query: str,
        top_k: int,
        alpha: float = 0.5,
        user_id: str | None = None,
        source_filter: str | None = None,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Weighted Sum 방식 하이브리드 검색 (기존 로직)

        Score = alpha * Vector_Score + (1 - alpha) * BM25_Score
        """
        # BM25 점수 계산
        bm25_scores = self.bm25.get_full_scores(query)

        # 벡터 쿼리
        query_vec = self.embedder.embed_query(query)

        # 1차 검색 (후보군 추출) - user_id 필터 적용
        candidates = self.vector_store.search(query_vec, top_k=top_k * 3, user_id=user_id)

        if not candidates:
            # 벡터 결과도 없으면 BM25 검색 시도 (BM25 인덱스가 있을 경우)
            if len(bm25_scores) > 0:
                return self.bm25.search(query, top_k, user_id=user_id)
            return []

        # source_filter 적용 (벡터 검색 결과 필터링)
        if source_filter:
            candidates = [(chunk, score) for chunk, score in candidates
                         if chunk.metadata.get("source") == source_filter]

        if not candidates:
            return []

        # 점수 결합
        hybrid_results = []

        vec_scores = np.array([score for _, score in candidates])
        # 코사인 유사도(-1~1)는 그대로 사용 (상대적 순위보다 절대적 유사도가 중요)
        norm_vec_scores = vec_scores

        # BM25 corpus 위치 lookup: (source, chunk_index) -> bm25_scores 인덱스
        # chunk_index만으로는 멀티 문서에서 충돌하므로 source와 함께 사용
        bm25_lookup: dict[Any, int] = {}
        if len(bm25_scores) > 0:
            for i, bm25_chunk in enumerate(self.bm25.chunks):
                key = (
                    bm25_chunk.metadata.get("source"),
                    bm25_chunk.metadata.get("chunk_index"),
                )
                bm25_lookup[key] = i

        for i, (chunk, _) in enumerate(candidates):
            current_bm25_score = 0.0
            key = (chunk.metadata.get("source"), chunk.metadata.get("chunk_index"))
            bm25_idx = bm25_lookup.get(key)
            if bm25_idx is not None and bm25_idx < len(bm25_scores):
                current_bm25_score = bm25_scores[bm25_idx]

            hybrid_results.append({
                "chunk": chunk,
                "vec_score": norm_vec_scores[i],
                "bm25_score": current_bm25_score
            })

        # BM25 점수 정규화 및 가중치 동적 조정
        cand_bm25_scores = np.array([r["bm25_score"] for r in hybrid_results])
        bm25_has_signal = np.max(cand_bm25_scores) > 0

        if bm25_has_signal:
            norm_bm25_scores = min_max_normalize(cand_bm25_scores)
            effective_alpha = alpha
        else:
            # BM25 매칭이 전혀 없는 경우 (cross-lingual 쿼리 등)
            # 벡터 검색 점수만으로 순위 결정
            norm_bm25_scores = np.zeros_like(cand_bm25_scores)
            effective_alpha = 1.0
            logger.debug(
                "bm25_no_signal_fallback_to_vector",
                query_preview=query[:30],
            )

        # 최종 점수 계산 및 정렬
        final_results = []
        for i, res in enumerate(hybrid_results):
            final_score = (effective_alpha * res["vec_score"]) + ((1 - effective_alpha) * norm_bm25_scores[i])
            final_results.append((res["chunk"], final_score))

        final_results.sort(key=lambda x: x[1], reverse=True)

        # 점수 임계값 필터링
        filtered_results = [
            (chunk, score) for chunk, score in final_results
            if score >= kwargs.get("score_threshold", 0.0)
        ]

        return filtered_results[:top_k]
    
    def fetch_full_document(
        self,
        source: str,
        user_id: str | None = None,
        max_chunks: int = 50,
    ) -> List[Chunk]:
        """주어진 ``source`` 의 모든 청크를 ``chunk_index`` 오름차순으로 반환한다.

        검색이 아닌 "조회"이므로 BM25/벡터 점수 계산을 건너뛴다. 요약·전체
        조망 질의에서 컨텍스트 누락(top_k 절단으로 헤더/날짜만 답변되는 현상)
        을 방지하기 위해 사용된다.

        Args:
            source: 대상 ``source`` 파일명.
            user_id: 사용자 ID 필터.
            max_chunks: LLM 컨텍스트 윈도우 보호용 상한.

        Returns:
            청크 리스트(``chunk_index`` 오름차순, 최대 ``max_chunks`` 개).
        """
        chunks = self.vector_store.get_all_by_source(
            source=source,
            user_id=user_id,
            limit=max_chunks,
        )
        logger.debug(
            "fetch_full_document",
            source=source,
            user_id=user_id,
            returned=len(chunks),
            max_chunks=max_chunks,
        )
        return chunks

    def delete_by_source(self, source: str, user_id: str | None = None) -> int:
        """source(+user_id) 매칭 청크를 BM25/벡터 저장소 모두에서 삭제

        Returns:
            벡터 저장소에서 삭제된 청크 수 (BM25와 동일해야 정상)
        """
        vec_deleted = self.vector_store.delete_by_source(source, user_id=user_id)
        bm25_deleted = self.bm25.delete_by_source(source, user_id=user_id)
        if vec_deleted != bm25_deleted:
            logger.warning(
                "delete_count_mismatch",
                vector_deleted=vec_deleted,
                bm25_deleted=bm25_deleted,
                source=source,
                user_id=user_id,
            )
        return vec_deleted

    def save(self, path: Path) -> None:
        """인덱스 저장"""
        self.vector_store.save(path)
        self.bm25.save(path)

    def load(self, path: Path) -> None:
        """인덱스 로드"""
        self.vector_store.load(path)
        self.bm25.load(path)
