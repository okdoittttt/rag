"""BM25 검색 모듈

rank_bm25를 사용한 키워드 검색 기능을 제공합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunking.chunk import Chunk
from rag.logger import get_logger
from rag.retrieval.tokenizer import tokenize_content, tokenize_query


logger = get_logger(__name__)


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Chunk를 JSON 직렬화 가능한 dict로 변환"""
    return {
        "content": chunk.content,
        "metadata": chunk.metadata,
    }


def _dict_to_chunk(data: dict[str, Any]) -> Chunk:
    """dict를 Chunk 객체로 복원"""
    return Chunk(
        content=data["content"],
        metadata=data.get("metadata", {}),
    )


class BM25Searcher:
    """BM25 검색기"""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]) -> None:
        """청크 추가 (누적)

        vector_store.add()와 동일하게 기존 청크에 누적된다.
        rank_bm25는 증분 업데이트를 지원하지 않으므로 전체 코퍼스를 재구축한다.

        Args:
            chunks: 인덱싱할 청크 리스트
        """
        if not chunks:
            return

        self.chunks = list(self.chunks) + list(chunks)

        tokenized_corpus = [
            tokenize_content(chunk.content)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("bm25_indexed", added=len(chunks), total=len(self.chunks))

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """키워드 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 개수
            user_id: 사용자 ID 필터. None이면 필터 없음.

        Returns:
            (청크, 점수) 튜플 리스트. 점수는 정규화되지 않음.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = tokenize_query(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 점수 내림차순 정렬 (user_id 필터를 위해 전체 정렬 후 필터링)
        sorted_indices = np.argsort(scores)[::-1]

        results: List[Tuple[Chunk, float]] = []
        for idx in sorted_indices:
            score = scores[idx]
            if score <= 0:
                break
            chunk = self.chunks[idx]
            if user_id is not None and chunk.metadata.get("user_id") != user_id:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break

        return results

    def get_full_scores(self, query: str) -> np.ndarray:
        """전체 문서에 대한 점수 반환 (Hybrid 검색용)"""
        if not self.bm25:
            return np.array([])

        tokenized_query = tokenize_query(query)
        return np.array(self.bm25.get_scores(tokenized_query))

    def delete_by_source(self, source: str, user_id: str | None = None) -> int:
        """source(+user_id) 매칭 청크 삭제 후 BM25 재인덱싱

        Args:
            source: chunk.metadata["source"]
            user_id: 사용자 ID. None이면 user_id 없는 청크만 삭제.

        Returns:
            삭제된 청크 수
        """
        if not self.chunks:
            return 0

        def _matches(chunk: Chunk) -> bool:
            if chunk.metadata.get("source") != source:
                return False
            chunk_user = chunk.metadata.get("user_id")
            if user_id is None:
                return chunk_user in (None, "")
            return chunk_user == user_id

        remaining = [c for c in self.chunks if not _matches(c)]
        deleted = len(self.chunks) - len(remaining)

        if deleted == 0:
            return 0

        self.chunks = remaining
        if remaining:
            self._rebuild_bm25()
        else:
            self.bm25 = None

        logger.info(
            "bm25_chunks_deleted",
            source=source,
            user_id=user_id,
            count=deleted,
            remaining=len(remaining),
        )
        return deleted

    def save(self, path: Path) -> None:
        """인덱스 저장 (JSON 포맷)"""
        if not self.bm25:
            return

        path.mkdir(parents=True, exist_ok=True)

        chunks_data = [_chunk_to_dict(chunk) for chunk in self.chunks]
        with open(path / "bm25_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False)

        logger.info("bm25_saved", path=str(path), count=len(self.chunks))

    def load(self, path: Path) -> None:
        """인덱스 로드 (JSON에서 청크를 읽고 BM25를 재구축)"""
        json_path = path / "bm25_chunks.json"

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            self.chunks = [_dict_to_chunk(d) for d in chunks_data]
            self._rebuild_bm25()
            return

        # 기존 pickle 파일이 있는 경우 경고
        pkl_path = path / "bm25.pkl"
        if pkl_path.exists():
            logger.warning(
                "bm25_pickle_deprecated",
                message="bm25.pkl은 보안 위험으로 더 이상 로드하지 않습니다. 문서를 다시 인덱싱하세요.",
                path=str(pkl_path),
            )
            return

        logger.warning("bm25_index_not_found", path=str(path))

    def _rebuild_bm25(self) -> None:
        """로드된 청크로부터 BM25 인덱스를 재구축"""
        if not self.chunks:
            return

        tokenized_corpus = [
            tokenize_content(chunk.content)
            for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("bm25_rebuilt_from_json", count=len(self.chunks))
