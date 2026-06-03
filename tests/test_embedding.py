"""Embedding 모듈 테스트"""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.chunking.chunk import Chunk
from rag.config import get_config
from rag.embedding.embedder import Embedder
from rag.embedding.faiss_store import FAISSStore


# 테스트용 임베딩 모델 (매우 작음)
TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _unit_vec(values: list[float], dim: int = 4) -> np.ndarray:
    """단위 정규화된 임베딩 벡터 생성 (코사인 유사도용)"""
    v = np.array(values + [0.0] * (dim - len(values)), dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _make_chunk(content: str, source: str, chunk_index: int, user_id: str | None = None) -> Chunk:
    meta = {"source": source, "chunk_index": chunk_index}
    if user_id is not None:
        meta["user_id"] = user_id
    return Chunk(content=content, metadata=meta)


@pytest.fixture(scope="module")
def embedder():
    """모듈 레벨에서 한 번만 모델 로딩 (로컬 백엔드 강제)"""
    return Embedder(model_name=TEST_MODEL, provider="local")


class TestEmbedder:
    """Embedder 테스트"""

    def test_embed_single_text(self, embedder):
        """단일 텍스트 임베딩"""
        texts = ["Hello world"]
        embeddings = embedder.embed(texts)

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 1
        assert embeddings.shape[1] == 384  # MiniLM 차원

    def test_embed_multiple_texts(self, embedder):
        """다중 텍스트 임베딩"""
        texts = ["Hello", "World", "Python"]
        embeddings = embedder.embed(texts)

        assert len(embeddings) == 3
        assert embeddings.shape[1] == 384

    def test_embed_empty_list(self, embedder):
        """빈 리스트 처리"""
        embeddings = embedder.embed([])

        assert len(embeddings) == 0

    def test_embed_query(self, embedder):
        """쿼리 임베딩"""
        embedding = embedder.embed_query("search query")

        assert len(embedding) == 1
        assert embedding.shape[1] == 384


class TestLocalEmbedderInstructionAndDevice:
    """로컬(SentenceTransformers) 백엔드의 instruction 비대칭·device 배선 검증.

    실제 모델 로드를 피하기 위해 ``SentenceTransformer`` 를 모킹한다. ``Embedder``
    가 백엔드에서 지연 import 하므로 원천 모듈(``sentence_transformers``)을 패치한다.
    """

    def _mock_model(self) -> MagicMock:
        """``encode`` 가 임의의 벡터를 반환하는 모킹 모델."""
        model = MagicMock()
        model.encode.return_value = np.zeros((1, 4), dtype=np.float32)
        return model

    def test_device_passed_to_sentence_transformer(self) -> None:
        """``config.embedding.device`` 가 ``SentenceTransformer`` 로 전달된다."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_st.return_value = self._mock_model()
            Embedder(model_name="dummy-model", provider="local")

            _, kwargs = mock_st.call_args
            assert "device" in kwargs

    def test_embed_query_applies_instruction_prompt(self) -> None:
        """``embed_query`` 는 쿼리 측 instruction 을 ``prompt`` 로 전달한다."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            model = self._mock_model()
            mock_st.return_value = model
            embedder = Embedder(model_name="dummy-model", provider="local")

            embedder.embed_query("이 논문의 문제는?")

            _, kwargs = model.encode.call_args
            assert kwargs.get("prompt")  # query_instruction 이 비어있지 않게 전달됨

    def test_embed_documents_has_no_instruction_prompt(self) -> None:
        """문서 임베딩(``embed``)에는 instruction 을 적용하지 않는다(비대칭)."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            model = self._mock_model()
            mock_st.return_value = model
            embedder = Embedder(model_name="dummy-model", provider="local")

            embedder.embed(["문서 본문 내용"])

            _, kwargs = model.encode.call_args
            assert "prompt" not in kwargs


class TestGeminiEmbedder:
    """Gemini Embedding API 백엔드 테스트 (``google-genai`` mock).

    실제 네트워크 호출 없이 ``genai.Client`` 를 모킹해 분기/지침/차원/폴백을 검증한다.
    """

    @pytest.fixture
    def mock_client(self):
        """``genai.Client`` 를 모킹하고 mock client 를 노출한다(API 키도 주입)."""
        with patch("google.genai.Client") as cls, patch.dict(
            os.environ, {"GOOGLE_API_KEY": "fake_key"}
        ):
            client = MagicMock()
            cls.return_value = client
            yield client

    def _resp(self, n: int) -> SimpleNamespace:
        """``embeddings[i].values`` 구조의 가짜 응답을 ``n`` 개 만든다."""
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3, 0.4]) for _ in range(n)]
        )

    def test_init_requires_api_key(self) -> None:
        """gemini provider 인데 키가 없으면 ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                Embedder(provider="gemini")

    def test_embed_returns_matrix(self, mock_client) -> None:
        """``embed`` 가 입력 수만큼의 벡터 행렬을 반환한다."""
        mock_client.models.embed_content.return_value = self._resp(2)
        emb = Embedder(provider="gemini")

        out = emb.embed(["doc a", "doc b"])

        assert out.shape == (2, 4)

    def test_embed_query_prepends_instruction_and_sets_dim(self, mock_client) -> None:
        """쿼리에 task 지침이 prepend 되고 output_dimensionality 가 설정된다."""
        mock_client.models.embed_content.return_value = self._resp(1)
        emb = Embedder(provider="gemini")

        emb.embed_query("질문")

        _, kwargs = mock_client.models.embed_content.call_args
        assert kwargs["contents"][0].startswith("task: search result | query: ")
        assert kwargs["config"].output_dimensionality == get_config().embedding.dimension

    def test_embed_documents_no_instruction(self, mock_client) -> None:
        """문서 임베딩에는 지침을 붙이지 않는다(원문 그대로)."""
        mock_client.models.embed_content.return_value = self._resp(1)
        emb = Embedder(provider="gemini")

        emb.embed(["plain document"])

        _, kwargs = mock_client.models.embed_content.call_args
        assert kwargs["contents"][0] == "plain document"

    def test_count_mismatch_falls_back_to_single(self, mock_client) -> None:
        """배치 응답 수가 입력과 다르면 항목별 호출로 폴백해 N개를 보장한다."""
        # 1) 배치 호출: 2개 기대했으나 1개만 반환(합쳐짐) → 폴백
        # 2~3) 항목별 단일 호출 2회
        mock_client.models.embed_content.side_effect = [
            self._resp(1),
            self._resp(1),
            self._resp(1),
        ]
        emb = Embedder(provider="gemini")

        out = emb.embed(["a", "b"])

        assert out.shape == (2, 4)


class TestFAISSStore:
    """FAISSStore 직접 테스트 (Embedder 의존 없음)"""

    def test_add_and_search(self):
        store = FAISSStore(dimension=4)
        chunks = [
            _make_chunk("apple", "a.txt", 0),
            _make_chunk("banana", "a.txt", 1),
        ]
        embs = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([0, 1, 0])])
        store.add(chunks, embs)
        assert store.total_chunks == 2

        results = store.search(_unit_vec([1, 0, 0]).reshape(1, -1), top_k=1)
        assert len(results) == 1
        assert results[0][0].content == "apple"

    def test_user_id_filter(self):
        store = FAISSStore(dimension=4)
        chunks = [
            _make_chunk("u1 apple", "a.txt", 0, user_id="u1"),
            _make_chunk("u2 apple", "b.txt", 0, user_id="u2"),
        ]
        embs = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([1, 0, 0])])
        store.add(chunks, embs)

        results = store.search(_unit_vec([1, 0, 0]).reshape(1, -1), top_k=5, user_id="u1")
        assert len(results) == 1
        assert results[0][0].metadata["user_id"] == "u1"

    def test_delete_by_source(self):
        store = FAISSStore(dimension=4)
        chunks = [
            _make_chunk("c1", "a.txt", 0, user_id="u1"),
            _make_chunk("c2", "a.txt", 1, user_id="u1"),
            _make_chunk("c3", "b.txt", 0, user_id="u1"),
        ]
        embs = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([0, 1, 0]), _unit_vec([0, 0, 1])])
        store.add(chunks, embs)
        assert store.total_chunks == 3

        deleted = store.delete_by_source("a.txt", user_id="u1")
        assert deleted == 2
        assert store.total_chunks == 1

        # 남은 청크는 b.txt
        results = store.search(_unit_vec([0, 0, 1]).reshape(1, -1), top_k=5)
        assert len(results) == 1
        assert results[0][0].metadata["source"] == "b.txt"

    def test_delete_isolates_users(self):
        """같은 source라도 다른 user_id의 청크는 보존"""
        store = FAISSStore(dimension=4)
        chunks = [
            _make_chunk("u1 doc", "report.pdf", 0, user_id="u1"),
            _make_chunk("u2 doc", "report.pdf", 0, user_id="u2"),
        ]
        embs = np.vstack([_unit_vec([1, 0, 0]), _unit_vec([0, 1, 0])])
        store.add(chunks, embs)

        deleted = store.delete_by_source("report.pdf", user_id="u1")
        assert deleted == 1
        assert store.total_chunks == 1
        # 남은 청크는 u2의 것
        results = store.search(_unit_vec([0, 1, 0]).reshape(1, -1), top_k=5)
        assert results[0][0].metadata["user_id"] == "u2"

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        store = FAISSStore(dimension=4)
        chunks = [_make_chunk("hello", "a.txt", 0, user_id="u1")]
        store.add(chunks, _unit_vec([1, 0, 0]).reshape(1, -1))
        store.save(tmp_path)

        new_store = FAISSStore(dimension=4)
        new_store.load(tmp_path)
        assert new_store.total_chunks == 1

        # 로드 후에도 user_id 필터 동작
        results = new_store.search(_unit_vec([1, 0, 0]).reshape(1, -1), top_k=5, user_id="u1")
        assert len(results) == 1

    def test_reindex_does_not_accumulate(self):
        """delete_by_source 후 같은 청크 재추가 시 총 개수가 유지되어야 함"""
        store = FAISSStore(dimension=4)
        chunks = [_make_chunk("doc", "a.txt", 0, user_id="u1")]
        emb = _unit_vec([1, 0, 0]).reshape(1, -1)

        store.add(chunks, emb)
        store.delete_by_source("a.txt", user_id="u1")
        store.add(chunks, emb)
        assert store.total_chunks == 1
