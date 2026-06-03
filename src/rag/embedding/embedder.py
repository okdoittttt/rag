"""임베딩 생성 모듈

provider 설정에 따라 두 가지 백엔드로 텍스트를 벡터로 변환한다.

- ``local``: SentenceTransformers 로 로컬에서 임베딩(모델 다운로드/메모리 필요).
- ``gemini``: Google Gemini Embedding API(``google-genai``) 로 임베딩(로컬 모델 0).

검색 품질을 위해 쿼리에는 task 지침을 prepend 하고 문서는 원문 그대로 임베딩한다
(비대칭 인코딩). 코사인 유사도 사용을 위해 항상 L2 정규화된 벡터를 반환한다.
"""

from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv

from rag.config import get_config
from rag.logger import get_logger


load_dotenv()

logger = get_logger(__name__)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """행 단위로 L2 정규화한다(영벡터는 그대로 둔다).

    Args:
        arr: shape ``[N, dim]`` 임베딩 배열.

    Returns:
        각 행이 단위 벡터인 배열. 코사인 유사도를 내적으로 계산하기 위함이다.
    """
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class Embedder:
    """텍스트 임베딩 생성기.

    ``config.embedding.provider`` 에 따라 로컬(SentenceTransformers) 또는 Gemini
    Embedding API 백엔드를 사용한다. 호출 인터페이스(``embed``/``embed_query``)는
    백엔드와 무관하게 동일하다.
    """

    def __init__(self, model_name: str | None = None, provider: str | None = None):
        """임베딩 백엔드를 초기화한다.

        Args:
            model_name: 사용할 모델명. ``None`` 이면 설정 파일값 사용.
            provider: ``"local"`` 또는 ``"gemini"``. ``None`` 이면 설정값
                (``config.embedding.provider``) 을 따른다.

        Raises:
            ValueError: provider 가 ``"gemini"`` 인데 ``GOOGLE_API_KEY`` 가 없을 때.
        """
        config = get_config()
        self.provider = provider or config.embedding.provider
        self.model_name = model_name or config.embedding.model
        self.dimension = config.embedding.dimension

        if self.provider == "gemini":
            self._init_gemini()
        else:
            self._init_local(config)

    def _init_local(self, config) -> None:
        """로컬 SentenceTransformers 백엔드를 로드한다.

        실행 디바이스는 ``config.embedding.device`` 를 따른다(``None`` 이면 자동 선택).

        Args:
            config: 전역 설정 객체.
        """
        # SentenceTransformer 는 로컬 백엔드에서만 필요하므로 지연 import 한다.
        from sentence_transformers import SentenceTransformer

        self.device = config.embedding.device
        logger.info("loading_embedding_model", model=self.model_name, device=self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def _init_gemini(self) -> None:
        """Gemini Embedding API 클라이언트를 초기화한다.

        ``GeminiLLM`` 과 동일하게 ``GOOGLE_API_KEY`` 환경변수를 사용한다.

        Raises:
            ValueError: ``GOOGLE_API_KEY`` 가 설정되지 않은 경우.
        """
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API Key not found. Please set GOOGLE_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=api_key)
        logger.info("gemini_embedder_initialized", model=self.model_name, dimension=self.dimension)

    def embed(self, texts: list[str]) -> np.ndarray:
        """문서 텍스트 리스트를 벡터로 변환한다(지침 미적용).

        Args:
            texts: 임베딩할 텍스트 리스트.

        Returns:
            임베딩 벡터 (numpy array, shape=``[N, dim]``). 빈 입력은 ``[]``.
        """
        if not texts:
            return np.array([])

        if self.provider == "gemini":
            return self._embed_gemini(texts, is_query=False)

        config = get_config()
        batch_size = config.embedding.batch_size
        logger.debug("embedding_texts", count=len(texts), batch_size=batch_size)
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 코사인 유사도를 위해 정규화
        )

    def embed_query(self, query: str) -> np.ndarray:
        """쿼리를 벡터로 변환한다(검색 지침 prepend).

        쿼리 측에 ``config.embedding.query_instruction`` 을 적용하면 검색 품질이
        향상된다. 문서 임베딩(``embed``)에는 지침을 적용하지 않는다(비대칭 인코딩).

        Args:
            query: 검색 쿼리.

        Returns:
            임베딩 벡터 (numpy array, shape=``[1, dim]``).
        """
        if self.provider == "gemini":
            return self._embed_gemini([query], is_query=True)

        config = get_config()
        instruction = config.embedding.query_instruction
        logger.debug("embedding_query", has_instruction=bool(instruction))
        return self.model.encode(
            [query],
            prompt=instruction or None,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # 코사인 유사도를 위해 정규화
        )

    def _embed_gemini(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Gemini Embedding API 로 텍스트를 임베딩한다.

        쿼리면 각 텍스트에 task 지침을 prepend 한다. 입력 토큰/요청 크기 제한을
        고려해 ``batch_size`` 단위로 나눠 호출하고, 입력별로 분리된 임베딩을 보장한다.

        Args:
            texts: 임베딩할 텍스트 리스트.
            is_query: 쿼리면 ``True`` (지침 prepend), 문서면 ``False``.

        Returns:
            L2 정규화된 임베딩 배열 (shape=``[N, dim]``).
        """
        config = get_config()
        instruction = config.embedding.query_instruction if is_query else ""
        inputs = [f"{instruction}{t}" if instruction else t for t in texts]
        batch_size = max(1, config.embedding.batch_size)

        vectors: list[list[float]] = []
        for i in range(0, len(inputs), batch_size):
            vectors.extend(self._embed_gemini_batch(inputs[i : i + batch_size]))

        return _l2_normalize(np.array(vectors, dtype=np.float32))

    def _embed_gemini_batch(self, batch: list[str]) -> list[list[float]]:
        """배치를 한 번에 임베딩하되, 입력 수와 응답 수가 다르면 개별 호출로 폴백한다.

        Gemini Embedding 2 는 입력 리스트를 하나로 합쳐 반환할 수 있어, 응답 길이를
        검증하고 불일치 시 항목별 호출로 분리 임베딩을 보장한다.

        Args:
            batch: 임베딩할(이미 지침이 적용된) 텍스트 배치.

        Returns:
            각 입력에 대응하는 float 벡터 리스트.

        Raises:
            errors.APIError: Gemini API 호출이 실패한 경우(조용한 손상 방지를 위해 전파).
        """
        from google.genai import errors, types

        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=self.dimension),
            )
            embeddings = getattr(result, "embeddings", None)
            if embeddings is not None and len(embeddings) == len(batch):
                return [list(e.values) for e in embeddings]

            # 응답이 합쳐졌거나 개수가 안 맞으면 입력별 개별 호출로 폴백한다.
            logger.warning(
                "gemini_embed_count_mismatch",
                expected=len(batch),
                got=(len(embeddings) if embeddings is not None else None),
            )
            return [self._embed_gemini_single(text) for text in batch]
        except errors.APIError as e:
            logger.error("gemini_embed_api_error", error=str(e))
            raise

    def _embed_gemini_single(self, text: str) -> list[float]:
        """단일 텍스트를 임베딩한다(배치 폴백용).

        Args:
            text: 임베딩할 텍스트.

        Returns:
            float 벡터.
        """
        from google.genai import types

        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        return list(result.embeddings[0].values)
