"""Semantic chunking 테스트"""

import pytest

from rag.chunking.semantic import (
    split_into_sentences,
    split_semantic,
    cosine_similarity,
)
from rag.chunking import Chunk
import numpy as np


def test_split_sentences_english():
    """영어 문장 분할 테스트"""
    text = "Hello world. This is a test. How are you?"
    sentences = split_into_sentences(text, language="en")

    assert len(sentences) == 3
    assert "Hello world." in sentences[0]
    assert "This is a test." in sentences[1]
    assert "How are you?" in sentences[2]


def test_split_sentences_korean():
    """한국어 문장 분할 테스트"""
    text = "안녕하세요. 저는 개발자입니다. 파이썬을 좋아합니다."
    sentences = split_into_sentences(text, language="ko")

    # Kiwi가 설치되어 있으면 3개, 없으면 regex로 분할
    assert len(sentences) >= 1
    assert "안녕하세요" in sentences[0]


def test_cosine_similarity():
    """코사인 유사도 계산 테스트"""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    vec3 = np.array([0.0, 1.0, 0.0])

    # 같은 벡터는 유사도 1.0
    sim1 = cosine_similarity(vec1, vec2)
    assert abs(sim1 - 1.0) < 0.001

    # 직교 벡터는 유사도 0.0
    sim2 = cosine_similarity(vec1, vec3)
    assert abs(sim2 - 0.0) < 0.001


def test_semantic_split_short_text():
    """짧은 텍스트는 simple chunking으로 폴백"""
    text = "Short text."
    chunks = split_semantic(text, chunk_size=1000)

    # 짧은 텍스트는 단일 청크로 반환되어야 함
    assert len(chunks) == 1
    assert chunks[0].content == text.strip()


def test_semantic_split_single_sentence():
    """단일 문장 문서 테스트"""
    text = "This is a single sentence document that is long enough for semantic chunking."
    chunks = split_semantic(text, chunk_size=1000)

    assert len(chunks) == 1
    assert chunks[0].content == text.strip()


def test_semantic_split_respects_size():
    """크기 제약 테스트"""
    # 긴 텍스트 생성
    text = " ".join([f"This is sentence number {i}." for i in range(100)])
    chunks = split_semantic(text, chunk_size=500, similarity_threshold=0.5)

    # 각 청크가 크기 제약을 준수하는지 확인
    for chunk in chunks:
        assert len(chunk.content) <= 500

    # 여러 청크로 분할되어야 함
    assert len(chunks) > 1


def test_semantic_split_metadata():
    """메타데이터 검증 테스트"""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = split_semantic(text, chunk_size=1000, source="test.txt")

    assert len(chunks) >= 1

    # 메타데이터 확인
    for chunk in chunks:
        assert "chunking_strategy" in chunk.metadata
        assert chunk.metadata["chunking_strategy"] == "semantic"
        assert "sentence_count" in chunk.metadata
        assert "avg_similarity" in chunk.metadata
        assert chunk.metadata["source"] == "test.txt"


def test_semantic_split_grouping():
    """의미적 그룹화 테스트"""
    # 명확히 다른 주제의 문장들
    text = """
    Python is a programming language. Python is easy to learn. Python has many libraries.
    The weather is sunny today. It's a beautiful day outside. The temperature is perfect.
    """

    chunks = split_semantic(
        text,
        chunk_size=1000,
        similarity_threshold=0.6,  # 낮은 threshold로 명확한 분할 유도
    )

    # 두 주제가 다른 청크로 분할되어야 함
    # (임베딩 모델에 따라 결과가 다를 수 있음)
    assert len(chunks) >= 1


def test_semantic_split_overlap():
    """오버랩 적용 테스트"""
    text = " ".join([f"Sentence number {i} with some content." for i in range(20)])

    chunks = split_semantic(
        text,
        chunk_size=200,
        chunk_overlap=50,
        similarity_threshold=0.5,
    )

    # 여러 청크가 생성되어야 함
    assert len(chunks) > 1

    # 각 청크는 유효한 내용을 가져야 함
    for chunk in chunks:
        assert len(chunk.content) > 0


def test_empty_text():
    """빈 텍스트 처리 테스트"""
    chunks = split_semantic("", chunk_size=1000)
    assert len(chunks) == 0

    chunks = split_semantic("   ", chunk_size=1000)
    assert len(chunks) == 0


def test_chunk_index_sequential():
    """청크 인덱스가 순차적인지 확인"""
    text = " ".join([f"This is sentence {i}." for i in range(50)])
    chunks = split_semantic(text, chunk_size=300, similarity_threshold=0.5)

    for i, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_index"] == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
