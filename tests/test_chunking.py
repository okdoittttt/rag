"""Chunking 모듈 테스트"""

from pathlib import Path

import pytest

from rag.chunking import (
    Chunk,
    split_text,
    split_markdown,
    chunk_document,
)
from rag.ingestion.document import Document


class TestChunk:
    """Chunk 클래스 테스트"""
    
    def test_create_chunk(self) -> None:
        """Chunk 생성 확인"""
        chunk = Chunk.create(
            content="Test content",
            source="/path/to/file.txt",
            chunk_index=0,
            start_char=0,
            end_char=12,
        )
        
        assert chunk.content == "Test content"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["source"] == "/path/to/file.txt"
    
    def test_chunk_length(self) -> None:
        """Chunk 길이 반환 확인"""
        chunk = Chunk(content="12345")
        
        assert len(chunk) == 5


class TestSplitText:
    """기본 텍스트 분할 테스트"""
    
    def test_split_short_text(self) -> None:
        """짧은 텍스트는 하나의 청크"""
        chunks = split_text("Hello world", chunk_size=100)
        
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"
    
    def test_split_long_text(self) -> None:
        """긴 텍스트 분할"""
        text = "A" * 300  # 300자
        chunks = split_text(text, chunk_size=100, chunk_overlap=0)
        
        assert len(chunks) == 3
    
    def test_split_preserves_sentence_boundary(self) -> None:
        """문장 경계 보존"""
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text(text, chunk_size=30, chunk_overlap=0)
        
        # 첫 청크가 문장 경계에서 끝나는지 확인
        assert chunks[0].content.endswith(".")
    
    def test_split_with_overlap(self) -> None:
        """오버랩 적용 확인"""
        text = "A" * 100 + " " + "B" * 100
        chunks = split_text(text, chunk_size=110, chunk_overlap=20)
        
        # 오버랩으로 인해 일부 내용이 중복되어야 함
        if len(chunks) > 1:
            # 두 번째 청크가 첫 번째 청크의 끝부분과 겹치는지
            assert len(chunks[1].content) > 0
    
    def test_split_empty_text(self) -> None:
        """빈 텍스트는 빈 리스트"""
        chunks = split_text("")
        
        assert chunks == []
    
    def test_chunk_metadata_positions(self) -> None:
        """청크 위치 메타데이터 확인"""
        text = "Hello world. This is test."
        chunks = split_text(text, chunk_size=15, chunk_overlap=0)
        
        # 첫 청크의 위치 정보 확인
        assert chunks[0].metadata["start_char"] == 0
        assert chunks[0].metadata["chunk_index"] == 0


class TestWordBoundarySplitting:
    """단어 경계 보존 및 미니 청크 병합 테스트"""

    def test_no_mid_word_cut_for_separatorless_text(self) -> None:
        """구분자 없는 긴 텍스트에서 단어 중간 절단이 발생하지 않음"""
        # "단어"를 공백으로 이어 붙인 긴 텍스트(문단/문장 구분자 없음)
        word = "단어"
        text = " ".join([word] * 200)
        chunks = split_text(text, chunk_size=80, chunk_overlap=0)

        assert len(chunks) > 1
        # 각 청크 본문 자체에 깨진 단어("단" 또는 "어" 단독)가 없어야 함
        for chunk in chunks:
            assert not chunk.content.startswith("어 ")
            assert not chunk.content.endswith(" 단")

    def test_adjacent_chunks_do_not_split_word(self) -> None:
        """인접 청크 경계가 단어 중간을 자르지 않음(토큰 온전성 검증)"""
        text = " ".join(["alpha", "bravo", "charlie", "delta"] * 60)
        chunks = split_text(text, chunk_size=70, chunk_overlap=0)

        words = set(text.split())
        for chunk in chunks:
            # 청크 양끝 토큰이 사전에 존재하는 온전한 단어여야 함
            tokens = chunk.content.split()
            assert tokens, "빈 청크가 생성되면 안 됨"
            assert tokens[0] in words
            assert tokens[-1] in words

    def test_mixed_korean_english_number(self) -> None:
        """한국어+영문+숫자 혼재 입력에서 정상 분할"""
        unit = "회의는 Aug 2023 에 진행되었고 결과는 95점 입니다. "
        text = unit * 30
        chunks = split_text(text, chunk_size=120, chunk_overlap=20)

        assert len(chunks) > 1
        # "Aug"가 "A" / "ug"처럼 쪼개져 청크 끝/시작에 남지 않아야 함
        for chunk in chunks:
            assert not chunk.content.endswith("A")
            assert not chunk.content.startswith("ug ")

    def test_overlap_start_is_word_boundary(self) -> None:
        """오버랩 시작점이 단어 경계인지 검증"""
        text = " ".join([f"word{i:03d}" for i in range(200)])
        chunks = split_text(text, chunk_size=90, chunk_overlap=30)

        words = set(text.split())
        # 첫 청크를 제외한 모든 청크는 온전한 단어로 시작해야 함
        for chunk in chunks[1:]:
            first_token = chunk.content.split()[0]
            assert first_token in words

    def test_mini_chunk_merged_into_neighbor(self) -> None:
        """50자 미만 미니 청크가 인접 청크에 병합되어 단독으로 남지 않음"""
        # 큰 본문 뒤에 아주 짧은 꼬리를 붙여 마지막 미니 청크 발생 유도
        text = "가나다라마바사아자차카타파하 " * 50 + "끝."
        chunks = split_text(text, chunk_size=200, chunk_overlap=0)

        # 마지막 청크가 50자 미만으로 단독 생성되면 안 됨
        assert all(
            len(c.content) >= 50 or len(chunks) == 1 for c in chunks
        )
        # 꼬리 텍스트("끝.")는 마지막 청크에 포함되어야 함
        assert chunks[-1].content.rstrip().endswith("끝.")

    def test_chunk_index_and_positions_consistency(self) -> None:
        """chunk_index 연속성과 start_char/end_char 정합성 검증"""
        text = " ".join([f"token{i:04d}" for i in range(300)])
        chunks = split_text(text, chunk_size=120, chunk_overlap=30)

        # chunk_index는 0부터 빈틈없이 연속
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

        # start_char/end_char는 원문 범위 안에서 단조 증가하는 정합성 유지
        for c in chunks:
            sc = c.metadata["start_char"]
            ec = c.metadata["end_char"]
            assert 0 <= sc < ec <= len(text)
        starts = [c.metadata["start_char"] for c in chunks]
        assert starts == sorted(starts)

    def test_long_unbreakable_token_falls_back(self) -> None:
        """경계 없는 초장문 토큰에서 폴백 동작(무한 루프/예외 없음)"""
        text = "x" * 500
        chunks = split_text(text, chunk_size=100, chunk_overlap=0)

        assert len(chunks) >= 1
        # 모든 본문을 합치면 원문 길이를 복원(폴백 강제 분할이 동작)
        assert sum(len(c.content) for c in chunks) == len(text)
        # 인덱스 연속성 유지
        assert [c.metadata["chunk_index"] for c in chunks] == list(
            range(len(chunks))
        )

    def test_short_paragraph_then_long_body_no_microchunk_explosion(self) -> None:
        """짧은 단락 뒤 긴 본문에서 마이크로청크 폭증이 일어나지 않음.

        과거 버그: 짧은 단락(split_point < chunk_overlap) 다음 시작점이 뒤로 밀려
        1자씩만 전진 → 거의 동일한 청크 수천 개가 양산되었다. 청크 수가 원문
        글자수가 아니라 chunk_size에 비례하는지 검증한다.
        """
        # 푸터 같은 짧은 단락 + 구분자 없는 긴 본문을 반복.
        short = "footer.\n\n"          # 오버랩(200)보다 짧은 단락
        long_body = "word " * 800       # 약 4000자, 내부에 \n\n 없음
        text = (short + long_body) * 5
        chunks = split_text(text, chunk_size=1500, chunk_overlap=200)

        # 청크 수는 원문 길이/청크 크기 수준이어야 한다(글자수에 비례하면 버그).
        # 수정 전이면 수천 개라 이 상한에서 실패한다.
        assert len(chunks) < len(text) // 200

    def test_overlap_does_not_create_one_char_shifted_duplicates(self) -> None:
        """인접 청크가 1글자만 민 근접 중복이 아님(슬라이딩 버그 회귀 방지)."""
        short = "x.\n\n"
        long_body = "alpha bravo charlie delta echo " * 100
        text = short + long_body
        chunks = split_text(text, chunk_size=500, chunk_overlap=100)

        contents = [c.content for c in chunks]
        for a, b in zip(contents, contents[1:]):
            # b 가 a 를 한 칸 민 것과 동일하면 1자 슬라이딩 버그다.
            assert not (len(a) == len(b) and a[1:] == b[:-1])


class TestSplitMarkdown:
    """Markdown 분할 테스트"""
    
    def test_split_by_headers(self) -> None:
        """헤더별 분할"""
        text = """# Title

Content for title.

## Section 1

Content for section 1.

## Section 2

Content for section 2.
"""
        chunks = split_markdown(text, chunk_size=1000)
        
        # 최소 3개 청크 (Title, Section 1, Section 2)
        assert len(chunks) >= 3
    
    def test_header_path_metadata(self) -> None:
        """헤더 경로 메타데이터 확인"""
        text = """# Main Title

## Sub Section

Content here.
"""
        chunks = split_markdown(text, chunk_size=1000)
        
        # 마지막 청크에 헤더 경로가 있어야 함
        last_chunk = chunks[-1]
        assert "header_path" in last_chunk.metadata
    
    def test_large_section_gets_split(self) -> None:
        """큰 섹션은 추가 분할"""
        text = "# Title\n\n" + "A" * 500
        chunks = split_markdown(text, chunk_size=100, chunk_overlap=0)
        
        # 500자 내용이 100자 청크로 분할되어야 함
        assert len(chunks) > 1
    
    def test_text_before_first_header(self) -> None:
        """첫 헤더 이전 텍스트 처리"""
        text = """Some intro text.

# First Header

Content.
"""
        chunks = split_markdown(text, chunk_size=1000)
        
        # 첫 청크가 intro 텍스트여야 함
        assert "intro" in chunks[0].content.lower()


class TestChunkDocument:
    """chunk_document 통합 테스트"""
    
    def test_chunk_txt_document(self) -> None:
        """txt 문서 청킹"""
        doc = Document(
            content="This is a test document with some content.",
            metadata={"source": "/test/file.txt", "extension": ".txt"}
        )
        
        chunks = chunk_document(doc)
        
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "/test/file.txt"
    
    def test_chunk_md_document(self) -> None:
        """md 문서 청킹 (Markdown 분할기 사용)"""
        doc = Document(
            content="# Header\n\nContent here.",
            metadata={"source": "/test/file.md", "extension": ".md"}
        )
        
        chunks = chunk_document(doc)
        
        assert len(chunks) >= 1
