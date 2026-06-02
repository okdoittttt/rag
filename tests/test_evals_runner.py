"""evals.ragas_runner 흐름 테스트.

LLM/검색기는 mock으로 대체하여 외부 의존(Qdrant, Gemini)을 만들지 않는다.
RAGAS 메트릭(``--with-ragas``)은 LLM 호출이 필요하므로 본 파일에서는
검증하지 않는다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evals import ragas_runner
from rag.chunking.chunk import Chunk


def _make_chunks() -> list[Chunk]:
    """검색기 mock이 반환할 청크 리스트.

    Returns:
        길이 2의 청크 리스트(인덱스 1, 7).
    """
    return [
        Chunk.create(
            content="hybrid 검색은 sparse와 dense를 결합한다.",
            source="intro.md",
            chunk_index=1,
            start_char=0,
            end_char=100,
        ),
        Chunk.create(
            content="BM25 점수는 키워드 빈도에서 유래한다.",
            source="bm25.md",
            chunk_index=7,
            start_char=0,
            end_char=80,
        ),
    ]


def _fake_search_documents(request: Any) -> tuple[list[Chunk], list[tuple[Chunk, float]]]:
    """``_search_documents`` 를 대체할 결정적 mock.

    Args:
        request: 무시되는 ``AskRequest``.

    Returns:
        ``(chunks, scored)`` 튜플. 점수는 내림차순.
    """
    chunks = _make_chunks()
    scored = [(chunks[0], 0.9), (chunks[1], 0.7)]
    return chunks, scored


class TestLoadGoldenSet:
    """``load_golden_set`` 의 입력 처리 검증."""

    def test_skips_comments_and_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "g.jsonl"
        path.write_text(
            "# header comment\n"
            "\n"
            '{"id": "q1", "question": "테스트?"}\n'
            "# trailing comment\n"
            '{"id": "q2", "question": "또?"}\n',
            encoding="utf-8",
        )
        cases = ragas_runner.load_golden_set(path)
        assert [c["id"] for c in cases] == ["q1", "q2"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ragas_runner.load_golden_set(tmp_path / "nope.jsonl")

    def test_invalid_json_raises_with_lineno(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": 1}\n{ broken json\n', encoding="utf-8")
        with pytest.raises(ValueError, match=":2"):
            ragas_runner.load_golden_set(path)


class TestRunOneCase:
    """``run_one_case`` 가 mock된 검색 결과로 메트릭을 합치는지 검증."""

    def test_search_only_record_has_metrics(self) -> None:
        case = {
            "id": "q1",
            "category": "single_fact",
            "question": "BM25는 무엇인가?",
            "ground_truth": "키워드 기반 sparse 검색.",
            "expected_chunk_ids": [1],
            "expected_sources": ["intro.md"],
        }
        with patch(
            "api.routes.ask._search_documents",
            side_effect=_fake_search_documents,
        ):
            record = ragas_runner.run_one_case(case, with_answer=False)

        assert record["id"] == "q1"
        assert record["category"] == "single_fact"
        assert record["retrieved_chunk_ids"] == [1, 7]
        assert record["retrieved_sources"] == ["intro.md", "bm25.md"]
        assert record["answer"] == ""
        # expected_chunk_ids=[1] 가 1위 등장 → recall@1 = mrr = 1.0
        assert record["recall_at_1"] == 1.0
        assert record["mrr"] == 1.0
        assert record["source_recall"] == 1.0
        assert record["latency_search_ms"] >= 0.0

    def test_with_answer_calls_llm(self) -> None:
        case = {"id": "q2", "question": "정답?"}

        class _FakeLLM:
            """``llm.generate`` 만 흉내내는 가짜 LLM."""

            def generate(self, prompt: str) -> str:
                return "FAKE_ANSWER"

        with patch(
            "api.routes.ask._search_documents",
            side_effect=_fake_search_documents,
        ), patch(
            "rag.generation.get_llm",
            return_value=_FakeLLM(),
        ), patch(
            "rag.generation.build_prompt",
            return_value="prompt-stub",
        ):
            record = ragas_runner.run_one_case(case, with_answer=True)

        assert record["answer"] == "FAKE_ANSWER"
        assert record["latency_answer_ms"] >= 0.0


class TestEvaluateAll:
    """``evaluate_all`` 의 CSV 저장과 집계 흐름 검증."""

    def test_writes_csv_and_returns_records(self, tmp_path: Path) -> None:
        golden = tmp_path / "g.jsonl"
        golden.write_text(
            '{"id": "q1", "category": "single_fact", "question": "Q1?", "expected_chunk_ids": [1]}\n'
            '{"id": "q2", "category": "single_fact", "question": "Q2?", "expected_chunk_ids": [7]}\n',
            encoding="utf-8",
        )
        out_csv = tmp_path / "out" / "result.csv"

        with patch(
            "api.routes.ask._search_documents",
            side_effect=_fake_search_documents,
        ):
            records = ragas_runner.evaluate_all(
                golden_path=golden,
                out_path=out_csv,
                with_answer=False,
                with_ragas=False,
            )

        assert len(records) == 2
        assert out_csv.exists()

        rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
        assert len(rows) == 2
        assert rows[0]["id"] == "q1"
        # 리스트 컬럼은 JSON 직렬화 보장
        assert json.loads(rows[0]["retrieved_chunk_ids"]) == [1, 7]

    def test_limit_truncates_cases(self, tmp_path: Path) -> None:
        golden = tmp_path / "g.jsonl"
        golden.write_text(
            "\n".join(
                json.dumps({"id": f"q{i}", "question": f"Q{i}?"})
                for i in range(5)
            ),
            encoding="utf-8",
        )
        out_csv = tmp_path / "small.csv"

        with patch(
            "api.routes.ask._search_documents",
            side_effect=_fake_search_documents,
        ):
            records = ragas_runner.evaluate_all(
                golden_path=golden,
                out_path=out_csv,
                limit=2,
            )

        assert len(records) == 2

    def test_with_ragas_promotes_with_answer(self, tmp_path: Path) -> None:
        """``with_ragas=True`` 이면 ``with_answer`` 가 자동 True 로 승격된다."""
        golden = tmp_path / "g.jsonl"
        golden.write_text(
            '{"id": "q1", "question": "Q?"}\n', encoding="utf-8"
        )
        out_csv = tmp_path / "x.csv"

        class _FakeLLM:
            def generate(self, prompt: str) -> str:
                return "ANS"

        with patch(
            "api.routes.ask._search_documents",
            side_effect=_fake_search_documents,
        ), patch(
            "rag.generation.get_llm",
            return_value=_FakeLLM(),
        ), patch(
            "rag.generation.build_prompt",
            return_value="p",
        ), patch(
            "evals.ragas_runner.run_ragas",
            return_value=[{"faithfulness": 0.8, "answer_relevancy": 0.7}],
        ):
            records = ragas_runner.evaluate_all(
                golden_path=golden,
                out_path=out_csv,
                with_answer=False,
                with_ragas=True,
            )

        assert records[0]["answer"] == "ANS"
        assert records[0]["faithfulness"] == 0.8


class TestCliArgs:
    """``_parse_args`` 기본값 검증."""

    def test_defaults(self) -> None:
        ns = ragas_runner._parse_args([])
        assert ns.top_k == 5
        assert ns.rerank is False
        assert ns.expand is False
        assert ns.with_answer is False
        assert ns.with_ragas is False
        assert ns.limit is None

    def test_overrides(self) -> None:
        ns = ragas_runner._parse_args(
            ["--rerank", "--expand", "--with-answer", "--with-ragas", "--top-k", "10"]
        )
        assert ns.rerank is True
        assert ns.expand is True
        assert ns.with_answer is True
        assert ns.with_ragas is True
        assert ns.top_k == 10
