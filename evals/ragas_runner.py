"""RAGAS 평가 러너.

골든셋(JSONL)을 로드해 현재 RAG 시스템에 질의하고, 결과를 CSV로
저장한다. 두 단계로 메트릭을 계산한다:

1. **검색 단계 메트릭** (`evals.metrics`): LLM 호출 없이 ``expected_chunk_ids``/
   ``expected_sources`` 비교로 ``recall@k``, ``mrr``, ``source_recall`` 산출.
2. **RAGAS 메트릭** (선택, ``--with-ragas``): LLM-as-judge로
   ``faithfulness``, ``answer_relevancy``, ``context_precision``,
   ``context_recall`` 산출. 비용이 크므로 기본 비활성.

Examples:
    빠른 회귀 체크(검색 메트릭만)::

        $ uv run python -m evals.ragas_runner \\
            --golden evals/golden_set.jsonl \\
            --out evals/history/quick.csv

    답변 생성 + RAGAS 평가까지::

        $ uv run python -m evals.ragas_runner \\
            --golden evals/golden_set.jsonl \\
            --out evals/history/full.csv \\
            --with-answer --with-ragas --rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from api.schemas import AskRequest
from evals.metrics import compute_search_metrics


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """JSONL 골든셋을 dict 리스트로 로드한다.

    Args:
        path: ``evals/golden_set.jsonl`` 파일 경로.

    Returns:
        각 케이스의 dict 리스트. 빈 줄과 ``#``으로 시작하는 주석 줄은 무시.

    Raises:
        FileNotFoundError: ``path`` 가 존재하지 않을 때.
        ValueError: JSONL 파싱 실패 시 줄 번호 포함 메시지.
    """
    if not path.exists():
        raise FileNotFoundError(f"golden set not found: {path}")

    cases: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON parse error at {path}:{lineno}: {exc.msg}"
                ) from exc
    return cases


def run_one_case(
    case: dict[str, Any],
    *,
    provider: str | None = None,
    top_k: int = 5,
    rerank: bool = False,
    expand: bool = False,
    with_answer: bool = False,
) -> dict[str, Any]:
    """단일 골든셋 케이스를 실행하여 평가용 레코드를 만든다.

    Args:
        case: 골든셋 한 줄(dict). ``question`` 필수, ``ground_truth``/
            ``expected_chunk_ids``/``expected_sources``/``category``/``id``는
            선택.
        provider: 사용할 LLM provider. ``None``이면 설정 기본값.
        top_k: 검색 결과 상위 ``k`` 개수.
        rerank: Reranker 활성 여부.
        expand: Query Rewriting 활성 여부.
        with_answer: ``True``면 LLM으로 답변까지 생성. 검색 메트릭만
            필요하면 ``False``로 두어 토큰 비용을 절감한다.

    Returns:
        다음 키를 포함하는 dict:

        - ``id``, ``category``, ``question``, ``ground_truth``
        - ``contexts`` (list[str]): 검색된 청크 본문
        - ``retrieved_chunk_ids`` (list[int]): 청크 인덱스
        - ``retrieved_sources`` (list[str]): 출처 파일명
        - ``answer`` (str): ``with_answer=True``일 때만 채워짐, 아니면 빈 문자열
        - ``latency_search_ms`` / ``latency_answer_ms``: 단계별 측정값
        - 검색 메트릭(``recall_at_*``, ``mrr``, ``source_recall``)
    """
    # 지연 import: 평가 모듈 단독으로 import해도 무거운 의존성을 끌어들이지 않도록.
    from api.routes.ask import _search_documents

    request = AskRequest(
        query=case["question"],
        top_k=top_k,
        rerank=rerank,
        expand=expand,
        provider=provider,
        user_id=case.get("user_id"),
        source_filter=case.get("source_filter"),
    )

    t0 = time.perf_counter()
    chunks, scored = _search_documents(request)
    latency_search_ms = (time.perf_counter() - t0) * 1000.0

    contexts = [c.content for c in chunks]
    retrieved_chunk_ids = [c.metadata.get("chunk_index") for c in chunks]
    retrieved_sources = [c.metadata.get("source") for c in chunks]
    retrieved_scores = [float(s) for _, s in scored]

    search_metrics = compute_search_metrics(
        retrieved=chunks,
        expected_chunk_ids=case.get("expected_chunk_ids"),
        expected_sources=case.get("expected_sources"),
    )

    answer = ""
    latency_answer_ms = 0.0
    if with_answer and chunks:
        from rag.generation import build_prompt, get_llm

        t1 = time.perf_counter()
        prompt = build_prompt(case["question"], chunks)
        llm = get_llm(provider=provider)
        answer = llm.generate(prompt)
        latency_answer_ms = (time.perf_counter() - t1) * 1000.0

    return {
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "question": case["question"],
        "ground_truth": case.get("ground_truth", ""),
        "contexts": contexts,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_sources": retrieved_sources,
        "retrieved_scores": retrieved_scores,
        "answer": answer,
        "latency_search_ms": latency_search_ms,
        "latency_answer_ms": latency_answer_ms,
        **search_metrics,
    }


def _to_ragas_dataset(records: list[dict[str, Any]]):
    """레코드 리스트를 RAGAS ``EvaluationDataset``으로 변환한다.

    RAGAS 0.2+ 의 ``SingleTurnSample`` 스키마(``user_input``,
    ``retrieved_contexts``, ``response``, ``reference``)에 맞춰 매핑한다.

    Args:
        records: ``run_one_case`` 반환값 리스트.

    Returns:
        ``ragas.EvaluationDataset`` 인스턴스.

    Raises:
        ImportError: ``ragas`` 가 설치되지 않은 경우.
    """
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=list(r["contexts"]),
            response=r["answer"],
            reference=r["ground_truth"],
        )
        for r in records
    ]
    return EvaluationDataset(samples=samples)


def run_ragas(records: list[dict[str, Any]]) -> list[dict[str, float]]:
    """레코드에 RAGAS 메트릭을 적용한다.

    LLM-as-judge 계열이므로 케이스당 4~6회 LLM 호출이 발생한다.
    ``GOOGLE_API_KEY`` 또는 ``OPENAI_API_KEY`` 등 RAGAS 기본 평가자
    설정에 필요한 환경변수를 호출 전에 설정해야 한다.

    Args:
        records: ``run_one_case`` 반환값 리스트. ``answer`` 가 비어 있으면
            ``faithfulness``/``answer_relevancy``는 의미가 없으므로 건너뛴다.

    Returns:
        케이스 순서와 동일한 길이의, 메트릭→점수 dict 리스트.

    Raises:
        ImportError: ``ragas`` 미설치 시.
    """
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = _to_ragas_dataset(records)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    df = result.to_pandas()
    score_cols = [c for c in df.columns if c not in {
        "user_input", "retrieved_contexts", "response", "reference",
    }]
    return [
        {col: float(row[col]) if row[col] is not None else float("nan") for col in score_cols}
        for _, row in df.iterrows()
    ]


def _write_csv(records: list[dict[str, Any]], out_path: Path) -> None:
    """레코드를 CSV로 저장한다.

    list/dict 컬럼은 JSON 문자열로 직렬화해 CSV 한 셀에 담는다.
    ``pandas`` 가 설치돼 있으면 사용하고, 없으면 표준 ``csv`` 모듈로
    fallback한다 — RAGAS 사용을 옵션 의존성으로 분리해도 빠른 회귀
    실행이 가능하도록 한다.

    Args:
        records: 저장할 레코드 리스트.
        out_path: 출력 CSV 경로. 부모 디렉터리는 자동 생성.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value

    flat = [{k: _serialize(v) for k, v in r.items()} for r in records]

    try:
        import pandas as pd  # type: ignore
        pd.DataFrame(flat).to_csv(out_path, index=False)
    except ImportError:
        import csv

        if not flat:
            out_path.write_text("", encoding="utf-8")
            return
        keys = list(flat[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat)


def evaluate_all(
    golden_path: Path,
    out_path: Path,
    *,
    provider: str | None = None,
    top_k: int = 5,
    rerank: bool = False,
    expand: bool = False,
    with_answer: bool = False,
    with_ragas: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """골든셋 전체를 실행해 메트릭을 산출하고 CSV로 저장한다.

    Args:
        golden_path: 골든셋 JSONL 경로.
        out_path: 결과 CSV 출력 경로.
        provider: LLM provider.
        top_k: 검색 상위 k.
        rerank: Reranker 사용 여부.
        expand: Query rewriting 사용 여부.
        with_answer: 답변 생성 여부(RAGAS 사용 시 필수).
        with_ragas: RAGAS 메트릭 계산 여부. ``True``인데 ``with_answer``가
            ``False``면 자동으로 ``with_answer=True``로 승격한다.
        limit: 디버그용 케이스 수 제한. ``None``이면 전체.

    Returns:
        저장된 레코드 리스트. 호출자가 추가 집계(예: 카테고리별 평균)에
        활용할 수 있다.
    """
    if with_ragas and not with_answer:
        with_answer = True  # RAGAS는 answer 없으면 무의미

    cases = load_golden_set(golden_path)
    if limit:
        cases = cases[:limit]

    records: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        record = run_one_case(
            case,
            provider=provider,
            top_k=top_k,
            rerank=rerank,
            expand=expand,
            with_answer=with_answer,
        )
        records.append(record)
        print(
            f"[{i}/{len(cases)}] {record['id'] or '-'} "
            f"recall@5={record.get('recall_at_5', 0):.2f} "
            f"mrr={record.get('mrr', 0):.2f}",
            file=sys.stderr,
        )

    if with_ragas:
        scores = run_ragas(records)
        for record, extra in zip(records, scores):
            record.update(extra)

    _write_csv(records, out_path)
    print(f"saved {len(records)} records → {out_path}", file=sys.stderr)
    return records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI 인자 파서.

    Args:
        argv: 인자 리스트. ``None``이면 ``sys.argv[1:]`` 사용.

    Returns:
        파싱된 ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        prog="evals.ragas_runner",
        description="Run RAG evaluation against the golden set.",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("evals/golden_set.jsonl"),
        help="골든셋 JSONL 경로 (기본: evals/golden_set.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="결과 CSV 경로 (기본: evals/history/{timestamp}.csv)",
    )
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--expand", action="store_true")
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="LLM으로 답변 생성까지 수행 (RAGAS 사용 시 자동 활성)",
    )
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="RAGAS 메트릭(faithfulness 등) 계산 — 비용 큼",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="디버그용 케이스 수 제한",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 엔트리포인트.

    Args:
        argv: 인자 리스트. ``None``이면 ``sys.argv``.

    Returns:
        프로세스 종료 코드. 성공 시 ``0``.
    """
    args = _parse_args(argv)

    out_path = args.out
    if out_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path("evals/history") / f"{stamp}.csv"

    evaluate_all(
        golden_path=args.golden,
        out_path=out_path,
        provider=args.provider,
        top_k=args.top_k,
        rerank=args.rerank,
        expand=args.expand,
        with_answer=args.with_answer,
        with_ragas=args.with_ragas,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
