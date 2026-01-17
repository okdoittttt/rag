"""Ask 명령어

질문에 대해 답변을 생성합니다.
"""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from rag.config import get_config
from rag.embedding import Embedder, get_vector_store
from rag.generation import build_prompt, GeminiLLM
from rag.retrieval import HybridSearcher
from rag.retrieval.reranker import Reranker

from rag.logger import setup_logging

console = Console()

# Reranker 인스턴스 (지연 로딩)
_reranker: Optional[Reranker] = None


def get_reranker(model_name: str) -> Reranker:
    """Reranker 싱글톤 인스턴스 반환"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker(model_name=model_name)
    return _reranker


def handle_ask(
    query: Annotated[str, typer.Argument(help="질문 내용")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="참조할 청크 개수")] = 5,
    show_context: Annotated[bool, typer.Option("--show-context", "-s", help="참조된 컨텍스트 표시")] = False,
    rerank: Annotated[bool, typer.Option("--rerank", "-r", help="Reranker로 결과 재정렬")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="상세 로그 출력")] = False,
):
    """질문에 대해 답변합니다."""
    if verbose:
        setup_logging(log_level="DEBUG")

    config = get_config()
    index_path = Path(config.project.index_path)
    
    if not index_path.exists():
        console.print("[red]Error:[/red] Index not found. Please run 'rag index' first.")
        raise typer.Exit(code=1)

    # Reranker 사용 여부 (CLI 옵션 또는 config)
    use_reranker = rerank or config.retrieval.use_reranker

    with console.status("[bold green]Thinking...[/bold green]"):
        # 1. 인덱스 로드
        embedder = Embedder()
        store = get_vector_store(config)
        searcher = HybridSearcher(embedder, store)
        
        try:
            searcher.load(index_path)
        except Exception as e:
            console.print(f"[red]Error loading index: {e}[/red]")
            raise typer.Exit(code=1)

        # 2. 검색 (Reranker 사용 시 더 많은 후보 검색)
        search_top_k = top_k * 3 if use_reranker else top_k
        results = searcher.search(query, top_k=search_top_k)
        
        # 3. Reranking (선택적)
        if use_reranker and results:
            console.print("[dim]Reranking...[/dim]")
            reranker = get_reranker(config.retrieval.reranker_model)
            results = reranker.rerank(query, results, top_k=top_k)
        
        chunks = [r[0] for r in results]
        
        if not chunks:
            console.print("[yellow]No relevant context found.[/yellow]")
            return

        # 4. 프롬프트 구성 및 LLM 호출
        prompt = build_prompt(query, chunks)
        
        try:
            llm = GeminiLLM()
            answer = llm.generate(prompt)
        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
            raise typer.Exit(code=1)

    # 5. 결과 출력
    console.print("\n[bold]Answer:[/bold]")
    console.print(Markdown(answer))
    
    # 컨텍스트 표시 (옵션)
    if show_context:
        console.print("\n[bold]References:[/bold]")
        for i, (chunk, score) in enumerate(results, 1):
            source = chunk.metadata.get("filename", "unknown")
            panel = Panel(
                chunk.content[:500] + ("..." if len(chunk.content) > 500 else ""),
                title=f"[{i}] {source} (Score: {score:.4f})",
            )
            console.print(panel)
