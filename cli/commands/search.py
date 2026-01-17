"""Search 명령어

답변 생성 없이 검색 결과만 확인합니다. (디버깅용)
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag.config import get_config
from rag.embedding import Embedder, VectorStore
from rag.retrieval import HybridSearcher


from rag.logger import setup_logging

console = Console()


def handle_search(
    query_text: Annotated[str, typer.Argument(help="검색어")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="반환 개수")] = 5,
    alpha: Annotated[float, typer.Option("--alpha", "-a", help="하이브리드 가중치 (0.0: BM25 ~ 1.0: Vector)")] = 0.5,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="상세 로그 출력")] = False,
):
    """검색 결과를 테이블로 보여줍니다."""
    if verbose:
        setup_logging(log_level="DEBUG")

    config = get_config()
    index_path = Path(config.project.index_path)
    
    if not index_path.exists():
        console.print("[red]Error:[/red] Index not found. Please run 'rag index' first.")
        raise typer.Exit(code=1)

    searcher = HybridSearcher(Embedder(), VectorStore(config.embedding.dimension))
    try:
        searcher.load(index_path)
    except Exception as e:
        console.print(f"[red]Index Load Error: {e}[/red]")
        raise typer.Exit(code=1)
        
    results = searcher.search(query_text, top_k=top_k, alpha=alpha)
    
    # 결과 테이블 출력
    table = Table(title=f"Search Results (Query: {query_text})")
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Source", style="green")
    table.add_column("Content (Snippet)", style="white")
    
    for i, (chunk, score) in enumerate(results, 1):
        content_snippet = chunk.content[:100].replace("\n", " ") + "..."
        source = chunk.metadata.get("filename", "unknown")
        table.add_row(str(i), f"{score:.4f}", source, content_snippet)
        
    console.print(table)
