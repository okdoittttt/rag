"""Ask 명령어

사용자 질문에 대해 검색하고 LLM을 통해 답변을 생성합니다.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from rag.config import get_config
from rag.embedding import Embedder, get_vector_store
from rag.generation import GeminiLLM, build_prompt
from rag.retrieval import HybridSearcher


from rag.logger import setup_logging

console = Console()


def handle_ask(
    query: Annotated[str, typer.Argument(help="질문 내용")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="참조할 청크 개수")] = 5,
    show_context: Annotated[bool, typer.Option("--show-context", "-s", help="참조된 컨텍스트 표시")] = False,
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

    with console.status("[bold green]Thinking...[/bold green]"):
        # 1. 인덱스 로드
        embedder = Embedder()
        store = get_vector_store(config)  # 팩토리 함수 사용
        searcher = HybridSearcher(embedder, store)
        
        try:
            searcher.load(index_path)
        except Exception as e:
            console.print(f"[red]Error loading index: {e}[/red]")
            raise typer.Exit(code=1)

        # 2. 검색
        results = searcher.search(query, top_k=top_k)
        chunks = [r[0] for r in results]
        
        if not chunks:
            console.print("[yellow]No relevant context found.[/yellow]")
            return

        # 3. 프롬프트 구성 및 LLM 호출
        prompt = build_prompt(query, chunks)
        
        try:
            llm = GeminiLLM()
            answer = llm.generate(prompt)
        except Exception as e:
            console.print(f"[red]LLM Error: {e}[/red]")
            raise typer.Exit(code=1)

    # 4. 결과 출력
    console.print("\n[bold]Answer:[/bold]")
    console.print(Markdown(answer))
    
    # 컨텍스트 표시 (옵션)
    if show_context:
        console.print("\n[bold]References:[/bold]")
        for i, (chunk, score) in enumerate(results, 1):
            source = chunk.metadata.get("filename", "unknown")
            console.print(
                Panel(
                    chunk.content,
                    title=f"[{i}] {source} (Score: {score:.4f})",
                    border_style="blue"
                )
            )
