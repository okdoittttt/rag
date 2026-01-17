"""Index 명령어

문서를 로드, 청킹, 임베딩하여 인덱스를 생성합니다.
"""

import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag.chunking import chunk_document
from rag.config import get_config
from rag.embedding import Embedder, VectorStore
from rag.ingestion import load_documents
from rag.retrieval import HybridSearcher


from rag.logger import setup_logging

console = Console()


def handle_index(
    path: Annotated[Path, typer.Argument(help="문서가 있는 경로 (파일 또는 폴더)")],
    reset: Annotated[bool, typer.Option("--reset", "-r", help="기존 인덱스 삭제 후 재생성")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="상세 로그 출력")] = False,
):
    """지정된 경로의 문서를 인덱싱합니다."""
    if verbose:
        setup_logging(log_level="DEBUG")

    config = get_config()
    index_path = Path(config.project.index_path)
    
    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(code=1)

    # 초기화 옵션 처리
    if reset and index_path.exists():
        console.print(f"[yellow]Removing existing index at {index_path}...[/yellow]")
        shutil.rmtree(index_path)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # 1. 문서 로드
        task_load = progress.add_task("Loading documents...", total=None)
        docs = load_documents(path)
        progress.update(task_load, completed=1, description=f"Loaded {len(docs)} documents")
        
        if not docs:
            console.print("[yellow]No documents found.[/yellow]")
            return

        # 2. 청킹
        task_chunk = progress.add_task("Chunking documents...", total=len(docs))
        all_chunks = []
        for doc in docs:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
            progress.advance(task_chunk)
            
        progress.update(task_chunk, description=f"Created {len(all_chunks)} chunks")

        # 3. 임베딩 및 인덱싱
        task_embed = progress.add_task("Embedding and Indexing...", total=None)
        
        embedder = Embedder()
        store = VectorStore(dimension=config.embedding.dimension)
        searcher = HybridSearcher(embedder, store)
        
        # 기존 인덱스가 있으면 로드
        if index_path.exists():
            try:
                searcher.load(index_path)
                console.print(f"[blue]Loaded existing index with {searcher.vector_store.total_chunks} chunks[/blue]")
            except Exception as e:
                console.print(f"[red]Failed to load existing index: {e}[/red]")
                # 로드 실패 시 새로 생성하거나 중단 선택 가능. 여기선 경고 후 진행.

        # 인덱스 추가 (임베딩은 내부에서 수행됨)
        searcher.index(all_chunks)
        searcher.save(index_path)
        
        progress.update(task_embed, completed=1, description="Indexing complete")

    console.print(f"\n[green]Successfully indexed {len(all_chunks)} chunks from {len(docs)} documents![/green]")
    console.print(f"Index saved to: [bold]{index_path}[/bold]")
