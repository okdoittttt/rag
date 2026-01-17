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
from rag.embedding import Embedder, get_vector_store
from rag.ingestion import load_documents, load_file
from rag.ingestion.metadata import IndexMetadata
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

    # 벡터 스토어 초기화 (reset 처리를 위해 먼저 생성)
    embedder = Embedder()
    store = get_vector_store(config)
    
    # 인덱스 메타데이터 (파일 해시 추적)
    index_meta = IndexMetadata(index_path)
    
    # 초기화 옵션 처리
    if reset:
        # 로컬 인덱스(FAISS 메타데이터 등) 삭제
        if index_path.exists():
            console.print(f"[yellow]Removing local index at {index_path}...[/yellow]")
            shutil.rmtree(index_path)
        
        # Qdrant 컬렉션 초기화 (Qdrant 사용 시)
        if config.embedding.store_type == "qdrant":
            console.print(f"[yellow]Clearing Qdrant collection '{config.embedding.qdrant.collection}'...[/yellow]")
            store.clear()
        
        # 메타데이터 초기화
        index_meta.clear()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # 1. 문서 로드 및 변경 감지
        task_load = progress.add_task("Loading documents...", total=None)
        
        # 지원 확장자 가져오기
        extensions = config.ingestion.supported_extensions
        
        # 파일 목록 수집
        path_obj = Path(path)
        if path_obj.is_file():
            all_files = [path_obj] if path_obj.suffix.lower() in extensions else []
        else:
            all_files = [f for f in path_obj.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
        
        # 변경된 파일만 필터링 (reset이 아닌 경우)
        if not reset:
            new_or_changed_files = [f for f in all_files if not index_meta.is_indexed(f)]
            skipped_count = len(all_files) - len(new_or_changed_files)
            if skipped_count > 0:
                console.print(f"[blue]Skipping {skipped_count} unchanged files[/blue]")
        else:
            new_or_changed_files = all_files
        
        # 변경된 파일만 로드
        docs = []
        for file_path in new_or_changed_files:
            doc = load_file(file_path)
            if doc:
                doc.metadata["_file_path"] = file_path  # 해시 추적용
                docs.append(doc)
        
        progress.update(task_load, completed=1, description=f"Loaded {len(docs)} documents")
        
        if not docs:
            if len(all_files) > 0:
                console.print("[green]All documents are already indexed. No changes detected.[/green]")
            else:
                console.print("[yellow]No documents found.[/yellow]")
            return

        # 2. 청킹
        task_chunk = progress.add_task("Chunking documents...", total=len(docs))
        all_chunks = []
        doc_chunk_counts: dict[Path, int] = {}  # 문서별 청크 수 추적
        
        for doc in docs:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
            file_path = doc.metadata.get("_file_path")
            if file_path:
                doc_chunk_counts[file_path] = len(chunks)
            progress.advance(task_chunk)
            
        progress.update(task_chunk, description=f"Created {len(all_chunks)} chunks")

        # 3. 임베딩 및 인덱싱
        task_embed = progress.add_task("Embedding and Indexing...", total=None)
        
        searcher = HybridSearcher(embedder, store)
        
        # 기존 인덱스가 있으면 로드 (reset이 아닐 때만)
        if not reset and index_path.exists():
            try:
                searcher.load(index_path)
                console.print(f"[blue]Loaded existing index with {searcher.vector_store.total_chunks} chunks[/blue]")
            except Exception as e:
                console.print(f"[red]Failed to load existing index: {e}[/red]")

        # 인덱스 추가 (임베딩은 내부에서 수행됨)
        searcher.index(all_chunks)
        searcher.save(index_path)
        
        # 인덱싱된 파일 메타데이터 업데이트
        for file_path, chunk_count in doc_chunk_counts.items():
            index_meta.mark_indexed(file_path, chunk_count)
        index_meta.save()
        
        progress.update(task_embed, completed=1, description="Indexing complete")

    console.print(f"\n[green]Successfully indexed {len(all_chunks)} chunks from {len(docs)} documents![/green]")
    console.print(f"Index saved to: [bold]{index_path}[/bold]")

