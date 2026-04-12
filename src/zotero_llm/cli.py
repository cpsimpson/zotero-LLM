from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from .pipeline import answer_with_context, ingest_pdfs, semantic_search

app = typer.Typer(help="Local semantic PDF search for Zotero storage.")
console = Console()

# Prefer env override; otherwise use a generic home-relative Zotero default.
DEFAULT_ZOTERO_STORAGE = Path(
    os.environ.get("ZOTERO_STORAGE_DIR", str(Path.home() / "Zotero" / "storage"))
)
DEFAULT_PARSED = Path("./parsed-pdfs")
DEFAULT_QDRANT = Path("./qdrant-data")
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "zotero_pdf_chunks"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _run_or_exit(fn):
    try:
        return fn()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def ingest(
    source: Path = typer.Option(DEFAULT_ZOTERO_STORAGE, help="Root folder containing PDFs."),
    parsed_out: Path = typer.Option(DEFAULT_PARSED, help="Where parsed .txt files are written."),
    qdrant_path: Path = typer.Option(DEFAULT_QDRANT, help="Local path for embedded Qdrant data."),
    qdrant_url: str | None = typer.Option(
        DEFAULT_QDRANT_URL,
        help="Qdrant server URL (e.g. http://localhost:6333).",
    ),
    collection: str = typer.Option(DEFAULT_COLLECTION, help="Qdrant collection name."),
    embedding_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Ollama embedding model."),
    ollama_host: str = typer.Option(DEFAULT_OLLAMA_HOST, help="Ollama base URL."),
) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        pdf_task = progress.add_task("Discovering PDFs", total=1, completed=0)
        delete_task = progress.add_task("Deleting stale docs", total=1, completed=1, visible=False)

        def on_progress(event: str, current: int, total: int, path: Path | None) -> None:
            label = path.name if path else ""
            if event == "start":
                progress.update(pdf_task, total=max(total, 1), completed=0, description="Scanning PDFs")
            elif event == "parsing":
                progress.update(pdf_task, description=f"Parsing {label}")
            elif event == "indexed":
                progress.update(pdf_task, completed=current, total=max(total, 1), description=f"Indexed {label}")
            elif event == "skipped":
                progress.update(pdf_task, completed=current, total=max(total, 1), description=f"Skipped {label}")
            elif event == "failed":
                progress.update(pdf_task, completed=current, total=max(total, 1), description=f"Failed {label}")
            elif event == "deleting_start":
                progress.update(delete_task, visible=True, total=max(total, 1), completed=0, description="Deleting stale docs")
            elif event == "deleting":
                progress.update(delete_task, completed=current, total=max(total, 1), description=f"Deleting {label}")
            elif event == "done":
                progress.update(pdf_task, completed=max(total, 1), total=max(total, 1), description="Ingest complete")
                progress.update(delete_task, visible=False)

        stats = _run_or_exit(
            lambda: ingest_pdfs(
                source_root=source,
                parsed_text_root=parsed_out,
                qdrant_path=qdrant_path,
                collection_name=collection,
                embedding_model=embedding_model,
                ollama_host=ollama_host,
                qdrant_url=qdrant_url,
                progress_callback=on_progress,
            )
        )
    console.print(
        "Seen {seen} PDFs | updated {updated} | skipped {skipped} | deleted {deleted} | indexed chunks {chunks} "
        "into '{collection}'.".format(
            seen=stats.total_pdfs_seen,
            updated=stats.processed_docs,
            skipped=stats.skipped_docs,
            deleted=stats.deleted_docs,
            chunks=stats.indexed_chunks,
            collection=collection,
        )
    )


def _print_results(query: str, results: list, limit: int) -> None:
    table = Table(title=f"Semantic results for: {query}")
    table.add_column("#")
    table.add_column("Score")
    table.add_column("Title")
    table.add_column("DOI")
    table.add_column("PDF")
    table.add_column("Preview")

    for i, result in enumerate(results[:limit], start=1):
        preview = result.chunk_text.replace("\n", " ")[:180]
        table.add_row(
            str(i),
            f"{result.score:.4f}",
            result.title,
            result.doi or "",
            result.pdf_path,
            preview,
        )
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Text query."),
    qdrant_path: Path = typer.Option(DEFAULT_QDRANT, help="Local path for embedded Qdrant data."),
    qdrant_url: str | None = typer.Option(
        DEFAULT_QDRANT_URL,
        help="Qdrant server URL (e.g. http://localhost:6333).",
    ),
    collection: str = typer.Option(DEFAULT_COLLECTION, help="Qdrant collection name."),
    embedding_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Ollama embedding model."),
    ollama_host: str = typer.Option(DEFAULT_OLLAMA_HOST, help="Ollama base URL."),
    limit: int = typer.Option(8, min=1, max=50, help="Number of semantic matches."),
) -> None:
    results = _run_or_exit(
        lambda: semantic_search(
            query,
            qdrant_path=qdrant_path,
            collection_name=collection,
            embedding_model=embedding_model,
            ollama_host=ollama_host,
            qdrant_url=qdrant_url,
            limit=limit,
        )
    )
    _print_results(query, results, limit)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from indexed PDFs."),
    qdrant_path: Path = typer.Option(DEFAULT_QDRANT, help="Local path for embedded Qdrant data."),
    qdrant_url: str | None = typer.Option(
        DEFAULT_QDRANT_URL,
        help="Qdrant server URL (e.g. http://localhost:6333).",
    ),
    collection: str = typer.Option(DEFAULT_COLLECTION, help="Qdrant collection name."),
    embedding_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Ollama embedding model."),
    chat_model: str = typer.Option(DEFAULT_CHAT_MODEL, help="Ollama chat model."),
    ollama_host: str = typer.Option(DEFAULT_OLLAMA_HOST, help="Ollama base URL."),
    limit: int = typer.Option(6, min=1, max=30, help="Context chunks to include."),
) -> None:
    results = _run_or_exit(
        lambda: semantic_search(
            question,
            qdrant_path=qdrant_path,
            collection_name=collection,
            embedding_model=embedding_model,
            ollama_host=ollama_host,
            qdrant_url=qdrant_url,
            limit=limit,
        )
    )
    if not results:
        console.print("No results found.")
        raise typer.Exit(code=1)

    answer = _run_or_exit(
        lambda: answer_with_context(
            question,
            context_results=results,
            chat_model=chat_model,
            ollama_host=ollama_host,
        )
    )
    console.print("\n[bold]Answer[/bold]")
    console.print(answer)
    console.print("\n[bold]Sources[/bold]")
    for idx, r in enumerate(results, start=1):
        console.print(f"[{idx}] {r.title} | DOI: {r.doi or 'N/A'} | {r.pdf_path}")


@app.command()
def shell(
    qdrant_path: Path = typer.Option(DEFAULT_QDRANT, help="Local path for embedded Qdrant data."),
    qdrant_url: str | None = typer.Option(
        DEFAULT_QDRANT_URL,
        help="Qdrant server URL (e.g. http://localhost:6333).",
    ),
    collection: str = typer.Option(DEFAULT_COLLECTION, help="Qdrant collection name."),
    embedding_model: str = typer.Option(DEFAULT_EMBED_MODEL, help="Ollama embedding model."),
    chat_model: str = typer.Option(DEFAULT_CHAT_MODEL, help="Ollama chat model."),
    ollama_host: str = typer.Option(DEFAULT_OLLAMA_HOST, help="Ollama base URL."),
    limit: int = typer.Option(6, min=1, max=30, help="Context chunks per question."),
) -> None:
    console.print("Interactive mode. Type /quit to exit.")
    while True:
        query = typer.prompt("zotero-llm")
        if query.strip().lower() in {"/quit", "quit", "exit"}:
            break
        results = _run_or_exit(
            lambda: semantic_search(
                query,
                qdrant_path=qdrant_path,
                collection_name=collection,
                embedding_model=embedding_model,
                ollama_host=ollama_host,
                qdrant_url=qdrant_url,
                limit=limit,
            )
        )
        if not results:
            console.print("No results.")
            continue
        _print_results(query, results, limit=min(limit, 5))
        answer = _run_or_exit(
            lambda: answer_with_context(
                query,
                context_results=results,
                chat_model=chat_model,
                ollama_host=ollama_host,
            )
        )
        console.print("\n[bold]Answer[/bold]")
        console.print(answer)
        console.print()


if __name__ == "__main__":
    app()
