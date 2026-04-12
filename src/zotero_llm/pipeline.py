from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import ollama
from liteparse import LiteParse, ParseError
from qdrant_client import QdrantClient
from qdrant_client.http import models

DOI_PATTERN = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk_text: str
    pdf_path: str
    text_path: str
    title: str
    doi: str | None
    zotero_key: str | None
    chunk_index: int


@dataclass(frozen=True)
class IngestStats:
    total_pdfs_seen: int
    processed_docs: int
    skipped_docs: int
    deleted_docs: int
    indexed_chunks: int


def discover_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def _safe_text_name(pdf_path: Path, source_root: Path) -> Path:
    relative = pdf_path.relative_to(source_root)
    return relative.with_suffix(".txt")


def _extract_title(text: str) -> str:
    for line in text.splitlines()[:30]:
        candidate = " ".join(line.split()).strip()
        if len(candidate) < 15:
            continue
        if candidate.isupper() and len(candidate.split()) <= 2:
            continue
        return candidate[:300]
    return "Untitled"


def _extract_doi(text: str) -> str | None:
    match = DOI_PATTERN.search(text[:15000])
    if not match:
        return None
    return match.group(1).rstrip(".,);")


def _doc_id(pdf_path: Path) -> str:
    return hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()


def _split_into_chunks(text: str, chunk_chars: int = 1400, overlap_chars: int = 250) -> list[str]:
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + chunk_chars, n)
        if end < n:
            split_point = cleaned.rfind(" ", start, end)
            if split_point > start + 200:
                end = split_point
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return chunks


def _ensure_collection(
    qdrant: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if qdrant.collection_exists(collection_name):
        info = qdrant.get_collection(collection_name)
        cfg = info.config.params.vectors
        if isinstance(cfg, dict):
            raise RuntimeError("Named vectors are not supported by this script.")
        existing_size = cfg.size if cfg else None
        if existing_size != vector_size:
            raise RuntimeError(
                f"Collection '{collection_name}' exists with vector size {existing_size}, "
                f"but embedding model returned size {vector_size}."
            )
        return

    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    for key in ("doc_id", "pdf_path", "title", "doi", "zotero_key"):
        qdrant.create_payload_index(
            collection_name=collection_name,
            field_name=key,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def _embed_texts(client: ollama.Client, model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = client.embed(model=model, input=texts)
    return [list(vec) for vec in response.embeddings]


def _zotero_key(pdf_path: Path, source_root: Path) -> str | None:
    try:
        rel = pdf_path.relative_to(source_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return None


def _state_file(qdrant_path: Path, collection_name: str) -> Path:
    safe_collection = re.sub(r"[^a-zA-Z0-9_.-]+", "_", collection_name)
    return qdrant_path / f"{safe_collection}.index_state.json"


def _qdrant_client(*, qdrant_path: Path, qdrant_url: str | None) -> QdrantClient:
    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    return QdrantClient(path=str(qdrant_path))


def _load_state(state_path: Path) -> dict[str, dict]:
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def _save_state(state_path: Path, state: dict[str, dict]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


def _file_sig(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (int(st.st_mtime_ns), int(st.st_size))


def ingest_pdfs(
    source_root: Path,
    parsed_text_root: Path,
    qdrant_path: Path,
    collection_name: str,
    embedding_model: str,
    ollama_host: str,
    qdrant_url: str | None = None,
    state_root: Path | None = None,
    batch_size: int = 16,
    progress_callback: Callable[[str, int, int, Path | None], None] | None = None,
) -> IngestStats:
    source_root = source_root.expanduser().resolve()
    parsed_text_root = parsed_text_root.expanduser().resolve()
    qdrant_path = qdrant_path.expanduser().resolve()
    parsed_text_root.mkdir(parents=True, exist_ok=True)
    if not qdrant_url:
        qdrant_path.mkdir(parents=True, exist_ok=True)
    if state_root is None:
        state_root = parsed_text_root / ".zotero-llm-state"
    state_root = state_root.expanduser().resolve()
    state_root.mkdir(parents=True, exist_ok=True)

    parser = LiteParse()
    ollama_client = ollama.Client(host=ollama_host)
    qdrant = _qdrant_client(qdrant_path=qdrant_path, qdrant_url=qdrant_url)
    state_path = _state_file(state_root, collection_name)
    state = _load_state(state_path)

    pdfs = discover_pdfs(source_root)
    if progress_callback:
        progress_callback("start", 0, len(pdfs), None)
    processed_docs = 0
    skipped_docs = 0
    deleted_docs = 0
    indexed_chunks = 0
    collection_ready = False

    # If state exists but collection is gone, force rebuild from scratch.
    if state and not qdrant.collection_exists(collection_name):
        state = {}

    current_pdf_paths = {str(p) for p in pdfs}
    stale_paths = [path for path in state.keys() if path not in current_pdf_paths]
    if stale_paths and qdrant.collection_exists(collection_name):
        if progress_callback:
            progress_callback("deleting_start", 0, len(stale_paths), None)
        for stale_pdf_path in stale_paths:
            if progress_callback:
                progress_callback(
                    "deleting",
                    deleted_docs + 1,
                    len(stale_paths),
                    Path(stale_pdf_path),
                )
            old = state.get(stale_pdf_path, {})
            doc_id = old.get("doc_id")
            if doc_id:
                qdrant.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="doc_id",
                                    match=models.MatchValue(value=str(doc_id)),
                                )
                            ]
                        )
                    ),
                )
            text_path_value = old.get("text_path")
            if isinstance(text_path_value, str):
                text_path = Path(text_path_value)
                if text_path.exists():
                    text_path.unlink(missing_ok=True)
            del state[stale_pdf_path]
            deleted_docs += 1

    for index, pdf in enumerate(pdfs, start=1):
        if progress_callback:
            progress_callback("parsing", index - 1, len(pdfs), pdf)
        mtime_ns, size = _file_sig(pdf)
        state_entry = state.get(str(pdf))
        if (
            state_entry
            and state_entry.get("mtime_ns") == mtime_ns
            and state_entry.get("size") == size
        ):
            skipped_docs += 1
            if progress_callback:
                progress_callback("skipped", index, len(pdfs), pdf)
            continue

        try:
            parsed = parser.parse(str(pdf), ocr_enabled=True)
        except (ParseError, FileNotFoundError, TimeoutError):
            if progress_callback:
                progress_callback("failed", index, len(pdfs), pdf)
            continue
        text = parsed.text.strip()
        if not text:
            if progress_callback:
                progress_callback("failed", index, len(pdfs), pdf)
            continue

        text_rel = _safe_text_name(pdf, source_root)
        text_path = parsed_text_root / text_rel
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")

        title = _extract_title(text)
        doi = _extract_doi(text)
        doc_id = _doc_id(pdf)
        key = _zotero_key(pdf, source_root)

        chunks = _split_into_chunks(text)
        if not chunks:
            continue

        vectors = _embed_texts(ollama_client, embedding_model, chunks)
        if not vectors:
            continue
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding response count does not match input chunk count.")

        if not collection_ready:
            _ensure_collection(qdrant, collection_name, len(vectors[0]))
            collection_ready = True

        qdrant.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )

        points: list[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))
            payload = {
                "doc_id": doc_id,
                "chunk_index": idx,
                "chunk_text": chunk,
                "pdf_path": str(pdf),
                "text_path": str(text_path),
                "title": title,
                "doi": doi,
                "zotero_key": key,
            }
            points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

        for start in range(0, len(points), batch_size):
            qdrant.upsert(collection_name=collection_name, points=points[start : start + batch_size])

        state[str(pdf)] = {
            "doc_id": doc_id,
            "mtime_ns": mtime_ns,
            "size": size,
            "text_path": str(text_path),
            "title": title,
            "doi": doi,
            "zotero_key": key,
            "chunks": len(points),
            "updated_at": int(time.time()),
        }

        processed_docs += 1
        indexed_chunks += len(points)
        if progress_callback:
            progress_callback("indexed", index, len(pdfs), pdf)

    _save_state(state_path, state)
    if progress_callback:
        progress_callback("done", len(pdfs), len(pdfs), None)
    return IngestStats(
        total_pdfs_seen=len(pdfs),
        processed_docs=processed_docs,
        skipped_docs=skipped_docs,
        deleted_docs=deleted_docs,
        indexed_chunks=indexed_chunks,
    )


def semantic_search(
    query: str,
    *,
    qdrant_path: Path,
    collection_name: str,
    embedding_model: str,
    ollama_host: str,
    qdrant_url: str | None = None,
    limit: int = 8,
) -> list[SearchResult]:
    client = ollama.Client(host=ollama_host)
    qdrant = _qdrant_client(
        qdrant_path=qdrant_path.expanduser().resolve(),
        qdrant_url=qdrant_url,
    )
    if not qdrant.collection_exists(collection_name):
        backend = qdrant_url or str(qdrant_path)
        raise RuntimeError(
            f"Collection '{collection_name}' not found at {backend}. "
            "Run `zotero-llm ingest` first."
        )

    query_vec = _embed_texts(client, embedding_model, [query])[0]
    response = qdrant.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    out: list[SearchResult] = []
    for p in response.points:
        payload = p.payload or {}
        out.append(
            SearchResult(
                score=float(p.score),
                chunk_text=str(payload.get("chunk_text", "")),
                pdf_path=str(payload.get("pdf_path", "")),
                text_path=str(payload.get("text_path", "")),
                title=str(payload.get("title", "Untitled")),
                doi=(str(payload["doi"]) if payload.get("doi") else None),
                zotero_key=(str(payload["zotero_key"]) if payload.get("zotero_key") else None),
                chunk_index=int(payload.get("chunk_index", -1)),
            )
        )
    return out


def answer_with_context(
    question: str,
    *,
    context_results: Iterable[SearchResult],
    chat_model: str,
    ollama_host: str,
) -> str:
    blocks: list[str] = []
    for i, r in enumerate(context_results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Source {i}]",
                    f"title: {r.title}",
                    f"doi: {r.doi or 'N/A'}",
                    f"pdf_path: {r.pdf_path}",
                    f"chunk_index: {r.chunk_index}",
                    f"text: {r.chunk_text}",
                ]
            )
        )
    context = "\n\n".join(blocks)
    prompt = (
        "You are answering questions about a local Zotero PDF collection.\n"
        "Use only the provided source snippets.\n"
        "If the answer is not in the snippets, say you could not find it.\n"
        "Always reference source numbers in your answer."
    )
    client = ollama.Client(host=ollama_host)
    response = client.chat(
        model=chat_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
    )
    return response.message.content.strip()
