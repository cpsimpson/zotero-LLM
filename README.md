# zotero-llm

Local semantic search and Q&A over PDFs in your Zotero storage using:

- `liteparse` for PDF parsing (with OCR support)
- local `ollama` embeddings/chat models
- local `qdrant` vector database

## 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

`liteparse` relies on the Node CLI. If you do not already have it:

```bash
npm install -g @llamaindex/liteparse
```

## 2) Start Ollama and pull models

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.2
```

## 3) Ingest Zotero PDFs

By default this scans recursively in `/Users/carolinesimpson/Zotero/storage`.

```bash
zotero-llm ingest
```

Output text files are written to `./parsed-pdfs` and vectors are stored in `./qdrant-data`.
This command is incremental: it only re-parses/re-embeds new or modified PDFs, skips unchanged files, and removes vectors for deleted PDFs.

## 4) Search

```bash
zotero-llm search "transformer interpretability for medical imaging"
```

Each result includes score, title, DOI (if found), and original PDF path.

## 5) Ask questions

```bash
zotero-llm ask "Which papers discuss retrieval-augmented generation benchmarks?"
```

This retrieves top semantic chunks, asks a local Ollama chat model to answer from those chunks, and prints sources.

## Optional: interactive shell

```bash
zotero-llm shell
```

## Useful overrides

```bash
zotero-llm ingest \
  --source /Users/carolinesimpson/Zotero/storage \
  --parsed-out ./parsed-pdfs \
  --qdrant-path ./qdrant-data \
  --collection zotero_pdf_chunks \
  --embedding-model nomic-embed-text \
  --ollama-host http://localhost:11434
```
