"""Lê data/corpus, fatia os artigos e grava embeddings no Postgres."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunking import load_chunks
from app.config import get_settings
from app.embeddings import get_embedder
from app.store import PostgresStore


def main() -> None:
    settings = get_settings()
    documents, chunks = load_chunks(ROOT / "data" / "corpus", settings)
    if not chunks:
        raise SystemExit("Nenhum trecho encontrado em data/corpus.")
    print(f"Documentos: {len(documents)}. Trechos: {len(chunks)}. Embeddings: {settings.embedding_provider}.")
    embedder = get_embedder(settings)
    vectors = embedder.embed_documents([chunk.search_text for chunk in chunks])
    store = PostgresStore(settings)
    store.migrate()
    store.replace_all(documents, chunks, vectors)
    print(f"Ingestão concluída. Trechos no banco: {store.count()}. Modelo: {embedder.name}.")


if __name__ == "__main__":
    main()
