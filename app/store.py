import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.bm25 import BM25Index
from app.chunking import Chunk, Document
from app.config import Settings


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    law TEXT NOT NULL,
    source_url TEXT NOT NULL,
    edition TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    law TEXT NOT NULL,
    article TEXT NOT NULL,
    article_key TEXT NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    content TEXT NOT NULL,
    search_text TEXT NOT NULL,
    embedding vector({dim}) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('portuguese', search_text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_article_key ON chunks (article_key);
"""


class PostgresStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._rows: list[dict] | None = None

    def connect(self):
        connection = psycopg.connect(
            self.settings.database_url,
            row_factory=dict_row,
            prepare_threshold=None,
            connect_timeout=10,
        )
        self._ensure_vector_extension(connection)
        register_vector(connection)
        return connection

    def _ensure_vector_extension(self, connection) -> None:
        # register_vector consulta pg_type. A extensão precisa existir e estar
        # confirmada nesta conexão antes dessa consulta, senão um banco recém
        # criado (como o rag_test do CI) quebra com "vector type not found".
        present = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector') AS present"
        ).fetchone()
        if present and present["present"]:
            return
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.commit()

    def migrate(self) -> None:
        statements = [item.strip() for item in SCHEMA.format(dim=self.settings.embedding_dim).split(";") if item.strip()]
        with self.connect() as connection:
            for statement in statements:
                connection.execute(statement)
            connection.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)")

    def clear(self) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
        self._rows = None

    def replace_all(self, documents: list[Document], chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("A quantidade de embeddings difere da quantidade de trechos.")
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            for document in documents:
                connection.execute(
                    """
                    INSERT INTO documents (id, title, law, source_url, edition)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (document.id, document.title, document.law, document.source_url, document.edition),
                )
            for chunk, vector in zip(chunks, vectors):
                connection.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, law, article, article_key, title, source_url, content, search_text, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.law,
                        chunk.article,
                        chunk.article_key,
                        chunk.title,
                        chunk.source_url,
                        chunk.content,
                        chunk.search_text,
                        Vector(vector),
                    ),
                )
        self._rows = None

    def count(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM chunks").fetchone()
        return int(row["total"])

    def vector_search(self, vector: list[float], k: int) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, document_id, law, article, article_key, title, source_url, content,
                       1 - (embedding <=> %s) AS score
                FROM chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (Vector(vector), Vector(vector), k),
            ).fetchall()
        return [_hit(row, row["score"], {"vector": float(row["score"])}) for row in rows]

    def fts_ids(self, query: str, limit: int) -> list[str]:
        try:
            with self.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT id
                    FROM chunks
                    WHERE tsv @@ plainto_tsquery('portuguese', %s)
                    ORDER BY ts_rank_cd(tsv, plainto_tsquery('portuguese', %s)) DESC
                    LIMIT %s
                    """,
                    (query, query, limit),
                ).fetchall()
        except psycopg.Error:
            return []
        return [row["id"] for row in rows]

    def lexical_search(self, query: str, k: int) -> list[dict]:
        rows = self._all_rows()
        if not rows:
            return []
        pool = rows
        if len(rows) > self.settings.bm25_full_scan_limit:
            allowed = set(self.fts_ids(query, self.settings.lexical_pool))
            chosen = [row for row in rows if row["id"] in allowed]
            pool = chosen or rows
        index = BM25Index.from_texts(
            [row["search_text"] for row in pool],
            k1=self.settings.bm25_k1,
            b=self.settings.bm25_b,
        )
        scores = index.scores(query)
        order = sorted(range(len(pool)), key=lambda position: scores[position], reverse=True)
        hits = []
        for position in order[:k]:
            if scores[position] <= 0:
                continue
            hits.append(_hit(pool[position], scores[position], {"bm25": scores[position]}))
        return hits

    def _all_rows(self) -> list[dict]:
        if self._rows is None:
            with self.connect() as connection:
                self._rows = connection.execute(
                    """
                    SELECT id, document_id, law, article, article_key, title, source_url, content, search_text
                    FROM chunks
                    """
                ).fetchall()
        return self._rows


def _hit(row: dict, score: float, detail: dict) -> dict:
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "law": row["law"],
        "article": row["article"],
        "article_key": row["article_key"],
        "title": row["title"],
        "source_url": row["source_url"],
        "content": row["content"],
        "score": float(score),
        "detail": {key: float(value) for key, value in detail.items()},
    }
