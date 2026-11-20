import psycopg
import pytest
from fastapi.testclient import TestClient

from app.chunking import Chunk, Document
from app.config import Settings, get_settings
from app.embeddings import HashEmbedding
from app.main import app
from app.retrieve import search
from app.runtime import get_runtime
from app.tracing import trace_question


def _database_available() -> bool:
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=3) as connection:
            connection.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _database_available(), reason="Postgres de teste indisponível")


@pytest.fixture
def ready_store():
    runtime = get_runtime()
    runtime.store.migrate()
    runtime.store.clear()
    return runtime.store


def _sample(store):
    documents = [
        Document("lgpd", "LGPD", "LGPD teste", "https://example.test/lgpd", "teste"),
        Document("cdc", "CDC", "CDC teste", "https://example.test/cdc", "teste"),
    ]
    chunks = [
        Chunk(
            id="lgpd-art-5-p1",
            document_id="lgpd",
            law="LGPD",
            article="Art. 5º",
            article_key="lgpd-art-5",
            title="LGPD teste",
            source_url="https://example.test/lgpd",
            content="LGPD. Art. 5º. Dado pessoal zetaunico é informação relacionada a pessoa natural identificada.",
            search_text="LGPD. LGPD teste. Art. 5º. Dado pessoal zetaunico é informação relacionada a pessoa natural identificada.",
        ),
        Chunk(
            id="cdc-art-2-p1",
            document_id="cdc",
            law="CDC",
            article="Art. 2º",
            article_key="cdc-art-2",
            title="CDC teste",
            source_url="https://example.test/cdc",
            content="CDC. Art. 2º. Consumidor é toda pessoa física ou jurídica que adquire produto como destinatário final.",
            search_text="CDC. CDC teste. Art. 2º. Consumidor é toda pessoa física ou jurídica que adquire produto como destinatário final.",
        ),
    ]
    embedder = HashEmbedding(get_settings().embedding_dim)
    store.replace_all(documents, chunks, embedder.embed_documents([chunk.search_text for chunk in chunks]))
    return embedder, chunks


def test_pgvector_returns_identical_embedding(ready_store):
    embedder, chunks = _sample(ready_store)
    hits = ready_store.vector_search(embedder.embed_query(chunks[0].search_text), 1)
    assert hits[0]["id"] == "lgpd-art-5-p1"
    assert hits[0]["score"] > 0.99


def test_bm25_over_postgres_finds_rare_token(ready_store):
    _sample(ready_store)
    hits = ready_store.lexical_search("zetaunico na LGPD", 2)
    assert hits[0]["article_key"] == "lgpd-art-5"
    fused = search(
        ready_store,
        get_runtime().embedder,
        "O que é dado pessoal zetaunico na LGPD?",
        "hybrid",
        get_settings(),
    )
    assert fused[0]["article_key"] == "lgpd-art-5"


def test_perguntar_returns_citation(ready_store):
    _sample(ready_store)
    with TestClient(app) as client:
        health = client.get("/health")
        page = client.get("/")
        ok = client.post(
            "/perguntar",
            json={"pergunta": "O que é dado pessoal zetaunico na LGPD?", "modo": "hybrid"},
        )
        invalid = client.post("/perguntar", json={"pergunta": "modo inexistente aqui", "modo": "bm25"})
    assert health.status_code == 200
    assert health.json()["trechos"] == 2
    assert "RAG Avaliado" in page.text
    assert ok.status_code == 200
    body = ok.json()
    assert body["citacoes"][0]["lei"] == "LGPD"
    assert body["citacoes"][0]["artigo"] == "Art. 5º"
    assert body["provedor_llm"] == "mock"
    assert body["fallback_mock"] is False
    assert "zetaunico" in body["resposta"].lower()
    assert invalid.status_code == 422


def test_empty_corpus_is_503(ready_store):
    with TestClient(app) as client:
        response = client.post("/perguntar", json={"pergunta": "O que é dado pessoal?"})
    assert response.status_code == 503


def test_langfuse_post_shape(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

    def fake_post(url, json, auth, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["auth"] = auth
        return DummyResponse()

    monkeypatch.setattr("app.tracing.httpx.post", fake_post)
    settings = Settings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
    )
    assert trace_question(settings, "pergunta", "hybrid", "resposta", [{"id": "lgpd-art-5-p1"}], "mock") is True
    assert captured["url"].endswith("/api/public/ingestion")
    assert captured["json"]["batch"][0]["type"] == "trace-create"
    assert captured["auth"] == ("pk-test", "sk-test")
    assert trace_question(get_settings(), "pergunta", "hybrid", "resposta", [], "mock") is False
