import json
from pathlib import Path

from app.chunking import load_chunks
from app.config import get_settings
from app.llm import MockLLM, get_llm

ROOT = Path(__file__).resolve().parents[1]


def test_corpus_chunks_cover_expected_articles():
    documents, chunks = load_chunks(ROOT / "data" / "corpus", get_settings())
    assert len(documents) == 5
    assert len(chunks) > 200
    keys = {chunk.article_key for chunk in chunks}
    ids = [chunk.id for chunk in chunks]
    assert len(ids) == len(set(ids))
    questions = json.loads((ROOT / "data" / "eval" / "questions.json").read_text(encoding="utf-8"))
    assert len(questions) >= 50
    assert len({item["id"] for item in questions}) == len(questions)
    for item in questions:
        assert item["question"]
        assert item["expected_answer"]
        assert item["expected_articles"]
        for article in item["expected_articles"]:
            assert article in keys


def test_mock_answer_cites_context():
    contexts = [
        {
            "law": "LGPD",
            "article": "Art. 5º",
            "title": "LGPD",
            "content": "LGPD. Art. 5º. Dado pessoal: informação relacionada a pessoa natural identificada ou identificável.",
        }
    ]
    result = MockLLM().generate("O que a LGPD considera dado pessoal?", contexts)
    assert "[1]" in result.text
    assert "identificada" in result.text
    assert result.prompt_chars > len(result.text)


def test_missing_gemini_key_falls_back_to_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()
    try:
        model, fallback = get_llm(get_settings())
        assert fallback is True
        assert model.provider == "mock"
    finally:
        get_settings.cache_clear()
