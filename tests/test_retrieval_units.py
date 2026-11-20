from app.bm25 import BM25Index
from app.metrics import faithfulness, quality_failures, retrieval_at_k, token_f1
from app.rerank import rerank
from app.retrieve import apply_lexical_prior
from app.rrf import reciprocal_rank_fusion
from app.textutil import mentioned_articles, tokens


def test_bm25_ranks_rare_term_first():
    index = BM25Index.from_texts(
        [
            "o intervalo de repouso é de uma hora",
            "zetaunico aparece apenas neste trecho da norma",
            "outro artigo fala de férias anuais",
        ]
    )
    scores = index.scores("onde está zetaunico")
    assert scores.index(max(scores)) == 1


def test_rrf_prefers_document_present_in_both_lists():
    fused = reciprocal_rank_fusion(
        [
            ["a", "b", "c"],
            ["b", "c", "a"],
        ],
        k=60,
    )
    assert fused[0][0] == "b"


def test_lexical_prior_keeps_a_strong_bm25_lead():
    fused = [
        {"id": "ambos", "score": 0.031, "detail": {"rrf": 0.031}},
        {"id": "so_lexical", "score": 0.016, "detail": {"rrf": 0.016}},
    ]
    lexical = [
        {"id": "so_lexical", "detail": {"bm25": 18.0}},
        {"id": "ambos", "detail": {"bm25": 10.0}},
    ]
    ranked = apply_lexical_prior(fused, lexical, alpha=0.05)
    assert ranked[0]["id"] == "so_lexical"


def test_reranker_boosts_named_article_and_law():
    hits = [
        {"id": "cdc", "law": "CDC", "article": "Art. 7º", "content": "CDC. Art. 7º. Direitos previstos neste código.", "detail": {}},
        {"id": "lgpd", "law": "LGPD", "article": "Art. 7º", "content": "LGPD. Art. 7º. Consentimento do titular para o tratamento.", "detail": {}},
    ]
    ranked = rerank("Quais hipóteses do art. 7 da LGPD autorizam o tratamento?", hits)
    assert ranked[0]["id"] == "lgpd"


def test_article_mentions_and_tokens_keep_numbers():
    assert mentioned_articles("Veja o art. 52 e o artigo 7º") == [52, 7]
    assert "20" in tokens("adicional de 20%")
    assert "de" not in tokens("tratamento de dados")


def test_faithfulness_is_high_when_answer_copies_context():
    context = "A multa simples é de até 2% do faturamento, limitada a 50 milhões de reais."
    answer = "A multa simples é de até 2% do faturamento, limitada a 50 milhões de reais."
    assert faithfulness(answer, [context]) > 0.9
    assert token_f1(answer, "multa de 2% limitada a 50 milhões de reais") > 0.4


def test_retrieval_metric_and_gate():
    scored = retrieval_at_k(["lgpd-art-5"], ["lgpd-art-7", "lgpd-art-5"], k=5)
    assert scored["hit"] == 1
    assert scored["rank"] == 2
    assert scored["mrr"] == 0.5
    report = {
        "modes": {
            "hybrid": {
                "hit_rate_at_5": 0.9,
                "recall_at_5": 0.9,
                "faithfulness": 0.8,
            }
        }
    }
    thresholds = {
        "mode": "hybrid",
        "min_hit_rate_at_5": 0.8,
        "min_recall_at_5": 0.8,
        "min_faithfulness": 0.55,
    }
    assert quality_failures(report, thresholds) == []
    report["modes"]["hybrid"]["hit_rate_at_5"] = 0.2
    assert quality_failures(report, thresholds)
