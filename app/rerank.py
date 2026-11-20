from app.bm25 import BM25Index
from app.textutil import mentioned_articles, tokens


def rerank(query: str, hits: list[dict]) -> list[dict]:
    """Reordena o conjunto híbrido com sinais lexicais e o número do artigo.

    É o reranker padrão, sem modelo neural, para o experimento rodar sem
    download extra. Um cross-encoder pode substituir esta função no mesmo
    contrato: recebe a pergunta e os hits, devolve a lista reordenada.
    """

    if not hits:
        return []
    index = BM25Index.from_texts([hit["content"] for hit in hits])
    bm25_scores = index.scores(query)
    peak = max(bm25_scores) or 1.0
    query_terms = set(tokens(query))
    wanted_articles = set(mentioned_articles(query))
    law_hints = _law_hints(query)
    ranked = []
    for hit, bm25_score in zip(hits, bm25_scores):
        content_terms = set(tokens(hit["content"]))
        coverage = (len(query_terms & content_terms) / len(query_terms)) if query_terms else 0.0
        article_number = _article_number(hit.get("article", ""))
        article_bonus = 1.0 if article_number in wanted_articles else 0.0
        law_bonus = 1.0 if hit.get("law") in law_hints else 0.0
        phrase_bonus = 1.0 if _phrase_hit(query, hit["content"]) else 0.0
        score = (
            0.40 * (bm25_score / peak)
            + 0.25 * coverage
            + 0.20 * article_bonus
            + 0.10 * law_bonus
            + 0.05 * phrase_bonus
        )
        updated = dict(hit)
        updated["score"] = score
        detail = dict(hit.get("detail") or {})
        detail["rerank"] = round(score, 6)
        updated["detail"] = detail
        ranked.append(updated)
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _article_number(label: str) -> int | None:
    numbers = mentioned_articles(label)
    return numbers[0] if numbers else None


def _law_hints(query: str) -> set[str]:
    folded = query.lower()
    hints = set()
    if "lgpd" in folded or "proteção de dados" in folded or "protecao de dados" in folded:
        hints.add("LGPD")
    if "marco civil" in folded or "neutralidade" in folded:
        hints.add("Marco Civil da Internet")
    if "cdc" in folded or "consumidor" in folded or "defesa do consumidor" in folded:
        hints.add("CDC")
    if "clt" in folded or "empregado" in folded or "empregador" in folded:
        hints.add("CLT")
    if "constitui" in folded:
        hints.add("Constituição")
    return hints


def _phrase_hit(query: str, content: str) -> bool:
    words = [word for word in query.lower().split() if len(word) > 3]
    if len(words) < 3:
        return False
    content_folded = content.lower()
    for start in range(len(words) - 2):
        phrase = " ".join(words[start : start + 3])
        if phrase in content_folded:
            return True
    return False
