from app.config import Settings
from app.embeddings import EmbeddingModel
from app.rerank import rerank
from app.rrf import reciprocal_rank_fusion
from app.store import PostgresStore

MODES = ("vector", "hybrid", "hybrid_rerank")


def search(store: PostgresStore, embedder: EmbeddingModel, query: str, mode: str, settings: Settings) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"Modo inválido: {mode}")
    k = settings.top_k
    if mode == "vector":
        return store.vector_search(embedder.embed_query(query), k)
    pool = max(settings.candidate_k, k)
    vector_hits = store.vector_search(embedder.embed_query(query), pool)
    lexical_hits = store.lexical_search(query, pool)
    fused_ids = reciprocal_rank_fusion(
        [[hit["id"] for hit in vector_hits], [hit["id"] for hit in lexical_hits]],
        k=settings.rrf_k,
    )
    by_id: dict[str, dict] = {}
    for hit in vector_hits + lexical_hits:
        current = by_id.get(hit["id"])
        if current is None:
            by_id[hit["id"]] = dict(hit)
            continue
        detail = dict(current.get("detail") or {})
        detail.update(hit.get("detail") or {})
        current["detail"] = detail
    fused = []
    for doc_id, score in fused_ids:
        hit = dict(by_id[doc_id])
        detail = dict(hit.get("detail") or {})
        detail["rrf"] = float(score)
        hit["detail"] = detail
        hit["score"] = float(score)
        fused.append(hit)
    fused = apply_lexical_prior(fused, lexical_hits, alpha=0.05)
    fused = fused[:pool]
    if mode == "hybrid_rerank" and settings.reranker_provider.lower() != "none":
        fused = rerank(query, fused)
    return fused[:k]


def apply_lexical_prior(fused: list[dict], lexical_hits: list[dict], alpha: float) -> list[dict]:
    """O RRF com k=60 quase não separa a 1ª posição da 2ª.

    Um prior proporcional ao BM25 devolve esse intervalo quando o termo raro
    está concentrado em um trecho, sem deixar de usar a fusão por rank.
    """

    if not fused or alpha <= 0:
        return fused
    bm25 = {hit["id"]: float((hit.get("detail") or {}).get("bm25") or 0.0) for hit in lexical_hits}
    peak = max(bm25.values(), default=0.0) or 1.0
    adjusted = []
    for hit in fused:
        updated = dict(hit)
        prior = alpha * (bm25.get(hit["id"], 0.0) / peak)
        detail = dict(updated.get("detail") or {})
        detail["lexical_prior"] = prior
        updated["detail"] = detail
        updated["score"] = float(updated.get("score") or 0.0) + prior
        adjusted.append(updated)
    adjusted.sort(key=lambda item: item["score"], reverse=True)
    return adjusted
