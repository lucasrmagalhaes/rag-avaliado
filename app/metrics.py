import math

from app.textutil import split_sentences, tokens


def token_f1(left: str, right: str) -> float:
    left_tokens = set(tokens(left))
    right_tokens = set(tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def token_recall(answer: str, source: str) -> float:
    answer_tokens = set(tokens(answer))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & set(tokens(source))) / len(answer_tokens)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Fidelidade lexical: fração dos termos de cada frase que aparece no contexto."""

    sentences = split_sentences(answer)
    if not sentences or not contexts:
        return 0.0
    scores = [max(token_recall(sentence, context) for context in contexts) for sentence in sentences]
    return sum(scores) / len(scores)


def retrieval_at_k(expected_articles: list[str], retrieved_keys: list[str], k: int) -> dict:
    top: list[str] = []
    for key in retrieved_keys:
        if key not in top:
            top.append(key)
    top = top[:k]
    expected = list(dict.fromkeys(expected_articles))
    found = [item for item in expected if item in top]
    rank = None
    for index, item in enumerate(top, start=1):
        if item in expected:
            rank = index
            break
    return {
        "hit": 1 if found else 0,
        "recall": (len(found) / len(expected)) if expected else 0.0,
        "mrr": 0.0 if rank is None else 1.0 / rank,
        "rank": rank,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("A avaliação não tem perguntas.")

    def mean(name: str) -> float:
        return sum(row[name] for row in rows) / len(rows)

    latencies = [row["latency_ms"] for row in rows]
    return {
        "n": len(rows),
        "hit_rate_at_5": mean("hit"),
        "recall_at_5": mean("recall"),
        "mrr": mean("mrr"),
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("relevancy"),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
    }


def quality_failures(report: dict, thresholds: dict) -> list[str]:
    mode = thresholds["mode"]
    metrics = report["modes"][mode]
    failures = []
    checks = (
        ("hit_rate_at_5", "min_hit_rate_at_5"),
        ("recall_at_5", "min_recall_at_5"),
        ("faithfulness", "min_faithfulness"),
    )
    for metric, floor_name in checks:
        floor = float(thresholds[floor_name])
        actual = float(metrics[metric])
        if actual + 1e-9 < floor:
            failures.append(f"{mode}.{metric}={actual:.4f} abaixo de {floor:.4f}")
    return failures
