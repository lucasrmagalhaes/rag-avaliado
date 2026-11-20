"""Compara vetor, híbrido e híbrido com reranker. Sai com erro se o piso de qualidade cair."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.metrics import aggregate, faithfulness, quality_failures, retrieval_at_k, token_f1
from app.retrieve import MODES
from app.runtime import get_runtime


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "eval" / "questions.json")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "data" / "eval" / "thresholds.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "eval_latest.json")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    questions = load_json(args.questions)
    thresholds = load_json(args.thresholds)
    if len(questions) < 50:
        raise SystemExit(f"O conjunto precisa ter pelo menos 50 perguntas. Encontrei {len(questions)}.")

    runtime = get_runtime()
    runtime.store.migrate()
    total = runtime.store.count()
    if total == 0:
        raise SystemExit("Corpus vazio. Rode python scripts/ingest.py antes da avaliação.")

    warmup_started = perf_counter()
    runtime.embedder.embed_query("aquecimento da avaliação")
    warmup_ms = (perf_counter() - warmup_started) * 1000

    per_mode: dict[str, list[dict]] = {}
    prompt_chars = 0
    completion_chars = 0
    for mode in MODES:
        rows = []
        for item in questions:
            started = perf_counter()
            state = runtime.graph.invoke({"question": item["question"], "mode": mode})
            latency_ms = (perf_counter() - started) * 1000
            contexts = state.get("contexts") or []
            answer = state.get("answer") or ""
            retrieved = [context["article_key"] for context in contexts]
            retrieval = retrieval_at_k(item["expected_articles"], retrieved, k=thresholds.get("k", 5))
            row = {
                "id": item["id"],
                "hit": retrieval["hit"],
                "recall": retrieval["recall"],
                "mrr": retrieval["mrr"],
                "rank": retrieval["rank"],
                "faithfulness": faithfulness(answer, [context["content"] for context in contexts]),
                "relevancy": token_f1(answer, item["expected_answer"]),
                "latency_ms": latency_ms,
                "retrieved": retrieved,
            }
            rows.append(row)
            prompt_chars += int(state.get("prompt_chars") or 0)
            completion_chars += int(state.get("completion_chars") or 0)
        per_mode[mode] = rows
        summary = aggregate(rows)
        print(
            f"{mode}: hit@5={summary['hit_rate_at_5']:.3f} recall@5={summary['recall_at_5']:.3f} "
            f"mrr={summary['mrr']:.3f} fidelidade={summary['faithfulness']:.3f} "
            f"relevancia={summary['answer_relevancy']:.3f} "
            f"p50={summary['latency_ms_p50']:.1f}ms p95={summary['latency_ms_p95']:.1f}ms"
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "embedding_provider": runtime.settings.embedding_provider,
        "embedding_model": runtime.embedder.name,
        "llm_provider": runtime.settings.llm_provider,
        "llm_effective": runtime.llm.provider,
        "fallback_mock": runtime.fallback_mock,
        "judge": "lexical",
        "judge_note": (
            "Fidelidade e relevância foram calculadas por sobreposição de termos, sem LLM juiz. "
            "Isso permite repetir o número sem chave de API. Não é a nota de um juiz RAGAS ou DeepEval."
        ),
        "k": thresholds.get("k", 5),
        "n_questions": len(questions),
        "n_chunks": total,
        "warmup_ms": round(warmup_ms, 2),
        "modes": {mode: aggregate(rows) for mode, rows in per_mode.items()},
        "questions": per_mode,
        "cost": {
            "billed": None,
            "currency": None,
            "reason": (
                "A execução usou embeddings locais e o provedor de LLM configurado. "
                "Nenhum valor foi lido de uma fatura. prompt_chars e completion_chars medem o texto montado."
            ),
            "prompt_chars_total": prompt_chars,
            "completion_chars_total": completion_chars,
        },
    }
    if runtime.llm.provider == "mock":
        report["cost"]["reason"] = (
            "Execução com embeddings locais e LLM mock, extrativo. Nenhuma chamada paga foi feita. "
            "prompt_chars e completion_chars medem o prompt que seria enviado e o texto da resposta."
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório: {args.report}")

    if args.skip_gate:
        return
    failures = quality_failures(report, thresholds)
    if failures:
        print("Piso de qualidade não atingido:")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)
    print(f"Piso de qualidade atingido no modo {thresholds['mode']}.")


if __name__ == "__main__":
    main()
