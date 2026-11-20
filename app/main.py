import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from app import __version__
from app.retrieve import MODES
from app.runtime import get_runtime
from app.schemas import Citacao, PerguntaIn, PerguntaOut
from app.tracing import trace_question

logger = logging.getLogger("rag_avaliado")
STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "eval_latest.json"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    runtime = get_runtime()
    if runtime.settings.auto_migrate:
        try:
            runtime.store.migrate()
        except Exception as exc:
            logger.warning("Migração não concluída: %s", exc)
    yield


app = FastAPI(
    title="RAG Avaliado",
    version=__version__,
    description="Perguntas com citação da fonte sobre normas públicas brasileiras, com qualidade medida.",
    lifespan=lifespan,
)


@app.get("/")
def pagina():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    try:
        total = get_runtime().store.count()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "erro", "database": str(exc)})
    return {"status": "ok", "database": "ok", "trechos": total}


@app.get("/modos")
def modos():
    return {"modos": list(MODES), "padrao": get_runtime().settings.retrieval_mode}


@app.get("/relatorio")
def relatorio():
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Ainda não há relatório. Rode scripts/evaluate.py.")
    return JSONResponse(content=json.loads(REPORT_PATH.read_text(encoding="utf-8")))


@app.post("/perguntar", response_model=PerguntaOut)
def perguntar(body: PerguntaIn) -> PerguntaOut:
    runtime = get_runtime()
    mode = (body.modo or runtime.settings.retrieval_mode).strip()
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"Modo inválido. Use um de: {', '.join(MODES)}.")
    try:
        total = runtime.store.count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Banco indisponível: {exc}") from exc
    if total == 0:
        raise HTTPException(status_code=503, detail="Corpus vazio. Rode python scripts/ingest.py.")
    started = perf_counter()
    try:
        state = runtime.graph.invoke({"question": body.pergunta.strip(), "mode": mode})
    except Exception as exc:
        logger.exception("Falha ao responder")
        raise HTTPException(status_code=502, detail=f"Falha ao gerar a resposta: {exc}") from exc
    elapsed_ms = (perf_counter() - started) * 1000
    contexts = state.get("contexts") or []
    answer = state.get("answer") or ""
    provider = state.get("llm_provider") or runtime.llm.provider
    trace_question(runtime.settings, body.pergunta, mode, answer, contexts, provider)
    citations = [
        Citacao(
            indice=index,
            chunk_id=item["id"],
            documento=item["document_id"],
            lei=item["law"],
            artigo=item["article"],
            titulo=item["title"],
            fonte=item["source_url"],
            trecho=item["content"][:500],
            score=round(float(item.get("score") or 0.0), 6),
        )
        for index, item in enumerate(contexts, start=1)
    ]
    return PerguntaOut(
        pergunta=body.pergunta.strip(),
        modo=mode,
        resposta=answer,
        citacoes=citations,
        provedor_llm=provider,
        fallback_mock=runtime.fallback_mock,
        latencia_ms=round(elapsed_ms, 2),
        prompt_chars=int(state.get("prompt_chars") or 0),
        completion_chars=int(state.get("completion_chars") or 0),
    )
