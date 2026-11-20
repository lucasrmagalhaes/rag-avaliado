import logging
import uuid
from datetime import datetime, timezone

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)


def trace_question(settings: Settings, question: str, mode: str, answer: str, contexts: list[dict], provider: str) -> bool:
    """Envia um trace ao Langfuse quando as chaves existem. Falha de rede não quebra a resposta."""

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return False
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    trace_id = str(uuid.uuid4())
    batch = [
        {
            "id": str(uuid.uuid4()),
            "type": "trace-create",
            "timestamp": now,
            "body": {
                "id": trace_id,
                "timestamp": now,
                "name": "rag.perguntar",
                "input": {"pergunta": question, "modo": mode},
                "output": {"resposta": answer},
                "metadata": {
                    "provedor_llm": provider,
                    "chunks": [item["id"] for item in contexts],
                },
                "tags": ["rag-avaliado", mode],
            },
        }
    ]
    try:
        response = httpx.post(
            f"{settings.langfuse_host.rstrip('/')}/api/public/ingestion",
            json={"batch": batch},
            auth=(settings.langfuse_public_key, settings.langfuse_secret_key),
            timeout=5.0,
        )
        response.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("Langfuse não recebeu o trace: %s", exc)
        return False
