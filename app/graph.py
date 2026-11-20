from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.embeddings import EmbeddingModel
from app.llm import LanguageModel
from app.retrieve import search
from app.store import PostgresStore


class RAGState(TypedDict, total=False):
    question: str
    mode: str
    contexts: list[dict]
    answer: str
    llm_provider: str
    fallback_mock: bool
    prompt_chars: int
    completion_chars: int
    prompt_tokens: int | None
    completion_tokens: int | None


def compile_graph(store: PostgresStore, embedder: EmbeddingModel, llm: LanguageModel, settings: Settings):
    """Grafo de dois nós: recuperar e gerar. O modo escolhe vetor, híbrido ou híbrido com reranker."""

    def retrieve_node(state: RAGState) -> dict:
        hits = search(store, embedder, state["question"], state["mode"], settings)
        return {"contexts": hits}

    def generate_node(state: RAGState) -> dict:
        result = llm.generate(state["question"], state.get("contexts") or [])
        return {
            "answer": result.text,
            "llm_provider": result.provider,
            "prompt_chars": result.prompt_chars,
            "completion_chars": result.completion_chars,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
