from functools import lru_cache

from app.config import Settings, get_settings
from app.embeddings import get_embedder
from app.graph import compile_graph
from app.llm import get_llm
from app.store import PostgresStore


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = PostgresStore(settings)
        self.embedder = get_embedder(settings)
        self.llm, self.fallback_mock = get_llm(settings)
        self.graph = compile_graph(self.store, self.embedder, self.llm, settings)


@lru_cache
def get_runtime() -> Runtime:
    return Runtime(get_settings())
