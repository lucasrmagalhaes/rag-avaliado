from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://rag:rag@127.0.0.1:5432/rag"
    llm_provider: str = "mock"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    retrieval_mode: str = "hybrid"
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    lexical_pool: int = 40
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_full_scan_limit: int = 5000
    reranker_provider: str = "lexical"
    chunk_max_chars: int = 550
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    auto_migrate: bool = True
    request_timeout_seconds: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
