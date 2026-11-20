import hashlib
import math

from app.config import Settings
from app.textutil import tokens


class EmbeddingModel:
    name = "base"
    dim = 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbedding(EmbeddingModel):
    """Embedding determinístico para testes, sem rede e sem modelo."""

    def __init__(self, dim: int = 384):
        self.name = "hash"
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokens(text) or ["vazio"]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


class LocalEmbedding(EmbeddingModel):
    """Modelo multilíngue ONNX via fastembed. Cabe em CPU e não exige chave."""

    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=settings.embedding_model)
        self.name = settings.embedding_model
        self.dim = settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [_normalize(_as_list(item)) for item in self.model.embed(texts)]
        _check_dim(vectors, self.dim, self.name)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        query_embed = getattr(self.model, "query_embed", None)
        if query_embed is not None:
            produced = list(query_embed([text]))
        else:
            produced = list(self.model.embed([text]))
        vector = _normalize(_as_list(produced[0]))
        _check_dim([vector], self.dim, self.name)
        return vector


def get_embedder(settings: Settings) -> EmbeddingModel:
    provider = settings.embedding_provider.lower()
    if provider == "hash":
        return HashEmbedding(settings.embedding_dim)
    if provider == "local":
        return LocalEmbedding(settings)
    raise ValueError(
        "EMBEDDING_PROVIDER deve ser local ou hash. "
        "O caminho local usa paraphrase-multilingual-MiniLM-L12-v2."
    )


def _as_list(vector) -> list[float]:
    return [float(value) for value in list(vector)]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _check_dim(vectors: list[list[float]], dim: int, name: str) -> None:
    for vector in vectors:
        if len(vector) != dim:
            raise ValueError(
                f"O modelo {name} gerou dimensão {len(vector)}, mas EMBEDDING_DIM está em {dim}. "
                "Ajuste a dimensão e recrie a tabela."
            )
