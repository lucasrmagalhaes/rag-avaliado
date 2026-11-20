from pydantic import BaseModel, Field


class PerguntaIn(BaseModel):
    pergunta: str = Field(min_length=3, max_length=2000)
    modo: str | None = None


class Citacao(BaseModel):
    indice: int
    chunk_id: str
    documento: str
    lei: str
    artigo: str
    titulo: str
    fonte: str
    trecho: str
    score: float


class PerguntaOut(BaseModel):
    pergunta: str
    modo: str
    resposta: str
    citacoes: list[Citacao]
    provedor_llm: str
    fallback_mock: bool
    latencia_ms: float
    prompt_chars: int
    completion_chars: int
