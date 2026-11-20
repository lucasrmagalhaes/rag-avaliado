from dataclasses import dataclass

import httpx

from app.config import Settings

SYSTEM_PROMPT = """Você responde perguntas sobre normas brasileiras usando somente o contexto.
Cite as fontes entre colchetes, como [1], conforme a numeração do contexto.
Se o contexto não trouxer a resposta, diga que o corpus não traz essa informação.
Não invente número de artigo, prazo ou percentual.
Escreva em português do Brasil, em frases curtas.
O corpus pode ser uma publicação original, diferente do texto consolidado vigente. Não atualize a regra com conhecimento externo."""


@dataclass
class LLMResult:
    text: str
    provider: str
    fallback_mock: bool
    prompt_chars: int
    completion_chars: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LanguageModel:
    provider = "base"

    def generate(self, question: str, contexts: list[dict]) -> LLMResult:
        raise NotImplementedError


class MockLLM(LanguageModel):
    """Resposta extrativa. Copia frases do contexto para os testes rodarem sem chave."""

    provider = "mock"

    def generate(self, question: str, contexts: list[dict]) -> LLMResult:
        prompt = build_user_prompt(question, contexts)
        answer = extractive_answer(question, contexts)
        return LLMResult(
            text=answer,
            provider=self.provider,
            fallback_mock=False,
            prompt_chars=len(SYSTEM_PROMPT) + len(prompt),
            completion_chars=len(answer),
        )


class OpenAILLM(LanguageModel):
    provider = "openai"

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, question: str, contexts: list[dict]) -> LLMResult:
        prompt = build_user_prompt(question, contexts)
        response = httpx.post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": self.settings.openai_model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"].strip()
        usage = payload.get("usage") or {}
        return LLMResult(
            text=text,
            provider=self.provider,
            fallback_mock=False,
            prompt_chars=len(SYSTEM_PROMPT) + len(prompt),
            completion_chars=len(text),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class GeminiLLM(LanguageModel):
    provider = "gemini"

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, question: str, contexts: list[dict]) -> LLMResult:
        prompt = build_user_prompt(question, contexts)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent"
        )
        response = httpx.post(
            url,
            params={"key": self.settings.gemini_api_key},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0},
            },
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        parts = payload["candidates"][0]["content"]["parts"]
        text = "\n".join(part.get("text", "") for part in parts).strip()
        usage = payload.get("usageMetadata") or {}
        return LLMResult(
            text=text,
            provider=self.provider,
            fallback_mock=False,
            prompt_chars=len(SYSTEM_PROMPT) + len(prompt),
            completion_chars=len(text),
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )


def get_llm(settings: Settings) -> tuple[LanguageModel, bool]:
    provider = settings.llm_provider.lower().strip()
    if provider == "mock":
        return MockLLM(), False
    if provider == "openai":
        if not settings.openai_api_key:
            return MockLLM(), True
        return OpenAILLM(settings), False
    if provider == "gemini":
        if not settings.gemini_api_key:
            return MockLLM(), True
        return GeminiLLM(settings), False
    raise ValueError("LLM_PROVIDER deve ser mock, openai ou gemini.")


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    if not contexts:
        context = "(nenhum trecho recuperado)"
    else:
        blocks = []
        for index, item in enumerate(contexts, start=1):
            blocks.append(f"[{index}] {item['law']}, {item['article']}, {item['title']}\n{item['content']}")
        context = "\n\n".join(blocks)
    return f"Pergunta:\n{question}\n\nContexto:\n{context}"


def extractive_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "O corpus não traz trechos para essa pergunta."
    from app.textutil import mentioned_articles, split_sentences, tokens

    question_terms = set(tokens(question))
    wanted = set(mentioned_articles(question))
    candidates: list[tuple[int, int, str]] = []
    for index, item in enumerate(contexts, start=1):
        article_bonus = 2 if wanted and _article_number(item.get("article", "")) in wanted else 0
        for sentence in split_sentences(item["content"]):
            sentence = sentence.strip()
            if len(sentence) < 40:
                continue
            overlap = len(set(tokens(sentence)) & question_terms)
            bonus = article_bonus + _definition_bonus(question, sentence)
            if overlap == 0 and bonus == 0:
                continue
            candidates.append((overlap + bonus, index, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[tuple[int, str]] = []
    seen = set()
    for score, index, sentence in candidates:
        if score <= 0:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((index, sentence))
        if len(chosen) == 3:
            break
    if not chosen:
        sentence = split_sentences(contexts[0]["content"])[0]
        chosen = [(1, sentence)]
    parts = []
    for index, sentence in chosen:
        text = sentence if sentence.endswith((".", "!", "?", ";")) else f"{sentence}."
        parts.append(f"{text} [{index}]")
    return " ".join(parts)


def _definition_bonus(question: str, sentence: str) -> int:
    from app.textutil import fold, tokens

    question_words = tokens(question)
    sentence_folded = fold(sentence)
    bonus = 0
    grams = list(question_words)
    grams.extend(
        f"{question_words[index]} {question_words[index + 1]}"
        for index in range(len(question_words) - 1)
    )
    for gram in grams:
        if f"{gram}:" in sentence_folded or f"considera-se {gram}" in sentence_folded or sentence_folded.startswith(f"{gram} "):
            bonus = max(bonus, 4)
    return bonus


def _article_number(label: str) -> int | None:
    from app.textutil import mentioned_articles

    numbers = mentioned_articles(label)
    return numbers[0] if numbers else None
