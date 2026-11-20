import re
import unicodedata

STOPWORDS = {
    "a",
    "o",
    "os",
    "as",
    "um",
    "uma",
    "uns",
    "umas",
    "de",
    "da",
    "do",
    "das",
    "dos",
    "e",
    "em",
    "no",
    "na",
    "nos",
    "nas",
    "para",
    "por",
    "com",
    "sem",
    "que",
    "se",
    "ou",
    "como",
    "ao",
    "aos",
    "à",
    "às",
    "pelo",
    "pela",
    "pelos",
    "pelas",
    "seu",
    "sua",
    "seus",
    "suas",
    "este",
    "esta",
    "esse",
    "essa",
    "isso",
    "isto",
    "aquele",
    "aquela",
}


def fold(text: str) -> str:
    text = text.lower().replace("º", "o").replace("ª", "a")
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def tokens(text: str) -> list[str]:
    folded = fold(text)
    raw = re.findall(r"[0-9a-z]+", folded)
    return [token for token in raw if token not in STOPWORDS and len(token) > 1]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[\.\!\?\;])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def mentioned_articles(text: str) -> list[int]:
    folded = fold(text)
    numbers = re.findall(r"\bart(?:igo|\.)?\s*(\d+)", folded)
    return [int(number) for number in numbers]
