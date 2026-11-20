from dataclasses import dataclass
from pathlib import Path
import re

from app.config import Settings, get_settings
from app.textutil import split_sentences


@dataclass(frozen=True)
class Document:
    id: str
    law: str
    title: str
    source_url: str
    edition: str


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    law: str
    article: str
    article_key: str
    title: str
    source_url: str
    content: str
    search_text: str

    @property
    def token_count(self) -> int:
        return len(self.content.split())


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        raise ValueError("Documento sem frontmatter")
    _, raw_meta, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in raw_meta.strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, body


def load_documents(corpus_dir: Path) -> list[tuple[Document, list[tuple[str, str, int]]]]:
    documents = []
    for path in sorted(corpus_dir.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        document = Document(
            id=meta["id"],
            law=meta["law"],
            title=meta["title"],
            source_url=meta["source_url"],
            edition=meta["edition"],
        )
        articles = []
        for section in re.split(r"(?m)^## ", body):
            section = section.strip()
            match = re.match(r"(Art\.\s*\d+\s*º?)\s*(.*)", section, flags=re.S)
            if not match:
                continue
            label = re.sub(r"\s+", " ", match.group(1)).strip()
            number = int(re.search(r"\d+", label).group())
            content = match.group(2).strip()
            articles.append((label, content, number))
        documents.append((document, articles))
    return documents


def _blocks(text: str, max_chars: int) -> list[str]:
    blocks: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            blocks.append(paragraph)
            continue
        buffer = ""
        for sentence in split_sentences(paragraph):
            if buffer and len(buffer) + 1 + len(sentence) > max_chars:
                blocks.append(buffer)
                buffer = sentence
            else:
                buffer = f"{buffer} {sentence}".strip()
        if buffer:
            blocks.append(buffer)
    return blocks or [text.strip()]


def chunk_document(document: Document, articles: list[tuple[str, str, int]], settings: Settings | None = None) -> list[Chunk]:
    settings = settings or get_settings()
    chunks: list[Chunk] = []
    max_chars = settings.chunk_max_chars
    for label, content, number in articles:
        article_key = f"{document.id}-art-{number}"
        header = f"{document.law}. {label}."
        blocks = _blocks(content, max_chars)
        packed: list[str] = []
        current = ""
        for block in blocks:
            candidate = block if not current else f"{current}\n{block}"
            if current and len(header) + 1 + len(candidate) > max_chars:
                packed.append(current)
                current = block
            else:
                current = candidate
        if current:
            packed.append(current)
        for index, piece in enumerate(packed, start=1):
            body = f"{header}\n{piece}".strip()
            search_text = f"{document.law}. {document.title}. {label}. {piece}"
            chunks.append(
                Chunk(
                    id=f"{article_key}-p{index}",
                    document_id=document.id,
                    law=document.law,
                    article=label,
                    article_key=article_key,
                    title=document.title,
                    source_url=document.source_url,
                    content=body,
                    search_text=search_text,
                )
            )
    return chunks


def load_chunks(corpus_dir: Path, settings: Settings | None = None) -> tuple[list[Document], list[Chunk]]:
    documents: list[Document] = []
    chunks: list[Chunk] = []
    for document, articles in load_documents(corpus_dir):
        documents.append(document)
        chunks.extend(chunk_document(document, articles, settings))
    return documents, chunks
