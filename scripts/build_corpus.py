"""Converte o HTML da publicacao original da Camara em Markdown do corpus.

Os HTML nao fazem parte do repositorio. Baixe as paginas oficiais e rode:

    python scripts/build_corpus.py --html-dir /caminho/dos/html

O Markdown gerado em data/corpus e a fonte da ingestao.
"""

from __future__ import annotations

import argparse
import re
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    {
        "file": "lgpd-camara.html",
        "out": "lgpd.md",
        "id": "lgpd",
        "law": "LGPD",
        "title": "Lei nº 13.709, de 14 de agosto de 2018 (LGPD), publicação original",
        "source_url": "https://www2.camara.leg.br/legin/fed/lei/2018/lei-13709-14-agosto-2018-787077-publicacaooriginal-156212-pl.html",
        "edition": "Publicação original da Câmara dos Deputados. Alterações posteriores, como a Lei nº 13.853/2019, não estão neste arquivo.",
        "articles": None,
    },
    {
        "file": "mci.html",
        "out": "marco-civil.md",
        "id": "mci",
        "law": "Marco Civil da Internet",
        "title": "Lei nº 12.965, de 23 de abril de 2014 (Marco Civil da Internet), publicação original",
        "source_url": "https://www2.camara.leg.br/legin/fed/lei/2014/lei-12965-23-abril-2014-778630-publicacaooriginal-143980-pl.html",
        "edition": "Publicação original da Câmara dos Deputados.",
        "articles": None,
    },
    {
        "file": "cdc.html",
        "out": "cdc.md",
        "id": "cdc",
        "law": "CDC",
        "title": "Lei nº 8.078, de 11 de setembro de 1990 (Código de Defesa do Consumidor), publicação original",
        "source_url": "https://www2.camara.leg.br/legin/fed/lei/1990/lei-8078-11-setembro-1990-365086-publicacaooriginal-1-pl.html",
        "edition": "Publicação original da Câmara dos Deputados. Alterações posteriores não estão neste arquivo.",
        "articles": None,
    },
    {
        "file": "clt.html",
        "out": "clt.md",
        "id": "clt",
        "law": "CLT",
        "title": "Decreto-Lei nº 5.452, de 1º de maio de 1943 (CLT), publicação original",
        "source_url": "https://www2.camara.leg.br/legin/fed/declei/1940-1949/decreto-lei-5452-1-maio-1943-415500-publicacaooriginal-1-pe.html",
        "edition": "Publicação original de 1943. O texto difere da CLT consolidada vigente. O adicional de hora suplementar neste arquivo, por exemplo, é de 20 por cento.",
        "articles": [2, 3, 58, 59, 71, 129, 130, 134, 442, 473, 477, 482, 487],
    },
    {
        "file": "cf.html",
        "out": "constituicao.md",
        "id": "cf",
        "law": "Constituição",
        "title": "Constituição da República Federativa do Brasil de 1988, publicação original, artigos 5º a 7º",
        "source_url": "https://www2.camara.leg.br/legin/fed/consti/1988/constituicao-1988-5-outubro-1988-322142-publicacaooriginal-1-pl.html",
        "edition": "Trechos da publicação original de 1988. Emendas constitucionais posteriores não estão neste arquivo.",
        "articles": [5, 6, 7],
    },
]


def html_to_text(path: Path) -> str:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", "", text)
    text = unescape(text).replace("\xa0", " ")
    text = text.replace("faItar", "faltar")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def extract_articles(text: str) -> dict[int, str]:
    parts = re.split(r"(?=\bArt\.\s*\d+)", text)
    best: dict[int, str] = {}
    for part in parts:
        match = re.match(r"\s*Art\.\s*(\d+)", part)
        if not match:
            continue
        number = int(match.group(1))
        body = clean_article(part)
        if number not in best or len(body) > len(best[number]):
            best[number] = body
    return best


def clean_article(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r", "").splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if line:
            lines.append(line)
    body = "\n".join(lines).strip()
    body = re.split(r"\nBrasília,", body, maxsplit=1)[0].strip()
    return body


def article_label(body: str, number: int) -> str:
    match = re.match(r"Art\.\s*\d+\s*(º|°)?", body)
    if match and match.group(1):
        return f"Art. {number}º"
    if number <= 9:
        return f"Art. {number}º"
    return f"Art. {number}"


def body_without_label(body: str) -> str:
    without = re.sub(r"^Art\.\s*\d+\s*(º|°)?\.?\s*", "", body, count=1)
    return without.strip()


def render(meta: dict, articles: dict[int, str]) -> str:
    wanted = meta["articles"] or sorted(articles)
    lines = [
        "---",
        f"id: {meta['id']}",
        f"law: {meta['law']}",
        f"title: {meta['title']}",
        f"source_url: {meta['source_url']}",
        f"edition: {meta['edition']}",
        "---",
        "",
    ]
    for number in wanted:
        if number not in articles:
            raise SystemExit(f"Artigo {number} ausente em {meta['file']}")
        label = article_label(articles[number], number)
        body = body_without_label(articles[number])
        lines.append(f"## {label}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", type=Path, default=Path("/tmp/laws"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "corpus")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for meta in DOCS:
        source = args.html_dir / meta["file"]
        articles = extract_articles(html_to_text(source))
        text = render(meta, articles)
        destination = args.out_dir / meta["out"]
        destination.write_text(text, encoding="utf-8")
        print(f"{destination.name}: {text.count('## Art.')} artigos, {len(text)} caracteres")


if __name__ == "__main__":
    main()
