import math

from app.textutil import tokens


class BM25Index:
    """Okapi BM25 sobre o texto dos trechos.

    O índice invertido cabe em memória no corpus de demonstração. Quando a
    tabela passa de bm25_full_scan_limit, a busca lexical usa o tsvector do
    Postgres só para escolher os candidatos e o BM25 reordena esse conjunto.
    """

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.doc_len = [len(doc) for doc in documents]
        self.avgdl = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for doc in documents:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)
            for term in counts:
                self.df[term] = self.df.get(term, 0) + 1

    @classmethod
    def from_texts(cls, texts: list[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        return cls([tokens(text) for text in texts], k1=k1, b=b)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> list[float]:
        query_terms = tokens(query)
        if not query_terms or not self.doc_count:
            return [0.0] * self.doc_count
        scored = []
        for index, counts in enumerate(self.tf):
            score = 0.0
            length = self.doc_len[index] or 1
            for term in query_terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                score += self.idf(term) * numerator / denominator
            scored.append(score)
        return scored
