\
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    title: str
    text: str
    score: float


class SimpleRAG:
    """
    Lightweight local semantic retrieval for the demo.
    Replace knowledge_base.md with reviewed institutional material for real research.
    """

    def __init__(
        self,
        knowledge_path: str = "knowledge_base.md",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.knowledge_path = Path(knowledge_path)
        self.embedder = SentenceTransformer(embedding_model)
        self.chunks = self._load_chunks()
        self.embeddings = self.embedder.encode(
            [c["text"] for c in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _load_chunks(self):
        raw = self.knowledge_path.read_text(encoding="utf-8")
        chunks = []
        current_title = "General"
        current_lines = []

        def flush():
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    chunks.append({"title": current_title, "text": text})

        for line in raw.splitlines():
            if line.startswith("## "):
                flush()
                current_title = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)
        flush()
        return chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        q = self.embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = np.dot(self.embeddings, q)
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                title=self.chunks[i]["title"],
                text=self.chunks[i]["text"],
                score=float(scores[i]),
            )
            for i in indices
        ]
