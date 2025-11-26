import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Default documents path
DEFAULT_DOCS_PATH = Path(__file__).parent.parent.parent / "docs"


@dataclass
class Chunk:
    """Represents a document chunk."""
    id: str
    source: str
    content: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "source": self.source, "content": self.content, "score": self.score}


class DocumentRetriever:
    """TF-IDF based document retriever."""
    
    def __init__(self, docs_path: Optional[Path] = None):
        self.docs_path = docs_path or DEFAULT_DOCS_PATH
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self._is_indexed = False

    def load_and_chunk_docs(self) -> List[Chunk]:
        """Load markdown docs and split into chunks by '##' headers."""
        self.chunks = []
        if not self.docs_path.exists():
            return self.chunks

        for doc_file in sorted(self.docs_path.glob("*.md")):
            content = doc_file.read_text(encoding="utf-8")
            source = doc_file.stem
            sections = self._split_by_sections(content)

            for i, sec in enumerate(sections):
                if sec.strip():
                    self.chunks.append(Chunk(
                        id=f"{source}::chunk{i}",
                        source=source,
                        content=sec.strip()
                    ))
        return self.chunks

    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by '##' markdown headers."""
        parts = re.split(r'(?=^##\s)', content, flags=re.MULTILINE)
        sections = [p.strip() for p in parts if p.strip()]
        return sections or [content.strip()]

    def build_index(self) -> None:
        """Build TF-IDF index on all chunks."""
        if not self.chunks:
            self.load_and_chunk_docs()
        if not self.chunks:
            return

        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform([c.content for c in self.chunks])
        self._is_indexed = True

    def search(self, query: str, top_k: int = 3) -> List[Chunk]:
        """Search chunks by TF-IDF cosine similarity."""
        if not self._is_indexed:
            self.build_index()
        if not self.chunks or not self.vectorizer:
            return []

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        return [Chunk(id=self.chunks[i].id,
                      source=self.chunks[i].source,
                      content=self.chunks[i].content,
                      score=float(sims[i])) for i in top_indices]

    def get_all_chunks(self) -> List[Chunk]:
        if not self.chunks:
            self.load_and_chunk_docs()
        return self.chunks


# Singleton retriever
_retriever_instance: Optional[DocumentRetriever] = None


def get_retriever(docs_path: Optional[Path] = None) -> DocumentRetriever:
    global _retriever_instance
    if _retriever_instance is None or (docs_path and docs_path != _retriever_instance.docs_path):
        _retriever_instance = DocumentRetriever(docs_path)
        _retriever_instance.build_index()
    return _retriever_instance


def search_docs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    retriever = get_retriever()
    return [c.to_dict() for c in retriever.search(query, top_k)]


if __name__ == "__main__":
    retriever = get_retriever()

    print("=== All Chunks ===")
    for c in retriever.get_all_chunks():
        print(f"\n{c.id}:\n  {c.content[:100]}...")

    queries = ["return policy beverages", "AOV average order value", "Summer Beverages 1997 dates"]
    for q in queries:
        print(f"\n=== Search: '{q}' ===")
        for r in retriever.search(q, top_k=3):
            print(f"\n{r.id} (score: {r.score:.3f}):\n  {r.content[:150]}...")
