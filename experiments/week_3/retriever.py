import os
import pickle
from typing import List, Tuple, Union

import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


class Retriever:
    """
    A simple document retriever using FAISS + SentenceTransformers.

    Methods
    -------
    add_documents(paths)
        Read .txt, .md or .pdf files, chunk them into fixed-size windows,
        embed them, and add to the FAISS index.
    query(q, top_k=5)
        Embed a query string and return the top_k most similar chunks.
    save(index_path, texts_path)
        Persist the FAISS index and the list of chunks to disk.
    load(index_path, texts_path)
        Reload a saved FAISS index and chunk list.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # texts[i] is the original text for vector index position i
        self.texts: List[str] = []

        # FAISS flat L2 index
        self.index = faiss.IndexFlatL2(embedding_dim)

    def _chunk_text(self, text: str) -> List[str]:
        """Break a long string into overlapping chunks."""
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(text), step):
            chunks.append(text[start : start + self.chunk_size])
        return chunks

    def add_documents(self, paths: List[str]) -> None:
        """
        Load each path (txt, md, or pdf), chunk, embed, and add to index.
        """
        new_chunks = []

        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                reader = PdfReader(path)
                raw = "".join(page.extract_text() or "" for page in reader.pages)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()

            for chunk in self._chunk_text(raw):
                new_chunks.append(chunk)
                self.texts.append(chunk)

        if new_chunks:
            embeddings = self.model.encode(new_chunks, convert_to_numpy=True)
            self.index.add(embeddings)

    def query(self, q: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return the top_k most similar chunks to the query string.

        Returns
        -------
        List of (chunk_text, distance) tuples, sorted by increasing distance.
        """
        q_emb = self.model.encode([q], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.texts[idx], float(dist)))
        return results

    def save(self, index_path: str, texts_path: str) -> None:
        """
        Write FAISS index to `index_path` and chunk list to `texts_path`.
        """
        faiss.write_index(self.index, index_path)
        with open(texts_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, index_path: str, texts_path: str) -> None:
        """
        Load FAISS index from `index_path` and chunk list from `texts_path`.
        """
        self.index = faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)


if __name__ == "__main__":
    # Quick smoke‐test
    import tempfile
    from pathlib import Path

    # create two tiny temp files
    t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    t1.write(b"Hello world\nThis is a test of FAISS retrieval.")
    t1.close()
    t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    t2.write(b"Another document.  FAISS + SentenceTransformer demo.")
    t2.close()

    r = Retriever(chunk_size=50, chunk_overlap=10)
    r.add_documents([t1.name, t2.name])

    for chunk, dist in r.query("test FAISS", top_k=2):
        print(f"→ {chunk!r} (distance={dist:.4f})")
# nafees-iqbal github
# Halum