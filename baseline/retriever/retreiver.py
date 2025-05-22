import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import os
import pickle
from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of ~chunk_size tokens (approx. words here), with overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


class Retriever:
    def __init__(self, embed_model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "cpu"):
        self.embedder = SentenceTransformer(embed_model_name, device=device)
        self.index = None
        self.documents: List[Tuple[str, str]] = []  # (doc_id, text)

    def add_documents(self, paths: List[str]):
        """
        Load documents from file paths (.txt, .md, .pdf), chunk them, embed, and add to FAISS index.
        """
        texts = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.txt', '.md']:
                with open(path, 'r', encoding='utf-8') as f:
                    raw = f.read()
            elif ext == '.pdf':
                reader = PdfReader(path)
                raw = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                continue
            for i, chunk in enumerate(chunk_text(raw)):
                doc_id = f"{os.path.basename(path)}_chunk{i}"
                texts.append((doc_id, chunk))
        # prepare embeddings
        corpus = [t[1] for t in texts]
        embeddings = self.embedder.encode(corpus, convert_to_numpy=True)
        # build FAISS
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents = texts

    def query(self, question: str, top_k: int = 1) -> List[Tuple[str, str]]:
        """
        Embed the question and retrieve top_k chunks.
        Returns list of (doc_id, chunk).
        """
        q_emb = self.embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            doc_id, text = self.documents[idx]
            results.append((doc_id, text))
        return results

    def save(self, path: str):
        """
        Save FAISS index and document metadata.
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str):
        """
        Load FAISS index and document metadata.
        """
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "docs.pkl"), "rb") as f:
            self.documents = pickle.load(f)