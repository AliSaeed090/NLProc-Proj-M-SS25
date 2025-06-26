import os
import csv
from typing import List, Optional
from datetime import datetime

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

class Retriever:
    def __init__(self, embed_model_name: str = "BAAI/bge-large-en", device: str = "cpu"):
        self.embedder = HuggingFaceEmbeddings(model_name=embed_model_name)
        self.vectorstore: Optional[FAISS] = None

    def _extract_text(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in ['.txt', '.md']:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif ext == '.pdf':
                reader = PdfReader(path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            elif ext == '.csv':
                return self._extract_csv_text(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
        return ""

    def _extract_csv_text(self, path: str) -> str:
        lines = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    lines.append(", ".join(header))
                for row in reader:
                    lines.append(", ".join(row))
        except Exception as e:
            print(f"[WARN] Failed to parse CSV {path}: {e}")
        return "\n".join(lines)

    def _extract_date(self, path: str) -> str:
        base = os.path.basename(path)
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(base[:10], fmt).strftime("%Y-%m-%d")
            except:
                continue
        try:
            ts = os.path.getmtime(path)
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        except:
            return "unknown"

    def add_documents(self, paths: List[str]):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        documents = []
        
        for path in paths:
            text = self._extract_text(path)
            if not text.strip():
                continue

            chunks = splitter.split_text(text)
            file_name = os.path.basename(path)
            source_type = os.path.splitext(path)[1].lower().strip('.')
            doc_date = self._extract_date(path)

            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": file_name,
                        "source_type": source_type,
                        "date": doc_date,
                        "chunk_id": i
                    }
                ))

        if documents:
            self.vectorstore = FAISS.from_documents(documents, self.embedder)

    def query(self, question: str, top_k: int = 5, filter_source_type: Optional[str] = None, filter_date_after: Optional[str] = None):
        if self.vectorstore is None:
            raise ValueError("No documents indexed yet.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        results = retriever.get_relevant_documents(question)

        filtered_results = []
        for doc in results:
            if filter_source_type and doc.metadata.get("source_type") != filter_source_type:
                continue
            if filter_date_after:
                date = doc.metadata.get("date")
                if date == "unknown" or date < filter_date_after:
                    continue
            filtered_results.append(doc)
        
        return [(doc.metadata, doc.page_content) for doc in filtered_results[:top_k]]

    def save(self, path: str):
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load(self, path: str):
        self.vectorstore = FAISS.load_local(path, self.embedder)
