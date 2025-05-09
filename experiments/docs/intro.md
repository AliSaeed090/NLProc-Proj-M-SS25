Introduction

Welcome to the Retriever project! This repository provides a minimal, reusable Python class for indexing and retrieving chunks of text from a variety of document formats (plain text, Markdown, and PDF) using FAISS and SentenceTransformers.

Why a Retriever?

Building a document retriever is often the first step in larger information-retrieval or question-answering systems. Instead of searching raw text files line by line, this retriever embeds chunks of your documents into a vector space and uses approximate nearest neighbors to find the most relevant passages quickly.

Features

Automatic Chunking: Splits long documents into overlapping windows of configurable size.

Vector Embeddings: Uses SentenceTransformers (default: all-MiniLM-L6-v2) to embed text chunks.

Fast Indexing & Search: Leverages FAISS for efficient similarity search over thousands of vectors.

Multi-Format Support: Load .txt, .md, and .pdf files with zero extra effort.

Persistence: Save and load your FAISS index and chunk metadata for reuse.

Extensible: Easy to swap in different models, change chunking logic, or integrate into larger pipelines.

Getting Started

Install dependencies:

pip install sentence-transformers faiss-cpu PyPDF2

Add your documents:

from retriever import Retriever

retriever = Retriever(chunk_size=500, chunk_overlap=50)
retriever.add_documents(["docs/intro.md", "reports/whitepaper.pdf"])

Query the index:

results = retriever.query("What does the retriever do?", top_k=3)
for text, distance in results:
    print(f"{distance:.4f}\t{text}")

Persist and reload:

retriever.save("index.faiss", "chunks.pkl")
# Later
new_retriever = Retriever()
new_retriever.load("index.faiss", "chunks.pkl")

What’s Inside

retriever.py: Core class for document processing, embedding, indexing, and search.

tests/test_retriever.py: Simple unit tests to verify that adding documents and querying works as expected.

README.md: High-level installation and usage instructions.

intro.md: (You’re reading it!) A friendly introduction to get you up and running.

Feel free to explore, experiment, and extend this starter retriever to fit your specific needs. Happy building!

