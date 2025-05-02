# RAG Project – Summer Semester 2025

## Overview
This repository hosts the code for a semester-long project on building and experimenting with Retrieval-Augmented Generation (RAG) systems. Students start with a shared baseline and then explore specialized variations in teams.

## Structure
- `baseline/`: Common starter system (retriever + generator)
- `experiments/`: Each team's independent exploration
- `evaluation/`: Common tools for comparing results
- `utils/`: Helper functions shared across code

## Getting Started
1. Clone the repo
2. `cd baseline/`
3. Install dependencies: `pip install -r ../requirements.txt`

## Teams & Tracks
## Week 2 task 
## Team Neural Narrators
## Hafiz Muhammad Ali Saeed (2161224)
## Ahsan Munir (2121328)
## Muhammad Sohail Anwar (2112858)

## How Vector Search Works

1. **Embedding**  
   We use a pre-trained SentenceTransformer (e.g. `all-MiniLM-L6-v2`) to convert each text snippet into a fixed-length vector that captures its meaning.

2. **Indexing**  
   All vectors are stored in a FAISS `IndexFlatIP`, which performs fast inner-product (cosine) similarity search over normalized embeddings.

3. **Querying**  
   A user’s query is encoded into a vector the same way, then FAISS returns the top-k snippets whose vectors are closest (highest cosine similarity) to the query vector.

## What We Observed

- **Exact synonyms first**: “feline” ≈ “cat” gives the highest score.  
- **Close paraphrases next**: “mat” vs. “rug” or “sits” vs. “lies” still rank highly.  
- **Irrelevant content drops off**: Unrelated sentences score near zero.  
- **Conceptual grouping**: Queries about cars vs. bikes retrieve both speed-focused and spec-focused sentences because they share the broader “vehicle” concept, not just exact words.
