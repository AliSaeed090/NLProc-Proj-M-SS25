# Retrieval-Augmented Text Generation Pipeline

This repository provides a simple retrieval-augmented text generation system using:

* **SentenceTransformers** + **FAISS** for document embedding and retrieval
* **Hugging Face Transformers** (e.g., Flan-T5) for sequence-to-sequence generation
* A lightweight **logging** mechanism for tracing queries and generated answers

## Project Structure

```
├── generator/            # Seq2Seq Generator module
│   ├── __init__.py
│   └── generator.py      # Generator class with JSONL logging
│
├── retriever/            # FAISS-backed Retriever module
│   ├── __init__.py
│   └── retriever.py      # Retriever class: read, chunk, embed, index, query
│
├── logger/               # Simple logging utility
│   ├── __init__.py
│   └── logger.py         # Logs question, context IDs, prompt, answer
│
├── pipeline.py           # End-to-end orchestration: load/index docs, retrieve, generate, log
├── test_inputs.json      # Sample test cases (question/context/expected_answer)
├── tests/                # Pytest suite
│   └── test_pipeline.py  # Parametrized tests for non-empty, consistency, expected answer
└── README.md             # This file
```

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   ```

2. **Create a Python environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .\.venv\\Scripts\\activate    # Windows PowerShell
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *Requirements should include:*

   * `transformers`
   * `torch`
   * `sentence-transformers`
   * `faiss-cpu` (or `faiss-gpu`)
   * `PyPDF2`
   * `pytest`

## Usage

1. **Prepare documents**

   * Place your `.txt`, `.md`, or `.pdf` files under `data/` (or update `docs_dir` in `pipeline.py`).

2. **Build or load the embeddings**

   ```bash
   python pipeline.py
   ```

   * If embeddings exist in `./embeddings/`, they will be loaded.
   * Otherwise, documents in `data/` will be indexed and saved.

3. **Run a query**

   ```bash
   python pipeline.py
   # It will prompt question in __main__ or you can modify pipeline.run()
   ```

## Testing

Ensure `test_inputs.json` exists at project root with at least one test case:

```json
[
  {
    "question": "What is the capital of France?",
    "context": "France's capital city is Paris, known for the Eiffel Tower.",
    "expected_answer": "Paris"
  }
]
```

Run the pytest suite:

```bash
pytest -q
```

## Customization

* **Retriever settings**: adjust `chunk_size` and `chunk_overlap` in `Retriever()`
* **Model selection**: change `model_name` in `Generator()` (e.g., `flan-t5-large`)
* **Logging**: modify `Logger` or adjust `log_file` path in `Generator`

 

 
