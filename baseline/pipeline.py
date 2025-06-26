import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from glob import glob
from retriever.retreiver import Retriever
from generator.generator import Generator
from logger.logger import Logger

class Pipeline:
    def __init__(self):
        self.embeddings_dir = "./embeddings"
        self.docs_dir = "./data"

        self.retriever = Retriever()
        self.generator = Generator()
        self.logger = Logger()

        self.memory = []

        index_path = os.path.join(self.embeddings_dir, "index.faiss")
        docs_meta  = os.path.join(self.embeddings_dir, "docs.pkl")

        if os.path.exists(index_path) and os.path.exists(docs_meta):
            print(f"✅ Loading existing embeddings from '{self.embeddings_dir}'")
            self.retriever.load(self.embeddings_dir)
        else:
            if not os.path.isdir(self.docs_dir):
                raise FileNotFoundError(f"Documents folder '{self.docs_dir}' not found.")
            print(f"⚙️  No embeddings found—indexing documents in '{self.docs_dir}'...")
            patterns = ["*.txt", "*.md", "*.pdf", "*.csv"]
            paths = []
            for pat in patterns:
                paths += glob(os.path.join(self.docs_dir, pat))
            if not paths:
                raise ValueError(f"No .txt/.md/.pdf files found in '{self.docs_dir}'.")
            self.retriever.add_documents(paths)
            os.makedirs(self.embeddings_dir, exist_ok=True)
            self.retriever.save(self.embeddings_dir)
            print(f"✅ Embeddings built and saved to '{self.embeddings_dir}'")

    def run(self, question: str) -> str:
        # Retrieve with optional filters
        retrieved = self.retriever.query(
            question,
            top_k=5,
            filter_source_type=None,        # e.g., "pdf" to filter
            filter_date_after=None          # e.g., "2023-01-01" to filter
        )

        chunks = [text for _, text in retrieved]
        metas  = [str(meta) for meta, _ in retrieved]
        ids    = [meta.get("source") for meta, _ in retrieved]

        # Build prompt with memory
        prompt = self.generator.build_prompt(chunks, metas, question, self.memory)

        # Generate
        answer = self.generator.generate_answer(prompt)

        # # Basic post-check: warn if not grounded
        # if ("I don’t know" not in answer) and not any(chunk in answer for chunk in chunks):
        #     answer = "⚠️ The answer may not be based on the provided context.\n" + answer

        # Update memory
        self.memory.append((question, answer))

        # Log
        self.logger.log(question, ids, prompt, answer)
        return answer

if __name__ == "__main__":
    pipe = Pipeline()
    q = "What is neural network?"
    print(pipe.run(q))
