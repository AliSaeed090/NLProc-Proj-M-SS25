# pipeline.py
import os
# Force single-threaded BLAS / OpenMP
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# macOS fork safety disable
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
        self.docs_dir = "./data"  # must be a string, not a tuple

        self.retriever = Retriever()
        self.generator = Generator()
        self.logger = Logger()

        index_path = os.path.join(self.embeddings_dir, "index.faiss")
        docs_meta  = os.path.join(self.embeddings_dir, "docs.pkl")

        if os.path.exists(index_path) and os.path.exists(docs_meta):
            print(f"✅ Loading existing embeddings from '{self.embeddings_dir}'")
            self.retriever.load(self.embeddings_dir)
        else:
            if not os.path.isdir(self.docs_dir):
                raise FileNotFoundError(f"Documents folder '{self.docs_dir}' not found.")
            print(f"⚙️  No embeddings found—indexing documents in '{self.docs_dir}'...")
            patterns = ["*.txt", "*.md", "*.pdf"]
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
        retrieved    = self.retriever.query(question)
        chunks       = [text for _, text in retrieved]
        prompt       = self.generator.build_prompt(chunks, question)
        answer       = self.generator.generate_answer(prompt)
        retrieved_ids = [doc_id for doc_id, _ in retrieved]
        self.logger.log(question, retrieved_ids, prompt, answer)
        return answer

if __name__ == "__main__":
    pipe = Pipeline()
    question = "what is nural network"
    print(pipe.run(question))

