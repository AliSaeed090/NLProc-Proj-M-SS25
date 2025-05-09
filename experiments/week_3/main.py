from retriever import Retriever

# 1) Initialize
ret = Retriever(
    model_name="all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50
)

# 2) Add your files (txt, md, pdf)
ret.add_documents(["./docs/intro.md", "./papers/report.pdf"])

# 3) Query
results = ret.query("what is the conclusion?", top_k=3)
for text, distance in results:
    print(f"{distance:.4f}\t{text}")

# 4) Save / Load
ret.save("my_index.faiss", "my_chunks.pkl")

# later...
new_ret = Retriever()
new_ret.load("my_index.faiss", "my_chunks.pkl")
