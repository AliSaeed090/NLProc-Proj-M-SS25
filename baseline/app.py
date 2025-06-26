import streamlit as st
from pipeline import Pipeline
from metrics.recall_evaluator import compute_recall_at_k
from metrics.latency_benchmark import benchmark_latency
import os
import time 

# Initialize pipeline (load or build index)
@st.cache_resource
def load_pipeline():
    return Pipeline()

pipe = load_pipeline()

# App title
st.title("📄 RAG Question Answering + Metrics Dashboard")

# Mode switch
mode = st.radio("Select Mode", ["Ask a Question", "Metrics Dashboard"])

if mode == "Ask a Question":
    # --- FILE UPLOAD ---
    st.sidebar.markdown("### 📂 Upload documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .txt, .md, , .csv or .pdf files",
        type=["csv","txt", "md", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        save_dir = "./data"
        os.makedirs(save_dir, exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join(save_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.sidebar.success(f"Saved: {file.name}")
        # Rebuild index
        st.sidebar.info("Rebuilding index with new documents...")
        pipe.retriever.add_documents([os.path.join(save_dir, f.name) for f in uploaded_files])
        pipe.retriever.save(pipe.embeddings_dir)
        st.sidebar.success("Index updated!")

    # --- QUESTION INPUT ---
    question = st.text_input("Enter your question:", "")

    # --- FILTERS ---
    filter_source_type = st.selectbox("Filter source type", options=["","csv", "pdf", "txt", "md"], index=0)
    filter_date_after = st.text_input("Filter documents after date (YYYY-MM-DD)", "")

    # --- RUN ---
    if st.button("Get Answer") and question.strip():
        with st.spinner("Retrieving and generating..."):
            retrieved = pipe.retriever.query(
                question,
                top_k=5,
                filter_source_type=filter_source_type if filter_source_type else None,
                filter_date_after=filter_date_after if filter_date_after else None
            )

            chunks = [text for _, text in retrieved]
            metas  = [str(meta) for meta, _ in retrieved]

            prompt = pipe.generator.build_prompt(chunks, metas, question, pipe.memory)

            # Live token display
            st.markdown("### 💬 Answer")
            answer_placeholder = st.empty()
            full_answer = ""
            
            answer = pipe.generator.generate_answer(prompt)
            sleep_time = st.sidebar.slider("Token stream speed (seconds)", 0.01, 0.2, 0.05, 0.01)

            for token in answer.split():
                full_answer += token + " "
                answer_placeholder.markdown(full_answer)
                time.sleep(sleep_time)

            pipe.memory.append((question, answer))
            ids = [meta.get("source") for meta, _ in retrieved]
            pipe.logger.log(question, ids, prompt, answer)

            # Show retrieved chunks
            st.markdown("### 📑 Retrieved Context Chunks")
            for meta, text in retrieved:
                st.markdown(f"**Metadata:** `{meta}`")
                st.write(text)
                st.markdown("---")

elif mode == "Metrics Dashboard":
    st.header("📊 Recall@k Evaluation")
    benchmark_data = [
    (
        "What is the role of cyber security in information technology?",
        ["A_Study_Of_Cyber_Security_Challenges_And_Its_Emerg_chunk0"]
    ),
    (
        "What are the major challenges faced by cyber security today?",
        ["A_Study_Of_Cyber_Security_Challenges_And_Its_Emerg_chunk0"]
    ),
    (
        "What is the first thing that comes to mind when thinking about cyber security?",
        ["A_Study_Of_Cyber_Security_Challenges_And_Its_Emerg_chunk0"]
    ),
    (
        "How are governments and companies trying to prevent cyber crimes?",
        ["A_Study_Of_Cyber_Security_Challenges_And_Its_Emerg_chunk0"]
    ),
    (
        "What does this paper focus on regarding cyber security?",
        ["A_Study_Of_Cyber_Security_Challenges_And_Its_Emerg_chunk0"]
    )
]

    k = st.slider("Select k", 1, 10, 5)
    if st.button("Run Recall@k"):
        with st.spinner("Running benchmark..."):
            recall = compute_recall_at_k(pipe.retriever, benchmark_data, k=k)
        st.success(f"Recall@{k}: {recall:.2%}")

    st.header("⚡ Latency Benchmark")
    test_query = st.text_input("Query for latency test", "What is a neural network?")
    if st.button("Run Latency Test"):
        with st.spinner("Running latency test..."):
            latency, answer = benchmark_latency(pipe, test_query)
        st.info(f"Latency: {latency:.2f} seconds")
        st.text_area("Generated Answer", answer, height=200)

# --- SIDEBAR INFO ---
st.sidebar.markdown("### ℹ️ Info")
st.sidebar.markdown(
"""
This app uses:
- **FAISS + SentenceTransformer** for retrieval
- **FLAN-T5 Large** (or LLaMA3/Cerebras) for generation
- **Streamlit** for UI  
- Supports: file uploads, dynamic filters, live token streaming, benchmark metrics
"""
)
