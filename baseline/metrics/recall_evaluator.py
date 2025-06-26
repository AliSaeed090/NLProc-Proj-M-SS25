# metrics/recall_evaluator.py
from typing import List, Tuple

def compute_recall_at_k(
    retriever,
    benchmark_data: List[Tuple[str, List[str]]],
    k: int = 5
) -> float:
    """
    Compute Recall@k for benchmark queries.

    Args:
        retriever: your Retriever instance
        benchmark_data: list of (question, list of gold doc_ids)
        k: number of top documents to retrieve

    Returns:
        recall@k value (0.0 - 1.0)
    """
    hits = 0
    total = len(benchmark_data)

    for question, gold_doc_ids in benchmark_data:
        results = retriever.query(question, top_k=k)
        retrieved_ids = [meta["source"].replace(".pdf", "").replace(".md", "").replace(".txt", "") +
                        f"_chunk{meta.get('chunk_id', meta.get('row_id', 0))}" for (meta, _) in results]

        print(f"Question: {question}")
        print(f"Gold IDs: {gold_doc_ids}")
        print(f"Retrieved IDs: {retrieved_ids}\n")

        if any(doc_id in retrieved_ids for doc_id in gold_doc_ids):
            hits += 1

    recall_at_k = hits / total if total > 0 else 0.0
    return recall_at_k
