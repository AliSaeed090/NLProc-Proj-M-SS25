from typing import List, Tuple

def _retrieve_doc_ids(retriever, question: str, k: int) -> List[str]:
    """Helper function to retrieve document IDs from retriever."""
    results = retriever.query(question, top_k=k)
    return [
        meta["source"].replace(".pdf", "").replace(".md", "").replace(".txt", "") +
        f"_chunk{meta.get('chunk_id', meta.get('row_id', 0))}"
        for (meta, _) in results
    ]

def compute_recall_at_k(
    retriever,
    benchmark_data: List[Tuple[str, List[str]]],
    k: int = 5
) -> float:
    """
    Compute Recall@k for benchmark queries (binary version: 1 if any relevant doc is retrieved).

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
        retrieved_ids = _retrieve_doc_ids(retriever, question, k)

        print(f"Question: {question}")
        print(f"Gold IDs: {gold_doc_ids}")
        print(f"Retrieved IDs: {retrieved_ids}\n")

        if any(doc_id in retrieved_ids for doc_id in gold_doc_ids):
            hits += 1

    recall_at_k = hits / total if total > 0 else 0.0
    return recall_at_k

def compute_precision_recall_f1_at_k(
    retriever,
    benchmark_data: List[Tuple[str, List[str]]],
    k: int = 5
) -> Tuple[float, float, float]:
    """
    Compute Precision@k, Recall@k, and F1-score@k for benchmark queries.

    Args:
        retriever: your Retriever instance
        benchmark_data: list of (question, list of gold doc_ids)
        k: number of top documents to retrieve

    Returns:
        Tuple of (precision@k, recall@k, f1@k) values (0.0 - 1.0)
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_queries = len(benchmark_data)
    
    if total_queries == 0:
        return 0.0, 0.0, 0.0

    for question, gold_doc_ids in benchmark_data:
        retrieved_ids = _retrieve_doc_ids(retriever, question, k)
        gold_set = set(gold_doc_ids)
        retrieved_set = set(retrieved_ids)
        
        # Calculate true positives (relevant documents retrieved)
        tp = len(gold_set & retrieved_set)
        
        # Precision: TP / (TP + FP) = TP / k
        precision = tp / k if k > 0 else 0.0
        
        # Recall: TP / (TP + FN) = TP / total relevant
        recall = tp / len(gold_doc_ids) if len(gold_doc_ids) > 0 else 0.0
        
        # F1: harmonic mean of precision and recall
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    # Calculate macro averages
    avg_precision = total_precision / total_queries
    avg_recall = total_recall / total_queries
    avg_f1 = total_f1 / total_queries

    return avg_precision, avg_recall, avg_f1

# Individual metric functions for convenience
def compute_precision_at_k(retriever, benchmark_data, k=5) -> float:
    """Compute average Precision@k"""
    precision, _, _ = compute_precision_recall_f1_at_k(retriever, benchmark_data, k)
    return precision

def compute_recall_at_k(retriever, benchmark_data, k=5) -> float:
    """Compute average Recall@k"""
    _, recall, _ = compute_precision_recall_f1_at_k(retriever, benchmark_data, k)
    return recall

def compute_f1_at_k(retriever, benchmark_data, k=5) -> float:
    """Compute average F1-score@k"""
    _, _, f1 = compute_precision_recall_f1_at_k(retriever, benchmark_data, k)
    return f1