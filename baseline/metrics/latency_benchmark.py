# metrics/latency_benchmark.py
import time
from typing import List, Tuple

def benchmark_latency(pipeline, query: str) -> Tuple[float, str]:
    """
    Measure latency for a single query.

    Args:
        pipeline: your Pipeline instance
        query: user query

    Returns:
        latency (seconds), answer
    """
    start = time.time()
    answer = pipeline.run(query)
    end = time.time()
    latency = end - start
    return latency, answer
