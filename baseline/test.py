import os
import json
import pytest

# Ensure single-threaded BLAS/OpenMP for consistency
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from pipeline import Pipeline

# Load test cases from JSON file
TEST_INPUTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'test_inputs.json')
with open(TEST_INPUTS_PATH, 'r', encoding='utf-8') as f:
    TEST_CASES = json.load(f)

@pytest.fixture(scope='session')
def pipeline():
    """Initialize and return a Pipeline instance for testing."""
    return Pipeline()

@pytest.mark.parametrize('case', TEST_CASES)
def test_pipeline_non_empty_and_expected(case, pipeline):
    """
    For each test case, ensure the pipeline returns a non-empty answer and contains the expected substring.
    """
    question = case['question']
    expected = case['expected_answer']

    answer = pipeline.run(question)

    # Non-empty
    assert isinstance(answer, str) and answer.strip(), \
        f"Empty answer for question: {question}"
    # Contains expected term (case-insensitive)
    assert expected.lower() in answer.lower(), \
        f"Answer '{answer}' does not contain expected '{expected}' for question: {question}"

@pytest.mark.parametrize('case', TEST_CASES)
def test_pipeline_consistency(case, pipeline):
    """
    Check that repeated runs on the same question produce identical answers.
    """
    question = case['question']

    answer1 = pipeline.run(question)
    answer2 = pipeline.run(question)

    assert answer1 == answer2, \
        f"Inconsistent answers for '{question}': '{answer1}' vs '{answer2}'"
