import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# src layout: importable straight from a checkout, no install required.
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIXTURES = Path(__file__).parent / "fixtures"
SEED_CSV = FIXTURES / "seed_sample.csv"
TEST_JSONL = ROOT / "data" / "test.jsonl"


@pytest.fixture(scope="session")
def seed_csv():
    return str(SEED_CSV)


@pytest.fixture(scope="session")
def dataset():
    from guardrailgym.schema import load_items
    if not TEST_JSONL.exists():
        pytest.skip("data/test.jsonl is not present")
    return load_items(TEST_JSONL)


SEED_CORPUS = ROOT / "data" / "seed" / "custom_policy_5k_llmshield.csv"


@pytest.fixture(scope="session")
def seed_corpus():
    """The real seed CSV. Skips rather than fails where it is not distributed."""
    if not SEED_CORPUS.exists():
        pytest.skip("the seed corpus is not present")
    return str(SEED_CORPUS)
