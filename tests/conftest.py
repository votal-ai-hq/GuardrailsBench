import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
