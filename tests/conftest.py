"""Shared test configuration.

Test tiers
    tests/unit/         default; no network; reads only data/fixtures and data/lexicon
    tests/integration/  marked ``realdata``; skipped unless HEVAJRA_RAW_DIR points at the
                        fetched CBETA and Derge files (``hevajra-matrix fetch``)
    tests/live/         marked ``live``; skipped unless ANTHROPIC_API_KEY is set

Multilingual test inputs live in ``data/fixtures/`` (the language policy forbids
non-Latin script in ``.py`` files).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "data" / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    raw = os.environ.get("HEVAJRA_RAW_DIR")
    skip_raw = pytest.mark.skip(reason="set HEVAJRA_RAW_DIR to run real-data tests")
    skip_live = pytest.mark.skip(reason="set ANTHROPIC_API_KEY to run live API tests")
    for item in items:
        if "realdata" in item.keywords and not (raw and Path(raw).is_dir()):
            item.add_marker(skip_raw)
        if "live" in item.keywords and not os.environ.get("ANTHROPIC_API_KEY"):
            item.add_marker(skip_live)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def raw_dir() -> Path:
    """Directory with the fetched source texts (real-data tier only)."""
    return Path(os.environ["HEVAJRA_RAW_DIR"])
