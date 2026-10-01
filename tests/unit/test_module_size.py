"""Engineering rule (impl_decisions, docs/contributing.md): one responsibility per module,
<= about 400 lines. The tolerance allows a few lines over, never a module that needs splitting."""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "hevajra_matrix"
MAX_LINES = 450


def test_no_module_exceeds_the_size_limit():
    sizes = {p.relative_to(PACKAGE).as_posix(): len(p.read_text(encoding="utf-8").splitlines())
             for p in PACKAGE.rglob("*.py")}
    too_big = {name: n for name, n in sizes.items() if n > MAX_LINES}
    assert not too_big, f"split these modules (limit ~400 lines): {too_big}"


def test_llm_failure_reasons_have_one_definition():
    """The refusal/truncation/invalid reason mapping lives in core.types only (no drifting copies)."""
    copies = [p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*.py")
              if "def _failure_reason" in p.read_text(encoding="utf-8")]
    assert copies == []
