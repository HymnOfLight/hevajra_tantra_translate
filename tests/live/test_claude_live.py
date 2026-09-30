"""One real claude-check call (costs a fraction of a cent).

Skipped unless ANTHROPIC_API_KEY is set (see tests/conftest.py) and the SDK is
installed. Never run in CI.
"""

from __future__ import annotations

import pytest

from hevajra_matrix.llm.anthropic_client import AnthropicClient
from hevajra_matrix.llm.check import run_check


@pytest.mark.live
def test_claude_check_against_the_live_api():
    pytest.importorskip("anthropic")
    summary = run_check(AnthropicClient())
    assert summary["status"] == "ok", summary
    assert summary["served_model"] == "claude-opus-5-5" and summary["fallback_used"] is False
    assert summary["passed"] is True
    assert summary["request_id"] and summary["usage"]["output_tokens"] > 0
    assert summary["from_cache"] is False
