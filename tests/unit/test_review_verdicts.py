"""review.verdicts: committed format round trip, decision rule and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from hevajra_matrix.core.types import Segment, Verdict
from hevajra_matrix.review.verdicts import (
    COLUMNS,
    VerdictError,
    VerdictRecord,
    decision,
    load,
    read_file,
    save,
    validate,
)

REF = Segment("D417:1a.1.1", "bo_ref", "bo", "ka kha", "1a.1", "1a.1", "verse", fingerprint="aaaaaaaaaaaa")
WIT = Segment("T0892:0587a01.1", "zh_wit", "zh", "abc", "0587a01", "0587a01", "verse", fingerprint="bbbbbbbbbbbb")
INDEX = {REF.id: REF, WIT.id: WIT}


def record(**kw) -> VerdictRecord:
    base = dict(batch_id="verify_1", item_id=f"verify:{REF.id}", task="verify", unit_id=REF.id,
                fingerprint=REF.fingerprint, stratum="pos:abridged:other", inclusion_prob=0.5,
                blind_relation="abridged", blind_wit_ids=(WIT.id,), final_relation="abridged",
                final_wit_ids=(WIT.id,), annotator="ab", blind_date="2026-10-01", final_date="2026-10-02")
    base.update(kw)
    return VerdictRecord(**base)


def test_round_trip(tmp_path: Path) -> None:
    rows = [record(flags=frozenset({"unsure", "scope_list"}), quote_ref="ka", quote_zh="ab", minutes=2.5,
                   note="own words, any language"),
            record(item_id="resolve:x", task="resolve", inclusion_prob=None, polarity_flip=True,
                   blind_relation="reversal", final_relation="reversal")]
    path = tmp_path / "zh_wit" / "verify_1.csv"
    save(rows, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(COLUMNS)
    assert load(tmp_path) == rows
    assert isinstance(load(tmp_path)[0], Verdict)


def test_load_missing_directory_is_empty(tmp_path: Path) -> None:
    assert load(tmp_path / "none") == []


def test_text_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    path.write_text(",".join(COLUMNS) + ",ref_text\n", encoding="utf-8")
    with pytest.raises(VerdictError, match="unexpected \\['ref_text'\\]"):
        read_file(path)


def test_bad_boolean_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    save([record()], path)
    path.write_text(path.read_text(encoding="utf-8").replace(",false,", ",maybe,"), encoding="utf-8")
    with pytest.raises(VerdictError, match="line 2"):
        read_file(path)


def test_decision_rule() -> None:
    assert decision(record()) == ("abridged", (WIT.id,))
    assert decision(record(final_relation="", final_wit_ids=())) is None                  # not revealed yet
    gold = record(task="gold", final_relation="", final_wit_ids=(), blind_relation="equivalent")
    assert decision(gold) == ("equivalent", (WIT.id,))


def test_valid_rows_have_no_problems() -> None:
    orphan = record(item_id="verify:+x", unit_id="+" + WIT.id, fingerprint=WIT.fingerprint,
                    blind_relation="addition", final_relation="has_counterpart", stratum="pos:witness_only",
                    inclusion_prob=1.0)
    assert validate([record(), orphan], INDEX) == []


@pytest.mark.parametrize("change, message", [
    (dict(unit_id="D417:9a.1.1"), "unknown unit"),
    (dict(unit_id="not an id"), "bad reference unit id"),
    (dict(task="guess"), "task 'guess'"),
    (dict(final_relation="omitted"), "final_relation 'omitted'"),
    (dict(final_relation="no_counterpart"), "no_counterpart must have no witness ids"),
    (dict(blind_wit_ids=()), "blind: relation abridged needs witness ids"),
    (dict(final_wit_ids=("T0892:0999a01.1",)), "unknown segment"),
    (dict(final_relation="reversal"), "reversal requires polarity_flip"),
    (dict(flags=frozenset({"interesting"})), "unknown flag"),
    (dict(inclusion_prob=None), "needs its inclusion_prob"),
    (dict(inclusion_prob=1.5), "not in (0, 1]"),
    (dict(final_date="02/10/2026"), "not YYYY-MM-DD"),
    (dict(quote_zh="x" * 61), "licence rule"),
    (dict(blind_relation="", final_relation=""), "no decision"),
    (dict(fingerprint=""), "fingerprint is empty"),
    (dict(unit_id="+" + WIT.id, final_relation="abridged"), "is not one of addition"),
])
def test_validate_reports_problems(change, message) -> None:
    problems = validate([record(**change)], INDEX)
    assert any(message in p for p in problems), problems


def test_duplicate_items_are_reported() -> None:
    assert any("duplicate" in p for p in validate([record(), record()], INDEX))


def test_stale_fingerprint_is_not_a_validation_error() -> None:
    assert validate([record(fingerprint="cccccccccccc")], INDEX) == []
