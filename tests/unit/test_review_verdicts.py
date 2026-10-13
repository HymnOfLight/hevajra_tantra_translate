"""review.verdicts: committed format round trip, decision rule and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from hevajra_matrix.core.types import Segment, Verdict
from hevajra_matrix.review.verdicts import (
    COLUMNS,
    CONVENTION_COLUMNS,
    CONVENTIONS,
    VerdictError,
    VerdictRecord,
    decision,
    decision_convention,
    load,
    normalise_convention,
    read_file,
    same_convention,
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
                   note="own words, any language", machine_convention="none", blind_convention="transcribed_name",
                   final_convention="none"),
            record(item_id="resolve:x", task="resolve", inclusion_prob=None, polarity_flip=True,
                   blind_relation="reversal", final_relation="reversal")]
    path = tmp_path / "zh_wit" / "verify_1.csv"
    save(rows, path)
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(COLUMNS)
    assert load(tmp_path) == rows
    assert isinstance(load(tmp_path)[0], Verdict)


def test_machine_polarity_flip_is_committed_and_old_files_still_load(tmp_path: Path) -> None:
    # The reveal sheet exported and imported machine_polarity_flip, but the committed format
    # dropped it, so the misclassification table lost the machine's polarity call.
    path = tmp_path / "zh_wit" / "verify_1.csv"
    save([record(machine_relation="paraphrase", machine_status="PRESENT", machine_polarity_flip=True)], path)
    assert "machine_polarity_flip" in path.read_text(encoding="utf-8").splitlines()[0]
    assert read_file(path)[0].machine_polarity_flip is True
    legacy = [c for c in COLUMNS if c != "machine_polarity_flip"]
    lines = path.read_text(encoding="utf-8").splitlines()
    index = COLUMNS.index("machine_polarity_flip")
    old = [",".join(legacy)] + [",".join(v for k, v in enumerate(line.split(",")) if k != index) for line in lines[1:]]
    path.write_text("\n".join(old) + "\n", encoding="utf-8")
    assert read_file(path)[0].machine_polarity_flip is False


def _drop_columns(path: Path, dropped: set[str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    keep = [k for k, c in enumerate(lines[0].split(",")) if c not in dropped]
    path.write_text("\n".join(",".join(line.split(",")[k] for k in keep) for line in lines) + "\n", encoding="utf-8")


def test_conventions_are_committed_and_old_files_still_load(tmp_path: Path) -> None:
    assert len(COLUMNS) == 27 and CONVENTION_COLUMNS <= set(COLUMNS)
    assert COLUMNS.index("machine_convention") == COLUMNS.index("machine_polarity_flip") + 1
    assert COLUMNS.index("blind_convention") == COLUMNS.index("blind_wit_ids") + 1
    assert COLUMNS.index("final_convention") == COLUMNS.index("final_wit_ids") + 1
    path = tmp_path / "zh_wit" / "verify_1.csv"
    rec = record(machine_convention="transcribed_name", blind_convention="none", final_convention="transcribed_name")
    save([rec], path)
    assert read_file(path) == [rec]
    _drop_columns(path, set(CONVENTION_COLUMNS))                     # a 24-column file of before collate.v2
    old = read_file(path)[0]
    assert (old.machine_convention, old.blind_convention, old.final_convention) == ("", "", "")
    assert old == record()
    _drop_columns(path, {"machine_polarity_flip"})                    # and a 23-column one
    assert read_file(path)[0].machine_polarity_flip is False


def test_a_partially_legacy_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "zh_wit" / "verify_1.csv"
    save([record(blind_convention="none", final_convention="none")], path)
    _drop_columns(path, {"final_convention"})
    with pytest.raises(VerdictError, match="all three or none"):
        read_file(path)


def test_convention_states() -> None:
    assert CONVENTIONS == ("none", "standing_rendering", "transcribed_name")
    assert normalise_convention("") == normalise_convention(None) == normalise_convention("  ") == "none"
    assert normalise_convention(" transcribed_name ") == "transcribed_name"
    assert normalise_convention("?") == normalise_convention(" ? ") == ""     # not decided yet
    assert same_convention("", "none") and same_convention("none", "") and same_convention("", "")
    assert not same_convention("", "transcribed_name") and not same_convention("none", "standing_rendering")


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


def test_a_gold_row_in_a_verdict_file_is_refused(tmp_path: Path) -> None:
    # Gold belongs in data/annotations/gold/; loaded from here it would be applied with
    # review-verdict precedence and be invisible to G1 (review finding, known issue 5).
    path = tmp_path / "zh_wit" / "gold_1.csv"
    save([record(), record(item_id=f"gold:{REF.id}", task="gold", stratum="", inclusion_prob=None)], path)
    with pytest.raises(VerdictError, match="line 3: task 'gold' does not belong"):
        load(tmp_path)


def test_decision_rule() -> None:
    assert decision(record()) == ("abridged", (WIT.id,))
    assert decision(record(final_relation="", final_wit_ids=())) is None                  # not revealed yet
    gold = record(task="gold", final_relation="", final_wit_ids=(), blind_relation="equivalent")
    assert decision(gold) == ("equivalent", (WIT.id,))


def test_decision_convention_follows_the_decision_rule() -> None:
    both = dict(blind_convention="transcribed_name", final_convention="none")
    assert decision_convention(record(**both)) == "none"                                        # final
    assert decision_convention(record(final_relation="", final_wit_ids=(), **both)) == ""        # not revealed yet
    gold = dict(task="gold", final_relation="", final_wit_ids=(), final_convention="")
    assert decision_convention(record(**gold, blind_convention="transcribed_name")) == "transcribed_name"
    assert decision_convention(record(**gold)) == ""                                             # legacy gold
    resolve = record(task="resolve", blind_convention="transcribed_name", final_convention="transcribed_name")
    assert decision_convention(resolve) == "transcribed_name"


def test_valid_rows_have_no_problems() -> None:
    orphan = record(item_id="verify:+x", unit_id="+" + WIT.id, fingerprint=WIT.fingerprint,
                    blind_relation="addition", final_relation="has_counterpart", stratum="pos:witness_only",
                    inclusion_prob=1.0, blind_convention="none", final_convention="none")
    named = record(item_id="verify:n", machine_convention="none", blind_convention="transcribed_name",
                   final_convention="standing_rendering")
    absent = record(item_id="verify:a", final_relation="no_counterpart", final_wit_ids=(), final_convention="none")
    assert validate([record(), orphan, named, absent], INDEX) == []


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
    (dict(final_convention="proper_name"), "convention 'proper_name' is not one of none"),
    (dict(machine_convention="name"), "machine_convention 'name'"),
    (dict(final_relation="no_counterpart", final_wit_ids=(), final_convention="transcribed_name"),
     "final: convention transcribed_name needs a linked relation"),
    (dict(final_relation="lacuna", final_wit_ids=(), final_convention="standing_rendering"),
     "needs a linked relation"),
    (dict(blind_relation="unresolved", blind_wit_ids=(), blind_convention="transcribed_name"),
     "blind: convention transcribed_name needs a linked relation"),
    (dict(unit_id="+" + WIT.id, blind_relation="addition", final_relation="has_counterpart",
          final_convention="transcribed_name"), "needs a linked relation"),
    (dict(final_relation="", final_wit_ids=(), final_convention="none"), "final_convention 'none' without"),
    (dict(flags=frozenset({"transcribed_name"})), "is a convention: put it in the convention column"),
    (dict(flags=frozenset({"convention_unsupported"})), "set by the code checks only"),
])
def test_validate_reports_problems(change, message) -> None:
    problems = validate([record(**change)], INDEX)
    assert any(message in p for p in problems), problems


def test_duplicate_items_are_reported() -> None:
    assert any("duplicate" in p for p in validate([record(), record()], INDEX))


def test_stale_fingerprint_is_not_a_validation_error() -> None:
    assert validate([record(fingerprint="cccccccccccc")], INDEX) == []
