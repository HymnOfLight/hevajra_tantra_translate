"""evaluation.gold: the ``convention`` column (collate.v2), its three states, validation,
legacy files and the conversions to and from verdicts."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hevajra_matrix.evaluation import gold as G

from test_evaluation_gold import GOLD, REF, WINDOWS, WIT, only, row

NAMED = (
    row("D:1a.1.1", "equivalent", ["T:1.1"], convention="transcribed_name"),
    row("D:1a.2.1", "abridged", ["T:2.1"], convention="standing_rendering"),
    row("D:1a.3.1", "no_counterpart", convention="none"),
    row("D:1a.4.1", "lacuna", window="w02", convention="none"),
    row("D:1a.5.1", "paraphrase", ["T:5.1"], window="w02"),                 # not annotated
    only("T:9.1", "addition"),
)


def saved(tmp_path: Path, rows) -> Path:
    path = tmp_path / WIT / "test.csv"
    G.save(G.GoldSet("test", WIT, tuple(rows), WINDOWS, REF), path)
    G.write_windows(WINDOWS, path.with_name("windows.csv"))
    return path


def test_convention_round_trip_keeps_the_three_states(tmp_path: Path):
    back = G.load(saved(tmp_path, NAMED), reference=REF)
    assert [r.convention for r in back.rows] == ["transcribed_name", "standing_rendering", "none", "none", "", ""]
    assert back == G.GoldSet("test", WIT, NAMED, WINDOWS, REF)
    assert G.GOLD_COLUMNS.index("convention") == G.GOLD_COLUMNS.index("relation") + 1


def test_a_file_without_the_column_still_loads(tmp_path: Path):
    path = saved(tmp_path, GOLD.rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    k = lines[0].split(",").index("convention")
    path.write_text("\n".join(",".join(x for i, x in enumerate(line.split(",")) if i != k) for line in lines) + "\n",
                    encoding="utf-8")
    assert "convention" not in path.read_text(encoding="utf-8").splitlines()[0]
    back = G.load(path, reference=REF)
    assert back.rows == GOLD.rows and all(r.convention == "" for r in back.rows)


@pytest.mark.parametrize("bad, message", [
    (row("D:1.1", "equivalent", ["T:1.1"], convention="name"), "convention 'name' is not one of"),
    (row("D:1.1", "no_counterpart", convention="transcribed_name"), "needs a linked relation"),
    (row("D:1.1", "unresolved", convention="standing_rendering"), "needs a linked relation"),
    (dataclasses.replace(only("T:1.1", "addition"), convention="transcribed_name"), "carries no convention"),
    (row("D:1.1", "equivalent", ["T:1.1"], flags=frozenset({"transcribed_name"})), "is a convention"),
    (row("D:1.1", "equivalent", ["T:1.1"], flags=frozenset({"convention_unsupported"})), "code checks only"),
])
def test_invalid_conventions_are_refused(tmp_path: Path, bad, message):
    with pytest.raises(G.GoldError, match=message):
        G.load(saved(tmp_path, [bad]))
    with pytest.raises(G.GoldError, match=message):
        G.from_verdicts(G.to_verdicts(G.GoldSet("test", WIT, (bad,))), "test", WIT)


def test_verdict_round_trip_keeps_the_convention():
    gold = G.GoldSet("test", WIT, NAMED, (), REF)
    verdicts = G.to_verdicts(gold)
    assert [v.blind_convention for v in verdicts] == [r.convention for r in NAMED]
    assert G.from_verdicts(verdicts, "test", WIT, REF).rows == NAMED


def test_gold_links_carry_the_convention_as_a_flag():
    links = {link.ref_id: link for link in G.GoldSet("test", WIT, NAMED, WINDOWS, REF).alignment.links}
    assert links["D:1a.1.1"].flags == {"transcribed_name"} and links["D:1a.2.1"].flags == {"standing_rendering"}
    assert links["D:1a.3.1"].flags == frozenset() and links["D:1a.5.1"].flags == frozenset()
