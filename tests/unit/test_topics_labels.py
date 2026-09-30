"""topics.labels: the committed label file, stale labels, and topic agreement (critique A6, A7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hevajra_matrix.core.types import Segment
from hevajra_matrix.topics import (
    LABEL_COLUMNS,
    TOPICS,
    TopicError,
    TopicLabel,
    agreement,
    cohen_kappa,
    current_labels,
    format_topics,
    human_agreement,
    load_codebook,
    load_labels,
    parse_topics,
    prelabel_agreement,
    write_labels,
)

ROOT = Path(__file__).resolve().parents[2]
HEADER = ",".join(LABEL_COLUMNS)


@pytest.fixture(scope="module")
def cb():
    return load_codebook(ROOT / "data" / "codebook" / "topics.yaml")


def _csv(tmp_path: Path, *rows: str, header: str = HEADER, bom: bool = False) -> Path:
    path = tmp_path / "bo_derge_D417_418.csv"
    path.write_text("\n".join((header, *rows)) + "\n", encoding="utf-8-sig" if bom else "utf-8")
    return path


def _label(uid: str, topics=(), second=(), prelabel=(), fp: str = "fp") -> TopicLabel:
    return TopicLabel(uid, fp, frozenset(topics), frozenset(prelabel), "A" if topics else "", "",
                      frozenset(second), "B" if second else "", "")


# --------------------------------------------------------------------------- cells
def test_parse_and_format_topics_round_trip(cb):
    assert parse_topics(" sexual ; female_agent;", cb) == {"sexual", "female_agent"}
    assert parse_topics("", cb) == frozenset()
    assert format_topics({"ritual", "sexual"}) == "sexual;ritual"   # vocabulary order
    assert format_topics(set()) == ""
    with pytest.raises(TopicError, match="neutral must appear alone"):
        format_topics({"neutral", "harm"})


# --------------------------------------------------------------------------- load / write
def test_load_labels_reads_every_column(tmp_path, cb):
    path = _csv(tmp_path,
                "D418:9a.1.2,a1b2c3d4e5f6,bone_corpse;ritual,bone_corpse,AB,2026-10-01,bone_corpse,CD,2026-10-02",
                "D418:9a.3.1,0f0f0f0f0f0f,neutral,,AB,2026-10-01,,,",
                "D418:9a.3.2,123456789abc,,mantra_control,,,,,")
    labels = load_labels(path, cb)
    first = labels["D418:9a.1.2"]
    assert first == TopicLabel("D418:9a.1.2", "a1b2c3d4e5f6", frozenset({"bone_corpse", "ritual"}),
                               frozenset({"bone_corpse"}), "AB", "2026-10-01", frozenset({"bone_corpse"}),
                               "CD", "2026-10-02")
    assert first.group == "sensitive" and labels["D418:9a.3.1"].group == "neutral"
    assert labels["D418:9a.3.2"].group == "unlabelled"          # pre-labelled, not yet confirmed


def test_write_then_load_round_trips(tmp_path, cb):
    labels = [_label("D417:1b.2.1", {"frame"}, {"frame"}, {"frame"}),
              _label("D417:1b.2.3", {"mantra_control"}),
              _label("D417:1b.3.1")]
    path = tmp_path / "out.csv"
    write_labels(path, labels)
    assert path.read_text(encoding="utf-8").splitlines()[0] == HEADER
    assert list(load_labels(path, cb).values()) == labels


def test_columns_may_come_in_any_order_and_a_bom_is_tolerated(tmp_path, cb):
    header = ",".join(reversed(LABEL_COLUMNS))     # second_date, second_coder, ..., unit_id
    path = _csv(tmp_path, ",,,,AB,,harm,fp,u1", header=header, bom=True)
    label = load_labels(path, cb)["u1"]
    assert (label.topics, label.coder, label.fingerprint) == ({"harm"}, "AB", "fp")


@pytest.mark.parametrize("header", [
    HEADER + ",text",                           # a text column must never be committed
    HEADER.replace(",second_date", ""),
    HEADER.replace("topics,", "topic,", 1),
])
def test_wrong_columns_are_refused(tmp_path, cb, header):
    with pytest.raises(TopicError, match="columns must be exactly"):
        load_labels(_csv(tmp_path, header=header), cb)


@pytest.mark.parametrize("row, message", [
    ("u1,fp,violence,,AB,,,,", "topics: unknown topic(s) ['violence']"),
    ("u1,fp,sexual harm,,AB,,,,", "separate topics with ';'"),
    ("u1,fp,neutral;harm,,AB,,,,", "topics: neutral must appear alone"),
    ("u1,fp,harm,theft;neutral,AB,,,,", "prelabel_topics: neutral must appear alone"),
    ("u1,fp,harm,,AB,,flesh,CD,", "second_topics: unknown topic(s) ['flesh']"),
    ("u1,fp,harm,,,,,,", "coder is empty"),
    ("u1,fp,harm,,AB,,harm,,", "second_coder is empty"),
    ("u1,fp,harm,,AB,,harm,ab,", "second_coder must be a different person"),
    ("u1,fp,harm,,AB,30.9.2026,,,", "date '30.9.2026' is not YYYY-MM-DD"),
    ("u1,fp,harm,,AB,,harm,CD,2026/10/01", "second_date '2026/10/01' is not YYYY-MM-DD"),
    (",fp,harm,,AB,,,,", "unit_id is empty"),
    ("u1,,harm,,AB,,,,", "fingerprint is empty"),
    ("u1,fp,harm,,AB,,,", "exactly one value per column"),
    ("u1,fp,harm,,AB,,,,,extra", "exactly one value per column"),
])
def test_invalid_rows_are_rejected_with_a_clear_error(tmp_path, cb, row, message):
    with pytest.raises(TopicError) as err:
        load_labels(_csv(tmp_path, row), cb)
    assert message in str(err.value) and "line 2" in str(err.value)


def test_unknown_topic_error_lists_the_allowed_values(tmp_path, cb):
    with pytest.raises(TopicError) as err:
        load_labels(_csv(tmp_path, "u1,fp,violence,,AB,,,,"), cb)
    assert all(t in str(err.value) for t in TOPICS)


def test_every_invalid_row_is_reported_and_duplicates_are_refused(tmp_path, cb):
    path = _csv(tmp_path, "u1,fp,harm,,AB,,,,", "u2,fp,violence,,AB,,,,", "u1,fp,theft,,AB,,,,",
                "u3,fp,neutral;frame,,AB,,,,")
    with pytest.raises(TopicError) as err:
        load_labels(path, cb)
    text = str(err.value)
    assert "3 invalid row(s)" in text
    assert "line 3" in text and "line 4: duplicate unit_id u1" in text and "line 5" in text


def test_an_empty_file_with_the_right_header_has_no_labels(tmp_path, cb):
    assert load_labels(_csv(tmp_path), cb) == {}


def test_write_labels_refuses_invalid_sets(tmp_path):
    with pytest.raises(TopicError):
        write_labels(tmp_path / "x.csv", [_label("u1", {"neutral", "harm"})])


# --------------------------------------------------------------------------- stale labels
def test_current_labels_sets_aside_changed_and_missing_units():
    seg = lambda uid, fp: Segment(uid, "bo_derge_D417_418", "bo", "x", "1a.1", "1a.1", "prose", fingerprint=fp)
    labels = {"u1": _label("u1", {"harm"}, fp="aaa"), "u2": _label("u2", {"frame"}, fp="bbb"),
              "u3": _label("u3", {"neutral"}, fp="ccc")}
    current, stale = current_labels(labels, [seg("u1", "aaa"), seg("u2", "changed")])
    assert list(current) == ["u1"] and stale == ("u2", "u3")


# --------------------------------------------------------------------------- kappa
def test_cohen_kappa_hand_computed():
    # 2x2: a=3 (both), b=1, c=1, d=5 -> p_o 0.8, p_e 0.4*0.4 + 0.6*0.6 = 0.52, kappa 0.28/0.48
    pairs = [(True, True)] * 3 + [(True, False), (False, True)] + [(False, False)] * 5
    assert cohen_kappa(pairs) == pytest.approx(0.28 / 0.48)
    # three categories: p_o = 4/6; first S,S,S,N,N,F; second S,S,N,N,F,F -> p_e = (3*2 + 2*2 + 1*2)/36
    three = list(zip("SSSNNF", "SSNNFF"))
    assert cohen_kappa(three) == pytest.approx((4 / 6 - 12 / 36) / (1 - 12 / 36))
    assert cohen_kappa([("S", "S"), ("N", "N")]) == pytest.approx(1.0)
    assert cohen_kappa([("S", "N"), ("N", "S")]) == pytest.approx(-1.0)


def test_cohen_kappa_is_undefined_without_pairs_or_variation():
    assert cohen_kappa([]) is None
    assert cohen_kappa([("neutral", "neutral")] * 4) is None


def test_agreement_per_topic_and_on_the_group():
    pairs = ([({"sexual"}, {"sexual"})] * 3 + [({"sexual"}, {"neutral"}), ({"neutral"}, {"sexual"})]
             + [({"neutral"}, {"neutral"})] * 5)
    result = agreement([(frozenset(a), frozenset(b)) for a, b in pairs])
    assert result.n_units == 10
    sexual = result.per_topic["sexual"]
    assert (sexual.both, sexual.first_only, sexual.second_only) == (3, 1, 1)
    assert sexual.kappa == pytest.approx(0.28 / 0.48)
    assert sexual.positive_agreement == pytest.approx(6 / 8)
    assert result.group_kappa == pytest.approx(0.28 / 0.48)       # sensitive vs neutral
    assert result.group_agreement == pytest.approx(0.8)
    unused = result.per_topic["theft"]
    assert unused.kappa is None and unused.positive_agreement is None
    assert set(result.per_topic) == set(TOPICS)


def test_agreement_on_no_units_is_not_estimable():
    result = agreement([])
    assert (result.n_units, result.group_kappa, result.group_agreement) == (0, None, None)


def test_human_agreement_uses_the_two_coder_columns_only():
    base = {"u1": _label("u1", {"harm"}, {"harm"}), "u2": _label("u2", {"frame"}, {"neutral"}),
            "u3": _label("u3", {"sexual"}), "u4": _label("u4", (), {"sexual"})}
    with_prelabels = {u: TopicLabel(x.unit_id, x.fingerprint, x.topics, frozenset({"theft"}), x.coder, x.date,
                                    x.second_topics, x.second_coder, x.second_date) for u, x in base.items()}
    result = human_agreement(base)
    assert result.n_units == 2                          # only units coded by both humans
    assert result == human_agreement(with_prelabels)    # prelabels never enter kappa_topic (A6)
    assert result.group_agreement == pytest.approx(0.5)


def test_prelabel_agreement_is_separate_and_names_its_human_column():
    labels = {"u1": _label("u1", {"harm"}, {"ritual"}, prelabel={"harm"}),
              "u2": _label("u2", {"frame"}, {"frame"}, prelabel={"neutral"}),
              "u3": _label("u3", {"sexual"}, prelabel={"sexual"}),             # not double-coded
              "u4": _label("u4", {"neutral"}, {"neutral"})}                  # no usable prelabel
    blind = prelabel_agreement(labels)                     # against the blind second coder
    anchored = prelabel_agreement(labels, against="first")
    assert blind.n_units == 2 and anchored.n_units == 3
    assert blind.per_topic["harm"].first_only == 1 and anchored.per_topic["harm"].both == 1
    with pytest.raises(ValueError):
        prelabel_agreement(labels, against="both")
