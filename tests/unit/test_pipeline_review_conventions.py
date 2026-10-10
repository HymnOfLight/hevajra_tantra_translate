"""Conventions in the review stages (collate.v2): the verified glossary beside the sheets,
the import's glossary check, and a blind convention fixed once revealed."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import yaml

from hevajra_matrix.collate.verify import CheckLexicon
from hevajra_matrix.core.lexicon import Negators, StandingEntry, StandingRenderings, VariantForms
from hevajra_matrix.core.textnorm import bo_syllables
from hevajra_matrix.core.types import Segment
from hevajra_matrix.pipeline import human_data as ann
from hevajra_matrix.pipeline.context import load_texts
from hevajra_matrix.pipeline.review import glossary_check
from hevajra_matrix.review import verdicts as verdict_files

from test_pipeline_review import WITNESS, cli, finished, fill  # noqa: F401  (finished is a fixture)
from test_pipeline_support import context

TSHEG = "\u0f0b"
KYE, RDO, RJE, SA = "\u0f40\u0fb1\u0f7a", "\u0f62\u0fa1\u0f7c", "\u0f62\u0f97\u0f7a", "\u0f66"
NAME_BO = TSHEG.join((KYE, RDO, RJE))
NAME_ZH = "\u7a7a\u667a\u91d1\u525b"
OTHER_ZH = "\u5927\u60b2"


def seg(sid: str, witness: str, lang: str, text: str) -> Segment:
    return Segment(sid, witness, lang, text, "1a.1", "", "verse")


def texts_of(ref_text: str, wit_text: str) -> SimpleNamespace:
    return SimpleNamespace(reference=(seg("D:1a.1.1", "bo_ref", "bo", ref_text),),
                           witness=(seg("T:1.1", "zh_wit", "zh", wit_text), seg("T:2.1", "zh_wit", "zh", OTHER_ZH)),
                           witness_id="zh_wit")


GLOSSARY = StandingRenderings((StandingEntry("hevajra", (NAME_BO,), (NAME_ZH,)),
                               StandingEntry("other", (SA,), (OTHER_ZH,))))


def lexicon(glossary: StandingRenderings) -> CheckLexicon:
    return CheckLexicon(Negators({}, {}), VariantForms({}), standing_renderings=glossary)


def test_the_glossary_check_follows_the_verified_entries() -> None:
    lex = lexicon(GLOSSARY)
    fused = texts_of(NAME_BO + SA + TSHEG + RDO, NAME_ZH + "\u8aaa")              # a particle fused to rje
    assert glossary_check(fused, lex)("D:1a.1.1", ("T:1.1",)) is None
    inside = texts_of(SA + NAME_BO, NAME_ZH)                                         # not syllable-initial
    assert "needs a verified entry" in glossary_check(inside, lex)("D:1a.1.1", ("T:1.1",))
    crossed = texts_of(NAME_BO, NAME_ZH)                                             # the rendering of ANOTHER entry
    assert "needs a verified entry" in glossary_check(crossed, lex)("D:1a.1.1", ("T:2.1",))
    assert glossary_check(crossed, lex)("D:1a.1.1", ("T:2.1", "T:1.1")) is None
    assert "unknown segment(s) ['T:9.1']" in glossary_check(crossed, lex)("D:1a.1.1", ("T:9.1",))
    empty = glossary_check(fused, lexicon(StandingRenderings.empty()))("D:1a.1.1", ("T:1.1",))
    assert "no entry is verified for zh_wit" in empty


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _verify_glossary(root: Path, run: Path) -> tuple[str, str]:
    """Rewrite the run root's glossary with one verified entry built from the fixture text
    of the first planned unit and the first witness segment of its context; returns them."""
    ctx = context(root, run, offline=True)
    texts = load_texts(ctx)
    row = _rows(run / "review" / "verify.blind.csv")[0]
    ref = {s.id: s for s in texts.reference}[row["unit_id"]]
    wit_id = row["zh_context_loci"].split()[0]
    wit = {s.id: s for s in texts.witness}[wit_id]
    path = root / "data" / "codebook" / "collate.v2" / "standing_renderings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "schema_version": 1, "reference": texts.reference_id, "witness": texts.witness_id,
        "entries": [{"id": "fixture", "reference_forms": [bo_syllables(ref.text)[0]],
                     "witness_renderings": ["".join(c for c in wit.text if "\u4e00" <= c <= "\u9fff")[:2]],
                     "status": "verified", "verified_by": "test"}],
        "considered_not_listed": []}, allow_unicode=True), encoding="utf-8")
    assert len(ann.standing_renderings(ctx, texts).entries) == 1
    return row["item_id"], wit_id


def test_glossary_beside_the_sheets_and_the_checked_import(finished, capsys) -> None:  # noqa: F811
    root, run = finished
    assert cli(root, run, "sample", "verification") == 0
    assert cli(root, run, "review", "export", "--task", "verify") == 0
    blind = run / "review" / "verify.blind.csv"
    assert _rows(run / "review" / "standing_renderings.csv") == [], "nothing verified: header only"
    fill(blind, blind_relation="no_counterpart")
    first = _rows(blind)[0]
    rows = _rows(blind)
    rows[0].update(blind_relation="equivalent", blind_wit_loci=first["zh_context_loci"].split()[0],
                   blind_convention="standing_rendering")
    _write(blind, rows)
    args = ("review", "import", "--task", "verify", "--file", str(blind), "--annotator", "ann")
    capsys.readouterr()
    assert cli(root, run, *args) == 2
    assert "line 2: standing_rendering is accepted only for a verified glossary entry" in capsys.readouterr().err

    item, wit_id = _verify_glossary(root, run)
    assert cli(root, run, "review", "export", "--task", "verify", "--force") == 0
    listed = _rows(run / "review" / "standing_renderings.csv")
    assert [r["entry"] for r in listed] == ["fixture"]
    _write(blind, [dict(r, blind_relation="no_counterpart") for r in _rows(blind)])
    rows = _rows(blind)
    rows[0].update(blind_relation="equivalent", blind_wit_loci=wit_id, blind_convention="standing_rendering")
    _write(blind, rows)
    assert cli(root, run, *args) == 0
    committed = root / "data" / "annotations" / "verdicts" / WITNESS / "verify.csv"
    saved = {v.item_id: v for v in verdict_files.read_file(committed)}
    assert saved[item].blind_convention == "standing_rendering"
    assert {v.blind_convention for k, v in saved.items() if k != item} == {"none"}

    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "verify") == 0
    assert (run / "review" / "standing_renderings.csv").is_file()
    reveal = _rows(run / "review" / "verify.reveal.csv")
    assert {r["machine_convention"] for r in reveal} == {"none"}, "a v1 consensus carries no convention"
    assert cli(root, run, "review", "reveal", "--file", str(run / "review" / "verify.reveal.csv")) == 0
    final = {v.item_id: v for v in verdict_files.read_file(committed)}
    assert final[item].final_convention == "standing_rendering" and final[item].machine_convention == "none"

    assert cli(root, run, *args) == 0, "the same blind sheet again is harmless"
    rows = _rows(blind)
    rows[0]["blind_convention"] = ""
    _write(blind, rows)
    capsys.readouterr()
    assert cli(root, run, *args) == 2
    assert "a blind decision is fixed once revealed" in capsys.readouterr().err


def test_a_glossary_of_another_pair_of_texts_is_not_shown(finished) -> None:  # noqa: F811
    root, run = finished
    path = root / "data" / "codebook" / "collate.v2" / "standing_renderings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"schema_version": 1, "reference": "bo_other", "witness": "zh_other",
                                    "entries": [{"id": "x", "reference_forms": [KYE], "witness_renderings": [NAME_ZH],
                                                 "status": "verified", "verified_by": "test"}]},
                                   allow_unicode=True), encoding="utf-8")
    ctx = context(root, run, offline=True)
    assert ann.standing_renderings(ctx, load_texts(ctx)).entries == ()
    path.unlink()
    assert ann.standing_renderings(ctx, load_texts(ctx)).entries == ()


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_a_reveal_under_another_prompt_version_is_refused(finished, capsys) -> None:  # noqa: F811
    """The verdicts record the collate digest; it must be the version the run collated with."""
    from test_pipeline_collate_v2 import select_prompt

    root, run = finished
    assert cli(root, run, "sample", "verification") == 0
    assert cli(root, run, "review", "export", "--task", "verify") == 0
    blind = run / "review" / "verify.blind.csv"
    fill(blind, blind_relation="no_counterpart")
    assert cli(root, run, "review", "import", "--task", "verify", "--file", str(blind), "--annotator", "ann") == 0
    assert cli(root, run, "review", "export", "--task", "reveal", "--batch", "verify") == 0
    select_prompt(root, "collate.v2")
    capsys.readouterr()
    assert cli(root, run, "review", "reveal", "--file", str(run / "review" / "verify.reveal.csv")) == 2
    assert "was collated with collate.v1" in capsys.readouterr().err
    committed = root / "data" / "annotations" / "verdicts" / WITNESS / "verify.csv"
    assert all(not v.final_relation for v in verdict_files.read_file(committed))
