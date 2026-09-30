import math

from hevajra_matrix import metrics
from hevajra_matrix.align import AlignParams, AnchorBackend, align_chapter, alignment_summary
from hevajra_matrix.anchors import load_lexicon
from hevajra_matrix.attribution import (CounterfactualHarness, CounterfactualItem, LABELS, asserts_motive, attribute,
                                        evidence_gate)
from hevajra_matrix.matrix import Cell, Unit, WitnessMatrix, calibrate_length_residuals, cells_from_beads, units_from_reference
from hevajra_matrix.segments import Segment


def seg(w, lang, i, text, ch="I.1", kind="prose"):
    return Segment(witness=w, seg_id=f"{w}:{i}", lang=lang, text=text, start=str(i), end=str(i), kind=kind, chapter=ch)


def test_alignment_uses_anchors_and_allows_null():
    lex = load_lexicon()
    be = AnchorBackend(lex)
    ref = [seg("bo", "bo", i, t) for i, t in enumerate([
        "རྡོ་རྗེ་སྙིང་པོས་གསོལ་བ།",                    # Vajragarbha asked
        "རྩ་ནི་སུམ་ཅུ་རྩ་གཉིས་ཏེ།",                    # 32 nāḍīs
        "ལ་ལ་ན་ར་ས་ན་ཨ་བ་དྷཱུ་ཏཱི།",                    # lalanā rasanā avadhūtī
        "ཤ་ལྔ་དང་བདུད་རྩི་ལྔ་ཟ་བར་བྱ།",                 # five meats & five nectars — no Chinese counterpart
        "ཏིང་ངེ་འཛིན་ལས་བཞེངས་སོ།",                    # rose from samādhi
    ])]
    wit = [seg("zh", "zh", i, t) for i, t in enumerate([
        "金剛藏菩薩白佛言，", "彼血脈相有三十二種，", "謂羅羅拏辣娑拏阿嚩底，", "從是三摩地起。",
    ])]
    beads = align_chapter(ref, wit, be, AlignParams())
    summ = alignment_summary(beads)
    assert summ["ref_null"] == 1, summ
    null_bead = next(b for b in beads if b.shape == (1, 0))
    assert null_bead.ref == [3]
    pairs = {(tuple(b.ref), tuple(b.wit)) for b in beads}
    assert ((0,), (0,)) in pairs and ((1,), (1,)) in pairs and ((4,), (3,)) in pairs
    assert next(b for b in beads if b.ref == [1]).similarity > 0.5


def _toy_matrix():
    """Two Sanskrit editions (gold + variant), Tibetan and Chinese translations."""
    lex = load_lexicon()
    m = WitnessMatrix(reference="sa_a", reference_grade="gold")
    ref = [seg("sa_a", "sa", i, t, kind="verse") for i, t in enumerate(["vajragarbha", "dvātriṃśat nāḍī", "śvānamāṃsa", "samādhi"])]
    for s, uid in zip(ref, ["I.1.1", "I.1.2", "I.1.3", "I.1.4"]):
        s.seg_id = uid
    units = units_from_reference(ref)
    m.add_units(units)
    # variant Sanskrit edition lacks verse 3
    for u in units:
        m.add_cell(Cell(unit=u.unit_id, witness="sa_b", status="ABSENT" if u.unit_id == "I.1.3" else "PRESENT",
                        d_cov=0.0 if u.unit_id == "I.1.3" else 1.0, chapter="I.1", kind="verse"))
    # Tibetan retains everything; Chinese lacks verses 3 and 4
    bo = [seg("bo", "bo", i, t) for i, t in enumerate(["རྡོ་རྗེ་སྙིང་པོ།", "རྩ་སུམ་ཅུ་རྩ་གཉིས།", "ཁྱིའི་ཤ།", "ཏིང་ངེ་འཛིན།"])]
    zh = [seg("zh", "zh", i, t) for i, t in enumerate(["金剛藏，", "三十二血脈，"])]
    be = AnchorBackend(lex)
    for wname, wsegs in (("bo", bo), ("zh", zh)):
        beads = align_chapter(ref, wsegs, be, AlignParams())
        cells = cells_from_beads(units, ref, wsegs, beads, wname, lex, "I.1")
        calibrate_length_residuals(cells, wname)
        for c in cells:
            m.add_cell(c)
    return m


def test_cells_statuses_and_decomposition():
    m = _toy_matrix()
    assert m.get("I.1.1", "zh").status == "PRESENT"
    assert m.get("I.1.3", "zh").status == "ABSENT" and m.get("I.1.4", "zh").status == "ABSENT"
    assert all(m.get(u, "bo").status == "PRESENT" for u in ["I.1.1", "I.1.2", "I.1.3", "I.1.4"])
    dec = metrics.decompose(m, "zh", ["sa_a", "sa_b"], cowitness="bo")
    assert dec.n_counted == 4 and dec.n_deviating == 2
    assert dec.by_unit["I.1.3"] == "vorlage_explained"   # sa_b also lacks it
    assert dec.by_unit["I.1.4"] == "residual"            # Sanskrit unanimous, Tibetan retains
    r = dec.rates()
    assert math.isclose(r["total_deviation"], 0.5)
    # structure table and wide view
    row = next(r for r in m.structure() if r["chapter"] == "I.1" and r["witness"] == "zh")
    assert row["n_absent"] == 2 and row["coverage"] == 0.5
    assert m.wide("status")[0]["zh"] == "PRESENT"


def test_evidence_gate_and_attribution_never_names_motive():
    m = _toy_matrix()
    assert "censorship" not in LABELS
    g = evidence_gate(m, "I.1.4", "zh", ["sa_a", "sa_b"], cowitness="bo")
    assert g.sufficient and g.n_sanskrit_informative == 2
    a4 = attribute(m, "I.1.4", "zh", ["sa_a", "sa_b"], cowitness="bo")
    assert a4.top == "unexplained_at_translation" and a4.distribution["abstain"] > 0
    a3 = attribute(m, "I.1.3", "zh", ["sa_a", "sa_b"], cowitness="bo")
    assert a3.top == "vorlage_attested"
    a1 = attribute(m, "I.1.1", "zh", ["sa_a", "sa_b"], cowitness="bo")
    assert a1.top == "abstain"
    # missing co-witness evidence → gate closes → abstain
    m.add_cell(Cell(unit="I.1.4", witness="bo", status="LACUNA", chapter="I.1", kind="verse"))
    a4b = attribute(m, "I.1.4", "zh", ["sa_a", "sa_b"], cowitness="bo")
    assert a4b.top == "abstain" and "INSUFFICIENT_COWITNESS" in a4b.evidence


def test_block_bootstrap_and_distance():
    m = _toy_matrix()
    groups = metrics.deviation_rate_by_chapter(m, "zh")
    ci = metrics.block_bootstrap(groups, metrics.rate, n_boot=200)
    assert math.isclose(ci["estimate"], 0.5) and ci["n_blocks"] == 1
    d = metrics.witness_distance(m, "zh", "bo")
    assert d is not None and d > 0
    ws, D = metrics.distance_matrix(m, ["zh", "bo", "sa_b"])
    newick = metrics.average_linkage_newick(ws, D)
    assert newick.endswith(";") and "zh" in newick and "bo" in newick
    assert metrics.kendall_tau_distance([0, 1, 2], [2, 1, 0]) == 1.0


def test_counterfactual_harness_scoring():
    assert asserts_motive("The translator deliberately omitted the passage as self-censorship.")
    assert asserts_motive("譯者為迴避性描寫而刪略之。")
    assert not asserts_motive("The Vorlage used by the translator probably lacked this verse.")

    def fake_model(prompt: str) -> str:
        return "The Sanskrit manuscript tradition varies here." if "Additional fact" in prompt else "This is clearly censorship."

    h = CounterfactualHarness(fake_model)
    item = CounterfactualItem("I.1.4", "sa_a", "samādhi", "0587c13", "從是三摩地起", "the Tibetan translation also lacks this unit")
    pa, pb = h.build_prompts(item)
    assert "Additional fact" in pb and "Additional fact" not in pa
    res = h.run([item, item])
    assert res.rate_a == 1.0 and res.rate_b == 0.0 and res.diff == 1.0
