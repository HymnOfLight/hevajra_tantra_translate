import json
from pathlib import Path

from hevajra_matrix.align import AlignParams, AnchorBackend, align_chapter
from hevajra_matrix.anchors import load_lexicon
from hevajra_matrix.attribution import LABELS, CounterfactualItem
from hevajra_matrix.llm import tasks
from hevajra_matrix.llm.backends import MockBackend, load_backend
from hevajra_matrix.llm.embeddings import EmbeddingSimilarityBackend, HybridSimilarityBackend
from hevajra_matrix.registry import DATA_DIR
from hevajra_matrix.segments import Segment


def seg(w, i, lang, text):
    return Segment(witness=w, seg_id=f"{w}:{i}", lang=lang, text=text, start=str(i), end=str(i), kind="prose", chapter="I.1")


# --------------------------------------------------------------------------- backends / config
def test_models_yaml_profiles_load_without_gpu():
    from hevajra_matrix.llm.backends import OpenAICompatibleBackend, TransformersBackend

    b = load_backend("qwen3-32b-awq")
    assert isinstance(b, OpenAICompatibleBackend) and b.model_id == "qwen3-32b-awq"
    t = load_backend("qwen3-8b")
    assert isinstance(t, TransformersBackend) and t._model is None  # lazy: nothing loaded
    assert isinstance(load_backend("mock"), MockBackend)


def test_call_log_records_hashes(tmp_path: Path):
    b = MockBackend()
    b.generate("hello", system="sys")
    assert len(b.log.records) == 1 and b.log.records[0].prompt_sha256 != b.log.records[0].system_sha256
    b.log.dump(tmp_path / "log.json")
    assert json.loads((tmp_path / "log.json").read_text())[0]["model_id"] == "mock"


def test_parse_json_tolerates_think_and_fences():
    assert tasks.parse_json("<think>hmm</think>```json\n{\"a\": 1}\n```") == {"a": 1}
    assert tasks.parse_json("Sure: {\"a\": [1,2]} done") == {"a": [1, 2]}


# --------------------------------------------------------------------------- R4 components
def test_extract_components_rejects_non_substrings():
    text = "金剛藏白佛言：一切如來身語心。"
    resp = json.dumps({"actor": ["金剛藏"], "action": ["白佛言"], "patient": ["一切如來身語心"],
                       "instrument": [], "negation": ["不"], "condition": [], "quantity": ["一切"], "result": [],
                       "lexicon_keys": ["name:vajragarbha", "bogus:key"]}, ensure_ascii=False)
    b = MockBackend(lambda p, s: resp)
    cs = tasks.extract_components(b, text, "zh", load_lexicon(DATA_DIR / "anchors" / "terms.yaml"))
    assert cs.slots["actor"] == ["金剛藏"] and cs.slots["negation"] == []      # 不 is not in the text
    assert "negation:不" in cs.rejected
    assert cs.lexicon_keys == ["name:vajragarbha"]                          # bogus key filtered against lexicon


def test_component_deviation_flags_negation_loss():
    ref = tasks.ComponentSet("a", "sa", "m", slots={"actor": ["x"], "negation": ["na"], "action": ["y"]})
    wit = tasks.ComponentSet("b", "zh", "m", slots={"actor": ["甲"], "action": ["乙"]})
    d = tasks.component_deviation(ref, wit)
    assert d["negation_lost"] == 1 and d["n_lost"] == 1 and abs(d["d_comp"] - 1 / 3) < 1e-9


# --------------------------------------------------------------------------- R2 judge
def test_judge_review_rows_only_touches_suspicious_beads():
    rows = [{"shape": "1:1", "ref_text": "a", "wit_text": "b", "review_status": "", "review_note": ""},
            {"shape": "1:0", "ref_text": "c", "wit_text": "", "review_status": "", "review_note": ""},
            {"shape": "0:1", "ref_text": "", "wit_text": "d", "review_status": "human:ok", "review_note": ""}]
    b = MockBackend(lambda p, s: json.dumps({"verdict": "make_null", "confidence": 0.9, "shared_content": [], "note": ""}))
    out = tasks.judge_review_rows(b, rows, "bo", "zh")
    assert out[0]["review_status"] == "" and out[1]["review_status"] == "llm:make_null" and out[2]["review_status"] == "human:ok"
    assert len(b.calls) == 1


def test_judge_bead_invalid_verdict_becomes_unsure():
    b = MockBackend(lambda p, s: json.dumps({"verdict": "definitely", "confidence": "high"}))
    j = tasks.judge_bead(b, "x", "y", "sa", "zh")
    assert j.verdict == "unsure" and j.confidence == 0.0


# --------------------------------------------------------------------------- R5 / R6 / R9 guards
def test_topic_label_restricted_to_fixed_set():
    b = MockBackend(lambda p, s: json.dumps({"topic": "heresy", "cues": ["zzz"]}))
    assert tasks.label_topic(b, "I.1.1", "vajrasattva", "sa")["topic"] == "other"


def test_attribution_judge_downgrades_motive_language():
    facts = ["zh lacks unit", "bo retains unit"]
    b = MockBackend(lambda p, s: json.dumps({"distribution": {"unexplained_at_translation": 0.9, "abstain": 0.1},
                                             "cited_facts": [1, 2, 7], "note": "the translator deliberately omitted it (self-censorship)"}))
    j = tasks.judge_attribution(b, "I.1.1", "zh", facts)
    assert j.motive_flag and j.top == "abstain" and j.cited_facts == [1, 2]
    b2 = MockBackend(lambda p, s: json.dumps({"distribution": {"shared_with_cowitness": 2, "abstain": 2}, "note": ""}))
    j2 = tasks.judge_attribution(b2, "I.1.1", "zh", facts)
    assert abs(sum(j2.distribution.values()) - 1) < 1e-9 and set(j2.distribution) <= set(LABELS)


def test_explain_cell_flags_motive_words():
    b = MockBackend(lambda p, s: "The Chinese omits the verse, apparently toned down for readers.")
    r = tasks.explain_cell(b, "I.1.1", "zh", "ref", "wit", "ABSENT", "1:0", None, None)
    assert r["motive_flag"] is True


# --------------------------------------------------------------------------- R7 / R8
def test_counterfactual_across_models():
    items = [CounterfactualItem("I.1.1", "sa", "text", "0587c", "zh", "the Tibetan also lacks this unit")]
    naive = MockBackend(lambda p, s: "This looks like self-censorship." if "Additional fact" not in p else "Vorlage variation.", model_id="naive")
    stubborn = MockBackend(lambda p, s: "Censorship.", model_id="stubborn")
    rows = tasks.counterfactual_across_models([naive, stubborn], items)
    assert rows[0]["drop"] == 1.0 and rows[1]["drop"] == 0.0


def test_contamination_probe_scores_memorised_text():
    text = "如是我聞，一時婆伽梵住一切如來身語心金剛喜金剛婦人之陰處。爾時，世尊告金剛藏菩薩言。" * 2
    items = tasks.probe_items_from_segments([("cbeta_T0892_zh", "0587c", text)], prefix_chars=20, min_chars=40)
    memorised = MockBackend(lambda p, s: text[20:])
    forgetful = MockBackend(lambda p, s: "南無阿彌陀佛" * 10)
    r1 = tasks.contamination_probe(memorised, items)
    r2 = tasks.contamination_probe(forgetful, items)
    assert r1["memorised_rate"] == 1.0 and r2["memorised_rate"] == 0.0
    assert "cbeta_T0892_zh" in r1["by_source"]


def test_segment_propositions_falls_back_when_not_tiling():
    text = "甲乙丙丁。戊己庚辛。"
    good = MockBackend(lambda p, s: json.dumps({"propositions": ["甲乙丙丁。", "戊己庚辛。"]}, ensure_ascii=False))
    bad = MockBackend(lambda p, s: json.dumps({"propositions": ["甲乙", "庚辛壬"]}, ensure_ascii=False))
    assert tasks.segment_propositions(good, text, "zh") == ["甲乙丙丁。", "戊己庚辛。"]
    assert tasks.segment_propositions(bad, text, "zh") == [text]


# --------------------------------------------------------------------------- B1 embeddings in the aligner
def toy_encode(texts):
    # 3-dim bag: "vajra"-ness, "mantra"-ness, other
    out = []
    for t in texts:
        v = [float(any(k in t for k in ("金剛", "vajra", "རྡོ་རྗེ"))), float(any(k in t for k in ("真言", "mantra", "སྔགས"))), 1.0]
        out.append(v)
    return out


def test_embedding_and_hybrid_backends_drive_alignment():
    ref = [seg("bo", 1, "bo", "རྡོ་རྗེ་སྙིང་པོ།"), seg("bo", 2, "bo", "སྔགས་ནི།")]
    wit = [seg("zh", 1, "zh", "金剛藏。"), seg("zh", 2, "zh", "真言曰。")]
    emb = EmbeddingSimilarityBackend(model_id="toy", encode=toy_encode)
    emb.warm(ref + wit)
    assert emb.score([ref[0]], [wit[0]]) > emb.score([ref[0]], [wit[1]])
    beads = align_chapter(ref, wit, emb, AlignParams(anchor_weight=4.0))
    assert [b.shape for b in beads] == [(1, 1), (1, 1)]
    hyb = HybridSimilarityBackend(AnchorBackend(load_lexicon(DATA_DIR / "anchors" / "terms.yaml")), emb, weight=0.5)
    assert hyb.has_anchors([wit[0]]) and hyb.score([ref[0]], [wit[0]]) >= 0.5 * emb.score([ref[0]], [wit[0]])
