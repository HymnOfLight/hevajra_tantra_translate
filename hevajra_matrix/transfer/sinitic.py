"""Same-language multi-witness matrix (Laozi and similar Chinese texts).

What changes relative to the Hevajra pipeline (docs/03 §7, docs/05 §2):

* the reference is a *coordinate scheme* (the 81 received chapters, clause-split),
  not a source-language text → ``reference_grade="coordinate_only"``; the quantity
  of interest is the spread between witnesses (σ_S), not the distance to a gold text;
* no cross-lingual anchors: similarity is character n-gram overlap after
  variant-character normalisation (通假 / 异体 / 避讳 / 古今字 from ``variants.yaml``);
* a new dimension ``d_lex`` (lexical distance after normalisation) and, in the review
  table, ``d_lex_raw`` so that the share explained by variant spelling is visible;
* chapter *order* is itself a measurement (Mawangdui: 德 before 道; Guodian: its own
  sequence) → Kendall-τ against the received order;
* anthology witnesses (Guodian) mark unincluded chapters ``NA`` (out of scope), never
  ``ABSENT``; damaged graphs (□) yield ``LACUNA``.

Everything downstream — matrix, decomposition, bootstrap, distances, reports — is the
unchanged Hevajra code, which is the point of the transfer test.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml

from .. import metrics
from ..align import AlignParams, align_chapter, alignment_summary
from ..anchors import AnchorLexicon
from ..ids import register_scheme
from ..matrix import Cell, Unit, WitnessMatrix, _write_csv, calibrate_length_residuals, cells_from_beads, sha256_of
from ..report import markdown_summary, status_strip_svg, structure_heatmap_svg
from ..segments import Segment

PART = "L"
N_CHAPTERS = 81
CHAPTERS = register_scheme(PART, N_CHAPTERS)

_CLAUSE_PUNCT = "。？！，；：、"
_CHAPTER_RE = re.compile(r"^#\s*(\d{1,2})([上下]?)(\?)?\s*(?:@(\S+))?\s*(.*)$")
_LACUNA_CHARS = set("□▢■囗〼")
_STRIP_RE = re.compile(r"[\s。？！，；：、「」『』（）()《》〈〉\[\]【】…—－-]+")


# --------------------------------------------------------------------------- variants
@dataclass
class VariantTable:
    mapping: dict[str, str] = field(default_factory=dict)
    category: dict[str, str] = field(default_factory=dict)

    def normalize(self, text: str) -> str:
        return "".join(self.mapping.get(ch, ch) for ch in text)

    def explained(self, a: str, b: str) -> list[str]:
        """Character pairs that differ raw but agree after normalisation (for the audit column)."""
        out = []
        for x, y in zip(a, b):
            if x != y and self.mapping.get(x, x) == self.mapping.get(y, y):
                out.append(f"{x}={y}")
        return out


def load_variants(path: Path) -> VariantTable:
    vt = VariantTable()
    if not path.exists():
        return vt
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for group in data.get("groups", []):
        canon = group["canonical"]
        cat = group.get("category", "")
        if not group.get("normalize", True):
            continue
        for v in group.get("variants", []):
            vt.mapping[v] = canon
            vt.category[v] = cat
    return vt


# --------------------------------------------------------------------------- ingest
def _strip(text: str) -> str:
    return _STRIP_RE.sub("", text)


def _clauses(text: str) -> list[str]:
    out, buf = [], ""
    for ch in text:
        buf += ch
        if ch in _CLAUSE_PUNCT:
            if _strip(buf):
                out.append(buf.strip())
            buf = ""
    if _strip(buf):
        out.append(buf.strip())
    return out


def lacuna_fraction(text: str) -> float:
    core = _strip(text)
    return (sum(1 for c in core if c in _LACUNA_CHARS) / len(core)) if core else 0.0


def parse_witness_file(path: Path, witness: str, lang: str = "zh", lacuna_threshold: float = 0.5) -> list[Segment]:
    """Plain-text witness format:

        // comment
        #19 @甲簡1-2  optional native coordinate, optional trailing note
        絕智棄辯，民利百倍。…
        #64下         half-chapter markers are accepted and folded into the chapter
        #1?           '?' = chapter identification uncertain (kept in extra)

    File order is the witness' own chapter order (recorded in ``extra['order']``).
    """
    segs: list[Segment] = []
    chapter: int | None = None
    half, uncertain, native, order = "", False, "", -1
    seen: list[int] = []
    k = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = _CHAPTER_RE.match(line)
        if m:
            chapter = int(m.group(1))
            half, uncertain, native = m.group(2) or "", bool(m.group(3)), m.group(4) or ""
            if chapter not in seen:
                seen.append(chapter)
            order = seen.index(chapter)
            k = 0
            continue
        if chapter is None:
            continue
        for cl in _clauses(line):
            k += 1
            lf = lacuna_fraction(cl)
            segs.append(Segment(
                witness=witness, seg_id=f"{witness}:{chapter}{half}:{k}", lang=lang, text=cl,
                start=f"c{chapter}{half}:{k}" + (f"@{native}" if native else ""), end=f"c{chapter}{half}:{k}",
                kind="lacuna" if lf >= lacuna_threshold else "prose",
                chapter=f"{PART}.{chapter}", local_chapter=chapter,
                extra={"order": order, "half": half, "uncertain": uncertain, "native": native, "lacuna_fraction": round(lf, 3)},
            ))
    return segs


def chapter_order(segments: Sequence[Segment]) -> list[int]:
    seen: list[int] = []
    for s in segments:
        if s.local_chapter is not None and s.local_chapter not in seen:
            seen.append(s.local_chapter)
    return seen


def order_deviation(order: Sequence[int], reference: Sequence[int] | None = None) -> dict:
    """Kendall-τ distance of the witness' chapter sequence to the received order, on shared chapters,
    plus the share of 德-part chapters (38–81) placed before 道-part chapters (1–37)."""
    ref = [c for c in (reference or range(1, N_CHAPTERS + 1)) if c in set(order)]
    wit = [c for c in order if c in set(ref)]
    tau = metrics.kendall_tau_distance(ref, wit) if len(wit) >= 2 else 0.0
    de_first = 0
    pairs = 0
    pos = {c: i for i, c in enumerate(wit)}
    for a in wit:
        for b in wit:
            if a >= 38 and b <= 37:
                pairs += 1
                de_first += pos[a] < pos[b]
    return {"n_chapters": len(wit), "kendall_tau_distance": round(tau, 4),
            "de_before_dao_share": round(de_first / pairs, 4) if pairs else None, "sequence": wit}


# --------------------------------------------------------------------------- similarity
def _grams(text: str, n: int) -> set[str]:
    core = text
    if len(core) < n:
        return {core} if core else set()
    return {core[i:i + n] for i in range(len(core) - n + 1)}


def dice(a: str, b: str) -> float:
    """Dice coefficient over character unigrams+bigrams of the (already normalised, stripped) strings."""
    if not a or not b:
        return 0.0
    ga = _grams(a, 1) | _grams(a, 2)
    gb = _grams(b, 1) | _grams(b, 2)
    return 2 * len(ga & gb) / (len(ga) + len(gb))


class CharNgramBackend:
    """``align.SimilarityBackend`` for same-language witnesses: Dice on normalised char n-grams."""

    def __init__(self, variants: VariantTable) -> None:
        self.variants = variants
        self._cache: dict[str, str] = {}

    def norm(self, s: Segment) -> str:
        key = f"{s.witness}|{s.seg_id}"
        if key not in self._cache:
            self._cache[key] = self.variants.normalize(_strip(s.text)).replace("□", "")
        return self._cache[key]

    def score(self, ref: Sequence[Segment], wit: Sequence[Segment]) -> float:
        return dice("".join(self.norm(s) for s in ref), "".join(self.norm(s) for s in wit))


# --------------------------------------------------------------------------- registry
@dataclass
class TransferWitness:
    id: str
    label: str
    layer: str            # primary | edition | indirect
    scope: str            # complete | anthology | fragment | quotation
    date: str = ""
    reference: bool = False
    cowitness: str | None = None
    file: str | None = None
    note: str = ""


def load_transfer_witnesses(path: Path) -> list[TransferWitness]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for w in data["witnesses"]:
        out.append(TransferWitness(id=w["id"], label=w.get("label", w["id"]), layer=w.get("layer", "primary"),
                                   scope=w.get("scope", "complete"), date=str(w.get("date", "")),
                                   reference=bool(w.get("reference", False)), cowitness=w.get("cowitness"),
                                   file=w.get("file", f"{w['id']}.txt"), note=w.get("note", "")))
    return out


# --------------------------------------------------------------------------- reorder repair
def reorder_pass(ru: list[Unit], ws: list[Segment], cells: list[Cell], backend: CharNgramBackend,
                 low: float = 0.35, high: float = 0.6) -> list[dict]:
    """Non-monotone repair after the DP: a reference unit whose monotone partner is a poor
    match (< ``low``) but which has a strong match (≥ ``high``) elsewhere in the same
    witness chapter is re-linked to that segment and receives ``d_ord`` = positional
    displacement. Bead is marked ``1:1*``. Returns audit rows."""
    lens = [max(1, len(_strip(s.text))) for s in ws]
    total = sum(lens) or 1
    pos, acc = [], 0
    for ln in lens:
        pos.append((acc + ln / 2) / total)
        acc += ln
    ref_norm = {u.unit_id: backend.variants.normalize(_strip(u.text)) for u in ru}
    unit_pos = {u.unit_id: u.position for u in ru}
    rows = []
    for c in cells:
        if c.status != "PRESENT" or c.unit.startswith("+") or (c.similarity or 0.0) >= low:
            continue
        scores = [(dice(ref_norm[c.unit], backend.norm(s)), j) for j, s in enumerate(ws)]
        best, j = max(scores) if scores else (0.0, -1)
        if j < 0 or best < high or ws[j].start == c.span_start:
            continue
        rows.append({"unit": c.unit, "witness": c.witness, "from": c.span_start, "to": ws[j].start,
                     "similarity_before": round(c.similarity or 0.0, 3), "similarity_after": round(best, 3)})
        c.similarity, c.d_lex, c.d_ord = best, 1 - best, abs(pos[j] - unit_pos[c.unit])
        c.span_start, c.span_end, c.text, c.bead, c.provenance = ws[j].start, ws[j].end, ws[j].text[:80], "1:1*", "reorder_pass"
    return rows


# --------------------------------------------------------------------------- run
def _units_from_reference(ref_segs: list[Segment]) -> list[Unit]:
    units: list[Unit] = []
    by_ch: dict[str, list[Segment]] = {}
    for s in ref_segs:
        by_ch.setdefault(s.chapter or "", []).append(s)
    for ch, segs in by_ch.items():
        lens = [max(1, len(_strip(s.text))) for s in segs]
        total = sum(lens) or 1
        acc = 0
        for i, (s, ln) in enumerate(zip(segs, lens)):
            uid = f"{ch}.p{i + 1:02d}"
            s.seg_id = uid
            units.append(Unit(unit_id=uid, chapter=ch, kind="prose", text=s.text, length=ln, position=(acc + ln / 2) / total))
            acc += ln
    return units


def run_transfer(config_dir: Path, texts_dir: Path, out_dir: Path, params: AlignParams | None = None) -> WitnessMatrix:
    params = params or AlignParams(anchor_weight=6.0, anchor_conflict=0.0)
    witnesses = load_transfer_witnesses(config_dir / "witnesses.yaml")
    variants = load_variants(config_dir / "variants.yaml")
    backend = CharNgramBackend(variants)
    empty_lex = AnchorLexicon(terms=[])

    available = {w.id: texts_dir / (w.file or f"{w.id}.txt") for w in witnesses}
    present = {wid: p for wid, p in available.items() if p.exists()}
    ref_w = next((w for w in witnesses if w.reference), None)
    if ref_w is None or ref_w.id not in present:
        raise FileNotFoundError("reference witness text not found (mark one witness `reference: true` and provide its file)")

    ref_segs = [s for s in parse_witness_file(present[ref_w.id], ref_w.id) if s.kind == "prose"]
    units = _units_from_reference(ref_segs)
    matrix = WitnessMatrix(reference=ref_w.id, reference_grade="coordinate_only", chapters=list(CHAPTERS))
    matrix.add_units(units)
    matrix.inputs = {p.name: sha256_of(p) for p in present.values()}
    unit_by_ch: dict[str, list[Unit]] = {}
    for u in units:
        unit_by_ch.setdefault(u.chapter, []).append(u)
    ref_by_ch: dict[str, list[Segment]] = {}
    for s in ref_segs:
        ref_by_ch.setdefault(s.chapter or "", []).append(s)

    log: dict[str, dict] = {}
    review: list[dict] = []
    reorders: list[dict] = []
    orders: list[dict] = []
    by_id = {w.id: w for w in witnesses}
    ref_order = chapter_order(ref_segs)
    orders.append({"witness": ref_w.id, **{k: v for k, v in order_deviation(ref_order, ref_order).items() if k != "sequence"},
                   "sequence": " ".join(map(str, ref_order))})

    for wid, path in present.items():
        if wid == ref_w.id:
            continue
        w = by_id[wid]
        wsegs = parse_witness_file(path, wid)
        od = order_deviation(chapter_order(wsegs), ref_order)
        orders.append({"witness": wid, **{k: v for k, v in od.items() if k != "sequence"}, "sequence": " ".join(map(str, od["sequence"]))})
        w_by_ch: dict[str, list[Segment]] = {}
        for s in wsegs:
            w_by_ch.setdefault(s.chapter or "", []).append(s)
        cells: list[Cell] = []
        for ch in CHAPTERS:
            ru, rs, ws = unit_by_ch.get(ch, []), ref_by_ch.get(ch, []), w_by_ch.get(ch, [])
            if not ru:
                continue
            if not ws:
                status = "NA" if w.scope in ("anthology", "fragment", "quotation") else "UNALIGNED"
                cells.extend(Cell(unit=u.unit_id, witness=wid, status=status, chapter=ch, kind=u.kind) for u in ru)
                continue
            beads = align_chapter(rs, ws, backend, params)
            log[f"{wid}:{ch}"] = alignment_summary(beads)
            new = cells_from_beads(ru, rs, ws, beads, wid, empty_lex, ch)
            # walk cells in bead order to add d_lex / LACUNA and the review rows
            k = 0
            for b in beads:
                n, m = b.shape
                wit_text = "".join(ws[j].text for j in b.wit)
                wit_norm = "".join(backend.norm(ws[j]) for j in b.wit)
                ref_text = "".join(rs[i].text for i in b.ref)
                ref_norm = "".join(backend.norm(rs[i]) for i in b.ref)
                count = 1 if n == 0 else n
                lac = any(ws[j].kind == "lacuna" for j in b.wit)
                d_raw = 1 - dice(_strip(ref_text), _strip(wit_text)) if n and m else None
                d_norm = 1 - dice(ref_norm, wit_norm) if n and m else None
                for c in new[k:k + count]:
                    if n and m:
                        c.d_lex = d_norm
                        if lac:
                            c.status, c.d_cov = "LACUNA", None
                    if n and not m:
                        c.d_lex = None
                k += count
                review.append({
                    "witness": wid, "chapter": ch, "shape": f"{n}:{m}",
                    "ref_units": " ".join(ru[i].unit_id for i in b.ref), "ref_text": ref_text[:120], "wit_text": wit_text[:120],
                    "wit_coords": f"{ws[b.wit[0]].start}-{ws[b.wit[-1]].end}" if b.wit else "",
                    "similarity": round(b.similarity, 3), "d_lex_raw": "" if d_raw is None else round(d_raw, 3),
                    "d_lex": "" if d_norm is None else round(d_norm, 3),
                    "variants_explained": " ".join(variants.explained(_strip(ref_text), _strip(wit_text))),
                    "review_status": "", "review_note": "",
                })
            reorders.extend(reorder_pass(ru, ws, new, backend))
            cells.extend(new)
        log[f"{wid}:length_baseline"] = calibrate_length_residuals(cells, wid)
        for c in cells:
            if c.unit.startswith("+"):
                matrix.units.setdefault(c.unit, Unit(unit_id=c.unit, chapter=c.chapter or "", kind="orphan", text=c.text, length=0, position=0.0))
            matrix.add_cell(c)

    for wid in available:
        if wid not in present:
            log[f"{wid}:missing"] = {"note": "text file not supplied; column omitted"}

    # ------------------------------------------------------------------ statistics
    others = [w for w in matrix.witnesses]
    decs = []
    for wid in others:
        co = by_id[wid].cowitness
        decs.append(metrics.decompose(matrix, wid, [], cowitness=co if co in matrix.witnesses else None))
    dist = metrics.distance_matrix(matrix, others) if len(others) >= 2 else None
    newick = metrics.average_linkage_newick(*dist) if dist else None
    boots = {wid: metrics.block_bootstrap(metrics.deviation_rate_by_chapter(matrix, wid), metrics.rate) for wid in others}

    # ------------------------------------------------------------------ export
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = matrix.export(out_dir, params={**params.__dict__, "beads": list(params.beads)})
    manifest["alignment_log"] = log
    manifest["chapter_order"] = orders
    manifest["reorder_repairs"] = reorders
    manifest["deviation_rate_ci"] = boots
    manifest["decompositions"] = [{"witness": d.witness, "cowitness": d.cowitness, "n_counted": d.n_counted,
                                   "n_deviating": d.n_deviating, "counts": d.counts} for d in decs]
    manifest["variants_used"] = len(variants.mapping)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_dir / "alignment_review.csv", review)
    _write_csv(out_dir / "chapter_order.csv", orders)
    _write_csv(out_dir / "reorder_repairs.csv", reorders)
    structure_heatmap_svg(matrix, out_dir / "heatmap_coverage.svg", "coverage")
    structure_heatmap_svg(matrix, out_dir / "heatmap_d_lex.svg", "mean_d_lex")
    status_strip_svg(matrix, out_dir / "status_strip.svg")
    extra = {"参照等级": "coordinate_only —— 参照只是通行本的章/句坐标，不是金标准；报告的是见证间离散度 σ_S，不是相对源文本的偏移",
             "章序偏移 (Kendall τ 距离)": "; ".join(f"{o['witness']}={o['kendall_tau_distance']}" for o in orders if o["witness"] != ref_w.id)}
    for wid, b in boots.items():
        extra[f"{wid} 句级缺失率（章区组自助法）"] = f"{b['estimate']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]，{b['n_blocks']} 章"
    (out_dir / "summary.md").write_text(markdown_summary(matrix, decs, dist, newick, extra), encoding="utf-8")
    return matrix
