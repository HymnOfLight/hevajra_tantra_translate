"""End-to-end run: ingest → chapter gating → alignment → matrix → statistics → report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from . import metrics
from .align import AlignParams, AnchorBackend, align_chapter, alignment_summary
from .anchors import load_lexicon
from .ids import REF_CHAPTERS
from .ingest.cbeta import parse_cbeta_tei
from .ingest.derge import parse_derge_volume
from .ingest.sanskrit import load_reference_tsv
from .matrix import Cell, WitnessMatrix, calibrate_length_residuals, cells_from_beads, sha256_of, units_from_reference
from .registry import DATA_DIR, chapter_lookup, load_chapter_concordance, load_witnesses
from .report import markdown_summary, status_strip_svg, structure_heatmap_svg
from .segments import Segment

ALIGNABLE_KINDS = {"prose", "verse_line", "verse", "mantra", "provisional"}
ZH_WITNESS = "zh_T0892_song"
BO_WITNESS = "bo_derge_D417_418"


def _map_chapters_zh(segments: list[Segment], lookup: dict[int, str]) -> None:
    for s in segments:
        s.chapter = lookup.get(s.local_chapter) if s.local_chapter is not None else None


def _map_chapters_bo(segments: list[Segment], lookup: dict[int, str]) -> None:
    for s in segments:
        toh = (s.extra or {}).get("toh", "")
        part = 1 if toh == "D417" else 2 if toh == "D418" else None
        s.chapter = lookup.get(part * 100 + s.local_chapter) if part and s.local_chapter else None


def _local_key(s: Segment) -> str:
    """Witness-local chapter key: ``pin11`` for Chinese, ``D418:3`` for Tibetan."""
    if s.lang == "zh":
        return f"pin{s.local_chapter}"
    return f"{(s.extra or {}).get('toh', '')}:{s.local_chapter}"


def _ref_to_local_maps(concordance) -> dict[str, dict[str, str]]:
    """For each witness: reference chapter → witness-local chapter key (from the concordance).

    Several reference chapters may share one local key (merged chapters).
    """
    zh: dict[str, str] = {}
    bo: dict[str, str] = {}
    for c in concordance:
        pin = c.zh.get("pin")
        if pin is not None:
            zh[c.ref] = f"pin{pin}"
        n = c.bo.get("chapter")
        if n is not None:
            toh = "D417" if c.ref.startswith("I.") else "D418"
            bo[c.ref] = f"{toh}:{n}"
    return {ZH_WITNESS: zh, BO_WITNESS: bo}


def _review_rows(wname: str, label: str, ru, rs, ws, beads) -> list[dict]:
    rows = []
    for k, b in enumerate(beads):
        rows.append({
            "witness": wname, "chapters": label, "bead_no": k, "shape": f"{b.shape[0]}:{b.shape[1]}",
            "ref_units": " ".join(ru[i].unit_id for i in b.ref),
            "ref_text": " / ".join(rs[i].text for i in b.ref)[:160],
            "wit_span": f"{ws[b.wit[0]].start}-{ws[b.wit[-1]].end}" if b.wit else "",
            "wit_text": " / ".join(ws[j].text for j in b.wit)[:160],
            "similarity": round(b.similarity, 3), "cost": round(b.cost, 3),
            "review_status": "", "review_note": "",
        })
    return rows


def _length_table(ref_segments: list[Segment], targets: dict[str, list[Segment]], ref_to_local_by_witness) -> list[dict]:
    """L1 diagnostic: per reference chapter, reference length vs each witness' mapped length and ratio."""
    from . import normalize

    ref_len: dict[str, int] = {}
    for s in ref_segments:
        if s.chapter:
            ref_len[s.chapter] = ref_len.get(s.chapter, 0) + normalize.length(s.text, s.lang)
    rows = []
    for wname, wsegs in targets.items():
        loc_len: dict[str, int] = {}
        for s in wsegs:
            loc_len[_local_key(s)] = loc_len.get(_local_key(s), 0) + normalize.length(s.text, s.lang)
        r2l = ref_to_local_by_witness[wname]
        # merged groups share the witness length, so report the group and the pooled ratio
        for ch in REF_CHAPTERS:
            key = r2l.get(ch)
            siblings = [c for c in REF_CHAPTERS if r2l.get(c) == key] if key else [ch]
            pooled_ref = sum(ref_len.get(c, 0) for c in siblings)
            wl = loc_len.get(key, 0) if key else 0
            rows.append({
                "chapter": ch, "witness": wname, "witness_local": key or "", "merged_group": "+".join(siblings) if key else "",
                "ref_length": ref_len.get(ch, 0), "witness_length": wl,
                "ratio_pooled": round(wl / pooled_ref, 3) if pooled_ref and wl else "",
            })
    return rows


def load_sanskrit_references(ref_dir: Path) -> dict[str, list[Segment]]:
    out: dict[str, list[Segment]] = {}
    if ref_dir.exists():
        for p in sorted(ref_dir.glob("sa_*.tsv")):
            out[p.stem] = load_reference_tsv(p)
    return out


def run(raw_dir: Path, out_dir: Path, data_dir: Path = DATA_DIR, params: AlignParams | None = None,
        cbeta_file: str = "T18n0892.xml", derge_file: str = "derge_rgyud_bum_nga.txt") -> WitnessMatrix:
    params = params or AlignParams()
    lexicon = load_lexicon(data_dir / "anchors" / "terms.yaml")
    registry = load_witnesses(data_dir / "witnesses.yaml")
    concordance = load_chapter_concordance(data_dir / "concordance" / "chapters.yaml")
    zh_lookup = chapter_lookup(concordance, "zh")
    bo_lookup = chapter_lookup(concordance, "bo")

    # ------------------------------------------------------------------ ingest
    cbeta_path = raw_dir / cbeta_file
    derge_path = raw_dir / derge_file
    zh_doc = parse_cbeta_tei(cbeta_path, witness=ZH_WITNESS)
    _map_chapters_zh(zh_doc.segments, zh_lookup)
    bo_texts = parse_derge_volume(derge_path, ["D417", "D418"], witness=BO_WITNESS)
    bo_segments = [s for t in bo_texts for s in t.segments]
    _map_chapters_bo(bo_segments, bo_lookup)

    sa_refs = load_sanskrit_references(data_dir / "reference")
    ref_scheme = "sa_snellgrove1959"
    if sa_refs:
        reference = ref_scheme if ref_scheme in sa_refs else sorted(sa_refs)[0]
        ref_segments = sa_refs[reference]
        grade = "gold"
    else:
        reference, grade = BO_WITNESS, "provisional"
        ref_segments = [s for s in bo_segments if s.kind in ALIGNABLE_KINDS and s.chapter]

    matrix = WitnessMatrix(reference=reference, reference_grade=grade)
    matrix.inputs = {str(cbeta_path.name): sha256_of(cbeta_path), str(derge_path.name): sha256_of(derge_path)}
    for name in sa_refs:
        matrix.inputs[f"{name}.tsv"] = sha256_of(data_dir / "reference" / f"{name}.tsv")
    units = units_from_reference(ref_segments, provisional=(grade == "provisional"))
    matrix.add_units(units)
    unit_by_ch: dict[str, list] = {}
    for u in units:
        unit_by_ch.setdefault(u.chapter, []).append(u)
    ref_by_ch: dict[str, list[Segment]] = {}
    for s in ref_segments:
        if s.chapter:
            ref_by_ch.setdefault(s.chapter, []).append(s)

    backend = AnchorBackend(lexicon)
    log: dict[str, dict] = {}

    # ------------------------------------------------------------------ other Sanskrit witnesses: id-matched
    for name, segs in sa_refs.items():
        if name == reference:
            continue
        have = {s.seg_id: s for s in segs}
        for u in units:
            s = have.get(u.unit_id)
            if s is None:
                matrix.add_cell(Cell(unit=u.unit_id, witness=name, status="NA", chapter=u.chapter, kind=u.kind))
                continue
            flag = (s.extra or {}).get("flag")
            status = "LACUNA" if flag == "LACUNA" else "ABSENT" if flag == "ABSENT" or not s.text else "PRESENT"
            matrix.add_cell(Cell(unit=u.unit_id, witness=name, status=status, span_start=s.seg_id, span_end=s.seg_id,
                                 d_cov=1.0 if status == "PRESENT" else 0.0, d_split=0.0, chapter=u.chapter, kind=u.kind,
                                 text=s.text[:80]))

    # ------------------------------------------------------------------ translated witnesses: aligned
    targets = {ZH_WITNESS: [s for s in zh_doc.segments if s.kind in ALIGNABLE_KINDS]}
    if grade == "gold":
        targets[BO_WITNESS] = [s for s in bo_segments if s.kind in ALIGNABLE_KINDS]
    ref_to_local_by_witness = _ref_to_local_maps(concordance)
    review_rows: list[dict] = []
    for wname, wsegs in targets.items():
        # witness segments grouped by their *local* chapter; several reference
        # chapters may map onto one local chapter (merged chapters, e.g. zh pin 11 = I.11 + II.1)
        w_by_local: dict[str, list[Segment]] = {}
        for s in wsegs:
            w_by_local.setdefault(_local_key(s), []).append(s)
        ref_to_local = ref_to_local_by_witness[wname]
        groups: dict[str, list[str]] = {}
        for ch in REF_CHAPTERS:
            key = ref_to_local.get(ch)
            if key is not None:
                groups.setdefault(key, []).append(ch)
        all_cells: list[Cell] = []
        aligned_ref_chapters: set[str] = set()
        for key, chs in groups.items():
            ws = w_by_local.get(key, [])
            ru = [u for ch in chs for u in unit_by_ch.get(ch, [])]
            rs = [s for ch in chs for s in ref_by_ch.get(ch, [])]
            if not ru or not ws:
                continue
            beads = align_chapter(rs, ws, backend, params)
            label = "+".join(chs)
            log[f"{wname}:{label}"] = alignment_summary(beads)
            ch_of_unit = {u.unit_id: u.chapter for u in ru}
            new_cells = cells_from_beads(ru, rs, ws, beads, wname, lexicon, label if len(chs) > 1 else chs[0])
            for c in new_cells:
                if c.unit in ch_of_unit:
                    c.chapter = ch_of_unit[c.unit]
            all_cells.extend(new_cells)
            aligned_ref_chapters.update(chs)
            review_rows.extend(_review_rows(wname, label, ru, rs, ws, beads))
        for ch in REF_CHAPTERS:
            if ch in aligned_ref_chapters:
                continue
            ru = unit_by_ch.get(ch, [])
            if not ru:
                continue
            # chapter not mapped for this witness (concordance uncertain) → UNALIGNED, never ABSENT
            for u in ru:
                all_cells.append(Cell(unit=u.unit_id, witness=wname, status="UNALIGNED", chapter=ch, kind=u.kind))
            log[f"{wname}:{ch}"] = {"note": "no witness chapter mapped"}
        medians = calibrate_length_residuals(all_cells, wname)
        log[f"{wname}:length_baseline"] = medians
        for c in all_cells:
            if c.unit.startswith("+"):
                from .matrix import Unit
                matrix.units.setdefault(c.unit, Unit(unit_id=c.unit, chapter=c.chapter or "", kind="orphan", text=c.text, length=0, position=0.0))
            matrix.add_cell(c)

    # ------------------------------------------------------------------ statistics
    sanskrit_witnesses = [w for w in matrix.witnesses if w.startswith("sa_")] + ([reference] if grade == "gold" else [])
    decs = []
    if grade == "gold":
        decs.append(metrics.decompose(matrix, ZH_WITNESS, sanskrit_witnesses, cowitness=BO_WITNESS))
        decs.append(metrics.decompose(matrix, BO_WITNESS, sanskrit_witnesses, cowitness=ZH_WITNESS))
    else:
        decs.append(metrics.decompose(matrix, ZH_WITNESS, [], cowitness=None))
    stat_witnesses = [w for w in matrix.witnesses if w != reference]
    dist = metrics.distance_matrix(matrix, stat_witnesses) if len(stat_witnesses) >= 2 else None
    newick = metrics.average_linkage_newick(*dist) if dist else None
    boot = metrics.block_bootstrap(metrics.deviation_rate_by_chapter(matrix, ZH_WITNESS), metrics.rate)

    # ------------------------------------------------------------------ export
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = matrix.export(out_dir, params=asdict(params))
    manifest["alignment_log"] = log
    manifest["zh_title_note"] = zh_doc.title_note
    manifest["bo_colophons"] = {t.toh: t.colophon for t in bo_texts}
    manifest["zh_deviation_rate_ci"] = boot
    manifest["decompositions"] = [{"witness": d.witness, "cowitness": d.cowitness, "n_counted": d.n_counted,
                                   "n_deviating": d.n_deviating, "counts": d.counts} for d in decs]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "glosses_taisho_sanskrit.json").open("w", encoding="utf-8") as f:
        json.dump(zh_doc.glosses, f, ensure_ascii=False, indent=1)
    with (out_dir / "apparatus_cbeta.json").open("w", encoding="utf-8") as f:
        json.dump(zh_doc.apparatus, f, ensure_ascii=False, indent=1)
    from .matrix import _write_csv

    _write_csv(out_dir / "alignment_review.csv", review_rows)
    _write_csv(out_dir / "structure_lengths.csv", _length_table(ref_segments, targets, ref_to_local_by_witness))
    structure_heatmap_svg(matrix, out_dir / "heatmap_coverage.svg", "coverage")
    structure_heatmap_svg(matrix, out_dir / "heatmap_d_len.svg", "mean_d_len")
    status_strip_svg(matrix, out_dir / "status_strip.svg")
    extra = {"汉译缺失率（品区组自助法 95% 区间）": f"{boot['estimate']:.3f} [{boot['lo']:.3f}, {boot['hi']:.3f}]，{boot['n_blocks']} 个品区组",
             "卷一夹注": zh_doc.title_note or "", "藏译题记": (bo_texts[-1].colophon or "")[:200]}
    (out_dir / "summary.md").write_text(markdown_summary(matrix, decs, dist, newick, extra), encoding="utf-8")
    return matrix
