"""Command line entry point.

    python -m hevajra_matrix fetch  --raw data/raw
    python -m hevajra_matrix run    --raw data/raw --out out [--similarity hybrid --embedding-model BAAI/bge-m3]
    python -m hevajra_matrix transfer --config data/transfer/laozi --texts data/transfer/laozi/texts --out out/laozi
    python -m hevajra_matrix llm-judge|llm-extract|llm-attribute|llm-probe|llm-counterfactual --llm qwen3-32b-awq ...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .align import AlignParams
from .registry import DATA_DIR

SOURCES = {
    "T18n0892.xml": {
        "url": "https://raw.githubusercontent.com/cbeta-org/xml-p5/master/T/T18/T18n0892.xml",
        "license": "CC BY-NC-SA 4.0 (CBETA)",
        "witness": "zh_T0892_song",
    },
    "derge_rgyud_bum_nga.txt": {
        "url": "https://raw.githubusercontent.com/Esukhia/derge-kangyur/master/text/080_%E0%BD%A2%E0%BE%92%E0%BE%B1%E0%BD%B4%E0%BD%91%E0%BC%8B%E0%BD%A0%E0%BD%96%E0%BD%B4%E0%BD%98%E0%BC%8D_%E0%BD%84.txt",
        "license": "see Esukhia/derge-kangyur README (to verify)",
        "witness": "bo_derge_D417_418",
    },
}


def cmd_fetch(args: argparse.Namespace) -> int:
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for name, meta in SOURCES.items():
        dest = raw / name
        if dest.exists() and not args.force:
            print(f"exists  {dest}")
        else:
            print(f"fetch   {meta['url']}")
            with urllib.request.urlopen(meta["url"], timeout=60) as r:
                dest.write_bytes(r.read())
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        manifest[name] = {**meta, "sha256": sha, "fetched_at": datetime.now(timezone.utc).isoformat(), "bytes": dest.stat().st_size}
        print(f"        sha256 {sha[:16]}…  {dest.stat().st_size} bytes")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run

    params = AlignParams(prior_null=args.prior_null, anchor_weight=args.anchor_weight)
    m = run(Path(args.raw), Path(args.out), data_dir=Path(args.data), params=params,
            similarity=args.similarity, embedding_model=args.embedding_model)
    print(f"reference={m.reference} ({m.reference_grade}) units={len(m.units)} cells={len(m.cells)} → {args.out}/")
    return 0


def cmd_transfer(args: argparse.Namespace) -> int:
    from .transfer.sinitic import run_transfer

    m = run_transfer(Path(args.config), Path(args.texts), Path(args.out))
    print(f"reference={m.reference} ({m.reference_grade}) witnesses={m.witnesses} units={len(m.units)} → {args.out}/")
    return 0


# --------------------------------------------------------------------------- LLM subcommands
def _backend(args: argparse.Namespace):
    from .llm.backends import load_backend

    return load_backend(args.llm, Path(args.models) if args.models else None)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    from .matrix import _write_csv as w

    path.parent.mkdir(parents=True, exist_ok=True)
    w(path, rows)


def cmd_llm_judge(args: argparse.Namespace) -> int:
    """R2: fill review_status/review_note of alignment_review.csv for suspicious beads."""
    from .llm.tasks import judge_review_rows

    b = _backend(args)
    path = Path(args.review)
    rows = _read_csv(path)
    if args.limit:
        rows = rows[: args.limit]
    rows = judge_review_rows(b, rows, ref_lang=args.ref_lang, wit_lang=args.wit_lang,
                             only_shapes=tuple(args.shapes.split(",")), min_confidence=args.min_confidence)
    out = Path(args.out or path.with_name(path.stem + "_llm.csv"))
    _write_csv(out, rows)
    _dump_log(b, out.with_suffix(".calls.json"))
    n = sum(1 for r in rows if str(r.get("review_status", "")).startswith("llm:"))
    print(f"{n} beads judged by {b.model_id} → {out}")
    return 0


def cmd_llm_extract(args: argparse.Namespace) -> int:
    """R4: lexicon-constrained component extraction for cells.csv rows (derived layer only)."""
    from .anchors import load_lexicon
    from .llm.tasks import extract_components

    b = _backend(args)
    lexicon = load_lexicon(Path(args.data) / "anchors" / "terms.yaml")
    rows = _read_csv(Path(args.cells))
    rows = [r for r in rows if r.get("status") in ("PRESENT", "PARTIAL") and r.get("text")]
    if args.witness:
        rows = [r for r in rows if r["witness"] == args.witness]
    if args.limit:
        rows = rows[: args.limit]
    lang = args.lang
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            cs = extract_components(b, r["text"], lang, lexicon)
            f.write(json.dumps({"unit": r["unit"], "witness": r["witness"], **cs.to_row()}, ensure_ascii=False) + "\n")
    _dump_log(b, out_path.with_suffix(".calls.json"))
    print(f"{len(rows)} segments → {out_path}")
    return 0


def cmd_llm_attribute(args: argparse.Namespace) -> int:
    """R6: fixed-label attribution judgement for deviating cells, facts drawn from the matrix only."""
    from .llm.tasks import judge_attribution

    b = _backend(args)
    cells = _read_csv(Path(args.cells))
    by_unit: dict[str, dict[str, dict]] = {}
    for r in cells:
        by_unit.setdefault(r["unit"], {})[r["witness"]] = r
    targets = [r for r in cells if r["witness"] == args.witness and r["status"] in ("ABSENT", "PARTIAL")]
    if args.limit:
        targets = targets[: args.limit]
    out_rows = []
    for r in targets:
        facts = [f"{r['witness']} status at {r['unit']}: {r['status']} (bead {r['bead']}, evidence grade {r['evidence']})"]
        for w, o in by_unit[r["unit"]].items():
            if w != r["witness"]:
                facts.append(f"{w} status at {r['unit']}: {o['status']}")
        j = judge_attribution(b, r["unit"], r["witness"], facts)
        out_rows.append({"unit": j.unit, "witness": j.witness, "top": j.top, "motive_flag": j.motive_flag,
                         "cited_facts": " ".join(map(str, j.cited_facts)), "model_id": j.model_id,
                         **{f"p_{k}": round(v, 3) for k, v in j.distribution.items()}, "note": j.note})
    out = Path(args.out)
    _write_csv(out, out_rows)
    _dump_log(b, out.with_suffix(".calls.json"))
    flagged = sum(1 for r in out_rows if r["motive_flag"])
    print(f"{len(out_rows)} cells judged by {b.model_id}; {flagged} downgraded for motive language → {out}")
    return 0


def cmd_llm_probe(args: argparse.Namespace) -> int:
    """R8: memorisation probe over the segments of a run (cells.csv text column or a TSV of passages)."""
    from .llm.tasks import contamination_probe, probe_items_from_segments

    b = _backend(args)
    texts: list[tuple[str, str, str]] = []
    p = Path(args.passages)
    if p.suffix == ".csv":
        for r in _read_csv(p):
            if r.get("text"):
                texts.append((r.get("witness", p.stem), r.get("unit", ""), r["text"]))
    else:
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                texts.append((parts[0], parts[1], parts[2]))
    items = probe_items_from_segments(texts, prefix_chars=args.prefix_chars, min_chars=args.min_chars)
    if args.limit:
        items = items[: args.limit]
    res = contamination_probe(b, items, threshold=args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    _dump_log(b, out.with_suffix(".calls.json"))
    print(f"{res['n']} probes, memorised_rate={res['memorised_rate']:.3f} mean_overlap={res['mean_overlap']:.3f} → {out}")
    return 0


def cmd_llm_counterfactual(args: argparse.Namespace) -> int:
    """R7: over-attribution counterfactual across several models (docs/02 §9)."""
    from .attribution import CounterfactualItem
    from .llm.backends import load_backend
    from .llm.tasks import counterfactual_across_models

    items = []
    for r in _read_csv(Path(args.items)):
        items.append(CounterfactualItem(unit=r["unit"], sa_witness=r.get("sa_witness", "sa"), sa_text=r["sa_text"],
                                        zh_span=r.get("zh_span", ""), zh_text=r.get("zh_text", ""),
                                        cowitness_fact=r["cowitness_fact"]))
    if args.limit:
        items = items[: args.limit]
    backends = [load_backend(n, Path(args.models) if args.models else None) for n in args.llm.split(",")]
    rows = counterfactual_across_models(backends, items)
    _write_csv(Path(args.out), rows)
    for r in rows:
        print(f"{r['model_id']}: motive rate {r['motive_rate_without_fact']:.2f} → {r['motive_rate_with_fact']:.2f} (drop {r['drop']:.2f})")
    return 0


def _dump_log(backend, path: Path) -> None:
    log = getattr(backend, "log", None)
    if log is not None and log.records:
        log.dump(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hevajra_matrix")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download CBETA T0892 and the Derge volume into --raw")
    f.add_argument("--raw", default="data/raw")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="ingest, align, build matrix, report")
    r.add_argument("--raw", default="data/raw")
    r.add_argument("--out", default="out")
    r.add_argument("--data", default=str(DATA_DIR))
    r.add_argument("--prior-null", type=float, default=AlignParams.prior_null)
    r.add_argument("--anchor-weight", type=float, default=AlignParams.anchor_weight)
    r.add_argument("--similarity", choices=["anchors", "embedding", "hybrid"], default="anchors",
                   help="B0 anchors (default) | B1 embeddings | hybrid (needs sentence-transformers + GPU)")
    r.add_argument("--embedding-model", default="BAAI/bge-m3")
    r.set_defaults(func=cmd_run)

    t = sub.add_parser("transfer", help="same-language multi-witness matrix (Laozi etc.)")
    t.add_argument("--config", default=str(DATA_DIR / "transfer" / "laozi"))
    t.add_argument("--texts", default=str(DATA_DIR / "transfer" / "laozi" / "texts"))
    t.add_argument("--out", default="out/laozi")
    t.set_defaults(func=cmd_transfer)

    def llm_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--llm", default="mock", help="profile name in data/llm/models.yaml, or 'mock'")
        p.add_argument("--models", default=None, help="path to models.yaml (default data/llm/models.yaml)")
        p.add_argument("--limit", type=int, default=0)

    j = sub.add_parser("llm-judge", help="R2: LLM verdicts on suspicious alignment beads (review table only)")
    llm_common(j)
    j.add_argument("--review", default="out/alignment_review.csv")
    j.add_argument("--out", default=None)
    j.add_argument("--ref-lang", default="bo")
    j.add_argument("--wit-lang", default="zh")
    j.add_argument("--shapes", default="1:0,0:1,1:3,3:1,2:2")
    j.add_argument("--min-confidence", type=float, default=0.6)
    j.set_defaults(func=cmd_llm_judge)

    e = sub.add_parser("llm-extract", help="R4: lexicon-constrained component extraction → JSONL (derived layer)")
    llm_common(e)
    e.add_argument("--cells", default="out/cells.csv")
    e.add_argument("--data", default=str(DATA_DIR))
    e.add_argument("--witness", default=None)
    e.add_argument("--lang", default="zh")
    e.add_argument("--out", default="out/derived/components.jsonl")
    e.set_defaults(func=cmd_llm_extract)

    a = sub.add_parser("llm-attribute", help="R6: fixed-label attribution judgement for deviating cells")
    llm_common(a)
    a.add_argument("--cells", default="out/cells.csv")
    a.add_argument("--witness", default="zh_T0892_song")
    a.add_argument("--out", default="out/derived/attribution_llm.csv")
    a.set_defaults(func=cmd_llm_attribute)

    pr = sub.add_parser("llm-probe", help="R8: memorisation / contamination probe")
    llm_common(pr)
    pr.add_argument("--passages", default="out/cells.csv", help="cells.csv or TSV: source<TAB>ref<TAB>text")
    pr.add_argument("--prefix-chars", type=int, default=40)
    pr.add_argument("--min-chars", type=int, default=80)
    pr.add_argument("--threshold", type=float, default=0.5)
    pr.add_argument("--out", default="out/derived/contamination_probe.json")
    pr.set_defaults(func=cmd_llm_probe)

    cf = sub.add_parser("llm-counterfactual", help="R7: over-attribution counterfactual across models")
    llm_common(cf)
    cf.add_argument("--items", required=True, help="CSV: unit,sa_witness,sa_text,zh_span,zh_text,cowitness_fact")
    cf.add_argument("--out", default="out/derived/counterfactual.csv")
    cf.set_defaults(func=cmd_llm_counterfactual)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
