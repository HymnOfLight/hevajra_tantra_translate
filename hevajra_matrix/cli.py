"""Command line entry point.

    python -m hevajra_matrix fetch  --raw data/raw
    python -m hevajra_matrix run    --raw data/raw --out out
"""

from __future__ import annotations

import argparse
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
    m = run(Path(args.raw), Path(args.out), data_dir=Path(args.data), params=params)
    print(f"reference={m.reference} ({m.reference_grade}) units={len(m.units)} cells={len(m.cells)} → {args.out}/")
    return 0


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
    r.set_defaults(func=cmd_run)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
