"""Report sections built from the JSON the stages write (pure; used by ``report.markdown``):
the E3 decomposition (``stats/details.json: e3``) and the confidence reliability of the
instrument (``evaluation/scores.json: sources.<source>.reliability``) with the P-negation
polarity recall (``evaluation/perturbations.json: negation``)."""

from __future__ import annotations

from typing import Any, Mapping


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def e3(d: Mapping[str, Any]) -> list[str]:
    """The decomposition counts as measured and under the A2 bound, and the excess over
    independence (O, E, O - E, phi) of the Vorlage and shared indicators (synthesis 6.4)."""
    def counts(c: Mapping[str, int]) -> str:
        return ", ".join(f"{k} {v}" for k, v in c.items())

    def excess(label: str, e: Mapping[str, Any]) -> str:
        return (f"  - {label}: O {e['observed']}, E {_num(e['expected'])}, O - E {_num(e['excess'])}, "
                f"phi {_num(e['phi'])} (n = {e['n']})")

    bound = d.get("bound") or {}
    return [f"- E3 decomposition of deviations (Vorlage, shared, residual): {d.get('n_deviating')} deviating units "
            f"of {d.get('units')}; manuscripts {', '.join(d.get('manuscripts') or [])}; co-witness "
            f"{d.get('cowitness')}; m_min {d.get('m_min')}",
            f"  - as measured: {counts(d.get('counts') or {})}",
            f"  - A2 bound ({d.get('verified_deviating')} deviating units human-verified; "
            f"{bound.get('unverified_shared')} unverified shared counted as not shared): "
            f"{counts(bound.get('counts') or {})}",
            excess("Vorlage (a manuscript lacks or varies the unit)", d["vorlage"]),
            excess("shared (the co-witness deviates too)", d["shared"]),
            excess("shared, A2 bound", bound["shared"])]


def reliability(by_source: Mapping[str, Mapping[str, Any] | None], negation: Mapping[str, int] | None) -> list[str]:
    """Per labelled source: Brier score of its confidence against the constant predictor and
    the reliability table; then the P-negation polarity recall when it was run."""
    out = []
    for label, r in by_source.items():
        if not r:
            continue
        beats = bool(r.get("beats_constant"))
        out += [f"### Confidence reliability, {label}", "",
                f"Over {r['n']} units with a confidence, status correct in {_num(r.get('accuracy'))}. Brier score "
                f"{_num(r.get('brier'))} (nominal high/medium/low probabilities) vs {_num(r.get('brier_constant'))} "
                f"for a constant predictor: confidence {'beats' if beats else 'does not beat'} the constant, so it "
                f"is {'informative' if beats else 'used only as a stratum'}.", "",
                "| confidence | nominal | units | correct | accuracy |", "|---|---|---|---|---|"]
        out += [f"| {name} | {_num(row['nominal'])} | {row['n']} | {row['correct']} | {_num(row['accuracy'])} |"
                for name, row in (r.get("table") or {}).items()] + [""]
    if negation and negation.get("n"):
        out += [f"P-negation (reported, never gated): polarity recall {negation['hits']}/{negation['n']} "
                "perturbed units.", ""]
    return out
