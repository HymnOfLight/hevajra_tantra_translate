"""Human-readable outputs: SVG heatmaps and a Markdown summary. No plotting dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .ids import REF_CHAPTERS
from .matrix import DIMS, WitnessMatrix
from .metrics import Decomposition, summarize_dim

STATUS_COLOR = {"PRESENT": "#2b8a3e", "PARTIAL": "#f59f00", "ABSENT": "#c92a2a",
                "LACUNA": "#868e96", "UNALIGNED": "#adb5bd", "NA": "#f1f3f5"}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def structure_heatmap_svg(matrix: WitnessMatrix, path: Path, metric: str = "coverage") -> None:
    """Chapter × witness heatmap of a structure-level metric (coverage by default)."""
    rows = matrix.structure()
    ws = matrix.witnesses
    cell, left, top = 34, 70, 120
    width, height = max(460, left + cell * len(ws) + 20), top + cell * len(REF_CHAPTERS) + 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">']
    out.append(f'<text x="{left}" y="20" font-size="14">{_esc(metric)} — reference: {_esc(matrix.reference)} ({matrix.reference_grade})</text>')
    for j, w in enumerate(ws):
        x = left + j * cell + cell / 2
        out.append(f'<text x="{x}" y="{top - 6}" transform="rotate(-60 {x} {top - 6})" text-anchor="start">{_esc(w)}</text>')
    lookup = {(r["chapter"], r["witness"]): r for r in rows}
    for i, ch in enumerate(REF_CHAPTERS):
        y = top + i * cell
        out.append(f'<text x="{left - 6}" y="{y + cell * 0.65}" text-anchor="end">{ch}</text>')
        for j, w in enumerate(ws):
            r = lookup.get((ch, w))
            v = r.get(metric, "") if r else ""
            if v == "" or v is None:
                color, label = "#f1f3f5", ""
            else:
                v = float(v)
                if metric == "coverage":
                    g = int(255 * v)
                    color = f"rgb({255 - g},{g},90)"
                else:
                    mag = min(1.0, abs(v))
                    color = f"rgb({int(255 * mag)},{int(255 * (1 - mag))},120)" if v >= 0 else f"rgb(120,{int(255 * (1 - mag))},{int(255 * mag)})"
                label = f"{v:.2f}"
            out.append(f'<rect x="{left + j * cell}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{color}"/>')
            if label:
                out.append(f'<text x="{left + j * cell + cell / 2 - 1}" y="{y + cell * 0.65}" text-anchor="middle" font-size="9">{label}</text>')
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def status_strip_svg(matrix: WitnessMatrix, path: Path, witnesses: Sequence[str] | None = None, max_units: int = 4000) -> None:
    """Unit × witness status strip: one thin column per unit, one row per witness."""
    ws = list(witnesses or matrix.witnesses)
    units = [u for u in matrix.ordered_units() if not u.startswith("+")][:max_units]
    px, row_h, left, top = max(1, min(4, 1600 // max(1, len(units)))), 18, 160, 40
    width = left + px * len(units) + 10
    height = top + row_h * len(ws) + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">']
    out.append(f'<text x="{left}" y="18">status per reference unit ({len(units)} units) — green PRESENT, amber PARTIAL, red ABSENT, grey LACUNA</text>')
    for i, w in enumerate(ws):
        y = top + i * row_h
        out.append(f'<text x="{left - 6}" y="{y + 13}" text-anchor="end">{_esc(w)}</text>')
        for k, u in enumerate(units):
            c = matrix.get(u, w)
            color = STATUS_COLOR.get(c.status if c else "NA", "#f1f3f5")
            out.append(f'<rect x="{left + k * px}" y="{y}" width="{px}" height="{row_h - 2}" fill="{color}"/>')
    # chapter ticks
    last = None
    for k, u in enumerate(units):
        ch = matrix.units[u].chapter
        if ch != last:
            x = left + k * px
            out.append(f'<line x1="{x}" y1="{top - 4}" x2="{x}" y2="{top + row_h * len(ws)}" stroke="#000" stroke-width="0.5"/>')
            out.append(f'<text x="{x + 2}" y="{top + row_h * len(ws) + 14}" font-size="9">{ch}</text>')
            last = ch
    out.append("</svg>")
    path.write_text("\n".join(out), encoding="utf-8")


def markdown_summary(matrix: WitnessMatrix, decompositions: Sequence[Decomposition], distances: tuple[list[str], list[list[float | None]]] | None,
                     newick: str | None, extra: dict | None = None) -> str:
    lines = ["# 见证矩阵运行摘要", ""]
    lines.append(f"- 参照：`{matrix.reference}`（{matrix.reference_grade}）")
    lines.append(f"- 行数（参照单元）：{sum(1 for u in matrix.units if not u.startswith('+'))}，孤儿行：{sum(1 for u in matrix.units if u.startswith('+'))}")
    lines.append(f"- 列（见证）：{', '.join(matrix.witnesses)}")
    if matrix.reference_grade == "provisional":
        lines.append("- **注意**：未提供梵文电子文本，本次以藏译德格本为临时参照；所有数值只描述汉译相对藏译的关系，不构成相对梵文的偏移。")
    lines.append("")
    lines.append("## L1 结构层（品 × 见证）")
    lines.append("")
    lines.append("| 品 | 见证 | 单元 | 计数 | 存在 | 缺失 | 孤儿 | 覆盖 | d_len 均值 | d_lit 均值 | d_ord 均值 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in matrix.structure():
        if r["n_units"] == 0 and r["n_orphan"] == 0:
            continue
        lines.append(f"| {r['chapter']} | {r['witness']} | {r['n_units']} | {r['n_counted']} | {r['n_present']} | {r['n_absent']} | {r['n_orphan']} | {r['coverage']} | {r['mean_d_len']} | {r['mean_d_lit']} | {r['mean_d_ord']} |")
    lines.append("")
    lines.append("## 各维度分布（PRESENT/PARTIAL 单元）")
    lines.append("")
    lines.append("| 见证 | 维度 | n | 均值 | 中位数 | p90 |")
    lines.append("|---|---|---|---|---|---|")
    for w in matrix.witnesses:
        for dim in DIMS:
            s = summarize_dim(matrix, w, dim)
            if s.get("n"):
                lines.append(f"| {w} | {dim} | {s['n']} | {s['mean']:.3f} | {s['median']:.3f} | {s['p90']:.3f} |")
    if decompositions:
        lines.append("")
        lines.append("## 三分分解")
        lines.append("")
        lines.append("| 见证 | 同源见证 | 计数单元 | 偏移单元 | Vorlage 可解释 | 共享 | 残余 |")
        lines.append("|---|---|---|---|---|---|---|")
        for d in decompositions:
            r = d.rates()
            lines.append(f"| {d.witness} | {d.cowitness or '—'} | {d.n_counted} | {d.n_deviating} | {d.counts['vorlage_explained']} ({r['vorlage_explained']:.3f}) | {d.counts['shared_with_cowitness']} ({r['shared_with_cowitness']:.3f}) | {d.counts['residual']} ({r['residual']:.3f}) |")
        if any(not d.sanskrit_witnesses or d.sanskrit_witnesses == [matrix.reference] for d in decompositions):
            lines.append("")
            lines.append("只有一列梵文参照时，“Vorlage 可解释”恒为 0，分解退化为两项；见 docs/03 §3。")
    if distances:
        ws, D = distances
        lines.append("")
        lines.append("## 见证间距离")
        lines.append("")
        lines.append("| | " + " | ".join(ws) + " |")
        lines.append("|---|" + "---|" * len(ws))
        for i, a in enumerate(ws):
            lines.append(f"| {a} | " + " | ".join("—" if D[i][j] is None else f"{D[i][j]:.3f}" for j in range(len(ws))) + " |")
        if newick:
            lines.append("")
            lines.append(f"UPGMA: `{newick}`")
    if extra:
        lines.append("")
        lines.append("## 其他")
        lines.append("")
        for k, v in extra.items():
            lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)
