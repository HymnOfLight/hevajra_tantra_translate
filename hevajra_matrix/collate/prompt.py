"""The static part of a T1 request: prompt versions, template, examples and glossary.

Two prompt versions exist (``VERSIONS``); ``config/llm.yaml: tasks.collate.prompt``
selects one. Both keep the task name ``collate``: the version is captured by the prompt
sha and the schema sha, which every cache key and the instrument digest cover.

    collate.v1  ``prompts/collate.v1.md``, then the three synthetic examples of
                ``data/codebook/collate_examples.yaml`` (other language pairs:
                ``collate_examples.<ref>-<wit>.yaml``); the schema ``SCHEMA_V1``
    collate.v2  ``prompts/collate.v2.md``, then the examples of
                ``data/codebook/collate.v2/examples[.<ref>-<wit>].yaml``, then the
                STANDING RENDERINGS block: the **verified** entries of
                ``data/codebook/collate.v2/standing_renderings[.<ref>-<wit>].yaml``
                ("(none)" while no entry is verified); the schema ``SCHEMA_V2``, which
                adds a ``convention`` to every unit

The glossary travels inside ``Examples`` (``glossary``), so whoever builds requests from
an ``Examples`` (the collate stage, the perturbations, ``prereg``) shows the model the
same entries that V12 (``collate.verify``) later accepts. Only the forms of an entry are
rendered; counts, notes and the name of the verifier never reach the model.

An example of v2 may list its own ``standing_renderings`` (rendered inside the example,
after its comment), so that no example shows an entry of the real glossary; a form of the
real glossary, whatever its status, may not appear among them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..core.io import read_yaml
from ..core.lexicon import StandingEntry, StandingRenderings, load_standing_renderings
from ..core.textnorm import for_quote
from ..core.types import Convention, Relation, WitnessOnlyKind
from ..llm.client import canonical_json
from ..llm.schema import strict, validate
from .windows import prompt_kind

PROMPT_V1 = "collate.v1"
PROMPT_V2 = "collate.v2"
DEFAULT_PAIR = ("bo", "zh")           # (reference, witness) languages of the unsuffixed files
N_EXAMPLES = 3
CONFIDENCE = ("high", "medium", "low")
LANG_NAMES = {"bo": "Tibetan", "zh": "Chinese", "sa": "Sanskrit", "en": "English"}
GLOSSARY_HEADING = "## Standing renderings of this witness"
_LAYOUT_RE = re.compile(r"[\t\r\n]+")


def _schema(conventions: bool) -> dict[str, Any]:
    """The T1 answer schema; with ``conventions`` every unit also records its convention
    (placed before ``relation``: the reading protocol decides the convention first)."""
    convention = {"convention": {"type": "string", "enum": [c.value for c in Convention]}} if conventions else {}
    return strict({
        "type": "object",
        "properties": {
            "units": {"type": "array", "items": {"type": "object", "properties": {
                "ref": {"type": "string"},
                "wit": {"type": "array", "items": {"type": "string"}},
                **convention,
                "relation": {"type": "string", "enum": [r.value for r in Relation]},
                "polarity_flip": {"type": "boolean"},
                "confidence": {"type": "string", "enum": list(CONFIDENCE)},
                "ref_quote": {"type": "string"},
                "wit_quote": {"type": "string"},
            }}},
            "witness_only": {"type": "array", "items": {"type": "object", "properties": {
                "wit": {"type": "array", "items": {"type": "string"}},
                "kind": {"type": "string", "enum": [k.value for k in WitnessOnlyKind]},
                "wit_quote": {"type": "string"},
            }}},
        },
    })


SCHEMA_V1: Mapping[str, Any] = _schema(conventions=False)
SCHEMA_V2: Mapping[str, Any] = _schema(conventions=True)


@dataclass(frozen=True)
class PromptVersion:
    """Where one prompt version keeps its files (paths relative to ``data/``)."""

    id: str
    template_file: str                 # under the package's ``prompts/``
    examples_dir: Path
    examples_stem: str
    schema: Mapping[str, Any]
    glossary_stem: str | None = None   # None: the version has no conventions and no glossary

    @property
    def conventions(self) -> bool:
        return self.glossary_stem is not None


VERSIONS: Mapping[str, PromptVersion] = {
    PROMPT_V1: PromptVersion(PROMPT_V1, "collate.v1.md", Path("codebook"), "collate_examples", SCHEMA_V1),
    PROMPT_V2: PromptVersion(PROMPT_V2, "collate.v2.md", Path("codebook") / "collate.v2", "examples", SCHEMA_V2,
                             "standing_renderings"),
}


def version(prompt: str) -> PromptVersion:
    if prompt not in VERSIONS:
        raise ValueError(f"unknown collate prompt {prompt!r}; known: {sorted(VERSIONS)}")
    return VERSIONS[prompt]


def schema_for(prompt: str = PROMPT_V1) -> Mapping[str, Any]:
    return version(prompt).schema


# --------------------------------------------------------------------------- files
def _pair_file(directory: Path, stem: str, ref_lang: str, wit_lang: str) -> Path:
    if (ref_lang, wit_lang) == DEFAULT_PAIR:
        return directory / f"{stem}.yaml"
    return directory / f"{stem}.{ref_lang}-{wit_lang}.yaml"


def examples_file(ref_lang: str = DEFAULT_PAIR[0], wit_lang: str = DEFAULT_PAIR[1], prompt: str = PROMPT_V1) -> Path:
    """Examples of one language pair, relative to ``data/``. v1:
    ``codebook/collate_examples.yaml`` for the default pair (Tibetan reference, Chinese
    witness), else ``codebook/collate_examples.<ref>-<wit>.yaml`` (e.g. ``sa-zh``); v2:
    the same names with the stem ``examples`` under ``codebook/collate.v2/``."""
    v = version(prompt)
    return _pair_file(v.examples_dir, v.examples_stem, ref_lang, wit_lang)


def glossary_file(ref_lang: str = DEFAULT_PAIR[0], wit_lang: str = DEFAULT_PAIR[1],
                  prompt: str = PROMPT_V2) -> Path | None:
    """The standing-renderings file of one language pair, relative to ``data/`` (None for
    a version without a glossary)."""
    v = version(prompt)
    return None if v.glossary_stem is None else _pair_file(v.examples_dir, v.glossary_stem, ref_lang, wit_lang)


def example_pairs(data_dir: Path, prompt: str = PROMPT_V1) -> list[tuple[str, str]]:
    """The language pairs that have an examples file of ``prompt``, the default pair first.
    The glob is not recursive, so v1 never sees the ``collate.v2/`` directory."""
    v = version(prompt)
    found = sorted((Path(data_dir) / v.examples_dir).glob(f"{v.examples_stem}.*-*.yaml"))
    extra = [tuple(p.name.split(".")[1].split("-", 1)) for p in found]
    return [DEFAULT_PAIR, *[(a, b) for a, b in extra if (a, b) != DEFAULT_PAIR]]


def load_template(prompt: str = PROMPT_V1) -> str:
    """The collation template shipped with the package (``prompts/<prompt>.md``)."""
    name = version(prompt).template_file
    return resources.files("hevajra_matrix").joinpath("prompts").joinpath(name).read_text("utf-8")


# --------------------------------------------------------------------------- examples
@dataclass(frozen=True)
class Examples:
    """The rendered few-shot block and the language pair it illustrates.

    ``glossary`` is None for a version without conventions (v1); for v2 it holds the
    verified standing renderings shown after the examples (possibly none), and
    ``glossary_texts`` the (reference, witness) text ids the glossary file was written for.
    """

    reference_lang: str
    witness_lang: str
    text: str
    prompt: str = PROMPT_V1
    glossary: StandingRenderings | None = None
    glossary_texts: tuple[str, str] | None = None


def load_examples(data_dir: Path, ref_lang: str = DEFAULT_PAIR[0], wit_lang: str = DEFAULT_PAIR[1],
                  prompt: str = PROMPT_V1) -> Examples:
    """Read, check and render the examples of one language pair (``examples_file``).

    The file must hold exactly three examples of the requested pair; each answer must be
    valid against the version's schema, name only handles its example shows and record
    every reference line exactly once. (That the answers also pass the V-checks is a unit
    test.) A missing file raises ``FileNotFoundError``: examples are written by a
    researcher, never borrowed from another language pair or version. v2 also needs the
    pair's standing-renderings file (``entries: []`` when nothing is listed).
    """
    v = version(prompt)
    path = Path(data_dir) / examples_file(ref_lang, wit_lang, prompt)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: write three synthetic {ref_lang}->{wit_lang} examples "
                                f"in the format of {examples_file(*DEFAULT_PAIR, prompt).name}")
    doc = read_yaml(path)
    if not isinstance(doc, Mapping) or set(doc) != {"schema_version", "reference_lang", "witness_lang", "examples"}:
        raise ValueError(f"{path}: expected keys schema_version, reference_lang, witness_lang, examples")
    if (doc["reference_lang"], doc["witness_lang"]) != (ref_lang, wit_lang):
        raise ValueError(f"{path}: illustrates {doc['reference_lang']}->{doc['witness_lang']}, "
                         f"expected {ref_lang}->{wit_lang}")
    items = doc["examples"]
    if doc["schema_version"] != 1 or not isinstance(items, list) or len(items) != N_EXAMPLES:
        raise ValueError(f"{path}: schema_version 1 with exactly {N_EXAMPLES} examples required")
    blocks = [_render_example(n, ex, ref_lang, wit_lang, f"{path}: example {n}", v)
              for n, ex in enumerate(items, start=1)]
    if not v.conventions:
        return Examples(ref_lang, wit_lang, "\n\n".join(blocks))
    glossary, texts, forms = load_glossary(data_dir, ref_lang, wit_lang, prompt)
    for n, ex in enumerate(items, start=1):
        reused = _reused_forms(example_glossary(ex, f"{path}: example {n}"), forms, ref_lang, wit_lang)
        if reused:
            raise ValueError(f"{path}: example {n} lists form(s) of the real standing renderings {reused}; "
                             "examples must invent their own")
    return Examples(ref_lang, wit_lang, "\n\n".join(blocks), prompt, glossary, texts)


def load_glossary(data_dir: Path, ref_lang: str, wit_lang: str,
                  prompt: str = PROMPT_V2) -> tuple[StandingRenderings, tuple[str, str], dict[str, set[str]]]:
    """The verified standing renderings of one language pair, the (reference, witness)
    text ids the file names, and every form the file lists under ``entries`` whatever its
    status (side "ref" or "wit" -> forms; used to keep examples apart from it)."""
    rel = glossary_file(ref_lang, wit_lang, prompt)
    if rel is None:
        raise ValueError(f"{prompt} has no standing renderings")
    path = Path(data_dir) / rel
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: list the verified standing renderings of the {ref_lang}->"
                                f"{wit_lang} witness (entries: [] when there are none)")
    doc = read_yaml(path)
    texts = (doc.get("reference"), doc.get("witness")) if isinstance(doc, Mapping) else (None, None)
    if not all(isinstance(t, str) and t for t in texts):
        raise ValueError(f"{path}: the file must name its reference and witness text ids")
    glossary = load_standing_renderings(path, texts[0], texts[1], ref_lang, wit_lang)
    forms: dict[str, set[str]] = {"ref": set(), "wit": set()}
    for entry in doc.get("entries") or ():
        forms["ref"].update(str(f) for f in entry.get("reference_forms") or ())
        forms["wit"].update(str(f) for f in entry.get("witness_renderings") or ())
    return glossary, (texts[0], texts[1]), forms


def example_glossary(ex: Any, where: str) -> StandingRenderings:
    """An example's own ``standing_renderings`` (a list of {reference, witness} pairs)."""
    if not isinstance(ex, Mapping) or "standing_renderings" not in ex:
        return StandingRenderings()
    items = ex["standing_renderings"]
    if not isinstance(items, list) or not items:
        raise ValueError(f"{where}: standing_renderings must be a non-empty list of {{reference, witness}}")
    entries = []
    for i, item in enumerate(items, start=1):
        if (not isinstance(item, Mapping) or set(item) != {"reference", "witness"}
                or not all(isinstance(item[k], str) and item[k].strip() for k in item)):
            raise ValueError(f"{where}: standing rendering {i} needs a non-empty reference and witness")
        entries.append(StandingEntry(f"example.{i}", (item["reference"].strip(),), (item["witness"].strip(),)))
    return StandingRenderings(tuple(entries))


def _reused_forms(own: StandingRenderings, real: Mapping[str, set[str]], ref_lang: str,
                  wit_lang: str) -> list[str]:
    """Forms of an example's glossary that contain, or occur in, a form of the real one."""
    out = []
    for side, lang, attr in (("ref", ref_lang, "reference_forms"), ("wit", wit_lang, "witness_renderings")):
        keys = {for_quote(f, lang) for f in real[side]} - {""}
        for entry in own.entries:
            for form in getattr(entry, attr):
                key = for_quote(form, lang)
                if any(key in k or k in key for k in keys):
                    out.append(form)
    return out


def _render_example(n: int, ex: Any, ref_lang: str, wit_lang: str, where: str, v: PromptVersion) -> str:
    keys = {"title", "comment", "witness", "reference", "answer"}
    if not isinstance(ex, Mapping) or not keys <= set(ex) <= keys | ({"standing_renderings"} if v.conventions
                                                                     else set()):
        raise ValueError(f"{where}: expected keys {sorted(keys)}"
                         + (" and optionally standing_renderings" if v.conventions else ""))
    wit = [(f"z{i:04d}", line["kind"], line["text"]) for i, line in enumerate(ex["witness"], start=1)]
    ref = [(f"r{i:03d}", line["kind"], line["text"]) for i, line in enumerate(ex["reference"], start=1)]
    answer = ex["answer"]
    errors = validate(answer, v.schema)
    if errors:
        raise ValueError(f"{where}: answer violates the schema: {errors[:3]}")
    named = [h for u in answer["units"] for h in u["wit"]] + [h for w in answer["witness_only"] for h in w["wit"]]
    unknown = sorted({h for h in named if h not in {z for z, _, _ in wit}}
                     | {u["ref"] for u in answer["units"] if u["ref"] not in {r for r, _, _ in ref}})
    if unknown or [u["ref"] for u in answer["units"]] != [r for r, _, _ in ref]:
        raise ValueError(f"{where}: the answer must record r001.. once each, in order, and name only shown "
                         f"handles (unknown: {unknown})")
    own = example_glossary(ex, where) if v.conventions else None
    if own is not None:
        _check_example_conventions(answer, own, ref_lang, wit_lang, where)
    return "\n".join([
        f"### Example {n}: {ex['title']}",
        "",
        str(ex["comment"]).strip(),
        "",
        *([render_glossary(own, ref_lang, wit_lang), ""] if own is not None and own.entries else []),
        render_context(wit, wit_lang),
        "",
        render_body(ref, ref_lang, [(wit[0][0], wit[-1][0])]),
        "",
        "ANSWER",
        canonical_json(answer),
    ])


def _check_example_conventions(answer: Mapping[str, Any], own: StandingRenderings, ref_lang: str, wit_lang: str,
                               where: str) -> None:
    """A unit with a convention quotes both sides; a standing rendering quotes a listed pair."""
    for u in answer["units"]:
        if u["convention"] == Convention.NONE.value:
            continue
        if not (u["ref_quote"].strip() and u["wit_quote"].strip()):
            raise ValueError(f"{where}: {u['ref']} has convention {u['convention']} but not both quotes")
        if u["convention"] == Convention.STANDING_RENDERING.value and not any(
                any(for_quote(r, ref_lang) in for_quote(u["ref_quote"], ref_lang) for r in e.reference_forms)
                and any(for_quote(w, wit_lang) in for_quote(u["wit_quote"], wit_lang) for w in e.witness_renderings)
                for e in own.entries):
            raise ValueError(f"{where}: {u['ref']} claims a standing rendering the example does not list")


# --------------------------------------------------------------------------- rendering
def system_prompt(template: str, examples: Examples) -> str:
    """Template, examples and (v2) the standing renderings: the cached ``system`` block."""
    if examples.glossary is None:
        return f"{template.rstrip()}\n\n{examples.text}\n"
    block = render_glossary(examples.glossary, examples.reference_lang, examples.witness_lang)
    return f"{template.rstrip()}\n\n{examples.text}\n\n{GLOSSARY_HEADING}\n\n{block}\n"


def render_glossary(glossary: StandingRenderings, ref_lang: str, wit_lang: str) -> str:
    """The STANDING RENDERINGS block: one ``forms = renderings`` line per entry, or
    "(none)". Only the forms are printed."""
    lines = [" / ".join(_flat(f) for f in e.reference_forms) + " = " + " / ".join(_flat(w) for w in
                                                                                    e.witness_renderings)
             for e in glossary.entries]
    header = f"STANDING RENDERINGS ({LANG_NAMES.get(ref_lang, ref_lang)} -> {LANG_NAMES.get(wit_lang, wit_lang)})"
    return "\n".join([header, *(lines or ["(none)"])])


def _flat(text: str) -> str:
    return _LAYOUT_RE.sub(" ", text).strip()


def _line(handle: str, kind: str, text: str) -> str:
    """One prompt line; tabs and line breaks inside the text would break the format."""
    return f"{handle}\t{prompt_kind(kind)}\t{_flat(text)}"


def render_context(lines: Sequence[tuple[str, str, str]], lang: str) -> str:
    """The witness text block: a header and one (handle, kind, text) line per segment."""
    return "\n".join([f"WITNESS TEXT ({LANG_NAMES.get(lang, lang)})", *(_line(*x) for x in lines)])


def render_body(lines: Sequence[tuple[str, str, str]], lang: str, core: Sequence[tuple[str, str]]) -> str:
    """The reference chunk block and the core-window ranges."""
    ranges = ", ".join(a if a == b else f"{a}-{b}" for a, b in core) or "(empty)"
    return "\n".join([
        f"REFERENCE CHUNK ({LANG_NAMES.get(lang, lang)})",
        *(_line(*x) for x in lines),
        "",
        f"CORE WINDOW: {ranges}",
    ])
