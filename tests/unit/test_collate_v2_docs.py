"""The glossary instructions of data/codebook/collate.v2/README.md hold (review findings C7,
C9, C10): the tripwire test the README names for a verification exists; the README's
template entry (with ``status``) loads, and one without ``status`` is refused; only a change
to the verified entries changes the v2 system prompt, a proposed entry changes nothing.

Multilingual strings come from the committed glossary file; no non-Latin literal appears here.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import yaml

from hevajra_matrix.collate.collator import PROMPT_V2, load_examples, load_template, system_prompt
from hevajra_matrix.core.lexicon import LexiconError, load_standing_renderings

from test_collate_support import DATA
from test_collate_v2 import glossary_doc

README = DATA / "codebook" / "collate.v2" / "README.md"
TESTS = Path(__file__).resolve().parent


def _template_entry() -> dict:
    """The README's yaml template entry, its placeholders filled with the shipped forms."""
    block = re.search(r"```yaml\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert block, "the README shows a template entry"
    entry = yaml.safe_load(block.group(1))[0]
    real = glossary_doc()["entries"][0]
    return {**entry, "id": "doc_template", "reference_forms": list(real["reference_forms"]),
            "witness_renderings": list(real["witness_renderings"])}


def _glossary_file(tmp_path: Path, entries: list[dict]) -> Path:
    doc = {k: v for k, v in glossary_doc().items() if k != "entries"}
    path = tmp_path / "standing_renderings.yaml"
    path.write_text(yaml.safe_dump({**doc, "entries": entries}, allow_unicode=True), encoding="utf-8")
    return path


def _load(path: Path):
    doc = glossary_doc()
    return load_standing_renderings(path, doc["reference"], doc["witness"])


def test_the_readme_names_an_existing_tripwire_test() -> None:
    text = README.read_text(encoding="utf-8")
    named = re.findall(r"tests/unit/(test_\w+\.py)::(test_\w+)", text)
    assert named, "README step 3 names the test to update in the same commit"
    for file, func in named:
        assert re.search(rf"^def {func}\(", (TESTS / file).read_text(encoding="utf-8"), re.M), (file, func)


def test_the_readme_template_entry_loads_and_status_is_required(tmp_path: Path) -> None:
    entry = _template_entry()
    assert entry["status"] == "proposed" and entry["verified_by"] is None
    assert _load(_glossary_file(tmp_path, [entry])).entries == ()
    signed = {**entry, "status": "verified", "verified_by": "A. Researcher"}
    assert [e.id for e in _load(_glossary_file(tmp_path, [signed])).entries] == ["doc_template"]
    without = {k: v for k, v in entry.items() if k != "status"}
    with pytest.raises(LexiconError, match="status"):
        _load(_glossary_file(tmp_path, [without]))


def _v2_prompt(data: Path) -> str:
    return system_prompt(load_template(PROMPT_V2), load_examples(data, prompt=PROMPT_V2))


def test_only_verified_entries_change_the_v2_prompt(tmp_path: Path) -> None:
    data = tmp_path / "data"
    shutil.copytree(DATA / "codebook" / "collate.v2", data / "codebook" / "collate.v2")
    shipped = _v2_prompt(data)
    assert shipped == _v2_prompt(DATA)
    target = data / "codebook" / "collate.v2" / "standing_renderings.yaml"
    doc = glossary_doc()
    real = doc["entries"][0]
    proposed = {**_template_entry(), "id": "doc_template", "status": "proposed", "verified_by": None}

    def write(entries: list[dict]) -> str:
        target.write_text(yaml.safe_dump({**doc, "entries": entries}, allow_unicode=True), encoding="utf-8")
        return _v2_prompt(data)

    assert write([real, proposed]) == shipped, "adding a proposed entry changes nothing"
    assert write([real, {**proposed, "status": "retired"}]) == shipped, "retiring a proposed entry neither"
    assert write([]) == shipped, "removing a proposed entry neither"
    assert write([{**real, "status": "verified", "verified_by": "A. Researcher"}]) != shipped
