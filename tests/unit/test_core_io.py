"""core.io: round trips, BOM handling, append-only JSONL, hashing, git and the run manifest."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import hevajra_matrix
from hevajra_matrix.config import Settings, load_settings
from hevajra_matrix.core.io import (
    append_jsonl,
    git_commit,
    git_dirty,
    read_csv,
    read_jsonl,
    read_yaml,
    sha256_file,
    write_csv,
    write_jsonl,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TIBETAN = "\u0f62\u0fa1\u0f7c\u0f0b\u0f62\u0f97\u0f7a"      # rdo rje
CHINESE = "\u91d1\u525b"                                   # jin gang
MANIFEST_KEYS = {"package_version", "git_commit", "git_dirty", "python", "created_at",
                 "config_sha256", "prereg_sha256", "inputs"}


def test_read_yaml(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text(f"key: [1, {CHINESE}]\n", encoding="utf-8")
    assert read_yaml(p) == {"key": [1, CHINESE]}
    p.write_bytes(b"\xef\xbb\xbfkey: 1\n")
    assert read_yaml(p) == {"key": 1}
    p.write_text("", encoding="utf-8")
    assert read_yaml(p) is None


@pytest.mark.parametrize("bom", [False, True])
def test_csv_round_trip(tmp_path, bom):
    p = tmp_path / "sub" / "rows.csv"
    rows = [
        {"unit_id": "D417:8a.4.3", "text": TIBETAN, "note": 'comma, "quote"\nnew line'},
        {"unit_id": "T0892:0592a29.1", "text": CHINESE, "note": None},
        {"unit_id": "+T0892:0601c01.2"},
    ]
    write_csv(p, rows, ["unit_id", "text", "note"], bom=bom)
    assert p.read_bytes().startswith(b"\xef\xbb\xbf") is bom
    back = read_csv(p)
    assert list(back[0]) == ["unit_id", "text", "note"]          # the BOM never leaks into a header
    assert back[0] == {k: v for k, v in rows[0].items()}
    assert back[1]["note"] == "" and back[2] == {"unit_id": "+T0892:0601c01.2", "text": "", "note": ""}
    assert not list(p.parent.glob(".*.tmp"))                     # atomic write cleans up


def test_write_csv_rejects_unknown_columns(tmp_path):
    with pytest.raises(ValueError):
        write_csv(tmp_path / "x.csv", [{"a": 1, "b": 2}], ["a"])


def test_jsonl_round_trip_and_append(tmp_path):
    p = tmp_path / "log" / "records.jsonl"
    records = [{"id": "D417:8a.4.3", "text": TIBETAN}, {"id": "x", "n": 3, "flags": ["a"]}]
    write_jsonl(p, records)
    assert TIBETAN in p.read_text(encoding="utf-8")             # ensure_ascii=False: readable file
    assert read_jsonl(p) == records
    append_jsonl(p, {"id": "y"})
    append_jsonl(tmp_path / "new" / "ledger.jsonl", {"id": "z"})
    assert read_jsonl(p) == records + [{"id": "y"}]
    assert read_jsonl(tmp_path / "new" / "ledger.jsonl") == [{"id": "z"}]


def test_read_jsonl_skips_blank_lines_and_names_bad_lines(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert read_jsonl(p) == [{"a": 1}, {"a": 2}]
    p.write_text('{"a": 1}\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"r\.jsonl:2"):
        read_jsonl(p)


def test_sha256_file(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"abc" * 1_000_000)
    assert sha256_file(p) == hashlib.sha256(b"abc" * 1_000_000).hexdigest()


def test_git_helpers_outside_a_repository(tmp_path):
    assert git_commit(tmp_path) is None
    assert git_dirty(tmp_path) is None


def test_git_commit_of_this_checkout():
    commit = git_commit(REPO_ROOT)
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_manifest_keys_and_values(tmp_path):
    settings = load_settings(REPO_ROOT)
    source = tmp_path / "T18n0892.xml"
    source.write_text("<TEI/>", encoding="utf-8")
    path = write_manifest(tmp_path / "run", settings, {"cbeta": source},
                          {"instrument_digests": {"collate": "abc"}, "served_models": ["claude-opus-5-5"]})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert MANIFEST_KEYS <= set(manifest)
    assert manifest["package_version"] == hevajra_matrix.__version__
    assert manifest["inputs"]["cbeta"]["sha256"] == sha256_file(source)
    assert set(manifest["config_sha256"]) == {"run.yaml", "llm.yaml", "preregistration.yaml"}
    assert manifest["prereg_sha256"] == sha256_file(REPO_ROOT / "config" / "preregistration.yaml")
    assert manifest["instrument_digests"] == {"collate": "abc"}
    commit = git_commit(REPO_ROOT)
    assert manifest["git_commit"] == commit


def test_manifest_without_git_and_with_clashing_extra(tmp_path):
    settings = Settings(root=tmp_path, run={}, llm={}, prereg={}, shas={"run.yaml": "0" * 64})
    manifest = json.loads(write_manifest(tmp_path, settings, {}).read_text(encoding="utf-8"))
    assert manifest["git_commit"] is None and manifest["git_dirty"] is None
    assert manifest["prereg_sha256"] is None and manifest["inputs"] == {}
    with pytest.raises(ValueError, match="git_commit"):
        write_manifest(tmp_path, settings, {}, {"git_commit": "forged"})


def test_manifest_requires_existing_inputs(tmp_path):
    settings = Settings(root=tmp_path, run={}, llm={}, prereg={}, shas={})
    with pytest.raises(FileNotFoundError):
        write_manifest(tmp_path, settings, {"missing": tmp_path / "absent.txt"})
