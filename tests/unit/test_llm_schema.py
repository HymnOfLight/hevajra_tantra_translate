"""llm.schema: strict(), for_api() and the dependency-free validator."""

from __future__ import annotations

import copy

import pytest

from hevajra_matrix.llm.schema import for_api, strict, validate

# The T1 collator schema of the design (section 3.1), written without the strict
# boilerplate that strict() must add.
RELATIONS = [
    "equivalent",
    "paraphrase",
    "generalised",
    "abridged",
    "expanded",
    "substitution",
    "reversal",
    "category_name_omitted",
    "transliterated",
    "no_counterpart",
]
T1_LOOSE = {
    "type": "object",
    "properties": {
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "wit": {"type": "array", "items": {"type": "string"}},
                    "relation": {"type": "string", "enum": RELATIONS},
                    "polarity_flip": {"type": "boolean"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "ref_quote": {"type": "string"},
                    "wit_quote": {"type": "string"},
                },
            },
        },
        "witness_only": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "wit": {"type": "array", "items": {"type": "string"}},
                    "kind": {
                        "type": "string",
                        "enum": ["addition", "translator_note", "paratext", "belongs_elsewhere"],
                    },
                    "wit_quote": {"type": "string"},
                },
            },
        },
    },
}
T1 = strict(T1_LOOSE)

UNIT = {
    "ref": "r001",
    "wit": ["z0001"],
    "relation": "equivalent",
    "polarity_flip": False,
    "confidence": "high",
    "ref_quote": "",
    "wit_quote": "q",
}


def _objects(schema):
    if schema.get("type") == "object":
        yield schema
    for sub in schema.get("properties", {}).values():
        yield from _objects(sub)
    if "items" in schema:
        yield from _objects(schema["items"])


# ------------------------------------------------------------------------------ strict


def test_strict_closes_every_object_and_requires_every_property_in_order():
    objects = list(_objects(T1))
    assert len(objects) == 3
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])
    assert T1["properties"]["units"]["items"]["required"][:3] == ["ref", "wit", "relation"]


def test_strict_overrides_partial_required_and_open_objects():
    out = strict(
        {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": True,
        }
    )
    assert out["required"] == ["a", "b"] and out["additionalProperties"] is False


def test_strict_handles_empty_object_and_is_idempotent_and_pure():
    assert strict({"type": "object"}) == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
        "required": [],
    }
    before = copy.deepcopy(T1_LOOSE)
    assert strict(T1) == T1
    strict(T1_LOOSE)
    assert before == T1_LOOSE


# ----------------------------------------------------------------------------- for_api


def test_for_api_moves_unsupported_constraints_into_description():
    schema = strict(
        {
            "type": "object",
            "properties": {
                "quote": {"type": "string", "maxLength": 200, "description": "verbatim"},
                "tags": {"type": "array", "items": {"type": "string", "maxLength": 5}, "minItems": 2},
                "wit": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            },
        }
    )
    out = for_api(schema)
    quote = out["properties"]["quote"]
    assert "maxLength" not in quote and quote["description"] == "verbatim\n\n{maxLength: 200}"
    tags = out["properties"]["tags"]
    assert "minItems" not in tags and tags["description"] == "{minItems: 2}"
    assert tags["items"] == {"type": "string", "description": "{maxLength: 5}"}
    assert out["properties"]["wit"]["minItems"] == 1  # 0 and 1 are supported
    assert out["additionalProperties"] is False and out["required"] == ["quote", "tags", "wit"]
    assert schema["properties"]["quote"]["maxLength"] == 200  # input untouched


def test_for_api_keeps_a_supported_schema_unchanged():
    assert for_api(T1) == T1


# ---------------------------------------------------------------------------- validate


def test_valid_t1_answer_has_no_errors():
    assert validate({"units": [UNIT], "witness_only": []}, T1) == []


def test_errors_carry_a_json_path():
    bad_unit = {**UNIT, "relation": "motive", "wit": ["z1", 2]}
    del bad_unit["confidence"]
    errors = validate({"units": [UNIT, bad_unit], "witness_only": [], "extra": 1}, T1)
    assert "$: unexpected property 'extra'" in errors
    assert "$.units[1]: missing required property 'confidence'" in errors
    assert any(e.startswith("$.units[1].relation: 'motive' is not one of") for e in errors)
    assert "$.units[1].wit[1]: expected string, got integer" in errors
    assert len(errors) == 4


@pytest.mark.parametrize(
    ("schema", "value", "ok"),
    [
        ({"type": "integer"}, 3, True),
        ({"type": "integer"}, 3.0, True),
        ({"type": "integer"}, 3.5, False),
        ({"type": "integer"}, True, False),
        ({"type": "number"}, 2.5, True),
        ({"type": "number"}, False, False),
        ({"type": "boolean"}, 0, False),
        ({"type": "boolean"}, True, True),
        ({"type": "string"}, None, False),
        ({"type": "null"}, None, True),
        ({"type": ["string", "null"]}, None, True),
        ({"type": "array"}, (), False),
        ({"type": "object"}, [], False),
        ({"enum": [1, "a"]}, True, False),  # JSON true is not 1
        ({"enum": [True]}, 1, False),
        ({"enum": [1, "a"]}, 1.0, True),
        ({"type": "string", "maxLength": 3}, "abc", True),
        ({"type": "string", "maxLength": 3}, "abcd", False),
        ({"type": "array", "minItems": 1}, [], False),
        ({"type": "array", "minItems": 1}, [0], True),
    ],
)
def test_keyword_rules(schema, value, ok):
    assert (validate(value, schema) == []) is ok


def test_max_length_counts_code_points():
    # IAST "sunya" with precomposed diacritics: five code points, not seven UTF-8 bytes.
    assert validate("\u015b\u016bnya", {"type": "string", "maxLength": 5}) == []
    assert validate("\u015b\u016bnya", {"type": "string", "maxLength": 4}) != []


def test_type_mismatch_stops_descent():
    assert validate("x", T1) == ["$: expected object, got string"]


def test_annotations_are_allowed():
    assert validate("x", {"type": "string", "description": "d", "title": "t"}) == []


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "pattern": "^a"},
        {"type": "object", "properties": {"never_present": {"type": "integer", "minimum": 0}}},
        {"type": "object", "additionalProperties": {"type": "string"}},
        {"type": "date"},
        {"anyOf": [{"type": "string"}]},
    ],
)
def test_unsupported_schema_features_fail_loudly(schema):
    with pytest.raises(ValueError):
        validate({}, schema)
