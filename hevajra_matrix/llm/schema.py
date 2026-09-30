"""Minimal JSON-schema helpers for Claude structured outputs (no jsonschema dependency).

Three pure functions over plain ``dict`` schemas:

``strict(schema)``
    Closes every object schema (``additionalProperties: false``) and makes every
    property required, which structured outputs demand. Request builders apply it
    once; the result is what ``LLMRequest.schema`` holds and what the cache key covers.
``for_api(schema)``
    The copy actually sent to the API. Structured outputs reject some validation
    keywords (``maxLength``, ``minItems`` above 1, ...); they are moved into the
    node's ``description`` so the model still sees them, mirroring the SDK's own
    ``transform_schema``. The full schema stays authoritative for ``validate``.
``validate(instance, schema)``
    Checks a parsed answer against the full schema and returns human-readable error
    messages (empty list = valid). This is what enforces the keywords the API cannot.

Only the subset used by this project's task schemas is implemented: ``type``
(object, array, string, integer, number, boolean, null), ``properties``,
``required``, ``additionalProperties: false``, ``items``, ``enum``, ``maxLength`` and
``minItems``, plus the annotations ``description`` and ``title``. Any other keyword
raises ``ValueError`` so that a constraint can never be silently ignored.
"""

from __future__ import annotations

import copy
from typing import Any, Iterator, Mapping

_ANNOTATIONS = frozenset({"description", "title"})
_ASSERTIONS = frozenset(
    {"type", "properties", "required", "additionalProperties", "items", "enum", "maxLength", "minItems"}
)
# Keywords the structured-outputs API accepts as-is; minItems only with value 0 or 1.
_API_KEYWORDS = frozenset(
    {"type", "properties", "required", "additionalProperties", "items", "enum", "description", "title"}
)

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: (
        (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer())
    ),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def strict(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy in which every object schema is closed and fully required.

    Idempotent. ``required`` lists the properties in their declared order.
    """
    out = copy.deepcopy(dict(schema))
    for node in _nodes(out):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("properties", {})
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
    return out


def for_api(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the copy of ``schema`` to send as ``output_config.format.schema``.

    Keywords outside the API-supported set are removed and appended to the node's
    ``description`` as ``{keyword: value, ...}`` (sorted), the format the SDK's own
    ``transform_schema`` uses, so the model still learns limits the API cannot enforce.
    """
    node: dict[str, Any] = {}
    moved: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _API_KEYWORDS or (key == "minItems" and value in (0, 1)):
            node[key] = copy.deepcopy(value)
        else:
            moved[key] = value
    if isinstance(node.get("properties"), Mapping):
        node["properties"] = {name: for_api(sub) for name, sub in node["properties"].items()}
    if isinstance(node.get("items"), Mapping):
        node["items"] = for_api(node["items"])
    if moved:
        note = "{" + ", ".join(f"{k}: {moved[k]}" for k in sorted(moved)) + "}"
        node["description"] = f"{node['description']}\n\n{note}" if node.get("description") else note
    return node


def validate(instance: Any, schema: Mapping[str, Any]) -> list[str]:
    """Return the ways ``instance`` violates ``schema`` (empty list when valid).

    Messages are prefixed with a JSONPath-like location such as ``$.units[3].relation``.
    Raises ``ValueError`` if the schema uses a keyword outside the supported subset.
    """
    for node in _nodes(schema):
        unknown = sorted(set(node) - _ASSERTIONS - _ANNOTATIONS)
        unknown += [f"type {t!r}" for t in _type_names(node) if t not in _TYPE_CHECKS]
        if unknown:
            raise ValueError(f"unsupported JSON-schema feature(s): {unknown}")
        if node.get("additionalProperties") not in (None, True, False):
            raise ValueError("additionalProperties must be a boolean")
    errors: list[str] = []
    _check(instance, schema, "$", errors)
    return errors


def _type_names(schema: Mapping[str, Any]) -> list[str]:
    declared = schema.get("type")
    if declared is None:
        return []
    return [declared] if isinstance(declared, str) else list(declared)


def _nodes(schema: Mapping[str, Any]) -> Iterator[Any]:
    """Yield ``schema`` and every nested schema reachable through properties/items."""
    yield schema
    for sub in (schema.get("properties") or {}).values():
        yield from _nodes(sub)
    if isinstance(schema.get("items"), Mapping):
        yield from _nodes(schema["items"])


def _check(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    names = _type_names(schema)
    if names and not any(_TYPE_CHECKS[n](value) for n in names):
        errors.append(f"{path}: expected {' or '.join(names)}, got {_json_type(value)}")
        return
    if "enum" in schema and not any(_same(value, option) for option in schema["enum"]):
        errors.append(f"{path}: {_show(value)} is not one of {list(schema['enum'])}")
    if isinstance(value, str) and "maxLength" in schema and len(value) > schema["maxLength"]:
        errors.append(f"{path}: string of length {len(value)} exceeds maxLength {schema['maxLength']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} item(s), fewer than minItems {schema['minItems']}")
        if "items" in schema:
            for i, item in enumerate(value):
                _check(item, schema["items"], f"{path}[{i}]", errors)
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required", ()):
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        if schema.get("additionalProperties") is False:
            for name in sorted(set(value) - set(properties)):
                errors.append(f"{path}: unexpected property {name!r}")
        for name, sub in properties.items():
            if name in value:
                _check(value[name], sub, f"{path}.{name}", errors)


def _same(a: Any, b: Any) -> bool:
    """JSON equality: ``true`` is not ``1`` (Python says it is)."""
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _json_type(value: Any) -> str:
    for name in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _TYPE_CHECKS[name](value):
            return name
    return type(value).__name__


def _show(value: Any, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
