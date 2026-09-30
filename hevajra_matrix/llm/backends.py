"""Text-generation backends.

Three interchangeable implementations behind one protocol:

* ``OpenAICompatibleBackend`` – talks to a local vLLM / Ollama / llama.cpp server
  (``/v1/chat/completions``). This is the recommended way to run Qwen3-32B-AWQ or
  Gemma-3-27B on a single RTX 5090: the server owns the GPU, the pipeline stays
  dependency-free.
* ``TransformersBackend`` – in-process HuggingFace ``transformers`` loading (bf16 or
  4-bit), for notebooks and small models (Qwen3-8B, Gemma-3-12B).
* ``MockBackend`` – deterministic canned responses; used by the tests and by
  ``--dry-run`` so that every LLM task can be exercised without a GPU.

All backends expose ``generate(prompt, system=None, json_schema=None) -> str`` and
record every call (prompt hash, model id, parameters) so that model outputs are
as auditable as human annotations.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import yaml


class LLMBackend(Protocol):
    model_id: str

    def generate(self, prompt: str, system: str | None = None, json_schema: dict | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str: ...


@dataclass
class CallRecord:
    model_id: str
    prompt_sha256: str
    system_sha256: str | None
    temperature: float
    max_tokens: int
    seconds: float
    n_prompt_chars: int
    n_response_chars: int
    response_sha256: str


@dataclass
class CallLog:
    records: list[CallRecord] = field(default_factory=list)

    def add(self, model_id: str, prompt: str, system: str | None, response: str, temperature: float,
            max_tokens: int, seconds: float) -> None:
        self.records.append(CallRecord(
            model_id=model_id, prompt_sha256=_sha(prompt), system_sha256=_sha(system) if system else None,
            temperature=temperature, max_tokens=max_tokens, seconds=round(seconds, 3),
            n_prompt_chars=len(prompt), n_response_chars=len(response), response_sha256=_sha(response),
        ))

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps([r.__dict__ for r in self.records], ensure_ascii=False, indent=1), encoding="utf-8")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- OpenAI-compatible (vLLM / Ollama / llama.cpp)
class OpenAICompatibleBackend:
    def __init__(self, model_id: str, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
                 timeout: float = 600.0, seed: int | None = 20260930, log: CallLog | None = None) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.seed = seed
        self.log = log or CallLog()

    def generate(self, prompt: str, system: str | None = None, json_schema: dict | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict[str, Any] = {"model": self.model_id, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature}
        if self.seed is not None:
            body["seed"] = self.seed
        if json_schema is not None:
            # vLLM: guided_json ; OpenAI-style: response_format json_schema. Send both, servers ignore unknown keys.
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": json_schema}}
            body["guided_json"] = json_schema
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"] or ""
        self.log.add(self.model_id, prompt, system, text, temperature, max_tokens, time.time() - t0)
        return text


# --------------------------------------------------------------------------- in-process transformers
class TransformersBackend:
    """Lazy in-process loader. Requires ``torch`` + ``transformers`` (and ``bitsandbytes`` for 4-bit)."""

    def __init__(self, model_id: str, load_in_4bit: bool = False, dtype: str = "bfloat16", device_map: str = "auto",
                 log: CallLog | None = None) -> None:
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.dtype = dtype
        self.device_map = device_map
        self.log = log or CallLog()
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        kwargs: dict[str, Any] = {"device_map": self.device_map}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                               bnb_4bit_quant_type="nf4")
        else:
            kwargs["torch_dtype"] = getattr(torch, self.dtype)
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def generate(self, prompt: str, system: str | None = None, json_schema: dict | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        self._load()
        import torch  # type: ignore

        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        if json_schema is not None:
            messages[-1]["content"] += "\n\nReturn only JSON conforming to this schema:\n" + json.dumps(json_schema, ensure_ascii=False)
        ids = self._tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self._model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                       temperature=max(temperature, 1e-5), pad_token_id=self._tok.eos_token_id)
        text = self._tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        self.log.add(self.model_id, prompt, system, text, temperature, max_tokens, time.time() - t0)
        return text


# --------------------------------------------------------------------------- mock
class MockBackend:
    """Deterministic backend for tests / dry runs.

    ``responder`` maps a prompt to a response; default echoes a minimal JSON object
    or a fixed sentence so that parsers have something to chew on.
    """

    def __init__(self, responder: Callable[[str, str | None], str] | None = None, model_id: str = "mock") -> None:
        self.model_id = model_id
        self.responder = responder
        self.log = CallLog()
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, prompt: str, system: str | None = None, json_schema: dict | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        self.calls.append((prompt, system))
        if self.responder is not None:
            text = self.responder(prompt, system)
        elif json_schema is not None:
            text = json.dumps({k: ([] if v.get("type") == "array" else "" if v.get("type") == "string" else None)
                               for k, v in json_schema.get("properties", {}).items()}, ensure_ascii=False)
        else:
            text = "mock response"
        self.log.add(self.model_id, prompt, system, text, temperature, max_tokens, 0.0)
        return text


# --------------------------------------------------------------------------- config
def load_backend(name: str, config_path: Path | None = None) -> LLMBackend:
    """Instantiate a backend from ``data/llm/models.yaml`` by profile name (or ``mock``)."""
    if name == "mock":
        return MockBackend()
    config_path = config_path or Path(__file__).resolve().parent.parent.parent / "data" / "llm" / "models.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profiles = {p["name"]: p for p in cfg["generation"]}
    if name not in profiles:
        raise KeyError(f"unknown LLM profile {name!r}; known: {sorted(profiles)}")
    p = profiles[name]
    if p.get("serve") == "transformers":
        return TransformersBackend(p["model_id"], load_in_4bit=p.get("quant") == "bnb-4bit", dtype=p.get("dtype", "bfloat16"))
    return OpenAICompatibleBackend(p.get("served_name", p["model_id"]), base_url=p.get("base_url", "http://localhost:8000/v1"))
