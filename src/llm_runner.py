"""
Local LLM runner via Ollama HTTP API.
Phase 1: basic prompt -> response pipeline.
"""
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional


OLLAMA_BASE = "http://localhost:11434"

MUTATION_MODEL = "llama3.2:3b"          # fast, high-volume mutation generation
JUDGE_MODEL    = "llama3.1:8b-instruct-q4_K_M"  # quality eval on held-out set


@dataclass
class ModelResponse:
    model: str
    prompt: str
    response: str
    done: bool
    eval_duration_ms: float = 0.0
    raw: dict = field(default_factory=dict)


def _get(endpoint: str) -> dict:
    url = f"{OLLAMA_BASE}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(f"Ollama not reachable at {OLLAMA_BASE}. Is it running? ({e})") from e


def _post(endpoint: str, payload: dict) -> dict:
    url = f"{OLLAMA_BASE}{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ConnectionError(f"Ollama not reachable at {OLLAMA_BASE}. Is it running? ({e})") from e


def list_models() -> list[str]:
    result = _get("/api/tags")
    return [m["name"] for m in result.get("models", [])]


def generate(
    prompt: str,
    model: str = "llama3.1:8b-instruct-q4_K_M",
    system: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> ModelResponse:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    if system:
        payload["system"] = system

    raw = _post("/api/generate", payload)
    return ModelResponse(
        model=model,
        prompt=prompt,
        response=raw.get("response", ""),
        done=raw.get("done", False),
        eval_duration_ms=raw.get("eval_duration", 0) / 1e6,
        raw=raw,
    )


def chat(
    messages: list[dict],
    model: str = "llama3.1:8b-instruct-q4_K_M",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> ModelResponse:
    """messages: list of {"role": "user"|"assistant"|"system", "content": "..."}"""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    raw = _post("/api/chat", payload)
    content = raw.get("message", {}).get("content", "")
    prompt_echo = next((m["content"] for m in messages if m["role"] == "user"), "")
    return ModelResponse(
        model=model,
        prompt=prompt_echo,
        response=content,
        done=raw.get("done", False),
        eval_duration_ms=raw.get("eval_duration", 0) / 1e6,
        raw=raw,
    )
