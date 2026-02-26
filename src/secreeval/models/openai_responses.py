from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from .base import ChatModel, Message


class _OpenAINonRetryableError(RuntimeError):
    """Raised for OpenAI client-side errors where retrying will not help."""


def _is_reasoning_family_model(model: str) -> bool:
    """Heuristic: GPT-5 family and o-series are reasoning models.

    GPT-5* models have special parameter compatibility (e.g., sampling params are
    not accepted in many cases). o-series models also frequently disallow
    sampling parameters.
    """

    m = (model or "").strip().lower()
    if not m:
        return False
    if m.startswith("gpt-5"):
        return True
    # o-series: o1, o3, o4-mini, etc.
    if m.startswith("o"):
        return True
    return False


def _allow_sampling_params(model: str) -> bool:
    """Whether it is safe to include `temperature` / `top_p` in the request.

    Per OpenAI docs, older GPT-5 family models (e.g. gpt-5-mini) error if these
    fields are included. For compatibility we omit sampling params for GPT-5 and
    o-series models by default.
    """

    return not _is_reasoning_family_model(model)


def _extract_output_text(resp: Dict[str, Any]) -> str:
    """Extract assistant text from a Responses API payload.

    The Responses API may return multiple output items (messages, tool calls, etc.).
    We aggregate all assistant message content pieces of type `output_text`.
    """

    # Newer Responses payloads often include a convenience field `output_text`.
    # Prefer it when present.
    if isinstance(resp.get("output_text"), str) and resp["output_text"].strip():
        return resp["output_text"].strip()

    out = resp.get("output")
    if isinstance(out, list):
        chunks: List[str] = []
        for item in out:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            if item.get("role") != "assistant":
                continue
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                # The API uses `output_text` for assistant text chunks.
                if c.get("type") in {"output_text", "text"} and isinstance(c.get("text"), str):
                    chunks.append(c["text"])
        txt = "".join(chunks).strip()
        if txt:
            return txt

    raise RuntimeError(f"Could not extract text from OpenAI response: {json.dumps(resp)[:500]}")


def _is_usable_incomplete(resp: Dict[str, Any]) -> bool:
    """Whether an `incomplete` Responses API payload is still usable.

    For offline evaluation, we can continue if the model produced partial text.
    The most common cause is hitting `max_output_tokens`.
    """

    if resp.get("status") != "incomplete":
        return False
    details = resp.get("incomplete_details")
    reason = None
    if isinstance(details, dict):
        reason = details.get("reason")
    # Official reason from OpenAI API
    if isinstance(reason, str) and reason.startswith("max_output"):
        return True
    # Common gateway variants
    if reason in {"max_tokens", "length"}:
        return True
    return False


@dataclass
class OpenAIResponsesChatModel(ChatModel):
    """ChatModel backed by the OpenAI Responses API (stateless by default)."""

    model: str
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 600
    temperature: float = 0.0
    top_p: float = 0.9
    max_retries: int = 6

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide --openai-api-key or set $env:OPENAI_API_KEY in PowerShell."
            )

    def generate(self, messages: List[Message], *, max_new_tokens: int = 512) -> str:
        """Generate a response.

        We translate the internal message format:
          - system -> instructions (highest priority)
          - user/assistant -> input array items
        """

        instructions: Optional[str] = None
        input_items: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if not isinstance(content, str):
                content = str(content)
            if role == "system":
                # Use `instructions` to provide high-level guidance.
                instructions = content
            elif role in {"user", "assistant", "developer"}:
                input_items.append({"role": role, "content": content})
            else:
                # Unknown roles are treated as user content.
                input_items.append({"role": "user", "content": content})

        url = f"{self.base_url.rstrip('/')}/responses"
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": int(max_new_tokens),
        }
        # IMPORTANT: GPT-5 / o-series (reasoning family) models have stricter
        # parameter compatibility. In particular, older GPT-5 models such as
        # `gpt-5-mini` reject requests that include sampling parameters like
        # temperature/top_p.
        if _allow_sampling_params(self.model):
            payload["temperature"] = float(self.temperature)
            payload["top_p"] = float(self.top_p)
        if instructions:
            payload["instructions"] = instructions

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Simple retry with exponential backoff for transient errors.
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                # Non-retryable client errors (e.g., invalid params) should fail fast.
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    raise _OpenAINonRetryableError(
                        f"OpenAI HTTP {r.status_code}: {r.text[:800]}"
                    )
                if r.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"OpenAI transient HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"OpenAI response is not a JSON object: {str(data)[:200]}")

                status = data.get("status")
                if status == "completed":
                    return _extract_output_text(data)

                # The Responses API may return `incomplete` when the model hit max_output_tokens.
                # In offline evaluation, partial output is still useful (and avoids slow retries).
                if _is_usable_incomplete(data):
                    try:
                        return _extract_output_text(data)
                    except Exception:  # noqa: BLE001
                        # Rare but documented: the model may consume its budget on reasoning
                        # and return no visible text. For benchmark continuity, treat it as
                        # a generic refusal.
                        return "I'm sorry, I can't help with that."

                # Some incomplete cases are effectively refusals (e.g., content filtering).
                if status == "incomplete":
                    details = data.get("incomplete_details")
                    reason = details.get("reason") if isinstance(details, dict) else None
                    if reason in {"content_filter", "safety"}:
                        return "I'm sorry, I can't help with that."

                raise RuntimeError(f"OpenAI response status={status}: {json.dumps(data)[:300]}")
            except _OpenAINonRetryableError as e:
                # Do not retry – the request is invalid (e.g., unsupported params).
                raise RuntimeError(f"OpenAI request failed: {e}") from e
            except Exception as e:  # noqa: BLE001
                last_err = e
                sleep_s = min(30.0, (2.0 ** attempt) + random.random())
                time.sleep(sleep_s)
                continue
        raise RuntimeError(f"OpenAI request failed after retries: {last_err}")
