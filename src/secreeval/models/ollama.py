from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

from .base import ChatModel, Message


@dataclass
class OllamaChatModel(ChatModel):
    model: str
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    top_p: float = 0.9
    num_predict: int = 512
    timeout: int = 600

    def generate(self, messages: List[Message], *, max_new_tokens: int = 512) -> str:
        url = f"{self.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": min(max_new_tokens, self.num_predict),
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        # Ollama returns {message: {role, content}}
        content = data.get("message", {}).get("content")
        if content is None:
            raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)[:500]}")
        return str(content)
