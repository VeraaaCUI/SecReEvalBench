from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


Message = Dict[str, str]


class ChatModel(ABC):
    @abstractmethod
    def generate(self, messages: List[Message], *, max_new_tokens: int = 512) -> str:
        raise NotImplementedError


class JudgeModel(ABC):
    """A judge model returns a score (usually a probability) and a binary decision."""

    @abstractmethod
    def score(self, text: str, prompt: Optional[str] = None) -> tuple[float, bool]:
        raise NotImplementedError
