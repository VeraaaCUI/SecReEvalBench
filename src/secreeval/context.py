from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


Message = Dict[str, str]


@dataclass
class ConversationState:
    """Stores dialogue turns as (user, assistant) pairs."""

    turns: List[Tuple[str, str]] = field(default_factory=list)

    def reset(self) -> None:
        self.turns.clear()

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append((user, assistant))

    def build_messages(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
        history_window: int,
    ) -> List[Message]:
        """Build chat messages with a turn-based sliding window.

        history_window:
          -1 => full history
           0 => no history
          N>0 => last N turns
        """
        msgs: List[Message] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})

        if history_window == 0 or not self.turns:
            msgs.append({"role": "user", "content": user_prompt})
            return msgs

        selected = self.turns if history_window < 0 else self.turns[-history_window:]
        for u, a in selected:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})

        msgs.append({"role": "user", "content": user_prompt})
        return msgs
