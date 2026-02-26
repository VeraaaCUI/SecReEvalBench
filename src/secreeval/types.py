from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PromptItem:
    text: str
    label: str
    # prompt maliciousness in [0,1]
    ms: float
    # "malicious" or "not_malicious" source category
    category: str


@dataclass(frozen=True)
class Scenario:
    direction: str
    scenario_idx: int
    scenario_name: str
    judge_question: str
    # malicious scenario prompts (17 prompts: 1 benign + 16 adversarial)
    prompts: List[PromptItem]
    # paired benign scenario prompts (16 prompts, one per technique label)
    benign_prompts: Optional[List[PromptItem]] = None


@dataclass
class TurnResult:
    direction: str
    scenario_idx: int
    scenario_name: str
    sequence: str
    turn_idx: int
    prompt_label: str
    prompt_ms: float
    prompt_category: str
    response_text: Optional[str]
    # judge outputs
    mr_unsafe_prob: float
    mr_is_unsafe: bool
    refusal_prob: float
    is_refusal: bool
    logic_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "scenario_idx": self.scenario_idx,
            "scenario_name": self.scenario_name,
            "sequence": self.sequence,
            "turn_idx": self.turn_idx,
            "prompt_label": self.prompt_label,
            "prompt_ms": self.prompt_ms,
            "prompt_category": self.prompt_category,
            "response_text": self.response_text,
            "mr_unsafe_prob": self.mr_unsafe_prob,
            "mr_is_unsafe": self.mr_is_unsafe,
            "refusal_prob": self.refusal_prob,
            "is_refusal": self.is_refusal,
            "logic_score": self.logic_score,
        }
