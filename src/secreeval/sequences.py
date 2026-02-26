from __future__ import annotations

from typing import List

from .types import PromptItem, Scenario


SEQUENCES = {
    "one_off",
    "successive",
    "successive_reverse",
    "alternative",
    "sequential_ascending",
    "sequential_descending",
}


def build_sequence(scenario: Scenario, sequence: str) -> List[PromptItem]:
    if sequence not in SEQUENCES:
        raise ValueError(f"Unknown sequence: {sequence}")

    base = scenario.prompts

    if sequence == "one_off":
        # same order as dataset; runner will reset context every turn
        return list(base)

    if sequence == "successive":
        return list(base)

    if sequence == "successive_reverse":
        return list(base[1:]) + [base[0]]

    if sequence == "alternative":
        if not scenario.benign_prompts:
            raise ValueError("Scenario missing benign_prompts; cannot build alternative sequence")
        benign = scenario.benign_prompts
        malicious = base[1:]
        if len(benign) != len(malicious):
            raise ValueError(
                f"Alternative requires equal benign/malicious length (got {len(benign)} vs {len(malicious)})"
            )
        out: List[PromptItem] = []
        for b, m in zip(benign, malicious):
            out.append(b)
            out.append(m)
        return out

    if sequence in {"sequential_ascending", "sequential_descending"}:
        # Sort all prompts by Ms. Includes benign prompt (Ms=0).
        reverse = sequence == "sequential_descending"
        # stable sort: include original index as tiebreaker
        indexed = list(enumerate(base))
        indexed.sort(key=lambda x: (x[1].ms, x[0]), reverse=reverse)
        return [p for _, p in indexed]

    raise RuntimeError("Unhandled sequence")
