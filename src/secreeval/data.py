from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .types import PromptItem, Scenario


def load_seceval_json(path: str | Path) -> Dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _level_to_ms(level: object) -> Optional[float]:
    """Map dataset `level` -> Ms in [0, 1].

    The level-annotated dataset schema (SecEvalDatabase-with_levels.JSON) uses integer levels:

      - 1 -> 1.00 (obvious malicious)
      - 2 -> 0.66 (medium / mixed malicious)
      - 3 -> 0.33 (subtle malicious)
      - 4 -> 0.00 (benign)

    We accept level as str/int/float. Returns None if missing/unknown.
    """

    if level is None:
        return None

    # Common case: "1"/"2"/"3"/"4"
    try:
        lv_int = int(str(level).strip())
    except Exception:
        lv_int = None

    if lv_int == 1:
        return 1.0
    if lv_int == 2:
        return 0.66
    if lv_int == 3:
        return 0.33
    if lv_int == 4:
        return 0.0

    # Fallback: if the dataset stores Ms directly as a float string (e.g., "0.66")
    try:
        v = float(str(level).strip())
        if 0.0 <= v <= 1.0:
            return v
    except Exception:
        pass

    return None


def _promptitem_from_raw(p: Dict, *, default_ms: float, default_category: str) -> PromptItem:
    """Convert one raw prompt dict into a PromptItem.

    - If `level` exists, we compute Ms via `_level_to_ms`.
    - Else we fall back to `default_ms` (legacy behavior).

    We normalize `category` per-prompt to "malicious" vs "not_malicious" based on Ms,
    because the level-annotated file can contain benign prompts under the "malicious" dataset bucket
    (the first prompt of each malicious scenario).
    """

    text = p.get("user_input", "")
    label = p.get("label", "")

    ms = _level_to_ms(p.get("level"))
    if ms is None:
        ms = float(default_ms)

    category = "not_malicious" if ms <= 0 else "malicious"
    if default_category in {"malicious", "not_malicious"}:
        category = "not_malicious" if ms <= 0 else "malicious"

    return PromptItem(text=text, label=label, ms=ms, category=category)


def build_scenarios(
    data: Dict,
    *,
    ms_mode: str = "fixed",
) -> List[Scenario]:
    """Build scenarios from SecEval JSON.

    Supported dataset schemas
    -------------------------

    1) Legacy schema (SecEvalDatabase.JSON)
       - categories: ["malicious", "not_malicious"]
       - malicious scenarios contain 17 prompts (first prompt benign)
       - not_malicious scenarios contain 16 benign prompts

    2) Level-annotated schema (SecEvalDatabase-with_levels.JSON)
       - same structure as legacy, but each prompt has a `level` field in {1,2,3,4}
       - level mapping: 1->1.00, 2->0.66, 3->0.33, 4->0.00 (benign)

    `ms_mode` (kept for backward-compatibility)
    ------------------------------------------

    - "fixed":
        - if `level` exists, use the mapping above
        - else fall back to legacy behavior: first prompt Ms=0, others Ms=1 in malicious scenarios

      You can later override Ms using a prompt-judge via `--prompt-ms-mode guard`.
    """
    if ms_mode not in {"fixed"}:
        raise ValueError(f"Unsupported ms_mode: {ms_mode}")

    cats = {c.get("category_name"): c for c in data.get("categories", []) if c.get("category_name")}
    if "malicious" not in cats:
        raise ValueError("Expected a 'malicious' category in dataset JSON")

    mal = cats["malicious"].get("security_directions", [])

    # Benign prompts are optional; they are required only for the 'alternative' sequence.
    benign_cat = None
    for k in ("not_malicious", "benign", "safe"):
        if k in cats:
            benign_cat = cats[k]
            break

    ben_by_dir: Dict[str, List[Dict]] = {}
    if benign_cat is not None:
        ben_dirs = benign_cat.get("security_directions", [])
        ben_by_dir = {d.get("direction_name"): d.get("scenarios", []) for d in ben_dirs if d.get("direction_name")}

    scenarios: List[Scenario] = []
    for d in mal:
        direction = d["direction_name"]
        mal_scenarios = d["scenarios"]
        paired_ben = ben_by_dir.get(direction) if ben_by_dir else None
        if ben_by_dir and paired_ben is None:
            raise ValueError(f"Missing benign direction in dataset: {direction}")

        for idx, sc in enumerate(mal_scenarios):
            prompts: List[PromptItem] = []
            for p_i, p in enumerate(sc["prompts"]):
                # If the dataset provides `level`, use it; else fallback to legacy fixed Ms.
                default_ms = 0.0 if p_i == 0 else 1.0
                prompts.append(_promptitem_from_raw(p, default_ms=default_ms, default_category="malicious"))

            benign_prompts: Optional[List[PromptItem]] = None
            if paired_ben is not None:
                # Pair with benign scenario (aligned by scenario index)
                ben_sc = paired_ben[idx]
                benign_prompts = []
                for p in ben_sc.get("prompts", []):
                    benign_prompts.append(_promptitem_from_raw(p, default_ms=0.0, default_category="not_malicious"))

            scenarios.append(
                Scenario(
                    direction=direction,
                    scenario_idx=idx,
                    scenario_name=sc.get("scenario_name", ""),
                    judge_question=sc.get("judge_question", ""),
                    prompts=prompts,
                    benign_prompts=benign_prompts,
                )
            )

    return scenarios
