from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base import ChatModel, Message


@dataclass
class TransformersChatModel(ChatModel):
    model_name_or_path: str
    device: str | None = None
    dtype: str | None = "auto"
    trust_remote_code: bool = False
    temperature: float = 0.0
    top_p: float = 0.9

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Transformers backend requires optional deps. Install with: pip install -e '.[transformers]'"
            ) from e

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            device_map="auto" if self.device is None else None,
            trust_remote_code=self.trust_remote_code,
        )
        if self.device is not None:
            import torch

            self.model.to(torch.device(self.device))
        self.model.eval()

    def generate(self, messages: List[Message], *, max_new_tokens: int = 512) -> str:
        import torch

        # Prefer chat template if available.
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: plain concatenation
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role.upper()}: {content}")
            parts.append("ASSISTANT:")
            prompt = "\n".join(parts)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "top_p": self.top_p,
        }
        # Remove None keys
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # Strip prompt
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()
