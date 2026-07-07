from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationResult:
    text: str
    started_at: float
    ended_at: float
    input_token_count: int
    output_token_count: int

    @property
    def duration_sec(self) -> float:
        return self.ended_at - self.started_at


class LocalQwenGenerator:
    def __init__(
        self,
        *,
        model_name: str,
        model_path: str,
        device: str,
        max_input_tokens: int,
        max_new_tokens: int,
        system_prompt: str = "",
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.device = torch.device(device)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.tokenizer: Any
        self.model: Any
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> GenerationResult:
        input_text = self.format_prompt(prompt)
        encoded = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        input_token_count = int(encoded["input_ids"].shape[1])
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        started_at = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        ended_at = time.perf_counter()
        output_ids = generated[0, input_token_count:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            started_at=started_at,
            ended_at=ended_at,
            input_token_count=input_token_count,
            output_token_count=int(output_ids.shape[0]),
        )

    def format_prompt(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        method = self.tokenizer.apply_chat_template
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "enable_thinking" in inspect.signature(method).parameters:
            kwargs["enable_thinking"] = False
        formatted = method(messages, **kwargs)
        assert isinstance(formatted, str), type(formatted)
        return formatted
