from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

from transformers import AutoTokenizer

from workflow.replica import PromptEncoding
from workflow.schema import ExecutionConfig


class PromptTokenizerBackend(Protocol):
    truncation_side: str

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        /,
        **kwargs: object,
    ) -> object: ...

    def encode(self, prompt: str, /, **kwargs: object) -> list[int]: ...

    def decode(self, token_ids: object, /, **kwargs: object) -> object: ...


PromptTokenizerFactory = Callable[[ExecutionConfig], PromptTokenizerBackend]


class PromptTokenizer:
    def __init__(
        self,
        execution: ExecutionConfig,
        *,
        tokenizer_factory: PromptTokenizerFactory | None = None,
    ) -> None:
        factory = tokenizer_factory or _load_prompt_tokenizer
        self._execution = execution
        self._tokenizer = factory(execution)
        if execution.truncation_side is not None:
            self._tokenizer.truncation_side = execution.truncation_side

    def build_prompt(
        self,
        user_prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> PromptEncoding:
        if self._execution.use_chat_template:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._execution.enable_thinking,
            )
            if not isinstance(rendered, str):
                raise TypeError("chat template must render a string")
            prompt = rendered
        elif system_prompt is None:
            prompt = user_prompt
        else:
            prompt = f"{system_prompt}\n\n{user_prompt}"

        token_ids = self._tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )
        return PromptEncoding(text=prompt, token_ids=tuple(token_ids))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        output = self._tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
        )
        if not isinstance(output, str):
            raise TypeError("tokenizer decode must return a string")
        return output


def _load_prompt_tokenizer(execution: ExecutionConfig) -> PromptTokenizerBackend:
    tokenizer = AutoTokenizer.from_pretrained(
        execution.model_path,
        local_files_only=True,
    )
    if tokenizer is None:
        raise RuntimeError("AutoTokenizer returned no tokenizer")
    return cast(PromptTokenizerBackend, tokenizer)
