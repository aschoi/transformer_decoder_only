from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from tokenizers import Tokenizer

EOS_TOKEN = "<|endoftext|>"
END_TOKEN = "<|end|>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

SUPPORTED_ROLES = ("system", "user", "assistant")

@dataclass(frozen=True, slots=True)
class ChatSpecialTokens:

    eos: int
    end: int
    system: int
    user: int
    assistant: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Tokenizer) -> "ChatSpecialTokens":
        resolved: dict[str, int] = {}

        for field, token in (
            ("eos", EOS_TOKEN),
            ("end", END_TOKEN),
            ("system", SYSTEM_TOKEN),
            ("user", USER_TOKEN),
            ("assistant", ASSISTANT_TOKEN),
        ):
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"Tokenizer is missing required chat token: {token}")
            resolved[field] = token_id

        return cls(**resolved)

    def role_token_id(self, role: str) -> int:
        if role == "system":
            return self.system
        if role == "user":
            return self.user
        if role == "assistant":
            return self.assistant
        raise ValueError(f"Unsupported role: {role!r}. Expected one of {SUPPORTED_ROLES}")

    @property
    def stop_ids(self) -> frozenset[int]:

        return frozenset({self.end, self.eos})


def _normalize_message(message: Any) -> tuple[str, str] | None:
    """
    Uses same template rules as tokenize_sft_corpus
    """

    if not isinstance(message, Mapping):
        return None

    role = str(message.get("role", "")).strip().lower()
    content = message.get("content")

    if not isinstance(content, str):
        return None
    if not content:
        return None
    if role not in SUPPORTED_ROLES:
        return None

    return role, content


def encode_chat_prompt(
    messages: Iterable[Any],
    tokenizer: Tokenizer,
    special: ChatSpecialTokens | None = None,
    *,
    add_generation_prompt: bool = True,
    strict: bool = True,
) -> list[int]:
    """

    Args:
        messages:                <iterable>  dict {"role": ..., "content": ...}
        tokenizer:               <Tokenizer>
        special:                 <ChatSpecialTokens>  resolved ids; looked up if omitted
        add_generation_prompt:   <bool>  append `<|assistant|>\\n`
                                 False reproduces a finished training example instead.
        strict:                  <bool>  

    Return:
        <list[int]>  token ids, ready for the model
    """

    if special is None:
        special = ChatSpecialTokens.from_tokenizer(tokenizer)

    def encode_text(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    newline_ids = encode_text("\n")

    input_ids: list[int] = []

    for position, message in enumerate(messages):
        normalized = _normalize_message(message)

        if normalized is None:
            if strict:
                raise ValueError(
                    f"message at index {position} is not usable: expected a mapping with a "
                    f"non-empty string 'content' and a role in {SUPPORTED_ROLES}, got {message!r}"
                )
            continue

        role, content = normalized

        input_ids.append(special.role_token_id(role))
        input_ids.extend(newline_ids)
        input_ids.extend(encode_text(content))
        input_ids.append(special.end)
        input_ids.extend(newline_ids)

    if not input_ids:
        raise ValueError("conversation contains no usable messages")

    if add_generation_prompt:
        input_ids.append(special.assistant)
        input_ids.extend(newline_ids)

    return input_ids
