from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import load_dataset, DatasetDict

from pathlib import Path
import shutil
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents

TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer_llama_style.json"
OUTPUT_DIR = ROOT / "data" / "tokenized" / "SFT_DATASET"
WORKING_DIR = ROOT / "data" / "tokenized" / "temp"
LOG_DIR = ROOT / "logs" / "tokenized_llamaStyle_SFT_DATASET"     


DATASET_NAME = "HuggingFaceTB/smol-smoltalk"

EOS_TOKEN = "<|endoftext|>"

IGNORE_INDEX = -11
BATCH_SIZE = 0
NUM_DOCUMENTS = 0 
MAX_TOKENS_PER_FILE = 0
NUM_TASKS = 14         
NUM_WORKERS = 8    

ROLE_PREFIXES = {
    "system": "System:\n",
    "user": "User:\n",
    "assistant": "Assistant:\n"
}

_TOKENIZER_CACHE: dict[str, Tokenizer] = {}

def get_tokenizer(tokenizer_path: str) -> Tokenizer:

    tokenizer = _TOKENIZER_CACHE.get(tokenizer_path)

    if tokenizer is None:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        _TOKENIZER_CACHE[tokenizer_path] = tokenizer

    return tokenizer


def encode_text(
    tokenizer: Tokenizer,
    text: str
) -> list[int]:
    """
    Encode text w/out allowing the tokenizer to insert BOS, EOS, special tokens.
    """

    encoded_ids = tokenizer.encode(text, add_special_tokens=False).ids
    return encoded_ids


def append_tokens(
    input_ids: list[int],
    labels: list[int],
    token_ids: list[int],
    supervise: bool
) -> None:

    input_ids.extend(token_ids)

    if supervise:
        labels.extend(token_ids)
    else:
        labels.extend([IGNORE_INDEX] * len(token_ids))



def tokenize_conversation(
    example: dict[str, Any],
    tokenizer_path: str,
    eos_token_id: int,
    max_seq_len: int,
    bos_token_id: int | None
) -> dict[str, Any]:
    """
    Conversation function.

    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
            
            converts to
    
    input_ids = [...]
    labels    = [-100, -100, ..., assistant tokens ..., EOS]
    """

    tokenizer = get_tokenizer(tokenizer_path)
    messages = example.get("messages")

    if not isinstance(messages, list):
        return {
            "input_ids": [],
            "labels": [],
            "length": 0,
            "num_supervised_tokens": 0,
            "keep": False,
            "source": example.get("source", "unknown")
        }

    input_ids: list[int] = []
    labels: list[int] = []

    # BOS is optional
    if bos_token_id is not None:
        input_ids.append(bos_token_id)
        labels.append(IGNORE_INDEX)

    for message in messages:

        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")

        if role not in ROLE_PREFIXES:
            continue
        if content is None:
            continue

        content = str(content).strip()

        if not content:
            continue


        # Role prefix is context only, so must not train on it.
        prefix_ids = encode_text(tokenizer, ROLE_PREFIXES[role])

        append_tokens(
            input_ids, 
            labels, 
            prefix_ids, 
            supervise=False
        )

        # message content
        content_ids = encode_text(tokenizer, content)
        if role == "assistant":
            # THIS is the actual SFT target.
            append_tokens(
                input_ids,
                labels,
                content_ids,
                supervise=True
            )

            # Teach model when the assistant turn should end.
            input_ids.append(eos_token_id)
            labels.append(eos_token_id)

        else:
            # System/user content is context only.
            append_tokens(
                input_ids,
                labels,
                content_ids,
                supervise=False
            )

            # Separate prompt turns.
            separater_ids = encode_text(tokenizer, "\n\n")
            append_tokens(input_ids, labels, separater_ids, supervise=False)

        if len(input_ids) >= max_seq_len:
            break

    # Truncate
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]

    num_supervised_tokens = sum(
        label != IGNORE_INDEX
        for label in labels
    )

    keep = (len(input_ids) >= 2 and num_supervised_tokens > 0)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "length": len(input_ids),
        "num_supervised_tokens": num_supervised_tokens,
        "keep": keep,
        "source": example.get("source", "unknown")
    }



def main() -> None:


    output_dir = Path(OUTPUT_DIR)


    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    # Verify that the tokenizer actually contains the EOS token
    # before launching DataTrove pipeline
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Tokenizer does not contain req EOS token: {EOS_TOKEN}")

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"EOS token ID: {eos_id}")

    # Disposable Smoke test directory
    for path in (OUTPUT_DIR, LOG_DIR):
        if path.exists():
            shutil.rmtree(path)


    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        # cache_dir="/workspace/data/huggingface"  # explicity for Runpod 
    )





if __name__ == "__main__":
    main()
