from __future__ import annotations

import argparse
import json
from typing import Any

from datasets import load_dataset, DatasetDict

from pathlib import Path
import shutil
from tokenizers import Tokenizer



def check_tokenized_example(
    raw_example: dict[str, Any],
    tokenizer: Tokenizer,
    eos_token_id: int,
    end_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
    max_seq_len: int
) -> None:

    """
    raw_example
        {
            role: content,
            role: content,
            ...
        }
        {
            "user": "a block of words",
            "assistant: "The first response that will be learned",
            "user": "Another inquiry",
            "assistant": "The next response"
            ...
        }
    """

    result = tokenize_conversation(
        raw_example,
        tokenizer,
        eos_token_id,
        end_token_id,
        system_token_id,
        user_token_id,
        assistant_token_id,
        max_seq_len
    )

    input_ids = result["input_ids"]
    labels = result["labels"]

    print("\n" + "=" * 50)
    print("  RAW MESSAGES")
    print("=" * 50)
    for message in raw_example["messages"]:
        print(f"{message['role']}: {message['content']!r}\n")


    isSupervised = False
    print("\n" + "=" * 50)
    print("  TOKENIZED")
    print("=" * 50)
    for i, (token_id, label) in enumerate(zip(input_ids, labels)):
        token = tokenizer.id_to_token(token_id)
        supervised = label != IGNORE_INDEX

        if supervised != isSupervised:
            print()
            isSupervised = supervised

        print(
            f"{i:4d}   "
            f"id={token_id:6d}   "
            f"label={label:6d}   "
            f"supervised={str(supervised):5s}   "
            f"token={token!r}"
        )

    print("\n" + "=" * 50)
    print("  DECODED")
    print("=" * 50)
    print(tokenizer.decode(input_ids, skip_special_tokens=False))

    print("\n" + "-" * 50)
    print(f"Sqeuence length: {len(input_ids)}")
    print(f"Supervised tokens: {sum(label != IGNORE_INDEX for label in labels)}")
    print("-" * 50)

    assert len(input_ids) == len(labels), "len(input_ids) == len(labels): FAIL"
    assert len(input_ids) <= max_seq_len, "len(input_ids) <= max_seq_len: FAIL"
    assert all(isinstance(token_id, int) for token_id in input_ids), "all(isinstance(token_id, int) for token_id in input_ids): FAIL"

        

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
    tokenizer: Tokenizer,
    eos_token_id: int,
    end_token_id: int,
    system_token_id: int,
    user_token_id: int,
    assistant_token_id: int,
    max_seq_len: int
) -> dict[str, list[int]]:


    input_ids: list[int] = []
    labels: list[int] = []

    def append_context_ids(token_ids: list[int]) -> None:
        input_ids.extend(token_ids)
        labels.extend([IGNORE_INDEX] * len(token_ids))

    def append_target_ids(token_ids: list[int]) -> None:
        input_ids.extend(token_ids)
        labels.extend(token_ids)

    def append_context_text(text: str) -> None:
        token_ids = encode_text(tokenizer, text)
        append_context_ids(token_ids)

    def append_target_text(text: str) -> None:
        token_ids = encode_text(tokenizer, text)
        append_target_ids(token_ids)



    messages = example["messages"]

    if not isinstance(messages, list):
        return {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }

    last_valid_role: str | None=None

    for message in messages:

        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")

        if not isinstance(content, str):
            continue

        if not content:
            continue

        if role == "system":
            append_context_ids([system_token_id])
            append_context_text("\n")
            append_context_text(content)

            append_context_ids([end_token_id])
            append_context_text("\n")

        elif role == "user":
            append_context_ids([user_token_id])

            append_context_text("\n")
            append_context_text(content)

            append_context_ids([end_token_id])
            append_context_text("\n")

        elif role == "assistant":

            append_context_ids([assistant_token_id])
            append_context_text("\n")
            
            append_target_text(content)
            append_target_ids([end_token_id])

            append_context_text("\n")
            # the actual assistant reponse is supervised
            # append_target(content)
        
        else:
            continue

        last_valid_role = role

    if last_valid_role == "assistant":
        append_target_ids([eos_token_id])
    
    # truncate
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }




def has_supervised_tokens(example: dict[str, Any]) -> bool:

    return any(
        label != IGNORE_INDEX
        for label in example["labels"]
    )




ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer_llama_style.json"
OUTPUT_DIR = ROOT / "data" / "tokenized" / "smol_talk_dataset"


# WORKING_DIR = ROOT / "data" / "tokenized" / "temp"
# LOG_DIR = ROOT / "logs" / "tokenized_llamaStyle_smol_talk_dataset"     


DATASET_NAME = "HuggingFaceTB/smol-smoltalk"

EOS_TOKEN = "<|endoftext|>"
END_TOKEN = "<|end|>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

IGNORE_INDEX = -100
MAX_SEQ_LEN = 2048

_TOKENIZER_CACHE: dict[str, Tokenizer] = {}



NUM_PROC = 8

VALIDATION_FRACTION = .01
MAX_SHARD_SIZE = "1GB"


def main() -> None:


    output_dir = Path(OUTPUT_DIR)

    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    vocab_size = tokenizer.get_vocab_size()


    special_tokens = {}
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    end_token_id = tokenizer.token_to_id(END_TOKEN)
    system_token_id = tokenizer.token_to_id(SYSTEM_TOKEN)
    user_token_id = tokenizer.token_to_id(USER_TOKEN)
    assistant_token_id = tokenizer.token_to_id(ASSISTANT_TOKEN)
    special_tokens["EOS"] = eos_token_id
    special_tokens["END"] = end_token_id
    special_tokens["SYSTEM"] = system_token_id
    special_tokens["USER"] = user_token_id
    special_tokens["ASSISTANT"] = assistant_token_id

    for name, token_id in special_tokens.items():
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required {name} token")       

    print(f"Tokenizer vocabulary size: {vocab_size}")
    print(f"EOS token ID: {eos_token_id}")
    print(f"END token ID: {end_token_id}")
    print(f"SYSTEM token ID: {system_token_id}")
    print(f"USER token ID: {user_token_id}")
    print(f"ASSISTANT token ID: {assistant_token_id}")


    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        # cache_dir="/workspace/data/huggingface"  # explicity for Runpod 
    )



    # Sanity Check testing
    for i in range(5):
        print("*" * 50)
        print(f"Example: {i}")
        print("*" * 50)

        check_tokenized_example(
            dataset[i],
            tokenizer,
            eos_token_id,
            end_token_id,
            system_token_id,
            user_token_id,
            assistant_token_id,
            MAX_SEQ_LEN
        )

    


    original_columns = dataset.column_names


    dataset = dataset.map(
        tokenize_conversation,
        fn_kwargs={
            "tokenizer": tokenizer,
            "eos_token_id": eos_token_id,
            "end_token_id": end_token_id,
            "system_token_id": system_token_id,
            "user_token_id": user_token_id,
            "assistant_token_id": assistant_token_id,
            "max_seq_len": MAX_SEQ_LEN,
            
        },
        remove_columns=original_columns,
        num_proc=NUM_PROC,
        desc="Tokenizing SFT conversation"
    )

    dataset = dataset.filter(
        has_supervised_tokens,
        num_proc=NUM_PROC,
        desc="filtering unusable examples"
    )

    split_dataset = dataset.train_test_split(
        test_size=VALIDATION_FRACTION,
        seed=42,
        shuffle=True
    )

    output = DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output.save_to_disk(str(OUTPUT_DIR), max_shard_size=MAX_SHARD_SIZE)

    print()
    print(f"Train examples:           {len(output['train'])}")
    print(f"Validation examples:      {len(output['validation'])}")
    print(f"saved to:      {OUTPUT_DIR}")



if __name__ == "__main__":
    main()
