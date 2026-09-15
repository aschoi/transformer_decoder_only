from pathlib import Path
import sys

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer_llama_style.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from convoDecoderModel.chat_template import (  # noqa: E402
    ASSISTANT_TOKEN,
    END_TOKEN,
    EOS_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ChatSpecialTokens,
    encode_chat_prompt,
)
from convoDecoderModel.training_sft.tokenize_sft_corpus import (  # noqa: E402
    IGNORE_INDEX,
    MAX_SEQ_LEN,
    tokenize_conversation,
)


SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant."}

CONVERSATIONS = [
    # single turn, the common serving case
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
    # multi turn
    [
        {"role": "user", "content": "My name is Alex."},
        {"role": "assistant", "content": "Nice to meet you, Alex."},
        {"role": "user", "content": "What is my name?"},
        {"role": "assistant", "content": "Your name is Alex."},
    ],
    # leading system message
    [
        SYSTEM_MESSAGE,
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello!"},
    ],
    # content whose own newlines and markdown could perturb BPE at the seams
    [
        {"role": "user", "content": "Reformat this:\n\n- one\n- two\n"},
        {"role": "assistant", "content": "```python\ndef f(x):\n    return x * 2\n```"},
    ],
    # non-ascii, emoji, and leading/trailing whitespace inside the content
    [
        {"role": "user", "content": "  ¿Cómo estás? 🌍  "},
        {"role": "assistant", "content": "Estoy bien, gracias — 😄"},
    ],
]


def prompt_boundary(
    tokenized: dict[str, list[int]],
    special: ChatSpecialTokens,
    newline_length: int,
) -> int:

    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]

    assistant_positions = [
        index for index, token_id in enumerate(input_ids) if token_id == special.assistant
    ]
    if not assistant_positions:
        raise AssertionError("training example contains no assistant turn")

    boundary = assistant_positions[-1] + 1 + newline_length

    assert labels[boundary] != IGNORE_INDEX, "supervision should begin right after <|assistant|>\\n"
    assert labels[boundary - 1] == IGNORE_INDEX, "the opening <|assistant|>\\n must not be supervised"

    return boundary


def tokenize_like_training(
    messages: list[dict[str, str]],
    tokenizer: Tokenizer,
    special: ChatSpecialTokens,
) -> dict[str, list[int]]:

    return tokenize_conversation(
        {"messages": messages},
        tokenizer,
        special.eos,
        special.end,
        special.system,
        special.user,
        special.assistant,
        MAX_SEQ_LEN,
    )


def main() -> None:

    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    special = ChatSpecialTokens.from_tokenizer(tokenizer)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False).ids

    # Test 1: the control tokens resolve, and to the ids the corpus was built with
    for token, token_id in (
        (EOS_TOKEN, special.eos),
        (END_TOKEN, special.end),
        (SYSTEM_TOKEN, special.system),
        (USER_TOKEN, special.user),
        (ASSISTANT_TOKEN, special.assistant),
    ):
        assert tokenizer.token_to_id(token) == token_id, f"{token} id mismatch"

    assert special.stop_ids == frozenset({special.end, special.eos}), "stop set must cover both boundaries"
    assert len(special.stop_ids) == 2, "turn boundary and conversation boundary must be distinct tokens"
    print(f"Test 1: special token ids resolve ({special}): PASS")

    # Test 2: an inference prompt equals the training prefix, exactly
    for index, messages in enumerate(CONVERSATIONS):
        context = messages[:-1]

        training = tokenize_like_training(messages, tokenizer, special)
        boundary = prompt_boundary(training, special, len(newline_ids))
        expected = training["input_ids"][:boundary]

        actual = encode_chat_prompt(context, tokenizer, special, add_generation_prompt=True)

        assert actual == expected, (
            f"conversation {index}: runtime prompt diverges from the training layout\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"  expected text: {tokenizer.decode(expected, skip_special_tokens=False)!r}\n"
            f"  actual text:   {tokenizer.decode(actual, skip_special_tokens=False)!r}"
        )

    print(f"Test 2: inference prompt matches training prefix for {len(CONVERSATIONS)} conversations: PASS")

    # Test 3: the prompt ends by opening the assistant turn, so the model replies
    # rather than continuing a document
    for messages in CONVERSATIONS:
        prompt = encode_chat_prompt(messages[:-1], tokenizer, special)

        assert prompt[-(1 + len(newline_ids))] == special.assistant, "prompt must open an assistant turn"
        assert prompt[-len(newline_ids):] == newline_ids, "assistant token must be followed by a newline"
        assert special.eos not in prompt, "a prompt must never contain the conversation terminator"

    print("Test 3: prompts end with an opened assistant turn: PASS")

    # Test 4: without a generation prompt we reproduce the training example itself,
    # bar the trailing <|endoftext|> the corpus builder adds after the last reply
    for index, messages in enumerate(CONVERSATIONS):
        training = tokenize_like_training(messages, tokenizer, special)
        expected = training["input_ids"]

        assert expected[-1] == special.eos, "training example should end with <|endoftext|>"

        actual = encode_chat_prompt(messages, tokenizer, special, add_generation_prompt=False)

        assert actual == expected[:-1], f"conversation {index}: full-conversation encoding diverges"

    print("Test 4: full-conversation encoding matches the training example: PASS")

    # Test 5: multi-turn prompts grow by exactly one turn at a time
    long_conversation = CONVERSATIONS[1]
    short_prompt = encode_chat_prompt(long_conversation[:1], tokenizer, special)
    long_prompt = encode_chat_prompt(long_conversation[:3], tokenizer, special)

    generation_prompt_length = 1 + len(newline_ids)
    shared = len(short_prompt) - generation_prompt_length

    assert len(long_prompt) > len(short_prompt), "adding turns must extend the prompt"
    assert long_prompt[:shared] == short_prompt[:shared], "earlier turns must be preserved verbatim"
    print("Test 5: multi-turn prompts extend rather than rewrite history: PASS")

    # Test 6: strict mode rejects what training would have silently dropped
    unusable = [
        ([], "empty conversation"),
        ([{"role": "user", "content": ""}], "empty content"),
        ([{"role": "user", "content": None}], "non-string content"),
        ([{"role": "moderator", "content": "hi"}], "unsupported role"),
        (["not a message"], "non-mapping message"),
    ]

    for messages, description in unusable:
        try:
            encode_chat_prompt(messages, tokenizer, special)
        except ValueError:
            pass
        else:
            raise AssertionError(f"strict mode should have rejected: {description}")

    # ...and lenient mode drops them the way tokenize_conversation does
    mixed = [
        {"role": "moderator", "content": "dropped"},
        {"role": "user", "content": "kept"},
    ]
    lenient = encode_chat_prompt(mixed, tokenizer, special, strict=False)
    expected_lenient = encode_chat_prompt(mixed[1:], tokenizer, special)

    assert lenient == expected_lenient, "lenient mode must mirror tokenize_conversation's skipping"
    print("Test 6: strict mode rejects unusable messages, lenient mode mirrors training: PASS")

    # Test 7: show the wire format, so a human can eyeball it
    sample = encode_chat_prompt(
        [SYSTEM_MESSAGE, {"role": "user", "content": "What is the capital of France?"}],
        tokenizer,
        special,
    )
    print()
    print("Test 7: rendered prompt:")
    print("-" * 75)
    print(repr(tokenizer.decode(sample, skip_special_tokens=False)))
    print("-" * 75)

    print("\nAll SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
