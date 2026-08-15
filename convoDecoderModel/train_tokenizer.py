from tokenizers import Tokenizer
from tokenizers import Regex
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


PRETOKEN_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


SPECIAL_TOKENS = [
    "<|endoftext|>", 
    "<|pad|>", 
    "<|system|>", 
    "<|user|>", 
    "<|assistant|>", 
    "<|end|>", 
]

VOCAB_SIZE = 32_768
MIN_FREQ = 2
MAX_TOKEN_LENGTH = 64
NUM_DOCUMENTS = 1_000_000
BATCH_SIZE = 1000
initial_alphabet = ByteLevel.alphabet()



def batch_texts(reader, batch_size=BATCH_SIZE):
    batch = []

    for document in reader():
        batch.append(document.text)

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def main():


    print("Initialize reader")
    # Dataset streaming
    reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT", limit=NUM_DOCUMENTS)

    print("Initialize tokenizer")
    # Create Tokenizer
    tokenizer = Tokenizer(BPE())
    # Classic Pretokenizer (gpt2 style)
    # tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    # Llama style pretokenizer
    tokenizer.pre_tokenizer = Sequence([
        Split(Regex(PRETOKEN_PATTERN), behavior="isolated", invert=True),
        ByteLevel(add_prefix_space=False, use_regex=False)
    ])
    tokenizer.decoder = ByteLevelDecoder()

    print("Train tokenizer")
    # Train Tokenizer
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ, 
        special_tokens=SPECIAL_TOKENS, 
        initial_alphabet=initial_alphabet, 
        max_token_length=MAX_TOKEN_LENGTH,
        show_progress=True 
    )

    print("Tokenizer train from iterator")
    tokenizer.train_from_iterator(
        batch_texts(reader, BATCH_SIZE),
        trainer=trainer,
        length=NUM_DOCUMENTS
    )

    print("Validate Vocabulary")
    # Validate Vocabulary
    assert tokenizer.get_vocab_size() == VOCAB_SIZE

    for token in SPECIAL_TOKENS:
        assert tokenizer.token_to_id(token) is not None

    test_strings = [
        "Hello, world!",
        "def main():\n    print('hello')",
        "Unicode: 안녕하세요 你好 😀"
    ]

    for text in test_strings:
        encoding = tokenizer.encode(text)
        decoded = tokenizer.decode(encoding.ids)
        assert decoded == text, (text, decoded)

    tokenizer.save("tokenizer_llama_style.json")



if __name__ == "__main__":
    main()


