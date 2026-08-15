from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


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


def main():


    test_strings = [
        "Hello, world!",
        "def main():\n    print('hello')",
        "Unicode: 안녕하세요 你好 😀",
        "Hello, this is a tokenizer test",
        "A bunch of word and letters and things and such",
        "efoiajvaneka aejfae afeaaifjeanz",
        "12094 280101 31 14 381jofe#Q2341",
        "once a in the attic.!"
    ]


    tokenizer = Tokenizer.from_file('tokenizer/tokenizer_llama_style.json')

    for text in test_strings:

        print("text:", text)
        encoding = tokenizer.encode(text)
        print(f"encoding token: {encoding.tokens}")
        print(f"encoding id: {encoding.ids}")

        decoded = tokenizer.decode(encoding.ids)
        print(f'decoded: {decoded}')
        assert decoded == text, (text, decoded)
        print()



if __name__ == "__main__":
    main()


