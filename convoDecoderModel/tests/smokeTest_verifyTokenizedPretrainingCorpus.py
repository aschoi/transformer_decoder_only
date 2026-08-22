from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer_llama_style.json"
DATASET_PATH = ROOT / "data" / "tokenized" / "fineweb_edu_10bt" / "smoke_00000_unshuffled.ds"
CONTEXT_LENGTH = 2048
EOS_TOKEN = "<|endoftext|>"


class TokenDataset:
    """
    class definition for TokenDataset
    """

    path: Path
    context_length: int

    def __init__(
        self,
        path: Path,
        context_length: int 
    ) -> None:
        """
        Constructure for TokenDataset

        Args:
            path:               <Path>
            context_length:     <int>

        """

        self.path = path
        if not path.is_file():
            raise FileNotFoundError(path)
        
        metadata_path = Path(f"{path}.metadata")
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata_lines = metadata_path.read_text().splitlines()
        if len(metadata_lines) < 2:
            raise ValueError(f"Malformed metadata file: {metadata_path}")
        
        # DataTrove stores:
        #   tokenizer_path|token_size
        #   token_count
        #   human-readable token count

        _, token_size_text = metadata_lines[0].rsplit("|", 1)
        self.token_size = int(token_size_text)
        self.expected_num_tokens = int(metadata_lines[1])
        if self.token_size == 2:
            dtype = np.dtype("<u2")
        elif self.token_size == 4:
            dtype = np.dtype("<u4")
        else:
            raise ValueError(f"Unsupported token size: {self.token_size}")

        expected_bytes = self.expected_num_tokens * self.token_size
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                "Token file size does not match metadata:\n"
                f"expected: {expected_bytes} bytes\n"
                f"actual:   {actual_bytes}   bytes"
            )

        self.tokens = np.memmap(path, dtype=dtype, mode='r')

        self.context_length = context_length
        if len(self.tokens) <= self.context_length:
            raise ValueError("Dataset is too small for the requested context length.")


    def __len__(self) -> int:
        """
        returns length
        """
        return (
            len(self.tokens) - 1
        ) // self.context_length


    def __getitem__(
        self,
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns item at index
        """

        if idx < 0 or len(self) <= idx:
            raise IndexError(idx)

        start = idx * self.context_length
        end = start + self.context_length + 1

        # Convert uint16/uint32 disk representation -> PyTorch int64
        # So as to mesh with PyTorch embedding indices 
        tokens = np.array(self.tokens[start:end], dtype=np.int64, copy=True)
        tokens = torch.from_numpy(tokens)

        input_ids = tokens[:-1]
        targets = tokens[1:]

        return input_ids, targets



def main() -> None:

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    dataset = TokenDataset(DATASET_PATH, CONTEXT_LENGTH)

    print(f"Token file: {DATASET_PATH}")
    print(f"Token width: {dataset.token_size} bytes")
    print(f"Total token count: {len(dataset.tokens):,}")
    print(f"Training Sequences: {len(dataset):,}")
    print()

    # Test 1: contstruct training sample
    X, y = dataset[0]
    print(f"X.shape: {X.shape},  X dtype: {type(X)}")
    print(f"y.shape: {y.shape},  y dtype: {type(y)}")
    assert X.shape == (CONTEXT_LENGTH,), "Test 1: Constructed X training sample: FAIL"
    assert y.shape == (CONTEXT_LENGTH,), "Test 1: Constructed y training sample: FAIL"
    print("Test 1: Constructed training sample: PASS")


    # Test 2: Verify next-token shifting
    assert torch.equal(X[1:], y[:-1]), "Test2: Next-token shift: FAIL"
    print("Test 2: Next-token shift: PASS")


    # Test 3: Verify token IDs are valid
    vocab_size = tokenizer.get_vocab_size()
    assert X.min().item() >= 0, "Test 3: Token ID range. X min >= 0: FAIL"
    assert X.max().item() < vocab_size, "Test 3: Token ID range. X max < vocab_size: FAIL"
    assert y.min().item() >= 0, "Test 3: Token ID range. y min >= 0: FAIL"
    assert y.max().item() < vocab_size, "Test 3: Token ID range. y max < vocab_size: FAIL"
    print("Test 3: Token ID range: PASS")


    # Test 4: Verify document index
    idx_path = Path(f"{DATASET_PATH}.index")
    document_ends = np.memmap(idx_path, dtype=np.dtype("<u8"), mode='r')
    assert len(document_ends) > 0, "Test 4 - Document idx: len(document_ends) > 0: FAIL"
    assert int(document_ends[-1]) == len(dataset.tokens), "Test 4: int(document_ends[-1]) == len(dataset.tokens): FAIL"
    assert np.all(document_ends[1:] > document_ends[:-1]), "Test 4: Verify Doc idx: np.all(document_ends[1:] > document_ends[:-1]): FAIL"
    print(f"Documents index: {len(document_ends):,}")
    print("Test 4: Document index: PASS")


    # Test 5: Verify every sampled Document ends in EOS
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise ValueError(f"Test 5: EOS token missing from tokenizer: {EOS_TOKEN}")
    # Check the first 100 documents. (Enough for just SMOKE test)
    for doc_end in document_ends[500:700]:
        doc_end = int(doc_end)
        assert dataset.tokens[doc_end - 1] == eos_id, f"Test 5: document {doc_end} ends with EOS: FAIL"

    print("Test 5: EOS boundaries: PASS")


    # Test 6: decode actual stored tokens
    preview_ids = X[500:700].tolist()
    decoded = tokenizer.decode(preview_ids, skip_special_tokens=False)
    print()
    print("Test 6: Decoded Preview:")
    print("-" * 75)
    print(decoded)
    print("-" * 75)

    print("\nAll SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()


