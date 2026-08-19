from pathlib import Path
import shutil
import torch
from tokenizers import Tokenizer
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


ROOT = Path(__file__).resolve().parent

TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer_llama_style.json"
OUTPUT_DIR = ROOT/ "data" / "tokenized" / "fineweb_edu_10bt"
WORKING_DIR = ROOT / "data" / "tokenized" / "temp"
LOG_DIR = ROOT / "logs" / "tokenized_llamaStyle_finewebEdu_10bt"     


DATASET_PATH = "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"

EOS_TOKEN = "<|endoftext|>"

BATCH_SIZE = 1_000
NUM_DOCUMENTS = 10_000  
MAX_TOKENS_PER_FILE = 250_000_000
NUM_TASKS = 14         
NUM_WORKERS = 8         


def main() -> None:
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

    executor = LocalPipelineExecutor(
        pipeline=[
            # ParquetReader(dataset_path, glob_pattern="*.parquet"),
            ParquetReader(DATASET_PATH, glob_pattern="*.parquet"), 
            DocumentTokenizer(
                output_folder=str(OUTPUT_DIR),
                tokenizer_name_or_path=str(TOKENIZER_PATH),
                save_filename="fineweb_edu",
                eos_token=EOS_TOKEN,
                local_working_dir=str(WORKING_DIR),
                save_index=True,
                save_loss_metadata=False,
                batch_size=BATCH_SIZE,
                max_tokens_per_file=MAX_TOKENS_PER_FILE,
                
                shuffle_documents=True,
                seed=77,
                save_final_metadata=True,
            )
        ],
        tasks=NUM_TASKS,        
        workers=NUM_WORKERS,
        logging_dir=str(LOG_DIR),
        skip_completed=True
    )

    executor.run()


if __name__ == "__main__":
    main()
