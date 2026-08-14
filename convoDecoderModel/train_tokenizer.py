
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer




def main():



    # ======== DATA AND TOKENIZER PARAMETERS ======== #
    dataset_name = "bentrevett/multi30k"
    SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[BOS]', '[EOS]']
    unk_token = '[UNK]'
    vocab_size = 32_000
    max_seq_len = 2048
    
    src_pre_tokenizer = 'whitespace'
    src_tokenizer_min_freq = 2 
    tgt_txt = 'de'
    tgt_vocab_size = 5000
    tgt_pre_tokenizer = 'whitespace'
    tgt_tokenizer_min_freq = 2
    tokenizerDir_savePath = "english2German/checkpoints/tokenizers"
    tokenizer_directory = Path(tokenizerDir_savePath)
    tokenizerSrc_savePath = str(tokenizer_directory / "english_bpe.json")
    tokenizerTgt_savePath = str(tokenizer_directory / "german_bpe.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    pipeline = [
        ParquetReader(
            "hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT"
        ),

        DocumentTokenizer(
            output_folder="./data/tokenized",
            local_working_dir="./data/tmp",
            tokenizer_name_or_path="./tokenizer",
            eos_token="<|endoftext|>",
            max_tokens_per_file=100_000_000,
            shuffle_documents=True,
            seed=42,
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        logging_dir="./logs/tokenization",
        tasks=14,
        workers=4,
    )

    executor.run()






if __name__ == "__main__":
    main()


