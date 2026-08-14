import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time

from .model import Transformer
from .train import TransformerTrainer
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from collections.abc import Iterable
from pathlib import Path
import json
import re









def main():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ======== MODEL AND TRAINING PARAMETERS ======== #
    d_model = 768
    num_attn_heads = 12
    num_layers = 12         # Each layer is a causal decoder-only block
    d_ff = 2048
    dropout = 0.1
    norm = "rmsnorm"

    activate = 'swiglu'
    position_encoding = 'rope'
    tie_embeddings = True
    max_seq_len = 2048

    param_init = 'xavier_normal'

    batch_size = 32
    shuffle = True
    cur_step_count = 0
    warmup_steps = 3000
    epochs = 2


    # Create Model
    model = Transformer(
        src_vocab_size=source_vocab_size,
        tgt_vocab_size=target_vocab_size,
        src_pad_id=SRC_PAD_ID,
        tgt_pad_id=TGT_PAD_ID,
        d_model=d_model,
        num_attn_heads=num_attn_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activate,
        max_seq_len=max_seq_len,
        param_init=param_init
    ).to(device)


if __name__ == "__main__":
    main()
