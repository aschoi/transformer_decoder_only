from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from datatrove.utils.dataset import DatatroveFolderDataset
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk


import os
import math
import time
import json
import random
import contextlib
import inspect
import random
from pathlib import Path
from typing import Any, Iterator
from dataclasses import asdict, dataclass
from functools import partial

from .model import AcaiModelConfig, AcaiTransformer


@dataclass(frozen=True, slots=True)
class SftConfig:
    """
    class structure that stores configuration for Supervised Fine Tuning run

    """

    train_data_dir: str
    validation_data_dir: str | None=None
    pretrained_checkpoint_path: str | None=None
    resume_fromCheckpoint_path: str | None=None

    eval_batches: int = 50
    eval_interval: int = 500
    checkpoint_dir: str = "checkpoints/sft"
    checkpoint_interval: int = 1_000
    gradient_checkpointing: bool = False
    output_dir: str = "output/sft"
    log_interval: int = 10

    max_steps: int = 10_000_000_000
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 16

    lr: float = 1.0e-5
    min_lr: float = 1.0e-6
    warmup_steps: int = 200
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    precision: str = "auto"

    num_workers: int = 2
    seed: int = 173
    compile_model: bool = False
    # pretraining_epochs
    # sft_training_epochs   # for the future, have sft training epochs



class ResumableRandomSampler(Sampler[int]):
    """
    
    """
    def __init__(
        self,
        dataset_size: int,
        epoch: int = 0,
        start_index: int = 0,
        seed: int = 23
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset size must be greater than 0")

        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = epoch
        self.set_start_index(start_index)

    def set_start_index(
        self,
        start_index: int
    ) -> None:
        if not 0 <= start_index <= self.dataset_size:
            raise ValueError("Start index must be between 0 and dataset_size")
        self.start_index = start_index

    def advance_epoch(self) -> None:
        self.epoch += 1
        self.start_index = 0

    def get_epoch(self) -> None:
        return self.epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()

        return iter(indices[self.start_index:])


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_precision(
    name: str,
    device: torch.device
) -> torch.dtype | None:
    if name not in {"auto", "fp32", "bf16", "fp16"}:
        raise ValueError("precision must be one of: auto, fp32, bf1`6, fp16")

    if device.type != 'cuda':
        if name in {"bf16", "fp16"}:
            raise ValueError("bf16, fp16 precision requires CUDA")
        return None

    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This current CUDA device does not support bf16")
        return torch.bfloat16

    if name == 'fp16':
        return torch.float16

    return None


def autocast_context(
    device: torch.device,
    dtype: torch.dtype | None,
) -> contextlib.AbstractContextManager:
    
    if dtype is None:
        return contextlib.nullcontext()
    
    return torch.autocast(device_type=device.type, dtype=dtype)





def get_lr(
    cur_step: int,
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float
) -> float:

    if max_steps <= 0:
        raise ValueError("Max steps must be positive")
    if warmup_steps < 0:
        raise ValueError("Warmup steps must be >= 0")
    if warmup_steps > max_steps:
        raise ValueError("Warm up steps must be smaller than max steps")


    if warmup_steps > 0 and cur_step < warmup_steps:
        return max_lr * (cur_step + 1) / warmup_steps

    # Merely proactive boundary edge cases. 
    if cur_step >= max_steps:
        return min_lr
    if max_steps == warmup_steps:
        return min_lr

    decay_ratio = (cur_step - warmup_steps) / (max_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coefficient * (max_lr - min_lr)


def load_sft_dataset(
    data_dir: str,
    *,
    isTraining: bool
) -> HFDataset:
    """
    """

    loaded_datasetDict = load_from_disk(data_dir)

    if isinstance(loaded_datasetDict, DatasetDict):
        split_name = "train" if isTraining else "validation"
        if split_name not in loaded_datasetDict:
            raise KeyError(f"This <DatabaseDict> does not contain {split_name}")
        dataset = loaded_datasetDict[split_name]

    elif isinstance(loaded_datasetDict, HFDataset):
        dataset = loaded_datasetDict
    else:
        raise TypeError(f"Unsupported Dataset type: {type(loaded_datasetDict)}")

    if len(dataset) == 0:
        raise ValueError(f"This dataset is empty")
    required_cols = {"input_ids", "labels"}
    missing_cols = required_cols - set(dataset.column_names)
    if missing_cols:
        raise ValueError(f"This dataset is missing required columns: {sorted(missing_cols)}")

    return dataset


    

def make_data_loader(
    data_dir,
    *,
    model_config,
    sft_config,
    isTraining,
    conversations_already_processed
) -> DataLoader:
    
    dataset = load_sft_dataset(data_dir, isTraining=isTraining)

    sampler: ResumableRandomSampler | None=None
    if isTraining:
        samples_per_epoch = (len(dataset) // sft_config.micro_batch_size) * sft_config.micro_batch_size

        if samples_per_epoch == 0:
            raise ValueError("dataset does not contain enough conversations for one micro-batch")

        start_epoch = conversations_already_processed // samples_per_epoch
        start_index = conversations_already_processed % samples_per_epoch

        sampler = ResumableRandomSampler(
            len(dataset),
            seed=sft_config.seed,
            epoch=start_epoch,
            start_index=start_index
        )

    collate_fn = partial(
        collate_sft_batch,
        pad_token_id=model_config.pad_token_id,
        max_seq_len=model_config.max_seq_len
    )


    loader = DataLoader(
        dataset,
        batch_size=sft_config.micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=sft_config.num_workers,
        pin_memory=True,
        persistent_workers=sft_config.num_workers > 0,
        drop_last=isTraining,
        collate_fn=collate_fn
    )

    return loader



def next_batch(
    iterator: Iterator[dict[str, torch.Tensor]],
    loader: DataLoader,
    total_epochs: int
) -> tuple[dict[str, torch.Tensor], Iterator[dict[str, torch.Tensor]]]:

    
    try:
        return next(iterator), iterator
    except StopIteration:
        if isinstance(loader.sampler, ResumableRandomSampler):
            loader.sampler.advance_epoch()

            if loader.sampler.get_epoch() >= total_epochs:
                return None, iterator

        iterator = iter(loader)
        return next(iterator), iterator


def load_pretrained_weights(
    path: Path,
    model: AcaiTransformer,
    expected_model_config: AcaiModelConfig
) -> None:

    checkpoint_state = torch.load(path, map_location='cpu', weights_only=False)

    if not isinstance(checkpoint_state, dict) or "model" not in checkpoint_state:
        raise ValueError("pretrained checkpoint must contain a 'model' state_dict")
    if "AcaiModelConfig" in checkpoint_state:
        saved_model_config = AcaiModelConfig.from_dict(checkpoint_state["AcaiModelConfig"])
        if saved_model_config != expected_model_config:
            raise ValueError("pretrained checkpoint model config does nto match this run")

    model.load_state_dict(checkpoint_state['model'])



def save_checkpoint(
    path: Path,
    model: AcaiTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    completed_steps: int,
    samples_processed: int,
    tokens_processed: int,
    supervised_tokens_processed: int,
    model_config: AcaiModelConfig,
    sft_config: SftConfig
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "completed_steps": completed_steps,
        "samples_processed": samples_processed,
        "tokens_processed": tokens_processed,
        "supervised_tokens_processed": supervised_tokens_processed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "AcaiModelConfig": model_config.to_dict(),
        "SftConfig": asdict(sft_config),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
    }

    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    expected_model_config
) -> tuple[int, int, int, int]:
    
    checkpoint_state = torch.load(path, map_location='cpu', weights_only=False)
    saved_model_config = AcaiModelConfig.from_dict(checkpoint_state["AcaiModelConfig"])

    if saved_model_config != expected_model_config:
        raise ValueError("Checkpoint model config does not match this run")

    model.load_state_dict(checkpoint_state["model"])
    optimizer.load_state_dict(checkpoint_state["optimizer"])

    if "scaler" in checkpoint_state:
        scaler.load_state_dict(checkpoint_state["scaler"])
    if "torch_rng_state" in checkpoint_state:
        torch.set_rng_state(checkpoint_state["torch_rng_state"])
    if "cuda_rng_state" in checkpoint_state:
        torch.cuda.set_rng_state_all(checkpoint_state["cuda_rng_state"])

    return (
        int(checkpoint_state['completed_steps']),
        int(checkpoint_state.get('samples_processed', 0)),
        int(checkpoint_state.get("tokens_processed", 0)),
        int(checkpoint_state.get('supervised_tokens_processed', 0))
    )


def build_optimizer(
    model,
    sft_config
):
    
    decay_parameters = []
    no_decay_parameters = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay_parameters if parameter.ndim >= 2 else no_decay_parameters).append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": sft_config.weight_decay
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0
        }
    ]

    optimizer_arguments = {
        "lr": sft_config.lr,
        "betas": (sft_config.beta1, sft_config.beta2)
    }

    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        optimizer_arguments["fused"] = True

    return torch.optim.AdamW(parameter_groups, **optimizer_arguments)


def shift_sft_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply causal next token shift while preserving preprocessing labels
    """

    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)

    X = input_ids[:, :-1]
    y = labels[:, 1:]
    X_attention_mask = attention_mask[:, :-1]

    return X, y, X_attention_mask



@torch.no_grad()
def evaluate(
    model,
    loader,
    *,
    numEvalBatches,
    amp_dtype,
    device
) -> tuple[float, float]:
    """
    """

    prevTrainingMode = model.training
    model.eval()
    loss_numerator = 0.0
    evaluatedSupervisedTokenCount = 0

    try:
        for iBatch, curBatch in enumerate(loader):

            if iBatch >= numEvalBatches:
                break

            X, y, _ = shift_sft_batch(curBatch, device=device)
            supervised_tokens = int((y != IGNORE_INDEX).sum().item())

            if supervised_tokens == 0:
                continue

            with autocast_context(device, amp_dtype):
                outputFeatures = model(X, tnsr_labels=y, return_logits=False)

            if outputFeatures.loss is None:
                raise RuntimeError("Model did not return validation loss")

            loss_numerator += outputFeatures.loss.item() * supervised_tokens
            evaluatedSupervisedTokenCount += supervised_tokens

    finally:
        model.train(prevTrainingMode)
    
    if evaluatedSupervisedTokenCount == 0:
        raise RuntimeError("Evaluate - evaluated token count == 0. Validation loader did not provide batches")

    # Validation calculations
    mean_loss = loss_numerator / evaluatedSupervisedTokenCount
    perplexity = math.exp(min(mean_loss, 20.0))

    return mean_loss, perplexity 


def validate_config(
    model_config: AcaiModelConfig,
    sft_config: SftConfig
) -> None:
    if model_config.pad_token_id is None:
        raise ValueError("sft dynamic padding requires pad token id")
    if sft_config.micro_batch_size <= 0:
        raise ValueError("micro batch size must be postive")
    # More validations checks to be added


def collate_sft_batch(
    conversations: list[dict[str, Any]],
    pad_token_id: int,
    max_seq_len: int
) -> dict[str, torch.Tensor]:
    """
    Right pad variable length sft examples
    """

    if not conversations:
        raise ValueError("Batch is empty")

    lengths = [len(convo['input_ids']) for convo in conversations]
    batch_seq_len = max(lengths)

    if batch_seq_len < 2:
        raise ValueError("sequences must contain at least 2 tokens")
    if batch_seq_len > max_seq_len:
        raise ValueError(f"datset seq length > max seq length")

    batch_size = len(conversations)
    
    input_ids = torch.full(
        (batch_size, batch_seq_len),
        fill_value=pad_token_id,
        dtype=torch.int64
    )
    labels = torch.full(
        (batch_size, batch_seq_len),
        fill_value=IGNORE_INDEX,
        dtype=torch.int64
    )
    attention_mask = torch.zeros(
        (batch_size, batch_seq_len),
        dtype=torch.bool
    )

    for iConvo, convo in enumerate(conversations):
        convo_input_ids = convo["input_ids"]
        convo_labels = convo["labels"]

        if len(convo_input_ids) != len(convo_labels):
            raise ValueError("input ids and labels lengths are not equal")

        seq_len = len(convo_input_ids)
        if seq_len < 2:
            raise ValueError("This conversation has less than 2 tokens")

        input_ids[iConvo, :seq_len] = torch.as_tensor(convo_input_ids, dtype=torch.int64)
        labels[iConvo, :seq_len] = torch.as_tensor(convo_labels, dtype=torch.int64)
        attention_mask[iConvo, :seq_len] = True

    shifted_supervised = (labels[:, 1:] != IGNORE_INDEX).any(dim=1)
    if not bool(shifted_supervised.all()):
        raise ValueError("Encoutered an sft conversation with no supervised target tokens after causal shift")

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask" : attention_mask
    }




IGNORE_INDEX = -100

def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this training configuration")

    # Model Configuration
    model_config = AcaiModelConfig(
        vocab_size=32_768,
        d_model=768,
        num_layers=18,
        num_attn_heads=12,
        d_ff=2048,
        max_seq_len=2048, 
        rope_baseFreq=10_000.0,
        dropout_rate=0.0,
        pad_token_id=1,
        tie_word_embeddings=True,
        init_std=0.02
    )


    # Supervised Fine Tuning Configuration
    sft_config = SftConfig(
        train_data_dir="convoDecoderModel/data/tokenized/smol_talk_dataset/train",
        validation_data_dir="convoDecoderModel/data/tokenized/smol_talk_dataset/validation",
        pretrained_checkpoint_path="convoDecoderModel/checkpoints/checkpoint_final.pt",
        eval_batches=100,
        eval_interval=200,
        checkpoint_dir="convoDecoderModel/checkpoints/sft",
        checkpoint_interval=500,
        gradient_checkpointing=False, # True
        output_dir="convoDecoderModel/output/sft",
        log_interval=100,
        max_steps=1_000_000_000, 
        micro_batch_size=4, 
        gradient_accumulation_steps=8,
        lr=2.0e-5,
        min_lr=2.0e-6,
        warmup_steps=400,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        max_grad_norm=1.0,
        precision="auto",
        num_workers=4,        
        seed=717,
        compile_model=False,
        # resume_fromCheckpoint_path="convoDecoderModel/checkpoints/checkpoint_final.pt"
        # pretraining_epochs
        # sft_training_epochs
    )
    EPOCHS = 1

    validate_config(model_config, sft_config)

    set_seed(sft_config.seed)
    device = torch.device("cuda")

    torch.set_float32_matmul_precision("high")
    amp_dtype = resolve_precision(sft_config.precision, device)
    use_grad_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    # Model
    model = AcaiTransformer(model_config).to(device)
    if sft_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()


    # New optimizer needed
    if sft_config.resume_fromCheckpoint_path is None:
        assert sft_config.pretrained_checkpoint_path is not None
        load_pretrained_weights(
            path=Path(sft_config.pretrained_checkpoint_path),
            model=model,
            expected_model_config=model_config
        )
        print("loaded pretrained weights")

    optimizer = build_optimizer(model, sft_config)

    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter count: {parameter_count}")
    print(f"trainiable parameters: {trainable_parameter_count}")


    start_step = 0
    samples_processed = 0
    tokens_processed = 0
    supervised_tokens_processed = 0


    if sft_config.resume_fromCheckpoint_path is not None:
        start_step, samples_processed, tokens_processed, supervised_tokens_processed = load_checkpoint(
            path=Path(sft_config.resume_fromCheckpoint_path),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_model_config=model_config
        )


    train_loader = make_data_loader(
        sft_config.train_data_dir,
        model_config=model_config,
        sft_config=sft_config,
        isTraining=True,
        conversations_already_processed=samples_processed
    )

    validation_loader = None
    if sft_config.validation_data_dir is not None:
        validation_loader = make_data_loader(
            sft_config.validation_data_dir,
            model_config=model_config,
            sft_config=sft_config,
            isTraining=False,
            conversations_already_processed=0
        )

    output_dir = Path(sft_config.output_dir)
    checkpoint_dir = Path(sft_config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_config.save_json(output_dir/"model_config.json")
    with (output_dir / "sft_config.json").open('w', encoding="utf-8") as f:
        json.dump(asdict(sft_config), f, indent=4, sort_keys=True)
        f.write("\n")

    training_model = torch.compile(model) if sft_config.compile_model else model


    # TRAINING
    model.train()
    data_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    logging_tokens = 0
    logging_supervised_tokens = 0
    logging_start = time.perf_counter()
    completed_steps = start_step

    sftFinished = False

    for step in range(start_step, sft_config.max_steps):        

        curr_lr = get_lr(
            step,
            max_steps=sft_config.max_steps,
            warmup_steps=sft_config.warmup_steps,
            max_lr=sft_config.lr,
            min_lr=sft_config.min_lr
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        optimizer.zero_grad(set_to_none=True)

        micro_batches: list[tuple[dict[str, torch.Tensor], int]] = []
        step_supervised_tokens = 0


        accumulated_loss = 0.0

        # Gradient Accumulation 
        #   For Computational Effecieny, don't need to backprop every time.
        #   Accumulate gradients from multiple runs all at once, then with total gradient, do one backprop
        for _ in range(sft_config.gradient_accumulation_steps):

            batch, data_iterator = next_batch(data_iterator, train_loader, EPOCHS)
            if batch is None:
                sftFinished = True
                break

            supervised_tokens = int(
                (batch['labels'][:, 1:] != IGNORE_INDEX).sum().item()
            )

            if supervised_tokens == 0:
                raise RuntimeError("Training micro-batch has zero supervised target tokens")

            micro_batches.append((batch, supervised_tokens))
            step_supervised_tokens += supervised_tokens

        if sftFinished:
            break
        
        step_loss_numerator = 0.0

        for batch, micro_supervised_tokens in micro_batches:
            X, y, X_attention_mask = shift_sft_batch(batch, device=device)

            del X_attention_mask

            with autocast_context(device, amp_dtype):
                output = training_model(X, tnsr_labels=y, return_logits=False)

                if output.loss is None:
                    raise RuntimeError("model did not return sft training loss")
                loss = output.loss * (micro_supervised_tokens / step_supervised_tokens)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss_numerator += output.loss.detach().item() * micro_supervised_tokens

            batch_nonpadding_tokens = int(batch["attention_mask"].sum().item())
            batch_size = int(batch['input_ids'].size(0))

            samples_processed += batch_size
            tokens_processed += batch_nonpadding_tokens
            supervised_tokens_processed += micro_supervised_tokens
            logging_tokens += batch_nonpadding_tokens
            logging_supervised_tokens += micro_supervised_tokens

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            sft_config.max_grad_norm
        )

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        completed_steps = step + 1
        step_mean_loss = step_loss_numerator / step_supervised_tokens

        if completed_steps % sft_config.log_interval == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - logging_start
            tokens_per_second = logging_tokens / elapsed
            supervised_tokens_per_second = logging_supervised_tokens / elapsed

            print(
                f"completed steps: {completed_steps} | "
                f"sft loss: {step_mean_loss:.4f} | "
                f"lr: {curr_lr:.3e}\n"
                f"grad_norm: {float(grad_norm):.3f} | "
                f"samples: {samples_processed} | "
                f"tokens: {tokens_processed} | "
                f"supervised tokens: {supervised_tokens_processed}\n"
                f"tokens per second: {tokens_per_second:,.0f} | "
                f"supervised tokens per second: {supervised_tokens_per_second:,.0f}",
                flush=True
            )

            logging_tokens = 0
            logging_supervised_tokens = 0
            logging_start = time.perf_counter()

        if validation_loader is not None and completed_steps % sft_config.eval_interval == 0:
            validation_start = time.perf_counter()
            val_loss, val_perp = evaluate(
                model,
                validation_loader,
                numEvalBatches=sft_config.eval_batches,
                amp_dtype=amp_dtype,
                device=device
            )

            print(
                f"Validation | step: {completed_steps} | sft-loss: {val_loss:.4f} | target perplexity: {val_perp:.2f}", 
                flush=True
            )


        if completed_steps % sft_config.checkpoint_interval == 0:

            checkpoint_path = checkpoint_dir / f"checkpoint_{completed_steps:08d}.pt"
            checkpoint_start = time.perf_counter()

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                completed_steps=completed_steps,
                samples_processed=samples_processed,
                tokens_processed=tokens_processed,
                supervised_tokens_processed=supervised_tokens_processed,
                model_config=model_config,
                sft_config=sft_config
            )

            logging_start += time.perf_counter() - checkpoint_start


    # Final Checkpoint
    save_checkpoint(
        path=checkpoint_dir / "checkpoint_final.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_steps=completed_steps,
        samples_processed=samples_processed,
        tokens_processed=tokens_processed,
        supervised_tokens_processed=supervised_tokens_processed,
        model_config=model_config,
        sft_config=sft_config
    )



if __name__ == "__main__":
    main()






