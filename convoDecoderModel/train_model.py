


from __future__ import annotations

import argparse
import contextlib
import inspect
import math
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from datatrove.utils.dataset import DatatroveFolderDataset

try:
    from model import AcaiModelConfig, AcaiTransformer
except ModuleNotFoundError:
    from .model import AcaiModelConfig, AcaiTransformer


@dataclass(frozen=True, slots=True)
class TrainConfig:
    train_data_dir: str
    output_dir: str
    checkpoint_dir: str
    max_tokens: int
    max_steps: int | None
    micro_batch_size: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    validation_data: str | None=None
    gradient_accumulation_steps: int = 16
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0
    precision: str = "auto"
    num_workers: int = 2
    log_interval: int = 10
    eval_interval: int = 500
    eval_batches: int = 50
    checkpoint_interval: int = 1_000
    seed: int = 173
    token_size: int | None=None
    compile_model: bool=False
    gradient_checkpointing: bool=False
    resume: str | None=None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_precision(name, device):
    if device.type != "cuda":
        if name not in {"auto", 'fp32'}:
            raise ValueError("nope")
        return None
    if name == "auto":
        return (
            torch.bfloat16 
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    if name == 'bf16':
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("nope")
        return torch.bfloat16
    if name =='fp16':
        return torch.float16
    return None

def autocast_context(device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def get_lr(
    step: int,
    *, 
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float
) -> float:

    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps

    if step >= max_steps:
        return min_lr

    decay_ratio = ((step - warmup_steps) / (max_steps - warmup_steps))

    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coefficient * (max_lr - min_lr)


class SequentialOffsetSampler(Sampler[int]):
    def __init__(self, dataset_size, start_index):
        self.dataset_size = dataset_size
        self.set_start_index(start_index)

    def set_start_index(self, start_index):
        if not 0 <= start_index <= self.dataset_size:
            raise ValueError("nope nope")
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start_index, self.dataset_size))

    def __len__(self) -> int:
        return self.dataset_size - self.start_index


def make_data_loader(
        data_dir,
        *,
        model_config,
        train_config,
        training,
        samples_already_processed
):
    dataset = DatatroveFolderDataset(
        data_folder=data_dir,
        seq_len=model_config.max_seq_len,
        token_size=train_config.token_size,
        shuffle=training,
        seed=train_config.seed,
        return_positions=False
    )
    if len(dataset) == 0:
        raise ValueError("no full seq")

    usable_samples = len(dataset)
    sampler: SequentialOffsetSampler | None=None
    if training:
        usable_samples -= usable_samples % train_config.micro_batch_size
        if usable_samples == 0:
            raise ValueError("not enough usable samples")
        sampler = SequentialOffsetSampler(usable_samples, start_index=samples_already_processed % usable_samples)

    loader = DataLoader(
        dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=train_config.num_workers,
        pin_memory=True,
        persistent_workers=train_config.num_workers > 0,
        drop_last=training
    )

    return loader



def next_batch(
        iterator: Iterator[dict[str, torch.Tensor]],
        loader: DataLoader
):
    try:
        return next(iterator), iterator
    except StopIteration:
        if isinstance(loader.sampler, SequentialOffsetSampler):
            loader.sampler.set_start_index(0)
        iterator = iter(loader)
        return next(iterator), iterator


def save_checkpoint(
    *,
    path: Path,
    model: AcaiTransformer,
    optimizer: torch.optim.Optimizer,
    scaler,
    completed_steps,
    tokens_processed,
    model_config: AcaiModelConfig,
    train_config: TrainConfig
) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "completed_steps": completed_steps,
        "tokens_processed": tokens_processed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "AcaiModelConfig": model_config.to_dict(),
        "trainConfig": asdict(train_config),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
    }

    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)


def load_checkpoint(
    *,
    path,
    model,
    optimizer,
    scaler,
    expected_model_config
):
    checkpoint_state = torch.load(path, map_location='cpu', weights_only=False)
    saved_model_config = AcaiModelConfig.from_dict(checkpoint_state["AcaiModelConfig"])
    if saved_model_config != expected_model_config:
        raise ValueError("Checkpoint model config does not match this run")

    model.load_state_dict(checkpoint_state["model"])
    optimizer.load_state_dict(checkpoint_state["optimizer"])
    if "scaler" in checkpoint_state:
        scaler.load_state_dict(checkpoint_state["scaler"])
        torch.set_rng_state(checkpoint_state["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint_state["cuda_rng_state"])
    return (
        int(checkpoint_state["completed_steps"]),
        int(checkpoint_state.get("tokens_processed", 0))
    )


def build_optimizer(
        model,
        train_config
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
            "weight_decay": train_config.weight_decay
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0
        }
    ]
    optimizer_arguments = {
        "lr": train_config.lr,
        "betas": (train_config.beta1, train_config.beta2)
    }
    if "fused" in inspect.signature(torch.optim.AdamW).parameters:
        optimizer_arguments["fused"] = True
    return torch.optim.AdamW(parameter_groups, **optimizer_arguments)





@torch.no_grad()
def evaluate(
    model,
    loader,
    *,
    eval_batches,
    amp_dtype,
    device
) -> tuple[float, float]:

    was_training = model.training
    model.eval()
    total_loss = 0.0
    evaluated_tokens = 0

    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= eval_batches:
                break
            tokens = batch["input_ids"].to(device, non_blocking=True)
            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]
            with autocast_context(device, amp_dtype):
                output = model(input_ids, tnsr_labels=labels, return_logits=False)
            if output.loss is None:
                raise RuntimeError("Model did not return validation loss")
            token_count = labels.numel()
            total_loss += output.loss.item() * token_count
            evaluated_tokens += token_count
    finally:
        model.train(was_training)
    
    if evaluated_tokens == 0:
        raise RuntimeError("validation loader yielded no batches")
    mean_loss = total_loss / evaluated_tokens
    return mean_loss, math.exp(min(mean_loss, 20.0))




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


    # Training Configuration
    train_config = TrainConfig(
        train_data_dir="data/tokenized/train",
        checkpoint_dir="checkpoints",
        output_dir="output",
        validation_data="data/tokenized/validation",
        max_tokens=10_000_000_000,
        max_steps=100,
        micro_batch_size=4,
        lr=3.0e-4,
        min_lr=3.0e-5,
        warmup_steps=20,
        weight_decay=0.1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=16,
        beta1=0.9,
        beta2=0.95,
        precision="auto",
        num_workers=2,
        log_interval=1,
        eval_interval=25,
        eval_batches=25,
        checkpoint_interval=50,
        seed=717,
        token_size=2,
        compile_model=False,
        gradient_checkpointing=False,
    )

    set_seed(train_config.seed)
    device = torch.device("cuda")

    torch.set_float32_matmul_precision("high")
    amp_dtype  =resolve_precision(train_config.precision, device)
    use_grad_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)



    # Model
    model = AcaiTransformer(model_config).to(device)
    if train_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = build_optimizer(model, train_config)

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {parameter_count}")



    # microBatchSize * maxSeqLen
    tokens_per_microBatch = train_config.micro_batch_size * model_config.max_seq_len

    # tokensPerMicrobatch * gradAccSteps
    tokens_per_optimizerStep = tokens_per_microBatch * train_config.gradient_accumulation_steps

    if train_config.max_steps is not None:
        total_steps = train_config.max_steps
    else:
        total_steps = math.ceil(train_config.max_tokens / tokens_per_optimizerStep)

    start_step = 0
    tokens_processed = 0
    print(f"Token per micro batch: {tokens_per_microBatch}")
    print(f"tokens per optimizer step: {tokens_per_optimizerStep}")
    print(f"optimizer steps: {total_steps}")


    if train_config.resume is not None:
        start_step, tokens_processed = load_checkpoint(
            path=Path(train_config.resume),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_model_config=model_config
        )

    samples_processed = tokens_processed // model_config.max_seq_len    
    train_loader = make_data_loader(
        train_config.train_data_dir,
        model_config=model_config,
        train_config=train_config,
        training=True,
        samples_already_processed=samples_processed
    )

    validation_loader = None
    if train_config.validation_data is not None:
        validation_loader = make_data_loader(
            train_config.validation_data,
            model_config=model_config,
            train_config=train_config,
            training=False,
            samples_already_processed=0
        )

    output_dir = Path(train_config.output_dir)
    checkpoint_dir = Path(train_config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_config.save_json(output_dir/"model_config.json")
    with (output_dir / "train_config.json").open('w', encoding="utf-8") as f:
        json.dump(asdict(train_config), f, indent=4, sort_keys=True)
        f.write("\n")

    training_model = torch.compile(model) if train_config.compile_model else model


    # TRAINING
    model.train()
    data_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    logging_tokens = 0
    logging_start = time.perf_counter()

    for step in range(start_step, total_steps):
        curr_lr = get_lr(
            step,
            max_steps=total_steps,
            warmup_steps=train_config.warmup_steps,
            max_lr=train_config.lr,
            min_lr=train_config.min_lr
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        # Gradient Accumulation
        for _ in range(train_config.gradient_accumulation_steps):
            batch, data_iterator = next_batch(data_iterator, train_loader)

            tokens = batch["input_ids"].to(device, non_blocking=True)
            X = tokens[:, :-1]
            y = tokens[:, 1:]

            with autocast_context(device, amp_dtype):

                output = training_model(X, tnsr_labels=y, return_logits=False)
                if output.loss is None:
                    raise RuntimeError("oops")
                
                loss = output.loss / train_config.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += output.loss.detach().item()
            batch_tokens = X.numel()
            tokens_processed += batch_tokens
            logging_tokens += batch_tokens

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

        # Optimizer 
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()


        completed_steps = step + 1
        # Logging
        if completed_steps % train_config.log_interval == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - logging_start
            tokens_per_second = logging_tokens / elapsed
            mean_loss = accumulated_loss / train_config.gradient_accumulation_steps
            print(
                f"completed steps: {completed_steps} | "
                f"loss: {mean_loss:.4f} | "
                f"lr: {curr_lr:.3e}\n"
                f"grad_norm: {grad_norm:.3f} | "
                f"tokens: {tokens_processed} | "
                f"tokens per second: {tokens_per_second:,.0f}",
                flush=True
            )

            logging_tokens = 0
            logging_start = time.perf_counter()

        if validation_loader is not None and completed_steps % train_config.eval_interval == 0:
            val_loss, val_perplexity = evaluate(
                model,
                validation_loader,
                eval_batches=train_config.eval_batches,
                amp_dtype=amp_dtype,
                device=device
            )
            print(
                f"validation | "
                f"step: {completed_steps} | "
                f"loss: {val_loss:.4f} | "
                f"perplexity: {val_perplexity:.2f} | ",
                flush=True
            )

        # Checkpointing
        if completed_steps % train_config.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{completed_steps:08d}.pt"
            checkpoint_start = time.perf_counter()
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                completed_steps=completed_steps,
                tokens_processed=tokens_processed,
                model_config=model_config,
                train_config=train_config
            )
            logging_start += time.perf_counter() - checkpoint_start


    # Final Checkpoint
    save_checkpoint(
        path=checkpoint_dir / "checkpoint_final.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        completed_steps=total_steps,
        tokens_processed=tokens_processed,
        model_config=model_config,
        train_config=train_config
    )


if __name__ == "__main__":
    main()







