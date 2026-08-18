from __future__ import annotations

import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

from .model import AcaiModelConfig, AcaiTransformer

def load_model(
    checkpoint_path: Path,
    device: torch.device
) -> AcaiTransformer:

    checkpoint = torch.load( 
        checkpoint_path,
        map_location='cpu',
        weights_only=False
    )

    config = AcaiModelConfig.from_dict(checkpoint["AcaiModelConfig"])

    model = AcaiTransformer(config)

    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict, strict=True)

    model.eval()

    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
    return model


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None=None
) -> torch.Tensor:

    # Greedy
    if temperature == 0.0:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

    if temperature < 0.0:
        raise ValueError("temperature must be greater than 0")

    logits = logits.float()
    logits = logits / temperature

    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be postive")

        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(
            logits,
            k=top_k,
            dim=-1
        )

        cutoff = values[:, -1].unsqueeze(-1)

        logits = logits.masked_fill(
            logits < cutoff,
            float("-inf")
        )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        return torch.multinomial(
            probabilities,
            num_samples=1
        )



@torch.inference_mode()
def generate(
    model: AcaiTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    eos_token_id: int = 0,
    *,
    temperature: float = 0.8,
    top_k: int | None=50
) -> list[int]:

    encoding = tokenizer.encode(prompt, add_special_tokens=False)

    prompt_ids = encoding.ids

    if not prompt_ids:
        raise ValueError("Prompt must produce at least one token")

    if len(prompt_ids) > model.config.max_seq_len:
        raise ValueError(f"Prompt contains {len(prompt_ids)} tokens, but maximum sequence length allowed: {model.config.max_seq_len}")

    input_ids = torch.tensor([prompt_ids], dtype=torch.int64, device=device)


    for _ in range(max_new_tokens):
        if input_ids.size(1) >= model.config.max_seq_len:
            break

        output = model(input_ids, return_logits=True)

        if output.logits is None:
            raise RuntimeError("Model did not return logits. Inference requires return_logits=True")

        next_token_logits = output.logits[:, -1, :]

        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k
        )

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

    return input_ids[0].tolist()



def main() -> None:
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--checkpoint"
    # )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    TOKENIZER_PATH = "convoDecoderModel/tokenizer/tokenizer_llama_style.json"
    CHECKPOINT_PATH = "convoDecoderModel/checkpoints/checkpoint_00003000.pt"
    PROMPT = "What is the color of the sky?"
    MAX_NEW_TOKENS = 100
    TEMPERATURE = 2
    TOP_K = 7

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    model = load_model(CHECKPOINT_PATH, device)

    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != model.config.vocab_size:
        raise RuntimeError("tokenizer/model vocab size mismatch")

    EOS_TOKEN_ID = tokenizer.token_to_id("<|endoftext|>")

    if EOS_TOKEN_ID is None:
        raise RuntimeError("Tokenizer does not contain eos")

    generated_ids = generate(
        model,
        tokenizer,
        PROMPT,
        device=device,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        eos_token_id=EOS_TOKEN_ID
    )

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    print()
    print(text)
    print()



if __name__ == "__main__":
    main()



