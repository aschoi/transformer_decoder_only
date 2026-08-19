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
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
        model = model.to(device=device, dtype=dtype)
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
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    max_seq_len: int,
    eos_token_id: int = 0,
    temperature: float = 0.8,
    top_k: int | None=50
) -> torch.Tensor:

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape: (batch_size, seq_len)")
    
    if max_new_tokens <= 0:
        return input_ids

    model.eval()

    batch_size, prompt_length = input_ids.shape
    required_capacity = prompt_length + max_new_tokens
    if required_capacity > max_seq_len:
        raise ValueError(f"Requested sequence lengh, {required_capacity} exceeds max sequence length, {max_seq_len}")

    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype

    input_ids = input_ids.to(device)

    kv_caches = model.create_kv_caches(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        device=device,
        dtype=dtype
    )

    # PREFILL
    # Process entire prompt exactly once. 
    # Every decoder layer stores ALL prompt K/V states.
    output = model(input_ids, kv_caches=kv_caches)
    logits = output.logits
    next_token = sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        top_k=top_k
    )
    generated_ids = torch.cat(
        (input_ids, next_token),
        dim=1
    )
    # EOS directly after prompt.
    if (eos_token_id is not None 
        and batch_size == 1 
        and next_token.item() == eos_token_id
    ):
        return generated_ids

    # Decode
    for _ in range(max_new_tokens - 1):
        output = model(next_token, kv_caches=kv_caches)
        logits = output.logits
        next_token = sample_next_token(
            logits[:, -1, :], 
            temperature=temperature, 
            top_k=top_k
        )

        generated_ids = torch.cat(
            (generated_ids, next_token),
            dim=1
        )

        if (eos_token_id is not None
            and batch_size == 1
            and next_token.item() == eos_token_id
        ):
            break

    return generated_ids


def buildPrompt():
    pass



def main() -> None:
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--checkpoint"
    # )


    

    PROMPT = "Describe how to bake cookies."
    MAX_NEW_TOKENS = 100
    TEMPERATURE = 1
    TOP_K = 50
    TOKENIZER_PATH = "convoDecoderModel/tokenizer/tokenizer_llama_style.json"
    CHECKPOINT_PATH = "convoDecoderModel/checkpoints/checkpoint_00019000.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    model = load_model(CHECKPOINT_PATH, device)

    tokenizer_vocab_size = tokenizer.get_vocab_size()
    if tokenizer_vocab_size != model.config.vocab_size:
        raise RuntimeError("tokenizer/model vocab size mismatch")

    EOS_TOKEN_ID = tokenizer.token_to_id("<|endoftext|>")

    if EOS_TOKEN_ID is None:
        raise RuntimeError("Tokenizer does not contain eos")

    if len(PROMPT) == 0:
        raise ValueError("Prompt must contain at least one token.")



    encoding = tokenizer.encode(PROMPT)
    input_ids = torch.tensor(
        [encoding.ids],
        dtype=torch.int64,
        device=device
    )

    if input_ids.size(1) == 0:
        raise ValueError("Prompt must contain at least one token.")

    generated_ids = generate(
        model,
        input_ids,
        max_new_tokens=200,
        max_seq_len=model.config.max_seq_len,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        eos_token_id=EOS_TOKEN_ID
    )

    text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    print()
    print(text)
    print()

    # generated_ids = generate(
    #     model,
    #     tokenizer,
    #     PROMPT,
    #     device=device,
    #     max_new_tokens=MAX_NEW_TOKENS,
    #     temperature=TEMPERATURE,
    #     top_k=TOP_K,
    #     eos_token_id=EOS_TOKEN_ID
    # )

    # text = tokenizer.decode(
    #     generated_ids,
    #     skip_special_tokens=True
    # )

    # print()
    # print(text)
    # print()



if __name__ == "__main__":
    main()



