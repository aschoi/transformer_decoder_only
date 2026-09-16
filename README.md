# ACAI — Decoder-Only Transformer Language Model  


ACAI is a conversational language model implemented with PyTorch. It is a **152.6M-parameter decoder-only Transformer language model**, with custom transformer components, custom pretraining and supervised fine-tuning, and autoregressive inference.   



This is end-to-end training and implementation. It is not a wrapper for existing architecture. User facing UI available at https://www.alexander-choi.com/acai  


## Model Architecture

ACAI uses a Llama-style architecture built around:  
- **Rotary Position Embeddings (RoPE)** 
- **Causal self-attention** 
- **SwiGLU feed-forward layers** 
- **RMSNorm**
- **KV-cached inference**.


## Configuration / Hyperparameters

| | |
| --- | --- |
| Parameters | 152,596,224 (127.4M non-embedding) |
| Layers | 18 |
| Model dimension | 768 |
| Attention heads | 12 (head dim 64) |
| FFN hidden dimension | 2048 |
| Context length | 2048 |
| Vocabulary | 32,768 |
| RoPE base frequency | 10,000 |  

**The implementation does not use `torch.nn.Transformer`. The rotary embeddings, decoder, attention mechanism, normalization, feed-forward network, and inference cache are implemented directly in the project.**


## Additional Features

- Mixed-precision training with automatic BF16 / FP16 selection
- Gradient accumulation and gradient clipping
- Optional gradient checkpointing
- AdamW optimization with warmup and cosine learning-rate decay
- Validation loss and perplexity evaluation
- **Resumable model and optimizer checkpoints**
- Supervised fine-tuning with masked prompt tokens
- **Resumable shuffled SFT sampling**
- Per-layer KV caches for autoregressive inference
- Cached-vs-uncached numerical correctness testing
- CPU latency benchmarking for KV-cache performance  



## Tokenizer

The pipeline uses the included `tokenizer_llama_style.json`: a byte-level BPE tokenizer with Llama-style regex pre-tokenization and six special tokens:

```text
<|endoftext|>  <|pad|>  <|system|>  <|user|>  <|assistant|>  <|end|>
```

To train a replacement tokenizer:

```bash
python -m convoDecoderModel.train_tokenizer
```

## Pretraining

- Pretraining corpus: **FineWeb-Edu**  
- Pretraining token count: **10B+**  


## Supervised Fine-tuning

- SFT dataset: **SmolTalk**  
- SFT example count: **500K+**  

## KV Cache Inference

Autoregressive generation normally becomes increasingly expensive because every generated token causes the model to recompute attention keys and values for the entire preceding sequence.  

ACAI implements a **per-layer KV cache** to avoid that redundant work.  

During prompt prefill:

```text
prompt ──► K₁,V₁ ... Kₙ,Vₙ stored in cache
```

During token-by-token decoding:

```text
new token
    │
    ├──► compute new Q
    ├──► compute only new K,V
    │
    └──► attend to cached K,V + new K,V
```

RoPE positions are offset using the current cache sequence length, allowing newly generated tokens to receive the correct absolute rotary position while previously computed keys and values are reused.

KV caching is explicitly restricted to inference and cannot be enabled during training.  


## KV Cache Benchmark

Cached and uncached generation were first checked for numerical consistency before latency testing:

```text
Cached vs. uncached logits: PASS
Prefill max |Δlogit|:       0
Decode max |Δlogit|:        2.12e-05
```

Representative results from the full **CPU / FP32** benchmark:

| Prompt tokens | Generated tokens | No cache | KV cache | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 10.39 s | 4.73 s | 2.20× | 54.5% |
| 512 | 128 | 48.19 s | 5.05 s | 9.54× | 89.5% |
| 1,024 | 128 | 91.37 s | 5.40 s | 16.94× | 94.1% |
| 1,536 | 128 | 154.43 s | 7.04 s | **21.93×** | **95.4%** |
| 128 | 512 | 132.30 s | 17.70 s | 7.47× | 86.6% |

These measurements are intended to quantify the effect of KV caching within the same implementation and hardware environment; absolute latency will vary by hardware and runtime configuration.  



## Installation

Create a Python environment and install the dependencies from the repository root:

```bash
python3 -m venv .venv_tfd
source .venv_tfd/bin/activate
python -m pip install -r requirements.txt
```

Both training entry points require a CUDA GPU and a CUDA-enabled PyTorch installation.  
Inference supports CUDA and CPU.


Run the commands below from the repository root unless otherwise indicated.  
Local chat inference uses python constants within the script.  
For UI experience visit: https://www.alexander-choi.com/acai  


### Chat inference

Edit the constants in `main()` in [inference.py](convoDecoderModel/inference.py):

```python
PROMPT = "What is gravity?"
SYSTEM_PROMPT = None
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.77
TOP_K = 50
CHECKPOINT_PATH = "convoDecoderModel/checkpoints/sft/checkpoint_final.pt"
```

Then run:

```bash
python -m convoDecoderModel.inference
```

The script formats the prompt, loads the checkpoint and tokenizer, and prints the generated completion. CUDA is selected when available; otherwise, inference runs on CPU.

Generation supports temperature sampling, top-k filtering, and greedy decoding with `TEMPERATURE = 0.0`. For the single-conversation inference path, generation stops on `<|end|>` or `<|endoftext|>`.

The formatted prompt plus the requested new tokens must fit within the model’s 2,048-token context.

Conversation prompts use this structure:

```text
<|system|>
You are a helpful assistant.<|end|>
<|user|>
What is gravity?<|end|>
<|assistant|>
```




## Project Status

ACAI is under active development.

Current work includes improving conversational response quality, and experimenting with training performance gains. 

## License

This project is licensed under the MIT License. See `LICENSE` for details.  
