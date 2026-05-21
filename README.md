# llama-cpp-mini

A small C++20 Llama inference engine that loads GGUF weights, runs a decoder-only transformer, and streams generated text on Apple Silicon.

## Highlights

- Loads `Llama-3.2-1B-Instruct-Q4_K_M.gguf` directly from GGUF.
- Current local model file: 770MB.
- Measured load time: 1056 ms.
- Measured prefill speed: 4.51 tok/s.
- Measured decode speed: 11.50 tok/s.
- Implements GGUF metadata parsing, dequantization, GQA attention, RoPE, KV cache, SwiGLU, RMSNorm, and byte-level BPE tokenization.

## Architecture

```
token IDs
   |
   v
token embedding
   |
   v
+-------------------------------+
| repeated transformer layers   |
| RMSNorm -> GQA + RoPE -> add  |
| RMSNorm -> SwiGLU MLP -> add  |
+-------------------------------+
   |
   v
final RMSNorm
   |
   v
output projection
   |
   v
logits -> sampler -> next token
```

## Demo

Prompt:

```text
The capital of France is
```

Output snippet:

```text
Paris. The Eiffel Tower is a famous landmark in Paris.
```

## Build

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```bash
./build/llama_mini -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
                   -p "The capital of France is" \
                   -n 50
```

Debug token-id mode is still available:

```bash
./build/llama_mini --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
                   --tokens "128000" \
                   --n-generate 20
```

## Performance

| Chip | Quantization | Model size | Load | Prefill | Decode |
|---|---:|---:|---:|---:|---:|
| Apple M4 | Q4_K_M | 770MB | 1056 ms | 4.51 tok/s | 11.50 tok/s |

## Implementation Notes

The tensor layer is a compact float32 row-major container with 1D to 4D indexing, reshape, fill, random initialization, and simple debug printing. The basic math path includes matmul, stable softmax, RMSNorm, and SiLU.

The GGUF loader parses the binary header, metadata key/value table, tensor table, alignment, and tensor data region. Tensors are converted to float32 as they are loaded so the forward path can stay simple.

Attention uses separate Q, K, and V projections, grouped-query attention for shared K/V heads, RoPE on Q and K, and a causal decode path that only attends to cached positions up through the current token.

The KV cache stores one K and V buffer per layer. Decoding appends one position at a time, then reuses all cached history for the next token.

The feed-forward block follows the Llama layout: RMSNorm, gate projection, up projection, SiLU-gated product, down projection, and residual add. The final norm plus output projection produces logits over the vocabulary.

The tokenizer reads `tokenizer.ggml.tokens` and `tokenizer.ggml.merges` from GGUF metadata. Encoding uses byte-level BPE, and decoding maps token pieces back to bytes before printing UTF-8 text.

## Limitations / Future Work

- Single-threaded decode.
- No batching.
- No speculative decoding.
- Limited quantization coverage: Q4_K, Q6_K, and Q8_0.
- No memory planning for very large models.
- No chat-template formatting in the CLI yet.
