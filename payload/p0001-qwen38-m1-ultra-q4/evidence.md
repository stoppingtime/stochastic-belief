# Evidence ledger — SP-P0001-Q38-M1U-Q4-v1

This file records **external facts used as premises**. It deliberately separates evidence from theorem proving: Lean checks deductions from the premises; it does not authenticate Apple specifications, Hugging Face model metadata, or a concrete GGUF tensor layout.

## E1 — M1 Ultra rated memory bandwidth

Apple's M1 Ultra announcement states that the unified-memory bandwidth is **800 GB/s** and that the chip can be configured with up to **128 GB** of unified memory.

Source: https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/

Formal use:

```text
appleRatedBps = 800000000000 byte/s
```

This is a published product specification, not a kernel theorem about sustained application bandwidth.

## E2 — Qwen3.8-27B architecture

Qwen's model card and `config.json` give the text-model dimensions used in the count:

- hidden size: 5120;
- intermediate size: 17408;
- hidden layers: 64;
- layout: 16 groups, each containing three Gated DeltaNet blocks followed by one full-attention block;
- Gated DeltaNet: 16 Q/K heads, 48 V heads, head dimension 128;
- full attention: 24 Q heads, 4 KV heads, head dimension 256;
- vocabulary size: 248320;
- untied embeddings/output head;
- native maximum context: 262144.

Sources:

- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/README.md
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json

Formal use: these values determine the parameter counts and the growing KV term.

## E3 — GGML block encodings

The llama.cpp tensor-encoding documentation records the relevant storage densities:

- Q4_K: 4.5 bits/weight, equivalently 256 weights per 144-byte superblock;
- Q6_K: 6.5625 bits/weight, equivalently 256 weights per 210-byte superblock;
- Q8_0: 32 quantized values plus one FP16 scale, i.e. 34 bytes per 32 weights;
- F32: 4 bytes/value.

Sources:

- https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes
- https://github.com/ggml-org/llama.cpp/discussions/17393

These are encoding facts. The additional claim that the selected GGUF assigns particular tensor classes to Q4_K, Q6_K, Q8_0 or F32 is a separate artifact-level premise.

## E4 — Pinned GGUF artifact

The instantiated scenario is anchored to:

```text
repository: 6block/Qwen3.8-27B-GGUF
file:       Qwen3.8-27B-Q4_K_M.gguf
SHA-256:    038b8d86da2e388e4c3f5bafdf5a8aa4dcb630861a18d430b3a79c538c1a3beb
```

The proof uses the selected recipe's tensor-class assumptions only to construct a **conservative representation-byte floor**. A future audit should independently parse the pinned GGUF and mechanically verify the tensor names, shapes and encodings against this premise.

## E5 — 819.2 GB/s raw line-rate scenario

A more permissive scenario uses LPDDR5-6400 over an aggregate 1024-bit interface:

\[
6.4\times 10^9\;\frac{\text{transfers}}{\text{s}}
\times 1024\;\frac{\text{bit}}{\text{transfer}}
\times \frac{1\;\text{byte}}{8\;\text{bit}}
=819.2\times 10^9\;\frac{\text{byte}}{\text{s}}.
\]

Secondary cross-check: ComputerBase reports M1 Ultra as LPDDR5-6400 with 819.2 GB/s theoretical bandwidth.

Source: https://www.computerbase.de/news/prozessoren/apple-silicon-m1-ultra-2x-m1-max.79884/

This line-rate scenario is intentionally **more permissive** than Apple's 800 GB/s rating. It is useful for an upper-bound argument because a larger bandwidth cap makes the resulting throughput ceiling harder, not easier, to violate.

## Evidence not promoted to theorem premises

The archive does not currently treat any published FP32 TFLOPS figure as a rigorous compute cap for Q4_K_M decoding. Quantized Metal kernels combine unpacking, integer work, dequantization, floating-point arithmetic and reductions, so a headline FP32 peak is not by itself a sound upper bound on that mixed instruction stream.

Likewise, a cache-residency number is not silently assumed. The formal model exposes `cacheCarry` explicitly. The principal results set it to zero; the 96 MiB case is marked only as a sensitivity scenario.
