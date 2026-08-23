# External evidence boundary

Lean checks the deduction *from* these premises; it does not certify the premises themselves.

## Frozen model/artifact anchors

- Apple M1 Ultra public memory-bandwidth figure: 800 GB/s.
  - https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/
- Qwen3.8-27B architecture/configuration:
  - https://huggingface.co/Qwen/Qwen3.8-27B
  - https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- Concrete GGUF scenario:
  - repository: `6block/Qwen3.8-27B-GGUF`
  - file: `Qwen3.8-27B-Q4_K_M.gguf`
  - SHA-256: `038b8d86da2e388e4c3f5bafdf5a8aa4dcb630861a18d430b3a79c538c1a3beb`
- GGML/GGUF quantization encodings:
  - https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes

## Formal premise that still needs physical justification

For each decode step, the proof assumes actual DRAM traffic is at least `dramFloor ctx cacheCarry`. `cacheCarry` is deliberately explicit because public M1 Ultra documentation is not sufficient to turn a cache-capacity number into an unconditional theorem about GPU weight residency across decode steps.

A future stronger bridge proof should derive the per-token DRAM lower bound from a concrete runtime/tensor access trace or from a sufficiently complete hardware memory model.
