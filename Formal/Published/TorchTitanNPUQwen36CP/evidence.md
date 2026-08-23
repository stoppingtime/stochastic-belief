# Evidence ledger — TorchTitan-NPU Qwen3.6 context parallelism

**Model ID:** `SP-P0002-TTNPU-Q36-CP-v1`

This ledger separates implementation evidence from propositions proved in Lean. A source-code observation is not silently promoted to a theorem; conversely, a Lean theorem is not presented as proof that a Python collective or an NPU kernel implements the modeled operation.

## Frozen revisions

| Item | Frozen value |
|---|---|
| TorchTitan-NPU public mirror | `botcanlearn/torchtitan-npu-upstream` |
| Upstream project | `cann/torchtitan-npu` |
| Branch | `override-refactor` |
| Commit | `7864d5dc17930667d663bbadd1ce2bc722de2753` |
| Git tree | `a2083d3d601007d3a47a7200022d55ee89f90608` |
| TorchTitan dependency | `c91448d20480c7b294314e68976823050002ebec` |
| Lean toolchain | `leanprover/lean4:v4.33.0` |

The branch name alone is not an evidence anchor because it can move. Every claim below is scoped to the two full commit identifiers above.

## Source-to-model correspondence

### 1. Sequence/head redistribution

Source: `torchtitan_npu/override/qwen3_5/parallelize.py`, blob `1600dbe56d3842c480319c8285c6abed135e1d8d`.

Relevant functions:

- `sequence_to_head_shard`: interprets the local tensor as sequence-sharded (`Shard(1)`) and redistributes it to a shard on `head_dim`;
- `head_to_sequence_shard`: performs the reverse redistribution;
- `exchange_sequence_heads`: splits every projected tensor by head shard, packs rank-major pieces, performs the sequence-to-head redistribution, and splits the result back into the original projected tensors.

Mathematical abstraction: a coordinate transposition

\[
(	ext{sequence rank},\text{local sequence},\text{head rank},\text{local head})
\longleftrightarrow
(	ext{head rank},\text{sequence rank},\text{local sequence},\text{local head}).
\]

Lean proves that the two coordinate maps are mutual inverses. The bridge assumption is that DTensor/HCCL realizes this coordinate map exactly at the logical tensor level.

### 2. Qwen3.6 Gated DeltaNet CP path

Source: `torchtitan_npu/override/qwen3_5/gated_delta.py`, blob `35eaec0d1032938d15ee3229f7aa657a6d0726b8`.

`ContextParallelGatedDeltaNet.forward` computes local projections for Q, K, V, decay and beta, calls `exchange_sequence_heads`, applies causal convolution and the GDN kernel to the full sequence for locally owned heads, then calls `head_to_sequence_shard` before the output projection. The convolution multiplies a delayed source only when the segment identifiers agree, so a delayed tap is suppressed at a document boundary.

Mathematical abstraction: an arbitrary exact operator on one complete head sequence. Lean does not need to know the internal recurrence; it proves that every operator with no cross-head reads commutes with the coordinate transposition.

External obligation: the Triton-Ascend GDN kernel and its causal convolution must implement the intended per-head operator, including resets, dtype conversions and autograd.

### 3. Full-attention CP path

Source: `torchtitan_npu/override/qwen3_5/varlen_attention.py`, blob `ac81ee49016efefb6bef75b71b6eb7aa7ecadee1`.

`AscVarlenAttention.forward` optionally expands K/V heads for GQA, exchanges Q/K/V from sequence sharding to head sharding, calls `npu_fusion_attention_v3` in TND layout with per-segment cumulative lengths, and transforms the result back to sequence sharding. Its override description explicitly states “Qwen3.5/3.6 Context Parallel.”

Mathematical abstraction: a head-local causal-attention operator on the complete logical sequence. The formal theorem covers GQA after the implementation has produced the per-query-head K/V association; it does not prove the correctness of that repetition or of the NPU fused-attention kernel.

### 4. Variable-length CP metadata

Sources:

- `torchtitan_npu/patches/torchtitan/distributed/context_parallel.py`, blob `23c5538ca812ff34f330c8fde232568c7adb65cc`;
- `torchtitan_npu/patches/torchtitan/distributed/varlen_cp.py`, blob `6b1230df171155954a8cf128f97f581b4d45fb82`;
- `torchtitan_npu/override/qwen3_5/parallelize.py`, same blob as section 1.

`CPVarlenMetadata.from_global` maps rank-local Q positions back to global packed positions, splits local runs when the document ID changes or positions cease to be contiguous, and assigns each run a K/V prefix from the document start through the run stop. `build_sequence_metadata` later gathers reset markers and treats a local run as a true document start when its local Q length equals its causal K-prefix length.

The Lean model represents a local run by three global coordinates:

\[
d=	ext{document start},\qquad a=	ext{local run start},\qquad b=	ext{exclusive run stop},
\]

with \(d\le a<b\). It proves

\[
b-a=b-d \iff a=d,
\]

and proves that the generated indices \(d+j\), for \(0\le j<b-d\), are exactly the integers in \([d,b)\). These two statements are the arithmetic core of reset detection and causal-prefix construction.

### 5. Concrete long-text configuration

Source: `torchtitan_npu/models/qwen3_5/config_registry.py`, blob `2448b69407135cfb913ee2fe194f029776c1c9ec`.

`qwen35_27b_long_text_sft` fixes:

- sequence length `32768`;
- context-parallel degree `4`;
- `context_parallel_load_balancer = None`;
- varlen attention;
- the Qwen CP attention and GDN overrides above.

The names are `qwen3_5` because the upstream implementation is shared by the Qwen3.5/3.6 family. This publication uses “Qwen3.6” in the operational sense already used by the override’s attention description and the user-facing integration context; it does not claim that the Python package has a separate `qwen3_6` directory.

### 6. Dependency pin

Source: `requirements.txt`, blob `1b4f49f86267fbff25f4643105211051756fcb34`.

The plugin pins TorchTitan to `c91448d20480c7b294314e68976823050002ebec`. The corresponding upstream commit is independently addressable in `pytorch/torchtitan`. This matters because the override hooks, model configuration types and DTensor calls are version-sensitive.

## What is machine-checked

`Proof.lean` checks, using Lean Core only:

1. both redistribution maps are inverse functions;
2. every tensor entry survives a round trip at the same logical coordinate;
3. every head-separable exact operator commutes with exchange and restore;
4. token-local maps commute with both layout views;
5. the reset predicate is equivalent to starting at the document boundary;
6. the generated causal K-prefix is sound and complete;
7. the concrete CP=4 sequence/head divisibility facts.

The CI additionally runs `lake build`, `leanchecker --fresh`, and an axiom audit.

## What remains an engineering obligation

The formal result does **not** establish any of the following without additional evidence:

- DTensor or HCCL implements the modeled permutation without a runtime bug;
- `npu_fusion_attention_v3` computes exact causal attention;
- the Triton-Ascend GDN kernel computes the intended recurrence and backward pass;
- BF16 outputs, gradients or optimizer states are bitwise equal to a non-CP run;
- checkpoint save/load, optimizer stepping, loss scaling or distributed liveness is correct;
- a particular training log demonstrates long-run convergence.

Those claims require executable tests, numerical comparisons and, for kernel-level assurance, separate formal or implementation verification. This publication deliberately stops at the exact semantic invariant that the CP layout transformation must preserve.

## Source links

- Branch snapshot: `https://github.com/botcanlearn/torchtitan-npu-upstream/tree/7864d5dc17930667d663bbadd1ce2bc722de2753`
- Parallelization: `https://github.com/botcanlearn/torchtitan-npu-upstream/blob/7864d5dc17930667d663bbadd1ce2bc722de2753/torchtitan_npu/override/qwen3_5/parallelize.py`
- Gated DeltaNet: `https://github.com/botcanlearn/torchtitan-npu-upstream/blob/7864d5dc17930667d663bbadd1ce2bc722de2753/torchtitan_npu/override/qwen3_5/gated_delta.py`
- Varlen attention: `https://github.com/botcanlearn/torchtitan-npu-upstream/blob/7864d5dc17930667d663bbadd1ce2bc722de2753/torchtitan_npu/override/qwen3_5/varlen_attention.py`
- Varlen CP metadata: `https://github.com/botcanlearn/torchtitan-npu-upstream/blob/7864d5dc17930667d663bbadd1ce2bc722de2753/torchtitan_npu/patches/torchtitan/distributed/varlen_cp.py`
- TorchTitan pin: `https://github.com/pytorch/torchtitan/commit/c91448d20480c7b294314e68976823050002ebec`
