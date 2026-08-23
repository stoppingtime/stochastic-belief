# Evidence ledger — P0002 Qwen3.6 Context Parallel

## Frozen review target

- Canonical upstream: `cann/torchtitan-npu`
- Public review mirror: `botcanlearn/torchtitan-npu-upstream`
- Branch: `override-refactor`
- Commit: `7864d5dc17930667d663bbadd1ce2bc722de2753`
- Review date: 2026-08-24

The mirror branch metadata identifies `7864d5dc17930667d663bbadd1ce2bc722de2753` as the current
`override-refactor` head used by this archive. This commit is a provenance
anchor, not a claim that every future branch head has the same behavior.

## Source files represented by the model

| File | Frozen Git blob SHA | Model role |
|---|---|---|
| `torchtitan_npu/models/qwen3_5/config_registry.py` | `2448b69407135cfb913ee2fe194f029776c1c9ec` | selects the 27B long-text path, sequence length 32768, CP degree 4, and both CP overrides |
| `torchtitan_npu/override/qwen3_5/parallelize.py` | `1600dbe56d3842c480319c8285c6abed135e1d8d` | defines sequence/head redistribution, multi-tensor packing, and global sequence metadata preparation |
| `torchtitan_npu/override/qwen3_5/varlen_attention.py` | `ac81ee49016efefb6bef75b71b6eb7aa7ecadee1` | exchanges Q/K/V, invokes TND fused attention, and restores sequence sharding |
| `torchtitan_npu/override/qwen3_5/gated_delta.py` | `35eaec0d1032938d15ee3229f7aa657a6d0726b8` | exchanges projected GDN quantities, shards head-local parameters, applies causal convolution and recurrence, then restores sequence sharding |

Review URLs:

- https://github.com/botcanlearn/torchtitan-npu-upstream/tree/override-refactor
- https://github.com/botcanlearn/torchtitan-npu-upstream/blob/override-refactor/torchtitan_npu/models/qwen3_5/config_registry.py
- https://github.com/botcanlearn/torchtitan-npu-upstream/blob/override-refactor/torchtitan_npu/override/qwen3_5/parallelize.py
- https://github.com/botcanlearn/torchtitan-npu-upstream/blob/override-refactor/torchtitan_npu/override/qwen3_5/varlen_attention.py
- https://github.com/botcanlearn/torchtitan-npu-upstream/blob/override-refactor/torchtitan_npu/override/qwen3_5/gated_delta.py

The project documentation for custom CP explains the same architectural idea:
sequence-sharded activations are transformed to head sharding before the
attention computation and transformed back afterward.

- https://gitcode.com/cann/torchtitan-npu/blob/master/docs/feature_guides/parallelism/custom_cp.md

## Extracted external facts

The long-text registry supplies:

\[
L=32768,\qquad P=4,\qquad L_{local}=8192.
\]

The reviewed Qwen3.6 model shape used in the manuscripts is:

\[
H_Q=24,\quad H_{KV}=4,\quad H_{GDN,QK}=16,\quad H_{GDN,V}=48.
\]

The formal proof checks the corresponding CP=4 factorizations:

\[
24=4\cdot6,\qquad 4=4\cdot1,\qquad
16=4\cdot4,\qquad 48=4\cdot12.
\]

## Code-to-model correspondence

The Lean transform

\[
(\mathcal T X)[r_h,r_s,i]=X[r_s,i,r_h]
\]

models the intended semantics of changing DTensor placement from sequence
sharding to head sharding. The Python implementation contains additional
packing, concatenation, physical layout, dtype conversion, and collective
details. The formal theorem applies to the implementation only if those
details realize the same element map.

The local Lean kernel is intentionally abstract. For full attention it stands
for the causal/varlen GQA operator on one complete-sequence head block. For GDN
it stands for the whole head-block-local pipeline, including causal
convolution, reset handling, decay, beta, and recurrent state updates.

## External/formal boundary

Lean verifies:

- exact CP=4 integer factorizations;
- transport round-trip;
- packed/componentwise transport equivalence;
- equality of CP and dense compositions for every head-block-separable local
  kernel receiving the same metadata;
- equality of arbitrary downstream pure observations.

Lean does **not** verify:

- GitHub or GitCode content authenticity beyond the recorded hashes;
- DTensor or HCCL implementation correctness;
- CANN fused-attention numerical semantics;
- Triton-Ascend GDN numerical and backward semantics;
- the runtime correctness of `build_sequence_metadata`;
- FSDP, optimizer, checkpoint, or weight-save behavior.

Those are external premises and must be tested or separately formalized. The
manuscripts keep this boundary explicit so the formal result cannot be
misreported as an unconditional proof of the entire training stack.
