<!-- MODEL-ID: SP-P0002-TTNPU-Q36-CP-v1 -->
<!-- UPSTREAM-COMMIT: 7864d5dc17930667d663bbadd1ce2bc722de2753 -->
<!-- CP-DEGREE: 4 -->
<!-- GLOBAL-SEQ-LEN: 32768 -->
<!-- LOCAL-SEQ-LEN: 8192 -->
<!-- FULL-Q-HEADS: 24 -->
<!-- FULL-KV-HEADS: 4 -->
<!-- GDN-QK-HEADS: 16 -->
<!-- GDN-V-HEADS: 48 -->

# Conditional Correctness of Qwen3.6 Context Parallel in TorchTitan-NPU `override-refactor`

**Model ID: SP-P0002-TTNPU-Q36-CP-v1**

## Abstract

This manuscript studies the Context Parallel (CP) implementation used by the Qwen3.5/3.6 long-text path in the `override-refactor` branch of `cann/torchtitan-npu`. The reviewed source is frozen at commit `7864d5dc17930667d663bbadd1ce2bc722de2753`. The configuration shards a sequence of length 32768 across CP degree 4, leaving 8192 tokens on each sequence rank. It applies CP to two very different sequence operators: variable-length TND full attention and Gated DeltaNet (GDN), which combines causal convolution with recurrent state updates.

The proof does not equate “the job runs” with algorithmic correctness, and it does not infer semantic equivalence from a close loss trace. Instead, it separates three layers:

1. the **transport layer**, which changes layout from sequence sharding to head-block sharding and later applies the inverse layout change;
2. the **local-operator layer**, where each head block processes the complete global sequence;
3. the **code-to-model layer**, which asks whether DTensor, HCCL, CANN kernels, chunk boundaries, and sequence metadata actually realize the mathematical objects.

Lean 4 verifies the first layer and proves the following conditional theorem. If the collective implements a lossless rank-axis transpose, if the local fused operator implements the same head-block-local mathematical function as the non-CP baseline, and if both paths receive identical global sequence metadata, then the CP output is pointwise equal to the dense output. The local kernel is abstract, so the theorem covers causal softmax attention, variable-length attention, and a complete GDN head-block pipeline containing causal convolution, reset handling, and recurrence.

This is a conditional correctness result. It is stronger than an informal statement that the code “looks like an All-to-All,” because the round trip and the composed forward function are machine checked. It is narrower than a claim that the entire NPU training stack has been proved correct, because the semantics of DTensor/HCCL, fused kernels, metadata reconstruction, floating-point execution, and distributed backward remain external obligations.

---

## 1. External evidence and frozen source

### 1.1 Reviewed branch and files

The reviewed branch head is:

```text
override-refactor
7864d5dc17930667d663bbadd1ce2bc722de2753
```

The files directly represented by this model are:

```text
torchtitan_npu/models/qwen3_5/config_registry.py
torchtitan_npu/override/qwen3_5/parallelize.py
torchtitan_npu/override/qwen3_5/varlen_attention.py
torchtitan_npu/override/qwen3_5/gated_delta.py
```

The Python package is named `qwen3_5`, while the override description explicitly identifies the path as “Qwen3.5/3.6 Context Parallel.” This manuscript uses the user-facing name Qwen3.6 and preserves the exact source paths so that product naming is not confused with package naming.

### 1.2 Long-text configuration

The long-text configuration fixes

\[
L=32768,\qquad P=4,
\]

where \(L\) is global sequence length and \(P\) is the context-parallel degree. The initial local sequence length is therefore

\[
L_{\mathrm{local}}=\frac{32768}{4}=8192.
\]

The configuration loads both the variable-length attention CP override and the GDN CP override.

### 1.3 Transport skeleton

`sequence_to_head_shard` declares the local tensor as sequence-sharded and redistributes it to head sharding. `head_to_sequence_shard` performs the reverse redistribution.

`exchange_sequence_heads` additionally packs head chunks from several projected tensors before the collective. Packing is intended to reduce communication calls only. It must not change element ownership. The formal development therefore includes a componentwise theorem showing that transposing a packed tuple is the same as transposing each tensor independently and packing the local results afterward.

### 1.4 Full-attention path

`AscVarlenAttention.forward` follows this order:

1. expand K/V when required by the GQA case handled by the implementation;
2. exchange Q, K, and V from sequence sharding to head sharding;
3. give each head rank the complete sequence for its local head block;
4. invoke `npu_fusion_attention_v3` in TND layout;
5. exchange the result back to sequence sharding.

For the reviewed 27B model,

\[
H_Q=24,\qquad H_{KV}=4.
\]

With \(P=4\),

\[
H_Q^{\mathrm{local}}=6,\qquad H_{KV}^{\mathrm{local}}=1.
\]

Each CP rank therefore owns one aligned GQA block: one KV head serving six Q heads. Lean checks these closed integer factorizations.

### 1.5 Gated DeltaNet path

`ContextParallelGatedDeltaNet.forward` exchanges five projected quantities: Q, K, V, decay, and beta. It then:

- applies causal convolution with the local head-block convolution weights;
- derives reset boundaries from cumulative sequence lengths;
- shards `A_log` and `dt_bias` consistently with head ownership;
- runs the recurrent GDN kernel on the complete sequence;
- exchanges the output back to sequence sharding;
- combines it with the output gate that remained sequence-sharded, then applies the output projection.

The reviewed head counts are

\[
H_{QK}=16,\qquad H_V=48.
\]

At \(P=4\), the local counts are 4 and 12. They are intentionally not forced into a fictitious common scalar-head shape. The model treats the entire contiguous head chunk owned by one CP rank as an abstract payload.

---

## 2. Mathematical assumptions

The theorem does not execute Python or inspect an HCCL trace. It uses three explicit assumptions.

### Assumption A: transport is a lossless rank-axis transpose

Represent a global sequence position by

\[
(r_s,i),\qquad
r_s\in\{0,\ldots,P-1\},\quad
i\in\{0,\ldots,L_{\mathrm{local}}-1\},
\]

and a global head block by

\[
r_h\in\{0,\ldots,P-1\}.
\]

In the sequence-sharded layout, write an element as

\[
X[r_s,i,r_h].
\]

The mathematical sequence-to-head transform is

\[
(\mathcal T X)[r_h,r_s,i]=X[r_s,i,r_h]. \tag{1}
\]

Its inverse is

\[
(\mathcal T^{-1}Y)[r_s,i,r_h]=Y[r_h,r_s,i]. \tag{2}
\]

The payload at a head block may be a six-Q-head block, a one-KV-head block, or a compound GDN projection block. Assumption A says that the actual collective implements (1) without loss, duplication, reordering, or an incorrect chunk boundary.

### Assumption B: the local kernel matches the baseline head-block operator

For a head block \(r_h\), let

\[
\mathcal K_m(Q_{r_h},K_{r_h},V_{r_h})
\]

denote the local full-sequence operator. The metadata \(m\) may contain causal boundaries, cumulative sequence lengths, reset markers, and scale parameters. The operator may be exact causal attention, variable-length TND attention, or the entire GDN head-block pipeline including convolution and recurrence.

Assumption B says that the CP fused kernel and the non-CP baseline implement the same mathematical \(\mathcal K_m\) on the same inputs.

### Assumption C: metadata describes the same global segmentation

CP must not merge two packed samples into one causal segment and must not omit a recurrence reset. The formal theorem passes one identical metadata value to the dense and CP executions. The code-to-model bridge must therefore establish that `build_sequence_metadata` reconstructs the same global segment boundaries as the baseline batch.

---

## 3. General equivalence theorem

Define the dense head-separable result by

\[
Y_{\mathrm{dense}}[r_s,i,r_h]
=
\mathcal K_m
\left(
Q[\cdot,\cdot,r_h],
K[\cdot,\cdot,r_h],
V[\cdot,\cdot,r_h]
\right)[r_s,i].
\tag{3}
\]

The CP algorithm first applies \(\mathcal T\), evaluates the full-sequence operator on every local head block, and then applies \(\mathcal T^{-1}\):

\[
Y_{\mathrm{cp}}
=
\mathcal T^{-1}
\left(
r_h\mapsto
\mathcal K_m
\left(
(\mathcal TQ)[r_h,\cdot,\cdot],
(\mathcal TK)[r_h,\cdot,\cdot],
(\mathcal TV)[r_h,\cdot,\cdot]
\right)
\right).
\tag{4}
\]

### Theorem 1: transport round trip

For every tensor \(X\),

\[
\mathcal T^{-1}(\mathcal TX)=X,
\qquad
\mathcal T(\mathcal T^{-1}X)=X.
\tag{5}
\]

**Proof.** Fix \((r_s,i,r_h)\). By definition,

\[
(\mathcal T^{-1}\mathcal TX)[r_s,i,r_h]
=(\mathcal TX)[r_h,r_s,i]
=X[r_s,i,r_h].
\]

The other direction is identical. Lean applies function extensionality to the three indices, after which the proof reduces to `rfl`. ∎

### Theorem 2: CP forward equals dense forward

Under the mathematical interpretation of Assumptions A–C, for arbitrary Q, K, V and any head-separable kernel \(\mathcal K_m\),

\[
\boxed{Y_{\mathrm{cp}}=Y_{\mathrm{dense}}}. \tag{6}
\]

**Proof.** Fix any output index \((r_s,i,r_h)\). Expanding (4) and the inverse transform gives

\[
Y_{\mathrm{cp}}[r_s,i,r_h]
=
\mathcal K_m
\left(
(\mathcal TQ)[r_h,\cdot,\cdot],
(\mathcal TK)[r_h,\cdot,\cdot],
(\mathcal TV)[r_h,\cdot,\cdot]
\right)[r_s,i].
\]

Equation (1) gives

\[
(\mathcal TQ)[r_h,r'_s,i']=Q[r'_s,i',r_h],
\]

and similarly for K and V. The right-hand side is exactly (3). Since the output index was arbitrary, function extensionality yields equality of the whole tensors. In Lean, after the definitions are unfolded, the indexed equality again reduces to `rfl`. ∎

### Theorem 3: packing is semantically transparent

Let

\[
P(Q,K,V)[r_s,i,r_h]
=
(Q[r_s,i,r_h],K[r_s,i,r_h],V[r_s,i,r_h]).
\]

Then

\[
\mathcal T(P(Q,K,V))
=
P(\mathcal TQ,\mathcal TK,\mathcal TV).
\tag{7}
\]

Thus the packing performed by `exchange_sequence_heads` is semantics-preserving provided that its chunk and split boundaries match the abstract payload boundaries.

### Corollary: downstream observations are preserved

For every pure downstream function \(\Phi\), including a continuation through later layers to a scalar loss,

\[
\Phi(Y_{\mathrm{cp}})=\Phi(Y_{\mathrm{dense}}). \tag{8}
\]

Lean obtains this by congruence from (6). If both sides are interpreted as the same differentiable real-valued function, their mathematical derivatives agree wherever they exist. This manuscript does not formalize a concrete autograd engine, BF16 rounding, or distributed gradient collectives.

---

## 4. Full-attention instantiation

The payload types for Q, K, and V need not have equal head counts. At a head-block rank \(r_h\), choose:

- Q payload: six consecutive Q heads;
- K payload: one KV head;
- V payload: one KV head.

The local operator implements the GQA mapping from the one KV head to the six corresponding Q heads. The partition is aligned because

\[
24=4\times6,\qquad
4=4\times1,\qquad
6=1\times6.
\]

No fictitious physical duplication is required in the theorem. The payload is a structured block, and GQA remains inside the local-kernel specification.

Causal masking, TND layout, variable-length boundaries, and scaling are part of metadata and \(\mathcal K_m\). The transport theorem states that CP does not alter their mathematical inputs if every head block receives the same global token order as the baseline.

---

## 5. Gated DeltaNet instantiation

GDN is recurrent, so correctness cannot be justified by treating it as ordinary attention. Define one compound payload

\[
Z=(Q,K,V,a,\beta,\theta_{\mathrm{conv}},A_{\log},b_{\Delta t}),
\]

and let \(\mathcal G_m(Z)\) perform:

1. segment-aware causal convolution;
2. decay-gate construction;
3. reset-aware Gated Delta recurrence;
4. production of the full-sequence output for one head block.

The GDN CP path is the unary instance of the same theorem:

\[
\mathcal T^{-1}\bigl(\mathcal G_m(\mathcal TZ)\bigr)
=
\mathcal G_m(Z).
\tag{9}
\]

Two code-level conditions remain essential:

- `shard_local_heads` must align convolution weights, `A_log`, and `dt_bias` with the exchanged activation head block;
- cumulative sequence lengths must induce exactly the same reset positions in global coordinates as the baseline.

Lean proves that transport does not change GDN once these conditions hold. It does not replace their source-level verification.

---

## 6. Formal verification

`Proof.lean` targets Lean 4.33.0 and imports only `Init`. It checks:

- \(32768=4\times8192\);
- the 24/4 full-attention head split into 6/1 heads per CP rank;
- the 16/48 GDN head split into 4/12 heads per CP rank;
- mutual inverse properties of the rank-axis transforms;
- equivalence of packed and componentwise transport;
- CP/dense equality for arbitrary metadata and arbitrary three-input head-block kernels;
- CP/dense equality for an arbitrary unary compound kernel, used for GDN;
- preservation of arbitrary downstream pure observations.

The proof contains no `sorry`, `admit`, custom axiom, or `native_decide`. CI is expected to run:

```bash
lake build
lake env leanchecker --fresh Problems.TorchTitanNPUQwen36CP.Proof
lake env lean Problems/TorchTitanNPUQwen36CP/Proof.lean
```

and audit the declarations printed by `#print axioms`.

---

## 7. Interpretation boundary

This proof does **not** by itself establish unconditional correctness of the whole Qwen3.6 training implementation. It does not prove that:

1. DTensor redistribution is never misordered for the installed torch/torch_npu/HCCL stack;
2. actual `torch.chunk`, concatenation, and split boundaries always match the modeled payload partition;
3. `build_sequence_metadata` is correct for every packed, padded, and cross-rank sample layout;
4. `npu_fusion_attention_v3` is bitwise identical to a reference attention implementation in BF16;
5. the Triton-Ascend GDN forward, backward, checkpoint replay, and recurrent reset behavior are bitwise identical to a reference implementation;
6. FSDP, gradient scaling, optimizer updates, checkpoint saving, and reload contain no independent defect.

Those claims belong to the code-to-model bridge. A complete engineering campaign should add deterministic CP1/CP4 comparisons from the same checkpoint and global batch, intermediate-state comparisons, packed-sequence boundary tests, randomized transport inversion tests, 32K/64K loss and gradient comparisons, and save/reload continuation tests.

The formal theorem gives those tests a precise contract. When a comparison fails, the investigation can distinguish transport, metadata, local-kernel semantics, and the surrounding training pipeline instead of treating every mismatch as an undifferentiated “CP issue.”

---

## 8. Conclusion

At the reviewed commit, the Qwen3.6 CP algorithm has the abstract form

\[
\text{sequence shard}
\xrightarrow{\mathcal T}
\text{head shard with full sequence}
\xrightarrow{\mathcal K_m}
\text{head-sharded output}
\xrightarrow{\mathcal T^{-1}}
\text{sequence-sharded output}.
\]

Lean verifies the pointwise identity

\[
\boxed{
\mathcal T^{-1}\circ\mathcal K_m\circ\mathcal T
=
\mathcal K_m
}
\]

for every head-block-separable \(\mathcal K_m\), with identical metadata on both paths. Full attention and GDN are both instances of this result; their different head counts are represented by different structured payloads rather than erased.

The defensible engineering conclusion is therefore:

> The communication skeleton of Qwen3.6 CP in `override-refactor` is mathematically correct under the stated transport, kernel, and metadata premises. The remaining correctness risk lies in the code-to-model correspondence, especially DTensor chunk boundaries, variable-length metadata, fused-kernel semantics, and the backward/training path.
