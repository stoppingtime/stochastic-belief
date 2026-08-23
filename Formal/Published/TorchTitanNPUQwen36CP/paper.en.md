<!-- MODEL-ID: SP-P0002-TTNPU-Q36-CP-v1 -->
<!-- OVERRIDE-COMMIT: 7864d5dc17930667d663bbadd1ce2bc722de2753 -->
<!-- TORCHTITAN-COMMIT: c91448d20480c7b294314e68976823050002ebec -->
<!-- CP-DEGREE: 4 -->
<!-- GLOBAL-SEQUENCE-LENGTH: 32768 -->
<!-- LOCAL-SEQUENCE-LENGTH: 8192 -->
<!-- FULL-ATTN-Q-HEADS: 24 -->
<!-- FULL-ATTN-KV-HEADS: 4 -->
<!-- GDN-QK-HEADS: 16 -->
<!-- GDN-V-HEADS: 48 -->

# Semantic Correctness of Qwen3.6 Context Parallelism in TorchTitan-NPU `override-refactor`

**Model ID: `SP-P0002-TTNPU-Q36-CP-v1`**

## Abstract

This paper isolates and formally verifies the discrete semantic invariant behind the Qwen3.6 context-parallel path in a fixed revision of TorchTitan-NPU. In the implementation, the input initially appears as contiguous sequence shards: each context-parallel rank owns a portion of the sequence and all heads. Projected Q, K, V and Gated DeltaNet tensors are then redistributed so that a rank owns a subset of heads and the complete logical sequence. Full attention or Gated DeltaNet is evaluated in that view, after which the result is redistributed back to sequence sharding.

The implementation evidence is frozen at TorchTitan-NPU mirror commit `7864d5dc17930667d663bbadd1ce2bc722de2753` on `override-refactor`, together with the TorchTitan dependency commit `c91448d20480c7b294314e68976823050002ebec`. The repository uses the path `qwen3_5` for the implementation shared by the Qwen3.5/3.6 family; the attention override itself describes its target as “Qwen3.5/3.6 Context Parallel.” The proof is therefore tied to exact commits and file blobs rather than to a moving branch name or an ambiguous product label.

We model the redistribution as a transposition of four finite coordinates,

\[
(s,\ell,h,r)\mapsto(h,s,\ell,r),
\]

where \(s\) is the sequence-rank coordinate, \(\ell\) a local sequence position, \(h\) the head-rank coordinate, and \(r\) a local head. Lean 4 proves that the forward and reverse transpositions are mutual inverses. Consequently no logical entry can be lost, duplicated, or returned to the wrong coordinate in the abstract model.

The central theorem is stronger than a round-trip check. Let \(F\) be any exact operator that consumes the complete sequence for one head, returns a complete sequence for that head, and has no dependency on other heads. Then

\[
T^{-1}\!\left(F_{\mathrm{head}}(Tx)\right)=F_{\mathrm{reference}}(x).
\]

No linearity or finite-window assumption is required. The result therefore covers, at the layout-semantic level, both causal full attention and a recurrent Gated DeltaNet operator. Token-local projections are handled by separate commutation lemmas.

Packed variable-length training introduces a second indexing obligation. For a contiguous local run, let \(d\) be the document start, \(a\) the run start, and \(b\) its exclusive stop, with \(d\le a<b\). The implementation identifies a true document reset by comparing the local query length \(b-a\) with the causal key-prefix length \(b-d\). Lean proves

\[
b-a=b-d\quad\Longleftrightarrow\quad a=d.
\]

It also proves that the generated key indices \(d+j\), for \(0\le j<b-d\), are exactly the half-open interval \([d,b)\). Thus the abstract metadata rule neither crosses a document boundary nor omits a key in the causal prefix.

The conclusion is explicitly conditional. Lean does not verify that DTensor/HCCL realizes the modeled permutation, that CANN or Triton kernels implement the declared head-local functions, or that BF16, autograd, optimizer, checkpoint, and training-loop behavior are correct. It verifies the deduction from those implementation contracts. This boundary is essential: a proof of the layout invariant is valuable precisely because it is stated narrowly enough to be true and testable.

---

## 1. Scope and frozen evidence

### 1.1 Versioned object of study

The following revisions define the object of study:

| Object | Frozen value |
|---|---|
| TorchTitan-NPU mirror | `botcanlearn/torchtitan-npu-upstream` |
| Branch name | `override-refactor` |
| Commit | `7864d5dc17930667d663bbadd1ce2bc722de2753` |
| Git tree | `a2083d3d601007d3a47a7200022d55ee89f90608` |
| TorchTitan dependency | `c91448d20480c7b294314e68976823050002ebec` |
| Proof checker | Lean 4.33.0, Core `Init` only |

Every implementation statement in this paper can be traced to a path and blob SHA in [`evidence.md`](evidence.md). A branch name is retained for orientation, but the commit is the actual evidence anchor.

### 1.2 Concrete long-text configuration

The frozen `qwen35_27b_long_text_sft` configuration selects

\[
P=4,\qquad L=32768,\qquad S=L/P=8192,
\]

uses variable-length attention, disables the context-parallel load balancer, and imports the Qwen CP full-attention and GDN overrides. The proof therefore treats contiguous equal-length sequence shards. The metadata code has a broader head-tail path, but that path is not enabled by the concrete configuration and is not needed for the main result.

### 1.3 Claims separated by layer

The proof has three independent components.

**Layout correctness.** The sequence-sharded and head-sharded views are connected by a bijection, with an explicit inverse.

**Operator correctness under a contract.** Every exact head-separable operator commutes with the view change.

**Packed-sequence metadata correctness.** Reset detection and causal-prefix enumeration are exact for each contiguous local run.

The result does not claim to verify Python, a collective implementation, or a hardware kernel. Those systems must satisfy the contracts stated by the model.

---

## 2. From implementation to mathematical object

### 2.1 The redistribution path

In `torchtitan_npu/override/qwen3_5/parallelize.py`, `sequence_to_head_shard` wraps a local tensor as a DTensor with placement `Shard(1)` and redistributes it to `Shard(head_dim)`. `head_to_sequence_shard` performs the reverse operation. `exchange_sequence_heads` first splits each projected tensor along its head dimension, packs pieces in rank order, runs the redistribution, and splits the packed result back into Q, K, V, decay, beta, or the corresponding attention tensors.

At the logical level, the operation changes ownership rather than values:

- before exchange, a rank owns a sequence interval and all head groups;
- after exchange, a rank owns one head group and all sequence intervals;
- the tensor identity is preserved across the temporary packing;
- reverse exchange restores sequence ownership.

The proof models this ownership change directly. It does not model temporary buffer addresses or collective scheduling, because those are implementation choices rather than semantic coordinates.

### 2.2 Why attention and GDN share one theorem

Full attention and Gated DeltaNet differ radically along the sequence axis. Attention forms weighted causal combinations; GDN applies causal convolution and a gated recurrence. For context parallelism, however, both satisfy the same structural interface: one output head is determined by the complete logical sequence associated with that head, and the computation does not read a different output head.

This is the only property needed by the commutation theorem. The per-head function may be nonlinear, recurrent, stateful along the sequence, or sensitive to document resets. This abstraction avoids a common proof mistake: proving a special case for a linear toy operator and then assuming the same argument covers a recurrence.

### 2.3 Token-local maps

The input projections and the final pointwise or linear maps are evaluated independently at each token/head coordinate. For any pointwise map \(f\), coordinate transposition satisfies

\[
T(f(x))=f(T(x)).
\]

Lean records this in two dedicated lemmas rather than burying it in the main theorem. The complete pipeline can therefore be decomposed into token-local maps, an exchange, a head-local sequence operator, the inverse exchange, and token-local output maps.

---

## 3. Finite coordinate model

Let \(P\) be the context-parallel degree, \(S\) the local sequence length, and \(H\) the number of local heads in one head shard. Let \(\alpha\) be the element type.

The sequence view is the function space

\[
X:\operatorname{Fin}(P)\times\operatorname{Fin}(S)
\times\operatorname{Fin}(P)\times\operatorname{Fin}(H)\to\alpha.
\]

We write its coordinates as \((s,\ell,h,r)\): sequence rank, local sequence position, head rank, and local head.

The head view contains exactly the same logical entries but exposes coordinates as

\[
Y:\operatorname{Fin}(P)\times\operatorname{Fin}(P)
\times\operatorname{Fin}(S)\times\operatorname{Fin}(H)\to\alpha,
\]

written \((h,s,\ell,r)\).

Define

\[
(TX)(h,s,\ell,r)=X(s,\ell,h,r), \tag{1}
\]

and

\[
(T^{-1}Y)(s,\ell,h,r)=Y(h,s,\ell,r). \tag{2}
\]

Using `Fin` is more than notation. An out-of-range rank, sequence position, or head cannot inhabit the corresponding coordinate type. Bounds therefore appear in the statement rather than as unchecked integer conventions.

---

## 4. Bijection of the two layouts

### Theorem 1 — sequence-view round trip

For every sequence-layout tensor \(X\),

\[
T^{-1}(TX)=X. \tag{3}
\]

### Proof

Fix an arbitrary legal coordinate \((s,\ell,h,r)\). By (2),

\[
(T^{-1}(TX))(s,\ell,h,r)=(TX)(h,s,\ell,r).
\]

Equation (1) reduces the right-hand side to \(X(s,\ell,h,r)\). The functions agree at every coordinate, so function extensionality gives (3). ∎

### Theorem 2 — head-view round trip

For every head-layout tensor \(Y\),

\[
T(T^{-1}Y)=Y. \tag{4}
\]

The proof is symmetric.

These two statements establish a genuine bijection. Equality of element counts would be insufficient: a faulty exchange could duplicate one entry while dropping another and still preserve the total count. Mutual inverse functions rule out that failure mode.

The Lean declarations are:

```lean
headToSequence_sequenceToHead
sequenceToHead_headToSequence
roundTrip_preserves_entry
```

The first two prove function equality; the third exposes the invariant at an arbitrary single coordinate.

---

## 5. Commutation with head-separable operators

### 5.1 Sequence fiber of one head

Fix a head coordinate \((h,r)\). Its complete logical sequence is

\[
x_{h,r}(s,\ell)=X(s,\ell,h,r),
\]

which is a function on \(\operatorname{Fin}(P)\times\operatorname{Fin}(S)\).

Let

\[
F:(\operatorname{Fin}(P)\times\operatorname{Fin}(S)\to\alpha)
\to
(\operatorname{Fin}(P)\times\operatorname{Fin}(S)\to\beta)
\]

be any exact operator on one such fiber. No assumption of linearity, locality, or statelessness is imposed.

The unsharded reference applies \(F\) separately to every head:

\[
(F_{\mathrm{ref}}X)(s,\ell,h,r)
=F(x_{h,r})(s,\ell). \tag{5}
\]

The CP path applies \(T\), evaluates \(F\) in the head view, then applies \(T^{-1}\).

### Theorem 3 — semantic equivalence of head-wise CP

\[
T^{-1}\!\left(F_{\mathrm{head}}(TX)\right)=F_{\mathrm{ref}}X. \tag{6}
\]

### Proof

Evaluate the left-hand side at \((s,\ell,h,r)\). The sequence supplied to \(F\) is

\[
(s',\ell')\mapsto(TX)(h,s',\ell',r).
\]

By (1), this is pointwise equal to

\[
(s',\ell')\mapsto X(s',\ell',h,r)=x_{h,r}.
\]

The output at \((s,\ell)\) is therefore \(F(x_{h,r})(s,\ell)\), which is the right-hand side of (5). Function extensionality yields (6). ∎

Lean proves the theorem for arbitrary types, dimensions and head-local functions:

```lean
headwise_context_parallel_correct
contextParallelFunction_eq_reference
```

This is the precise sense in which one theorem covers both full attention and GDN. It does not assert that either hardware kernel is correct; it states that a correct per-head kernel remains semantically correct when surrounded by the modeled CP exchange.

---

## 6. Consequence for derivatives and gradients

Equation (6) is equality of forward functions, not merely equality on one test input. Any extensional semantic observer must therefore return the same result for the CP function and the reference function. Lean formalizes this general fact as `every_extensional_observer_agrees`.

If a mathematical derivative construction \(D\) is defined extensionally, then

\[
D\!\left(T^{-1}\circ F_{\mathrm{head}}\circ T\right)
=D(F_{\mathrm{ref}}). \tag{7}
\]

This is the correct mathematical backward corollary. It must not be confused with verification of the actual PyTorch/NPU backward path. The latter still requires evidence that:

- each kernel backward is the derivative of its forward;
- collective backward realizes the transpose/inverse linear map;
- dtype conversions and reductions satisfy the chosen numerical tolerance;
- hooks, activation checkpointing and optimizer wiring preserve the gradient path.

A saved model identical to its initial weights is therefore not ruled out by the layout theorem. Such a symptom can arise after the mathematically correct forward path, for example through a disconnected gradient, a skipped optimizer step, or incorrect checkpoint serialization.

---

## 7. Variable-length packed metadata

### 7.1 Local run geometry

Consider a packed document beginning at global position \(d\). A contiguous run owned by one CP rank begins at \(a\) and ends at the exclusive position \(b\), with

\[
d\le a<b.
\]

The local Q run has length

\[
q=b-a. \tag{8}
\]

Its exact causal K/V prefix begins at the document start and ends at \(b\), hence has length

\[
k=b-d. \tag{9}
\]

A reset is correct exactly when the local run itself begins at the document start.

### Theorem 4 — exact reset criterion

Under \(d\le a<b\),

\[
q=k\quad\Longleftrightarrow\quad a=d. \tag{10}
\]

### Proof

Substituting (8) and (9) gives \(b-a=b-d\). Since both subtractions are taken within their ordered ranges, equality holds exactly when the subtrahends agree. The reverse direction follows immediately by substitution. Lean proves the natural-number statement with the ordering hypotheses explicit. ∎

This theorem corresponds to the implementation’s comparison between local Q-run lengths and causal K-prefix lengths when reconstructing global reset locations.

### 7.2 Exact causal-prefix enumeration

The metadata construction enumerates

\[
K(j)=d+j,\qquad 0\le j<b-d. \tag{11}
\]

Two properties are required.

**Soundness.** Every generated index satisfies

\[
d\le K(j)<b. \tag{12}
\]

**Completeness.** For every \(x\) with \(d\le x<b\), choosing \(j=x-d\) yields a legal index and \(K(j)=x\).

Together,

\[
\{K(j):0\le j<b-d\}
=\{x\in\mathbb N:d\le x<b\}. \tag{13}
\]

Lean declaration `prefixKey_exact` proves both directions. The half-open formulation matters: it excludes \(b\) while retaining \(b-1\), eliminating the two most common boundary errors.

---

## 8. Concrete CP=4 arithmetic

The frozen long-text configuration satisfies

\[
32768=4\cdot8192.
\]

All relevant head counts are also divisible by four:

| Logical head family | Global count | Count per CP head rank |
|---|---:|---:|
| full-attention Q | 24 | 6 |
| full-attention KV | 4 | 1 |
| GDN Q/K | 16 | 4 |
| GDN V | 48 | 12 |

The closed Lean theorems `concrete_sequence_partition` and `concrete_head_partitions` check both quotients and zero remainders. This apparently elementary step is part of the model contract: if a head count were not divisible by the CP degree, the implementation would need replication, padding, or unequal shards, and the equal-product coordinate model would have to be revised.

---

## 9. Correspondence table

| Formal object | Frozen implementation | Remaining bridge obligation |
|---|---|---|
| \(T\), sequence to head | `sequence_to_head_shard`, `exchange_sequence_heads` | DTensor/HCCL realizes the coordinate transposition |
| \(T^{-1}\), head to sequence | `head_to_sequence_shard` | reverse redistribution is the inverse |
| full-attention head operator | `AscVarlenAttention.forward` | fused TND attention has declared causal semantics |
| GDN head operator | `ContextParallelGatedDeltaNet.forward` | convolution, recurrence and reset are correct |
| local run \((d,a,b)\) | `CPVarlenMetadata.from_global` | global/local positions describe the true run |
| reset test | `build_sequence_metadata` | compared lengths belong to the same segment |
| prefix \([d,b)\) | `k_global_gather_indices` | gather returns values at the listed indices in order |

The table is also a test plan. A runtime implementation should be judged against these contracts rather than against the weaker observation that a loss curve decreases.

---

## 10. Engineering validation still required

A complete implementation assurance argument should add at least four test layers.

### 10.1 Pure permutation tests

Use integer tensors whose values uniquely encode tensor kind, token, rank and head. Verify sequence→head→sequence identity element by element for CP degrees 2, 4 and 8, including simultaneous packing and splitting of Q, K, V, decay and beta.

### 10.2 Kernel/reference tests

On small FP64 or FP32 examples, compare CP full attention, GDN and causal convolution against an unsharded reference. Check forward values, input gradients and parameter gradients separately. BF16 comparisons should report tolerances explicitly instead of being conflated with exact semantic tests.

### 10.3 Exhaustive short packed sequences

Enumerate document partitions and CP cut positions for short sequences. Confirm that reset markers occur exactly at document starts and that every query sees exactly the same-document causal prefix. Include length-one documents and cuts both on and inside document boundaries.

### 10.4 Training-state tests

Assert directly that expected parameters receive nonzero gradients, that optimizer steps change parameters, that the serialized checkpoint equals the in-memory post-step state, and that resume produces the same next step as an uninterrupted run within a declared tolerance.

These tests address failures outside the theorem, including the practical symptom of unchanged saved weights.

---

## 11. Interpretation

The machine-checked result is exact within its stated model: the sequence/head exchange is a bijection, every exact head-separable operator commutes with it, the reset predicate is necessary and sufficient, and the generated causal key prefix is precisely the intended interval.

Therefore, if the frozen implementation satisfies three bridge contracts—correct collective transposition, correct head-local kernels, and correct mapping of packed document/run coordinates—then its context-parallel forward function is extensionally identical to the unsharded reference function.

The proof rules out a broad class of semantic indexing errors at the model level and supplies precise obligations for implementation tests. It does not rule out numerical-kernel defects, collective runtime failures, autograd disconnections, optimizer omissions, or checkpoint bugs. Keeping those claims separate is not a weakness of the result; it is what makes the result auditable and reusable.
