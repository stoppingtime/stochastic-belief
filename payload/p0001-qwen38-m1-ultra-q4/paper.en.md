<!-- MODEL-ID: SP-P0001-Q38-M1U-Q4-v1 -->
<!-- STATIC-FLOOR-BYTES: 15660093440 -->
<!-- KV-BYTES-PER-CONTEXT-TOKEN: 65536 -->
<!-- RAW-BUS-BPS: 819200000000 -->
<!-- APPLE-RATED-BPS: 800000000000 -->
<!-- RAW-CTX0-MILLI-TPS: 52311 -->
<!-- APPLE-CTX0-MILLI-TPS: 51085 -->
<!-- RAW-CTX262144-MILLI-TPS: 24945 -->
<!-- APPLE-CTX262144-MILLI-TPS: 24360 -->

# A bandwidth ceiling for single-token Qwen3.8-27B Q4_K_M decoding on M1 Ultra

**Model ID: SP-P0001-Q38-M1U-Q4-v1**

## Abstract

We study a deliberately narrow question: under batch-one, non-speculative, non-MTP autoregressive decoding of Qwen3.8-27B Q4_K_M on M1 Ultra, how large can the one-target-token decode rate be under an explicit memory-bandwidth model?

The argument does not identify the GGUF file size with DRAM traffic per token, nor does it infer a physical limit from an observed benchmark. Instead, it derives a conservative representation-byte floor from the model architecture and the block encodings of the selected quantization recipe. The statement that real DRAM traffic is at least this floor is kept as an explicit physical premise. The mathematical step is then a conservation law: if one output token requires at least \(D\) bytes to cross DRAM and the memory path can carry at most \(B\) bytes per second, any throughput \(r\) must satisfy

\[
rD\le B.
\]

The Lean formalization avoids floating-point arithmetic by measuring throughput in milli-token/s. For the principal short-context instance with `cacheCarry=0`, the exact discrete ceilings of the abstract model are **51.085 token/s** at Apple's published \(800\times10^9\) byte/s bandwidth and **52.311 token/s** under the more permissive \(819.2\times10^9\) byte/s raw line-rate scenario. At context length \(262144\), these become **24.360 token/s** and **24.945 token/s**, respectively.

“Exact” here means exact at milli-token/s granularity **inside the stated model**. It is not an empirical claim that an M1 Ultra can attain the ceiling, and it does not mean Lean has authenticated Apple's specification, the GGUF tensor layout, or a particular runtime's DRAM traffic.

---

## 1. Scope, quantities, and units

The target is one ordinary target-model decode step producing one target token. Throughout the principal result we fix:

- batch size \(1\);
- no speculative decoding;
- no MTP acceptance of multiple target tokens;
- FP16 KV storage for the growing full-attention cache;
- `cacheCarry=0`, so no cross-token on-chip residency is silently deducted from the DRAM floor.

Let

- \(B\in\mathbb N\) be a bandwidth cap in byte/s;
- \(r\in\mathbb N\) be throughput in milli-token/s;
- \(A\in\mathbb N\) be the actual DRAM traffic per generated token, in byte/token;
- \(D(L,C)\in\mathbb N\) be the modeled lower bound on that traffic at context length \(L\) with an allowance \(C\) for bytes retained on chip across decode steps.

Because \(r\) is measured in thousandths of a token per second, the bandwidth-feasibility relation is the integer inequality

\[
rA\le 1000B. \tag{1}
\]

No floating-point approximation enters (1).

---

## 2. External evidence is not part of the proof kernel

The model uses external facts about M1 Ultra bandwidth, Qwen3.8-27B architecture, GGML block encodings, and the tensor-class assumptions of the selected GGUF artifact. Their provenance is recorded separately in `evidence.md`.

Lean proves implications from mathematical premises. It does not prove that a vendor page or an artifact description is true. The bridge from the physical execution to the theorem is therefore stated explicitly:

> **DRAM-traffic premise.** At context length \(L\) and cache allowance \(C\), the actual traffic \(A\) obeys
>
> \[
> D(L,C)\le A. \tag{2}
> \]

Every throughput statement below is conditional on (1) and (2).

This distinction rules out a common shortcut. A 16.5 GiB GGUF file does not imply 16.5 GiB of DRAM traffic on every token: a file contains more than the set of tensors necessarily streamed by a particular step, while caching may remove some accesses from DRAM. We therefore construct a smaller representation floor tensor by tensor.

---

## 3. Static representation floor from the model architecture

For the text model we use

\[
H=5120,\qquad I=17408,\qquad N_{\mathrm{layer}}=64.
\]

The 64 layers comprise 48 Gated DeltaNet layers and 16 full-attention layers. The vocabulary size is

\[
V=248320.
\]

### 3.1 Feed-forward networks

Each SwiGLU FFN has three large matrices: gate, up, and down. Hence

\[
N_{\mathrm{FFN}}
=64\times3\times5120\times17408
=17\,112\,760\,320. \tag{3}
\]

### 3.2 Large Gated DeltaNet projections

There are 16 Q/K heads and 48 V heads, each with dimension 128, so

\[
d_{QK}=16\times128=2048,
\qquad
d_V=48\times128=6144.
\]

For each GDN layer the conservative floor includes the three large tier-quantized projections

\[
W_{qkv}:5120\to(2d_{QK}+d_V),
\]

\[
W_z:5120\to d_V,
\qquad
W_o:d_V\to5120.
\]

Across 48 layers this gives

\[
N_{\mathrm{GDN,Q4}}=5\,536\,481\,280. \tag{4}
\]

The selected recipe represents the `in_proj_a` and `in_proj_b` class as F32. Their count is

\[
N_{\mathrm{SSM,F32}}
=48\times2\times5120\times48
=23\,592\,960. \tag{5}
\]

### 3.3 Full attention

Full attention uses 24 Q heads, 4 KV heads, and head dimension 256. The Q projection carries an additional output-gate half, so its output width is counted as \(2\times24\times256\). Across all 16 full-attention layers, Q/K/V/O contribute

\[
N_{\mathrm{FA,Q8}}=1\,677\,721\,600. \tag{6}
\]

### 3.4 Output projection

Input embeddings and the language-model output projection are untied. A decode step does not stream the entire input embedding table; it looks up the current token's row. Producing the complete logit vector, however, requires the output projection. We therefore include

\[
N_{\mathrm{out,Q6}}
=248320\times5120
=1\,271\,398\,400. \tag{7}
\]

### 3.5 Converting weights into encoded bytes

The block encodings used by the model are

\[
\mathrm{Q4_K}:\;256\text{ weights}\to144\text{ bytes},
\]

\[
\mathrm{Q6_K}:\;256\text{ weights}\to210\text{ bytes},
\]

\[
\mathrm{Q8_0}:\;32\text{ weights}\to34\text{ bytes},
\qquad
\mathrm{F32}:\;1\to4\text{ bytes}.
\]

The relevant counts are block-aligned, so no partial-block correction is needed. Define

\[
\begin{aligned}
D_{\mathrm{static}}
={}&144\frac{N_{\mathrm{FFN}}+N_{\mathrm{GDN,Q4}}}{256}\\
&+34\frac{N_{\mathrm{FA,Q8}}}{32}
+4N_{\mathrm{SSM,F32}}
+210\frac{N_{\mathrm{out,Q6}}}{256}.
\end{aligned} \tag{8}
\]

Substitution of (3)--(7) yields

\[
\boxed{D_{\mathrm{static}}=15\,660\,093\,440\ \text{byte/token}}. \tag{9}
\]

Equation (9) is a conservative **representation floor**, not a measurement of physical DRAM traffic. We intentionally omit RMSNorm vectors, convolution weights, small state/bias tensors, and kernel-intermediate traffic. Omitting bytes decreases \(D\), which makes the resulting throughput ceiling larger. That is the conservative direction for an impossibility bound.

---

## 4. Context-dependent KV traffic

Only the 16 full-attention layers contribute a KV cache that grows linearly with context. For one historical token in one such layer, FP16 K and V require

\[
4\text{ KV heads}\times256
\times2\text{ tensors}\times2\text{ byte}.
\]

Across the 16 full-attention layers,

\[
\boxed{
D_{\mathrm{KV/token}}
=16\times4\times256\times2\times2
=65\,536\ \text{byte/context-token}
}. \tag{10}
\]

Thus the representation-level working-set floor at context length \(L\) is

\[
R(L)=15\,660\,093\,440+65\,536L. \tag{11}
\]

If \(C\) bytes can remain usefully resident on chip between consecutive decode steps and therefore need not cross DRAM again, define

\[
D(L,C)=\max\{0,R(L)-C\}. \tag{12}
\]

The Lean model uses truncated natural-number subtraction for the same quantity. The principal theorem takes \(C=0\). This does not assert that caches are absent; it keeps cross-token residency as an explicit parameter rather than an unreviewed assumption.

---

## 5. General bandwidth theorem

### Theorem 1 — Conservation upper bound

Let \(B,r,A,D,T\in\mathbb N\). Assume

\[
D\le A, \tag{13}
\]

\[
rA\le1000B, \tag{14}
\]

and

\[
1000B<TD. \tag{15}
\]

Then

\[
r<T. \tag{16}
\]

### Proof

Suppose instead that \(r\ge T\). Since \(A\ge D\), monotonicity of multiplication on natural numbers gives

\[
TD\le rA. \tag{17}
\]

Bandwidth feasibility (14) gives

\[
rA\le1000B. \tag{18}
\]

Therefore

\[
TD\le1000B. \tag{19}
\]

But (15) states

\[
1000B<TD, \tag{20}
\]

contradicting (19). Hence \(r<T\). ∎

Nothing in this theorem is specific to neural networks. The architecture is needed only to establish a defensible \(D\); once \(D\) is fixed, the bound is ordinary data-conservation arithmetic.

---

## 6. Exact discrete ceilings

An upper bound alone does not show that the stated integer is the largest feasible integer in the abstract model. For an integer \(q\), check the adjacent pair

\[
qD\le1000B<(q+1)D. \tag{21}
\]

In the floor-traffic model \(A=D\), the left inequality makes \(q\) feasible while the right inequality rules out \(q+1\). Thus \(q\) is the exact maximum at milli-token/s granularity.

This model-internal feasibility statement must not be confused with hardware attainability. Real executions may have \(A>D\), may fail to sustain the bandwidth cap, or may incur additional scheduling and kernel overheads.

---

## 7. Numerical instances

### 7.1 Apple's published 800 GB/s figure

Take

\[
B_{\mathrm{Apple}}=800\,000\,000\,000\ \text{byte/s},
\qquad L=0,\ C=0.
\]

Then \(D=15\,660\,093\,440\), and the adjacent-integer check is

\[
51\,085D\le1000B_{\mathrm{Apple}}<51\,086D. \tag{22}
\]

Hence the exact discrete ceiling of the abstract model is

\[
\boxed{51.085\ \text{token/s}}. \tag{23}
\]

### 7.2 The more permissive 819.2 GB/s raw line-rate scenario

Set

\[
B_{\mathrm{raw}}=819\,200\,000\,000\ \text{byte/s}.
\]

At short context,

\[
52\,311D\le1000B_{\mathrm{raw}}<52\,312D, \tag{24}
\]

so

\[
\boxed{52.311\ \text{token/s}}. \tag{25}
\]

This is intentionally the more generous ceiling: the model grants the machine more bandwidth than Apple's published rating. A claim of ordinary one-target-token decode substantially above this value must therefore invalidate or change some premise, for example by using speculative/MTP decoding, batching, or a demonstrated cross-token residency \(C>0\).

### 7.3 Native maximum context \(L=262144\)

Equation (11) gives

\[
R(262144)
=15\,660\,093\,440+65\,536\times262\,144
=32\,839\,962\,624\ \text{byte/token}. \tag{26}
\]

The adjacent-integer certificates then give

\[
\boxed{
\begin{aligned}
B=800\text{ GB/s}:&\quad 24.360\text{ token/s},\\
B=819.2\text{ GB/s}:&\quad 24.945\text{ token/s}.
\end{aligned}}
\tag{27}
\]

At this context length the growing historical KV traffic is comparable to the static encoded-weight floor, so the bandwidth ceiling falls sharply.

---

## 8. Interpretation boundary

The conclusions must be read together with their assumptions.

**First, 800 GB/s is an Apple product specification.** It is not derived here from first-principles DRAM timing as an unconditional physical theorem.

**Second, 819.2 GB/s is a raw line-rate scenario.** It is obtained from LPDDR5-6400 and an aggregate 1024-bit interface and is used precisely because it gives the upper-bound argument a more generous bandwidth budget. It is not a statement of sustained application payload bandwidth.

**Third, 15,660,093,440 byte/token is a representation floor.** Lean verifies that the number follows from the frozen tensor classes and encoding formulas. Whether the pinned GGUF artifact in fact assigns every relevant tensor to those classes remains an artifact-audit obligation.

**Fourth, equation (2) is the physical bridge.** If a future microarchitectural analysis establishes that a substantial subset of the relevant representation remains resident across tokens, reducing actual DRAM traffic below the present `dramFloor`, the correct response is to increase \(C\) and recompute the theorem instance.

**Fifth, speculative/MTP decoding is out of scope.** If one target verification pass accepts multiple output tokens, the relation between target passes and visible output token/s changes. Output throughput can then exceed the single-target-step ceiling without contradicting this theorem.

---

## 9. Formal verification

The machine-checkable development is `Proof.lean`. All critical numerical inequalities are discharged over natural numbers, with throughput discretized to milli-token/s. `audit_conclusion` bundles the principal exact ceilings.

The intended verification sequence is

```bash
lake build
lake env leanchecker --fresh Problems.Qwen38M1UltraQ4.Proof
lake env lean Problems/Qwen38M1UltraQ4/Proof.lean
```

The final command exposes `#print axioms` output. CI accepts only a subset of Lean's standard `{propext, Classical.choice, Quot.sound}` set and rejects `sorryAx` or `Lean.trustCompiler`.

Accordingly, the result has two deliberately separate components:

\[
\text{external evidence and physical premises}
\quad+\quad
\text{kernel-checked mathematical deduction}.
\]

A stronger claim about the physical M1 Ultra requires an independent audit of the first component; the Lean proof certifies the second.
