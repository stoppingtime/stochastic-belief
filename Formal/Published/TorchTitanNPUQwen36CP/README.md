# Qwen3.6 Context Parallel correctness — TorchTitan-NPU override-refactor

**Model ID:** `SP-P0002-TTNPU-Q36-CP-v1`

This directory contains one synchronized mathematical publication in four human-readable forms:

- [`paper.zh.md`](paper.zh.md) — Chinese, rendered directly by GitHub;
- [`paper.en.md`](paper.en.md) — English, rendered directly by GitHub;
- [`paper.zh.tex`](paper.zh.tex) — Chinese LaTeX manuscript;
- [`paper.en.tex`](paper.en.tex) — English LaTeX manuscript.

The machine-checked layer is [`Proof.lean`](Proof.lean). [`evidence.md`](evidence.md) records the exact implementation revision and explains where the source-to-model bridge remains an engineering premise.

## Main result

For the coordinate model corresponding to TorchTitan-NPU’s sequence-to-head exchange:

\[
\operatorname{HeadToSequence}
\circ F_{\mathrm{head}}
\circ \operatorname{SequenceToHead}
=
F_{\mathrm{reference}}
\]

for every exact operator \(F\) that acts independently on each complete head sequence. The two exchange maps are mutual inverses. The variable-length metadata proof additionally establishes that the local reset test is exact and that the generated K/V prefix is precisely the causal interval from the packed-document start to the local segment stop.

## Frozen implementation evidence

- TorchTitan-NPU mirror commit: `7864d5dc17930667d663bbadd1ce2bc722de2753`
- TorchTitan commit: `c91448d20480c7b294314e68976823050002ebec`
- Concrete long-text configuration: sequence length `32768`, CP degree `4`, load balancer disabled
- Lean: `4.33.0`, Core only

## Reproduce

From the repository root:

```bash
lake build
lake env leanchecker --fresh Formal.Published.TorchTitanNPUQwen36CP.Proof
lake env lean Formal/Published/TorchTitanNPUQwen36CP/Proof.lean
```

The theorem does not certify NPU kernels, collectives, floating-point rounding, autograd, optimizer stepping or training convergence. Those boundaries are stated in all four manuscripts and in the evidence ledger.

<!-- CI-only replay marker; this branch is not intended for merge. -->
