# stochastic-belief

A small research scratchpad for optimization experiments and a public CI home
for reproducible Lean 4 formal verification.

The historical Python experiments remain at the repository root. The original
M1 Ultra/Qwen3.8 proof lives under [`Formal/`](Formal/README.md). Reviewed,
bilingual formal-problem releases live under
[`Formal/Published/`](Formal/Published/README.md).

## Verification layers

The repository currently maintains:

- a conditional bandwidth roofline for Qwen3.8-27B Q4_K_M on M1 Ultra;
- a conditional transport-correctness theorem for Qwen3.6 Context Parallel in
  TorchTitan-NPU `override-refactor`;
- Chinese/English LaTeX and GitHub-renderable Markdown manuscripts for each
  published problem;
- isolated Lean builds, fresh kernel replay, and axiom audits.

```bash
(cd Formal && sha256sum -c PROOF-SHA256)
lake build
python3 scripts/check_published.py
```

## Relationship to the private archive

This public repository is canonical for released artifacts. The private
`stoppingtime/stochastic-problems` repository mirrors releases from here over
anonymous HTTPS. No cross-repository credential is required or maintained.

## Lean resource snapshots

[`resources/`](resources/README.md) is refreshed automatically from selected
Lean community Markdown sources, including `best-of-lean4` and Lean prover
community learning and verification guides. Licenses and attribution are
mirrored with the snapshots.
