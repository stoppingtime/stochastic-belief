# stochastic-belief

A small research scratchpad for optimization experiments, now also hosting reproducible Lean 4 formal verification.

The historical Python experiments remain at the repository root. New machine-checked work lives under [`Formal/`](Formal/README.md), with the Lean version pinned by `lean-toolchain` and checked on every relevant push/PR.

## Formal proof currently tracked

The repository verifies a conditional bandwidth roofline for **Qwen3.8-27B Q4_K_M on M1 Ultra**. The strengthened theorem certifies exact milli-token/s ceilings *inside the declared abstract model*, while keeping the hardware/GGUF evidence boundary explicit.

```bash
lake build
lake env lean4checker --fresh Formal.Qwen38M1UltraBound
```

## Lean resource snapshots

[`resources/`](resources/README.md) is refreshed automatically from selected Lean community Markdown sources, including `best-of-lean4` and Lean prover community learning/verification guides. Licenses and attribution are mirrored with the snapshots.
