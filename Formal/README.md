# Formal verification

This directory contains machine-checkable Lean 4 models and proofs that live next to the repository's historical optimization experiments.

## Qwen3.8-27B Q4_K_M on M1 Ultra

`Qwen38M1UltraBound.lean` formalizes a bandwidth-conservation model for batch-1, non-speculative autoregressive decoding. The current instantiated result is intentionally conditional on external hardware/model facts.

The strengthened proof now certifies **exact discrete ceilings inside the abstract model**, not only one-sided upper bounds. At milli-token/s granularity it proves both:

- the named ceiling is feasible when actual DRAM traffic is exactly the declared conservative floor; and
- the next milli-token/s value is impossible under the same bandwidth cap.

Main instantiated exact ceilings:

| model instance | context | exact ceiling in the abstract floor-traffic model |
|---|---:|---:|
| 819.2 GB/s raw line-rate | 0 | 52.311 tok/s |
| 819.2 GB/s raw line-rate | 262,144 | 24.945 tok/s |
| Apple-rated 800 GB/s | 0 | 51.085 tok/s |
| Apple-rated 800 GB/s | 262,144 | 24.360 tok/s |

These are **not** claims that real hardware can attain those rates. The bridge from the physical machine and a concrete GGUF file to the formal premises remains an external-evidence obligation.

## Reproduce

```bash
(cd Formal && sha256sum -c PROOF-SHA256)
lake build
lake env leanchecker --fresh Formal.Qwen38M1UltraBound
lake env lean Formal/Qwen38M1UltraBound.lean
```

`Formal.lean` imports the main theorem module and `Formal` is the Lake default target, so `lake build` cannot report success while silently skipping the proof. The final command prints the axioms used by the audit declarations. CI rejects any axiom outside Lean's standard set `{propext, Classical.choice, Quot.sound}`.

The canonical toolchain is pinned by `lean-toolchain`. Monthly attestations are written under `Formal/attestations/` after a successful scheduled verification.
