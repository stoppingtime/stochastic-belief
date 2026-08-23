# Verification attestations

The scheduled verification workflow writes one Markdown attestation per month after all of the following pass:

1. pinned Lean toolchain installation;
2. pinned proof SHA-256 validation;
3. `lake build` of the default `Formal` target;
4. `leanchecker --fresh Formal.Qwen38M1UltraBound`;
5. axiom-set validation.

The monthly commit is both an audit trail and meaningful repository activity, so the public-repository scheduled workflows do not rely on an otherwise dormant repository staying active indefinitely.
