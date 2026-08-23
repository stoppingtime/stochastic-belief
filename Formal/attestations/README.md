# Verification attestations

The scheduled verification workflow writes one Markdown attestation per month after all of the following pass:

1. pinned Lean toolchain installation;
2. `lake build`;
3. `lean4checker --fresh Formal.Qwen38M1UltraBound`;
4. axiom-set validation;
5. source SHA-256 recording.

The monthly commit is both an audit trail and meaningful repository activity, so the public-repository scheduled workflows do not rely on an otherwise dormant repository staying active indefinitely.
