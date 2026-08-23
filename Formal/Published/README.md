# Published formal problems

This directory is the **public canonical release set** for human-readable
mathematical models and machine-checkable Lean proofs.

Each released problem contains:

```text
paper.zh.tex
paper.en.tex
paper.zh.md
paper.en.md
Proof.lean
metadata.json
publication.json
evidence.md
```

The Chinese and English manuscripts share machine-readable anchors, but each
language is written as an independent mathematical exposition rather than a
sentence-by-sentence translation.

## Credential-free repository relationship

`stoppingtime/stochastic-belief` is canonical for released artifacts.
`stoppingtime/stochastic-problems` is the private research archive. Its own
scheduled workflow clones this public repository over anonymous HTTPS and
stores a byte-for-byte release mirror under `PublicMirror/`.

This direction requires no deploy key, PAT, GitHub App private key, or
cross-repository secret:

```text
public canonical release
        |
        | anonymous read
        v
private PublicMirror
```

Private drafts and review notes never enter the public repository. There is no
automated private-to-public write path; publication is an explicit review
action in the public repository, while archival synchronization is automatic.

## Verification

The public workflow checks the four-manuscript contract, model anchors,
evidence boundary, publication allowlist, an isolated `lake build`,
`leanchecker --fresh`, and the declared `#print axioms` targets for every
problem.
