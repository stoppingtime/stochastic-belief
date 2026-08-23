# Published mathematical problems

This directory is the public, machine-verifiable publication surface for mathematical models and proofs. It contains only artifacts intended to be public: synchronized Chinese/English LaTeX and Markdown manuscripts, evidence ledgers, metadata, and Lean sources.

## Keyless synchronization model

No cross-repository credential is used.

```text
stochastic-belief (public publication authority)
        |
        | anonymous HTTPS read of a fixed public commit
        v
stochastic-problems (private archival mirror + private notes)
```

The private repository periodically clones this public repository without authentication and copies each allowlisted publication into its archive using its own repository-scoped `GITHUB_TOKEN`. Private `notes/` directories remain private and are never read or written by the public repository. This direction is possible without a PAT, deploy key, GitHub App private key, or user-maintained secret because public Git data is anonymously readable and the only write occurs inside the private workflow's own repository.

The public repository is therefore authoritative for publishable theorem statements and manuscripts. The private repository is authoritative for non-public research notes and keeps a content-addressed archival copy of each public problem.

## Per-problem contract

Every published problem directory contains:

1. `paper.zh.tex` — Chinese LaTeX manuscript;
2. `paper.en.tex` — English LaTeX manuscript;
3. `paper.zh.md` — Chinese GitHub-renderable manuscript;
4. `paper.en.md` — English GitHub-renderable manuscript;
5. `Proof.lean` — machine-checked formal layer;
6. `evidence.md` — external evidence and source-to-model bridge;
7. `metadata.json`, `publication.json`, `PUBLIC-PROVENANCE.json`, and `README.md`.

The four manuscripts carry the same machine-readable anchor map. They are expected to be independently readable documents, not generated summaries or redirects.

## Verification

`.github/workflows/verify-published-problems.yml` performs:

- exact publication file-set validation;
- bilingual and dual-format anchor equality;
- minimum readability and document-structure checks;
- evidence/provenance consistency checks;
- `lake build` of every registered proof;
- `leanchecker --fresh` replay;
- `#print axioms` validation against Lean's standard trusted set;
- rejection of `sorry`, `admit`, `native_decide`, custom axioms, `sorryAx`, and `Lean.trustCompiler`.

A mathematical theorem remains conditional on the external facts stated in its evidence ledger. Passing CI establishes that the Lean deduction and publication contract are reproducible; it does not turn unverified hardware, source-code, or experimental assumptions into formal theorems.
