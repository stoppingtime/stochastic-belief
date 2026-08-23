# Published stochastic-problems artifacts

This directory is populated only by trusted workflows stored on the public repository's `master` branch.

The private canonical repository is `stoppingtime/stochastic-problems`. It does **not** mirror its private Git history here. Publication bundles arrive under

```text
incoming/stochastic-problems/<problem-id>
```

and are treated as untrusted data.

## Two-stage trust boundary

The publication path deliberately separates **validation authority** from **write authority**.

The validation job has only `contents: read`. It materializes only file names already authorized by the public policy, verifies the bundle, and runs the incoming Lean source in a fresh temporary project. It cannot write the public default branch.

Only after that job succeeds does a second job receive `contents: write`. The second job downloads the already-validated artifact, copies data files into `Formal/Published/`, and commits them. It does not execute the incoming Lean source.

A separate pull-request workflow performs the same incoming validation with read-only permissions. It exists to make the trust contract observable and reviewable without granting publication authority. A second read-only audit workflow prints the manifest and actual SHA-256 values of the incoming files, which makes byte-level mirror drift diagnosable without executing incoming code.

## Validation performed before publication

Before a bundle is copied into this directory, the public side independently checks:

1. the problem ID against a public hard-coded allowlist;
2. the exact file set and every SHA-256 in `EXPORT-MANIFEST.json`;
3. absence of nested paths, symlinks, executable workflow files, private-key markers, and unapproved file types;
4. the four synchronized human manuscripts: Chinese/English LaTeX and Chinese/English Markdown;
5. their shared `MODEL-ID` and numerical anchors against `metadata.json`;
6. the public destination and private canonical commit recorded by `PUBLIC-PROVENANCE.json`;
7. that `Proof.lean` imports only Lean Core `Init` and contains none of the prohibited trust escapes;
8. `lake build` in a fresh temporary project;
9. `leanchecker --fresh` replay of the compiled proof;
10. `#print axioms` output for the declarations named by the public allowlist.

Each imported problem also receives `SYNC-RECEIPT.json`, which records the private canonical commit, manifest hash, Lean source hash, public destination, and required audit declarations.

A new private problem cannot become public merely by changing the private repository: its problem ID and destination must first be added to the public validator's `PUBLICATIONS` table through an ordinary public-repository change.
