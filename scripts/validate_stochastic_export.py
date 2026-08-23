#!/usr/bin/env python3
"""Validate one history-free publication bundle before it enters the public tree.

Incoming branches are treated as untrusted data.  Publication authority is
therefore duplicated on the public side: a private allowlist is necessary to
export a problem, but a hard-coded public allowlist is also necessary to import
it.  New problem IDs require an explicit public-repository change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path, PurePosixPath

SCHEMA = 1
CANONICAL_REPO = "stoppingtime/stochastic-problems"
PUBLIC_REPO = "stoppingtime/stochastic-belief"
ALLOWED_TOOLCHAIN = "leanprover/lean4:v4.33.0"
ALLOWED_EXT = {".lean", ".md", ".tex", ".json", ".txt", ".toml"}
MAX_FILES = 32
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
DECL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*(?:\.[A-Za-z_][A-Za-z0-9_']*)*$")

PUBLICATIONS = {
    "p0001-qwen38-m1-ultra-q4": {
        "public_root": "Formal/Published/Qwen38M1UltraQ4",
        "model_id": "SP-P0001-Q38-M1U-Q4-v1",
        "audit_declarations": [
            "Qwen38M1Ultra.threshold_impossible",
            "Qwen38M1Ultra.certifyExactCeiling",
            "Qwen38M1Ultra.raw_ctx_0_exact",
            "Qwen38M1Ultra.apple_rated_ctx_0_exact",
            "Qwen38M1Ultra.audit_conclusion",
        ],
        "required_files": {
            "metadata.json",
            "evidence.md",
            "paper.zh.tex",
            "paper.en.tex",
            "paper.zh.md",
            "paper.en.md",
            "Proof.lean",
            "PUBLIC-PROVENANCE.json",
        },
    },
}

ANCHOR_FIELDS = {
    "MODEL-ID": ("model_id", None),
    "STATIC-FLOOR-BYTES": ("formal_constants", "static_representation_floor_bytes"),
    "KV-BYTES-PER-CONTEXT-TOKEN": ("formal_constants", "kv_bytes_per_context_token"),
    "RAW-BUS-BPS": ("formal_constants", "raw_bus_bytes_per_second"),
    "APPLE-RATED-BPS": ("formal_constants", "apple_rated_bytes_per_second"),
    "RAW-CTX0-MILLI-TPS": ("formal_constants", "raw_ctx0_exact_milli_tps"),
    "APPLE-CTX0-MILLI-TPS": ("formal_constants", "apple_ctx0_exact_milli_tps"),
    "RAW-CTX262144-MILLI-TPS": ("formal_constants", "raw_ctx262144_exact_milli_tps"),
    "APPLE-CTX262144-MILLI-TPS": ("formal_constants", "apple_ctx262144_exact_milli_tps"),
}
ANCHOR_RE = re.compile(r"(?:<!--\s*|%\s*)([A-Z0-9-]+):\s*([^\s%<]+)")

SECRET_MARKERS = (
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "github_pat_",
    "ghp_",
    "AKIA",
)
LEAN_FORBIDDEN = (
    "sorry",
    "admit",
    "native_decide",
    "axiom ",
    "unsafe",
    "run_tac",
    "debug.skipKernelTC",
    "Lean.trustCompiler",
    "@[implemented_by",
    "@[extern",
    "initialize ",
)


def die(msg: str) -> None:
    raise SystemExit(f"export validation failed: {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(s: str) -> PurePosixPath:
    p = PurePosixPath(s)
    if p.is_absolute() or not p.parts or any(x in ("", ".", "..") for x in p.parts):
        die(f"unsafe path {s!r}")
    if any(x.startswith(".git") or x == ".github" for x in p.parts):
        die(f"forbidden path component in {s!r}")
    return p


def read_utf8(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeError as exc:
        die(f"non-UTF-8 text in {path}: {exc}")
    for marker in SECRET_MARKERS:
        if marker in text:
            die(f"secret-like marker {marker!r} in {path}")
    return text


def expected_anchors(meta: dict) -> dict[str, str]:
    result = {}
    for anchor, (outer, inner) in ANCHOR_FIELDS.items():
        value = meta[outer] if inner is None else meta[outer][inner]
        result[anchor] = str(value)
    return result


def check_quartet(payload: Path, meta: dict) -> None:
    expected = expected_anchors(meta)
    quartet = ("paper.zh.tex", "paper.en.tex", "paper.zh.md", "paper.en.md")
    anchor_maps = []
    for name in quartet:
        path = payload / name
        text = read_utf8(path)
        if len(text) < 6000:
            die(f"{name} is a stub rather than a reviewable manuscript")
        found = {m.group(1): m.group(2) for m in ANCHOR_RE.finditer(text[:6000])}
        for key, value in expected.items():
            if found.get(key) != value:
                die(f"{name}: anchor {key}={found.get(key)!r}, expected {value!r}")
        anchor_maps.append(found)
    if any(m != anchor_maps[0] for m in anchor_maps[1:]):
        die("the four manuscript anchor maps differ")


def check_lean(path: Path) -> None:
    text = read_utf8(path)
    imports = re.findall(r"^\s*import\s+([^\s]+)\s*$", text, flags=re.MULTILINE)
    if imports != ["Init"]:
        die(f"Proof.lean must import exactly Init; got {imports}")
    for token in LEAN_FORBIDDEN:
        if token in text:
            die(f"Proof.lean contains forbidden token {token!r}")
    if "#print axioms" not in text:
        die("Proof.lean has no #print axioms audit hooks")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    src = args.source.resolve()
    manifest_path = src / "EXPORT-MANIFEST.json"
    if not manifest_path.is_file():
        die("missing EXPORT-MANIFEST.json")
    manifest = json.loads(read_utf8(manifest_path))

    required_keys = {
        "schema", "canonical_repo", "canonical_commit", "public_repo",
        "problem_id", "toolchain", "lean_entry", "audit_declarations", "files",
    }
    if set(manifest) != required_keys:
        die(f"manifest keys differ from schema: {sorted(set(manifest) ^ required_keys)}")
    if manifest["schema"] != SCHEMA:
        die("unsupported schema")
    if manifest["canonical_repo"] != CANONICAL_REPO:
        die("canonical_repo mismatch")
    if manifest["public_repo"] != PUBLIC_REPO:
        die("public_repo mismatch")
    if not isinstance(manifest["canonical_commit"], str) or not HEX40_RE.fullmatch(manifest["canonical_commit"]):
        die("canonical_commit must be lowercase 40-hex")
    if manifest["toolchain"] != ALLOWED_TOOLCHAIN:
        die("toolchain is not on the public allowlist")

    pid = manifest["problem_id"]
    policy = PUBLICATIONS.get(pid)
    if policy is None:
        die(f"problem_id {pid!r} is not authorized by the public repository")
    if manifest["audit_declarations"] != policy["audit_declarations"]:
        die("audit declaration list differs from the public allowlist")
    if any(not DECL_RE.fullmatch(x) for x in manifest["audit_declarations"]):
        die("invalid audit declaration name")

    entries = manifest["files"]
    if not isinstance(entries, list) or not (1 <= len(entries) <= MAX_FILES):
        die("invalid file-entry count")

    seen: set[PurePosixPath] = set()
    normalized = []
    total = 0
    prefix = PurePosixPath("payload") / pid
    for e in entries:
        if not isinstance(e, dict) or set(e) != {"path", "sha256"}:
            die("each file entry must contain exactly path and sha256")
        p = safe_rel(e["path"])
        if p in seen:
            die(f"duplicate file {p}")
        seen.add(p)
        if p.suffix.lower() not in ALLOWED_EXT:
            die(f"extension not allowed: {p}")
        try:
            rel_inside = p.relative_to(prefix)
        except ValueError:
            die(f"file must be under payload/{pid}/: {p}")
        if len(rel_inside.parts) != 1:
            die(f"nested payload paths are not allowed in schema 1: {p}")
        if not isinstance(e["sha256"], str) or not HEX64_RE.fullmatch(e["sha256"]):
            die(f"invalid sha256 for {p}")
        real = src.joinpath(*p.parts)
        if not real.exists() or not real.is_file() or real.is_symlink() or not stat.S_ISREG(real.stat().st_mode):
            die(f"missing, non-regular, or symlink file: {p}")
        size = real.stat().st_size
        if size > MAX_FILE_BYTES:
            die(f"file too large: {p}")
        total += size
        if total > MAX_TOTAL_BYTES:
            die("export exceeds total size limit")
        if sha256(real) != e["sha256"]:
            die(f"sha256 mismatch: {p}")
        read_utf8(real)
        normalized.append((rel_inside, real))

    actual_names = {p.name for p in seen}
    if actual_names != policy["required_files"]:
        die(f"payload file set differs from public policy: {sorted(actual_names ^ policy['required_files'])}")

    payload_root = src.joinpath(*prefix.parts)
    actual_disk = set()
    for root, dirs, files in os.walk(payload_root, followlinks=False):
        rootp = Path(root)
        for d in dirs:
            if (rootp / d).is_symlink():
                die(f"symlink directory forbidden: {rootp / d}")
        for name in files:
            q = rootp / name
            actual_disk.add(PurePosixPath(q.relative_to(src).as_posix()))
    if actual_disk != seen:
        die("manifest does not describe the payload tree exactly")

    entry = safe_rel(manifest["lean_entry"])
    if entry != prefix / "Proof.lean":
        die("lean_entry must be the authorized Proof.lean")

    meta = json.loads(read_utf8(payload_root / "metadata.json"))
    provenance = json.loads(read_utf8(payload_root / "PUBLIC-PROVENANCE.json"))
    if meta.get("problem_id") != pid or meta.get("model_id") != policy["model_id"]:
        die("metadata identity differs from public policy")
    if provenance.get("problem_id") != pid or provenance.get("model_id") != policy["model_id"]:
        die("provenance identity differs from public policy")
    if provenance.get("canonical_repo") != CANONICAL_REPO:
        die("provenance canonical repo mismatch")
    if provenance.get("canonical_commit") != manifest["canonical_commit"]:
        die("provenance canonical commit mismatch")
    if provenance.get("public_repo") != PUBLIC_REPO or provenance.get("public_root") != policy["public_root"]:
        die("provenance public destination mismatch")

    check_quartet(payload_root, meta)
    check_lean(payload_root / "Proof.lean")

    out = args.output.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for rel_inside, real in normalized:
        shutil.copyfile(real, out / rel_inside.name)

    receipt = {
        "schema": 1,
        "canonical_repo": CANONICAL_REPO,
        "canonical_commit": manifest["canonical_commit"],
        "problem_id": pid,
        "model_id": policy["model_id"],
        "public_root": policy["public_root"],
        "toolchain": ALLOWED_TOOLCHAIN,
        "lean_entry": "Proof.lean",
        "lean_entry_sha256": sha256(out / "Proof.lean"),
        "audit_declarations": manifest["audit_declarations"],
        "file_count": len(normalized),
        "total_bytes": total,
        "manifest_sha256": sha256(manifest_path),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
