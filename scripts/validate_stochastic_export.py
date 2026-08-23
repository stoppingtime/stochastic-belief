#!/usr/bin/env python3
import argparse, hashlib, json, os, re, shutil, stat
from pathlib import Path, PurePosixPath

SCHEMA = 1
CANONICAL_REPO = "stoppingtime/stochastic-problems"
PUBLIC_REPO = "stoppingtime/stochastic-belief"
ALLOWED_TOOLCHAIN = "leanprover/lean4:v4.33.0"
ALLOWED_EXT = {".lean", ".md", ".json", ".txt", ".toml"}
FORBIDDEN_MARKERS = (
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "github_pat_",
    "ghp_",
    "AKIA",
)
PROBLEM_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,80}$")
DECL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*(?:\.[A-Za-z_][A-Za-z0-9_']*)*$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_FILES = 100
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 10 * 1024 * 1024


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


def check_text_for_secrets(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="strict")
    for marker in FORBIDDEN_MARKERS:
        if marker in text:
            die(f"secret-like marker {marker!r} in {path}")


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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    required = {
        "schema", "canonical_repo", "canonical_commit", "public_repo",
        "problem_id", "toolchain", "lean_entry", "audit_declarations", "files"
    }
    if set(manifest) != required:
        die(f"manifest keys differ from required schema: {sorted(set(manifest) ^ required)}")
    if manifest["schema"] != SCHEMA:
        die("unsupported schema")
    if manifest["canonical_repo"] != CANONICAL_REPO:
        die("canonical_repo mismatch")
    if manifest["public_repo"] != PUBLIC_REPO:
        die("public_repo mismatch")
    if not isinstance(manifest["canonical_commit"], str) or not HEX40_RE.fullmatch(manifest["canonical_commit"]):
        die("canonical_commit must be lowercase 40-hex")
    pid = manifest["problem_id"]
    if not isinstance(pid, str) or not PROBLEM_RE.fullmatch(pid):
        die("invalid problem_id")
    if manifest["toolchain"] != ALLOWED_TOOLCHAIN:
        die("toolchain is not on the public CI allowlist")

    decls = manifest["audit_declarations"]
    if not isinstance(decls, list) or not (1 <= len(decls) <= 32):
        die("audit_declarations must contain 1..32 names")
    if len(set(decls)) != len(decls) or any(not isinstance(x, str) or not DECL_RE.fullmatch(x) for x in decls):
        die("invalid or duplicate audit declaration")

    entries = manifest["files"]
    if not isinstance(entries, list) or not (1 <= len(entries) <= MAX_FILES):
        die("files must contain 1..100 entries")

    seen = set()
    total = 0
    normalized = []
    prefix = PurePosixPath("payload") / pid
    for e in entries:
        if not isinstance(e, dict) or set(e) != {"path", "sha256"}:
            die("each file entry must contain exactly path and sha256")
        if not isinstance(e["path"], str) or not isinstance(e["sha256"], str):
            die("file entry types invalid")
        p = safe_rel(e["path"])
        if p in seen:
            die(f"duplicate file {p}")
        seen.add(p)
        if p.suffix.lower() not in ALLOWED_EXT:
            die(f"extension not allowed: {p}")
        try:
            p.relative_to(prefix)
        except ValueError:
            die(f"file must be under payload/{pid}/: {p}")
        if not HEX64_RE.fullmatch(e["sha256"]):
            die(f"invalid sha256 for {p}")
        real = src.joinpath(*p.parts)
        if not real.exists() or not real.is_file() or real.is_symlink():
            die(f"missing, non-regular, or symlink file: {p}")
        if not stat.S_ISREG(real.stat().st_mode):
            die(f"non-regular file: {p}")
        size = real.stat().st_size
        if size > MAX_FILE_BYTES:
            die(f"file too large: {p}")
        total += size
        if total > MAX_TOTAL_BYTES:
            die("export exceeds total size limit")
        actual = sha256(real)
        if actual != e["sha256"]:
            die(f"sha256 mismatch: {p}")
        check_text_for_secrets(real)
        normalized.append((p, real, actual, size))

    entry = safe_rel(manifest["lean_entry"])
    if entry not in seen or entry.suffix != ".lean":
        die("lean_entry must name an exported .lean file")
    if not any(str(p).endswith("README.md") for p in seen):
        die("export must contain a README.md")
    if not any(str(p).endswith("PUBLIC-PROVENANCE.json") for p in seen):
        die("export must contain PUBLIC-PROVENANCE.json")

    payload_root = src.joinpath(*prefix.parts)
    actual_files = set()
    for root, dirs, files in os.walk(payload_root, followlinks=False):
        rootp = Path(root)
        for d in list(dirs):
            if (rootp / d).is_symlink():
                die(f"symlink directory forbidden: {rootp / d}")
        for name in files:
            q = rootp / name
            actual_files.add(PurePosixPath(q.relative_to(src).as_posix()))
    if actual_files != seen:
        extra = sorted(map(str, actual_files - seen))
        missing = sorted(map(str, seen - actual_files))
        die(f"manifest/file-set mismatch extra={extra} missing={missing}")

    out = args.output.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for p, real, _, _ in normalized:
        rel_inside = p.relative_to(prefix)
        dest = out.joinpath(*rel_inside.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real, dest)

    copied_entry = out.joinpath(*entry.relative_to(prefix).parts)
    receipt = {
        "schema": 1,
        "canonical_repo": CANONICAL_REPO,
        "canonical_commit": manifest["canonical_commit"],
        "problem_id": pid,
        "toolchain": ALLOWED_TOOLCHAIN,
        "lean_entry": copied_entry.relative_to(out).as_posix(),
        "lean_entry_sha256": sha256(copied_entry),
        "audit_declarations": decls,
        "file_count": len(normalized),
        "total_bytes": total,
        "manifest_sha256": sha256(manifest_path),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
