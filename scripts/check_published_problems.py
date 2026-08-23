#!/usr/bin/env python3
"""Validate public mathematical publications before Lean compilation.

The checker deliberately enforces only properties that are mechanically
checkable: file-set completeness, bilingual/dual-format anchor identity,
minimum independent readability, provenance agreement, and absence of obvious
proof escapes.  Mathematical substance remains the responsibility of the
manuscript, Lean statement, evidence ledger, and human review together.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "Formal" / "Published"
QUARTET = ("paper.zh.tex", "paper.en.tex", "paper.zh.md", "paper.en.md")
BASE_REQUIRED = {
    "README.md",
    "PUBLIC-PROVENANCE.json",
    "metadata.json",
    "publication.json",
    "evidence.md",
    "Proof.lean",
    *QUARTET,
}
FORBIDDEN_PLACEHOLDERS = (
    "TODO",
    "TBD",
    "FIXME",
    "PLACEHOLDER",
    "待补",
    "待定",
    "此处补充",
)
ANCHOR_RE = re.compile(r"(?:<!--\s*|%\s*)([A-Z0-9-]+):\s*([^\s%<]+)")
CUSTOM_AXIOM_RE = re.compile(r"(?m)^\s*axiom\s+")
SORRY_RE = re.compile(r"(?m)\b(sorry|admit|native_decide)\b")


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostics only
        fail(f"{path}: invalid JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{path}: top-level JSON value must be an object")
    return value


def extract_anchors(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {match.group(1): match.group(2) for match in ANCHOR_RE.finditer(text[:8000])}


def check_manuscript(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    upper = text.upper()
    for marker in FORBIDDEN_PLACEHOLDERS:
        if marker.upper() in upper:
            fail(f"{path}: unfinished placeholder {marker!r}")

    # Each file must be independently readable, not a redirect or a caption.
    if len(text) < 6000:
        fail(f"{path}: manuscript is shorter than the 6000-character publication floor")

    if path.name == "paper.zh.md":
        for phrase in ("## 摘要", "证明对象", "布局双射", "结论边界"):
            if phrase not in text:
                fail(f"{path}: missing Chinese reasoning landmark {phrase!r}")
    elif path.name == "paper.en.md":
        for phrase in ("## Abstract", "Scope", "bijection", "Interpretation"):
            if phrase not in text:
                fail(f"{path}: missing English reasoning landmark {phrase!r}")
    elif path.suffix == ".tex":
        for phrase in ("\\begin{abstract}", "\\end{abstract}", "\\end{document}"):
            if phrase not in text:
                fail(f"{path}: incomplete LaTeX document; missing {phrase!r}")


def check_problem(problem: Path) -> None:
    metadata_path = problem / "metadata.json"
    publication_path = problem / "publication.json"
    metadata = load_json(metadata_path)
    publication = load_json(publication_path)
    provenance = load_json(problem / "PUBLIC-PROVENANCE.json")

    if metadata.get("schema") != 1 or publication.get("schema") != 1:
        fail(f"{problem}: unsupported schema")
    if metadata.get("problem_id") != publication.get("problem_id"):
        fail(f"{problem}: problem_id differs between metadata and publication")
    if metadata.get("model_id") != publication.get("model_id"):
        fail(f"{problem}: model_id differs between metadata and publication")
    if provenance.get("problem_id") != metadata.get("problem_id"):
        fail(f"{problem}: provenance problem_id mismatch")
    if provenance.get("model_id") != metadata.get("model_id"):
        fail(f"{problem}: provenance model_id mismatch")
    if metadata.get("directory_name") != problem.name:
        fail(f"{problem}: metadata directory_name mismatch")

    files = publication.get("files")
    if not isinstance(files, list) or len(files) != len(set(files)):
        fail(f"{problem}: publication files must be a unique list")
    if not BASE_REQUIRED.issubset(set(files)):
        fail(f"{problem}: publication allowlist omits a required artifact")
    for name in files:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            fail(f"{problem}: unsafe publication path {name!r}")
        if len(Path(name).parts) != 1:
            fail(f"{problem}: nested publication paths are not permitted")
        if not (problem / name).is_file():
            fail(f"{problem}: allowlisted file is missing: {name}")

    actual = {p.name for p in problem.iterdir() if p.is_file()}
    if actual != set(files):
        fail(
            f"{problem}: publication set mismatch; "
            f"extra={sorted(actual - set(files))}, missing={sorted(set(files) - actual)}"
        )

    anchors = metadata.get("anchors")
    if not isinstance(anchors, dict) or not anchors:
        fail(f"{problem}: metadata anchors must be a non-empty object")
    expected = {str(k): str(v) for k, v in anchors.items()}
    observed: dict[str, dict[str, str]] = {}
    for name in QUARTET:
        path = problem / name
        check_manuscript(path)
        found = extract_anchors(path)
        observed[name] = found
        if found != expected:
            fail(f"{path}: anchors differ from metadata: got={found}, expected={expected}")
    first = observed[QUARTET[0]]
    if any(observed[name] != first for name in QUARTET[1:]):
        fail(f"{problem}: four manuscript anchor maps are not identical")

    evidence = (problem / "evidence.md").read_text(encoding="utf-8")
    source = metadata.get("source")
    if not isinstance(source, dict):
        fail(f"{problem}: source metadata missing")
    for value in (source.get("commit"), source.get("torchtitan_commit")):
        if value and str(value) not in evidence:
            fail(f"{problem}/evidence.md: missing frozen source identifier {value}")
    if "does not" not in evidence.lower() and "不" not in evidence:
        fail(f"{problem}/evidence.md: interpretation boundary is not explicit")

    proof = (problem / "Proof.lean").read_text(encoding="utf-8")
    if not proof.lstrip().startswith("import Init"):
        fail(f"{problem}/Proof.lean: public proof must import Lean Core Init first")
    if CUSTOM_AXIOM_RE.search(proof):
        fail(f"{problem}/Proof.lean: custom axiom declaration found")
    if SORRY_RE.search(proof):
        fail(f"{problem}/Proof.lean: forbidden proof escape found")
    if "#print axioms" not in proof:
        fail(f"{problem}/Proof.lean: no axiom-audit hooks")

    declarations = publication.get("audit_declarations")
    if not isinstance(declarations, list) or not declarations:
        fail(f"{problem}: audit_declarations must be a non-empty list")
    if len(declarations) != len(set(declarations)):
        fail(f"{problem}: duplicate audit declaration")

    print(f"validated {problem.relative_to(ROOT)} ({metadata['model_id']})")


def main() -> int:
    if not PUBLISHED.is_dir():
        fail("Formal/Published directory missing")
    problems = sorted(
        p for p in PUBLISHED.iterdir()
        if p.is_dir() and (p / "metadata.json").is_file()
    )
    if not problems:
        fail("no published mathematical problems found")
    for problem in problems:
        check_problem(problem)
    print(f"published-problem checks passed for {len(problems)} problem(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
