#!/usr/bin/env python3
"""Validate the public, human-readable formal-problem releases."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "Formal" / "Published"
QUARTET = ("paper.zh.tex", "paper.en.tex", "paper.zh.md", "paper.en.md")
REQUIRED = (*QUARTET, "Proof.lean", "metadata.json", "publication.json", "evidence.md")
FORBIDDEN_TEXT = ("TODO", "TBD", "FIXME", "PLACEHOLDER", "待补", "待定", "此处补充")
ANCHOR_RE = re.compile(r"(?:<!--\s*|%\s*)([A-Z0-9-]+):\s*([^\s%<]+)")


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"{path}: invalid JSON: {exc}")


def anchors(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in ANCHOR_RE.finditer(text[:6000])}


def check_problem(problem: Path) -> None:
    print(f"checking {problem.relative_to(ROOT)}")
    for name in REQUIRED:
        if not (problem / name).is_file():
            fail(f"{problem}: missing {name}")

    meta = load_json(problem / "metadata.json")
    publication = load_json(problem / "publication.json")
    if meta.get("schema") != 1 or publication.get("schema") != 1:
        fail(f"{problem}: unsupported schema")
    if meta.get("problem_id") != publication.get("problem_id"):
        fail(f"{problem}: problem_id mismatch")
    if meta.get("model_id") != publication.get("model_id"):
        fail(f"{problem}: model_id mismatch")

    expected = {str(k): str(v) for k, v in meta.get("anchors", {}).items()}
    if not expected:
        fail(f"{problem}: metadata has no anchors")

    quality = meta.get("quality_contract", {})
    minimum = int(quality.get("minimum_characters", 6000))
    required_strings = quality.get("required_strings", {})

    reference = None
    for name in QUARTET:
        path = problem / name
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_TEXT:
            if marker.casefold() in text.casefold():
                fail(f"{path}: unfinished marker {marker!r}")
        if len(text) < minimum:
            fail(f"{path}: too short ({len(text)} < {minimum})")
        if path.suffix == ".tex":
            if "\\begin{abstract}" not in text or "\\end{document}" not in text:
                fail(f"{path}: incomplete LaTeX document")
        found = anchors(path)
        for key, value in expected.items():
            if found.get(key) != value:
                fail(f"{path}: anchor {key}={found.get(key)!r}, expected {value!r}")
        if reference is None:
            reference = found
        elif found != reference:
            fail(f"{problem}: quartet anchor maps differ")
        for phrase in required_strings.get(name, []):
            if phrase.casefold() not in text.casefold():
                fail(f"{path}: missing required reasoning phrase {phrase!r}")

    evidence = (problem / "evidence.md").read_text(encoding="utf-8")
    if "Lean" not in evidence or "external" not in evidence.casefold():
        fail(f"{problem}/evidence.md: formal/external boundary is not explicit")

    files = publication.get("files")
    if not isinstance(files, list) or len(files) != len(set(files)):
        fail(f"{problem}: publication files must be a unique list")
    if not set(REQUIRED).issubset(set(files)):
        fail(f"{problem}: publication allowlist omits a required artifact")
    if any("/" in name or name.startswith(".") for name in files):
        fail(f"{problem}: nested or hidden publication path")
    if any(not (problem / name).is_file() for name in files):
        fail(f"{problem}: publication allowlist names a missing file")

    proof = (problem / "Proof.lean").read_text(encoding="utf-8")
    forbidden_patterns = (
        r"(?m)^\s*sorry\b",
        r"(?m)^\s*admit\b",
        r"(?m)^\s*axiom\b",
        r"\bnative_decide\b",
    )
    for pattern in forbidden_patterns:
        if re.search(pattern, proof):
            fail(f"{problem}/Proof.lean: forbidden proof escape matching {pattern!r}")
    if "#print axioms" not in proof:
        fail(f"{problem}/Proof.lean: missing axiom audit hooks")


def main() -> int:
    if not PUBLISHED.is_dir():
        fail("Formal/Published is missing")
    problems = sorted(
        p for p in PUBLISHED.iterdir()
        if p.is_dir() and (p / "metadata.json").is_file()
    )
    if not problems:
        fail("no public formal problems found")
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
