"""Prove every rubric copy is faithful to the source the dataset actually uses.

The experiment claims its judge and answer prompts are the datasets' own, and both
``run_config.json`` and the paper's provenance block cite a sha256 as evidence. A
sha256 only means something if the file it describes is the file upstream uses, so
this checks bytes -- and the two datasets need different notions of "the source":

*   **MemTrapBench** builds its prompts from ``.txt`` files it ships, so the ground
    truth is those files. The copies are compared to them byte-for-byte.
*   **PersistBench** builds its prompts from Python string literals in
    ``src/benchmark/prompts.py`` (``GENERATOR_SYSTEM_PROMPT`` and the three
    ``JUDGE_SYSTEM_PROMPT_SINGLE_*``). The copies were extracted from those
    literals, so the ground truth is the literal's *value*.

Line endings are the trap here, and they cost real debugging time once already: this
repo is checked out on Windows, where every file is CRLF. Python's tokenizer
normalises CRLF to LF inside a string literal, so an ast-extracted prompt has LF even
though the file on disk has CRLF. A copy written back in text mode therefore has the
right characters and the wrong bytes -- and ``Path.read_text()`` hides the difference
by normalising on the way in, so a hash mismatch shows up with no visible difference
anywhere. The rule this script enforces:

    a prompt file's bytes are what gets sent, and its sha256 describes exactly that.

Hence the two source kinds are compared by different means, and both the copy's own
hash and the source's hash are reported. The MemTrapBench copies keep upstream's CRLF
(that is what upstream's own runner reads); the PersistBench copies are stored LF-only
so that one hash is simultaneously the file hash and the upstream-literal hash.

Run: ``python -m external_benchmarks.audit_rubrics``
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

from . import paths

CRLF = b"\r\n"
LF = b"\n"

MANIFEST = paths.RUBRICS_ROOT / "MANIFEST.json"
MEMTRAP_ROOT = paths.ROOT / "external" / "MemTrapBench"
PERSIST_PROMPTS = (
    paths.ROOT / "external" / "structured-memory" / "src" / "benchmark" / "prompts.py"
)

#: persist rubric copy -> the literal its content was extracted from.
PERSIST_LITERALS = {
    "generator.txt": "GENERATOR_SYSTEM_PROMPT",
    "judge_cross_domain.txt": "JUDGE_SYSTEM_PROMPT_SINGLE_CROSS_DOMAIN",
    "judge_sycophancy.txt": "JUDGE_SYSTEM_PROMPT_SINGLE_SYCOPHANCY",
    "judge_beneficial.txt": "JUDGE_SYSTEM_PROMPT_SINGLE_POSITIVE_MEMORY",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalise(payload: bytes) -> bytes:
    return payload.replace(CRLF, LF)


def literal_values(path: Path) -> dict[str, str]:
    """Every top-level string literal in a module, by assignment name."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found[target.id] = node.value.value
    return found


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, list[str]]] = []

    def add(self, name: str, status: str, notes: list[str] | None = None) -> None:
        self.rows.append((name, status, notes or []))

    @property
    def failures(self) -> list[tuple[str, str, list[str]]]:
        return [row for row in self.rows if not row[1].startswith("OK")]

    def print(self) -> None:
        width = max(len(name) for name, _, _ in self.rows)
        for name, status, notes in self.rows:
            print(f"{name:{width}}  {status}")
            for note in notes:
                print(f"{'':{width}}    {note}")


def check_memtrap(report: Report, manifest: dict) -> None:
    for name, entry in sorted(manifest["memtrapbench"]["files"].items()):
        copy = paths.RUBRICS_ROOT / "memtrap" / name
        source = MEMTRAP_ROOT / entry["source"]
        if not source.is_file():
            report.add(name, "FAIL missing upstream", [str(source)])
            continue
        a, b = copy.read_bytes(), source.read_bytes()
        recorded = entry["sha256"]
        actual = sha256_bytes(a)
        notes = [f"sha256 {actual[:16]}... upstream {sha256_bytes(b)[:16]}..."]
        if recorded != actual:
            report.add(name, "FAIL manifest sha stale", notes + [
                f"manifest says {recorded[:16]}...",
            ])
        elif a == b:
            report.add(name, "OK identical to upstream file", notes)
        elif normalise(a) == normalise(b):
            report.add(name, "FAIL newline-only difference", notes + [
                "hash does not describe upstream's bytes",
            ])
        else:
            report.add(name, "FAIL content differs", notes)


def check_persist(report: Report, manifest: dict) -> None:
    values = literal_values(PERSIST_PROMPTS)
    for name, literal in PERSIST_LITERALS.items():
        copy = paths.RUBRICS_ROOT / "persist" / name
        entry = manifest["persistbench"]["files"][name]
        if literal not in values:
            report.add(name, "FAIL literal not found", [literal])
            continue
        expected = values[literal].encode("utf-8")
        a = copy.read_bytes()
        actual = sha256_bytes(a)
        notes = [
            f"sha256 {actual[:16]}... literal {sha256_bytes(expected)[:16]}...",
            f"literal {literal}",
        ]
        if a == expected:
            report.add(name, "OK identical to upstream literal", notes)
        elif normalise(a) == normalise(expected):
            report.add(name, "FAIL newline-only difference", notes + [
                "stored as CRLF; the literal tokenises to LF",
            ])
        else:
            report.add(name, "FAIL content differs", notes)
        if entry["sha256"] != actual:
            report.add(name, "FAIL manifest sha stale", [f"manifest says {entry['sha256'][:16]}..."])
    source_sha = sha256_bytes(PERSIST_PROMPTS.read_bytes())
    if source_sha != manifest["persistbench"]["source_sha256"]:
        report.add("prompts.py", "FAIL source sha stale", [f"now {source_sha[:16]}..."])


def refresh_manifest(manifest: dict) -> dict:
    """Rewrite every recorded hash from the files on disk.

    The manifest is the artefact the paper's provenance block cites, so it must be
    derived from the check rather than maintained beside it -- the previous version
    had already drifted, recording a newline-normalised hash for one source and byte
    hashes for the others. Source, commit and repo fields are preserved: those
    describe where the copies came from and no script can rediscover them.
    """
    for name, entry in manifest["memtrapbench"]["files"].items():
        copy = paths.RUBRICS_ROOT / "memtrap" / name
        entry["sha256"] = sha256_bytes(copy.read_bytes())
        entry["source_bytes"] = copy.stat().st_size
    manifest["persistbench"]["source_sha256"] = sha256_bytes(PERSIST_PROMPTS.read_bytes())
    values = literal_values(PERSIST_PROMPTS)
    for name, literal in PERSIST_LITERALS.items():
        copy = paths.RUBRICS_ROOT / "persist" / name
        entry = manifest["persistbench"]["files"][name]
        entry["sha256"] = sha256_bytes(copy.read_bytes())
        entry["literal"] = literal
        entry["source_bytes"] = copy.stat().st_size
        if literal in values:
            entry["literal_sha256"] = sha256_bytes(values[literal].encode("utf-8"))
    return manifest


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    update = "--update" in argv
    if not MANIFEST.is_file():
        print(f"missing {MANIFEST}", file=sys.stderr)
        return 2
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = Report()
    check_memtrap(report, manifest)
    check_persist(report, manifest)
    report.print()
    failures = report.failures
    print(f"\n{len(report.rows) - len(failures)}/{len(report.rows)} checks passed")

    if update:
        # Refreshing on a failed audit would launder a bad copy into a "verified"
        # hash, so the rewrite is only allowed once every content check passes.
        content_failures = [f for f in failures if "content differs" in f[1] or "missing" in f[1]]
        if content_failures:
            print("\nrefusing to update the manifest: copies differ from their source",
                  file=sys.stderr)
            return 1
        refreshed = refresh_manifest(manifest)
        MANIFEST.write_text(
            json.dumps(refreshed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"manifest refreshed: {MANIFEST}")

    return 1 if failures and not update else 0


if __name__ == "__main__":
    sys.exit(main())
