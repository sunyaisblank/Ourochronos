#!/usr/bin/env python3
"""Check the reviewed usage behind the scoped RUSTSEC-2026-0295 disposition.

This is a change guard, not a formal call-graph proof. The locked z3 dependency
remains affected. Its unused ApplyResult/Tactic owner is neither constructed nor
exported by the reviewed Solver/Model/AST consumers. Any dependency or consumer
change requires renewed review before retaining the audit exception.
"""
import hashlib
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = {
    "src/temporal/global_solver.rs": "bd7a5c0461569deba905dbef5d506ace9b426755a0fdfa8154e08a0d69e618de",
    "examples/solver_reuse.rs": "69878a6f7f238b4d63ec19edcb30f0d2f90bd442cd94250034e3b3f3c4e723d4",
}
MANIFESTS = {
    "48a2902fb42c47836787d220d6878c8e73813011f7dd049d0724f379cd006ffd",
    # Cargo's normalized publication manifest has the same reviewed targets.
    "f81d4ac45befb10b5cc35e8c0f617663d399a88382727e6f49d526a8b14efc88",
}
LOCK_SHA256 = "8c9c8f6647d861afc71beb90658aa81946b88f210347817e6c27526a49e0aa7d"


def check(root=ROOT):
    if hashlib.sha256((root / "Cargo.toml").read_bytes()).hexdigest() not in MANIFESTS:
        raise ValueError("Cargo targets/dependency configuration requires renewed review")
    if hashlib.sha256((root / "Cargo.lock").read_bytes()).hexdigest() != LOCK_SHA256:
        raise ValueError("Locked dependencies require renewed advisory review")
    for name, digest in REVIEWED.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Z3 consumer requires renewed review: {name}")
    forbidden = re.compile(r"\b(?:Tactic|ApplyResult|Goal|z3_sys|Z3_apply_result\w*|Z3_tactic_apply\w*)\b")
    surface = re.compile(r"\bz3\b")
    for directory in ("src", "tests", "examples"):
        if not (root / directory).is_dir():
            raise ValueError(f"Missing reviewed source directory: {directory}")
    def unreadable(error):
        raise error

    for base, directories, files in os.walk(root, onerror=unreadable):
        if Path(base) == root:
            directories[:] = [name for name in directories if name not in ("target", ".git")]
        for directory in directories:
            if (Path(base) / directory).is_symlink():
                raise ValueError("Source directory symlink requires renewed review")
        for filename in files:
            if not filename.endswith(".rs"):
                continue
            path = Path(base) / filename
            if path.is_symlink():
                raise ValueError("Rust source symlink requires renewed review")
            source = path.read_text()
            name = path.relative_to(root).as_posix()
            if forbidden.search(source):
                raise ValueError(f"Affected or raw solver API requires renewed review: {name}")
            if surface.search(source) and name not in REVIEWED:
                raise ValueError(f"Unreviewed Z3 consumer: {name}")
    return "Reviewed Z3 dependency and Solver/Model/AST consumers are unchanged."


if __name__ == "__main__":
    print(check())
