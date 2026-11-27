"""Validate introduced commits, excluding only reviewed pre-policy history."""

import os
import re
import subprocess
import sys
from pathlib import Path

BASELINE = ".github/commit-policy-baseline.txt"
ZERO = "0" * 40


def git(*args: str) -> str:
    """Run Git and return its output without surrounding whitespace."""
    return subprocess.check_output(["git", *args], encoding="utf-8").strip()


def has_commit(ref: str) -> bool:
    """Return whether Git can resolve the revision to a commit."""
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            stdout=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def baseline_commits(base_ref: str | None) -> set[str]:
    """Read the reviewed commit IDs and validate any objects present in this clone."""
    # Once adopted, a proposed change cannot exempt its own commits by editing
    # the working copy. The first adoption is reviewed with its baseline file.
    if (
        base_ref
        and subprocess.call(
            ["git", "cat-file", "-e", f"{base_ref}:{BASELINE}"],
            stderr=subprocess.DEVNULL,
        )
        == 0
    ):
        content = git("show", f"{base_ref}:{BASELINE}")
    else:
        content = Path(BASELINE).read_text(encoding="utf-8")

    width = len(git("rev-parse", "--verify", "HEAD"))
    recorded = set()
    for number, line in enumerate(content.splitlines(), 1):
        oid = line.partition("#")[0].strip()
        if not oid:
            continue
        if not re.fullmatch(rf"[0-9a-fA-F]{{{width}}}", oid) or int(oid, 16) == 0:
            msg = f"{BASELINE}:{number}: expected a full commit object ID"
            raise SystemExit(msg)
        recorded.add(oid.lower())

    # An independent history can be absent from a fork. If an object is
    # available, verify that the reviewed ID belongs to a commit.
    objects = subprocess.check_output(
        ["git", "cat-file", "--batch-check"],
        input="".join(f"{oid}\n" for oid in recorded),
        encoding="utf-8",
    )
    for line in objects.splitlines():
        oid, kind, *_ = line.split()
        if kind not in {"commit", "missing"}:
            msg = f"{BASELINE}: {oid} is a {kind}, expected a commit"
            raise SystemExit(msg)
    return recorded


def introduced_commits(base: str, head: str) -> list[str]:
    """List new commits in history order, omitting reviewed pre-policy commits."""
    if head == ZERO:  # A deleted push ref.
        return []
    base_ref = None
    if base != ZERO and has_commit(base):
        base_ref = base
    else:
        # A newly created branch has no before SHA. A force-pushed branch can
        # also lose its old tip. Exclude the default branch if it is a different
        # ref. On the default branch itself check all post-adoption commits.
        default = os.getenv("DEFAULT_BRANCH")
        if default and os.getenv("GITHUB_REF") != f"refs/heads/{default}":
            ref = f"refs/remotes/origin/{default}"
            if has_commit(ref):
                base_ref = ref
    revision_range = f"{base_ref}..{head}" if base_ref else head
    candidates = git("rev-list", "--reverse", revision_range).splitlines()
    recorded = baseline_commits(base_ref)
    return [commit for commit in candidates if commit not in recorded]


def main(base: str, head: str) -> int:
    """Run Commitizen for each introduced commit and report failures by SHA."""
    commits = introduced_commits(base, head)
    print(f"Checking {len(commits)} introduced commit(s).", flush=True)
    failed = False
    for commit in commits:
        # Validate the original commit (including its body and trailers), even
        # when a pull request was checked out at GitHub's synthetic merge ref.
        result = subprocess.run(
            [sys.executable, "-m", "commitizen", "check", "--rev-range", f"{commit}^!"],
            stdout=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode:
            print(f"Invalid commit message in {commit}.", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
