"""Resolve the two adjacent vLLM Buildkite nightly-CI commits.

Usage:
    python -m vllm_test_audit.get_commits

The default output is ``<night-k-1-sha> <night-k-sha>``. Run
``list_tests.py`` from a local vLLM Git checkout with those arguments. This
command relies on the authenticated, read-only ``bk`` CLI profile; it never
reads or prints its token.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

BUILDKITE_PIPELINE = "ci"
VLLM_REPOSITORY = "vllm-project/vllm"
NIGHTLY_MESSAGE = "Full CI run - nightly"
FULL_SHA_RE = re.compile(r"[0-9a-f]{40}")
MAX_BUILD_PAGES = 500
BUILDS_PER_PAGE = 25


class BuildkiteError(RuntimeError):
    """Raised when Buildkite cannot provide a trustworthy nightly range."""


class BuildkiteClient(Protocol):
    """The small read-only Buildkite interface used by this module."""

    def list_build_page(self, page: int) -> list[Mapping[str, Any]]:
        """Return one page of main-branch CI builds, newest first."""


class BuildkiteCli:
    """Use the authenticated local ``bk`` CLI without exposing its token."""

    def list_build_page(self, page: int) -> list[Mapping[str, Any]]:
        endpoint = (
            f"/pipelines/{BUILDKITE_PIPELINE}/builds?branch=main"
            f"&per_page={BUILDS_PER_PAGE}&page={page}"
        )
        result = subprocess.run(
            ["bk", "api", endpoint, "--no-pager"],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip() or "unknown bk CLI error"
            raise BuildkiteError(f"Buildkite query failed for page {page}: {detail}")
        try:
            builds = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise BuildkiteError(f"Buildkite returned invalid JSON for page {page}") from error
        if not isinstance(builds, list) or not all(isinstance(build, Mapping) for build in builds):
            raise BuildkiteError(f"Buildkite returned an invalid build list for page {page}")
        return builds


@dataclass(frozen=True)
class NightlyCommit:
    """An exact vLLM source revision selected from a scheduled nightly CI build."""

    date: str
    build_id: int
    build_url: str
    build_state: str
    vllm_sha: str
    commit_url: str


def is_vllm_nightly_build(build: Mapping[str, Any]) -> bool:
    """Return whether a build is the scheduled vLLM nightly-CI lane."""
    env = build.get("env")
    return (
        build.get("message") == NIGHTLY_MESSAGE
        and build.get("source") == "schedule"
        and build.get("branch") == "main"
        and isinstance(env, Mapping)
        and env.get("NIGHTLY") == "1"
        and env.get("RUN_ALL") == "1"
    )


def verify_vllm_commit(sha: str) -> str:
    """Verify a SHA exists in vLLM and return its canonical GitHub URL."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{VLLM_REPOSITORY}/commits/{sha}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "vllm-test-audit-nightly-resolver"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            commit = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError) as error:
        raise BuildkiteError(f"Could not verify vLLM commit {sha}: {error}") from error
    if not isinstance(commit, Mapping) or commit.get("sha") != sha or not isinstance(commit.get("html_url"), str):
        raise BuildkiteError(f"vLLM did not verify commit {sha}")
    return commit["html_url"]


def nightly_commit_from_build(
    build: Mapping[str, Any], verifier: Callable[[str], str] = verify_vllm_commit
) -> NightlyCommit:
    """Validate Buildkite metadata and turn it into a verified vLLM revision."""
    build_id = build.get("number")
    commit = build.get("commit")
    created_at = build.get("created_at")
    state = build.get("state")
    web_url = build.get("web_url")
    if not isinstance(build_id, int) or not isinstance(commit, str) or not FULL_SHA_RE.fullmatch(commit):
        raise BuildkiteError("Nightly build is missing a full vLLM commit SHA or numeric build number")
    if not isinstance(created_at, str) or len(created_at) < 10:
        raise BuildkiteError(f"Nightly build {build_id} is missing its creation date")
    if not isinstance(state, str) or not isinstance(web_url, str):
        raise BuildkiteError(f"Nightly build {build_id} is missing state or web URL")
    return NightlyCommit(
        date=created_at[:10],
        build_id=build_id,
        build_url=web_url,
        build_state=state,
        vllm_sha=commit,
        commit_url=verifier(commit),
    )


def get_recent_nightly_commits(
    client: BuildkiteClient | None = None, verifier: Callable[[str], str] = verify_vllm_commit
) -> tuple[NightlyCommit, NightlyCommit]:
    """Return `(night_k_minus_1, night_k)` from scheduled Buildkite nightlies.

    Buildkite's REST list response embeds every job, so it has no lightweight
    message filter. Pages are inspected sequentially from newest to oldest and
    the lookup stops as soon as two qualifying nightlies are found.
    """
    buildkite = client or BuildkiteCli()
    newest_first: list[NightlyCommit] = []
    for page in range(1, MAX_BUILD_PAGES + 1):
        builds = buildkite.list_build_page(page)
        if not builds:
            break
        for build in builds:
            if is_vllm_nightly_build(build):
                newest_first.append(nightly_commit_from_build(build, verifier))
                if len(newest_first) == 2:
                    return newest_first[1], newest_first[0]
    raise BuildkiteError("Could not find two scheduled 'Full CI run - nightly' builds on main")


def main(argv: Sequence[str] | None = None) -> None:
    """Print adjacent nightly SHAs, or their full evidence as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit Buildkite and commit evidence as JSON")
    args = parser.parse_args(argv)
    try:
        previous, current = get_recent_nightly_commits()
    except BuildkiteError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if args.json:
        print(json.dumps({"previous": asdict(previous), "current": asdict(current)}, sort_keys=True))
    else:
        print(previous.vllm_sha, current.vllm_sha)


if __name__ == "__main__":
    main()
