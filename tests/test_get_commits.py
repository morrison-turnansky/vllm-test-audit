"""Tests for resolving adjacent vLLM Buildkite nightly commits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from vllm_test_audit.get_commits import BuildkiteError, get_recent_nightly_commits, is_vllm_nightly_build


class FakeBuildkite:
    """An ordered, paginated Buildkite response fixture."""

    def __init__(self, pages: Mapping[int, list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.requested_pages: list[int] = []

    def list_build_page(self, page: int) -> list[Mapping[str, Any]]:
        self.requested_pages.append(page)
        return self.pages.get(page, [])


def build(
    number: int,
    date: str,
    sha: str,
    *,
    message: str = "Full CI run - nightly",
    source: str = "schedule",
    nightly: str = "1",
) -> dict[str, Any]:
    return {
        "number": number,
        "commit": sha,
        "message": message,
        "source": source,
        "branch": "main",
        "env": {"NIGHTLY": nightly, "RUN_ALL": "1", "CONTINUE_ON_FAILURE": "1"},
        "state": "failed",
        "created_at": f"{date}T06:00:00.000Z",
        "web_url": f"https://buildkite.com/vllm/ci/builds/{number}",
    }


def test_selects_adjacent_scheduled_nightlies_in_diff_order() -> None:
    current = "a" * 40
    previous = "b" * 40
    client = FakeBuildkite(
        {
            1: [
                build(20, "2026-09-03", "c" * 40, source="api"),
                build(19, "2026-09-03", "d" * 40, message="PR #99 /ci run"),
            ],
            2: [build(18, "2026-09-03", current), build(17, "2026-09-02", previous)],
        }
    )

    night_k_minus_1, night_k = get_recent_nightly_commits(
        client, lambda sha: f"https://github.com/vllm-project/vllm/commit/{sha}"
    )

    assert (night_k_minus_1.date, night_k_minus_1.vllm_sha) == ("2026-09-02", previous)
    assert (night_k.date, night_k.vllm_sha) == ("2026-09-03", current)
    assert night_k.build_id == 18
    assert client.requested_pages == [1, 2]


def test_rejects_non_nightly_builds() -> None:
    assert not is_vllm_nightly_build(build(1, "2026-09-03", "a" * 40, source="api"))
    assert not is_vllm_nightly_build(build(1, "2026-09-03", "a" * 40, nightly="0"))


def test_rejects_malformed_nightly_commit() -> None:
    client = FakeBuildkite(
        {
            1: [build(2, "2026-09-03", "not-a-sha")],
            2: [build(1, "2026-09-02", "b" * 40)],
        }
    )

    with pytest.raises(BuildkiteError, match="full vLLM commit SHA"):
        get_recent_nightly_commits(client, lambda sha: sha)
