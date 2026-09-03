"""Tests for listing changed Python tests between Git refs."""

from __future__ import annotations

import subprocess
import textwrap
from collections.abc import Mapping
from pathlib import Path

import pytest

from vllm_test_audit.list_tests import get_function_ranges, list_changed_tests, main


def git(repo: Path, *args: str) -> str:
    """Run Git in a test repository and return its standard output."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def write_files(repo: Path, files: Mapping[str, str]) -> None:
    """Write a source snapshot into a test repository."""
    for relative_path, source in files.items():
        path = repo / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip())


def commit_snapshot(repo: Path, files: Mapping[str, str], message: str) -> str:
    """Commit a source snapshot and return its commit ID."""
    write_files(repo, files)
    git(repo, "add", ".")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """Create an initialized Git repository for a two-commit comparison."""
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Test User")
    git(tmp_path, "config", "user.email", "test@example.com")
    return tmp_path


class TestGetFunctionRanges:
    """Tests for AST function range extraction."""

    def test_includes_decorators_and_class_qualification(self) -> None:
        """Report decorator-inclusive ranges for sync and async functions."""
        source = textwrap.dedent(
            """\
            @decorator
            def test_module():
                pass

            class TestSuite:
                @decorator
                async def test_async(self):
                    pass
            """
        )

        assert get_function_ranges(source) == [
            (1, 3, "test_module"),
            (6, 8, "TestSuite::test_async"),
        ]


class TestListChangedTests:
    """Tests for identifying tests changed between two commits."""

    def test_detects_module_class_and_async_tests(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Return changed module-level, class-based, and async tests."""
        before = commit_snapshot(
            git_repo,
            {
                "tests/checks.py": """
                def test_module():
                    assert value == 1

                class TestSuite:
                    def test_class(self):
                        assert value == 1

                    async def test_async(self):
                        assert value == 1
                """,
            },
            "before",
        )
        after = commit_snapshot(
            git_repo,
            {
                "tests/checks.py": """
                def test_module():
                    assert value == 2

                class TestSuite:
                    def test_class(self):
                        assert value == 2

                    async def test_async(self):
                        assert value == 2
                """,
            },
            "after",
        )
        monkeypatch.chdir(git_repo)

        assert list_changed_tests(before, after) == [
            ("tests", "checks.py", "TestSuite::test_async"),
            ("tests", "checks.py", "TestSuite::test_class"),
            ("tests", "checks.py", "test_module"),
        ]

    def test_handles_an_added_test_file_after_a_modified_file(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not treat an added file's /dev/null header as a prior-file deletion."""
        before = commit_snapshot(
            git_repo,
            {"tests/existing.py": "def test_existing():\n    assert True\n"},
            "before",
        )
        after = commit_snapshot(
            git_repo,
            {
                "tests/existing.py": "def test_existing():\n    assert False\n",
                "tests/added.py": "def test_added():\n    assert True\n",
            },
            "after",
        )
        monkeypatch.chdir(git_repo)

        assert list_changed_tests(before, after) == [
            ("tests", "added.py", "test_added"),
            ("tests", "existing.py", "test_existing"),
        ]

    @pytest.mark.parametrize(
        ("before_source", "after_source"),
        [
            ("def production():\n    return 1\n", "def production():\n    return 2\n"),
            (
                "def helper():\n    return 1\n\ndef test_kept():\n    assert True\n",
                "def helper():\n    return 2\n\ndef test_kept():\n    assert True\n",
            ),
            (
                "import first\n\ndef test_kept():\n    assert True\n",
                "import second\n\ndef test_kept():\n    assert True\n",
            ),
            (
                "import pytest\n\n@pytest.fixture\ndef value():\n    return 1\n",
                "import pytest\n\n@pytest.fixture\ndef value():\n    return 2\n",
            ),
        ],
    )
    def test_excludes_non_test_changes(
        self,
        git_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
        before_source: str,
        after_source: str,
    ) -> None:
        """Do not report production, helper, fixture, or import changes."""
        before = commit_snapshot(git_repo, {"python_file.py": before_source}, "before")
        after = commit_snapshot(git_repo, {"python_file.py": after_source}, "after")
        monkeypatch.chdir(git_repo)

        assert list_changed_tests(before, after) == []

    def test_reports_deletion_within_a_surviving_test(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Report a test when a deleted line belongs to its surviving definition."""
        before = commit_snapshot(
            git_repo,
            {"tests/test_deletions.py": "def test_survives():\n    setup()\n    assert True\n"},
            "before",
        )
        after = commit_snapshot(
            git_repo,
            {"tests/test_deletions.py": "def test_survives():\n    assert True\n"},
            "after",
        )
        monkeypatch.chdir(git_repo)

        assert list_changed_tests(before, after) == [
            ("tests", "test_deletions.py", "test_survives"),
        ]

    def test_excludes_a_deleted_test_function(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not report a test function that exists only in the old commit."""
        before = commit_snapshot(
            git_repo,
            {"tests/test_deletions.py": "def test_removed():\n    assert True\n"},
            "before",
        )
        after = commit_snapshot(git_repo, {"tests/test_deletions.py": ""}, "after")
        monkeypatch.chdir(git_repo)

        assert list_changed_tests(before, after) == []


class TestMain:
    """Tests for command-line output."""

    def test_outputs_pytest_node_ids(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Print one pytest node ID per changed test."""
        before = commit_snapshot(
            git_repo,
            {
                "tests/checks.py": """
                def test_module():
                    assert value == 1

                class TestSuite:
                    def test_method(self):
                        assert value == 1
                """,
            },
            "before",
        )
        after = commit_snapshot(
            git_repo,
            {
                "tests/checks.py": """
                def test_module():
                    assert value == 2

                class TestSuite:
                    def test_method(self):
                        assert value == 2
                """,
            },
            "after",
        )
        monkeypatch.chdir(git_repo)

        main([before, after])

        assert capsys.readouterr().out.splitlines() == [
            "tests/checks.py::TestSuite::test_method",
            "tests/checks.py::test_module",
        ]
