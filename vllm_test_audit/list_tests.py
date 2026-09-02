"""List Python test functions changed between two Git refs.

Usage:
    python list_tests.py <commit-a> <commit-b>

Output:
    path/to/test_file.py::ClassName::test_name
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from collections.abc import Sequence

FunctionRange = tuple[int, int, str]
TestRecord = tuple[str, str, str]
ChangedLines = dict[str, set[int]]

HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class GitCommandError(RuntimeError):
    """Raised when a required Git command fails."""


class FunctionRangeVisitor(ast.NodeVisitor):
    """Collect function ranges while retaining enclosing class names."""

    def __init__(self) -> None:
        self.class_names: list[str] = []
        self.ranges: list[FunctionRange] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class and qualify functions it encloses."""
        self.class_names.append(node.name)
        self.generic_visit(node)
        self.class_names.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record a synchronous function's source range."""
        self._add_function_range(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record an asynchronous function's source range."""
        self._add_function_range(node)
        self.generic_visit(node)

    def _add_function_range(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Add a decorator-inclusive range for one function node."""
        decorator_lines = [decorator.lineno for decorator in node.decorator_list]
        start_line = min([node.lineno, *decorator_lines])
        end_line = node.end_lineno if node.end_lineno is not None else node.lineno
        qualified_name = "::".join([*self.class_names, node.name])
        self.ranges.append((start_line, end_line, qualified_name))


def run_git(args: Sequence[str]) -> str:
    """Run Git and return standard output or raise a clear error."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        command = " ".join(["git", *args])
        detail = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise GitCommandError(f"Git command failed: {command}\n{detail}")
    return result.stdout


def get_git_diff(commit_a: str, commit_b: str) -> str:
    """Return the zero-context Python diff from commit A to commit B."""
    run_git(["rev-parse", "--verify", f"{commit_a}^{{commit}}"])
    run_git(["rev-parse", "--verify", f"{commit_b}^{{commit}}"])
    return run_git(["diff", "-U0", commit_a, commit_b, "--", "*.py"])


def parse_changed_lines(diff: str) -> tuple[ChangedLines, ChangedLines]:
    """Return changed destination and deleted source line numbers by path."""
    added_or_modified_lines: ChangedLines = {}
    deleted_lines: ChangedLines = {}
    current_file: str | None = None
    old_line = 0
    new_line = 0

    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if line.startswith("+++ /dev/null"):
            current_file = None
            continue

        hunk_match = HUNK_RE.match(line)
        if hunk_match:
            old_line = int(hunk_match.group(1))
            new_line = int(hunk_match.group(2))
            continue

        if current_file is None:
            continue
        if line.startswith("+"):
            added_or_modified_lines.setdefault(current_file, set()).add(new_line)
            new_line += 1
        elif line.startswith("-"):
            deleted_lines.setdefault(current_file, set()).add(old_line)
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
            new_line += 1

    return added_or_modified_lines, deleted_lines


def get_function_ranges(source: str) -> list[FunctionRange]:
    """Return decorator-inclusive ranges for all synchronous and async functions."""
    visitor = FunctionRangeVisitor()
    visitor.visit(ast.parse(source))
    return visitor.ranges


def get_file_at_ref(commit: str, file_path: str) -> str:
    """Return a file's contents from a Git commit."""
    return run_git(["show", f"{commit}:{file_path}"])


def changed_function_names(function_ranges: Sequence[FunctionRange], lines: set[int]) -> set[str]:
    """Return functions whose inclusive source ranges contain changed lines."""
    return {
        qualified_name
        for start_line, end_line, qualified_name in function_ranges
        if any(start_line <= line <= end_line for line in lines)
    }


def is_test_function(qualified_name: str) -> bool:
    """Return whether a qualified function name has a test_ final component."""
    function_name = qualified_name.replace("::", ".").split(".")[-1]
    return function_name.startswith("test_")


def list_changed_tests(commit_a: str, commit_b: str) -> list[TestRecord]:
    """List tests changed in the Python diff from commit A to commit B."""
    diff = get_git_diff(commit_a, commit_b)
    added_lines, deleted_lines = parse_changed_lines(diff)
    changed_tests: set[TestRecord] = set()

    for file_path in sorted(set(added_lines) | set(deleted_lines)):
        try:
            source_at_b = get_file_at_ref(commit_b, file_path)
        except GitCommandError:
            if file_path not in added_lines:
                continue
            raise

        functions_at_b = get_function_ranges(source_at_b)
        functions_by_name_at_b = {
            qualified_name: (start_line, end_line, qualified_name)
            for start_line, end_line, qualified_name in functions_at_b
        }
        changed_names = changed_function_names(functions_at_b, added_lines.get(file_path, set()))

        if file_path in deleted_lines:
            functions_at_a = get_function_ranges(get_file_at_ref(commit_a, file_path))
            deleted_names = changed_function_names(functions_at_a, deleted_lines[file_path])
            changed_names.update(deleted_names & functions_by_name_at_b.keys())

        directory, filename = os.path.split(file_path)
        for qualified_name in changed_names:
            if is_test_function(qualified_name):
                changed_tests.add((directory, filename, qualified_name))

    return sorted(changed_tests)


def pytest_node_id(directory: str, filename: str, function_name: str) -> str:
    """Return the pytest node ID for a changed test record."""
    file_path = f"{directory}/{filename}" if directory else filename
    return f"{file_path}::{function_name}"


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line interface."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: list_tests.py <commit-a> <commit-b>", file=sys.stderr)
        raise SystemExit(1)

    try:
        results = list_changed_tests(args[0], args[1])
    except (GitCommandError, SyntaxError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for directory, filename, function_name in results:
        print(pytest_node_id(directory, filename, function_name))


if __name__ == "__main__":
    main()
