"""Verify README code snippets run without errors.

This test extracts fenced Python code blocks from the repository's top-level
`README.md`, executes each snippet in an isolated temporary process, and
fails if any snippet raises an exception. Keeping README examples runnable
helps ensure documentation correctness and trustworthiness.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import re as _re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

try:
    importlib.util.find_spec("qdk_chemistry")
except ImportError as exc:
    raise ImportError("The 'qdk_chemistry' package must be installed to run this test.") from exc

# Regular expression to match fenced Python code blocks in Markdown.
# Matches ```python, ```py, or ```python3 fences.
# The code block content is captured in group 1 (the only capturing group).
# The re.DOTALL flag allows '.' to match newlines within the code block.
# The re.MULTILINE flag allows '^' and '$' to match the start and end of lines.
# The re.IGNORECASE flag makes the language specifier case-insensitive.
FENCE_RE = _re.compile(
    r"^```(?:\s*(?:python|py|python3)(?:\b[^\n]*)?)\s*\r?\n(.*?)\r?\n?^```[ \t]*$",
    _re.DOTALL | _re.IGNORECASE | _re.MULTILINE,
)


def extract_snippets(text: str) -> list[tuple[str, int]]:
    """Extract fenced Python code snippets from the given text.

    Search for the fenced code blocks in `text` and return a list of
    (code, start_line) tuples where `code` is the extracted snippet, and
    `start_line` is the line number in `text` where the snippet starts (1-based).
    Only Python code blocks (```python, ```py, or ```python3) are extracted.
    In a given text, there may be zero or more snippets.

    Args:
        text (str): The text to extract snippets from, parsed from the README.md file

    Returns:
        List of (code, start_line) tuples where `code` is the extracted snippet, and
        `start_line` is the line number in `text` where the snippet starts (1-based).

    """
    snippets = []
    for match in FENCE_RE.finditer(text):
        code = match.group(1).strip()
        start_line = text[: match.start(1)].count("\n") + 1
        snippets.append((code, start_line))
        # TODO (NAB):  change output to logger rather than print() here and elsewhere, workitem: 41426
        print(f"Extracted snippet starting at line {start_line}:\n{code}\n")
    return snippets


def run_snippet(code: str, snippet_index: int, readme_path: Path, log_dir: Path | None = None) -> bool:
    """Run a code snippet in an isolated temporary process.

    Assumes the snippet is valid Python code and attempts to execute it in a subprocess.

    Args:
        code (str): The Python code snippet to run.
        snippet_index (int): The index of the snippet (for logging purposes).
        readme_path (Path): The path to the README.md file (for logging purposes).
        log_dir (Path | None): Optional directory to write stdout/stderr logs. If None,
            no logs are written.

    Returns:
        True if the snippet ran without errors, False if it raised an exception.

    """
    with tempfile.TemporaryDirectory() as td:
        fn = Path(td) / f"snippet_{snippet_index}.py"
        fn.write_text(code, encoding="utf-8")
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            out_path = log_dir / f"snippet_{snippet_index}.stdout.txt"
            err_path = log_dir / f"snippet_{snippet_index}.stderr.txt"
            meta_path = log_dir / f"snippet_{snippet_index}.meta.txt"
            meta_path.write_text(f"readme: {readme_path}\nsnippet_index: {snippet_index}\n")
        try:
            proc = subprocess.run(
                [sys.executable, "-B", str(fn)], cwd=td, capture_output=True, text=True, check=True, timeout=60
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            if log_dir:
                out_path.write_text(getattr(exc, "stdout", "") or "")
                err_path.write_text(getattr(exc, "stderr", "") or "")
            return False
        if log_dir:
            out_path.write_text(proc.stdout or "")
            err_path.write_text(proc.stderr or "")
        return True


def test_readme_snippets_run(tmp_path):
    """Main test. Extract and run all Python code snippets from the README.md, and build a report on failures.

    Args:
        tmp_path: pytest fixture providing a temporary directory for logs.

    Raises:
        AssertionError: If any snippet fails to run, with details about each failure.

    """
    readme_path = Path(__file__).resolve().parents[2] / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    # Extract the python snippets from the README
    snippets = extract_snippets(readme_text)
    failures = []
    readme_lines = readme_text.splitlines()
    # Run each snippet and collect failures
    for i, (code, start_line) in enumerate(snippets, start=1):
        ran_ok = run_snippet(code, i, readme_path, log_dir=tmp_path)
        if not ran_ok:
            # Check for the log files
            out_path = tmp_path / f"snippet_{i}.stdout.txt"
            err_path = tmp_path / f"snippet_{i}.stderr.txt"
            meta_path = tmp_path / f"snippet_{i}.meta.txt"
            stdout = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
            stderr = err_path.read_text(encoding="utf-8") if err_path.exists() else ""
            meta = meta_path.read_text(encoding="utf-8") if meta_path.exists() else ""

            # Get the failing line number in the README if possible, by parsing the traceback
            failure_line = None
            if stderr:
                # find all 'line N' occurrences that reference the snippet file
                match = _re.findall(r"File \".*/snippet_\d+\.py\", line (\d+)", stderr)
                if match:
                    try:
                        # take the last occurrence (deepest frame in the snippet)
                        snippet_err_line = int(match[-1])
                        failure_line = start_line + snippet_err_line - 1
                    except ValueError:
                        failure_line = None

            # Extract context lines from the README around the failure line
            anchor_line = start_line if failure_line is None else failure_line
            ctx_before = 3
            ctx_after = 3
            # Compute line indexes (0-based) for slicing the README text
            start_idx = max(0, anchor_line - 1 - ctx_before)
            end_idx = min(len(readme_lines), anchor_line - 1 + ctx_after + 1)
            ctx_slice = readme_lines[start_idx:end_idx]
            # Format lines with their original line numbers (1-based) for clarity
            # Annotate the failing line for quick identification
            anchor_idx = anchor_line - 1
            formatted_lines = []
            for line_num, line in enumerate(ctx_slice, start=start_idx):
                marker = "    <-- FAIL" if line_num == anchor_idx else ""
                formatted_lines.append(f"{line_num + 1:4}: {line}{marker}")
            ctx_text = "\n".join(formatted_lines)

            failures.append(
                {
                    "index": i,
                    "start_line": start_line,
                    "code": code,
                    "meta": meta,
                    "stdout": stdout,
                    "stderr": stderr,
                    "readme_context": ctx_text,
                }
            )

    if failures:
        pieces = []
        for f in failures:
            # Format the failure report, setting up indented blocks
            meta_block = textwrap.indent(str(f.get("meta", "")), "    ")
            code_block = textwrap.indent(str(f.get("code", "")), "    ")
            context_block = textwrap.indent(str(f.get("readme_context", "")), "    ")
            stdout_block = textwrap.indent(str(f.get("stdout", "")), "    ")
            stderr_block = textwrap.indent(str(f.get("stderr", "")), "    ")

            header = "=" * 75
            sep = "-" * 76
            block_lines = [
                header,
                f"Snippet {f['index']} (starts at README line {f['start_line']})",
                sep,
                "Meta:",
                meta_block,
                sep,
                "Code:",
                code_block,
                sep,
                "ERROR context:",
                context_block,
                sep,
                "Stdout:",
                stdout_block,
                sep,
                "Stderr:",
                stderr_block,
                sep,
            ]
            pieces.append("\n".join(block_lines).rstrip())

        raise AssertionError("README snippets failed:\n" + "\n".join(pieces))
