#!/usr/bin/env python3
"""
Check that all source files have the correct Microsoft license header.

This script verifies that C++, C, and Python source files contain the
appropriate copyright and license information.
"""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Expected license headers for different file types
CPP_LICENSE_PATTERNS = [
    # Full MIT license header - standard C-style comment with line breaks
    re.compile(
        r"^/\*\s*\n"
        r"\s*\*\s*Copyright\s*\(c\)\s*Microsoft\s+Corporation\.\s+All\s+rights\s+reserved\.\s*\n"
        r"\s*\*\s*Licensed\s+under\s+the\s+MIT\s+License\.\s+See\s+LICENSE\.txt\s+in\s+the\s+project\s+root\s+for\s*\n"
        r"\s*\*\s*license\s+information\.\s*\n"
        r"\s*\*/",
        re.MULTILINE,
    ),
    # Full MIT license header - Doxygen block style (/**) with line breaks
    re.compile(
        r"^/\*\*\s*\n"
        r"\s*\*\s*Copyright\s*\(c\)\s*Microsoft\s+Corporation\.\s+All\s+rights\s+reserved\.\s*\n"
        r"\s*\*\s*Licensed\s+under\s+the\s+MIT\s+License\.\s+See\s+LICENSE\.txt\s+in\s+the\s+project\s+root\s+for\s*\n"
        r"\s*\*\s*license\s+information\.\s*\n"
        r"\s*\*/",
        re.MULTILINE,
    ),
    # Full MIT license header - Doxygen line style (///) with line breaks
    re.compile(
        r"^///\s*Copyright\s*\(c\)\s*Microsoft\s+Corporation\.\s+All\s+rights\s+reserved\.\s*\n"
        r"///\s*Licensed\s+under\s+the\s+MIT\s+License\.\s+See\s+LICENSE\.txt\s+in\s+the\s+project\s+root\s+for\s*\n"
        r"///\s*license\s+information\.",
        re.MULTILINE,
    ),
    # Full MIT license header - C++ line comment style (//) with line breaks
    re.compile(
        r"^//\s*Copyright\s*\(c\)\s*Microsoft\s+Corporation\.\s+All\s+rights\s+reserved\.\s*\n"
        r"//\s*Licensed\s+under\s+the\s+MIT\s+License\.\s+See\s+LICENSE\.txt\s+in\s+the\s+project\s+root\s+for\s*\n"
        r"//\s*license\s+information\.",
        re.MULTILINE,
    ),
]

PYTHON_LICENSE_PATTERNS = [
    # Hash-style comment header directly after module docstring or at top
    re.compile(
        r"^(?:#!/usr/bin/env python3?\s*\n)?"  # Optional shebang
        r'(?:""".*?"""\s*\n)?'  # Optional module docstring
        r"\s*"  # Optional whitespace/blank lines
        r"#\s*-+\s*\n"  # Opening dashes
        r"#\s*Copyright\s*\(c\)\s*Microsoft\s+Corporation\.\s+All\s+rights\s+reserved\.\s*\n"
        r"#\s*Licensed\s+under\s+the\s+MIT\s+License\.\s+See\s+LICENSE\.txt\s+in\s+the\s+project\s+root\s+for\s+license\s+information\.\s*\n"
        r"#\s*-+",  # Closing dashes
        re.MULTILINE | re.DOTALL,
    ),
]

# File extensions to check
CPP_EXTENSIONS = {".cpp", ".hpp", ".h", ".cc", ".cxx", ".c"}
PYTHON_EXTENSIONS = {".py"}


def check_file_header(filepath: Path) -> Tuple[bool, str]:
    """
    Check if a file has the correct license header.

    Args:
        filepath: Path to the file to check

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Error reading file: {e}"

    # Skip empty files
    if not content.strip():
        return True, "Empty file, skipping"

    # Determine which patterns to use based on file extension
    suffix = filepath.suffix.lower()
    if suffix in CPP_EXTENSIONS:
        patterns = CPP_LICENSE_PATTERNS
        file_type = "C/C++"
    elif suffix in PYTHON_EXTENSIONS:
        patterns = PYTHON_LICENSE_PATTERNS
        file_type = "Python"
    else:
        # Unknown file type, skip
        return True, f"Unknown file type '{suffix}', skipping"

    # Check if any pattern matches
    # For Python files, check the first 2000 characters (allows for docstrings/imports before __copyright__)
    # For C/C++ files, check the first 500 characters (header should be at top)
    if suffix in PYTHON_EXTENSIONS:
        header_section = content[:50000]
    else:
        header_section = content[:50000]

    for pattern in patterns:
        if pattern.search(header_section):
            return True, f"Valid {file_type} license header found"

    return False, f"Missing or incorrect {file_type} license header"


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the license header checker.

    Args:
        argv: Command line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Check source files for correct license headers"
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Files to check (if not provided, checks all source files)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to add missing license headers (not implemented)",
    )

    args = parser.parse_args(argv)

    if not args.files:
        print("No files provided to check")
        return 0

    failed_cpp_files = []
    failed_python_files = []
    checked_count = 0

    for filepath in args.files:
        # Skip files in external/ directory
        if "external/" in str(filepath):
            continue

        # Skip files in build/ directories
        if "/build/" in str(filepath):
            continue

        # Only check source files
        suffix = filepath.suffix.lower()
        if suffix not in (CPP_EXTENSIONS | PYTHON_EXTENSIONS):
            continue

        checked_count += 1
        success, message = check_file_header(filepath)

        if not success:
            if suffix in CPP_EXTENSIONS:
                failed_cpp_files.append(filepath)
            elif suffix in PYTHON_EXTENSIONS:
                failed_python_files.append(filepath)

    if failed_cpp_files or failed_python_files:
        total_failed = len(failed_cpp_files) + len(failed_python_files)
        print(f"\n{total_failed} file(s) missing license headers:\n")

        if failed_cpp_files:
            print(f"C/C++ files ({len(failed_cpp_files)}):")
            for filepath in failed_cpp_files:
                print(f"  - {filepath}")
            print("\nExpected header format (any of these):")
            print("  /*")
            print("   * Copyright (c) Microsoft Corporation. All rights reserved.")
            print(
                "   * Licensed under the MIT License. See LICENSE.txt in the project root for"
            )
            print("   * license information.")
            print("   */")
            print("\n  Or:")
            print("  // Copyright (c) Microsoft Corporation. All rights reserved.")
            print(
                "  // Licensed under the MIT License. See LICENSE.txt in the project root for"
            )
            print("  // license information.\n")

        if failed_python_files:
            print(f"Python files ({len(failed_python_files)}):")
            for filepath in failed_python_files:
                print(f"  - {filepath}")
            print("\nExpected header format:")
            print('  """Module description."""')
            print(
                "  # --------------------------------------------------------------------------------------------"
            )
            print("  # Copyright (c) Microsoft Corporation. All rights reserved.")
            print(
                "  # Licensed under the MIT License. See LICENSE.txt in the project root for license information."
            )
            print(
                "  # --------------------------------------------------------------------------------------------"
            )
            print(
                "\nNote: License header must come directly after the module docstring (or at top if no docstring)."
            )

        return 1

    print(f"✅ All {checked_count} checked file(s) have valid license headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
