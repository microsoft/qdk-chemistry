"""Test that all documentation example scripts run without errors."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import subprocess
import sys
import unittest
from pathlib import Path
from typing import ClassVar

# Get the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "docs" / "examples"


class TestExampleScripts(unittest.TestCase):
    """Test case for all example scripts."""

    py_example_files: ClassVar[list[Path]] = []

    @classmethod
    def setUpClass(cls):
        """Collect all .py files from the examples directory."""
        if not EXAMPLES_DIR.exists():
            raise FileNotFoundError(f"Examples directory not found: {EXAMPLES_DIR}")

        cls.py_example_files = sorted(EXAMPLES_DIR.glob("*.py"))

        if not cls.py_example_files:
            raise FileNotFoundError(f"No Python example files found in {EXAMPLES_DIR}")

    def _run_python_example(self, example_file: Path):
        """Helper method to run a Python example file."""
        result = subprocess.run(
            [sys.executable, str(example_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=example_file.parent,
        )

        assert result.returncode == 0, (
            f"Example {example_file.name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


# Dynamically create test methods for each example file
def _create_test_methods():
    """Create individual test methods for each example file."""
    if EXAMPLES_DIR.exists():
        # Python examples
        py_example_files = sorted(EXAMPLES_DIR.glob("*.py"))

        for example_file in py_example_files:
            # Create a test method name from the file name
            # e.g., "basis_set.py" -> "test_py_basis_set"
            test_name = f"test_py_{example_file.stem}"

            # Create the test method
            def make_test(filepath):
                def test_method(self):
                    self._run_python_example(filepath)

                return test_method

            # Add the test method to the TestExampleScripts class
            setattr(TestExampleScripts, test_name, make_test(example_file))


# Generate test methods when the module is loaded
_create_test_methods()


if __name__ == "__main__":
    unittest.main()
