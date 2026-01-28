"""Test that all documentation example scripts run without errors."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from typing import ClassVar

from qdk_chemistry.plugins.qiskit import (
    QDK_CHEMISTRY_HAS_QISKIT,
    QDK_CHEMISTRY_HAS_QISKIT_AER,
    QDK_CHEMISTRY_HAS_QISKIT_NATURE,
)

# Get the examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "docs" / "source" / "_static" / "examples"
PYTHON_EXAMPLES_DIR = EXAMPLES_DIR / "python"

PYSCF_AVAILABLE = importlib.util.find_spec("pyscf") is not None


def check_example_requirements(example_file: Path) -> tuple[bool, bool, bool, bool]:
    """Check if an example file requires qiskit or pyscf.

    Args:
        example_file: Path to the example file to check

    Returns:
        Tuple of (requires_pyscf, requires_qiskit)

    """
    content = example_file.read_text()

    requires_pyscf = False
    requires_qiskit = False
    requires_qiskit_aer = False
    requires_qiskit_nature = False

    # Check for explicit imports
    if "import pyscf" in content or "from pyscf" in content:
        requires_pyscf = True

    if "import qiskit" in content or "from qiskit" in content:
        requires_qiskit = True

    # Check for plugin usage patterns in create() calls
    # Look for create(..., "pyscf") or create(..., 'pyscf') patterns
    if ', "pyscf' in content or ", 'pyscf" in content:
        requires_pyscf = True

    if ', "qiskit' in content or ", 'qiskit" in content:
        requires_qiskit = True

    # Look for create(..., algorithm_name="pyscf") or create(..., algorithm_name='pyscf') patterns
    if 'algorithm_name="pyscf' in content or "algorithm_name='pyscf" in content:
        requires_pyscf = True
    if 'algorithm_name="qiskit' in content or "algorithm_name='qiskit" in content:
        requires_qiskit = True

    if any(
        pattern in content
        for pattern in [
            "qiskit_regular_isometry",
            "qiskit_standard",
            "QiskitStandardPhaseEstimation",
            "RegularIsometryStatePreparation",
        ]
    ):
        requires_qiskit = True

    # check for plugin imports
    if "import qdk_chemistry.plugins.pyscf" in content:
        requires_pyscf = True
    if "import qdk_chemistry.plugins.qiskit" in content:
        requires_qiskit = True

    if any(
        pattern in content
        for pattern in [
            'create("qubit_mapper", "qiskit"',
            "create('qubit_mapper', 'qiskit'",
            'create("qubit_mapper", algorithm_name="qiskit"',
            "create('qubit_mapper', algorithm_name='qiskit'",
            "QiskitQubitMapper ",
        ]
    ):
        requires_qiskit_nature = True

    if any(
        pattern in content
        for pattern in [
            "qiskit_aer_simulator",
            "QiskitEnergyEstimator",
            "QiskitAerSimulator",
        ]
    ):
        requires_qiskit_aer = True

    return requires_pyscf, requires_qiskit, requires_qiskit_aer, requires_qiskit_nature


class TestExampleScripts(unittest.TestCase):
    """Test case for all example scripts."""

    py_example_files: ClassVar[list[Path]] = []

    @classmethod
    def setUpClass(cls):
        """Collect all .py files from the examples directory."""
        if not PYTHON_EXAMPLES_DIR.exists():
            raise FileNotFoundError(f"Python examples directory not found: {PYTHON_EXAMPLES_DIR}")

        cls.py_example_files = sorted(PYTHON_EXAMPLES_DIR.glob("*.py"))

        if not cls.py_example_files:
            raise FileNotFoundError(f"No Python example files found in {PYTHON_EXAMPLES_DIR}")

    def _run_python_example(self, example_file: Path):
        """Helper method to run a Python example file."""
        result = subprocess.run(
            [sys.executable, str(example_file)],
            check=False,
            capture_output=True,
            text=True,
            timeout=360,
            cwd=example_file.parent,
        )

        assert result.returncode == 0, (
            f"Example {example_file.name} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


# Dynamically create test methods for each example file
def _create_test_methods():
    """Create individual test methods for each example file."""
    if PYTHON_EXAMPLES_DIR.exists():
        # Python examples
        py_example_files = sorted(PYTHON_EXAMPLES_DIR.glob("*.py"))

        for example_file in py_example_files:
            # Create a test method name from the file name
            # e.g., "basis_set.py" -> "test_py_basis_set"
            test_name = f"test_py_{example_file.stem}"

            # Check requirements for this example
            requires_pyscf, requires_qiskit, requires_qiskit_aer, requires_qiskit_nature = check_example_requirements(
                example_file
            )

            # Create the test method
            def make_test(filepath, needs_pyscf, needs_qiskit, needs_qiskit_aer, needs_qiskit_nature):
                """Create a test method for the given example file."""

                def test_method(self):
                    """Test the example file runs without errors."""
                    # Skip if required packages are not available
                    if needs_pyscf and not PYSCF_AVAILABLE:
                        self.skipTest("PySCF not available")
                    if needs_qiskit and not QDK_CHEMISTRY_HAS_QISKIT:
                        self.skipTest("Qiskit not available")
                    if needs_qiskit_aer and not QDK_CHEMISTRY_HAS_QISKIT_AER:
                        self.skipTest("Qiskit Aer not available")
                    if needs_qiskit_nature and not QDK_CHEMISTRY_HAS_QISKIT_NATURE:
                        self.skipTest("Qiskit Nature not available")

                    self._run_python_example(filepath)

                return test_method

            # Add the test method to the TestExampleScripts class
            setattr(
                TestExampleScripts,
                test_name,
                make_test(example_file, requires_pyscf, requires_qiskit, requires_qiskit_aer, requires_qiskit_nature),
            )


# Generate test methods when the module is loaded
_create_test_methods()


if __name__ == "__main__":
    unittest.main()
