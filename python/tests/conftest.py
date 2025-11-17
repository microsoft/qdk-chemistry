"""Test configuration and fixtures for QDK/Chemistry Python tests."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import platform as plt
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import CasWavefunctionContainer, Configuration, Hamiltonian, Orbitals, Wavefunction
from qdk_chemistry.noise_models import QuantumErrorProfile

from .test_helpers import create_test_orbitals

# Dynamically add the build directory to Python path
# This ensures the _core module can be found regardless of platform
current_dir = Path(__file__).parent
python_dir = current_dir.parent
build_dir = python_dir / "build"

# Find the appropriate lib directory for the current Python version
if build_dir.exists():
    # Get current Python version info
    python_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    platform = sys.platform
    machine = plt.machine()

    # Try to find the most specific lib directory
    possible_patterns = [
        f"lib.{platform}-{machine}-{python_version}",
        f"lib.{platform}-{python_version}",
        f"lib.*{python_version}*",
        "lib.*",
    ]

    lib_dir_found = False
    for pattern in possible_patterns:
        for lib_dir in build_dir.glob(pattern):
            if lib_dir.is_dir():
                sys.path.insert(0, str(lib_dir))
                lib_dir_found = True
                break
        if lib_dir_found:
            break


@pytest.fixture
def basic_orbital():
    """Create a basic valid Orbitals object for testing."""
    # Restricted orbitals (3 AOs, 2 MOs)
    coeffs = np.array([[0.85, 0.15], [0.15, -0.85], [0.0, 0.0]])
    energies = np.array([-1.5, 0.8])
    occupations = np.array([2.0, 0.0])
    return Orbitals(coeffs, occupations, energies)


@pytest.fixture
def unrestricted_orbital():
    """Create an unrestricted Orbitals object for testing."""
    # Unrestricted with different alpha/beta channels (open shell)
    alpha_coeffs = np.array([[0.85, 0.15], [0.15, -0.85], [0.0, 0.0]])
    beta_coeffs = np.array([[0.75, 0.25], [0.25, -0.75], [0.0, 0.0]])
    alpha_energies = np.array([-1.5, 0.8])
    beta_energies = np.array([-1.3, 0.9])
    alpha_occupations = np.array([1.0, 0.0])
    beta_occupations = np.array([0.0, 0.0])
    return Orbitals(
        alpha_coeffs,
        beta_coeffs,
        alpha_occupations,
        beta_occupations,
        alpha_energies,
        beta_energies,
    )


@pytest.fixture
def temp_directory():
    """Create a temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_data_files_path():
    module_dir = Path(__file__).resolve().parent
    test_dir = module_dir / "test_data"
    return test_dir.resolve()


@pytest.fixture
def hamiltonian_4e4o(test_data_files_path):
    """Fixture to create the Qubit Hamiltonian for 4e4o ethylene 2det problem."""
    mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
    classical_hamiltonian = Hamiltonian.from_json_file(test_data_files_path / "ethylene_4e4o_2det.hamiltonian.json")
    return mapper.run(classical_hamiltonian)


@pytest.fixture
def hamiltonian_10e6o(test_data_files_path):
    """Fixture to create the Qubit Hamiltonian for 10e6o f2 problem."""
    mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
    classical_hamiltonian = Hamiltonian.from_json_file(test_data_files_path / "f2_10e6o.hamiltonian.json")
    return mapper.run(classical_hamiltonian)


@pytest.fixture
def wavefunction_4e4o():
    """Fixture to create the Wavefunction for 4e4o ethylene 2det problem."""
    test_orbitals = create_test_orbitals(4)
    det1 = Configuration("2200")
    det2 = Configuration("2020")
    dets = [det1, det2]
    coeffs = np.array([-0.9837947571031265, 0.17929828748875612])

    container = CasWavefunctionContainer(coeffs, dets, test_orbitals)
    return Wavefunction(container)


@pytest.fixture
def wavefunction_10e6o():
    """Fixture to create the Wavefunction for 10e6o f2 problem."""
    test_orbitals = create_test_orbitals(6)
    det1 = Configuration("222220")
    det2 = Configuration("220222")
    det3 = Configuration("222202")
    dets = [det1, det2, det3]
    coeffs = np.array([-0.9731147049456421, 0.22612369393111892, 0.04377037881377919])

    container = CasWavefunctionContainer(coeffs, dets, test_orbitals)
    return Wavefunction(container)


@pytest.fixture
def ref_energy_4e4o():
    """Fixture to provide the reference energy for 4e4o problem."""
    return -4.481197462370488


@pytest.fixture
def ref_energy_10e6o():
    """Fixture to provide the reference energy for 10e6o f2 problem."""
    return -33.347803073087476


@pytest.fixture
def simple_error_profile():
    """Fixture to create a simple QuantumErrorProfile."""
    return QuantumErrorProfile(
        name="simple_profile",
        description="A simple test error profile",
        errors={
            "h": {
                "type": "depolarizing_error",
                "rate": 0.01,
                "num_qubits": 1,
            },
            "cx": {
                "type": "depolarizing_error",
                "rate": 0.02,
                "num_qubits": 2,
            },
        },
    )
