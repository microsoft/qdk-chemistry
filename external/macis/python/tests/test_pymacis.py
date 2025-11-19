"""
Tests for PyMACIS Python bindings
"""

import os
import sys

import numpy as np
import pytest

# Add the directory containing pymacis to Python's path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pymacis
except ImportError:
    pytest.skip("pymacis not found, skipping tests", allow_module_level=True)

from prepq.library.problems.electronic import ElectronicProblem


def test_read_fcidump():
    """
    Test reading an FCIDUMP file and verifying the Hamiltonian structure
    """
    problem = ElectronicProblem("2e2o-fe_pnnp-can-2ae5327f")
    fcidump_path = str(problem.fcidump_path)
    H = pymacis.read_fcidump(fcidump_path)
    assert H is not None, "Failed to read FCIDUMP file"
    assert H.T.shape == (2, 2), "Incorrect shape for 1-body integrals"
    assert H.V.shape == (2, 2, 2, 2), "Incorrect shape for 2-body integrals"

    ref_V = np.zeros((2, 2, 2, 2))
    ref_T = np.zeros((2, 2))

    ref_V[0, 0, 0, 0] = 0.3034465207326911

    ref_V[0, 1, 0, 0] = 0.002188867979271676
    ref_V[1, 0, 0, 0] = 0.002188867979271676
    ref_V[0, 0, 0, 1] = 0.002188867979271676
    ref_V[0, 0, 1, 0] = 0.002188867979271676

    ref_V[0, 0, 1, 1] = 0.1752675845371727
    ref_V[1, 1, 0, 0] = 0.1752675845371727

    ref_V[0, 1, 0, 1] = 0.01495594810715078
    ref_V[0, 1, 1, 0] = 0.01495594810715078
    ref_V[1, 0, 0, 1] = 0.01495594810715078
    ref_V[1, 0, 1, 0] = 0.01495594810715078

    ref_V[0, 1, 1, 1] = 0.002491442225588175
    ref_V[1, 0, 1, 1] = 0.002491442225588175
    ref_V[1, 1, 0, 1] = 0.002491442225588175
    ref_V[1, 1, 1, 0] = 0.002491442225588175

    ref_V[1, 1, 1, 1] = 0.2831258252695618

    ref_T[0, 0] = -1.17200550962273
    ref_T[0, 1] = -0.03116941961754365
    ref_T[1, 0] = -0.03116941961754365
    ref_T[1, 1] = -1.053610187855433

    ref_core = -2588.786639503278

    assert np.allclose(H.T, ref_T), "1-body integrals do not match reference"
    assert np.allclose(H.V, ref_V), "2-body integrals do not match reference"
    assert np.isclose(H.core_energy, ref_core), "Core energy does not match reference"


def test_active_hamiltonian_noop():
    """
    Test that compute_active_hamiltonian returns the same Hamiltonian when full active space is specified
    """
    problem = ElectronicProblem("2e2o-fe_pnnp-can-2ae5327f")
    fcidump_path = str(problem.fcidump_path)
    H = pymacis.read_fcidump(fcidump_path)

    # Compute active Hamiltonian for 2 electrons in 2 orbitals
    H_active = pymacis.compute_active_hamiltonian(2, 0, H)

    assert H_active is not None, "Failed to compute active Hamiltonian"
    assert np.allclose(H_active.T, H.T), (
        "1-body integrals do not match in active Hamiltonian"
    )
    assert np.allclose(H_active.V, H.V), (
        "2-body integrals do not match in active Hamiltonian"
    )
    assert np.allclose(H_active.F_inactive, H.F_inactive), (
        "Inactive Fock matrix does not match"
    )
    assert np.isclose(H_active.core_energy, H.core_energy), (
        "Core energy mismatch in active Hamiltonian"
    )


def test_active_hamiltonian_2e2o_benzene():
    """
    Test the 2e2o active Hamiltonian for benzene (from 4e4o)
    """
    problem = ElectronicProblem("4e4o-benzene-can-55364ff9")
    fcidump_path = str(problem.fcidump_path)
    H = pymacis.read_fcidump(fcidump_path)

    H_active = pymacis.compute_active_hamiltonian(2, 2, H)

    assert H_active is not None, "Failed to compute active Hamiltonian"
    assert H_active.T.shape == (2, 2), (
        "Active Hamiltonian 1-body integrals shape mismatch"
    )
    assert H_active.V.shape == (2, 2, 2, 2), (
        "Active Hamiltonian 2-body integrals shape mismatch"
    )
    assert np.isclose(H_active.core_energy, -230.53576473787828), (
        "Core energy mismatch in active Hamiltonian"
    )


def test_casci_default():
    """
    Test CASCI calculation with default parameters (no determinants)
    """
    problem = ElectronicProblem("4e4o-benzene-can-55364ff9")
    fcidump_path = str(problem.fcidump_path)
    H = pymacis.read_fcidump(fcidump_path)

    # Perform CASCI calculation
    result = pymacis.casci(2, 2, H)

    assert "energy" in result, "CASCI energy not found in result"
    assert "determinants" not in result, "CASCI determinants not found in result"
    assert np.isclose(result["energy"], problem.e_cas), "CASCI energy mismatch"


def test_casci_with_determinants():
    """
    Test CASCI calculation with determinants in output
    """
    problem = ElectronicProblem("4e4o-benzene-can-55364ff9")
    fcidump_path = str(problem.fcidump_path)
    H = pymacis.read_fcidump(fcidump_path)

    # Perform CASCI calculation with determinants
    result = pymacis.casci(2, 2, H, {"return_determinants": True})

    assert "energy" in result, "CASCI energy not found in result"
    assert "determinants" in result, "CASCI determinants not found in result"
    assert len(result["determinants"]) == 36, "No determinants found in CASCI result"
    assert np.isclose(result["energy"], problem.e_cas), "CASCI energy mismatch"


def test_canonical_hf_determinant():
    """Test the creation of a canonical HF determinant"""
    norb = 4
    nalpha = 2
    nbeta = 2

    # Create canonical HF determinant
    det = pymacis.canonical_hf_determinant(nalpha, nbeta, norb)

    # Check determinant properties
    assert det == "2200"


def test_asci():
    """
    Test the ASCI calculation with a specific problem
    """
    problem = ElectronicProblem("10e10o-fe_porphyrin-can-52f9f405")
    fcidump_path = str(problem.fcidump_path)
    H = pymacis.read_fcidump(fcidump_path)

    initial_guess = [pymacis.canonical_hf_determinant(5, 5, 10)]
    C0 = [1.0]
    E0 = pymacis.compute_wfn_energy(initial_guess, C0, H)

    res = pymacis.asci(
        initial_guess,
        C0,
        E0,
        H,
        {
            "ci": {"max_subspace": 200},
            "asci": {"grow_factor": 2, "ntdets_max": 1000},
        },
    )
    assert "energy" in res, "ASCI energy not found in result"
    assert "determinants" in res, "ASCI determinants not found in result"
    assert "coefficients" in res, "ASCI coefficients not found in result"

    assert np.isclose(res["energy"], -16.71421439573152), "ASCI energy mismatch"
    assert len(res["determinants"]) == 1000, "No determinants found in ASCI result"
    assert np.isclose(np.linalg.norm(res["coefficients"]), 1.0), (
        "ASCI coefficients do not normalize to 1"
    )


def test_read_wavefunction():
    """Test that pymacis.read_wavefunction reproduces the reference CAS wave function data from ElectronicProblem"""
    # Use a problem that has a wavefunction file
    problem = ElectronicProblem("4e4o-benzene-can-55364ff9")

    # Check if CAS wavefunction file exists
    wfn_path = problem.wavefunction_cas_path
    if not wfn_path or not wfn_path.exists():
        pytest.skip(f"CAS wavefunction file not found: {wfn_path}")

    # Read using pymacis.read_wavefunction
    pymacis_result = pymacis.read_wavefunction(str(wfn_path))

    # Read using ElectronicProblem
    prepq_n_orbitals, prepq_coefficients, prepq_bitstrings = problem.read_wfn_file(
        "cas"
    )

    # Compare coefficients - this is the key comparison
    assert "coefficients" in pymacis_result, "pymacis result missing coefficients"
    pymacis_coefficients = pymacis_result["coefficients"]

    # Check that coefficients have the same length
    assert len(pymacis_coefficients) == len(prepq_coefficients), (
        f"Coefficient array length mismatch: pymacis={len(pymacis_coefficients)}, prepq={len(prepq_coefficients)}"
    )

    # Compare coefficients (allowing for small numerical differences)
    np.testing.assert_allclose(
        pymacis_coefficients,
        prepq_coefficients,
        rtol=1e-10,
        atol=1e-12,
        err_msg="Coefficients do not match between pymacis and ElectronicProblem",
    )

    # Compare number of orbitals if available in pymacis result
    if "norbitals" in pymacis_result:
        assert pymacis_result["norbitals"] == prepq_n_orbitals, (
            f"Number of orbitals mismatch: pymacis={pymacis_result['norbitals']}, prepq={prepq_n_orbitals}"
        )

    # Compare determinants/bitstrings if available in pymacis result
    if "determinants" in pymacis_result:
        pymacis_bitstrings = pymacis_result["determinants"]

        # Check that we have the same number of determinants
        assert len(pymacis_bitstrings) == len(prepq_bitstrings), (
            f"Number of determinants mismatch: pymacis={len(pymacis_bitstrings)}, prepq={len(prepq_bitstrings)}"
        )


def test_write_wavefunction(tmp_path):
    """Test that pymacis.write_wavefunction writes a valid wavefunction file"""
    # Use a problem that has a wavefunction file
    problem = ElectronicProblem("4e4o-benzene-can-55364ff9")

    # Check if CAS wavefunction file exists
    wfn_path = problem.wavefunction_cas_path
    if not wfn_path or not wfn_path.exists():
        pytest.skip(f"CAS wavefunction file not found: {wfn_path}")

    # Read using pymacis.read_wavefunction
    pymacis_result = pymacis.read_wavefunction(str(wfn_path))

    # Write to a temporary file
    temp_wfn_path = str(tmp_path / "temp_test.wfn")
    pymacis.write_wavefunction(
        temp_wfn_path,
        pymacis_result["norbitals"],
        pymacis_result["determinants"],
        pymacis_result["coefficients"],
    )

    # Read back the written wavefunction
    written_result = pymacis.read_wavefunction(temp_wfn_path)

    # Compare coefficients
    np.testing.assert_allclose(
        written_result["coefficients"],
        pymacis_result["coefficients"],
        rtol=1e-10,
        atol=1e-12,
        err_msg="Coefficients do not match after writing and reading",
    )


def test_fcidump_header_creation():
    """Test FCIDumpHeader creation and property access"""
    header = pymacis.FCIDumpHeader()

    # Test default values
    assert header.norb == 0
    assert header.nelec == 0
    assert header.ms2 == 0
    assert header.isym == 1
    assert len(header.orbsym) == 0

    # Test setting values
    header.norb = 4
    header.nelec = 4
    header.ms2 = 0  # Singlet
    header.isym = 1
    header.orbsym = [1, 1, 1, 1]

    # Verify values were set correctly
    assert header.norb == 4
    assert header.nelec == 4
    assert header.ms2 == 0
    assert header.isym == 1
    assert header.orbsym == [1, 1, 1, 1]


def test_read_fcidump_header():
    """Test reading FCIDUMP header information"""
    problem = ElectronicProblem("2e2o-fe_pnnp-can-2ae5327f")
    fcidump_path = str(problem.fcidump_path)

    header = pymacis.read_fcidump_header(fcidump_path)

    assert header.norb == 2
    assert header.nelec == 2
    assert header.ms2 == 0  # Singlet
    assert header.isym == 1
    assert len(header.orbsym) == header.norb


def test_write_fcidump_with_header(tmp_path):
    """Test writing FCIDUMP file with complete header information"""
    # Create test data
    norb = 2
    T = np.array([[1.0, 0.1], [0.1, 1.2]])
    V = np.random.random((norb, norb, norb, norb)) * 0.1
    core_energy = -5.0

    # Symmetrize V
    V = (
        V
        + np.transpose(V, (1, 0, 2, 3))
        + np.transpose(V, (0, 1, 3, 2))
        + np.transpose(V, (1, 0, 3, 2))
        + np.transpose(V, (2, 3, 0, 1))
        + np.transpose(V, (3, 2, 0, 1))
        + np.transpose(V, (2, 3, 1, 0))
        + np.transpose(V, (3, 2, 1, 0))
    ) / 8.0

    # Create header
    header = pymacis.FCIDumpHeader()
    header.norb = norb
    header.nelec = 2
    header.ms2 = 0  # Singlet
    header.isym = 1
    header.orbsym = [1, 1]

    # Write FCIDUMP file
    output_path = tmp_path / "test_with_header.fcidump"
    pymacis.write_fcidump(str(output_path), header, T, V, core_energy)

    # Verify file was created
    assert output_path.exists()

    # Read back and verify header
    read_header = pymacis.read_fcidump_header(str(output_path))
    assert read_header.norb == header.norb
    assert read_header.nelec == header.nelec
    assert read_header.ms2 == header.ms2
    assert read_header.isym == header.isym
    assert read_header.orbsym == header.orbsym

    # Read back full Hamiltonian and verify integrals
    H = pymacis.read_fcidump(str(output_path))
    assert np.allclose(H.T, T, atol=1e-14)
    assert np.allclose(H.V, V, atol=1e-14)
    assert np.isclose(H.core_energy, core_energy, atol=1e-14)


def test_write_fcidump_with_threshold(tmp_path):
    """Test writing FCIDUMP file with custom threshold"""
    # Create test data with small values near threshold
    norb = 2
    T = np.array([[1.0, 1e-14], [1e-14, 1.2]])  # Small off-diagonal
    V = np.zeros((norb, norb, norb, norb))
    V[0, 0, 0, 0] = 1.0
    V[0, 1, 0, 1] = 1e-14  # Small integral
    V[1, 0, 0, 1] = 1e-14  # Small integral
    V[1, 0, 1, 0] = 1e-14  # Small integral
    V[0, 1, 1, 0] = 1e-14  # Small integral
    V[1, 1, 1, 1] = 1.2
    core_energy = -5.0

    # Create header
    header = pymacis.FCIDumpHeader()
    header.norb = norb
    header.nelec = 2
    header.ms2 = 0
    header.isym = 1
    header.orbsym = [1, 1]

    # Write FCIDUMP file with tight threshold (should include small integrals)
    output_path1 = tmp_path / "test_tight_threshold.fcidump"
    pymacis.write_fcidump(str(output_path1), header, T, V, core_energy, 1e-16)

    # Write FCIDUMP file with loose threshold (should exclude small integrals)
    output_path2 = tmp_path / "test_loose_threshold.fcidump"
    pymacis.write_fcidump(str(output_path2), header, T, V, core_energy, 1e-13)

    # Read file contents
    with open(output_path1, "r") as f:
        content_tight = f.read()
    with open(output_path2, "r") as f:
        content_loose = f.read()

    # With tight threshold, small integrals should be present
    assert "1.00000000000000e-14" in content_tight

    # With loose threshold, small integrals should be absent
    assert "1.00000000000000e-14" not in content_loose

    # Both should have large integrals
    assert "1.00000000000000e+00" in content_tight
    assert "1.00000000000000e+00" in content_loose


def test_write_fcidump_round_trip(tmp_path):
    """Test reading and writing back an FCIDUMP file preserves data"""
    # Read existing FCIDUMP
    problem = ElectronicProblem("2e2o-fe_pnnp-can-2ae5327f")
    original_path = str(problem.fcidump_path)

    # Read header and Hamiltonian
    original_header = pymacis.read_fcidump_header(original_path)
    original_H = pymacis.read_fcidump(original_path)

    # Write to new file
    output_path = tmp_path / "roundtrip.fcidump"
    pymacis.write_fcidump(
        str(output_path),
        original_header,
        original_H.T,
        original_H.V,
        original_H.core_energy,
    )

    # Read back
    new_header = pymacis.read_fcidump_header(str(output_path))
    new_H = pymacis.read_fcidump(str(output_path))

    # Verify header preservation
    assert new_header.norb == original_header.norb
    assert new_header.nelec == original_header.nelec
    assert new_header.ms2 == original_header.ms2
    assert new_header.isym == original_header.isym
    assert new_header.orbsym == original_header.orbsym

    # Verify integral preservation
    assert np.allclose(new_H.T, original_H.T, atol=1e-14)
    assert np.allclose(new_H.V, original_H.V, atol=1e-14)
    assert np.isclose(new_H.core_energy, original_H.core_energy, atol=1e-14)


def test_write_fcidump_input_validation():
    """Test input validation for write_fcidump function"""
    norb = 2
    T_valid = np.random.random((norb, norb))
    V_valid = np.random.random((norb, norb, norb, norb))
    core_energy = 0.0

    # Create valid header
    header = pymacis.FCIDumpHeader()
    header.norb = norb
    header.nelec = 2
    header.ms2 = 0
    header.isym = 1
    header.orbsym = [1, 1]

    # Test invalid T matrix dimensions
    with pytest.raises(RuntimeError, match="T matrix must be 2-dimensional"):
        pymacis.write_fcidump(
            "test.fcidump", header, np.random.random((2,)), V_valid, core_energy
        )

    # Test non-square T matrix
    with pytest.raises(RuntimeError, match="T matrix must be square"):
        pymacis.write_fcidump(
            "test.fcidump", header, np.random.random((2, 3)), V_valid, core_energy
        )

    # Test invalid V tensor dimensions
    with pytest.raises(RuntimeError, match="V tensor must be 4-dimensional"):
        pymacis.write_fcidump(
            "test.fcidump", header, T_valid, np.random.random((2, 2, 2)), core_energy
        )

    # Test non-uniform V tensor dimensions
    with pytest.raises(RuntimeError, match="V tensor must have equal dimensions"):
        pymacis.write_fcidump(
            "test.fcidump", header, T_valid, np.random.random((2, 2, 2, 3)), core_energy
        )

    # Test dimension mismatch between T and V
    with pytest.raises(RuntimeError, match="T and V must have compatible dimensions"):
        pymacis.write_fcidump(
            "test.fcidump", header, T_valid, np.random.random((3, 3, 3, 3)), core_energy
        )

    # Test dimension mismatch with header.norb
    header_wrong = pymacis.FCIDumpHeader()
    header_wrong.norb = 3
    with pytest.raises(RuntimeError, match="Matrix dimensions must match header.norb"):
        pymacis.write_fcidump(
            "test.fcidump", header_wrong, T_valid, V_valid, core_energy
        )


def test_write_fcidump_file_format(tmp_path):
    """Test that written FCIDUMP file has correct format"""
    # Create simple test data
    norb = 2
    T = np.array([[1.0, 0.0], [0.0, 2.0]])
    V = np.zeros((norb, norb, norb, norb))
    V[0, 0, 0, 0] = 0.5
    V[1, 1, 1, 1] = 0.3
    core_energy = -1.5

    # Create header
    header = pymacis.FCIDumpHeader()
    header.norb = norb
    header.nelec = 2
    header.ms2 = 0
    header.isym = 1
    header.orbsym = [1, 2]

    # Write FCIDUMP file
    output_path = tmp_path / "format_test.fcidump"
    pymacis.write_fcidump(str(output_path), header, T, V, core_energy)

    # Read file content and verify format
    with open(output_path, "r") as f:
        content = f.read()

    # Check header format
    assert "&FCI NORB=2,NELEC=2,MS2=0," in content
    assert "ISYM=1" in content
    assert "ORBSYM=1,2" in content
    assert "&END" in content

    # Check that integrals are present
    assert (
        "5.00000000000000e-01        1        1        1        1" in content
    )  # V[0,0,0,0]
    assert (
        "3.00000000000000e-01        2        2        2        2" in content
    )  # V[1,1,1,1]
    assert (
        "1.00000000000000e+00        1        1        0        0" in content
    )  # T[0,0]
    assert (
        "2.00000000000000e+00        2        2        0        0" in content
    )  # T[1,1]
    assert (
        "-1.50000000000000e+00        0        0        0        0" in content
    )  # core energy


def test_fcidump_format_compatibility(tmp_path):
    """Test that FCIDUMP files in both formats (integral first and indices first) produce identical Hamiltonians"""
    # Create small test data for fast execution
    norb = 2
    T = np.array([[1.2, 0.3], [0.3, 1.8]])
    V = np.zeros((norb, norb, norb, norb))
    V[0, 0, 0, 0] = 0.6
    V[0, 1, 0, 1] = 0.2
    V[1, 0, 1, 0] = 0.2  # Should be same as above due to symmetry
    V[0, 1, 1, 0] = 0.2  # Should be same as above due to symmetry
    V[1, 0, 0, 1] = 0.2  # Should be same as above due to symmetry
    V[1, 1, 1, 1] = 0.7
    core_energy = -3.14159

    # Create header
    header = pymacis.FCIDumpHeader()
    header.norb = norb
    header.nelec = 2
    header.ms2 = 0
    header.isym = 1
    header.orbsym = [1, 1]

    # Create FCIDUMP file in default format (integral first)
    fcidump_integral_first = tmp_path / "test_integral_first.fcidump"
    pymacis.write_fcidump(str(fcidump_integral_first), header, T, V, core_energy)

    # Manually create FCIDUMP file in indices first format
    fcidump_indices_first = tmp_path / "test_indices_first.fcidump"
    with open(fcidump_indices_first, "w") as f:
        # Write header
        f.write("&FCI NORB=2,NELEC=2,MS2=0,\n")
        f.write("  ISYM=1,\n")
        f.write("  ORBSYM=1,1\n")
        f.write("&END\n")

        # Write integrals in indices first format: p q r s integral
        # Two-body integrals
        f.write("   1    1    1    1  6.00000000000000e-01\n")  # V[0,0,0,0]
        f.write("   1    2    1    2  2.00000000000000e-01\n")  # V[0,1,0,1]
        f.write("   1    2    2    1  2.00000000000000e-01\n")  # V[0,1,1,0]
        f.write("   2    1    1    2  2.00000000000000e-01\n")  # V[1,0,0,1]
        f.write("   2    1    2    1  2.00000000000000e-01\n")  # V[1,0,1,0]
        f.write("   2    2    2    2  7.00000000000000e-01\n")  # V[1,1,1,1]

        # One-body integrals
        f.write("   1    1    0    0  1.20000000000000e+00\n")  # T[0,0]
        f.write("   1    2    0    0  3.00000000000000e-01\n")  # T[0,1]
        f.write("   2    1    0    0  3.00000000000000e-01\n")  # T[1,0]
        f.write("   2    2    0    0  1.80000000000000e+00\n")  # T[1,1]

        # Core energy
        f.write("   0    0    0    0 -3.14159000000000e+00\n")

    # Read both files using pymacis
    H_integral_first = pymacis.read_fcidump(str(fcidump_integral_first))
    H_indices_first = pymacis.read_fcidump(str(fcidump_indices_first))

    # Verify that both formats produce identical Hamiltonians
    assert np.allclose(H_integral_first.T, H_indices_first.T, atol=1e-14), (
        "One-body integrals differ between integral-first and indices-first formats"
    )

    assert np.allclose(H_integral_first.V, H_indices_first.V, atol=1e-14), (
        "Two-body integrals differ between integral-first and indices-first formats"
    )

    assert np.isclose(
        H_integral_first.core_energy, H_indices_first.core_energy, atol=1e-14
    ), "Core energies differ between integral-first and indices-first formats"

    # Verify against expected values
    expected_T = T
    expected_V = V
    expected_core = core_energy

    assert np.allclose(H_integral_first.T, expected_T, atol=1e-14), (
        "Integral-first format T matrix doesn't match expected values"
    )
    assert np.allclose(H_indices_first.T, expected_T, atol=1e-14), (
        "Indices-first format T matrix doesn't match expected values"
    )

    assert np.allclose(H_integral_first.V, expected_V, atol=1e-14), (
        "Integral-first format V tensor doesn't match expected values"
    )
    assert np.allclose(H_indices_first.V, expected_V, atol=1e-14), (
        "Indices-first format V tensor doesn't match expected values"
    )

    assert np.isclose(H_integral_first.core_energy, expected_core, atol=1e-14), (
        "Integral-first format core energy doesn't match expected value"
    )
    assert np.isclose(H_indices_first.core_energy, expected_core, atol=1e-14), (
        "Indices-first format core energy doesn't match expected value"
    )
