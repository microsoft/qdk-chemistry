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


def test_canonical_hf_determinant():
    """Test the creation of a canonical HF determinant"""
    norb = 4
    nalpha = 2
    nbeta = 2

    # Create canonical HF determinant
    det = pymacis.canonical_hf_determinant(nalpha, nbeta, norb)

    # Check determinant properties
    assert det == "2200"


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
