"""Comprehensive tests for the Ansatz class serialization in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import pickle

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.data import (
    Ansatz,
    CanonicalFourCenterHamiltonianContainer,
    CasWavefunctionContainer,
    Configuration,
    Hamiltonian,
    Orbitals,
    Structure,
    Wavefunction,
)

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
    scf_energy_tolerance,
)
from .test_helpers import create_test_basis_set

try:
    import pyscf  # noqa: F401

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


class TestAnsatzSerialization:
    """Test ansatz serialization and deserialization."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
        basis_set = create_test_basis_set(3, "test-ansatz")
        return Orbitals(coeffs, None, None, basis_set)

    @pytest.fixture
    def test_structure(self):
        """Create a test structure (H2 molecule)."""
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        elements = ["H", "H"]
        return Structure(positions, elements)

    @pytest.fixture
    def test_basis_set(self, test_structure):
        """Create a test basis set."""
        return create_test_basis_set(2, "test-ansatz-basis", test_structure)

    @pytest.fixture
    def test_wavefunction(self, basic_orbitals):
        """Create a test wavefunction."""
        det1 = Configuration("20")
        coeffs = np.array([1.0])  # Single determinant with coefficient 1.0

        container = CasWavefunctionContainer(coeffs, [det1], basic_orbitals)
        return Wavefunction(container)

    @pytest.fixture
    def test_ansatz(self, test_wavefunction):
        """Create a test ansatz."""
        # Create a hamiltonian using the same orbitals as the wavefunction
        orbitals = test_wavefunction.get_orbitals()
        num_orbitals = orbitals.get_num_molecular_orbitals()

        # Create simple one and two body integrals
        one_body = np.eye(num_orbitals)
        two_body = np.zeros(num_orbitals**4)
        fock = np.eye(0)

        test_hamiltonian = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.0, fock))
        return Ansatz(test_hamiltonian, test_wavefunction)

    def test_json_serialization(self, test_ansatz):
        """Test JSON serialization for ansatz."""
        # Test to_json and from_json
        json_str = test_ansatz.to_json()
        json_data = json.loads(json_str)

        # Verify essential fields
        assert "wavefunction" in json_data
        assert "hamiltonian" in json_data

        # Test round-trip serialization
        ansatz_reconstructed = Ansatz.from_json(json_str)

        # Verify nested objects are preserved
        assert ansatz_reconstructed.get_wavefunction() is not None
        assert ansatz_reconstructed.get_hamiltonian() is not None

        # Verify wavefunction properties
        orig_wf = test_ansatz.get_wavefunction()
        recon_wf = ansatz_reconstructed.get_wavefunction()
        assert recon_wf.size() == orig_wf.size()
        assert np.isclose(
            recon_wf.norm(),
            orig_wf.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify hamiltonian properties
        orig_ham = test_ansatz.get_hamiltonian()
        recon_ham = ansatz_reconstructed.get_hamiltonian()
        assert recon_ham.get_core_energy() == orig_ham.get_core_energy()
        assert recon_ham.has_one_body_integrals() == orig_ham.has_one_body_integrals()
        assert recon_ham.has_two_body_integrals() == orig_ham.has_two_body_integrals()

    def test_hdf5_serialization(self, test_ansatz, tmp_path):
        """Test HDF5 serialization for ansatz."""
        filename = tmp_path / "test_ansatz.ansatz.h5"

        # Save to HDF5 file
        test_ansatz.to_hdf5_file(str(filename))

        # Load from HDF5 file
        ansatz_reconstructed = Ansatz.from_hdf5_file(str(filename))

        # Verify nested objects are preserved
        assert ansatz_reconstructed.get_wavefunction() is not None
        assert ansatz_reconstructed.get_hamiltonian() is not None

        # Verify wavefunction properties
        orig_wf = test_ansatz.get_wavefunction()
        recon_wf = ansatz_reconstructed.get_wavefunction()
        assert recon_wf.size() == orig_wf.size()
        assert np.isclose(
            recon_wf.norm(),
            orig_wf.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify hamiltonian properties
        orig_ham = test_ansatz.get_hamiltonian()
        recon_ham = ansatz_reconstructed.get_hamiltonian()
        assert recon_ham.get_core_energy() == orig_ham.get_core_energy()
        assert recon_ham.has_one_body_integrals() == orig_ham.has_one_body_integrals()
        assert recon_ham.has_two_body_integrals() == orig_ham.has_two_body_integrals()

    def test_json_file_io(self, test_ansatz, tmp_path):
        """Test JSON file I/O."""
        filename = tmp_path / "test_ansatz.ansatz.json"

        # Save to JSON file
        test_ansatz.to_json_file(str(filename))

        # Load from JSON file
        ansatz_reconstructed = Ansatz.from_json_file(str(filename))

        # Verify nested objects are preserved
        assert ansatz_reconstructed.get_wavefunction() is not None
        assert ansatz_reconstructed.get_hamiltonian() is not None

    def test_generic_file_io(self, test_ansatz, tmp_path):
        """Test generic file I/O with different formats."""
        json_filename = tmp_path / "test_ansatz_generic.ansatz.json"
        hdf5_filename = tmp_path / "test_ansatz_generic.ansatz.h5"

        # Test JSON format
        test_ansatz.to_file(str(json_filename), "json")
        ansatz_json = Ansatz.from_file(str(json_filename), "json")
        assert ansatz_json.get_wavefunction() is not None

        # Test HDF5 format
        test_ansatz.to_file(str(hdf5_filename), "hdf5")
        ansatz_hdf5 = Ansatz.from_file(str(hdf5_filename), "hdf5")
        assert ansatz_hdf5.get_wavefunction() is not None

        # Test invalid format
        with pytest.raises(RuntimeError, match="Unsupported file type"):
            test_ansatz.to_file(str(tmp_path / "test.ansatz.xyz"), "xyz")

        with pytest.raises(RuntimeError, match="Unsupported file type"):
            Ansatz.from_file(str(tmp_path / "test.ansatz.xyz"), "xyz")

    def test_error_handling(self):
        """Test error handling for malformed data."""
        # Test malformed JSON string
        bad_json_str = json.dumps({"wavefunction": "invalid"})

        with pytest.raises(RuntimeError):
            Ansatz.from_json(bad_json_str)

        # Test non-existent files
        with pytest.raises(RuntimeError):
            Ansatz.from_json_file("non_existent.ansatz.json")

        with pytest.raises(RuntimeError):
            Ansatz.from_hdf5_file("non_existent.ansatz.h5")

    def test_nested_serialization_consistency(self, test_ansatz):
        """Test that nested objects maintain consistency through serialization."""
        # Serialize ansatz
        json_str = test_ansatz.to_json()
        ansatz_reconstructed = Ansatz.from_json(json_str)

        # Verify that the orbitals in the wavefunction and hamiltonian are consistent
        recon_wf = ansatz_reconstructed.get_wavefunction()
        recon_ham = ansatz_reconstructed.get_hamiltonian()

        recon_wf_orbitals = recon_wf.get_orbitals()
        recon_ham_orbitals = recon_ham.get_orbitals()

        # They should have the same basis set properties
        wf_basis = recon_wf_orbitals.get_basis_set()
        ham_basis = recon_ham_orbitals.get_basis_set()

        assert wf_basis.get_num_atomic_orbitals() == ham_basis.get_num_atomic_orbitals()
        assert wf_basis.get_name() == ham_basis.get_name()

    def test_repr_method(self, test_ansatz):
        """Test that __repr__ returns the summary."""
        repr_str = repr(test_ansatz)
        summary_str = test_ansatz.get_summary()
        assert repr_str == summary_str
        assert "Ansatz" in repr_str

    def test_str_method(self, test_ansatz):
        """Test that __str__ returns the summary."""
        str_str = str(test_ansatz)
        summary_str = test_ansatz.get_summary()
        assert str_str == summary_str
        assert "Ansatz" in str_str

    def test_pickling_ansatz(self, test_ansatz):
        """Test that Ansatz can be pickled and unpickled correctly."""
        # Test pickling round-trip
        pickled_data = pickle.dumps(test_ansatz)
        ansatz_restored = pickle.loads(pickled_data)

        # Verify core properties
        assert ansatz_restored.has_hamiltonian() == test_ansatz.has_hamiltonian()
        assert ansatz_restored.has_wavefunction() == test_ansatz.has_wavefunction()
        assert ansatz_restored.has_orbitals() == test_ansatz.has_orbitals()

        # Verify hamiltonian data
        if test_ansatz.has_hamiltonian():
            orig_ham = test_ansatz.get_hamiltonian()
            restored_ham = ansatz_restored.get_hamiltonian()
            assert orig_ham.get_core_energy() == restored_ham.get_core_energy()

            if orig_ham.has_one_body_integrals():
                assert np.array_equal(orig_ham.get_one_body_integrals()[0], restored_ham.get_one_body_integrals()[0])
                assert np.array_equal(orig_ham.get_one_body_integrals()[1], restored_ham.get_one_body_integrals()[1])

        # Verify wavefunction data
        if test_ansatz.has_wavefunction():
            orig_wf = test_ansatz.get_wavefunction()
            restored_wf = ansatz_restored.get_wavefunction()
            assert orig_wf.size() == restored_wf.size()
            assert np.isclose(
                orig_wf.norm(),
                restored_wf.norm(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

            # Verify determinants
            orig_dets = orig_wf.get_active_determinants()
            restored_dets = restored_wf.get_active_determinants()
            assert len(orig_dets) == len(restored_dets)

            # Verify coefficients for each determinant
            for det in orig_dets:
                orig_coeff = orig_wf.get_coefficient(det)
                restored_coeff = restored_wf.get_coefficient(det)
                if isinstance(orig_coeff, complex):
                    assert np.isclose(
                        orig_coeff,
                        restored_coeff,
                        rtol=float_comparison_relative_tolerance,
                        atol=float_comparison_absolute_tolerance,
                    )
                else:
                    assert np.isclose(
                        orig_coeff,
                        restored_coeff,
                        rtol=float_comparison_relative_tolerance,
                        atol=float_comparison_absolute_tolerance,
                    )

        # Verify orbital consistency
        if test_ansatz.has_orbitals():
            orig_orbs = test_ansatz.get_orbitals()
            restored_orbs = ansatz_restored.get_orbitals()
            assert orig_orbs.get_num_molecular_orbitals() == restored_orbs.get_num_molecular_orbitals()
            assert np.array_equal(orig_orbs.get_coefficients(), restored_orbs.get_coefficients())

    def test_restricted_closed_shell_energy(self):
        """Test the energy evaluation for a restricted closed-shell system."""
        mol = Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], ["N", "N"])
        # get scf energy and wfn
        scf = algorithms.create("scf_solver")
        e_scf, hf_wfn = scf.run(mol, 0, 1, "cc-pvdz")
        # get hamiltonian
        h_ctor = algorithms.create("hamiltonian_constructor")
        hamiltonian = h_ctor.run(hf_wfn.get_orbitals())
        # get ansatz
        ansatz = Ansatz(hamiltonian, hf_wfn)
        e_rhf = ansatz.calculate_energy()
        # energy from ansatz should reproduce scf energy
        assert np.isclose(e_rhf, e_scf, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)

    @pytest.mark.skipif(not PYSCF_AVAILABLE, reason="pyscf not available")
    def test_restricted_open_shell_energy(self):
        """Test the energy evaluation for a restricted open-shell system."""
        try:
            import qdk_chemistry.plugins.pyscf as pyscf_plugin  # noqa: PLC0415

            pyscf_plugin.load()
        except ImportError:
            pytest.skip("pyscf not available, skipping O2 triplet Ansatz test")

        mol = Structure([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], ["O", "O"])
        # get scf energy and wfn
        scf = algorithms.create("scf_solver", "pyscf")
        scf.settings().set("scf_type", "restricted")
        e_scf, hf_wfn = scf.run(mol, 0, 3, "cc-pvdz")
        # get hamiltonian
        h_ctor = algorithms.create("hamiltonian_constructor")
        hamiltonian = h_ctor.run(hf_wfn.get_orbitals())
        # get ansatz
        ansatz = Ansatz(hamiltonian, hf_wfn)
        e_rohf = ansatz.calculate_energy()
        # energy from ansatz should reproduce scf energy
        assert np.isclose(e_rohf, e_scf, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)


def test_ansatz_data_type_name():
    """Test that Ansatz has the correct _data_type_name class attribute."""
    assert hasattr(Ansatz, "_data_type_name")
    assert Ansatz._data_type_name == "ansatz"
