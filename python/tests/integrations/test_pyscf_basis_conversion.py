"""Tests for PySCF basis set conversion utilities."""

# --
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --

from .pyscf_helpers import *  # noqa: F401, F403

class TestQDKChemistryPySCFBasisConversion:
    """Test suite for QDK/Chemistry-PySCF basis set conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple helium structure
        self.he_structure = Structure(["He"], np.array([[0.0, 0.0, 0.0]]))

        # Create a water structure
        self.h2o_structure = Structure(
            ["O", "H", "H"],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.757, 0.586, 0.0],
                    [-0.757, 0.586, 0.0],
                ]
            ),
        )

        # Create a hydrogen molecule structure
        self.h2_structure = Structure(
            ["H", "H"],
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.4, 0.0, 0.0],
                ]
            ),
        )

    def create_simple_basis_set(self, structure, basis_name="STO-3G"):
        """Create a simple basis set for testing."""
        scf_solver = algorithms.create("scf_solver")
        _, wavefunction = scf_solver.run(structure, 0, 1, basis_name.lower())
        return wavefunction.get_orbitals().get_basis_set()

    def test_qdk_to_pyscf_conversion_helium(self):
        """Test converting QDK/Chemistry basis set to PySCF for helium."""
        qdk_basis = self.create_simple_basis_set(self.he_structure)

        # Convert to PySCF
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)

        # Verify structure
        assert pyscf_mol.natm == 1
        assert pyscf_mol.atom_charges()[0] == 2  # Helium
        assert pyscf_mol.nao_nr() == qdk_basis.get_num_atomic_orbitals()

        # Verify coordinates match
        coords = pyscf_mol.atom_coords()
        assert np.allclose(
            coords[0], [0.0, 0.0, 0.0], rtol=float_comparison_relative_tolerance, atol=plain_text_tolerance
        )

    def test_qdk_to_pyscf_conversion_water(self):
        """Test converting QDK/Chemistry basis set to PySCF for water."""
        qdk_basis = self.create_simple_basis_set(self.h2o_structure)

        # Convert to PySCF
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)

        # Verify structure
        assert pyscf_mol.natm == 3
        assert pyscf_mol.atom_charges()[0] == 8  # Oxygen
        assert pyscf_mol.atom_charges()[1] == 1  # Hydrogen
        assert pyscf_mol.atom_charges()[2] == 1  # Hydrogen
        assert pyscf_mol.nao_nr() == qdk_basis.get_num_atomic_orbitals()

    def test_pyscf_to_qdk_conversion_helium(self):
        """Test converting PySCF molecule to QDK/Chemistry basis set for helium."""
        # Create PySCF molecule
        pyscf_mol = pyscf.gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)

        # Convert to QDK/Chemistry
        qdk_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.he_structure)

        # Verify basic properties
        assert qdk_basis.get_num_atoms() == 1
        assert qdk_basis.get_num_atomic_orbitals() == pyscf_mol.nao_nr()
        assert qdk_basis.has_structure()

    def test_pyscf_to_qdk_conversion_water(self):
        """Test converting PySCF molecule to QDK/Chemistry basis set for water."""
        # Create PySCF molecule
        pyscf_mol = pyscf.gto.M(
            atom="""
            O  0.0      0.0      0.0
            H  0.757    0.586    0.0
            H -0.757    0.586    0.0
            """,
            basis="sto-3g",
            verbose=0,
        )

        # Convert to QDK/Chemistry
        qdk_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.h2o_structure)

        # Verify basic properties
        assert qdk_basis.get_num_atoms() == 3
        assert qdk_basis.get_num_atomic_orbitals() == pyscf_mol.nao_nr()
        assert qdk_basis.has_structure()

    def test_pyscf_to_qdk_conversion_water_generally_contracted(self):
        """Test converting Pyscf molecule to QDK/Chemistry basis set for water with a generally contracted basis."""
        # Create PySCF molecule
        pyscf_mol = pyscf.gto.M(
            atom="""
             O  0.0      0.0      0.0
             H  0.757    0.586    0.0
             H -0.757    0.586    0.0
             """,
            basis="cc-pvdz",
            verbose=0,
        )

        # Convert to QDK/Chemistry
        qdk_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.h2o_structure)

        # Verify basic properties
        assert qdk_basis.get_num_atoms() == 3
        assert qdk_basis.get_num_atomic_orbitals() == pyscf_mol.nao_nr()
        assert qdk_basis.has_structure()

    def test_round_trip_conversion_helium(self):
        """Test round-trip conversion: QDK/Chemistry -> PySCF -> QDK/Chemistry for helium."""
        # Start with QDK/Chemistry basis
        original_basis = self.create_simple_basis_set(self.he_structure)

        # Convert to PySCF
        pyscf_mol = basis_to_pyscf_mol(original_basis)

        # Convert back to QDK/Chemistry
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.he_structure)

        # Compare key properties
        assert original_basis.get_num_atoms() == converted_basis.get_num_atoms()
        assert original_basis.get_num_shells() == converted_basis.get_num_shells()
        assert original_basis.get_num_atomic_orbitals() == converted_basis.get_num_atomic_orbitals()

        # Compare shell-by-shell (with more relaxed tolerances for numerical differences)
        orig_shells = original_basis.get_shells()
        conv_shells = converted_basis.get_shells()

        for i, (orig_shell, conv_shell) in enumerate(zip(orig_shells, conv_shells, strict=True)):
            assert orig_shell.atom_index == conv_shell.atom_index, f"Shell {i}: atom index mismatch"
            assert orig_shell.orbital_type == conv_shell.orbital_type, f"Shell {i}: orbital type mismatch"
            assert orig_shell.get_num_primitives() == conv_shell.get_num_primitives(), (
                f"Shell {i}: primitive count mismatch"
            )
            # Compare exponents and coefficients within tolerance (more relaxed for conversion)
            try:
                assert np.allclose(
                    orig_shell.exponents,
                    conv_shell.exponents,
                    rtol=plain_text_tolerance / 100,
                    atol=plain_text_tolerance,
                ), f"Shell {i}: exponent mismatch"
                assert np.allclose(
                    orig_shell.coefficients,
                    conv_shell.coefficients,
                    rtol=plain_text_tolerance / 100,
                    atol=plain_text_tolerance,
                ), f"Shell {i}: coefficient mismatch"
            except AssertionError:
                # If exact comparison fails, just verify they're reasonably close
                exp_diff = np.max(np.abs(orig_shell.exponents - conv_shell.exponents))
                coeff_diff = np.max(np.abs(orig_shell.coefficients - conv_shell.coefficients))
                assert exp_diff < plain_text_tolerance, f"Shell {i}: large exponent difference {exp_diff}"
                assert coeff_diff < plain_text_tolerance, f"Shell {i}: large coefficient difference {coeff_diff}"

    def test_round_trip_conversion_water(self):
        """Test round-trip conversion: QDK/Chemistry -> PySCF -> QDK/Chemistry for water."""
        # Start with QDK/Chemistry basis
        original_basis = self.create_simple_basis_set(self.h2o_structure)

        # Convert to PySCF
        pyscf_mol = basis_to_pyscf_mol(original_basis)

        # Convert back to QDK/Chemistry
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.h2o_structure)

        # Compare key properties
        assert original_basis.get_num_atoms() == converted_basis.get_num_atoms()
        assert original_basis.get_num_shells() == converted_basis.get_num_shells()
        assert original_basis.get_num_atomic_orbitals() == converted_basis.get_num_atomic_orbitals()

    def test_atomic_orbital_type_handling(self):
        """Test handling of spherical vs cartesian basis types."""
        # Create basis with spherical functions
        qdk_basis = self.create_simple_basis_set(self.he_structure)
        original_type = qdk_basis.get_atomic_orbital_type()

        # Convert to PySCF and back
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.he_structure)

        # Basis type should be preserved
        assert converted_basis.get_atomic_orbital_type() == original_type

    def test_shell_ordering_consistency(self):
        """Test that shell ordering is consistent after conversion."""
        qdk_basis = self.create_simple_basis_set(self.h2o_structure)

        # Get original shell ordering
        orig_shells = qdk_basis.get_shells()
        orig_shell_types = [shell.orbital_type for shell in orig_shells]
        orig_atom_indices = [shell.atom_index for shell in orig_shells]

        # Convert to PySCF and back
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.h2o_structure)

        # Get converted shell ordering
        conv_shells = converted_basis.get_shells()
        conv_shell_types = [shell.orbital_type for shell in conv_shells]
        conv_atom_indices = [shell.atom_index for shell in conv_shells]

        # Shell ordering should match
        assert orig_shell_types == conv_shell_types
        assert orig_atom_indices == conv_atom_indices

    def test_molecular_orbital_consistency(self):
        """Test that molecular orbitals remain consistent after basis conversion."""
        # Get QDK/Chemistry solution
        scf_solver = algorithms.create("scf_solver", "pyscf")
        qdk_energy, qdk_wavefunction = scf_solver.run(self.he_structure, 0, 1, "sto-3g")
        qdk_orbitals = qdk_wavefunction.get_orbitals()
        qdk_mos = qdk_orbitals.get_coefficients()[0]

        # Convert basis and solve with PySCF
        qdk_basis = qdk_orbitals.get_basis_set()
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)
        mf = pyscf.scf.RHF(pyscf_mol)
        mf.verbose = 0
        pyscf_energy = mf.kernel()
        pyscf_mos = mf.mo_coeff

        # Energies should match
        qdk_total_energy = qdk_energy + self.he_structure.calculate_nuclear_repulsion_energy()
        assert np.allclose(
            qdk_total_energy, pyscf_energy, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance
        )

        # MO coefficients should be similar (up to phase)
        # For helium with STO-3G, we expect 1 occupied orbital
        qdk_homo = qdk_mos[:, 0]
        pyscf_homo = pyscf_mos[:, 0]

        # Check if coefficients match (considering possible sign flip)
        overlap = np.abs(np.dot(qdk_homo, pyscf_homo))
        assert np.allclose(overlap, 1.0, rtol=float_comparison_relative_tolerance, atol=scf_orbital_tolerance)

    def test_basis_set_metadata_preservation(self):
        """Test that basis set metadata is preserved during conversion."""
        qdk_basis = self.create_simple_basis_set(self.he_structure)
        original_name = qdk_basis.get_name()

        # Convert to PySCF and back
        pyscf_mol = basis_to_pyscf_mol(qdk_basis)
        converted_basis = pyscf_mol_to_qdk_basis(pyscf_mol, self.he_structure)

        # Name should be preserved
        converted_name = converted_basis.get_name()
        assert original_name == converted_name

    def test_ecp_extraction_and_metadata(self):
        """Test that ECP shells and metadata are properly extracted from PySCF."""
        # Create a structure with heavy atoms that use ECPs
        ag_structure = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))

        # Create PySCF molecule with ECP
        pyscf_mol = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp="lanl2dz", verbose=0)

        # Convert to QDK/Chemistry basis
        qdk_basis = pyscf_mol_to_qdk_basis(pyscf_mol, ag_structure)

        # Verify ECP shells were extracted with radial powers
        assert qdk_basis.has_ecp_shells()
        assert qdk_basis.get_num_ecp_shells() > 0

        ecp_shells = qdk_basis.get_ecp_shells()
        for shell in ecp_shells:
            assert shell.has_radial_powers()
            assert len(shell.rpowers) > 0
            assert len(shell.rpowers) == len(shell.exponents) == len(shell.coefficients)

        # Verify ECP metadata
        assert qdk_basis.has_ecp_electrons()
        assert qdk_basis.get_ecp_name() == "lanl2dz"

        # Check ECP electron counts
        ecp_electrons = qdk_basis.get_ecp_electrons()
        assert len(ecp_electrons) == 1
        assert ecp_electrons[0] == 28  # LANL2DZ ECP for Ag removes 28 core electrons

    def test_ecp_roundtrip_conversion(self):
        """Test round-trip conversion of ECP shells and metadata: QDK -> PySCF -> QDK."""
        ag_structure = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))

        # Create PySCF molecule with ECP
        pyscf_mol_orig = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp="lanl2dz", verbose=0)
        qdk_basis_orig = pyscf_mol_to_qdk_basis(pyscf_mol_orig, ag_structure)

        # Store original data
        orig_ecp_shells = qdk_basis_orig.get_ecp_shells()
        orig_ecp_name = qdk_basis_orig.get_ecp_name()

        # Convert to PySCF (need to preserve charge and multiplicity)
        pyscf_mol_converted = basis_to_pyscf_mol(qdk_basis_orig, charge=0, multiplicity=2)

        # Verify qdk_ecp_name attribute was stored
        assert hasattr(pyscf_mol_converted, "qdk_ecp_name")
        assert pyscf_mol_converted.qdk_ecp_name == orig_ecp_name

        # Convert back to QDK/Chemistry
        qdk_basis_roundtrip = pyscf_mol_to_qdk_basis(pyscf_mol_converted, ag_structure)

        # Verify ECP name preserved
        assert qdk_basis_roundtrip.get_ecp_name() == orig_ecp_name

        # Verify ECP shells preserved with high precision
        assert qdk_basis_roundtrip.has_ecp_shells()
        assert qdk_basis_roundtrip.get_num_ecp_shells() == len(orig_ecp_shells)

        roundtrip_ecp_shells = qdk_basis_roundtrip.get_ecp_shells()
        for orig_shell, rt_shell in zip(orig_ecp_shells, roundtrip_ecp_shells, strict=True):
            assert orig_shell.atom_index == rt_shell.atom_index
            assert orig_shell.orbital_type == rt_shell.orbital_type
            assert orig_shell.has_radial_powers() == rt_shell.has_radial_powers()
            assert np.allclose(
                orig_shell.exponents,
                rt_shell.exponents,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                orig_shell.coefficients,
                rt_shell.coefficients,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.array_equal(orig_shell.rpowers, rt_shell.rpowers)

    def test_ecp_multi_atom_and_formats(self):
        """Test ECP handling in multi-atom systems and different PySCF ECP formats."""
        # Test 1: Multi-atom system with mixed ECP/no-ECP atoms
        agh_structure = Structure(["Ag", "H"], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        pyscf_mol_mixed = pyscf.gto.M(
            atom="Ag 0 0 0; H 2 0 0", basis={"Ag": "lanl2dz", "H": "sto-3g"}, ecp={"Ag": "lanl2dz"}, verbose=0
        )
        qdk_basis_mixed = pyscf_mol_to_qdk_basis(pyscf_mol_mixed, agh_structure)

        assert qdk_basis_mixed.has_ecp_electrons()
        assert qdk_basis_mixed.has_ecp_shells()
        ecp_electrons = qdk_basis_mixed.get_ecp_electrons()
        assert ecp_electrons == [28, 0]  # Ag with ECP, H without
        assert len(qdk_basis_mixed.get_ecp_shells_for_atom(0)) > 0  # Ag has ECP shells
        assert len(qdk_basis_mixed.get_ecp_shells_for_atom(1)) == 0  # H has no ECP shells

        # Test 2: Different ECP specification formats (string vs dict)
        ag_structure = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))
        pyscf_mol_str = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp="lanl2dz", verbose=0)
        pyscf_mol_dict = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp={"Ag": "lanl2dz"}, verbose=0)

        qdk_basis_str = pyscf_mol_to_qdk_basis(pyscf_mol_str, ag_structure)
        qdk_basis_dict = pyscf_mol_to_qdk_basis(pyscf_mol_dict, ag_structure)

        # Both formats should give identical results
        assert qdk_basis_str.get_ecp_name() == qdk_basis_dict.get_ecp_name() == "lanl2dz"
        assert qdk_basis_str.get_ecp_electrons() == qdk_basis_dict.get_ecp_electrons()
        assert qdk_basis_str.get_num_ecp_shells() == qdk_basis_dict.get_num_ecp_shells()

    def test_ecp_edge_cases(self):
        """Test ECP edge cases: shells without metadata and full structure format."""
        ag_structure = Structure(["Ag"], np.array([[0.0, 0.0, 0.0]]))

        # Edge case 1: ECP shells exist without ECP metadata
        shells = [Shell(0, OrbitalType.S, [1.0], [1.0])]
        ecp_shells = [Shell(0, OrbitalType.S, [10.0, 5.0], [50.0, 20.0], [0, 2])]
        qdk_basis_no_meta = BasisSet("test-basis", shells, ecp_shells, ag_structure, AOType.Spherical)

        assert qdk_basis_no_meta.has_ecp_shells()
        assert qdk_basis_no_meta.get_num_ecp_shells() == 1
        assert not qdk_basis_no_meta.has_ecp_electrons()  # No metadata set
        assert qdk_basis_no_meta.get_ecp_name() == "none"

        # Edge case 2: Full ECP structure format roundtrip
        pyscf_mol_orig = pyscf.gto.M(atom="Ag 0 0 0", spin=1, basis="lanl2dz", ecp="lanl2dz", verbose=0)
        qdk_basis_1 = pyscf_mol_to_qdk_basis(pyscf_mol_orig, ag_structure)

        # Convert to PySCF (creates full ECP structure dict, need to preserve charge and multiplicity)
        pyscf_mol_1 = basis_to_pyscf_mol(qdk_basis_1, charge=0, multiplicity=2)

        # Verify full structure format: [ncore, [[l, terms], ...]]
        assert isinstance(pyscf_mol_1.ecp, dict)
        assert "Ag" in pyscf_mol_1.ecp
        ecp_data = pyscf_mol_1.ecp["Ag"]
        assert isinstance(ecp_data, list)
        assert len(ecp_data) >= 2
        assert isinstance(ecp_data[0], int)  # ncore
        assert isinstance(ecp_data[1], list)  # [[l, terms], ...]

        # Convert back (should handle full structure format)
        qdk_basis_2 = pyscf_mol_to_qdk_basis(pyscf_mol_1, ag_structure)

        # Verify complete preservation
        assert qdk_basis_2.has_ecp_electrons()
        assert qdk_basis_2.has_ecp_shells()
        assert qdk_basis_2.get_ecp_name() == qdk_basis_1.get_ecp_name()
        assert qdk_basis_2.get_ecp_electrons() == qdk_basis_1.get_ecp_electrons()
        assert qdk_basis_2.get_num_ecp_shells() == qdk_basis_1.get_num_ecp_shells()

    def test_agh_def2svp_roundtrip(self):
        """Test AgH with def2-svp and ECP round-trip conversion."""
        # Setup AgH molecule with PySCF
        mol1 = pyscf.gto.Mole()
        mol1.atom = "Ag 0.0 0.0 0.0; H 0.0 0.0 1.617"
        mol1.basis = "def2-svp"
        mol1.ecp = "def2-svp"
        mol1.unit = "Angstrom"
        mol1.build()

        # Run SCF calculation
        scf1 = pyscf.scf.RHF(mol1)
        scf1.verbose = 0
        energy1 = scf1.kernel()
        assert hasattr(mol1, "_ecp")
        assert mol1._ecp

        # Create QDK Structure
        structure = Structure(symbols=["Ag", "H"], coordinates=mol1.atom_coords())

        # Convert PySCF Mole to QDK BasisSet
        qdk_basis = pyscf_mol_to_qdk_basis(mol1, structure, basis_name="def2-svp")

        # Convert QDK BasisSet back to PySCF Mole
        mol2 = basis_to_pyscf_mol(qdk_basis)
        assert hasattr(mol2, "_ecp")
        assert mol2._ecp

        # Run SCF calculation with converted basis
        scf2 = pyscf.scf.RHF(mol2)
        scf2.verbose = 0
        energy2 = scf2.kernel()

        # Verify round-trip conversion
        assert mol1.nao == mol2.nao
        assert mol1.nelectron == mol2.nelectron
        assert np.isclose(energy1, energy2, rtol=float_comparison_relative_tolerance, atol=scf_energy_tolerance)
