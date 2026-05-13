"""Tests for spin-S operator construction and higher-spin model Hamiltonians."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import numpy as np
import pytest

from qdk_chemistry.data import LatticeGraph, QubitHamiltonian
from qdk_chemistry.utils.model_hamiltonians import create_heisenberg_hamiltonian, create_ising_hamiltonian
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
from qdk_chemistry.utils.spin_operators import SpinEncoding, SpinSOperators, _spin_matrices


class TestSpinMatrices:
    """Test angular momentum matrix construction."""

    @pytest.mark.parametrize("spin", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    def test_commutation_relations(self, spin):
        """[Sˣ, Sʸ] = iSᶻ (and cyclic permutations)."""
        sx, sy, sz = _spin_matrices(spin)
        atol = 1e-12

        assert np.allclose(sx @ sy - sy @ sx, 1j * sz, atol=atol)
        assert np.allclose(sy @ sz - sz @ sy, 1j * sx, atol=atol)
        assert np.allclose(sz @ sx - sx @ sz, 1j * sy, atol=atol)

    @pytest.mark.parametrize("spin", [0.5, 1.0, 1.5, 2.0])
    def test_hermiticity(self, spin):
        """Spin matrices must be Hermitian."""
        sx, sy, sz = _spin_matrices(spin)
        atol = 1e-14
        assert np.allclose(sx, sx.conj().T, atol=atol)
        assert np.allclose(sy, sy.conj().T, atol=atol)
        assert np.allclose(sz, sz.conj().T, atol=atol)

    @pytest.mark.parametrize("spin", [0.5, 1.0, 1.5, 2.0])
    def test_casimir(self, spin):
        """S² = Sˣ² + Sʸ² + Sᶻ² = S(S+1)·I."""
        sx, sy, sz = _spin_matrices(spin)
        s_squared = sx @ sx + sy @ sy + sz @ sz
        expected = spin * (spin + 1) * np.eye(int(2 * spin + 1))
        assert np.allclose(s_squared, expected, atol=1e-12)

    def test_spin_half_matches_pauli(self):
        """For S=1/2, spin operators are Pauli/2."""
        sx, sy, sz = _spin_matrices(0.5)
        assert np.allclose(sx, np.array([[0, 0.5], [0.5, 0]]))
        assert np.allclose(sy, np.array([[0, -0.5j], [0.5j, 0]]))
        assert np.allclose(sz, np.array([[0.5, 0], [0, -0.5]]))

    def test_spin_one_eigenvalues(self):
        """Sᶻ for S=1 has eigenvalues {-1, 0, 1}."""
        _, _, sz = _spin_matrices(1.0)
        eigenvalues = np.sort(np.linalg.eigvalsh(sz))
        assert np.allclose(eigenvalues, [-1.0, 0.0, 1.0], atol=1e-14)


class TestSpinSOperators:
    """Test Pauli decomposition of spin-S operators."""

    @pytest.mark.parametrize("spin", [0.5, 1.0, 1.5, 2.0])
    def test_pauli_decomposition_roundtrip(self, spin):
        """Reconstruct matrix from Pauli decomposition and compare to original."""
        dim = int(2 * spin + 1)
        num_qubits = math.ceil(math.log2(dim)) if dim > 1 else 1
        total_qubits = num_qubits

        ops = SpinSOperators(spin, qubit_offset=0)

        for op_name, op_expr in [("sx", ops.sx), ("sy", ops.sy), ("sz", ops.sz)]:
            simplified = op_expr.simplify()
            terms = simplified.to_canonical_terms(total_qubits)
            pauli_strings = [t[1][::-1] for t in terms]
            coefficients = np.array([complex(t[0]) for t in terms])

            mat_reconstructed = pauli_to_dense_matrix(pauli_strings, coefficients)

            # Compare against the original spin matrix (embedded)
            sx_orig, sy_orig, sz_orig = _spin_matrices(spin)
            orig = {"sx": sx_orig, "sy": sy_orig, "sz": sz_orig}[op_name]
            dim_full = 1 << num_qubits
            embedded = np.zeros((dim_full, dim_full), dtype=np.complex128)
            embedded[:dim, :dim] = orig

            assert np.allclose(mat_reconstructed, embedded, atol=1e-12), (
                f"Pauli decomposition roundtrip failed for {op_name} with spin={spin}"
            )

    def test_spin_half_gives_pauli(self):
        """S=1/2 decomposition gives σ/2 coefficients."""
        ops = SpinSOperators(0.5, qubit_offset=0)
        sx_terms = ops.sx.simplify().to_canonical_terms(1)
        sy_terms = ops.sy.simplify().to_canonical_terms(1)
        sz_terms = ops.sz.simplify().to_canonical_terms(1)

        # Sˣ = 0.5 * X
        assert len(sx_terms) == 1
        assert sx_terms[0][1] == "X"
        assert abs(complex(sx_terms[0][0]) - 0.5) < 1e-14

        # Sʸ = 0.5 * Y
        assert len(sy_terms) == 1
        assert sy_terms[0][1] == "Y"
        assert abs(complex(sy_terms[0][0]) - 0.5) < 1e-14

        # Sᶻ = 0.5 * Z
        assert len(sz_terms) == 1
        assert sz_terms[0][1] == "Z"
        assert abs(complex(sz_terms[0][0]) - 0.5) < 1e-14

    def test_qubit_offset(self):
        """Operators with offset act on the correct qubits."""
        ops = SpinSOperators(0.5, qubit_offset=3)
        sz_terms = ops.sz.simplify().to_canonical_terms(5)
        # Should be Z on qubit 3, I everywhere else
        assert len(sz_terms) == 1
        # to_canonical_terms returns little-endian: index k = qubit k
        label_le = sz_terms[0][1]
        assert label_le[3] == "Z"
        for k in range(5):
            if k != 3:
                assert label_le[k] == "I"

    @pytest.mark.parametrize("spin", [1.0, 2.0, 2.5])
    def test_has_unphysical_states(self, spin):
        """Non-power-of-2 dims flag unphysical states."""
        ops = SpinSOperators(spin, qubit_offset=0)
        dim = int(2 * spin + 1)
        num_qubits = math.ceil(math.log2(dim))
        expected = dim < (1 << num_qubits)
        assert ops.has_unphysical_states == expected

    def test_no_unphysical_states_power_of_two(self):
        """S=1/2 (dim=2), S=3/2 (dim=4), S=7/2 (dim=8) have no unphysical states."""
        for spin in [0.5, 1.5, 3.5]:
            ops = SpinSOperators(spin, qubit_offset=0)
            assert not ops.has_unphysical_states
            assert ops.penalty_projector() is None

    @pytest.mark.parametrize("spin", [1.0, 2.0, 2.5])
    def test_penalty_projector(self, spin):
        """Penalty projector projects onto unphysical subspace."""
        dim = int(2 * spin + 1)
        num_qubits = math.ceil(math.log2(dim))
        dim_full = 1 << num_qubits

        ops = SpinSOperators(spin, qubit_offset=0)
        pen = ops.penalty_projector()
        assert pen is not None

        # Reconstruct penalty matrix
        pen_terms = pen.simplify().to_canonical_terms(num_qubits)
        pauli_strings = [t[1][::-1] for t in pen_terms]
        coefficients = np.array([complex(t[0]) for t in pen_terms])
        pen_mat = pauli_to_dense_matrix(pauli_strings, coefficients)

        # Should be projector onto states dim..dim_full-1
        expected = np.zeros((dim_full, dim_full), dtype=np.complex128)
        for k in range(dim, dim_full):
            expected[k, k] = 1.0

        assert np.allclose(pen_mat, expected, atol=1e-12)

    def test_invalid_spin_raises(self):
        """Invalid spin values raise ValueError."""
        with pytest.raises(ValueError):
            SpinSOperators(0.3, qubit_offset=0)
        with pytest.raises(ValueError):
            SpinSOperators(-0.5, qubit_offset=0)
        with pytest.raises(ValueError):
            SpinSOperators(0.0, qubit_offset=0)


class TestSpinEncoding:
    """Test qubit layout for mixed-spin lattices."""

    def test_uniform_spin_half(self):
        enc = SpinEncoding(0.5, num_sites=4)
        assert enc.num_sites == 4
        assert enc.total_qubits == 4
        assert enc.spins == [0.5, 0.5, 0.5, 0.5]
        assert enc.site_qubits(0) == range(0, 1)
        assert enc.site_qubits(3) == range(3, 4)
        assert not enc.has_unphysical_states

    def test_uniform_spin_three_half(self):
        enc = SpinEncoding(1.5, num_sites=3)
        assert enc.total_qubits == 6  # 2 qubits per site
        assert enc.site_qubits(0) == range(0, 2)
        assert enc.site_qubits(1) == range(2, 4)
        assert enc.site_qubits(2) == range(4, 6)
        assert not enc.has_unphysical_states  # dim=4 = 2^2

    def test_mixed_spins(self):
        enc = SpinEncoding([0.5, 1.5, 1.0, 0.5])
        assert enc.num_sites == 4
        # site 0: S=1/2 → 1 qubit, site 1: S=3/2 → 2 qubits,
        # site 2: S=1 → 2 qubits, site 3: S=1/2 → 1 qubit
        assert enc.total_qubits == 6
        assert enc.site_qubits(0) == range(0, 1)
        assert enc.site_qubits(1) == range(1, 3)
        assert enc.site_qubits(2) == range(3, 5)
        assert enc.site_qubits(3) == range(5, 6)
        assert enc.has_unphysical_states  # site 2 has S=1 (dim=3, 2 qubits=4 states)

    def test_scalar_requires_num_sites(self):
        with pytest.raises(ValueError):
            SpinEncoding(1.0)

    def test_site_operators_returns_correct_type(self):
        enc = SpinEncoding(1.5, num_sites=2)
        ops = enc.site_operators(0)
        assert isinstance(ops, SpinSOperators)
        assert ops.spin == 1.5
        assert ops.qubit_offset == 0


class TestHigherSpinHamiltonians:
    """Test Heisenberg/Ising builders with higher spin values."""

    def test_spin_three_half_heisenberg_hermitian(self):
        """Spin-3/2 Heisenberg on a 2-site chain produces a Hermitian QubitHamiltonian."""
        lattice = LatticeGraph.chain(2)
        qh = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0, spins=1.5)
        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_qubits == 4  # 2 sites × 2 qubits/site
        assert qh.is_hermitian()

    def test_spin_one_heisenberg_hermitian(self):
        """Spin-1 Heisenberg on a 3-site chain is Hermitian and requires penalty."""
        lattice = LatticeGraph.chain(3)
        qh = create_heisenberg_hamiltonian(
            lattice, jx=1.0, jy=1.0, jz=1.0, spins=1.0, penalty_strength=10.0
        )
        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_qubits == 6  # 3 sites × 2 qubits/site
        assert qh.is_hermitian()

    def test_spin_one_requires_penalty(self):
        """Spin-1 (dim=3, not power of 2) raises ValueError without penalty_strength."""
        lattice = LatticeGraph.chain(2)
        with pytest.raises(ValueError, match="penalty_strength"):
            create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0, spins=1.0)

    def test_spin_three_half_no_penalty_needed(self):
        """Spin-3/2 (dim=4 = 2^2) does not require penalty."""
        lattice = LatticeGraph.chain(2)
        # Should not raise
        qh = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0, spins=1.5)
        assert isinstance(qh, QubitHamiltonian)

    def test_spin_one_heisenberg_matrix_correctness(self):
        """Verify spin-1 Heisenberg 2-site matrix against direct construction."""
        lattice = LatticeGraph.chain(2)
        jx, jy, jz = 1.0, 1.0, 1.0

        qh = create_heisenberg_hamiltonian(
            lattice, jx=jx, jy=jy, jz=jz, spins=1.0, penalty_strength=100.0
        )

        # Build the matrix from the QubitHamiltonian
        pauli_strs, coeffs = qh.pauli_strings, qh.coefficients
        pauli_labels = ["".join(ps) for ps in pauli_strs]
        h_mat = pauli_to_dense_matrix(pauli_labels, coeffs)

        # Build expected matrix from spin-1 operators directly
        sx, sy, sz = _spin_matrices(1.0)
        I3 = np.eye(3, dtype=np.complex128)

        # Physical subspace is 3×3 per site = 9-dim, embedded in 4×4 per site = 16-dim
        # S_1^α ⊗ I + I ⊗ S_2^α for the full 4×4 ⊗ 4×4 space
        def embed(m):
            result = np.zeros((4, 4), dtype=np.complex128)
            result[:3, :3] = m
            return result

        sx_emb = embed(sx)
        sy_emb = embed(sy)
        sz_emb = embed(sz)
        I4 = np.eye(4, dtype=np.complex128)

        h_expected = (
            jx * np.kron(sx_emb, sx_emb)
            + jy * np.kron(sy_emb, sy_emb)
            + jz * np.kron(sz_emb, sz_emb)
        )

        # Add penalty for unphysical states
        pen1 = np.zeros((4, 4), dtype=np.complex128)
        pen1[3, 3] = 1.0
        h_expected += 100.0 * np.kron(pen1, I4)
        h_expected += 100.0 * np.kron(I4, pen1)

        # Compare only the physical subspace (first 3×3 block per site = rows/cols 0-8)
        assert np.allclose(h_mat, h_expected, atol=1e-10)

    def test_mixed_spin_lattice(self):
        """Mixed spin lattice (S=1/2 and S=3/2) produces valid Hamiltonian."""
        lattice = LatticeGraph.chain(3)
        spins = [0.5, 1.5, 0.5]  # site 0: 1 qubit, site 1: 2 qubits, site 2: 1 qubit

        qh = create_heisenberg_hamiltonian(lattice, jx=1.0, jy=1.0, jz=1.0, spins=spins)
        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_qubits == 4  # 1 + 2 + 1
        assert qh.is_hermitian()

    def test_ising_with_higher_spin(self):
        """Ising model with spin-3/2."""
        lattice = LatticeGraph.chain(2)
        qh = create_ising_hamiltonian(lattice, j=1.0, h=0.5, spins=1.5)
        assert isinstance(qh, QubitHamiltonian)
        assert qh.num_qubits == 4
        assert qh.is_hermitian()

    def test_biquadratic_heisenberg(self):
        """Bilinear-biquadratic model for spin-1."""
        lattice = LatticeGraph.chain(2)
        qh = create_heisenberg_hamiltonian(
            lattice,
            jx=1.0,
            jy=1.0,
            jz=1.0,
            j_biquadratic=0.5,
            spins=1.0,
            penalty_strength=10.0,
        )
        assert isinstance(qh, QubitHamiltonian)
        assert qh.is_hermitian()

        # The biquadratic term should add additional Pauli strings beyond the bilinear ones
        qh_no_bq = create_heisenberg_hamiltonian(
            lattice, jx=1.0, jy=1.0, jz=1.0, spins=1.0, penalty_strength=10.0
        )
        # Biquadratic model should have at least as many terms (typically more)
        assert len(qh.pauli_strings) >= len(qh_no_bq.pauli_strings)

    def test_biquadratic_spin_half_eigenvalues(self):
        """For spin-1/2, (S⃗ᵢ·S⃗ⱼ)² has eigenvalues 9/16 (singlet) and 1/16 (triplet)."""
        lattice = LatticeGraph.chain(2)

        # Pure biquadratic model: H = (S⃗₁·S⃗₂)²
        qh = create_heisenberg_hamiltonian(
            lattice, jx=0.0, jy=0.0, jz=0.0, j_biquadratic=1.0, spins=0.5
        )

        labels = ["".join(ps) for ps in qh.pauli_strings]
        mat = pauli_to_dense_matrix(labels, qh.coefficients)
        eigenvalues = np.sort(np.linalg.eigvalsh(mat.real))

        # Singlet: (S⃗·S⃗)² = (-3/4)² = 9/16
        # Triplet (3-fold): (S⃗·S⃗)² = (1/4)² = 1/16
        expected = np.sort([9 / 16, 1 / 16, 1 / 16, 1 / 16])
        assert np.allclose(eigenvalues, expected, atol=1e-12)
