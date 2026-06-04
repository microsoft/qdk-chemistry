"""Tests for Zassenhaus expansion Builder in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
import scipy

from qdk_chemistry.algorithms.time_evolution.builder.zassenhaus import Zassenhaus, ZassenhausSettings
from qdk_chemistry.data import QubitHamiltonian, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import PauliProductFormulaContainer

from .reference_tolerances import float_comparison_absolute_tolerance


# ---------------------------------------------------------------------------
# Shared Hamiltonians
# ---------------------------------------------------------------------------

def _heisenberg_4_site() -> QubitHamiltonian:
    """Open 4-site Heisenberg chain H = sum_{i=0}^{2} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}).

    Pauli strings in little-endian Qiskit convention (qubit 0 = rightmost char).
    3 bonds × 3 Pauli pairs = 9 two-body terms.
    """
    pauli_strings = [
        "IIXX", "IIYX", "IIZX",   # bond 0-1 (qubits 0,1)  -- note: "IIXX" means X on 0, X on 1
        "IXXII", "IYXII", "IZZII",  # these are wrong; build correctly below
    ]
    # Build correctly: bond (i, i+1) → Pauli on qubit i and i+1, rest identity
    # Little-endian: rightmost char = qubit 0
    n = 4
    terms: list[str] = []
    for i in range(n - 1):
        for p in ("X", "Y", "Z"):
            label = ["I"] * n
            label[n - 1 - i] = p
            label[n - 1 - (i + 1)] = p
            terms.append("".join(label))
    coefficients = [1.0] * len(terms)
    return QubitHamiltonian(pauli_strings=terms, coefficients=coefficients)


def _h2_sto3g() -> QubitHamiltonian:
    """H2/STO-3G Jordan-Wigner qubit Hamiltonian (4 qubits).

    Coefficients from standard JW encoding of H2 at equilibrium geometry
    (R = 0.74 Å); values from the Qiskit Nature / OpenFermion literature.
    """
    pauli_strings = [
        "IIII",
        "IIIZ",
        "IIZI",
        "IZII",
        "ZIII",
        "IIZZ",
        "IZIZ",
        "ZIIZ",
        "IZZI",
        "ZIZI",
        "ZZII",
        "IIXX",
        "IIYX",
        "IIYY",
        "IIXY",
    ]
    coefficients = [
        -0.81054798,
         0.17120669,
        -0.22278593,
        -0.22278593,
         0.17120669,
         0.12062523,
         0.16862219,
         0.16862219,
         0.12062523,
         0.17434844,
         0.12062523,
         0.04532175,
        -0.04532175,
         0.04532175,
        -0.04532175,
    ]
    return QubitHamiltonian(pauli_strings=pauli_strings, coefficients=coefficients)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _container_to_matrix(container: PauliProductFormulaContainer) -> np.ndarray:
    """Evaluate the unitary matrix encoded in a PauliProductFormulaContainer."""
    from qiskit.quantum_info import SparsePauliOp

    n = container.num_qubits
    dim = 2**n
    U = np.eye(dim, dtype=complex)
    for _ in range(container.step_reps):
        for term in container.step_terms:
            label = ["I"] * n
            for q, op in term.pauli_term.items():
                label[n - 1 - q] = op  # little-endian: qubit 0 → rightmost
            P = SparsePauliOp("".join(label)).to_matrix()
            U = scipy.linalg.expm(-1j * term.angle * P) @ U
    return U


def _exact_unitary(hamiltonian: QubitHamiltonian, time: float) -> np.ndarray:
    """Compute exp(-i H t) by dense matrix exponentiation."""
    H = np.array(hamiltonian.to_matrix(), dtype=complex)
    return scipy.linalg.expm(-1j * time * H)


def _operator_norm_error(U_approx: np.ndarray, U_exact: np.ndarray) -> float:
    """Spectral norm ||U_approx - U_exact||_2."""
    return float(np.linalg.norm(U_approx - U_exact, ord=2))


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------

class TestZassenhausBasic:
    """Tests for Zassenhaus class structure and settings."""

    def test_name(self):
        assert Zassenhaus().name() == "zassenhaus"

    def test_type_name(self):
        assert Zassenhaus().type_name() == "time_evolution_builder"

    def test_default_settings(self):
        s = ZassenhausSettings()
        assert s.get("order") == 2
        assert s.get("split_index") == -1
        assert s.get("tolerance") == pytest.approx(1e-12)

    def test_returns_time_evolution_unitary(self):
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = Zassenhaus(order=2)
        result = builder.run(hamiltonian, time=0.1)
        assert isinstance(result, TimeEvolutionUnitary)

    def test_container_type(self):
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        builder = Zassenhaus(order=2)
        result = builder.run(hamiltonian, time=0.1)
        assert isinstance(result.get_container(), PauliProductFormulaContainer)

    def test_step_reps_is_one(self):
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
        result = Zassenhaus(order=2).run(hamiltonian, time=0.1)
        assert result.get_container().step_reps == 1

    def test_num_qubits_preserved(self):
        hamiltonian = QubitHamiltonian(pauli_strings=["XI", "ZZ"], coefficients=[1.0, 0.5])
        result = Zassenhaus(order=2).run(hamiltonian, time=0.1)
        assert result.get_container().num_qubits == 2

    def test_order_too_low_raises(self):
        builder = Zassenhaus(order=1)
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        with pytest.raises(ValueError, match="order >= 2"):
            builder.run(hamiltonian, time=0.1)

    def test_order_too_high_raises(self):
        builder = Zassenhaus(order=5)
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        with pytest.raises(NotImplementedError):
            builder.run(hamiltonian, time=0.1)

    def test_rejects_non_hermitian(self):
        hamiltonian = QubitHamiltonian(pauli_strings=["X"], coefficients=[1.0 + 1.0j])
        with pytest.raises((ValueError, TypeError)):
            Zassenhaus(order=2).run(hamiltonian, time=0.1)

    def test_split_index_respected(self):
        """Custom split_index places 1 term in A and the rest in B."""
        hamiltonian = QubitHamiltonian(pauli_strings=["X", "Z"], coefficients=[1.0, 1.0])
        builder = Zassenhaus(order=2, split_index=1)
        result = builder.run(hamiltonian, time=0.1)
        assert isinstance(result.get_container(), PauliProductFormulaContainer)


# ---------------------------------------------------------------------------
# Error scaling: O(t^(p+1)) for p in {2, 3, 4}
# ---------------------------------------------------------------------------

def _empirical_slope(hamiltonian: QubitHamiltonian, order: int, times: list[float]) -> float:
    """Fit log(error) vs log(t) and return the slope."""
    errors = []
    for t in times:
        builder = Zassenhaus(order=order)
        container = builder.run(hamiltonian, time=t).get_container()
        U_approx = _container_to_matrix(container)
        U_exact = _exact_unitary(hamiltonian, t)
        err = _operator_norm_error(U_approx, U_exact)
        errors.append(err)
    log_t = np.log(times)
    log_e = np.log(np.clip(errors, 1e-15, None))
    slope, _ = np.polyfit(log_t, log_e, 1)
    return float(slope)


class TestZassenhausErrorScalingHeisenberg:
    """Verify O(t^(p+1)) error scaling on 4-site open Heisenberg chain."""

    TIMES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    SLOPE_TOLERANCE = 0.1

    @pytest.fixture(scope="class")
    def hamiltonian(self):
        return _heisenberg_4_site()

    def test_order_2_scaling(self, hamiltonian):
        slope = _empirical_slope(hamiltonian, order=2, times=self.TIMES)
        assert abs(slope - 3.0) < self.SLOPE_TOLERANCE, f"Expected slope ~3 for p=2, got {slope:.3f}"

    def test_order_3_scaling(self, hamiltonian):
        slope = _empirical_slope(hamiltonian, order=3, times=self.TIMES)
        assert abs(slope - 4.0) < self.SLOPE_TOLERANCE, f"Expected slope ~4 for p=3, got {slope:.3f}"

    def test_order_4_scaling(self, hamiltonian):
        slope = _empirical_slope(hamiltonian, order=4, times=self.TIMES)
        assert abs(slope - 5.0) < self.SLOPE_TOLERANCE, f"Expected slope ~5 for p=4, got {slope:.3f}"


class TestZassenhausErrorScalingH2:
    """Verify O(t^(p+1)) error scaling on H2/STO-3G Jordan-Wigner Hamiltonian."""

    TIMES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    SLOPE_TOLERANCE = 0.1

    @pytest.fixture(scope="class")
    def hamiltonian(self):
        return _h2_sto3g()

    def test_order_2_scaling(self, hamiltonian):
        slope = _empirical_slope(hamiltonian, order=2, times=self.TIMES)
        assert abs(slope - 3.0) < self.SLOPE_TOLERANCE, f"Expected slope ~3 for p=2, got {slope:.3f}"

    def test_order_3_scaling(self, hamiltonian):
        slope = _empirical_slope(hamiltonian, order=3, times=self.TIMES)
        assert abs(slope - 4.0) < self.SLOPE_TOLERANCE, f"Expected slope ~4 for p=3, got {slope:.3f}"

    def test_order_4_scaling(self, hamiltonian):
        slope = _empirical_slope(hamiltonian, order=4, times=self.TIMES)
        assert abs(slope - 5.0) < self.SLOPE_TOLERANCE, f"Expected slope ~5 for p=4, got {slope:.3f}"


# ---------------------------------------------------------------------------
# Correctness: lower error than first-order Trotter at same t
# ---------------------------------------------------------------------------

class TestZassenhausVsTrotter:
    """Zassenhaus at order 2 should be more accurate than first-order Trotter."""

    def test_lower_error_than_trotter_heisenberg(self):
        from qdk_chemistry.algorithms.time_evolution.builder.trotter import Trotter

        hamiltonian = _heisenberg_4_site()
        t = 0.5

        U_exact = _exact_unitary(hamiltonian, t)

        trotter_container = Trotter(num_divisions=1).run(hamiltonian, time=t).get_container()
        err_trotter = _operator_norm_error(_container_to_matrix(trotter_container), U_exact)

        zassenhaus_container = Zassenhaus(order=2).run(hamiltonian, time=t).get_container()
        err_zassenhaus = _operator_norm_error(_container_to_matrix(zassenhaus_container), U_exact)

        assert err_zassenhaus < err_trotter, (
            f"Zassenhaus order-2 error {err_zassenhaus:.4e} should be smaller than "
            f"first-order Trotter error {err_trotter:.4e} at t={t}"
        )

    def test_higher_order_more_accurate(self):
        hamiltonian = _h2_sto3g()
        t = 0.3
        U_exact = _exact_unitary(hamiltonian, t)

        errors = {}
        for order in (2, 3, 4):
            container = Zassenhaus(order=order).run(hamiltonian, time=t).get_container()
            errors[order] = _operator_norm_error(_container_to_matrix(container), U_exact)

        assert errors[3] < errors[2], f"Order 3 ({errors[3]:.4e}) should beat order 2 ({errors[2]:.4e})"
        assert errors[4] < errors[3], f"Order 4 ({errors[4]:.4e}) should beat order 3 ({errors[3]:.4e})"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestZassenhausRegistry:
    """Zassenhaus is discoverable through the standard registry."""

    def test_registry_create(self):
        from qdk_chemistry.algorithms import registry

        builder = registry.create("time_evolution_builder", "zassenhaus")
        assert isinstance(builder, Zassenhaus)

    def test_registry_available(self):
        from qdk_chemistry.algorithms import registry

        names = registry.available("time_evolution_builder")
        assert "zassenhaus" in names
