"""Tests for SelectSwap QROM operations (1D and 2D).

Tests two aspects:
  1. Correctness: within/apply + CNOT-copy pattern via qdk.Context
  2. Resource estimation: Toffoli (CCZ) and qubit counts via qsharp.estimate

Reference: Low et al. arXiv:1805.03662, Berry et al. arXiv:1902.02134.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import qdk
from qdk import qsharp

QSHARP = cast("Any", qsharp)
_QS_DIR = Path(__file__).resolve().parent.parent / "src" / "qdk_chemistry" / "utils" / "qsharp"
_PROJECT_ROOT = str(_QS_DIR)
_NS = "QDKChemistry.Utils.SelectSwap"


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════


def _make_context() -> Any:
    """Create a fresh qdk.Context with all Q# sources loaded."""
    return qdk.Context(project_root=_PROJECT_ROOT)


def _int_to_bools(value: int, width: int) -> list[bool]:
    """Convert integer to little-endian Bool array (matching Q# IntAsBoolArray)."""
    return [(value >> i) & 1 == 1 for i in range(width)]


def _bools_to_qs(data: list) -> str:
    """Convert nested Python bool list to Q# literal string."""
    if isinstance(data[0], list):
        return "[" + ", ".join(_bools_to_qs(row) for row in data) + "]"
    return "[" + ", ".join("true" if b else "false" for b in data) + "]"


def _make_random_data_1d(n_data: int, n_bits: int, seed: int = 42) -> list[list[bool]]:
    """Generate random Bool[][] data for 1D SelectSwap tests."""
    rng = np.random.default_rng(seed)
    return [_int_to_bools(int(rng.integers(0, 2**n_bits)), n_bits) for _ in range(n_data)]


def _make_random_data_2d(n_outer: int, n_inner: int, n_bits: int, seed: int = 42) -> list[list[list[bool]]]:
    """Generate random Bool[][][] data for 2D Select2DLoad tests."""
    rng = np.random.default_rng(seed)
    return [[_int_to_bools(int(rng.integers(0, 2**n_bits)), n_bits) for _ in range(n_inner)] for _ in range(n_outer)]


def _ceil_log2(n: int) -> int:
    """Ceiling of log2(n), i.e. number of address bits for n entries."""
    import math

    return math.ceil(math.log2(n))


def _phase_lookup_cost(n: int) -> int:
    """Toffoli cost of PhaseLookup (measurement-based unlookup) on n address qubits.

    PhaseLookup constructs power products on two halves of the address register.
    Each half of size m allocates 2^m - m - 1 AND gates (Toffolis).
    Ref: Gidney, arXiv:2505.15917.
    """
    if n <= 0:
        return 0
    n1 = n // 2  # floor(n/2)
    n2 = n - n1  # ceil(n/2)
    return max(0, 2**n1 - n1 - 1) + max(0, 2**n2 - n2 - 1)


def _expected_tof_1d(n_data: int, n_bits: int, lam: int) -> int:
    """Expected Toffoli count for 1D SelectSwap.

    Components:
      - Forward Select (unary iteration) on N/K entries: N/K - 2
      - Swap network (binary routing tree): (K-1)*b Fredkin gates
      - Adjoint Select (Unlookup via PhaseLookup): PhaseLookup(ceil_log2(N/K))
    """
    if lam == 0:
        return n_data - 2
    K = 2**lam
    n_select = _ceil_log2(n_data // K)
    select_cost = n_data // K - 2
    swap_cost = (K - 1) * n_bits
    unlookup_cost = _phase_lookup_cost(n_select)
    return select_cost + swap_cost + unlookup_cost


def _expected_qubits_1d(n_data: int, n_bits: int, lam: int) -> int:
    """Expected qubit count for 1D SelectSwap.

    Layout:
      - address register: ceil_log2(N)
      - output register: b
      - data register (λ>0): K*b
      - Select tree ancillas: ceil_log2(N/K) - 1
    Peak = 2*ceil_log2(N) - λ + b*(K+1) - 1  (for λ>0)
         = 2*ceil_log2(N) + b - 1             (for λ=0)
    """
    n_addr = _ceil_log2(n_data)
    if lam == 0:
        return 2 * n_addr + n_bits - 1
    K = 2**lam
    return 2 * n_addr - lam + n_bits * (K + 1) - 1


def _expected_tof_2d(n_outer: int, n_inner: int, n_bits: int, lam: int) -> int:
    """Expected Toffoli count for 2D Select2DLoad.

    Components:
      - UnaryIteration tree over N_out entries: N_out - 2 AND gates
      - Controlled Select at each leaf on N_in/K entries: N_in/K - 1 each
      - Swap network: (K-1)*b Fredkin gates
    Total = N_out * N_in / K - 2 + (K-1)*b
    """
    K = 2**lam
    return n_outer * n_inner // K - 2 + (K - 1) * n_bits


def _expected_qubits_2d(n_outer: int, n_inner: int, n_bits: int, lam: int) -> int:
    """Expected qubit count for 2D Select2DLoad.

    Layout:
      - outer address: ceil_log2(N_out)
      - inner address: ceil_log2(N_in)
      - target register: K*b
      - UnaryIteration ancillas: ceil_log2(N_out) - 1
      - Controlled Select ancillas: ceil_log2(N_in/K)
    Peak = 2*ceil_log2(N_out) + 2*ceil_log2(N_in) - λ + K*b - 1
    """
    K = 2**lam
    n_outer_addr = _ceil_log2(n_outer)
    n_inner_addr = _ceil_log2(n_inner)
    return 2 * n_outer_addr + 2 * n_inner_addr - lam + K * n_bits - 1


# ════════════════════════════════════════════════════════════════════════════
#  Statevector correctness tests
# ════════════════════════════════════════════════════════════════════════════


class TestSelectSwapCorrectness:
    """Verify SelectSwap loads the correct data for each address."""

    @pytest.mark.parametrize(
        "n_data,n_bits,num_swap_bits",
        [
            (4, 3, 0),  # no swap (plain Select)
            (4, 3, 1),  # 1 swap bit
            (8, 4, 0),  # 8 entries, no swap
            (8, 4, 1),  # 8 entries, 1 swap bit
            (8, 4, 2),  # 8 entries, 2 swap bits
        ],
    )
    def test_1d_all_addresses(self, n_data, n_bits, num_swap_bits):
        """For each address |i⟩, SelectSwap should load data[i] into output."""
        data = _make_random_data_1d(n_data, n_bits)
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelectSwap1DCorrectness({_bools_to_qs(data)}, {num_swap_bits})")
        assert result, f"SelectSwap 1D failed: n_data={n_data}, n_bits={n_bits}, num_swap_bits={num_swap_bits}"

    def test_1d_auto_lambda(self):
        """SelectSwap with numSwapBits=-1 (auto-optimal) should produce correct results."""
        data = _make_random_data_1d(8, 4)
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelectSwap1DCorrectness({_bools_to_qs(data)}, -1)")
        assert result, "SelectSwap 1D with auto lambda failed"

    @pytest.mark.parametrize(
        "n_outer,n_inner,n_bits,num_swap_bits",
        [
            (2, 4, 3, 0),  # no swap
            (2, 4, 3, 1),  # 1 swap bit
            (3, 4, 4, 0),  # non-power-of-2 outer
        ],
    )
    def test_2d_all_addresses(self, n_outer, n_inner, n_bits, num_swap_bits):
        """For each (i, j), Select2DLoad should load data[i][j] into target."""
        data = _make_random_data_2d(n_outer, n_inner, n_bits)
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelect2DLoadCorrectness({_bools_to_qs(data)}, {num_swap_bits})")
        assert result, (
            f"Select2DLoad failed: n_outer={n_outer}, n_inner={n_inner}, n_bits={n_bits}, num_swap_bits={num_swap_bits}"
        )


# ════════════════════════════════════════════════════════════════════════════
#  Resource estimation tests
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def qsharp_estimator():
    """Initialize qsharp with Q# project for resource estimation."""
    QSHARP.init(project_root=_PROJECT_ROOT)
    return QSHARP


def _estimate(qsharp_estimator, expr: str) -> dict:
    """Run qsharp.estimate and return logicalCounts."""
    result = qsharp_estimator.estimate(expr)
    return result["logicalCounts"]


class TestSelectSwapResourceEstimates:
    """Verify Toffoli (CCZ) and qubit counts from qsharp.estimate.

    Expected values computed from analytical formulas:
      Tof = (N/K - 2) + (K-1)*b + PhaseLookup(ceil_log2(N/K))
      Qubits = 2*ceil_log2(N) - λ + b*(K+1) - 1
    where K = 2^λ, and PhaseLookup accounts for measurement-based unlookup
    (Gidney arXiv:2505.15917).
    """

    @pytest.mark.parametrize(
        "n_data,n_bits",
        [
            (4, 4),
            (8, 4),
            (8, 8),
            (16, 4),
        ],
    )
    def test_1d_no_swap(self, qsharp_estimator, n_data, n_bits):
        """SelectSwap(lambda=0): plain Select, Toffoli = N - 2."""
        expected_tof = _expected_tof_1d(n_data, n_bits, 0)
        expected_qubits = _expected_qubits_1d(n_data, n_bits, 0)
        lc = _estimate(qsharp_estimator, f"{_NS}.EstimateSelectSwap1D({n_data}, {n_bits}, 0)")
        assert lc["cczCount"] == expected_tof, (
            f"n_data={n_data}, n_bits={n_bits}: tof={lc['cczCount']}, expected={expected_tof}"
        )
        assert lc["numQubits"] == expected_qubits, (
            f"n_data={n_data}, n_bits={n_bits}: qubits={lc['numQubits']}, expected={expected_qubits}"
        )

    @pytest.mark.parametrize(
        "n_data,n_bits,lam",
        [
            (8, 4, 1),
            (8, 4, 2),
            (16, 4, 1),
            (16, 8, 1),
            (16, 8, 2),
        ],
    )
    def test_1d_swap(self, qsharp_estimator, n_data, n_bits, lam):
        """SelectSwap(lambda>0): SWAP network trades qubits for Toffolis."""
        expected_tof = _expected_tof_1d(n_data, n_bits, lam)
        expected_qubits = _expected_qubits_1d(n_data, n_bits, lam)
        lc = _estimate(qsharp_estimator, f"{_NS}.EstimateSelectSwap1D({n_data}, {n_bits}, {lam})")
        assert lc["cczCount"] == expected_tof, (
            f"n_data={n_data}, n_bits={n_bits}, lam={lam}: tof={lc['cczCount']}, expected={expected_tof}"
        )
        assert lc["numQubits"] == expected_qubits, (
            f"n_data={n_data}, n_bits={n_bits}, lam={lam}: qubits={lc['numQubits']}, expected={expected_qubits}"
        )


class TestSelect2DResourceEstimates:
    """Verify 2D Select2DLoad Toffoli and qubit counts from qsharp.estimate.

    Expected values computed from analytical formulas:
      Tof = N_out * N_in / K - 2 + (K-1)*b
      Qubits = 2*ceil_log2(N_out) + 2*ceil_log2(N_in) - λ + K*b - 1
    """

    @pytest.mark.parametrize(
        "n_outer,n_inner,n_bits,lam",
        [
            (2, 4, 4, 0),
            (4, 4, 4, 0),
            (4, 8, 4, 0),
            (2, 4, 4, 1),
            (4, 8, 4, 1),
            (4, 8, 8, 2),
        ],
    )
    def test_2d_toffoli_and_qubits(self, qsharp_estimator, n_outer, n_inner, n_bits, lam):
        """Select2DLoad: Toffoli and qubit counts match analytical formulas."""
        expected_tof = _expected_tof_2d(n_outer, n_inner, n_bits, lam)
        expected_qubits = _expected_qubits_2d(n_outer, n_inner, n_bits, lam)
        lc = _estimate(
            qsharp_estimator,
            f"{_NS}.EstimateSelect2DLoad({n_outer}, {n_inner}, {n_bits}, {lam})",
        )
        assert lc["cczCount"] == expected_tof, (
            f"n_outer={n_outer}, n_inner={n_inner}, n_bits={n_bits}, lam={lam}: "
            f"tof={lc['cczCount']}, expected={expected_tof}"
        )
        assert lc["numQubits"] == expected_qubits, (
            f"n_outer={n_outer}, n_inner={n_inner}, n_bits={n_bits}, lam={lam}: "
            f"qubits={lc['numQubits']}, expected={expected_qubits}"
        )
