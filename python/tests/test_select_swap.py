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

QSHARP = cast(Any, qsharp)
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
    return [
        _int_to_bools(int(rng.integers(0, 2**n_bits)), n_bits)
        for _ in range(n_data)
    ]


def _make_random_data_2d(
    n_outer: int, n_inner: int, n_bits: int, seed: int = 42
) -> list[list[list[bool]]]:
    """Generate random Bool[][][] data for 2D Select2DLoad tests."""
    rng = np.random.default_rng(seed)
    return [
        [
            _int_to_bools(int(rng.integers(0, 2**n_bits)), n_bits)
            for _ in range(n_inner)
        ]
        for _ in range(n_outer)
    ]


# ════════════════════════════════════════════════════════════════════════════
#  Statevector correctness tests
# ════════════════════════════════════════════════════════════════════════════


class TestSelectSwapCorrectness:
    """Verify SelectSwap loads the correct data for each address."""

    @pytest.mark.parametrize("n_data,n_bits,num_swap_bits", [
        (4, 3, 0),   # no swap (plain Select)
        (4, 3, 1),   # 1 swap bit
        (8, 4, 0),   # 8 entries, no swap
        (8, 4, 1),   # 8 entries, 1 swap bit
        (8, 4, 2),   # 8 entries, 2 swap bits
    ])
    def test_1d_all_addresses(self, n_data, n_bits, num_swap_bits):
        """For each address |i⟩, SelectSwap should load data[i] into output."""
        data = _make_random_data_1d(n_data, n_bits)
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelectSwap1DCorrectness({_bools_to_qs(data)}, {num_swap_bits})")
        assert result, (
            f"SelectSwap 1D failed: n_data={n_data}, n_bits={n_bits}, "
            f"num_swap_bits={num_swap_bits}"
        )

    def test_1d_auto_lambda(self):
        """SelectSwap with numSwapBits=-1 (auto-optimal) should produce correct results."""
        data = _make_random_data_1d(8, 4)
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelectSwap1DCorrectness({_bools_to_qs(data)}, -1)")
        assert result, "SelectSwap 1D with auto lambda failed"


class TestSelect2DCorrectness:
    """Verify Select2DLoad loads the correct data for each (outer, inner) address."""

    @pytest.mark.parametrize("n_outer,n_inner,n_bits,num_swap_bits", [
        (2, 4, 3, 0),   # no swap
        (2, 4, 3, 1),   # 1 swap bit
        (3, 4, 4, 0),   # non-power-of-2 outer
    ])
    def test_2d_all_addresses(self, n_outer, n_inner, n_bits, num_swap_bits):
        """For each (i, j), Select2DLoad should load data[i][j] into target."""
        data = _make_random_data_2d(n_outer, n_inner, n_bits)
        ctx = _make_context()
        result = ctx.eval(f"{_NS}.TestSelect2DLoadCorrectness({_bools_to_qs(data)}, {num_swap_bits})")
        assert result, (
            f"Select2DLoad failed: n_outer={n_outer}, n_inner={n_inner}, "
            f"n_bits={n_bits}, num_swap_bits={num_swap_bits}"
        )


# ════════════════════════════════════════════════════════════════════════════
#  Resource estimation tests
# ════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def qsharp_estimator():
    """Initialize qsharp with Q# project for resource estimation."""
    QSHARP.init(project_root=_PROJECT_ROOT)
    yield QSHARP


def _estimate(qsharp_estimator, expr: str) -> dict:
    """Run qsharp.estimate and return logicalCounts."""
    result = qsharp_estimator.estimate(expr)
    return result["logicalCounts"]


class TestSelectSwapResourceEstimates:
    """Verify Toffoli (CCZ) and qubit counts from qsharp.estimate.

    Baseline values obtained from qsharp.estimate on the actual Q# implementation.
    These serve as regression tests — any change in counts indicates a code change.
    """

    @pytest.mark.parametrize("n_data,n_bits,expected_tof,expected_qubits", [
        (4, 4, 2, 7),
        (8, 4, 6, 9),
        (8, 8, 6, 13),
        (16, 4, 14, 11),
    ])
    def test_1d_no_swap(self, qsharp_estimator, n_data, n_bits, expected_tof, expected_qubits):
        """SelectSwap(lambda=0): plain Select, Toffoli = N - 2."""
        lc = _estimate(qsharp_estimator, f"{_NS}.EstimateSelectSwap1D({n_data}, {n_bits}, 0)")
        assert lc["cczCount"] == expected_tof, (
            f"n_data={n_data}, n_bits={n_bits}: "
            f"tof={lc['cczCount']}, expected={expected_tof}"
        )
        assert lc["numQubits"] == expected_qubits, (
            f"n_data={n_data}, n_bits={n_bits}: "
            f"qubits={lc['numQubits']}, expected={expected_qubits}"
        )

    @pytest.mark.parametrize("n_data,n_bits,lam,expected_tof,expected_qubits", [
        (8, 4, 1, 6, 16),
        (8, 4, 2, 12, 23),
        (16, 4, 1, 11, 18),
        (16, 8, 1, 15, 30),
        (16, 8, 2, 26, 45),
    ])
    def test_1d_swap(self, qsharp_estimator, n_data, n_bits, lam, expected_tof, expected_qubits):
        """SelectSwap(lambda>0): SWAP network trades qubits for Toffolis."""
        lc = _estimate(qsharp_estimator, f"{_NS}.EstimateSelectSwap1D({n_data}, {n_bits}, {lam})")
        assert lc["cczCount"] == expected_tof, (
            f"n_data={n_data}, n_bits={n_bits}, lam={lam}: "
            f"tof={lc['cczCount']}, expected={expected_tof}"
        )
        assert lc["numQubits"] == expected_qubits, (
            f"n_data={n_data}, n_bits={n_bits}, lam={lam}: "
            f"qubits={lc['numQubits']}, expected={expected_qubits}"
        )


class TestSelect2DResourceEstimates:
    """Verify 2D Select2DLoad Toffoli and qubit counts from qsharp.estimate."""

    @pytest.mark.parametrize("n_outer,n_inner,n_bits,lam,expected_tof,expected_qubits", [
        (2, 4, 4, 0, 6, 9),
        (4, 4, 4, 0, 14, 11),
        (4, 8, 4, 0, 30, 13),
        (2, 4, 4, 1, 6, 12),
        (4, 8, 4, 1, 18, 16),
        (4, 8, 8, 2, 30, 39),
    ])
    def test_2d_toffoli_and_qubits(
        self, qsharp_estimator, n_outer, n_inner, n_bits, lam, expected_tof, expected_qubits
    ):
        """Select2DLoad: Toffoli and qubit counts match baseline."""
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
