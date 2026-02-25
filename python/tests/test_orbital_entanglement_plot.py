"""Tests for the orbital entanglement chord diagram utility."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI

from qdk_chemistry.utils.visualization import plot_orbital_entanglement

# ── Mock wavefunction ───────────────────────────────────────────────────────


class _MockOrbitals:
    """Minimal orbitals stub."""

    def __init__(self, *, has_active: bool = False, active_indices: list[int] | None = None):
        self._has_active = has_active
        self._active_indices = active_indices or []

    def has_active_space(self) -> bool:
        return self._has_active

    def get_active_space_indices(self) -> tuple[list[int], ...]:
        return (self._active_indices,)


class MockWavefunction:
    """Duck-typed stand-in for ``qdk_chemistry.data.Wavefunction``."""

    def __init__(
        self,
        s1: np.ndarray,
        mi: np.ndarray,
        *,
        has_active: bool = False,
        active_indices: list[int] | None = None,
    ):
        self._s1 = np.asarray(s1, dtype=float)
        self._mi = np.asarray(mi, dtype=float)
        self._orbitals = _MockOrbitals(has_active=has_active, active_indices=active_indices)

    def has_single_orbital_entropies(self) -> bool:
        return True

    def has_mutual_information(self) -> bool:
        return True

    def get_single_orbital_entropies(self) -> np.ndarray:
        return self._s1

    def get_mutual_information(self) -> np.ndarray:
        return self._mi

    def get_orbitals(self) -> _MockOrbitals:
        return self._orbitals


class MockWavefunctionNoEntropy(MockWavefunction):
    def has_single_orbital_entropies(self) -> bool:
        return False


class MockWavefunctionNoMI(MockWavefunction):
    def has_mutual_information(self) -> bool:
        return False


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_symmetric(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a symmetric matrix with zero diagonal."""
    m = rng.random((n, n)) * 0.5
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m


@pytest.fixture
def small_wfn():
    """4-orbital mock wavefunction with non-trivial data."""
    rng = np.random.default_rng(123)
    s1 = rng.random(4) * np.log(4.0)
    mi = _make_symmetric(4, rng)
    return MockWavefunction(s1, mi)


@pytest.fixture
def small_wfn_active():
    """4-orbital wavefunction that reports active-space indices."""
    rng = np.random.default_rng(456)
    s1 = rng.random(4) * np.log(4.0)
    mi = _make_symmetric(4, rng)
    return MockWavefunction(
        s1,
        mi,
        has_active=True,
        active_indices=[3, 5, 7, 9],
    )


@pytest.fixture
def zero_entropy_wfn():
    """Wavefunction where all entropies are zero (edge case)."""
    n = 3
    return MockWavefunction(np.zeros(n), np.zeros((n, n)))


# ── Tests ───────────────────────────────────────────────────────────────────


class TestPlotOrbitalEntanglement:
    """Unit tests for ``plot_orbital_entanglement``."""

    def test_returns_figure_and_axes(self, small_wfn):
        fig, ax = plot_orbital_entanglement(small_wfn)
        assert fig is not None
        assert ax is not None

        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_default_labels_are_indices(self, small_wfn):
        """Default labels should be '0', '1', '2', '3' without active space."""
        _, ax = plot_orbital_entanglement(small_wfn)
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["0", "1", "2", "3"]

    def test_active_space_labels(self, small_wfn_active):
        """When orbitals have an active space, labels should be those indices."""
        _, ax = plot_orbital_entanglement(small_wfn_active)
        texts = [t.get_text() for t in ax.texts]
        assert texts == ["3", "5", "7", "9"]

    def test_custom_labels(self, small_wfn):
        labels = ["\u03c3", "\u03c3*", "\u03c0", "\u03c0*"]
        _, ax = plot_orbital_entanglement(small_wfn, labels=labels)
        texts = [t.get_text() for t in ax.texts]
        assert texts == labels

    def test_wrong_label_count_raises(self, small_wfn):
        with pytest.raises(ValueError, match="Number of labels"):
            plot_orbital_entanglement(small_wfn, labels=["a", "b"])

    def test_missing_entropy_raises(self):
        wfn = MockWavefunctionNoEntropy(np.zeros(2), np.zeros((2, 2)))
        with pytest.raises(RuntimeError, match="single-orbital entropies"):
            plot_orbital_entanglement(wfn)

    def test_missing_mi_raises(self):
        wfn = MockWavefunctionNoMI(np.zeros(2), np.zeros((2, 2)))
        with pytest.raises(RuntimeError, match="mutual information"):
            plot_orbital_entanglement(wfn)

    def test_zero_entropy_does_not_crash(self, zero_entropy_wfn):
        """All-zero entropies should produce equal arcs, not divide-by-zero."""
        fig, _ = plot_orbital_entanglement(zero_entropy_wfn)
        assert fig is not None

    def test_save_path_creates_file(self, small_wfn):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test_out.png"
            plot_orbital_entanglement(small_wfn, save_path=p)
            assert p.exists()
            assert p.stat().st_size > 0

    def test_save_svg(self, small_wfn):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test_out.svg"
            plot_orbital_entanglement(small_wfn, save_path=p)
            assert p.exists()
            assert p.stat().st_size > 0

    def test_existing_axes(self, small_wfn):
        """Drawing into a user-supplied axes should work."""
        fig, ax = plt.subplots()
        fig2, ax2 = plot_orbital_entanglement(small_wfn, ax=ax)
        assert ax2 is ax
        assert fig2 is fig

    def test_mi_threshold_filters_chords(self):
        """With a high threshold, weak chords should be omitted."""
        s1 = np.array([0.5, 0.5, 0.5])
        mi = np.array(
            [
                [0.0, 0.01, 0.8],
                [0.01, 0.0, 0.01],
                [0.8, 0.01, 0.0],
            ]
        )
        wfn = MockWavefunction(s1, mi)
        # With threshold = 0.5, only the (0,2) chord should survive.
        _, ax = plot_orbital_entanglement(wfn, mi_threshold=0.5)
        # Count PathPatch objects (chord lines)
        chord_patches = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        assert len(chord_patches) == 1

    def test_title_none_suppresses_title(self, small_wfn):
        _, ax = plot_orbital_entanglement(small_wfn, title=None)
        assert ax.get_title() == ""

    def test_custom_title(self, small_wfn):
        _, ax = plot_orbital_entanglement(small_wfn, title="My Plot")
        assert ax.get_title() == "My Plot"

    def test_s1_vmax_and_mi_vmax(self, small_wfn):
        """Custom v-max values should not crash."""
        fig, _ = plot_orbital_entanglement(
            small_wfn,
            s1_vmax=2.0,
            mi_vmax=3.0,
        )
        assert fig is not None

    def test_selected_indices_draws_outlines(self, small_wfn):
        """Passing selected_indices should add outline patches."""
        _, ax1 = plot_orbital_entanglement(small_wfn)
        n_patches_without = len(ax1.patches)

        _, ax2 = plot_orbital_entanglement(
            small_wfn,
            selected_indices=[0, 2],
        )
        n_patches_with = len(ax2.patches)
        # Should have more patches when outlines are drawn
        assert n_patches_with > n_patches_without

    def test_selected_indices_with_active_space(self, small_wfn_active):
        """selected_indices should match against label strings (active indices)."""
        # Active indices are [3, 5, 7, 9]; select orbitals 5 and 9
        fig, _ = plot_orbital_entanglement(
            small_wfn_active,
            selected_indices=[5, 9],
        )
        assert fig is not None

    def test_selection_color_and_linewidth(self, small_wfn):
        """Custom selection styling should not crash."""
        fig, _ = plot_orbital_entanglement(
            small_wfn,
            selected_indices=[1],
            selection_color="green",
            selection_linewidth=5.0,
        )
        assert fig is not None

    def test_large_system(self):
        """Smoke test with a larger orbital count."""
        rng = np.random.default_rng(789)
        n = 30
        s1 = rng.random(n) * np.log(4.0)
        mi = _make_symmetric(n, rng)
        wfn = MockWavefunction(s1, mi)
        fig, _ = plot_orbital_entanglement(
            wfn,
            figsize=(12, 13),
            gap_deg=1.5,
        )
        assert fig is not None

    def test_very_large_system_labels_staggered(self):
        """With many orbitals, labels should be staggered, not dropped."""
        rng = np.random.default_rng(101)
        n = 100
        # Make a few orbitals dominant, rest near-zero
        s1 = np.full(n, 0.01)
        s1[:5] = rng.random(5) * np.log(4.0)
        mi = np.zeros((n, n))
        wfn = MockWavefunction(s1, mi)
        _, ax = plot_orbital_entanglement(wfn, gap_deg=0.5)
        # All labels should still be drawn (staggered, not skipped)
        n_labels = len(ax.texts)
        assert n_labels == n

    def test_auto_line_scale(self):
        """line_scale=None should auto-scale based on orbital count."""
        rng = np.random.default_rng(202)
        for n_orbs in [4, 20, 100]:
            s1 = rng.random(n_orbs) * np.log(4.0)
            mi = _make_symmetric(n_orbs, rng)
            wfn = MockWavefunction(s1, mi)
            fig, ax = plot_orbital_entanglement(wfn)
            assert fig is not None
            plt.close(fig)

    def test_explicit_line_scale_overrides_auto(self, small_wfn):
        """Passing line_scale explicitly should be honoured."""
        fig, _ = plot_orbital_entanglement(small_wfn, line_scale=5.0)
        assert fig is not None

    def test_figsize_parameter(self, small_wfn):
        fig, _ = plot_orbital_entanglement(small_wfn, figsize=(8, 9))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8)
        assert h == pytest.approx(9)

    def test_gap_deg_zero(self, small_wfn):
        """Zero gap should work (arcs touch)."""
        fig, _ = plot_orbital_entanglement(small_wfn, gap_deg=0.0)
        assert fig is not None
