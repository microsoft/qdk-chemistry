"""Tests for SymmetryBlockedTensor and SymmetryBlockedIndexSet Python bindings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.data import symmetry as sym


@pytest.fixture
def restricted_spin():
    """A restricted (equivalent) spin vocabulary with alpha/beta labels."""
    syms = sym.SymmetryProduct([sym.axes.spin(1, True)])
    alpha = sym.SymmetryLabel([sym.axes.alpha()])
    beta = sym.SymmetryLabel([sym.axes.beta()])
    return syms, alpha, beta


@pytest.fixture
def unrestricted_spin():
    """An unrestricted (non-equivalent) spin vocabulary with alpha/beta labels."""
    syms = sym.SymmetryProduct([sym.axes.spin(1, False)])
    alpha = sym.SymmetryLabel([sym.axes.alpha()])
    beta = sym.SymmetryLabel([sym.axes.beta()])
    return syms, alpha, beta


class TestSymmetryBlockedTensorRank2:
    """Tests for the rank-2 (matrix) symmetry-blocked tensor."""

    def test_restricted_auto_aliases_beta(self, restricted_spin):
        """A restricted tensor auto-aliases the beta block to alpha storage."""
        syms, alpha, beta = restricted_spin
        block = np.arange(9, dtype=float).reshape(3, 3)
        sbt = sym.SymmetryBlockedTensorRank2(
            [syms, syms],
            [{alpha: 3, beta: 3}, {alpha: 3, beta: 3}],
            [((alpha, alpha), block)],
        )
        assert sbt.has_block((alpha, alpha))
        assert sbt.has_block((beta, beta))
        # Both spin sectors exist (num_blocks counts aliases) but storage is
        # shared, so block((alpha, alpha)) and block((beta, beta)) return the
        # same data.
        assert sbt.num_blocks() == 2
        assert np.allclose(sbt.block((alpha, alpha)), sbt.block((beta, beta)))

    def test_block_roundtrip(self, restricted_spin):
        """Stored numpy data is returned unchanged from block()."""
        syms, alpha, beta = restricted_spin
        block = np.random.default_rng(0).standard_normal((4, 4))
        sbt = sym.SymmetryBlockedTensorRank2(
            [syms, syms],
            [{alpha: 4, beta: 4}, {alpha: 4, beta: 4}],
            [((alpha, alpha), block)],
        )
        assert np.allclose(sbt.block((alpha, alpha)), block)
        assert np.allclose(sbt.block((beta, beta)), block)

    def test_unrestricted_distinct_blocks(self, unrestricted_spin):
        """An unrestricted tensor keeps alpha and beta blocks independent."""
        syms, alpha, beta = unrestricted_spin
        a_block = np.ones((2, 2))
        b_block = 2.0 * np.ones((2, 2))
        sbt = sym.SymmetryBlockedTensorRank2(
            [syms, syms],
            [{alpha: 2, beta: 2}, {alpha: 2, beta: 2}],
            [((alpha, alpha), a_block), ((beta, beta), b_block)],
        )
        assert np.allclose(sbt.block((alpha, alpha)), a_block)
        assert np.allclose(sbt.block((beta, beta)), b_block)
        # Independent storage: alpha and beta blocks differ.
        assert not np.allclose(sbt.block((alpha, alpha)), sbt.block((beta, beta)))

    def test_missing_block_raises(self, unrestricted_spin):
        """Requesting an absent block raises ValueError."""
        syms, alpha, beta = unrestricted_spin
        sbt = sym.SymmetryBlockedTensorRank2(
            [syms, syms],
            [{alpha: 2, beta: 2}, {alpha: 2, beta: 2}],
            [((alpha, alpha), np.zeros((2, 2)))],
        )
        with pytest.raises(ValueError, match="no block for the requested labels"):
            sbt.block((beta, beta))

    def test_extent_mismatch_raises(self, unrestricted_spin):
        """A block whose shape disagrees with extents raises ValueError."""
        syms, alpha, beta = unrestricted_spin
        with pytest.raises(ValueError, match="block shape does not match extents"):
            sym.SymmetryBlockedTensorRank2(
                [syms, syms],
                [{alpha: 3, beta: 3}, {alpha: 3, beta: 3}],
                [((alpha, alpha), np.zeros((2, 2)))],
            )

    def test_symmetries_accessor(self, restricted_spin):
        """The per-slot symmetries are recoverable from the tensor."""
        syms, alpha, beta = restricted_spin
        sbt = sym.SymmetryBlockedTensorRank2(
            [syms, syms],
            [{alpha: 1, beta: 1}, {alpha: 1, beta: 1}],
            [((alpha, alpha), np.zeros((1, 1)))],
        )
        slots = sbt.symmetries()
        assert len(slots) == 2
        assert slots[0].has_axis(sym.AxisName.Spin)


class TestSymmetryBlockedTensorRank1:
    """Tests for the rank-1 (vector) symmetry-blocked tensor."""

    def test_real_and_complex(self, unrestricted_spin):
        """Both real and complex rank-1 tensors store and return blocks."""
        syms, alpha, beta = unrestricted_spin
        real = sym.SymmetryBlockedTensorRank1([syms], [{alpha: 3, beta: 3}], [((alpha,), np.array([1.0, 2.0, 3.0]))])
        assert np.allclose(real.block((alpha,)), [1.0, 2.0, 3.0])

        vec = np.array([1 + 2j, 3 - 1j])
        cplx = sym.SymmetryBlockedTensorRank1Complex([syms], [{alpha: 2, beta: 2}], [((alpha,), vec)])
        assert np.allclose(cplx.block((alpha,)), vec)


class TestSymmetryBlockedIndexSet:
    """Tests for the symmetry-blocked index set."""

    def test_indices_returns_immutable_tuple(self, unrestricted_spin):
        """indices() returns a tuple of the sorted, unique indices."""
        syms, alpha, beta = unrestricted_spin
        sbis = sym.SymmetryBlockedIndexSet(syms, {alpha: 5, beta: 5}, {alpha: [0, 2, 4], beta: [1, 3]})
        assert sbis.indices(alpha) == (0, 2, 4)
        assert isinstance(sbis.indices(alpha), tuple)
        assert sbis.has(alpha)
        assert len(sbis.labels()) == 2

    def test_out_of_range_raises(self, restricted_spin):
        """An index >= its extent raises IndexError."""
        syms, alpha, beta = restricted_spin
        with pytest.raises(IndexError):
            sym.SymmetryBlockedIndexSet(syms, {alpha: 3, beta: 3}, {alpha: [0, 5]})

    def test_not_sorted_unique_raises(self, restricted_spin):
        """A non-increasing index list raises ValueError."""
        syms, alpha, beta = restricted_spin
        with pytest.raises(ValueError, match="must be strictly increasing"):
            sym.SymmetryBlockedIndexSet(syms, {alpha: 5, beta: 5}, {alpha: [2, 1]})


class TestSerialization:
    """Round-trip serialization tests for the SBT storage primitives."""

    def test_json_roundtrip(self, restricted_spin, tmp_path):
        """A rank-2 tensor survives a JSON round-trip."""
        syms, alpha, beta = restricted_spin
        block = np.arange(4, dtype=float).reshape(2, 2)
        sbt = sym.SymmetryBlockedTensorRank2(
            [syms, syms],
            [{alpha: 2, beta: 2}, {alpha: 2, beta: 2}],
            [((alpha, alpha), block)],
        )
        path = tmp_path / "sbt.json"
        sbt.to_json_file(str(path))
        loaded = sym.SymmetryBlockedTensorRank2.from_json_file(str(path))
        assert np.allclose(loaded.block((alpha, alpha)), block)
        # Restricted-spin aliasing survives the round-trip.
        assert np.allclose(loaded.block((beta, beta)), block)

    def test_index_set_json_roundtrip(self, unrestricted_spin, tmp_path):
        """An index set survives a JSON round-trip."""
        syms, alpha, beta = unrestricted_spin
        sbis = sym.SymmetryBlockedIndexSet(syms, {alpha: 5, beta: 5}, {alpha: [0, 2, 4], beta: [1, 3]})
        path = tmp_path / "sbis.json"
        sbis.to_json_file(str(path))
        loaded = sym.SymmetryBlockedIndexSet.from_json_file(str(path))
        assert loaded.indices(alpha) == (0, 2, 4)
        assert loaded.indices(beta) == (1, 3)


class TestSymmetryBlockedScalarCount:
    """Tests for the SymmetryBlockedScalarCount Python bindings."""

    def test_spin_blocked_holds_independent_channels(self, unrestricted_spin):
        """Per-spin counts are stored as independent (non-aliased) blocks."""
        syms, alpha, beta = unrestricted_spin
        count = sym.SymmetryBlockedScalarCount([syms], [{alpha: 1, beta: 1}], [((alpha,), 5), ((beta,), 3)])

        assert count.has_block((alpha,))
        assert count.has_block((beta,))
        assert count.value(alpha) == 5
        assert count.value(beta) == 3
        assert count.num_blocks() == 2
        assert count.symmetries()[0].has_axis(sym.AxisName.Spin)

    def test_trivial_holds_aggregate(self):
        """A trivial (axis-free) symmetry holds a single aggregate block."""
        trivial = sym.SymmetryProduct([])
        empty = sym.SymmetryLabel([])
        count = sym.SymmetryBlockedScalarCount([trivial], [{empty: 1}], [((empty,), 8)])

        assert not count.symmetries()[0].has_axis(sym.AxisName.Spin)
        assert count.num_blocks() == 1
        assert count.value(empty) == 8

    def test_missing_block_raises(self):
        """Requesting an absent label raises."""
        trivial = sym.SymmetryProduct([])
        empty = sym.SymmetryLabel([])
        count = sym.SymmetryBlockedScalarCount([trivial], [{empty: 1}], [((empty,), 4)])
        with pytest.raises(ValueError, match="no block"):
            count.value(sym.SymmetryLabel([sym.axes.alpha()]))

    def test_json_roundtrip(self, unrestricted_spin, tmp_path):
        """A spin-blocked count survives a JSON round-trip."""
        syms, alpha, beta = unrestricted_spin
        count = sym.SymmetryBlockedScalarCount([syms], [{alpha: 1, beta: 1}], [((alpha,), 7), ((beta,), 2)])
        path = tmp_path / "count.json"
        count.to_json_file(str(path))
        loaded = sym.SymmetryBlockedScalarCount.from_json_file(str(path))
        assert loaded.value(alpha) == 7
        assert loaded.value(beta) == 2

    def test_hdf5_roundtrip(self, unrestricted_spin, tmp_path):
        """A spin-blocked count survives an HDF5 round-trip."""
        syms, alpha, beta = unrestricted_spin
        count = sym.SymmetryBlockedScalarCount([syms], [{alpha: 1, beta: 1}], [((alpha,), 6), ((beta,), 4)])
        path = tmp_path / "count.h5"
        count.to_hdf5_file(str(path))
        loaded = sym.SymmetryBlockedScalarCount.from_hdf5_file(str(path))
        assert loaded.value(alpha) == 6
        assert loaded.value(beta) == 4


class TestSymmetryBlockedTensorVariants:
    """Round-trips for the rank/scalar SBT variants not exercised above.

    Covers rank-2 complex, rank 3 (real and complex), and rank 4 (real and complex).
    """

    def test_rank2_complex(self, unrestricted_spin, tmp_path):
        """A rank-2 complex tensor round-trips block data through JSON."""
        syms, alpha, beta = unrestricted_spin
        block = np.array([[1 + 1j, 2 - 1j], [3, 4j]])
        sbt = sym.SymmetryBlockedTensorRank2Complex(
            [syms, syms], [{alpha: 2, beta: 2}, {alpha: 2, beta: 2}], [((alpha, alpha), block)]
        )
        assert np.allclose(sbt.block((alpha, alpha)), block)
        path = tmp_path / "r2c.json"
        sbt.to_json_file(str(path))
        loaded = sym.SymmetryBlockedTensorRank2Complex.from_json_file(str(path))
        assert np.allclose(loaded.block((alpha, alpha)), block)

    def test_rank3_real(self, unrestricted_spin, tmp_path):
        """A rank-3 tensor (size == product of 3 extents) round-trips."""
        syms, alpha, beta = unrestricted_spin
        ext = {alpha: 2, beta: 2}
        block = np.arange(8, dtype=float).reshape(2, 4)  # 2 * 2 * 2 == 8
        sbt = sym.SymmetryBlockedTensorRank3([syms, syms, syms], [ext, ext, ext], [((alpha, alpha, alpha), block)])
        assert np.allclose(sbt.block((alpha, alpha, alpha)), block)
        path = tmp_path / "r3.json"
        sbt.to_json_file(str(path))
        loaded = sym.SymmetryBlockedTensorRank3.from_json_file(str(path))
        assert np.allclose(loaded.block((alpha, alpha, alpha)), block)

    def test_rank3_complex(self, unrestricted_spin):
        """A rank-3 complex tensor exposes its block."""
        syms, alpha, beta = unrestricted_spin
        ext = {alpha: 2, beta: 2}
        block = (np.arange(8) + 1j * np.arange(8)).reshape(2, 4)
        sbt = sym.SymmetryBlockedTensorRank3Complex(
            [syms, syms, syms], [ext, ext, ext], [((alpha, alpha, alpha), block)]
        )
        assert np.allclose(sbt.block((alpha, alpha, alpha)), block)

    def test_rank4_real(self, unrestricted_spin, tmp_path):
        """A rank-4 tensor (flat vector) round-trips."""
        syms, alpha, beta = unrestricted_spin
        ext = {alpha: 2, beta: 2}
        vec = np.arange(16, dtype=float)  # 2 ** 4 flat entries
        labels = (alpha, alpha, alpha, alpha)
        sbt = sym.SymmetryBlockedTensorRank4([syms, syms, syms, syms], [ext, ext, ext, ext], [(labels, vec)])
        assert np.allclose(np.asarray(sbt.block(labels)).ravel(), vec)
        path = tmp_path / "r4.json"
        sbt.to_json_file(str(path))
        loaded = sym.SymmetryBlockedTensorRank4.from_json_file(str(path))
        assert np.allclose(np.asarray(loaded.block(labels)).ravel(), vec)

    def test_rank4_complex(self, unrestricted_spin):
        """A rank-4 complex tensor exposes its block."""
        syms, alpha, beta = unrestricted_spin
        ext = {alpha: 2, beta: 2}
        vec = np.arange(16) + 1j * np.arange(16)
        labels = (alpha, alpha, alpha, alpha)
        sbt = sym.SymmetryBlockedTensorRank4Complex([syms, syms, syms, syms], [ext, ext, ext, ext], [(labels, vec)])
        assert np.allclose(np.asarray(sbt.block(labels)).ravel(), vec)


class TestHdf5Serialization:
    """HDF5 round-trip tests across SBT ranks and scalar types."""

    def test_rank1_real_hdf5(self, unrestricted_spin, tmp_path):
        """A rank-1 real tensor survives an HDF5 round-trip."""
        syms, alpha, beta = unrestricted_spin
        sbt = sym.SymmetryBlockedTensorRank1([syms], [{alpha: 3, beta: 3}], [((alpha,), np.array([1.0, 2.0, 3.0]))])
        path = tmp_path / "r1.h5"
        sbt.to_hdf5_file(str(path))
        loaded = sym.SymmetryBlockedTensorRank1.from_hdf5_file(str(path))
        assert np.allclose(loaded.block((alpha,)), [1.0, 2.0, 3.0])

    def test_rank2_real_hdf5(self, restricted_spin, tmp_path):
        """A restricted rank-2 tensor survives an HDF5 round-trip, aliasing included."""
        syms, alpha, beta = restricted_spin
        block = np.arange(4, dtype=float).reshape(2, 2)
        sbt = sym.SymmetryBlockedTensorRank2(
            [syms, syms], [{alpha: 2, beta: 2}, {alpha: 2, beta: 2}], [((alpha, alpha), block)]
        )
        path = tmp_path / "r2.h5"
        sbt.to_hdf5_file(str(path))
        loaded = sym.SymmetryBlockedTensorRank2.from_hdf5_file(str(path))
        assert np.allclose(loaded.block((alpha, alpha)), block)
        # Restricted-spin aliasing survives the HDF5 round-trip.
        assert np.allclose(loaded.block((beta, beta)), block)

    def test_rank2_complex_hdf5(self, unrestricted_spin, tmp_path):
        """A rank-2 complex tensor survives an HDF5 round-trip."""
        syms, alpha, beta = unrestricted_spin
        block = np.array([[1 + 1j, 0], [0, 2 - 2j]])
        sbt = sym.SymmetryBlockedTensorRank2Complex(
            [syms, syms], [{alpha: 2, beta: 2}, {alpha: 2, beta: 2}], [((alpha, alpha), block)]
        )
        path = tmp_path / "r2c.h5"
        sbt.to_hdf5_file(str(path))
        loaded = sym.SymmetryBlockedTensorRank2Complex.from_hdf5_file(str(path))
        assert np.allclose(loaded.block((alpha, alpha)), block)

    def test_rank4_real_hdf5(self, unrestricted_spin, tmp_path):
        """A rank-4 real tensor survives an HDF5 round-trip."""
        syms, alpha, beta = unrestricted_spin
        ext = {alpha: 2, beta: 2}
        vec = np.arange(16, dtype=float)
        labels = (alpha, alpha, alpha, alpha)
        sbt = sym.SymmetryBlockedTensorRank4([syms, syms, syms, syms], [ext, ext, ext, ext], [(labels, vec)])
        path = tmp_path / "r4.h5"
        sbt.to_hdf5_file(str(path))
        loaded = sym.SymmetryBlockedTensorRank4.from_hdf5_file(str(path))
        assert np.allclose(np.asarray(loaded.block(labels)).ravel(), vec)

    def test_index_set_hdf5(self, unrestricted_spin, tmp_path):
        """An index set survives an HDF5 round-trip."""
        syms, alpha, beta = unrestricted_spin
        sbis = sym.SymmetryBlockedIndexSet(syms, {alpha: 5, beta: 5}, {alpha: [0, 2, 4], beta: [1, 3]})
        path = tmp_path / "sbis.h5"
        sbis.to_hdf5_file(str(path))
        loaded = sym.SymmetryBlockedIndexSet.from_hdf5_file(str(path))
        assert loaded.indices(alpha) == (0, 2, 4)
        assert loaded.indices(beta) == (1, 3)


class TestSymmetryBlockedSparseMapRank4:
    """Tests for the SymmetryBlockedSparseMapRank4 Python bindings."""

    @staticmethod
    def _make(syms, alpha):
        """Build a small unrestricted rank-4 sparse map with two entries."""
        ext = {alpha: 2, sym.SymmetryLabel([sym.axes.beta()]): 2}
        labels = (alpha, alpha, alpha, alpha)
        block = {(0, 0, 0, 0): 1.5, (1, 1, 1, 1): 2.5}
        return sym.SymmetryBlockedSparseMapRank4([syms, syms, syms, syms], [ext, ext, ext, ext], [(labels, block)])

    def test_construct_and_lookup(self, unrestricted_spin):
        """Construction exposes blocks, entry counts, and per-entry lookup."""
        syms, alpha, beta = unrestricted_spin
        m = self._make(syms, alpha)
        labels = (alpha, alpha, alpha, alpha)
        assert m.num_blocks() == 1
        assert m.num_entries() == 2
        assert m.has_block(labels)
        assert m.get(labels, (0, 0, 0, 0)) == 1.5
        assert m.get(labels, (0, 1, 0, 1)) == 0.0
        assert m.block(labels) == {(0, 0, 0, 0): 1.5, (1, 1, 1, 1): 2.5}

    def test_json_roundtrip(self, unrestricted_spin, tmp_path):
        """A sparse map survives a JSON round-trip."""
        syms, alpha, beta = unrestricted_spin
        m = self._make(syms, alpha)
        path = tmp_path / "sbsm.json"
        m.to_json_file(str(path))
        loaded = sym.SymmetryBlockedSparseMapRank4.from_json_file(str(path))
        labels = (alpha, alpha, alpha, alpha)
        assert loaded.num_entries() == 2
        assert loaded.get(labels, (1, 1, 1, 1)) == 2.5

    def test_hdf5_roundtrip(self, unrestricted_spin, tmp_path):
        """A sparse map survives an HDF5 round-trip."""
        syms, alpha, beta = unrestricted_spin
        m = self._make(syms, alpha)
        path = tmp_path / "sbsm.h5"
        m.to_hdf5_file(str(path))
        loaded = sym.SymmetryBlockedSparseMapRank4.from_hdf5_file(str(path))
        labels = (alpha, alpha, alpha, alpha)
        assert loaded.num_entries() == 2
        assert loaded.get(labels, (0, 0, 0, 0)) == 1.5
