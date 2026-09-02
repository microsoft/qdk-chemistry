"""Tests for the single-particle symmetry vocabulary (qdk_chemistry.data.symmetry)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.data import symmetry as sym


class TestAxisName:
    """Tests for the AxisName enumeration and its string conversion."""

    def test_spin_axis_string(self):
        """AxisName.Spin maps to the canonical 'spin' tag."""
        assert sym.axis_name_to_string(sym.AxisName.Spin) == "spin"

    def test_all_axes_have_strings(self):
        """Every AxisName value has a non-empty human-readable name."""
        for axis in sym.AxisName.__members__.values():
            assert isinstance(sym.axis_name_to_string(axis), str)
            assert sym.axis_name_to_string(axis)


class TestSpinValue:
    """Tests for the SpinValue axis value (stored as 2*Ms)."""

    def test_alpha_beta_values(self):
        """The interned alpha/beta values carry 2*Ms = +1/-1."""
        assert sym.axes.alpha().value() == 1
        assert sym.axes.beta().value() == -1

    def test_equality_and_hash(self):
        """Spin values compare equal iff they carry the same 2*Ms."""
        a1 = sym.SpinValue(1)
        a2 = sym.SpinValue(1)
        b = sym.SpinValue(-1)
        assert a1 == a2
        assert a1 != b
        assert hash(a1) == hash(a2)

    def test_axis(self):
        """A spin value belongs to the Spin axis."""
        assert sym.axes.alpha().axis() == sym.AxisName.Spin


class TestSymmetryAxis:
    """Tests for SymmetryAxis construction and queries."""

    def test_spin_axis_factory(self):
        """axes.spin builds a Spin axis carrying alpha and beta labels."""
        axis = sym.axes.spin(1, True)
        assert axis.name() == sym.AxisName.Spin
        assert axis.equivalent() is True
        assert len(axis.labels()) == 2

    def test_admits(self):
        """The spin axis admits its alpha and beta labels."""
        axis = sym.axes.spin(1, True)
        assert axis.admits(sym.axes.alpha())
        assert axis.admits(sym.axes.beta())

    def test_equivalent_flag(self):
        """The equivalent flag round-trips through the factory."""
        assert sym.axes.spin(1, False).equivalent() is False


class TestSymmetries:
    """Tests for the SymmetryProduct vocabulary."""

    def test_has_axis(self):
        """A spin SymmetryProduct reports the Spin axis present."""
        syms = sym.SymmetryProduct([sym.axes.spin(1, True)])
        assert syms.has_axis(sym.AxisName.Spin)

    def test_axis_lookup(self):
        """The Spin axis can be retrieved by name."""
        syms = sym.SymmetryProduct([sym.axes.spin(1, True)])
        assert syms.axis(sym.AxisName.Spin).name() == sym.AxisName.Spin

    def test_equality_and_hash(self):
        """SymmetryProducts with identical axes compare equal and hash equal."""
        s1 = sym.SymmetryProduct([sym.axes.spin(1, True)])
        s2 = sym.SymmetryProduct([sym.axes.spin(1, True)])
        assert s1 == s2
        assert hash(s1) == hash(s2)


class TestSymmetryLabel:
    """Tests for the composite SymmetryLabel addressing key."""

    def test_get_and_has(self):
        """A spin label carries and returns its alpha value."""
        label = sym.SymmetryLabel([sym.axes.alpha()])
        assert label.has(sym.AxisName.Spin)
        assert label.get(sym.AxisName.Spin).value() == 1

    def test_equality_and_hash(self):
        """Labels with identical values compare and hash equal."""
        l1 = sym.SymmetryLabel([sym.axes.alpha()])
        l2 = sym.SymmetryLabel([sym.axes.alpha()])
        l3 = sym.SymmetryLabel([sym.axes.beta()])
        assert l1 == l2
        assert l1 != l3
        assert hash(l1) == hash(l2)

    def test_usable_as_dict_key(self):
        """Labels are hashable and usable as dict keys."""
        d = {sym.SymmetryLabel([sym.axes.alpha()]): 7}
        assert d[sym.SymmetryLabel([sym.axes.alpha()])] == 7


class TestExceptionSurface:
    """Tests for the standard exception surface exposed by the symmetry module."""

    def test_custom_error_names_are_not_exported(self):
        """The module no longer exports custom exception classes."""
        for name in (
            "QdkError",
            "SymmetryError",
            "SymmetryConditionError",
            "SymmetryBlockedTensorError",
            "BlockLabelInvalidError",
        ):
            assert not hasattr(sym, name)
