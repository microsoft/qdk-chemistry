"""Tests for the Pauli SELECT oracles in ``QDKChemistry.Utils.Select``."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk import TargetProfile
from qdk.qsharp import Pauli, QSharpError

from qdk_chemistry.utils.qsharp import create_qsharp_context, get_qsharp_context

# Without mid-circuit measurement ``AND`` is only a relative-phase Toffoli, so both lowerings are checked.
_PROFILES = [TargetProfile.Base, TargetProfile.Adaptive_RIF]
_CONTEXTS: dict[str, object] = {}


def _context(profile: TargetProfile):
    """A context per target profile, shared because each one compiles the Q# sources."""
    key = str(profile)
    if key not in _CONTEXTS:
        _CONTEXTS[key] = create_qsharp_context(profile)
    return _CONTEXTS[key]


def _unary_action_index(num_actions: int, address: int) -> int:
    """Mirror ``UnaryIterationActionIndex``: the action unary iteration runs for ``address``."""
    if num_actions == 1:
        return 0
    num_bits = (num_actions - 1).bit_length()
    address %= 1 << num_bits
    half = 1 << (num_bits - 1)
    if address < half:
        return address
    return half + _unary_action_index(num_actions - half, address - half)


def _params(terms: list[str], signs: list[int], states: list[int]) -> str:
    """Render a ``PauliSelectParams`` literal; ``terms[i][q]`` acts on target qubit ``q``."""
    paulis = ", ".join("[" + ", ".join(f"Pauli{p}" for p in term) + "]" for term in terms)
    return f"QDKChemistry.Utils.Select.PauliSelectParams([{paulis}], {signs}, {states})"


def _aliased_reference(terms: list[str], signs: list[int], states: list[int]) -> str:
    """The ``PauliSelect`` that applies, on every index state, the term unary iteration routes it to."""
    term_at = dict(zip(states, zip(terms, signs, strict=True), strict=True))
    num_actions = max(states) + 1
    num_bits = (num_actions - 1).bit_length()
    routed = [(s, term_at[a]) for s in range(1 << num_bits) if (a := _unary_action_index(num_actions, s)) in term_at]
    return _params([t for _, (t, _) in routed], [g for _, (_, g) in routed], [s for s, _ in routed])


def _controlled(operation: str, params: str, num_controls: int, num_index: int) -> str:
    """A lambda applying ``operation`` under ``num_controls`` controls on a flat register."""
    controls = f"qs[0..{num_controls - 1}]"
    index = f"qs[{num_controls}..{num_controls + num_index - 1}]"
    targets = f"qs[{num_controls + num_index}...]"
    return f"qs => Controlled QDKChemistry.Utils.Select.{operation}({controls}, ({params}, {index}, {targets}))"


def _random_case(rng: np.random.Generator, num_terms: int, num_targets: int, with_gap: bool):
    """Random Pauli terms, signs, and distinct control states, optionally skipping one index."""
    states = list(range(num_terms + int(with_gap)))
    if with_gap:
        states.pop(int(rng.integers(num_terms)))
    terms = ["".join(rng.choice(list("IXYZ"), num_targets)) for _ in states]
    signs = [int(s) for s in rng.choice([1, -1], len(states))]
    return terms, signs, states


class TestUnaryPauliSelect:
    """``UnaryPauliSelect`` must reproduce ``PauliSelect`` on every index state that names a term."""

    @pytest.mark.parametrize("profile", _PROFILES)
    @pytest.mark.parametrize("num_controls", [0, 1, 2, 3])
    @pytest.mark.parametrize("with_gap", [False, True])
    @pytest.mark.parametrize("num_terms", [1, 2, 3, 4, 5, 7, 8, 11])
    def test_matches_multi_controlled_select(
        self, num_terms: int, with_gap: bool, num_controls: int, profile: TargetProfile
    ) -> None:
        """Compare the full unitaries, controlled or not, including signs and aliased indices.

        Index states past the last control state run the term unary iteration routes them to,
        so the reference ``PauliSelect`` is given those aliased terms explicitly. Missing control
        states apply the identity in both.
        """
        rng = np.random.default_rng(1000 * num_terms + 10 * num_controls + int(with_gap))
        num_targets = 2
        terms, signs, states = _random_case(rng, num_terms, num_targets, with_gap)
        num_index = max(states).bit_length()
        num_qubits = num_controls + num_index + num_targets

        actual = _controlled("UnaryPauliSelect", _params(terms, signs, states), num_controls, num_index)
        expected = _controlled("PauliSelect", _aliased_reference(terms, signs, states), num_controls, num_index)
        assert _context(profile).eval(f"Std.Diagnostics.CheckOperationsAreEqual({num_qubits}, {actual}, {expected})")

    @pytest.mark.parametrize("profile", _PROFILES)
    @pytest.mark.parametrize("num_controls", [0, 1, 2])
    @pytest.mark.parametrize("num_terms", [1, 3, 6])
    def test_is_self_inverse(self, num_terms: int, num_controls: int, profile: TargetProfile) -> None:
        """SELECT squares to the identity, which the qubitization walk relies on."""
        terms, signs, states = _random_case(np.random.default_rng(num_terms), num_terms, 3, with_gap=False)
        num_index = max(states).bit_length()
        select = _controlled("UnaryPauliSelect", _params(terms, signs, states), num_controls, num_index)
        twice = f"qs => {{ ({select})(qs); ({select})(qs); }}"
        num_qubits = num_controls + num_index + 3
        assert _context(profile).eval(f"Std.Diagnostics.CheckOperationsAreEqual({num_qubits}, {twice}, qs => ())")

    def test_rejects_repeated_control_states(self) -> None:
        """Two terms on one index state cannot both be selected."""
        params = _params(["XI", "ZZ"], [1, 1], [1, 1])
        with pytest.raises(QSharpError, match="distinct"):
            get_qsharp_context().eval(f"QDKChemistry.Utils.Select.MakeUnarySelectCircuit({params}, 1, 2)")

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_lowers_to_qir(self, profile: TargetProfile) -> None:
        """The recursion runs on classical data, so it lowers under Base as well as Adaptive."""
        # A fresh context: compiling in one that has already evaluated many lambdas can trip qdk.
        context = create_qsharp_context(profile)
        select = context.code.QDKChemistry.Utils.Select
        terms, signs, states = _random_case(np.random.default_rng(5), 5, 2, with_gap=False)
        params = select.PauliSelectParams(
            pauliTerms=[[getattr(Pauli, p) for p in term] for term in terms], signs=signs, controlStates=states
        )
        qir = str(context.compile(select.MakeUnarySelectCircuit, params, 3, 2))
        assert "define" in qir
        if profile == TargetProfile.Base:
            assert "__quantum__qis__m" not in qir

    def test_cuts_toffolis(self) -> None:
        """Unary iteration needs L - 2 ANDs; multi-controlling every factor costs far more."""
        terms, signs, states = _random_case(np.random.default_rng(3), 16, 4, with_gap=False)
        terms = [term.replace("I", "X") for term in terms]
        context = get_qsharp_context()
        select = context.code.QDKChemistry.Utils.Select
        params = select.PauliSelectParams(
            pauliTerms=[[getattr(Pauli, p) for p in term] for term in terms], signs=signs, controlStates=states
        )

        def toffolis(entry) -> int:
            counts = context.logical_counts(entry, params, 4, 4)
            return counts["cczCount"] + counts["ccixCount"]

        assert toffolis(select.MakeUnarySelectCircuit) == len(terms) - 2
        assert toffolis(select.MakeSelectCircuit) > 10 * toffolis(select.MakeUnarySelectCircuit)
