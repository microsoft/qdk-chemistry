"""Tests for the settable, thread-safe Q# utilities context."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qdk import TargetProfile

from qdk_chemistry.algorithms.phase_estimation.circuit_builder.standard_builder import (
    QdkStandardQpeCircuitBuilder,
)
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import (
    QSHARP_UTILS,
    create_qsharp_context,
    get_qsharp_context,
    set_qsharp_context,
    use_qsharp_context,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import qdk


@pytest.fixture(autouse=True)
def _restore_shared_context() -> Iterator[None]:
    """Snapshot and restore the shared context so tests stay isolated."""
    saved = get_qsharp_context()
    try:
        yield
    finally:
        set_qsharp_context(saved)


class TestContextApi:
    """Unit tests for the context accessor / mutator API."""

    def test_set_qsharp_context_overrides_globally(self) -> None:
        """set_qsharp_context installs a user-provided context process-wide."""
        user_context = create_qsharp_context()
        set_qsharp_context(user_context)
        assert get_qsharp_context() is user_context
        assert QSHARP_UTILS.StatePreparation is user_context.code.QDKChemistry.Utils.StatePreparation

    def test_set_none_resets_to_fresh_default(self) -> None:
        """Passing None clears the override and lazily rebuilds a default context."""
        user_context = create_qsharp_context()
        set_qsharp_context(user_context)
        set_qsharp_context(None)
        assert get_qsharp_context() is not user_context

    def test_use_qsharp_context_scopes_and_restores(self) -> None:
        """use_qsharp_context overrides only inside the block and restores on exit."""
        before = get_qsharp_context()
        scoped = create_qsharp_context()
        with use_qsharp_context(scoped):
            assert get_qsharp_context() is scoped
        assert get_qsharp_context() is before

    def test_use_qsharp_context_is_thread_local(self) -> None:
        """A per-thread override must not leak into other threads."""
        baseline = get_qsharp_context()
        scoped = create_qsharp_context()
        seen: dict[str, bool] = {}
        started = threading.Barrier(2)

        def overriding_thread() -> None:
            with use_qsharp_context(scoped):
                started.wait()
                seen["override"] = get_qsharp_context() is scoped

        def plain_thread() -> None:
            started.wait()
            seen["plain"] = get_qsharp_context() is baseline

        t1 = threading.Thread(target=overriding_thread)
        t2 = threading.Thread(target=plain_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert seen["override"] is True
        assert seen["plain"] is True


def _make_state_prep_from_context(context: qdk.Context) -> Circuit:
    """Build a state-preparation Circuit whose Q# op comes from *context*."""
    context.eval("operation BellState(qs : Qubit[]) : Unit is Adj + Ctl { H(qs[0]); CNOT(qs[0], qs[1]); }")
    user_op = context.eval("BellState")
    return Circuit(
        qsharp_op=user_op,
        qsharp_factory=QsharpFactoryData(program=user_op, parameter={}),
    )


def _build_standard_qpe(state_prep: Circuit) -> Circuit:
    """Run the standard QPE builder and return the composed circuit."""
    hamiltonian = QubitOperator(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    builder = QdkStandardQpeCircuitBuilder(num_bits=2)
    builder.settings().set(
        "controlled_circuit_mapper",
        AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
    )
    builder.settings().set(
        "unitary_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=float(np.pi / 2.0)),
    )
    circuits = builder.run(state_preparation=state_prep, qubit_hamiltonian=hamiltonian)
    assert len(circuits) == 1
    return circuits[0]


class TestCrossContextComposition:
    """Regression tests for composing user operations with QSHARP_UTILS factories."""

    def test_user_op_from_owning_context_composes(self) -> None:
        """A user op built from get_qsharp_context() composes and compiles."""
        state_prep = _make_state_prep_from_context(get_qsharp_context())
        circuit = _build_standard_qpe(state_prep)
        circuit.get_qsharp_circuit()

    def test_bring_your_own_context_composes(self) -> None:
        """A caller can install their own context so ops and utils share it."""
        user_context = create_qsharp_context(target_profile=TargetProfile.Adaptive_RIF)
        set_qsharp_context(user_context)
        state_prep = _make_state_prep_from_context(user_context)
        circuit = _build_standard_qpe(state_prep)
        circuit.get_qsharp_circuit()
