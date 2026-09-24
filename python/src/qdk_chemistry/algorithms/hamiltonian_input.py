"""Shared handling of the two Hamiltonian input forms accepted by QDK/Chemistry algorithms.

Algorithms that consume a Hamiltonian accept either form:

* a **lattice Hamiltonian** -- a second-quantized :class:`~qdk_chemistry.data.Hamiltonian`
  such as the one :func:`~qdk_chemistry.utils.model_hamiltonians.create_hubbard_hamiltonian`
  builds from a :class:`~qdk_chemistry.data.LatticeGraph`. It carries the lattice
  connectivity together with the model parameters, and is not yet mapped to qubits.
* a **qubit Hamiltonian** -- a :class:`~qdk_chemistry.data.QubitOperator`, i.e. the
  Pauli representation produced by a ``qubit_mapper``.

Most algorithms only understand the qubit form. They declare that by leaving
:meth:`~qdk_chemistry.algorithms.hamiltonian_unitary_builder.base.HamiltonianUnitaryBuilder.accepts_lattice`
at its default, and :func:`validate_hamiltonian_input` then rejects a lattice Hamiltonian
with an error explaining how to map it first.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.data import Hamiltonian, QubitOperator

__all__: list[str] = [
    "HamiltonianInput",
    "describe_hamiltonian_input",
    "is_lattice_hamiltonian",
    "system_num_qubits",
    "validate_hamiltonian_input",
]

#: Either input form an algorithm may be handed for the system it acts on.
HamiltonianInput = Hamiltonian | QubitOperator


def is_lattice_hamiltonian(hamiltonian: object) -> bool:
    """Return whether *hamiltonian* is an unmapped lattice (fermionic) Hamiltonian.

    Args:
        hamiltonian: The object to classify.

    Returns:
        bool: ``True`` for a :class:`~qdk_chemistry.data.Hamiltonian`, ``False`` otherwise.

    """
    return isinstance(hamiltonian, Hamiltonian)


def describe_hamiltonian_input(hamiltonian: object) -> str:
    """Return a short human-readable name for the input form of *hamiltonian*.

    Args:
        hamiltonian: The object to describe.

    Returns:
        str: ``"lattice Hamiltonian"``, ``"qubit Hamiltonian"``, or the type name.

    """
    if is_lattice_hamiltonian(hamiltonian):
        return "lattice Hamiltonian"
    if isinstance(hamiltonian, QubitOperator):
        return "qubit Hamiltonian"
    return type(hamiltonian).__name__


def system_num_qubits(hamiltonian: HamiltonianInput) -> int:
    """Return the number of system qubits the Hamiltonian acts on.

    A lattice Hamiltonian is counted as two spin orbitals per spatial orbital, matching
    the spin-blocked register that a fermion-to-qubit encoding produces.

    Args:
        hamiltonian: The lattice or qubit Hamiltonian.

    Returns:
        int: The system register width in qubits.

    Raises:
        TypeError: If *hamiltonian* is neither supported input form.

    """
    if isinstance(hamiltonian, QubitOperator):
        return hamiltonian.num_qubits
    if is_lattice_hamiltonian(hamiltonian):
        return 2 * hamiltonian.get_orbitals().get_num_molecular_orbitals()
    raise TypeError(f"Expected a lattice Hamiltonian or a qubit Hamiltonian, but got {type(hamiltonian).__name__}.")


def validate_hamiltonian_input(
    hamiltonian: object,
    *,
    accepts_lattice: bool,
    algorithm_name: str,
    algorithm_kind: str = "algorithm",
) -> None:
    """Check that *hamiltonian* is an input form the caller supports.

    Args:
        hamiltonian: The object handed to the algorithm.
        accepts_lattice: Whether the caller can consume an unmapped lattice Hamiltonian.
        algorithm_name: Name of the calling algorithm, used in the error message.
        algorithm_kind: Noun describing the caller, for example ``"unitary builder"``.

    Raises:
        TypeError: If *hamiltonian* is not a supported input form, or is a lattice
            Hamiltonian while *accepts_lattice* is ``False``.

    """
    if isinstance(hamiltonian, QubitOperator):
        return

    if is_lattice_hamiltonian(hamiltonian):
        if accepts_lattice:
            return
        raise TypeError(
            f"The {algorithm_name!r} {algorithm_kind} builds from a qubit Hamiltonian, but it was given a "
            f"lattice Hamiltonian. Map it to qubits first, for example "
            f"create('qubit_mapper').run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(num_modes)), "
            f"or choose a lattice-aware algorithm such as the 'plaquette' unitary builder."
        )

    raise TypeError(
        f"The {algorithm_name!r} {algorithm_kind} expects a lattice Hamiltonian or a qubit Hamiltonian, "
        f"but got {type(hamiltonian).__name__}."
    )
