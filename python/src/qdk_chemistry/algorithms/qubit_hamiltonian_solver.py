"""QDK/Chemistry qubit hamiltonian solver abstractions and utilities.

This module provides the base classes QubitHamiltonianSolver and QubitHamiltonianSolverFactory
for solving qubit Hamiltonians for ground state energies using various algorithms.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

import numpy as np

from qdk_chemistry import data
from qdk_chemistry._core._algorithms import davidson_solver, syev_solver
from qdk_chemistry.algorithms import base


class QubitHamiltonianSolver(base.Algorithm):
    """Abstract base class for solving a qubit Hamiltonian."""

    def __init__(self):
        """Initialize the QubitHamiltonianSolver."""
        super().__init__()

    def type_name(self) -> str:
        """Return qubit_hamiltonian_solver as the algorithm type name."""
        return "qubit_hamiltonian_solver"

    @abstractmethod
    def _run_impl(self, qubit_hamiltonian: data.QubitHamiltonian) -> tuple[float, np.ndarray]:
        """Solve a qubit Hamiltonian.

        Args:
            qubit_hamiltonian: The :class:`~qdk_chemistry.data.QubitHamiltonian`.

        Returns:
            tuple[float, np.ndarray]: The ground state energy and corresponding eigenstate.

        """


class QubitHamiltonianSolverFactory(base.AlgorithmFactory):
    """Factory class for creating QubitHamiltonianSolver instances."""

    def algorithm_type_name(self) -> str:
        """Return qubit_hamiltonian_solver as the algorithm type name."""
        return "qubit_hamiltonian_solver"

    def default_algorithm_name(self) -> str:
        """Return qdk_sparse_matrix_solver as the default algorithm name."""
        return "qdk_sparse_matrix_solver"


class SparseMatrixSolverSettings(data.Settings):
    """Settings configuration for a SparseMatrixSolver.

    SparseMatrixSolver-specific settings:
        tol (float, default=1e-8): Tolerance for the solver.
        max_m (int, default=20): Maximum subspace dimension for the solver.

    """

    def __init__(self):
        """Initialize SparseMatrixSolverSettings."""
        super().__init__()
        self._set_default("tol", "float", 1e-8)
        self._set_default("max_m", "int", 20)


class SparseMatrixSolver(QubitHamiltonianSolver):
    """Qubit Hamiltonian solver using sparse matrix methods."""

    def __init__(self, tol: float = 1e-8, max_m: int = 20):
        """Initialize the SparseMatrixSolver."""
        super().__init__()
        self._settings = SparseMatrixSolverSettings()
        self._settings.set("tol", tol)
        self._settings.set("max_m", max_m)

    def _run_impl(self, qubit_hamiltonian: data.QubitHamiltonian) -> tuple[float, np.ndarray]:
        """Solve a qubit Hamiltonian using sparse matrix methods.

        Args:
            qubit_hamiltonian: The :class:`~qdk_chemistry.data.QubitHamiltonian`.

        Returns:
            tuple[float, np.ndarray]: The ground state energy and corresponding eigenstate.

        """
        sparse_matrix = qubit_hamiltonian.pauli_ops.to_matrix(sparse=True)
        sparse_matrix_real = sparse_matrix.real.copy()
        eigenvalue, eigenvector = davidson_solver(
            sparse_matrix_real, tol=self._settings.get("tol"), max_m=self._settings.get("max_m")
        )
        return eigenvalue, eigenvector

    def name(self) -> str:
        """Return the name of the qubit hamiltonian solver."""
        return "qdk_sparse_matrix_solver"


class DenseMatrixSolver(QubitHamiltonianSolver):
    """Qubit Hamiltonian solver using dense matrix methods."""

    def __init__(self):
        """Initialize the DenseMatrixSolver."""
        super().__init__()

    def _run_impl(self, qubit_hamiltonian: data.QubitHamiltonian) -> tuple[float, np.ndarray]:
        """Solve a qubit Hamiltonian using dense matrix methods.

        Args:
            qubit_hamiltonian: The :class:`~qdk_chemistry.data.QubitHamiltonian`.

        Returns:
            tuple[float, np.ndarray]: The ground state energy and corresponding eigenstate.

        """
        dense_matrix = qubit_hamiltonian.pauli_ops.to_matrix()
        dense_matrix_real = dense_matrix.real.copy()
        eigenvalues, eigenvectors = syev_solver(dense_matrix_real)
        ground_state_energy = eigenvalues[0]
        ground_state_vector = eigenvectors[:, 0]
        return ground_state_energy, ground_state_vector

    def name(self) -> str:
        """Return the name of the qubit hamiltonian solver."""
        return "qdk_dense_matrix_solver"
