"""SOSSA qubit mapper for factorized SOS Hamiltonians."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from math import sqrt

import numpy as np

from qdk_chemistry.algorithms.qubit_mapper.qubit_mapper import QubitMapper
from qdk_chemistry.data import FactorizedHamiltonianContainer, Hamiltonian, MajoranaMapping, QubitOperator
from qdk_chemistry.data.sossa_qubit_operator import (
    FermionParity,
    RotatedMode,
    RotatedPauliContainer,
    RotatedPauliTerm,
    SOSContainer,
    SOSGenerator,
    SOSGeneratorKind,
    SpinPolicy,
)

_INV_SQRT_TWO = 1.0 / sqrt(2.0)


class SOSSAQubitMapper(QubitMapper):
    """Map a factorized Hamiltonian to a qubit operator containing SOS generators."""

    def name(self) -> str:
        """Return the algorithm variant name."""
        return "sossa"

    def _run_impl(self, hamiltonian: Hamiltonian, mapping: MajoranaMapping) -> QubitOperator:
        """Convert a factorized Hamiltonian to a structured SOS qubit operator."""
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError("SOSSAQubitMapper requires a Hamiltonian")
        container = hamiltonian.get_container()
        if not isinstance(container, FactorizedHamiltonianContainer):
            raise TypeError("SOSSAQubitMapper requires a Hamiltonian backed by FactorizedHamiltonianContainer")
        return self._map_factorized_container(container, mapping)

    @staticmethod
    def _one_body_generator(
        mapping: MajoranaMapping,
        basis_vector: np.ndarray,
        spin: int,
        num_orbitals: int,
        weight: float,
        creation: bool,
        kind: SOSGeneratorKind,
        source_index: int,
    ) -> SOSGenerator:
        """Construct one spin component of a D1 or Q1 generator."""
        mode = spin * num_orbitals
        scale = sqrt(weight) / 2.0
        rotated_mode = RotatedMode(basis_vector, spin)
        terms = (
            RotatedPauliTerm(scale, dict(mapping.majorana(2 * mode)), rotated_mode),
            RotatedPauliTerm(
                complex(0.0, -scale if creation else scale),
                dict(mapping.majorana(2 * mode + 1)),
                rotated_mode,
            ),
        )
        operator = QubitOperator(RotatedPauliContainer(terms, mapping.num_qubits, mapping.name, "blocked"))
        return SOSGenerator(
            kind,
            FermionParity.Odd,
            SpinPolicy.Specific,
            operator,
            spin=spin,
            source_index=(source_index,),
        )

    @classmethod
    def _map_factorized_container(
        cls,
        container: FactorizedHamiltonianContainer,
        mapping: MajoranaMapping,
    ) -> QubitOperator:
        """Map a validated factorized container to the unified qubit operator wrapper."""
        num_orbitals = container.get_num_orbitals()
        if mapping.base_encoding != "jordan-wigner":
            raise ValueError("SOSSAQubitMapper requires a Jordan-Wigner mapping")
        if not mapping.is_majorana_atomic or mapping.num_modes != 2 * num_orbitals:
            raise ValueError("mapping must provide atomic Majoranas for 2N spin orbitals")
        if mapping.tapering is not None:
            raise ValueError("SOSSAQubitMapper does not support tapered mappings")

        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(container.get_h1_majorana(), dtype=float))
        generators: list[SOSGenerator] = []
        source_index = 0
        for positive in (True, False):
            for index, eigenvalue in enumerate(eigenvalues):
                if (positive and eigenvalue <= 0.0) or (not positive and eigenvalue >= 0.0):
                    continue
                kind = SOSGeneratorKind.D1 if positive else SOSGeneratorKind.Q1
                for spin in (0, 1):
                    generators.append(
                        cls._one_body_generator(
                            mapping,
                            eigenvectors[:, index],
                            spin,
                            num_orbitals,
                            abs(float(eigenvalue)),
                            creation=not positive,
                            kind=kind,
                            source_index=source_index,
                        )
                    )
                source_index += 1

        num_ranks = container.get_num_ranks()
        num_bases = container.get_num_bases()
        num_copies = container.get_num_copies()
        rotations = np.asarray(container.get_u_matrices(), dtype=float)
        weights = np.asarray(container.get_w_matrices(), dtype=float)
        identity_weights = np.asarray(container.get_wb_matrix(), dtype=float)
        for rank in range(num_ranks):
            for copy in range(num_copies):
                terms = [RotatedPauliTerm(identity_weights[rank, copy] * _INV_SQRT_TWO, {})]
                for basis in range(num_bases):
                    offset = (rank * num_bases + basis) * num_orbitals
                    basis_vector = rotations[offset : offset + num_orbitals]
                    weight = weights[(rank * num_bases + basis) * num_copies + copy]
                    for spin in (0, 1):
                        mode = spin * num_orbitals
                        mapping_coefficient, center = mapping.bilinear(2 * mode + 1, 2 * mode)
                        terms.append(
                            RotatedPauliTerm(
                                weight * _INV_SQRT_TWO * 0.5 * mapping_coefficient,
                                dict(center),
                                RotatedMode(basis_vector, spin),
                            )
                        )
                operator = QubitOperator(RotatedPauliContainer(terms, mapping.num_qubits, mapping.name, "blocked"))
                generators.append(
                    SOSGenerator(
                        SOSGeneratorKind.SF,
                        FermionParity.Even,
                        SpinPolicy.Summed,
                        operator,
                        source_index=(rank, copy),
                    )
                )

        negative_sum = float(-np.sum(eigenvalues[eigenvalues < 0.0]))
        w0_square_sum = 0.0
        for rank in range(num_ranks):
            for copy in range(num_copies):
                w0 = identity_weights[rank, copy]
                for basis in range(num_bases):
                    w0 -= weights[(rank * num_bases + basis) * num_copies + copy]
                w0_square_sum += w0 * w0
        energy_shift = (
            container.get_core_energy() + container.get_bliss_shift() - 2.0 * negative_sum - 0.5 * w0_square_sum
        )
        return QubitOperator(
            SOSContainer(
                num_orbitals,
                mapping.num_qubits,
                energy_shift,
                generators,
                mapping.name,
                "blocked",
            )
        )
