"""Utility functions for SOSSA resource-estimation examples."""

import math

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    FactorizedHamiltonianContainer,
    Hamiltonian,
    MajoranaMapping,
    ModelOrbitals,
    StateVectorContainer,
    Wavefunction,
)


def heisenberg_queries(lambda_effective: float, target_precision: float) -> int:
    """Return the Heisenberg-limited unary-QPE query count.

    Args:
        lambda_effective: Effective SOSSA normalization, in the same energy
            units as ``target_precision``.
        target_precision: Desired positive energy precision.

    Returns:
        The ceiling of ``pi * lambda_effective / (2 * target_precision)``,
        bounded below by one query.

    """
    return max(
        1,
        math.ceil(math.pi * lambda_effective / (2.0 * target_precision)),
    )


def make_fake_hamiltonian(
    n: int,
    r: int,
    b: int,
    c: int,
    seed: int = 42,
) -> Hamiltonian:
    """Create a deterministic synthetic factorized Hamiltonian.

    Args:
        n: Number of spatial orbitals.
        r: Number of factorization ranks.
        b: Number of basis vectors per rank.
        c: Number of coefficient copies per basis vector.
        seed: Random seed used to generate the synthetic tensors.

    Returns:
        A Hamiltonian backed by a synthetic factorized container with dimensions
        ``(N, R, B, C)``.

    """
    rng = np.random.default_rng(seed)
    one_body = rng.standard_normal((n, n))
    one_body = (one_body + one_body.T) / 2

    basis_vectors = rng.standard_normal((r, b, n))
    basis_vectors /= np.linalg.norm(basis_vectors, axis=-1, keepdims=True)

    container = FactorizedHamiltonianContainer(
        one_body_integrals=one_body,
        u_matrices=basis_vectors.ravel(),
        w_matrices=rng.standard_normal((r, b, c)).ravel() * 0.1,
        wb_matrix=rng.standard_normal((r, c)) * 0.1,
        orbitals=ModelOrbitals(n),
        core_energy=0.0,
        inactive_fock_matrix=np.zeros_like(one_body),
    )
    return Hamiltonian(container)


def sossa_unary_qpe_circuit(
    hamiltonian: Hamiltonian,
    n_alpha: int,
    n_beta: int,
    num_queries: int | None = None,
    circuit_mapper: AlgorithmRef | None = None,
    rotation_bit_precision: int = 15,
    coefficient_bit_precision: int = 11,
    target_precision: float = 1e-3,
):
    """Build a unary-iteration QPE circuit driven by a SOSSA walk.

    Args:
        hamiltonian: Hamiltonian backed by a factorized Hamiltonian container.
        n_alpha: Number of alpha electrons in the reference determinant.
        n_beta: Number of beta electrons in the reference determinant.
        num_queries: Number of walk queries. When omitted, use the raw SOSSA
            normalization to derive a conservative Heisenberg-limited count.
        circuit_mapper: Optional SOSSA circuit mapper configuration.
        rotation_bit_precision: Bits used to discretize Givens rotation angles.
        coefficient_bit_precision: Bits used to discretize PREPARE amplitudes.
        target_precision: Target energy precision used when deriving the query count.

    Returns:
        A tuple containing the QPE circuit and SOS qubit operator.

    """
    container = hamiltonian.get_container()
    num_orbitals = container.get_num_orbitals()
    orbitals = container.get_orbitals()

    operator = create("qubit_mapper", "sum_of_squares").run(
        hamiltonian,
        MajoranaMapping.jordan_wigner(2 * num_orbitals),
    )

    if num_queries is None:
        walk = create("hamiltonian_unitary_builder", "sossa").run(operator)
        num_queries = heisenberg_queries(
            walk.get_container().normalization,
            target_precision,
        )

    hf_configuration = Configuration.canonical_hf_configuration(
        n_alpha,
        n_beta,
        num_orbitals,
    )
    reference = Wavefunction(StateVectorContainer(hf_configuration, orbitals))
    state_preparation = create("state_prep", "sparse_isometry").run(reference)

    if circuit_mapper is None:
        circuit_mapper = AlgorithmRef(
            "circuit_mapper",
            "sossa",
            outer_prepare_algorithm=AlgorithmRef("state_prep", "alias_sampling"),
            inner_prepare_algorithm="controlled_alias_sampling",
            select_algorithm="qrom_phase_gradient",
            rotation_bit_precision=rotation_bit_precision,
            coefficient_bit_precision=coefficient_bit_precision,
        )

    builder = create(
        "qpe_circuit_builder",
        "qdk_unary",
        num_queries=num_queries,
        circuit_mapper=circuit_mapper,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
    )
    circuit = builder.run(
        state_preparation=state_preparation,
        qubit_hamiltonian=operator,
    )[0]
    return circuit, operator
