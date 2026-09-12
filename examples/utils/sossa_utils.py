"""Utility functions for the SOSSA resource-estimation example.

The notebook walks the SOSSA pipeline through once by hand. These helpers cover
the parts that would otherwise be repeated: the Heisenberg-limited query count,
the synthetic Hamiltonian generator used by the trade-off sweep, and the guards
that keep an oversized problem away from the state-vector simulator.
"""

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

#: Container type emitted by the ``sossa`` hamiltonian unitary builder.
SOSSA_WALK_CONTAINER_TYPE = "sossa_walk"

#: The only QPE circuit builder that accepts a SOSSA walk. Iterative and standard QPE
#: drive a walk through a ``controlled_circuit_mapper``, and no controlled SOSSA mapper
#: exists, so they cannot build a SOSSA circuit at all.
SOSSA_QPE_BUILDER = "qdk_unary"


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


def require_sossa_walk(unitary_representation):
    """Return the SOSSA walk container, rejecting any other block encoding.

    The ``qdk_unary`` builder performs the same check and raises on a mismatch.
    Calling this first turns a mis-wired pipeline into an error at the point the
    walk is built, rather than several cells later.

    Args:
        unitary_representation: Result of running a ``hamiltonian_unitary_builder``.

    Returns:
        The underlying SOSSA walk container.

    Raises:
        TypeError: If the representation does not hold a SOSSA walk.

    """
    container = unitary_representation.get_container()
    if container.type != SOSSA_WALK_CONTAINER_TYPE:
        raise TypeError(
            f"Expected a '{SOSSA_WALK_CONTAINER_TYPE}' container from the 'sossa' "
            f"hamiltonian unitary builder, got '{container.type}'. Spectrum amplification "
            "only applies to a sum-of-squares block encoding."
        )
    return container


def simulation_qubit_estimate(walk_container, num_queries: int) -> int:
    """Return a screening estimate of the qubits a SOSSA QPE circuit would use.

    The estimate is the walk's own register total plus the phase register that
    addresses ``num_queries + 1`` reflection slots. It is a screen, not an exact
    width: the compiled oracles a circuit mapper emits can allocate additional
    alias-sampling and phase-gradient ancillas on top of it.

    It is deliberately computed from the container alone, so an oversized problem
    can be rejected before any circuit is constructed.

    Args:
        walk_container: SOSSA walk container from :func:`require_sossa_walk`.
        num_queries: Number of walk queries the schedule would apply.

    Returns:
        The estimated number of qubits.

    """
    return walk_container.num_qubits + int(num_queries).bit_length()


def make_fake_hamiltonian(
    n: int,
    r: int,
    b: int,
    c: int,
    seed: int = 42,
) -> Hamiltonian:
    """Create a deterministic synthetic factorized Hamiltonian.

    The tensors are random, so the resulting spectrum is meaningless. Only the
    ``(N, R, B, C)`` dimensions are physical, and those are what drive circuit
    size and therefore the resource estimate. Never interpret an energy computed
    from one of these.

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


def compiled_circuit_mapper(
    rotation_bit_precision: int = 15,
    coefficient_bit_precision: int = 11,
) -> AlgorithmRef:
    """Return the SOSSA circuit mapper used for resource estimation.

    Alias sampling and QROM-with-phase-gradient are the compilations a
    fault-tolerant cost model should see. They are too wide for a state-vector
    simulator, which is why the notebook validates with the ``direct`` oracles
    instead.

    Args:
        rotation_bit_precision: Bits used to discretize Givens rotation angles.
        coefficient_bit_precision: Bits used to discretize PREPARE amplitudes.

    Returns:
        An ``AlgorithmRef`` for the SOSSA circuit mapper.

    """
    return AlgorithmRef(
        "circuit_mapper",
        "sossa",
        outer_prepare_algorithm=AlgorithmRef("state_prep", "alias_sampling"),
        inner_prepare_algorithm="controlled_alias_sampling",
        select_algorithm="qrom_phase_gradient",
        rotation_bit_precision=rotation_bit_precision,
        coefficient_bit_precision=coefficient_bit_precision,
    )


def hartree_fock_state_preparation(hamiltonian: Hamiltonian, n_alpha: int, n_beta: int):
    """Prepare the canonical Hartree-Fock reference state for a factorized Hamiltonian.

    Args:
        hamiltonian: Hamiltonian backed by a factorized Hamiltonian container.
        n_alpha: Number of alpha electrons in the reference determinant.
        n_beta: Number of beta electrons in the reference determinant.

    Returns:
        The state-preparation circuit.

    """
    container = hamiltonian.get_container()
    configuration = Configuration.canonical_hf_configuration(
        n_alpha,
        n_beta,
        container.get_num_orbitals(),
    )
    reference = Wavefunction(
        StateVectorContainer(configuration, container.get_orbitals())
    )
    return create("state_prep", "sparse_isometry").run(reference)


def build_sossa_qpe_circuit(
    hamiltonian: Hamiltonian,
    n_alpha: int,
    n_beta: int,
    num_queries: int,
    circuit_mapper: AlgorithmRef | None = None,
):
    """Run the whole SOSSA pipeline and return the unary-QPE circuit.

    This repeats, in one call, the steps the notebook walks through individually.
    It exists for the trade-off sweep, which builds the same circuit for several
    systems; prefer the explicit steps when reading the pipeline for the first time.

    The walk container is built once, inside the QPE builder, which rejects any
    non-SOSSA block encoding itself. Building it here as well to call
    :func:`require_sossa_walk` would double the most expensive step of the sweep
    without adding a check.

    Args:
        hamiltonian: Hamiltonian backed by a factorized Hamiltonian container.
        n_alpha: Number of alpha electrons in the reference determinant.
        n_beta: Number of beta electrons in the reference determinant.
        num_queries: Number of walk queries.
        circuit_mapper: SOSSA circuit mapper configuration. Defaults to
            :func:`compiled_circuit_mapper`.

    Returns:
        A tuple of the QPE circuit and the sum-of-squares qubit operator.

    """
    container = hamiltonian.get_container()
    operator = create("qubit_mapper", "sum_of_squares").run(
        hamiltonian,
        MajoranaMapping.jordan_wigner(2 * container.get_num_orbitals()),
    )

    builder = create(
        "qpe_circuit_builder",
        SOSSA_QPE_BUILDER,
        num_queries=num_queries,
        circuit_mapper=circuit_mapper
        if circuit_mapper is not None
        else compiled_circuit_mapper(),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "sossa"),
    )
    circuit = builder.run(
        state_preparation=hartree_fock_state_preparation(hamiltonian, n_alpha, n_beta),
        qubit_hamiltonian=operator,
    )[0]
    return circuit, operator
