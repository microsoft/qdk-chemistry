"""QDK/Chemistry Sum of Squares Spectral Amplification (SOSSA) circuit mapper :cite:`Low2025`."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import replace
from typing import Any

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.block_encoding.sossa import SOSSABuilder
from qdk_chemistry.algorithms.state_preparation.state_preparation import StatePreparation
from qdk_chemistry.data import AlgorithmRef, SettingNotFound, Settings
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.sossa import SOSSABlockEncodingContainer, SOSSAInnerPrepare
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import CircuitMapper

__all__: list[str] = [
    "SOSSAMapper",
    "SOSSAMapperSettings",
]


def _sossa_mapper_setting_defaults() -> tuple[tuple[Any, ...], ...]:
    """Return the ``_set_default`` arguments of the settings that choose the SOSSA oracles.

    Shared by :class:`SOSSAMapperSettings` and the controlled SOSSA walk mapper's settings, so the
    two mappers build the same block encoding from the same options.

    Returns:
        One argument tuple per setting.

    """
    return (
        (
            "outer_prepare_algorithm",
            "algorithm_ref",
            AlgorithmRef("state_prep", "alias_sampling"),
        ),
        (
            "inner_prepare_algorithm",
            "string",
            "controlled_alias_sampling",
            "Inner PREPARE algorithm.",
            ["controlled_alias_sampling", "one_hot", "direct"],
        ),
        (
            "select_algorithm",
            "string",
            "qrom_phase_gradient",
            "SELECT algorithm.",
            ["qrom_phase_gradient", "one_hot", "direct"],
        ),
        (
            "one_hot_vector_pairing",
            "bool",
            False,
            "Pair angle vectors in one-hot SELECT.",
        ),
        (
            "spin_swap",
            "string",
            "controlled_swap",
            "Spin-swap implementation in SELECT.",
            ["controlled_swap", "bell_parity"],
        ),
        (
            "rotation_bit_precision",
            "int",
            10,
            "Number of bits for Givens rotation angle precision, from 1 to 30 inclusive.",
            (1, 30),
        ),
        (
            "compressed_angle_representation",
            "bool",
            False,
            "Use range-reduced Givens angle words.",
        ),
        (
            "coefficient_bit_precision",
            "int",
            10,
            "Number of bits for alias sampling coefficient precision, from 1 to 30 inclusive.",
            (1, 30),
        ),
        (
            "max_givens_rotation_batches",
            "int",
            1,
            "Maximum number of Givens rotation batches.",
            (1, 4096),
        ),
        (
            "lookup_method",
            "string",
            "select_swap",
            "Lookup method for Givens and inner alias-sampling tables.",
            ["select", "select_swap", "dirty_select_swap", "skew_tree"],
        ),
        (
            "max_swap_address_bits",
            "int",
            -1,
            "Maximum QROAM swap-address bits (-1: automatic, 0: no swaps).",
            (-1, 30),
        ),
        (
            "amortize_batch_lookups",
            "bool",
            False,
            "Amortize consecutive streamed Givens lookups.",
        ),
        (
            "measure_angle_words",
            "bool",
            False,
            "Erase streamed Givens angle words by measurement.",
        ),
        (
            "recompute_outer_registers",
            "bool",
            False,
            "Recompute free-rider registers from the outer index.",
        ),
        (
            "givens_rotation_layout",
            "string",
            "sweep",
            "Givens rotation layout in SELECT.",
            ["sweep", "tree"],
        ),
        (
            "givens_step",
            "string",
            "two_adders",
            "Phase-gradient implementation of each Givens rotation.",
            ["two_adders", "controlled_adder"],
        ),
    )


def _compressed_givens_angles(angles: np.ndarray, rot_bits: int) -> np.ndarray:
    r"""Reduce quantized Givens rows to :math:`[0, \pi)` up to a sign canceled by the SOSSA walk :cite:`Caesura2025`."""
    full = 1 << (rot_bits - 1)
    half = full >> 1
    scaled = (2.0**rot_bits * np.asarray(angles, dtype=float)) / (4.0 * np.pi)
    # Q# Round sends exact halves toward zero; working on its words adds no quantization error.
    words = (np.sign(scaled) * np.ceil(np.abs(scaled) - 0.5)).astype(np.int64) % full
    reduced = np.zeros_like(words)
    for row_index, row in enumerate(words):
        # Shifting angle j by pi leaves Z_j Z_{j+1} on the output side, and the Z_{j+1} moved past gate j + 1
        # negates its angle, so the sweep's first-applied (last) angle is reduced first.
        negate = False
        for j in range(len(row) - 1, -1, -1):
            word = (-int(row[j]) if negate else int(row[j])) % full
            negate = word >= half
            reduced[row_index, j] = word - half if negate else word
    return reduced * (4.0 * np.pi / 2**rot_bits)


class SOSSAMapperSettings(Settings):
    """Settings for the SOSSAMapper."""

    def __init__(self):
        """Initialize settings for SOSSAMapper."""
        super().__init__()
        for default in _sossa_mapper_setting_defaults():
            self._set_default(*default)


class SOSSAMapper(CircuitMapper):
    r"""Circuit mapper for the Sum of Squares Spectral Amplification (SOSSA) block encoding :cite:`Low2025`.

    Emits uncontrolled :math:`B = U^\dagger \cdot \mathrm{Ref}_B \cdot U`
    on ``[system | ancillas | phase gradient]``. The phase-gradient suffix is
    shared ancilla and is excluded from the reflected zero register.

    Unary QPE controls walk iterations through its own index register. Iterative and
    standard QPE query controlled walk powers instead from
    :class:`~qdk_chemistry.algorithms.controlled_circuit_mapper.controlled_sossa_mapper.ControlledSOSSAMapper`.
    """

    def __init__(self):
        """Initialize the SOSSAMapper."""
        super().__init__()
        self._settings = SOSSAMapperSettings()

    def name(self) -> str:
        """Return the algorithm name."""
        return "sossa"

    def type_name(self) -> str:
        """Return the algorithm type name."""
        return "circuit_mapper"

    def _build_outer_prepare_circuit(self, container: SOSSABlockEncodingContainer) -> Circuit:
        r"""Build the outer PREPARE circuit.

        Args:
            container: The SOSSA container with outer_prepare coefficients.

        Returns:
            The PREPARE circuit to embed in the block encoding.

        """
        prepare_algorithm = self._create_nested("outer_prepare_algorithm")
        prepare_settings = prepare_algorithm.settings()
        # Configure by capability so aliases of these PREPARE implementations keep the same SOSSA contract.
        for key, value in (
            ("bits_precision", self._settings.get("coefficient_bit_precision")),
            ("allocate_phase_gradient", False),
        ):
            try:
                prepare_settings.get(key)
            except SettingNotFound:
                continue
            prepare_settings.set(key, value)
        circuit = prepare_algorithm.run(container.outer_prepare)
        if circuit._qsharp_op is None:  # noqa: SLF001
            raise ValueError("The outer PREPARE circuit has no Q# operation to embed in the SOSSA block encoding.")
        if circuit.num_qubits is None:
            raise ValueError(
                f"State preparation '{prepare_algorithm.name()}' does not declare num_qubits, so the "
                "outer register cannot be sized."
            )
        return circuit

    def _build_inner_oracles(self, container: SOSSABlockEncodingContainer) -> tuple[Any, Any]:
        """Build the Q# inner PREPARE and free-rider load callables."""
        algorithm = self._settings.get("inner_prepare_algorithm")
        coeff_bits = self._settings.get("coefficient_bit_precision")
        coefficients = container.inner_prepare.conditional_coefficients.tolist()
        free_rider_data = container.inner_prepare.free_rider_data
        free_rider_data = free_rider_data.tolist() if free_rider_data is not None else []

        if algorithm == "controlled_alias_sampling":
            # The walk lends its system register (2N qubits) to the inner PREPARE's QROAM.
            available_dirty = 2 * container.metadata.num_spatial_orbitals if self._borrow_dirty else 0
            return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareAliasSamplingOracles(
                coefficients,
                free_rider_data,
                coeff_bits,
                self._max_swap_address_bits,
                available_dirty,
            )
        if algorithm == "one_hot":
            return QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareOneHotOracles(coefficients, free_rider_data, coeff_bits)
        if algorithm == "direct":
            return (
                QSHARP_UTILS.SOSSAWalk.MakeInnerPrepareDirect(coefficients, free_rider_data),
                QSHARP_UTILS.SOSSAWalk.MakeFreeRiderLoadOp(free_rider_data),
            )
        raise ValueError(f"Unsupported SOSSA inner PREPARE algorithm '{algorithm}'.")

    def _build_select(self, container: SOSSABlockEncodingContainer) -> Any:
        r"""Build the SELECT step.

        Args:
            container: The SOSSA container with rotation angles and structure.

        Returns:
            A Q# callable for the SELECT oracle.

        """
        algorithm = self._settings.get("select_algorithm")
        rot_bits = self._settings.get("rotation_bit_precision")

        meta = container.metadata
        num_free_rider_bits = container.layout.num_free_rider_bits
        inner_prep_bits = container.layout.inner_prep_bits
        inner_algorithm = self._settings.get("inner_prepare_algorithm")
        if inner_algorithm in ("controlled_alias_sampling", "one_hot"):
            sign_qubit_index = 2 * inner_prep_bits + 2 * self._settings.get("coefficient_bit_precision") + 1
        elif inner_algorithm == "direct":
            sign_qubit_index = inner_prep_bits
        else:
            raise ValueError(f"Unsupported SOSSA inner PREPARE algorithm '{inner_algorithm}'.")

        num_angles = max(meta.num_spatial_orbitals - 1, 0)
        max_givens_rotation_batches = int(self._settings.get("max_givens_rotation_batches"))
        if algorithm == "qrom_phase_gradient" and 0 < num_angles < max_givens_rotation_batches:
            raise ValueError(
                f"max_givens_rotation_batches must be at most the {num_angles} rotation angles of a "
                f"{meta.num_spatial_orbitals}-orbital system, got {max_givens_rotation_batches}"
            )
        if algorithm == "one_hot" and max_givens_rotation_batches != 1:
            raise ValueError(
                "select_algorithm 'one_hot' keeps one qubit per angle vector resident and does not stream "
                f"Givens angle batches, so max_givens_rotation_batches must be 1, got {max_givens_rotation_batches}"
            )
        one_hot_pairs = bool(self._settings.get("one_hot_vector_pairing"))
        if one_hot_pairs and (algorithm != "one_hot" or self._settings.get("givens_step") != "controlled_adder"):
            raise ValueError(
                "one_hot_vector_pairing needs select_algorithm 'one_hot' and givens_step 'controlled_adder'"
            )

        spin_parity = self._settings.get("spin_swap") == "bell_parity"
        recompute = bool(self._settings.get("recompute_outer_registers"))
        # The walk reads isSF as the first free-rider qubit, which recompute_outer_registers drops and the
        # one-hot inner PREPARE displaces with its x_o register, so those keep both swaps in SELECT.
        shared_spin_parity = spin_parity and not recompute and inner_algorithm != "one_hot"

        one_body_angles = container.select.one_body_rotation_angles
        two_body_angles = container.select.two_body_rotation_angles
        compressed = bool(self._settings.get("compressed_angle_representation"))
        controlled_adder = self._settings.get("givens_step") == "controlled_adder"
        if compressed:
            # The controlled adder's words sit on the one-bit-finer grid of the full angle.
            grid_bits = rot_bits + int(controlled_adder)
            one_body_angles = _compressed_givens_angles(np.asarray(one_body_angles), grid_bits)
            two_body_angles = _compressed_givens_angles(np.asarray(two_body_angles), grid_bits)

        one_hot_angles: list[list[float]] = []
        one_hot_dq_index: list[int] = []
        one_hot_sf_index: list[int] = []
        if algorithm == "one_hot":
            # Distinct nonzero vectors by exact value; b == B rows map to none since SELECT skips them.
            vector_index: dict[tuple[float, ...], int] = {}
            two_body = np.asarray(two_body_angles)
            rows = [(True, row) for row in np.asarray(one_body_angles)] + [
                (False, two_body[idx] if idx // meta.num_ranks < meta.num_bases and idx < len(two_body) else None)
                for idx in range(meta.num_ranks * (meta.num_bases + 1))
            ]
            for is_dq, row in rows:
                index = -1
                if row is not None and np.any(row != 0.0):
                    index = vector_index.setdefault(tuple(float(angle) for angle in row), len(vector_index))
                (one_hot_dq_index if is_dq else one_hot_sf_index).append(index)
            one_hot_angles = [list(vector) for vector in vector_index]

        select_data = {
            "numOrbitals": meta.num_spatial_orbitals,
            "numRanks": meta.num_ranks,
            "numBases": meta.num_bases,
            "numCopies": meta.num_copies,
            "numPositiveOneBody": meta.num_positive_one_body_terms,
            "OneBodyRotationAngles": np.asarray(one_body_angles).tolist(),
            "TwoBodyRotationAngles": np.asarray(two_body_angles).tolist(),
            "rotationBitPrecision": rot_bits,
            "compressedAngleWords": compressed,
            # Narrowest lambda within max_givens_rotation_batches passes; 6 on 19 angles gives lambda 4, so 5 passes.
            "rotationBatchSize": -(-num_angles // max_givens_rotation_batches),
            "borrowDirty": self._borrow_dirty,
            "maxSwapBits": self._max_swap_address_bits,
            "amortizeBatchLookups": bool(self._settings.get("amortize_batch_lookups")),
            "measureAngleWords": bool(self._settings.get("measure_angle_words")),
            "useSkewTree": self._settings.get("lookup_method") == "skew_tree",
            "recomputeOuterRegisters": recompute,
            "givensTree": self._settings.get("givens_rotation_layout") == "tree",
            "givensControlledAdder": controlled_adder,
            "numFreeRiderBits": num_free_rider_bits,
            "signQubitIndex": sign_qubit_index,
            "oneHotAngles": one_hot_angles,
            "oneHotDQIndex": one_hot_dq_index,
            "oneHotSFIndex": one_hot_sf_index,
            "oneHotPairedVectors": one_hot_pairs,
            "bellParitySpinSwap": spin_parity,
            # The mapper only embeds SELECT in the walk, which applies the shared trailing parity swap.
            "walkSharesSpinParity": shared_spin_parity,
        }
        if algorithm in ("qrom_phase_gradient", "one_hot"):
            return QSHARP_UTILS.SOSSAWalk.MakeSelectPhaseGradient(select_data)
        if algorithm == "direct":
            return QSHARP_UTILS.SOSSAWalk.MakeSelectDirectRotation(select_data)
        raise ValueError(f"Unsupported SOSSA SELECT algorithm '{algorithm}'.")

    def _compute_register_sizes(
        self, container: SOSSABlockEncodingContainer, outer_prepare_circuit: Circuit
    ) -> tuple[dict[str, int], Any]:
        """Compute the register widths and the Q# ``SOSSAWalkLayout`` describing them.

        Args:
            container: The SOSSA block encoding container.
            outer_prepare_circuit: The outer PREPARE circuit already created for this block encoding.

        Returns:
            The width map, and the Q# ``SOSSAWalkLayout`` built from it.

        """
        meta = container.metadata
        layout = container.layout
        num_orbitals = meta.num_spatial_orbitals
        num_system_qubits = 2 * num_orbitals
        outer_prep_bits = layout.outer_prep_bits
        inner_prep_bits = layout.inner_prep_bits
        num_free_rider_bits = layout.num_free_rider_bits

        outer_prepare_width = outer_prepare_circuit.num_qubits
        if outer_prepare_width is None:
            raise ValueError(
                "The outer PREPARE circuit does not declare num_qubits, so the outer register cannot be sized."
            )
        outer_gradient_bits = outer_prepare_circuit.metadata.num_phase_gradient_ancillas
        num_outer_qubits = outer_prepare_width - outer_gradient_bits
        if num_outer_qubits < outer_prep_bits:
            raise ValueError(
                f"The outer PREPARE circuit owns {num_outer_qubits} non-gradient qubits, but the SOSSA "
                f"layout indexes {outer_prep_bits}."
            )

        inner_algorithm = self._settings.get("inner_prepare_algorithm")
        num_free_rider_qubits = num_free_rider_bits
        if inner_algorithm in ("controlled_alias_sampling", "one_hot"):
            mu_inner = self._settings.get("coefficient_bit_precision")
            if inner_algorithm == "one_hot":
                # The one-hot x_o register sits before the free-rider bits and is loaded with them.
                num_free_rider_qubits += container.inner_prepare.conditional_coefficients.shape[0]
            num_inner_qubits = 2 * inner_prep_bits + 2 * mu_inner + 3 + num_free_rider_qubits
            num_reflect_inner = inner_prep_bits + mu_inner + 1
        elif inner_algorithm == "direct":
            # The extra qubit is the sign bit SELECT phases
            num_inner_qubits = inner_prep_bits + 1 + num_free_rider_bits
            num_reflect_inner = inner_prep_bits
        else:
            raise ValueError(f"Unsupported SOSSA inner PREPARE algorithm '{inner_algorithm}'.")

        select_algorithm = self._settings.get("select_algorithm")
        if select_algorithm in ("qrom_phase_gradient", "one_hot"):
            select_gradient_bits = int(self._settings.get("rotation_bit_precision"))
        elif select_algorithm == "direct":
            select_gradient_bits = 0
        else:
            raise ValueError(f"Unsupported SOSSA SELECT algorithm '{select_algorithm}'.")
        if outer_gradient_bits and select_gradient_bits and outer_gradient_bits != select_gradient_bits:
            raise ValueError(
                "The outer PREPARE and SELECT share one phase gradient register and must agree "
                f"on its width. Now, the outer PREPARE: {outer_gradient_bits} and SELECT: {select_gradient_bits}."
            )
        num_phase_gradient_qubits = max(outer_gradient_bits, select_gradient_bits)
        num_spin_qubits = 2  # spinDQ + spinSF, matches Q# SOSSAWalk.qs

        regs = {
            "num_system_qubits": num_system_qubits,
            "num_outer_qubits": num_outer_qubits,
            "num_outer_index_qubits": outer_prep_bits,
            "num_inner_qubits": num_inner_qubits,
            "num_reflect_inner": num_reflect_inner,
            "num_phase_gradient_qubits": num_phase_gradient_qubits,
            "num_outer_prepare_gradient_qubits": outer_gradient_bits,
            "num_ancilla_qubits": num_outer_qubits + num_reflect_inner + num_spin_qubits + num_phase_gradient_qubits,
        }
        register_layout = QSHARP_UTILS.SOSSAWalk.SOSSAWalkLayout(
            numSystemQubits=regs["num_system_qubits"],
            numOuterQubits=regs["num_outer_qubits"],
            numOuterIndexQubits=regs["num_outer_index_qubits"],
            numOuterPrepareGradientQubits=regs["num_outer_prepare_gradient_qubits"],
            numInnerQubits=regs["num_inner_qubits"],
            numReflectInner=regs["num_reflect_inner"],
            numFreeRiderQubits=num_free_rider_qubits,
            numPhaseGradientQubits=regs["num_phase_gradient_qubits"],
            sharedSpinParity=self._settings.get("spin_swap") == "bell_parity"
            and not self._settings.get("recompute_outer_registers")
            and inner_algorithm != "one_hot",
        )
        return regs, register_layout

    def build_block_encoding_ops(
        self, container: SOSSABlockEncodingContainer
    ) -> tuple[dict[str, Any], dict[str, int], Any]:
        """Build the Q# oracles of the block encoding and the registers they act on.

        The controlled SOSSA walk mapper composes the same oracles into a controlled walk.

        Args:
            container: The SOSSA block encoding container.

        Returns:
            The ``outerPrepareOp``, ``freeRiderOp``, ``innerPrepareOp`` and ``selectOp`` callables keyed by
            their Q# parameter names, the register width map, and the Q# ``SOSSAWalkLayout``.

        Raises:
            ValueError: If the layout reserves free-rider bits but the container carries no free-rider table.

        """
        algorithm = self._settings.get("select_algorithm")
        rot_bits = self._settings.get("rotation_bit_precision")
        compressed = bool(self._settings.get("compressed_angle_representation"))
        controlled_adder = self._settings.get("givens_step") == "controlled_adder"
        if compressed:
            if algorithm not in ("qrom_phase_gradient", "one_hot"):
                raise ValueError(
                    "compressed_angle_representation needs the 'qrom_phase_gradient' or 'one_hot' SELECT, which store "
                    f"angle words; the '{algorithm}' SELECT has none to compress."
                )
            if self._settings.get("givens_rotation_layout") != "sweep":
                raise ValueError(
                    "compressed_angle_representation needs givens_rotation_layout 'sweep': it carries each sign "
                    "flip into the next angle of the sweep chain, which the tree's angles do not follow."
                )
            # A full-angle word is the half-angle word one bit wider, so it keeps b - 1 of b + 1 bits.
            dropped_bits = 1 if controlled_adder else 2
            if rot_bits < dropped_bits + 1:
                raise ValueError(
                    f"compressed_angle_representation drops {dropped_bits} of the rotation_bit_precision bits, "
                    f"so it needs at least {dropped_bits + 1}, got {rot_bits}."
                )

        if self._settings.get("recompute_outer_registers"):
            meta = container.metadata
            if meta.num_copies == 0:
                raise ValueError(
                    "recompute_outer_registers recomputes the spin-free flag and rank from x_o, so it needs at "
                    "least one spin-free copy, got num_copies=0."
                )
            num_orbitals = meta.num_spatial_orbitals
            num_sf = meta.num_ranks * meta.num_copies
            rank_span = 1 << container.layout.rank_bits
            sf_span = rank_span * meta.num_copies
            # Low 2026 Eq. E9 places SF (r, c) at 2^nR*c + r, then D1 and Q1 from sf_span on.
            new_index = np.array(
                [sf_span + a for a in range(num_orbitals)]
                + [(row % meta.num_copies) * rank_span + row // meta.num_copies for row in range(num_sf)]
            )
            old_amplitudes, _ = StatePreparation._dense_state_vector(  # noqa: SLF001
                container.outer_prepare, "SOSSA outer PREPARE"
            )
            outer_amplitudes = np.zeros(sf_span + num_orbitals)
            outer_amplitudes[new_index] = old_amplitudes[: num_orbitals + num_sf]
            old_rows = container.inner_prepare.conditional_coefficients
            # Padding indices carry no outer weight, so any normalized inner row is valid there.
            inner_rows = np.zeros((sf_span + num_orbitals, old_rows.shape[1]))
            inner_rows[:, 0] = 1.0
            inner_rows[new_index] = old_rows
            outer_prep_bits = max(1, (sf_span + num_orbitals - 1).bit_length())
            container = SOSSABlockEncodingContainer(
                outer_prepare=SOSSABuilder._build_outer_prepare(outer_amplitudes, outer_prep_bits),  # noqa: SLF001
                inner_prepare=SOSSAInnerPrepare(conditional_coefficients=inner_rows),
                select=container.select,
                metadata=meta,
                layout=replace(container.layout, outer_prep_bits=outer_prep_bits, num_free_rider_bits=0),
                normalization=container.normalization,
                power=container.power,
                lambda_eff=container._lambda_eff,  # noqa: SLF001
            )

        free_rider = container.inner_prepare.free_rider_data
        if container.layout.num_free_rider_bits and (free_rider is None or free_rider.size == 0):
            raise ValueError(
                f"The register layout reserves {container.layout.num_free_rider_bits} free-rider bits "
                "but the container carries no free-rider table."
            )

        outer_prepare_circuit = self._build_outer_prepare_circuit(container)
        regs, register_layout = self._compute_register_sizes(container, outer_prepare_circuit)
        # Resolved per run (settings stay mutable); "select" is a width-0 cap, so only borrowing is passed on.
        lookup_method = self._settings.get("lookup_method")
        self._max_swap_address_bits = (
            0 if lookup_method == "select" else int(self._settings.get("max_swap_address_bits"))
        )
        self._borrow_dirty = lookup_method == "dirty_select_swap"
        inner_prepare_op, free_rider_op = self._build_inner_oracles(container)
        ops = {
            "outerPrepareOp": outer_prepare_circuit._qsharp_op,  # noqa: SLF001
            "freeRiderOp": free_rider_op,
            "innerPrepareOp": inner_prepare_op,
            "selectOp": self._build_select(container),
        }
        return ops, regs, register_layout

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct the SOSSA block encoding on the flat ``[system | ancilla]`` register.

        Args:
            unitary: The unitary representation containing the SOSSA decomposition.

        Returns:
            Circuit: The block encoding :math:`B`, declaring the full register width and the
            phase gradient qubits its caller must prepare.

        Raises:
            ValueError: If the container is not a :class:`SOSSABlockEncodingContainer` or its
                power is not one.

        """
        container = unitary.get_container()
        if not isinstance(container, SOSSABlockEncodingContainer):
            raise ValueError(f"The {unitary.get_container_type()} container type is not supported.")
        if container.power != 1:
            raise ValueError(f"The SOSSA mapper supports only unit power, got {container.power}.")

        ops, regs, register_layout = self.build_block_encoding_ops(container)
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.SOSSAWalk.MakeSOSSABlockEncodingCircuit,
            parameter={**ops, "layout": register_layout},
        )
        qsharp_op = QSHARP_UTILS.SOSSAWalk.MakeSOSSABlockEncodingOp(
            ops["outerPrepareOp"], ops["freeRiderOp"], ops["innerPrepareOp"], ops["selectOp"], register_layout
        )

        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            num_qubits=regs["num_system_qubits"] + regs["num_ancilla_qubits"],
            metadata=CircuitMetadata(num_phase_gradient_ancillas=regs["num_phase_gradient_qubits"]),
        )
