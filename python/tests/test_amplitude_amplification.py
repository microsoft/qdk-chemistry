"""Tests for amplitude amplification.

Three layers are exercised against each other:

* the Grover closed form on
  :class:`~qdk_chemistry.algorithms.amplitude_amplification.AmplitudeAmplification`
  are checked against an independent simulation of the two-dimensional invariant
  subspace, so the tests validate the mathematics rather than restating it;
* the Q# module ``QDKChemistry.Utils.AmplitudeAmplification`` is executed on the
  full-state simulator and compared against those same closed forms;
* the registry algorithm ``amplitude_amplification/qdk_amplitude_amplification`` is
  composed with the QPE circuit builder and an external executor, end to end.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.algorithms.amplitude_amplification import AmplitudeAmplification, phase_marking_oracle
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context

OVERLAPS = [1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]


def _simulate(overlap: float, rounds: int) -> float:
    """Return the Grover probability from an explicit 2D rotation."""
    angle = math.asin(math.sqrt(overlap))
    return math.sin((2 * rounds + 1) * angle) ** 2


#
# Round schedule
#


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("rounds", [0, 1, 2, 3, 7, 15])
def test_success_probability_matches_simulation(overlap: float, rounds: int):
    simulated = _simulate(overlap, rounds)
    assert AmplitudeAmplification.success_probability(overlap, rounds) == pytest.approx(simulated, abs=1e-12)


@pytest.mark.parametrize("overlap", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_invalid_overlap_is_rejected(overlap: float):
    with pytest.raises(ValueError, match="overlap"):
        AmplitudeAmplification._rotation_angle(overlap)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="rounds"):
        AmplitudeAmplification.success_probability(0.1, -1)


#
# Q# primitives on the full-state simulator
#

_HAS_QSHARP = importlib.util.find_spec("qdk.qsharp") is not None

_SHOTS = 4000
_TOLERANCE = 0.04

_HARNESS = """
namespace QDKChemistryAmplitudeAmplificationTests {
    open QDKChemistry.Utils.AmplitudeAmplification;
    import QDKChemistry.Utils.StandardPhaseEstimation.MakeStandardQPEOp;
    import Std.Canon.*;
    import Std.Arithmetic.*;
    import Std.Arrays.*;
    import Std.Convert.*;
    import Std.Math.*;
    import Std.Measurement.*;

    operation PrepareOverlap(theta : Double, register : Qubit[]) : Unit is Adj + Ctl {
        Ry(2.0 * theta, register[0]);
    }

    operation MarkFirstQubit(register : Qubit[], target : Qubit) : Unit is Adj + Ctl {
        CNOT(register[0], target);
    }

    operation RunPlainAmplification(theta : Double, rounds : Int) : Result {
        use register = Qubit[1];
        ApplyAmplitudeAmplification(PrepareOverlap(theta, _), MarkFirstQubit, rounds, register);
        return MResetZ(register[0]);
    }

    /// A marking oracle for QPE: the phase register holds an accepted index and
    /// every signal ancilla is |0>. The library ships no oracle, so callers
    /// supply their own; this is the one the QPE tests use.
    operation MarkAcceptedPhase(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        accepted : Int[],
        register : Qubit[],
        target : Qubit,
    ) : Unit is Adj {
        let phaseRegister = register[0..numPhaseQubits - 1];
        let signalAncillas = Subarray(signalAncillaIndices, register[numPhaseQubits...]);
        within {
            ApplyToEachCA(X, signalAncillas);
        } apply {
            for index in accepted {
                Controlled ApplyControlledOnInt(
                    signalAncillas,
                    (index, X, phaseRegister, target),
                );
            }
        }
    }

    function MakeAcceptedPhaseMarkerOp(
        numPhaseQubits : Int,
        signalAncillaIndices : Int[],
        accepted : Int[],
    ) : (Qubit[], Qubit) => Unit is Adj {
        MarkAcceptedPhase(numPhaseQubits, signalAncillaIndices, accepted, _, _)
    }

    operation MarkQpeAcceptance(
        numPhaseQubits : Int,
        phaseValue : Int,
        numAncillas : Int,
        ancillaValue : Int,
        signalAncillaIndices : Int[],
        accepted : Int[],
    ) : Result {
        use register = Qubit[numPhaseQubits + numAncillas];
        use flag = Qubit();
        // The QPE phase register is little-endian, as ApplyXorInPlace assumes.
        ApplyXorInPlace(phaseValue, register[0..numPhaseQubits - 1]);
        ApplyXorInPlace(ancillaValue, register[numPhaseQubits...]);
        MarkAcceptedPhase(numPhaseQubits, signalAncillaIndices, accepted, register, flag);
        let outcome = MResetZ(flag);
        ResetAll(register);
        return outcome;
    }

    operation RunPhaseIndexMarker(
        numPhaseQubits : Int,
        targetIndices : Int[],
        phaseValue : Int,
    ) : Result {
        use register = Qubit[numPhaseQubits];
        use flag = Qubit();
        ApplyXorInPlace(phaseValue, register);
        MarkPhaseIndices(numPhaseQubits, targetIndices, register, flag);
        let outcome = MResetZ(flag);
        ResetAll(register);
        return outcome;
    }

    operation RunPhaseThresholdMarker(
        numPhaseQubits : Int,
        threshold : Int,
        phaseValue : Int,
    ) : Result {
        use register = Qubit[numPhaseQubits];
        use flag = Qubit();
        ApplyXorInPlace(phaseValue, register);
        MarkPhaseAtOrBelow(numPhaseQubits, threshold, register, flag);
        let outcome = MResetZ(flag);
        ResetAll(register);
        return outcome;
    }

    operation PrepareGuidingState(theta : Double, systems : Qubit[]) : Unit is Adj + Ctl {
        Ry(2.0 * theta, systems[0]);
    }

    operation PrepareUniformPhaseRegister(register : Qubit[]) : Unit is Adj + Ctl {
        ApplyToEachCA(H, register);
    }

    /// Controlled U^power for the single-qubit unitary U|1> = exp(2 pi i / 4)|1>.
    operation ApplyControlledPower(power : Int, control : Qubit, targets : Qubit[]) : Unit is Adj + Ctl {
        Controlled R1([control], (2.0 * PI() * IntAsDouble(power) / 4.0, targets[0]));
    }

    /// Amplified QPE on three phase qubits and one system qubit, with the single
    /// bin 2/8 accepted. Returns the phase register least-significant-bit first.
    operation RunAmplifiedQpe(theta : Double, rounds : Int) : Result[] {
        let unitaries = [
            ApplyControlledPower(4, _, _),
            ApplyControlledPower(2, _, _),
            ApplyControlledPower(1, _, _),
        ];
        let signalAncillaIndices : Int[] = [];
        let preparation = MakeStandardQPEOp(
            PrepareGuidingState(theta, _),
            unitaries,
            PrepareUniformPhaseRegister,
            3,
            1,
        );
        let marker = MakeAcceptedPhaseMarkerOp(3, signalAncillaIndices, [2]);
        use register = Qubit[4];
        ApplyAmplitudeAmplification(preparation, marker, rounds, register);
        let outcome = [MResetZ(register[0]), MResetZ(register[1]), MResetZ(register[2])];
        ResetAll(register);
        return outcome;
    }
}
"""

_NAMESPACE = "QDKChemistryAmplitudeAmplificationTests"


@pytest.fixture(scope="module")
def qsharp_module():
    """Load the chemistry Q# utilities plus the test harness exactly once."""
    if not _HAS_QSHARP:
        pytest.skip("qdk.qsharp is not installed")
    # The utilities live on a dedicated context, so the harness has to be
    # evaluated there rather than on the global interpreter.
    context = get_qsharp_context()
    context.eval(_HARNESS)
    return context


def _acceptance_frequency(qsharp_module, expression: str) -> float:
    """Return the fraction of shots that land in the good subspace."""
    outcomes = qsharp_module.run(expression, shots=_SHOTS)
    return sum(1 for outcome in outcomes if str(outcome) == "One") / _SHOTS


@pytest.mark.parametrize(("overlap", "rounds"), [(0.05, 0), (0.05, 2), (0.05, 3), (0.1, 1), (0.1, 2), (0.25, 1)])
def test_plain_amplification_matches_the_closed_form(qsharp_module, overlap: float, rounds: int):
    theta = AmplitudeAmplification._rotation_angle(overlap)
    observed = _acceptance_frequency(qsharp_module, f"{_NAMESPACE}.RunPlainAmplification({theta}, {rounds})")
    assert observed == pytest.approx(AmplitudeAmplification.success_probability(overlap, rounds), abs=_TOLERANCE)


def test_plain_amplification_overshoots_when_the_overlap_is_underestimated(qsharp_module):
    # Three rounds are optimal for a = 0.02 but wrap past the maximum for a = 0.25.
    overlap = 0.25
    rounds = 3
    theta = AmplitudeAmplification._rotation_angle(overlap)
    observed = _acceptance_frequency(qsharp_module, f"{_NAMESPACE}.RunPlainAmplification({theta}, {rounds})")
    assert observed == pytest.approx(AmplitudeAmplification.success_probability(overlap, rounds), abs=_TOLERANCE)
    assert observed < 0.5

    # A single round is what a = 0.25 actually wants.
    better = _acceptance_frequency(qsharp_module, f"{_NAMESPACE}.RunPlainAmplification({theta}, 1)")
    assert better > observed


def test_marking_oracle_conjunction_holds(qsharp_module):
    # The loop is only correct if the oracle flips exactly on the good subspace.
    accepted = [0, 1, 6, 7]
    for phase_value in range(8):
        for ancilla_value in range(4):
            expression = f"{_NAMESPACE}.MarkQpeAcceptance(3, {phase_value}, 2, {ancilla_value}, [0, 1], {accepted})"
            outcome = str(qsharp_module.run(expression, shots=1)[0])
            assert (outcome == "One") == (phase_value in accepted and ancilla_value == 0)


@pytest.mark.parametrize("phase_value", range(8))
def test_phase_marking_oracles_match_their_criteria(qsharp_module, phase_value: int):
    index_result = qsharp_module.run(
        f"{_NAMESPACE}.RunPhaseIndexMarker(3, [1, 6], {phase_value})", shots=1
    )[0]
    threshold_result = qsharp_module.run(
        f"{_NAMESPACE}.RunPhaseThresholdMarker(3, 3, {phase_value})", shots=1
    )[0]
    assert (str(index_result) == "One") == (phase_value in {1, 6})
    assert (str(threshold_result) == "One") == (phase_value <= 3)


def test_phase_marking_oracle_validates_criteria():
    with pytest.raises(ValueError, match="exactly one"):
        phase_marking_oracle(3)
    with pytest.raises(ValueError, match="exactly one"):
        phase_marking_oracle(3, target_indices=[1], threshold=2)
    with pytest.raises(ValueError, match="target index"):
        phase_marking_oracle(3, target_indices=[8])
    with pytest.raises(ValueError, match="threshold"):
        phase_marking_oracle(3, threshold=-1)


@pytest.mark.skipif(not _HAS_QSHARP, reason="qdk.qsharp is not installed")
def test_phase_marking_oracle_returns_circuits():
    index_marker = phase_marking_oracle(3, target_indices=[6, 1, 1])
    threshold_marker = phase_marking_oracle(3, threshold=3)
    assert isinstance(index_marker, Circuit)
    assert isinstance(threshold_marker, Circuit)
    assert index_marker._qsharp_op is not None
    assert threshold_marker._qsharp_op is not None


def test_rotation_angle_agrees_with_the_qsharp_convention():
    # The Q# harness prepares Ry(2 * theta), whose |1> amplitude is sin(theta).
    for overlap in (0.05, 0.25, 0.81):
        assert math.sin(AmplitudeAmplification._rotation_angle(overlap)) ** 2 == pytest.approx(overlap)


@pytest.mark.parametrize("rounds", [0, 1, 2, 3])
def test_amplified_qpe_boosts_acceptance_without_changing_the_answer(qsharp_module, rounds: int):
    # A single system qubit whose |1> eigenstate has phase 1/4, prepared with a
    # deliberately poor 5% overlap. The accepted window is the single bin 2/8.
    overlap = 0.05
    theta = AmplitudeAmplification._rotation_angle(overlap)
    outcomes = qsharp_module.run(f"{_NAMESPACE}.RunAmplifiedQpe({theta}, {rounds})", shots=_SHOTS)

    # The phase register comes back least-significant-bit first.
    indices = [sum(1 << position for position, bit in enumerate(bits) if str(bit) == "One") for bits in outcomes]
    frequency = sum(1 for index in indices if index == 2) / _SHOTS
    assert frequency == pytest.approx(AmplitudeAmplification.success_probability(overlap, rounds), abs=_TOLERANCE)

    # Amplification changes how often the window is accepted, never what it
    # accepts: the only other outcome is the phase-0 bin of the |0> component.
    assert set(indices) <= {0, 2}


#
# Registry algorithm on top of the QPE circuit builder
#


def _diagonal_hamiltonian() -> QubitOperator:
    """Return H = (pi/4) ZI + (pi/4) IZ, whose spectrum is {pi/2, 0, 0, -pi/2}."""
    coefficient = math.pi / 4.0
    return QubitOperator(pauli_strings=["ZI", "IZ"], coefficients=np.array([coefficient, coefficient]))


def _guiding_state(amplitude: float, index: int, num_qubits: int = 2) -> Circuit:
    """Prepare a state with the given amplitude on one computational basis state."""
    vector = [0.0] * (1 << num_qubits)
    vector[index] = amplitude
    remainder = math.sqrt(max(0.0, 1.0 - amplitude**2))
    vector[(index + 1) % (1 << num_qubits)] = remainder

    parameters = {
        "rowMap": list(reversed(range(num_qubits))),
        "stateVector": vector,
        "expansionOps": [],
        "numQubits": num_qubits,
    }
    return Circuit(
        qsharp_factory=QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=parameters,
        ),
        qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(parameters),
    )


def _qpe_preparation(
    qubit_hamiltonian: QubitOperator,
    state_preparation: Circuit,
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
) -> tuple[Circuit, int, list[int]]:
    """Build a measurement-free QPE circuit plus its register layout.

    This is the composition an external caller performs before amplifying:
    amplitude amplification itself only sees the resulting circuit.

    Returns:
        The coherent preparation, the total qubit count, and the indices of the
        block-encoding signal ancillas within the trailing target register.

    """
    unitary = unitary or AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True)
    builder = create("qpe_circuit_builder", "qdk_standard")
    builder.settings().update("num_bits", num_bits)
    builder.settings().update("controlled_circuit_mapper", AlgorithmRef("controlled_circuit_mapper", mapper))
    builder.settings().update("unitary_builder", unitary)
    builder.settings().update("measurement", "none")
    preparation = builder.run(state_preparation=state_preparation, qubit_hamiltonian=qubit_hamiltonian)[0]

    unitary_algorithm = create(unitary.algorithm_type, unitary.algorithm_name, **unitary.settings)
    num_system_qubits = qubit_hamiltonian.num_qubits
    num_ancilla_qubits = unitary_algorithm.run(qubit_hamiltonian).get_num_qubits() - num_system_qubits
    num_qubits = num_bits + num_system_qubits + num_ancilla_qubits
    signal_ancilla_indices = list(range(num_system_qubits, num_system_qubits + num_ancilla_qubits))
    return preparation, num_qubits, signal_ancilla_indices


def _amplified_qpe_circuit(
    qsharp_module,
    qubit_hamiltonian: QubitOperator,
    state_preparation: Circuit,
    accepted_indices: list[int],
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
    **settings,
) -> Circuit:
    """Compose a coherent QPE preparation with amplitude amplification."""
    preparation, num_qubits, signal_ancilla_indices = _qpe_preparation(
        qubit_hamiltonian,
        state_preparation,
        num_bits=num_bits,
        mapper=mapper,
        unitary=unitary,
    )
    make_marker = getattr(qsharp_module.code, _NAMESPACE).MakeAcceptedPhaseMarkerOp
    marker = make_marker(num_bits, signal_ancilla_indices, accepted_indices)
    # The executor reverses the Q# results, so emitting the ancillas reversed and
    # ahead of the phase indices makes the key read phase register MSB first.
    ancilla_indices = list(range(num_qubits - len(signal_ancilla_indices), num_qubits))
    measured_indices = list(reversed(ancilla_indices)) + list(range(num_bits))

    algorithm = create("amplitude_amplification")
    for key, value in settings.items():
        algorithm.settings().update(key, value)
    marking_oracle = Circuit(
        qsharp_factory=QsharpFactoryData(
            program=make_marker,
            parameter={
                "numPhaseQubits": num_bits,
                "signalAncillaIndices": signal_ancilla_indices,
                "accepted": accepted_indices,
            },
        ),
        qsharp_op=marker,
    )
    return algorithm.run(preparation, marking_oracle, num_qubits=num_qubits, measured_indices=measured_indices)


def _dominant_accepted_phase(circuit: Circuit, num_bits: int, accepted_indices: list[int], shots: int = 400) -> str:
    """Execute a circuit and return the most common bitstring from the good subspace.

    Acceptance mirrors the marking oracle and is decided by the caller, not by
    amplitude amplification.

    """
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    counts: dict[str, int] = {}
    for bitstring, count in executor.run(circuit, shots=shots).bitstring_counts.items():
        phase_bits, ancilla_bits = bitstring[:num_bits], bitstring[num_bits:]
        if any(bit != "0" for bit in ancilla_bits) or int(phase_bits, 2) not in accepted_indices:
            continue
        counts[phase_bits] = counts.get(phase_bits, 0) + count
    assert counts, f"No shot landed in the accepted window {accepted_indices}."
    return max(counts, key=counts.get)


def test_amplitude_amplification_is_registered():
    assert available("amplitude_amplification") == ["qdk_amplitude_amplification"]
    default = create("amplitude_amplification")
    assert default.name() == "qdk_amplitude_amplification"
    assert default.type_name() == "amplitude_amplification"
    assert isinstance(default, AmplitudeAmplification)


def test_rounds_setting_defaults_to_one():
    algorithm = create("amplitude_amplification")
    assert algorithm.settings().get("rounds") == 1


def test_iterative_circuit_builder_has_no_measurement_setting():
    builder = create("qpe_circuit_builder", "qdk_iterative")
    with pytest.raises(Exception, match="measurement"):
        builder.settings().update("measurement", "none")


def test_measurement_plan_covers_every_policy():
    builder = create("qpe_circuit_builder", "qdk_standard")
    builder.settings().update("num_bits", 3)

    # The default measures only the phase register, in register order, so the
    # executor's reversal makes the key read most-significant bit first.
    assert builder._measurement_plan(3, 2) == ([0, 1, 2], ["Z", "Z", "Z"])

    # Zero measurement is what amplitude amplification asks for.
    builder.settings().update("measurement", "none")
    assert builder._measurement_plan(3, 2) == ([], [])

    # The eigenvector policy trails the phase bits with the system register,
    # reversed so that the key reads system qubit 0 first.
    builder.settings().update("measurement", "eigenvector")
    builder.settings().update("measurement_basis", "X")
    assert builder._measurement_plan(3, 2) == ([4, 3, 0, 1, 2], ["X", "X", "Z", "Z", "Z"])

    # A per-qubit basis string is accepted, and stays aligned with its indices.
    builder.settings().update("measurement_basis", "XY")
    indices, bases = builder._measurement_plan(3, 2)
    assert indices == [4, 3, 0, 1, 2]
    assert bases == ["Y", "X", "Z", "Z", "Z"]

    builder.settings().update("measurement_basis", "XYZ")
    with pytest.raises(ValueError, match="one letter per system"):
        builder._measurement_plan(3, 2)

    builder.settings().update("measurement_basis", "AB")
    with pytest.raises(ValueError, match="I, X, Y or Z"):
        builder._measurement_plan(3, 2)

    builder.settings().update("measurement", "bogus")
    with pytest.raises(ValueError, match="measurement must be one of"):
        builder._measurement_plan(3, 2)


@pytest.mark.skipif(not _HAS_QSHARP, reason="qdk.qsharp is not installed")
def test_measurement_setting_drives_the_standard_qpe_readout():
    """Every policy shares one circuit body; only the readout changes.

    ``|00>`` is an exact eigenvector of ``(pi/4)(ZI + IZ)`` with eigenvalue
    ``pi/2``, so QPE leaves the system register untouched and the phase register
    lands deterministically in a single bin. That makes all three readouts
    predictable from each other.
    """
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    hamiltonian = _diagonal_hamiltonian()
    num_bits, num_system = 3, 2

    def build(**settings) -> Circuit:
        builder = create("qpe_circuit_builder", "qdk_standard")
        builder.settings().update("num_bits", num_bits)
        for key, value in settings.items():
            builder.settings().update(key, value)
        return builder.run(state_preparation=_guiding_state(1.0, 0), qubit_hamiltonian=hamiltonian)[0]

    # The body is always the adjointable QPE operation, whatever the readout.
    phase_circuit = build()
    eigenvector_circuit = build(measurement="eigenvector", measurement_basis="Z")
    coherent_circuit = build(measurement="none")
    for circuit in (phase_circuit, eigenvector_circuit, coherent_circuit):
        assert circuit._qsharp_op is not None

    phase_counts = executor.run(phase_circuit, shots=64).bitstring_counts
    assert all(len(key) == num_bits for key in phase_counts)
    # |00> is an exact eigenvector, so the phase register is deterministic.
    assert len(phase_counts) == 1
    phase_key = next(iter(phase_counts))

    eigenvector_counts = executor.run(eigenvector_circuit, shots=64).bitstring_counts
    assert all(len(key) == num_bits + num_system for key in eigenvector_counts)
    # The phase bits are unchanged and the system register is still |00>.
    assert set(eigenvector_counts) == {phase_key + "0" * num_system}

    # Zero measurement is what amplitude amplification consumes: the circuit is
    # still executable, it simply reports no bits.
    assert set(executor.run(coherent_circuit, shots=8).bitstring_counts) == {""}

    # The basis really reaches Q#: |00> measured in X is uniform over the system
    # bits while the phase bits stay put.
    x_counts = executor.run(build(measurement="eigenvector", measurement_basis="X"), shots=256).bitstring_counts
    assert {key[:num_bits] for key in x_counts} == {phase_key}
    assert {key[num_bits:] for key in x_counts} == {"00", "01", "10", "11"}


def test_amplitude_amplification_requires_a_coherent_circuit():
    # Amplitude amplification is decoupled from any particular circuit builder:
    # it only requires a circuit carrying an adjointable Q# operation.
    algorithm = create("amplitude_amplification")
    algorithm.settings().update("rounds", 1)
    with pytest.raises(TypeError, match="adjointable"):
        algorithm.run(
            Circuit(qasm="OPENQASM 3.0;"),
            _guiding_state(1.0, 0),
            num_qubits=4,
        )


def test_amplitude_amplification_requires_a_marking_oracle_operation():
    algorithm = create("amplitude_amplification")
    algorithm.settings().update("rounds", 1)
    with pytest.raises(TypeError, match="marking oracle"):
        algorithm.run(_guiding_state(1.0, 0), Circuit(qasm="OPENQASM 3.0;"), num_qubits=1)


def test_register_bounds_are_validated():
    algorithm = create("amplitude_amplification")
    algorithm.settings().update("rounds", 0)
    preparation = _guiding_state(1.0, 0)
    with pytest.raises(ValueError, match="num_qubits"):
        algorithm.run(preparation, preparation, num_qubits=0)
    with pytest.raises(ValueError, match="measured_indices"):
        algorithm.run(preparation, preparation, num_qubits=2, measured_indices=[2])


@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_amplification_does_not_change_the_energy(qsharp_module, rounds: int):
    accepted = [0]
    circuit = _amplified_qpe_circuit(
        qsharp_module,
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 0),
        accepted,
        rounds=rounds,
    )
    # Amplification changes how often the window is accepted, never what it accepts.
    assert _dominant_accepted_phase(circuit, 4, accepted) == "0000"


def test_energy_window_selects_the_right_phase_bin(qsharp_module):
    # The |11> eigenvector has energy -lambda, which the qubitization walk maps
    # to the phase bin 1/2 -- index 8 of 16, big-endian 0b1000.
    accepted = [8]
    circuit = _amplified_qpe_circuit(
        qsharp_module,
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        accepted,
        rounds=2,
    )
    assert _dominant_accepted_phase(circuit, 4, accepted) == "1000"


def test_trotter_encoding_is_supported(qsharp_module):
    # The pauli-sequence mapper has no block-encoding ancillas, so the accepted
    # window is defined purely on the phase register.
    accepted = [4]
    circuit = _amplified_qpe_circuit(
        qsharp_module,
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        accepted,
        mapper="pauli_sequence",
        unitary=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
        rounds=1,
    )
    # e^{-iHt} with t = 1 maps the eigenvalue -pi/2 to the phase 1/4, bin 4 of 16.
    assert _dominant_accepted_phase(circuit, 4, accepted, shots=200) == "0100"
