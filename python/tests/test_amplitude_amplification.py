"""Tests for amplitude amplification.

Three layers are exercised against each other:

* the round-scheduling closed forms on
  :class:`~qdk_chemistry.algorithms.amplitude_amplification.base.AmplitudeAmplification`
  are checked against an independent simulation of the two-dimensional invariant
  subspace, so the tests validate the mathematics rather than restating it;
* the Q# module ``QDKChemistry.Utils.AmplitudeAmplification`` is executed on the
  full-state simulator and compared against those same closed forms;
* the registry algorithm ``amplitude_amplification/qdk_amplified_qpe`` is run end
  to end on top of the QPE circuit builder.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import cmath
import importlib.util
import math

import numpy as np
import pytest

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.algorithms import registry as algorithm_registry
from qdk_chemistry.algorithms.amplitude_amplification.base import AmplitudeAmplification
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import (
    StandardQpeCircuitBuilder,
    split_coherent_qpe_bitstring,
)
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context

from .reference_tolerances import qpe_energy_tolerance

OVERLAPS = [1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]


def _simulate(overlap: float, mark_phases: list[float], state_phases: list[float]) -> float:
    """Return the acceptance probability from an explicit 2D subspace simulation.

    The state is represented in the ``(|G>, |B>)`` basis. Each iterate applies a
    phase to the good component and then the partial reflection about the
    prepared state, exactly as the Q# ``GeneralizedAmplitudeAmplificationStep``
    does.
    """
    angle = math.asin(math.sqrt(overlap))
    prepared = (math.sin(angle), math.cos(angle))
    good, bad = complex(prepared[0]), complex(prepared[1])

    for mark_phase, state_phase in zip(mark_phases, state_phases, strict=True):
        good *= cmath.exp(1j * mark_phase)
        projection = prepared[0] * good + prepared[1] * bad
        factor = (1.0 - cmath.exp(1j * state_phase)) * projection
        good -= factor * prepared[0]
        bad -= factor * prepared[1]

    return abs(good) ** 2


def _standard_phases(rounds: int) -> tuple[list[float], list[float]]:
    """Return the phase sequence of plain Grover amplification."""
    return [math.pi] * rounds, [math.pi] * rounds


def _chebyshev(degree: float, argument: float) -> float:
    """Return the Chebyshev polynomial of the first kind, valid outside [-1, 1]."""
    if argument >= 1.0:
        return math.cosh(degree * math.acosh(argument))
    if argument <= -1.0:
        magnitude = math.cosh(degree * math.acosh(-argument))
        return magnitude if int(degree) % 2 == 0 else -magnitude
    return math.cos(degree * math.acos(argument))


def _fixed_point_success_probability(overlap: float, rounds: int, tolerance: float) -> float:
    """Return 1 - delta^2 T_L(T_{1/L}(1/delta) sqrt(1 - a))^2, an independent stdlib reference."""
    queries = 2 * rounds + 1
    scale = math.cosh(math.acosh(1.0 / tolerance) / queries)
    return 1.0 - tolerance**2 * _chebyshev(queries, scale * math.sqrt(1.0 - overlap)) ** 2


#
# Round schedule
#


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("rounds", [0, 1, 2, 3, 7, 15])
def test_success_probability_matches_simulation(overlap: float, rounds: int):
    mark_phases, state_phases = _standard_phases(rounds)
    simulated = _simulate(overlap, mark_phases, state_phases)
    assert AmplitudeAmplification.success_probability(overlap, rounds) == pytest.approx(simulated, abs=1e-12)


@pytest.mark.parametrize("min_overlap", OVERLAPS)
def test_fixed_point_rounds_never_overshoot(min_overlap: float):
    """The whole point of the schedule: acceptance holds above the plateau for every larger overlap."""
    tolerance = 0.1
    rounds = AmplitudeAmplification.fixed_point_rounds(min_overlap, tolerance)
    for fraction in (1.0, 1.5, 2.0, 5.0, 20.0):
        overlap = min(0.999, min_overlap * fraction)
        achieved = _fixed_point_success_probability(overlap, rounds, tolerance)
        assert achieved >= 1.0 - tolerance**2 - 1e-9


@pytest.mark.parametrize("min_overlap", [1e-3, 0.01, 0.05, 0.2])
def test_fixed_point_rounds_is_the_smallest_sufficient_count(min_overlap: float):
    tolerance = 0.1
    rounds = AmplitudeAmplification.fixed_point_rounds(min_overlap, tolerance)
    assert _fixed_point_success_probability(min_overlap, rounds, tolerance) >= 1.0 - tolerance**2 - 1e-9
    # The Yoder-Low-Chuang query bound is log(2/delta)/sqrt(a_min).
    assert 2 * rounds + 1 >= math.log(2.0 / tolerance) / math.sqrt(min_overlap)


def test_plain_rounds_overshoot_where_fixed_point_does_not():
    """Contrast the two schedules: the sinusoid collapses, the plateau does not."""
    tolerance = 0.1
    min_overlap = 0.01
    plain = 7  # optimal for a = 0.01
    fixed_point = AmplitudeAmplification.fixed_point_rounds(min_overlap, tolerance)

    assert AmplitudeAmplification.success_probability(min_overlap, plain) > 0.99
    # Four times the assumed overlap rotates the plain schedule well past the maximum.
    assert AmplitudeAmplification.success_probability(0.04, plain) < 0.05
    assert _fixed_point_success_probability(0.04, fixed_point, tolerance) > 0.99


#
# Fixed-point schedule
#


@pytest.mark.parametrize("min_overlap", [0.01, 0.05, 0.1, 0.25])
@pytest.mark.parametrize("tolerance", [0.5, 0.2, 0.05])
def test_fixed_point_meets_its_tolerance_everywhere_above_threshold(min_overlap: float, tolerance: float):
    rounds = AmplitudeAmplification.fixed_point_rounds(min_overlap, tolerance)
    mark_phases, state_phases = AmplitudeAmplification.fixed_point_phases(rounds, tolerance)
    assert len(mark_phases) == rounds
    assert len(state_phases) == rounds

    for index in range(41):
        overlap = min_overlap + (1.0 - min_overlap) * index / 40.0
        probability = _simulate(overlap, mark_phases, state_phases)
        assert probability >= 1.0 - tolerance**2 - 1e-9


@pytest.mark.parametrize("min_overlap", [0.02, 0.1])
def test_fixed_point_removes_the_overshoot_cliff(min_overlap: float):
    tolerance = 0.1
    rounds = AmplitudeAmplification.fixed_point_rounds(min_overlap, tolerance)
    mark_phases, state_phases = AmplitudeAmplification.fixed_point_phases(rounds, tolerance)

    fixed_point_worst = min(
        _simulate(min_overlap + (1.0 - min_overlap) * index / 200.0, mark_phases, state_phases) for index in range(201)
    )
    plain_worst = min(
        AmplitudeAmplification.success_probability(min_overlap + (1.0 - min_overlap) * index / 200.0, rounds)
        for index in range(201)
    )
    assert fixed_point_worst >= 1.0 - tolerance**2 - 1e-9
    assert plain_worst < 1e-2


def test_fixed_point_phase_symmetry():
    rounds = 5
    mark_phases, state_phases = AmplitudeAmplification.fixed_point_phases(rounds, 0.1)
    for index in range(rounds):
        assert mark_phases[index] == pytest.approx(state_phases[rounds - 1 - index])


@pytest.mark.parametrize("rounds", [1, 2, 3, 5, 8, 12])
@pytest.mark.parametrize("tolerance", [0.5, 0.1, 0.01])
def test_fixed_point_phases_realize_the_chebyshev_closed_form(rounds: int, tolerance: float):
    mark_phases, state_phases = AmplitudeAmplification.fixed_point_phases(rounds, tolerance)
    for index in range(101):
        overlap = 0.001 + 0.998 * index / 100.0
        simulated = _simulate(overlap, mark_phases, state_phases)
        predicted = _fixed_point_success_probability(overlap, rounds, tolerance)
        assert simulated == pytest.approx(predicted, abs=1e-9)


@pytest.mark.parametrize("overlap", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_invalid_overlap_is_rejected(overlap: float):
    with pytest.raises(ValueError, match="overlap"):
        AmplitudeAmplification._rotation_angle(overlap)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="rounds"):
        AmplitudeAmplification.success_probability(0.1, -1)
    with pytest.raises(ValueError, match="min_overlap"):
        AmplitudeAmplification.fixed_point_rounds(0.0, 0.1)
    with pytest.raises(ValueError, match="tolerance"):
        AmplitudeAmplification.fixed_point_phases(3, 1.0)
    with pytest.raises(ValueError, match="rounds"):
        AmplitudeAmplification.fixed_point_phases(0, 0.1)
    with pytest.raises(ValueError, match="tolerance"):
        AmplitudeAmplification.fixed_point_rounds(0.1, 0.0)


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

    operation RunFixedPointAmplification(
        theta : Double,
        markPhases : Double[],
        statePhases : Double[],
    ) : Result {
        use register = Qubit[1];
        ApplyFixedPointAmplitudeAmplification(
            PrepareOverlap(theta, _),
            MarkFirstQubit,
            markPhases,
            statePhases,
            register,
        );
        return MResetZ(register[0]);
    }

    operation MarkPhaseIndex(numBits : Int, value : Int, accepted : Int[]) : Result {
        use register = Qubit[numBits];
        use flag = Qubit();
        ApplyXorInPlace(value, register);
        ApplyAcceptedPhaseMark(register, accepted, flag);
        let outcome = MResetZ(flag);
        ResetAll(register);
        return outcome;
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
        // The QPE phase register is little-endian: register[0] is the least
        // significant bit, which is what ApplyXorInPlace assumes as well.
        ApplyXorInPlace(phaseValue, register[0..numPhaseQubits - 1]);
        ApplyXorInPlace(ancillaValue, register[numPhaseQubits...]);
        ApplyQpeAcceptanceMark(numPhaseQubits, signalAncillaIndices, accepted, register, flag);
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
        let marker = MakeQpeAcceptanceMarkerOp(3, signalAncillaIndices, [2]);
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


@pytest.mark.parametrize(("rounds", "tolerance"), [(2, 0.3), (4, 0.2), (6, 0.1)])
@pytest.mark.parametrize("overlap", [0.05, 0.5, 0.9])
def test_fixed_point_amplification_matches_the_chebyshev_closed_form(
    qsharp_module, rounds: int, tolerance: float, overlap: float
):
    theta = AmplitudeAmplification._rotation_angle(overlap)
    mark_phases, state_phases = AmplitudeAmplification.fixed_point_phases(rounds, tolerance)
    expression = f"{_NAMESPACE}.RunFixedPointAmplification({theta}, {list(mark_phases)}, {list(state_phases)})".replace(
        "'", ""
    )
    observed = _acceptance_frequency(qsharp_module, expression)
    predicted = _fixed_point_success_probability(overlap, rounds, tolerance)
    assert observed == pytest.approx(predicted, abs=_TOLERANCE)


@pytest.mark.parametrize(
    "accepted",
    [[0, 1, 6, 7], [0, 1, 2], [5, 6, 7], [2, 3, 5], [4], list(range(8)), []],
)
def test_accepted_phase_mark_is_exact(qsharp_module, accepted: list[int]):
    num_bits = 3
    for value in range(1 << num_bits):
        expression = f"{_NAMESPACE}.MarkPhaseIndex({num_bits}, {value}, {accepted})"
        outcome = str(qsharp_module.run(expression, shots=1)[0])
        assert (outcome == "One") == (value in accepted)


def test_qpe_acceptance_requires_the_window_and_clean_signal_ancillas(qsharp_module):
    accepted = [0, 1, 6, 7]
    for phase_value in range(8):
        for ancilla_value in range(4):
            expression = f"{_NAMESPACE}.MarkQpeAcceptance(3, {phase_value}, 2, {ancilla_value}, [0, 1], {accepted})"
            outcome = str(qsharp_module.run(expression, shots=1)[0])
            assert (outcome == "One") == (phase_value in accepted and ancilla_value == 0)


def test_accepted_phase_interval_lengths_detect_wrapped_windows():
    utils = QSHARP_UTILS.AmplitudeAmplification
    assert tuple(utils.AcceptedPhaseIntervalLengths(3, [0, 1, 6, 7])) == (True, 2, 2)
    assert tuple(utils.AcceptedPhaseIntervalLengths(3, [0, 1, 2])) == (True, 3, 0)
    assert tuple(utils.AcceptedPhaseIntervalLengths(3, [5, 6, 7])) == (True, 0, 3)
    is_wrapped, _, _ = tuple(utils.AcceptedPhaseIntervalLengths(3, [2, 3, 5]))
    assert is_wrapped is False


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


def _qubitization_builder_ref(num_bits: int = 4) -> AlgorithmRef:
    """Return an AlgorithmRef for the standard QPE circuit builder with qubitization."""
    return AlgorithmRef(
        "qpe_circuit_builder",
        "qdk_standard",
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
    )


def _amplified_qpe(shots: int = 400, **settings) -> AmplitudeAmplification:
    """Return a configured amplified QPE algorithm using the QDK simulator."""
    algorithm = create("amplitude_amplification", "qdk_amplified_qpe")
    algorithm.settings().update("reflect_to_good_space", _qubitization_builder_ref())
    algorithm.settings().update("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"))
    algorithm.settings().update("shots", shots)
    for key, value in settings.items():
        algorithm.settings().update(key, value)
    return algorithm


def test_amplitude_amplification_is_registered():
    assert available("amplitude_amplification") == ["qdk_amplified_qpe"]
    default = create("amplitude_amplification")
    assert default.name() == "qdk_amplified_qpe"
    assert default.type_name() == "amplitude_amplification"
    assert isinstance(default, AmplitudeAmplification)


def test_resolve_rounds_uses_the_fixed_point_schedule():
    algorithm = create("amplitude_amplification", "qdk_amplified_qpe")
    algorithm.settings().update("min_overlap", 0.05)
    assert algorithm.resolve_rounds() == AmplitudeAmplification.fixed_point_rounds(0.05, 0.1)


def test_explicit_rounds_override_the_schedule():
    algorithm = create("amplitude_amplification", "qdk_amplified_qpe")
    algorithm.settings().update("min_overlap", 0.05)
    algorithm.settings().update("rounds", 1)
    assert algorithm.resolve_rounds() == 1


def test_deriving_rounds_requires_a_lower_bound():
    algorithm = create("amplitude_amplification", "qdk_amplified_qpe")
    with pytest.raises(ValueError, match="min_overlap"):
        algorithm.resolve_rounds()


def test_coherent_qpe_bit_order_round_trips():
    assert split_coherent_qpe_bitstring("101101", 4) == ("1011", "01")
    assert split_coherent_qpe_bitstring("1011", 4) == ("1011", "")
    with pytest.raises(ValueError, match="phase register"):
        split_coherent_qpe_bitstring("101", 4)


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


def test_reflect_to_good_space_requires_a_coherent_circuit():
    # Amplitude amplification is decoupled from any particular circuit builder:
    # it asks the nested algorithm for a measurement-free circuit and only
    # requires that the answer carries an adjointable Q# operation to reflect about.
    class MeasuredOnlyBuilder(StandardQpeCircuitBuilder):
        def name(self) -> str:
            return "test_measured_only"

        def _run_impl(self, state_preparation, qubit_hamiltonian):  # noqa: ARG002
            # Ignores the `measurement` setting, as a non-Q# backend would.
            return [Circuit(qasm="OPENQASM 3.0;")]

    algorithm_registry.register(lambda: MeasuredOnlyBuilder())
    try:
        algorithm = create("amplitude_amplification", "qdk_amplified_qpe")
        algorithm.settings().update(
            "reflect_to_good_space",
            AlgorithmRef("qpe_circuit_builder", "test_measured_only", num_bits=4),
        )
        algorithm.settings().update("accepted_phase_indices", [0])
        algorithm.settings().update("rounds", 1)
        with pytest.raises(TypeError, match="did not produce a coherent circuit"):
            algorithm.run(
                state_preparation=_guiding_state(1.0, 0),
                qubit_hamiltonian=_diagonal_hamiltonian(),
            )
    finally:
        algorithm_registry.unregister("qpe_circuit_builder", "test_measured_only")


@pytest.mark.parametrize("rounds", [0, 1, 2])
def test_amplification_does_not_change_the_energy(rounds: int):
    # A guiding state with only 9% overlap on the |00> eigenvector of energy pi/2.
    algorithm = _amplified_qpe(
        rounds=rounds,
        accepted_phase_indices=[0],
    )
    result = algorithm.run(
        state_preparation=_guiding_state(0.3, 0),
        qubit_hamiltonian=_diagonal_hamiltonian(),
    )

    # Amplification changes how often the window is accepted, never what it accepts.
    assert result.bitstring_msb_first == "0000"
    assert result.raw_energy == pytest.approx(math.pi / 2.0, abs=qpe_energy_tolerance)


def test_energy_window_selects_the_right_phase_bin():
    # The |11> eigenvector has energy -lambda, which the qubitization walk maps
    # to the phase bin 1/2 -- index 8 of 16, big-endian 0b1000.
    lambda_norm = math.pi / 2.0
    algorithm = _amplified_qpe(
        rounds=2,
        min_energy=-lambda_norm - 0.05,
        max_energy=-lambda_norm + 0.05,
    )
    result = algorithm.run(
        state_preparation=_guiding_state(0.3, 3),
        qubit_hamiltonian=_diagonal_hamiltonian(),
    )

    assert result.bitstring_msb_first == "1000"
    assert result.raw_energy == pytest.approx(-lambda_norm, abs=qpe_energy_tolerance)


def test_fixed_point_schedule_runs_end_to_end():
    algorithm = _amplified_qpe(
        min_overlap=0.09,
        tolerance=0.3,
        accepted_phase_indices=[0],
    )
    result = algorithm.run(
        state_preparation=_guiding_state(0.3, 0),
        qubit_hamiltonian=_diagonal_hamiltonian(),
    )
    assert result.raw_energy == pytest.approx(math.pi / 2.0, abs=qpe_energy_tolerance)


def test_trotter_encoding_is_supported():
    # The pauli-sequence mapper has no block-encoding ancillas, so the accepted
    # window is defined purely on the phase register.
    algorithm = create("amplitude_amplification", "qdk_amplified_qpe")
    algorithm.settings().update(
        "reflect_to_good_space",
        AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_standard",
            num_bits=4,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
        ),
    )
    algorithm.settings().update("circuit_executor", AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"))
    algorithm.settings().update("shots", 200)
    algorithm.settings().update("rounds", 1)
    algorithm.settings().update("accepted_phase_indices", [4])

    result = algorithm.run(
        state_preparation=_guiding_state(0.3, 3),
        qubit_hamiltonian=_diagonal_hamiltonian(),
    )
    # e^{-iHt} with t = 1 maps the eigenvalue -pi/2 to the phase 1/4, bin 4 of 16.
    assert result.bitstring_msb_first == "0100"
    assert result.raw_energy == pytest.approx(-math.pi / 2.0, abs=qpe_energy_tolerance)


def test_an_empty_window_is_rejected():
    algorithm = _amplified_qpe(min_energy=100.0, max_energy=200.0)
    with pytest.raises(ValueError, match="no phase bin"):
        algorithm.run(
            state_preparation=_guiding_state(1.0, 0),
            qubit_hamiltonian=_diagonal_hamiltonian(),
        )


def test_an_undefined_window_is_rejected():
    algorithm = _amplified_qpe()
    with pytest.raises(ValueError, match="accepted_phase_indices"):
        algorithm.run(
            state_preparation=_guiding_state(1.0, 0),
            qubit_hamiltonian=_diagonal_hamiltonian(),
        )


def test_out_of_range_accepted_indices_are_rejected():
    algorithm = _amplified_qpe(accepted_phase_indices=[16])
    with pytest.raises(ValueError, match="accepted_phase_indices"):
        algorithm.run(
            state_preparation=_guiding_state(1.0, 0),
            qubit_hamiltonian=_diagonal_hamiltonian(),
        )
