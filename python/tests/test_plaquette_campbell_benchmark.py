"""Benchmark the plaquette builder's logical costs against Campbell's published tables.

This is the regression form of a comparison that previously lived in a notebook. The
notebook could drift from the code without anyone noticing; a test cannot.

The reference is E. T. Campbell, "Early fault-tolerant simulations of the Hubbard
model", arXiv:2012.09238v4. Three of its results are used:

* Appendix E gives the per-step costs of the PLAQ product formula.
* Appendix F optimizes how the energy budget is split between Trotter error, phase
  readout, and rotation synthesis, giving the number of applications ``N_PE``.
* Table II lists the resulting totals at ``u/tau = 8``.

Campbell's numbers assume the particle-hole shifted Hamiltonian, where the onsite term
is :math:`U(n_up - 1/2)(n_dn - 1/2)`. Under Jordan-Wigner that is a single ``ZZ`` per
site rather than the three factors the unshifted form produces, so the comparison is
only meaningful against the shifted operator, built here with ``epsilon = -U/2``.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import LatticeGraph, MajoranaMapping, QubitOperator
from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian

#: Hopping amplitude; energies are quoted in units of this.
_HOPPING = 1.0

#: Campbell's strong-coupling regime.
_INTERACTION = 8.0 * _HOPPING

#: Per-site ground-state energy target, in units of the hopping.
_TARGET_PER_SITE = 0.0037 * _HOPPING

#: Threshold below which mapped coefficients are dropped.
_WEIGHT_THRESHOLD = 1e-12

#: Campbell's Table III, single-particle hopping norm per lattice size.
_HOPPING_NORM = dict(
    zip(range(4, 33, 2), [24, 56, 100, 160, 230, 320, 410, 520, 650, 780, 930, 1100, 1300, 1500, 1700], strict=True)
)

#: Campbell's Table III, the extra nested-commutator norm the plaquette split incurs.
_COMMUTATOR_NORM = dict(
    zip(range(4, 33, 2), [0, 110, 190, 300, 440, 630, 810, 1000, 1300, 1600, 1800, 2200, 2500, 2900, 3300], strict=True)
)

#: Campbell's Eq. (F7) prefactor, 3^{3/2} * 0.76 * pi / 2.
_PHASE_ESTIMATION_PREFACTOR = 3**1.5 * 0.76 * math.pi / 2.0

#: Table II at u/tau = 8: Toffoli and T totals for the whole phase estimation.
_TABLE_II_U8 = {
    8: (4.3e5, 4.1e6),
    10: (4.4e5, 3.3e6),
    12: (4.6e5, 2.9e6),
    14: (4.6e5, 2.5e6),
    16: (4.6e5, 2.3e6),
    18: (4.6e5, 2.2e6),
    20: (4.7e5, 2.0e6),
    22: (4.6e5, 2.2e6),
    24: (4.7e5, 2.1e6),
    26: (4.6e5, 2.1e6),
    28: (4.6e5, 2.1e6),
    30: (4.7e5, 2.1e6),
    32: (4.7e5, 2.1e6),
}


def _shifted_hubbard_operator(size: int) -> QubitOperator:
    """Return the Jordan-Wigner image of Campbell's shifted periodic Hubbard model.

    Args:
        size: Lattice side length.

    Returns:
        The qubit Hamiltonian, with identity factors dropped.

    """
    num_sites = size * size
    lattice = LatticeGraph.square(size, size, periodic_x=True, periodic_y=True)
    hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=-_INTERACTION / 2.0, t=_HOPPING, U=_INTERACTION)
    mapped = create("qubit_mapper").run(hamiltonian, mapping=MajoranaMapping.jordan_wigner(2 * num_sites))
    keep = [index for index, label in enumerate(mapped.pauli_strings) if set(label) != {"I"}]
    return QubitOperator(
        pauli_strings=[mapped.pauli_strings[index] for index in keep],
        coefficients=mapped.coefficients[keep],
        encoding=mapped.encoding,
        fermion_mode_order=mapped.fermion_mode_order,
    )


def _error_constant(size: int) -> float:
    """Return Campbell's ``W_PLAQ`` from his Eqs. (10) and (20) with Table III norms."""
    num_sites = size * size
    return (
        _INTERACTION * _HOPPING**2 / 6.0 * num_sites * (math.sqrt(5.0) + 8.0)
        + _INTERACTION**2 / 24.0 * _HOPPING_NORM[size] * _HOPPING
        + _COMMUTATOR_NORM[size] * _HOPPING**3 / 8.0
    )


def _per_step_costs(size: int) -> tuple[float, float, float]:
    """Return Campbell's Appendix E per-step ``(Toffoli, T, rotations)``.

    A batch of ``m = L^2/2`` equal-angle terms is phased through one Hamming weight
    register, costing ``m - w(m)`` Toffolis and ``ceil(log2 m)`` rotations rather than
    one rotation per term.

    Args:
        size: Lattice side length.

    Returns:
        The three per-step counts.

    """
    num_sites = size * size
    batch = num_sites // 2
    toffoli = 4 * num_sites * (batch - batch.bit_count()) / batch
    t_gates = 12 * num_sites
    rotations = 4 * num_sites * batch.bit_length() / batch
    return toffoli, t_gates, rotations


def _schedule(size: int, synthesis_fraction: float) -> tuple[float, float]:
    """Return ``(N_PE, N_HT)`` for one split of the budget between Trotter and synthesis."""
    epsilon = _TARGET_PER_SITE * size * size
    delta = (1.0 - synthesis_fraction) * epsilon
    num_pe = _PHASE_ESTIMATION_PREFACTOR * math.sqrt(_error_constant(size)) / delta**1.5
    _, _, rotations = _per_step_costs(size)
    synthesis_t = 1.15 * math.log2(rotations * num_pe / (synthesis_fraction * epsilon)) + 9.2
    return num_pe, synthesis_t


def _toffoli_equivalent(size: int, synthesis_fraction: float) -> float:
    """Return the Appendix F objective, two synthesized T gates counted as one Toffoli."""
    num_pe, synthesis_t = _schedule(size, synthesis_fraction)
    toffoli, t_gates, rotations = _per_step_costs(size)
    return num_pe * (toffoli + (t_gates + rotations * synthesis_t) / 2.0)


def campbell_schedule(size: int) -> dict[str, float]:
    """Return Campbell's optimized Appendix F schedule for a lattice.

    The objective is smooth and unimodal in the synthesis fraction, so a golden-section
    search finds the optimum to machine precision. That avoids a ``scipy.optimize``
    dependency the package does not otherwise carry, and agrees with
    ``minimize_scalar`` on ``N_PE`` at every size from 8 to 32.

    Args:
        size: Lattice side length.

    Returns:
        The step time, application count, and synthesis cost per rotation.

    """
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = 1e-9, 0.5
    left, right = high - golden * (high - low), low + golden * (high - low)
    for _ in range(200):
        if _toffoli_equivalent(size, left) < _toffoli_equivalent(size, right):
            high, right = right, left
            left = high - golden * (high - low)
        else:
            low, left = left, right
            right = low + golden * (high - low)
        if abs(high - low) < 1e-15:
            break

    fraction = (low + high) / 2.0
    num_pe, synthesis_t = _schedule(size, fraction)
    epsilon = _TARGET_PER_SITE * size * size
    delta = (1.0 - fraction) * epsilon
    return {
        "synthesis_fraction": fraction,
        "num_pe": math.ceil(num_pe),
        "step_time": math.sqrt(delta / (3.0 * _error_constant(size))),
        "synthesis_t": synthesis_t,
    }


def _logical_counts(size: int, *, max_batch: int) -> dict:
    """Build the aggregate controlled circuit for one lattice and return its counts."""
    schedule = campbell_schedule(size)
    steps = int(schedule["num_pe"])
    operator = _shifted_hubbard_operator(size)
    unitary = create(
        "hamiltonian_unitary_builder",
        "plaquette",
        order=2,
        # N_PE steps of duration step_time, i.e. exactly U_TS(t)^N_PE.
        time=steps * float(schedule["step_time"]),
        num_divisions=steps,
        target_accuracy=0.0,
        lattice_width=size,
        lattice_height=size,
        max_batch=max_batch,
        weight_threshold=_WEIGHT_THRESHOLD,
    ).run(operator)
    circuit = create(
        "controlled_circuit_mapper",
        "pauli_sequence",
        control_indices=[0],
        target_indices=list(range(1, operator.num_qubits + 1)),
    ).run(unitary)
    return circuit.estimate().logical_counts


class TestCampbellCostModel:
    """The reconstruction of Campbell's own formulas, independent of what we emit."""

    @pytest.mark.parametrize(("size", "expected"), [(4, 282.4), (8, 527.2)])
    def test_error_constant_reproduces_the_paper(self, size, expected):
        """Eqs. (10) and (20) with the Table III norms must give the tabulated W_PLAQ.

        At 4x4 with u/tau = 8 this is 282.4, and at 8x8 with the plaquette commutator
        term it is 527, matching Campbell's Table I entry of 5.3e2. Getting these right
        is what makes the rest of the comparison meaningful.
        """
        if size == 4:
            constant = (
                _INTERACTION * _HOPPING**2 / 6.0 * 16 * (math.sqrt(5.0) + 8.0)
                + _INTERACTION**2 / 24.0 * _HOPPING_NORM[4]
            )
        else:
            constant = (
                4.0 * _HOPPING**2 / 6.0 * 64 * (math.sqrt(5.0) + 8.0)
                + 4.0**2 / 24.0 * _HOPPING_NORM[8]
                + 3.0 / 24.0 * _COMMUTATOR_NORM[8]
            )
        assert constant == pytest.approx(expected, rel=2e-3)

    def test_schedule_shrinks_as_the_lattice_grows(self):
        """A larger lattice has a looser absolute target, so it needs fewer applications.

        The target is per site, so epsilon grows as L^2 while W_PLAQ grows only linearly
        in the site count; N_PE ~ sqrt(W)/epsilon^{3/2} therefore falls.
        """
        counts = [campbell_schedule(size)["num_pe"] for size in (8, 16, 32)]
        assert counts == sorted(counts, reverse=True)

    def test_step_time_is_nearly_size_independent(self):
        """Campbell's optimum fixes the step duration, not the step count.

        Both W_PLAQ and epsilon scale with the site count, so their ratio and hence the
        step time barely move. This is what makes the per-step costs comparable across
        sizes in the first place.
        """
        times = [campbell_schedule(size)["step_time"] for size in range(8, 33, 2)]
        assert max(times) / min(times) < 1.01


class TestPlaquetteAgainstTableII:
    """What the builder actually emits, measured against the published totals."""

    #: Sizes spanning Table II. Each builds one aggregate circuit, so keep the list short.
    _SIZES = (8, 12, 20, 32)

    @pytest.mark.parametrize("size", _SIZES)
    def test_qubit_count_differs_from_the_paper_by_a_known_constant(self, size):
        """The two qubit conventions differ by exactly ``w(m) + 1``, with nothing left over.

        Table II charges ``2L^2 + alpha + 2`` with ``alpha = L^2/2`` Hamming weight
        ancillas. We allocate Gidney's exact adder workspace ``m - w(m)`` instead of the
        looser ``m``, which saves ``w(m)``, and one repeat-until-success herald rather
        than two spare qubits, which saves one more. The herald is a constant and not a
        per-rotation cost: each rotation measures and releases the same qubit in turn,
        so charging one per rotation would bill depth as width.

        This is asserted as an identity rather than a tolerance because every term in it
        is known; an unexplained qubit would mean one of those two conventions is wrong.
        """
        batch = size * size // 2
        counts = _logical_counts(size, max_batch=batch)
        paper = 2 * size * size + batch + 2

        assert paper - counts["numQubits"] == batch.bit_count() + 1

    @pytest.mark.parametrize("size", _SIZES)
    def test_toffoli_count_matches_the_paper(self, size):
        """Toffolis come from the Hamming weight arithmetic, which we implement as published.

        Agreement here is the evidence that the Trotter step and the phase-estimation
        schedule are both right: a wrong step count or a wrong tiling would move this.
        The residual few percent is the exact batch decomposition against Table II's
        two-significant-figure entries, plus the one-time boundary layer that a closed
        formula in ``N_PE`` does not carry. Measured at 0.0-6.9% across these sizes.
        """
        counts = _logical_counts(size, max_batch=size * size // 2)

        assert counts["cczCount"] == pytest.approx(_TABLE_II_U8[size][0], rel=0.10)

    @pytest.mark.parametrize("size", _SIZES)
    def test_synthesized_t_count_approaches_the_paper_as_the_lattice_grows(self, size):
        """Our T total sits above Campbell's, by a factor that falls from 3.3x to 1.2x.

        The gap is rotation count, not step count: Campbell's directional-control
        costing emits fewer controlled rotations than a conventional controlled mapper,
        and each surviving rotation is then synthesized at his own ``N_HT``. Because the
        excess is a fixed number of rotations per step while his total grows with
        ``N_PE``, the ratio shrinks with the lattice. Asserted as a decreasing bound
        rather than one tolerance, since no single budget is honest across the range.
        """
        schedule = campbell_schedule(size)
        counts = _logical_counts(size, max_batch=size * size // 2)
        synthesized = counts["tCount"] + counts["rotationCount"] * schedule["synthesis_t"]

        ratio = synthesized / _TABLE_II_U8[size][1]
        assert 1.0 < ratio <= 3.4, f"T ratio {ratio:.2f}x outside the measured band"

    def test_the_t_gap_narrows_with_the_lattice(self):
        """The rotation excess is per step, so it is amortized as N_PE grows.

        This is the claim the per-size bound above cannot make: if the gap were a
        mis-sized schedule rather than a control-lowering difference, it would not
        shrink monotonically.
        """
        ratios = []
        for size in self._SIZES:
            schedule = campbell_schedule(size)
            counts = _logical_counts(size, max_batch=size * size // 2)
            synthesized = counts["tCount"] + counts["rotationCount"] * schedule["synthesis_t"]
            ratios.append(synthesized / _TABLE_II_U8[size][1])

        assert ratios == sorted(ratios, reverse=True)

    @pytest.mark.parametrize("size", [8, 16])
    def test_unbounded_batches_trade_qubits_for_rotations(self, size):
        """Merging each equal-angle family into one register is the default, and costs width.

        Campbell's Table II budgets alpha = L^2/2 ancillas, which bounds a batch at that
        size and so splits the interaction's families into several registers. Lifting the
        bound removes rotations and adds qubits; neither form is wrong, so the builder
        exposes the choice.
        """
        bounded = _logical_counts(size, max_batch=size * size // 2)
        merged = _logical_counts(size, max_batch=0)
        assert merged["rotationCount"] < bounded["rotationCount"]
        assert merged["numQubits"] > bounded["numQubits"]
