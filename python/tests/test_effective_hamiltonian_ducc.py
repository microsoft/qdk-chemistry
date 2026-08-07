"""Tests for the native DUCC ``effective_hamiltonian_constructor`` (``ducc_level`` 0-2).

At level 0 there is no BCH dressing, so the effective active-space Hamiltonian
must reproduce the bare active-space (CASCI) problem exactly: CASCI on the
original full Hamiltonian equals FCI on the DUCC output Hamiltonian.

Levels 1-2 add the Baker-Campbell-Hausdorff dressing, evaluated by the generated
BTAS equations. Three properties are checked: (a) the dressed active-space energy
reproduces independent reference energies for closed-shell active spaces with a
frozen core, which validates the dressing absolutely; (b) when the active space
is the whole orbital set there is no external space, so the dressing collapses to
the bare Hamiltonian and the level-0 identity still holds at every level; (c) the
dressed output is convention-consistent -- the restricted and unrestricted
dressings of a closed-shell system agree, and independent CI solvers agree on the
restricted output.

References are computed on the standard-convention active Hamiltonian from
``hamiltonian_constructor`` (full 8-fold chemist integrals): the native MACIS
CAS solver for restricted (RHF) references and PySCF's ``direct_uhf`` FCI for
unrestricted (UHF) references (MACIS does not support unrestricted Hamiltonians).

The DUCC output stores its same-spin two-body block as the half-antisymmetrized
representative, which a conventional solver such as ``direct_uhf`` misreads (it
assumes the full chemist integral). A level-0 restricted output retains full
8-fold two-body symmetry and is diagonalized with MACIS (which re-antisymmetrizes
internally). A dressed (level > 0) output is spin-blocked -- its reduced 4-fold
two-body symmetry cannot be conveyed by a single restricted block -- so it is
diagonalized through the native qubit mapper (Jordan-Wigner ``to_matrix()``
lowest eigenvalue), which reads each spin channel independently and reproduces
the reference exactly for both restricted and unrestricted references. Separate
tests confirm the restricted and unrestricted outputs map to the same qubit
ground state.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    AmplitudeContainer,
    AmplitudeType,
    Ansatz,
    CanonicalFourCenterHamiltonianContainer,
    Configuration,
    Hamiltonian,
    MajoranaMapping,
    Orbitals,
    StateVectorContainer,
    Structure,
    Wavefunction,
)
from qdk_chemistry.data._spin_channels import spin_channel_matrix, spin_channel_vector
from qdk_chemistry.data.symmetry import axes, spin_index_set

from .reference_tolerances import ci_energy_tolerance

try:
    from pyscf.fci import direct_uhf

    from qdk_chemistry.plugins.pyscf.coupled_cluster import PyscfCoupledClusterCalculator

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")

LIH = "2\nLiH\nLi 0 0 0\nH 0 0 1.595\n"
H2O = "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n"
OH = "2\nOH\nO 0 0 0\nH 0 0 0.97\n"


def _scf(xyz, multiplicity, unrestricted):
    """Run the SCF solver and return the reference wavefunction."""
    solver = create("scf_solver")
    if unrestricted:
        solver.settings().set("scf_type", "unrestricted")
    _, wfn = solver.run(Structure.from_xyz(xyz), charge=0, spin_multiplicity=multiplicity, basis_or_guess="sto-3g")
    return wfn


def _hf_determinant(nmo, nocc_a, nocc_b, orbitals):
    """Single Hartree-Fock determinant wavefunction for the given occupation."""
    n_doubly = min(nocc_a, nocc_b)
    det = "2" * n_doubly + "u" * (nocc_a - n_doubly) + "d" * (nocc_b - n_doubly) + "0" * (nmo - max(nocc_a, nocc_b))
    container = StateVectorContainer(np.array([1.0]), [Configuration.from_spin_half_string(det)], orbitals)
    return Wavefunction(container)


def _full_ccsd(full_ham, hf_wfn):
    """Full-space CCSD wavefunction (full orbitals + full-space amplitudes)."""
    cc = PyscfCoupledClusterCalculator()
    cc.settings().set("store_amplitudes", True)
    _, cc_wfn, _ = cc.run(Ansatz(full_ham, hf_wfn))
    return cc_wfn


def _active_orbitals(base, nmo, active, inactive, unrestricted):
    """Orbitals carrying an active-space designation over the same MO basis."""
    equivalent = not unrestricted
    active_idx = spin_index_set(nmo, active, active, equivalent=equivalent)
    inactive_idx = spin_index_set(nmo, inactive, inactive, equivalent=equivalent)
    if unrestricted:
        coefficients, energies = base.coefficients(), base.energies()
        return Orbitals(
            coefficients_alpha=spin_channel_matrix(coefficients, axes.alpha()),
            coefficients_beta=spin_channel_matrix(coefficients, axes.beta()),
            energies_alpha=spin_channel_vector(energies, axes.alpha()),
            energies_beta=spin_channel_vector(energies, axes.beta()),
            ao_overlap=np.array(base.get_overlap_matrix()),
            basis_set=base.get_basis_set(),
            active_indices=active_idx,
            inactive_indices=inactive_idx,
        )
    return Orbitals(
        coefficients=spin_channel_matrix(base.coefficients(), axes.alpha()),
        energies=spin_channel_vector(base.energies(), axes.alpha()),
        ao_overlap=np.array(base.get_overlap_matrix()),
        basis_set=base.get_basis_set(),
        active_indices=active_idx,
        inactive_indices=inactive_idx,
    )


def _ci_restricted(ham, nelec_a, nelec_b):
    """MACIS CAS (FCI) total energy for a restricted Hamiltonian."""
    energy, _ = create("multi_configuration_calculator", "macis_cas").run(ham, nelec_a, nelec_b)
    return energy


def _ci_unrestricted(ham, nelec_a, nelec_b):
    """PySCF ``direct_uhf`` FCI total energy for an unrestricted Hamiltonian."""
    h1a, h1b = (np.asarray(x) for x in ham.get_one_body_integrals())
    n = round(h1a.size**0.5)
    vaa, vab, vbb = (np.asarray(x) for x in ham.get_two_body_integrals())
    energy, _ = direct_uhf.kernel(
        (h1a.reshape(n, n), h1b.reshape(n, n)),
        (vaa.reshape((n,) * 4), vab.reshape((n,) * 4), vbb.reshape((n,) * 4)),
        n,
        (nelec_a, nelec_b),
        ecore=ham.get_core_energy(),
    )
    return float(energy)


def _qubit_ground_state(ham):
    """Ground-state energy of a DUCC output Hamiltonian via the native qubit mapper.

    Maps the effective Hamiltonian to qubits (native ``qubit_mapper``,
    Jordan-Wigner), forms the dense operator with ``to_matrix()``, and returns the
    global lowest eigenvalue plus the (mapper-excluded) core energy. The mapper
    re-antisymmetrizes the same-spin block, so it reads the DUCC half-representative
    two-body convention correctly for both restricted and unrestricted outputs,
    unlike ``direct_uhf``. The global Fock-space minimum is the neutral ground state
    for these systems, so no particle-number or Sz projection is required.
    """
    n_spatial = np.asarray(ham.get_one_body_integrals()[0]).shape[0]
    qubit_ham = create("qubit_mapper", "qdk").run(ham, MajoranaMapping.jordan_wigner(2 * n_spatial))
    matrix = np.asarray(qubit_ham.to_matrix())
    return float(np.linalg.eigvalsh(matrix)[0]) + ham.get_core_energy()


def _ducc_output(xyz, multiplicity, unrestricted, active, inactive, level, wfn_hf=None):
    """Build the DUCC output Hamiltonian, its active orbitals and active electron counts."""
    if wfn_hf is None:
        wfn_hf = _scf(xyz, multiplicity, unrestricted)
    orbitals = wfn_hf.get_orbitals()
    nmo = orbitals.get_num_molecular_orbitals()
    nocc_a, nocc_b = wfn_hf.get_total_num_electrons()
    full_ham = create("hamiltonian_constructor").run(orbitals)
    cc_wfn = _full_ccsd(full_ham, _hf_determinant(nmo, nocc_a, nocc_b, orbitals))
    active_orbitals = _active_orbitals(orbitals, nmo, active, inactive, unrestricted)
    builder = create("effective_hamiltonian_constructor", "ducc")
    builder.settings().set("ducc_level", level)
    out_ham = builder.run(cc_wfn, full_ham, active_orbitals.active_indices())
    n_active_a = sum(1 for p in active if p < nocc_a)
    n_active_b = sum(1 for p in active if p < nocc_b)
    return out_ham, active_orbitals, n_active_a, n_active_b


def test_complex_amplitudes_not_yet_implemented():
    """DUCC reports its real-only generated BTAS limitation explicitly."""
    wfn_hf = _scf(LIH, 1, False)
    orbitals = wfn_hf.get_orbitals()
    nmo = orbitals.get_num_molecular_orbitals()
    nocc_a, nocc_b = wfn_hf.get_total_num_electrons()
    full_ham = create("hamiltonian_constructor").run(orbitals)
    cc_wfn = _full_ccsd(full_ham, _hf_determinant(nmo, nocc_a, nocc_b, orbitals))
    amplitudes = cc_wfn.get_container()
    t1, _ = amplitudes.get_t1_amplitudes()
    t2, _, _ = amplitudes.get_t2_amplitudes()
    complex_wfn = Wavefunction(
        AmplitudeContainer(
            orbitals,
            amplitudes.get_wavefunction(),
            AmplitudeType.CoupledCluster,
            np.asarray(t1, dtype=np.complex128),
            np.asarray(t2, dtype=np.complex128),
        )
    )
    p_space = spin_index_set(nmo, list(range(nmo)), list(range(nmo)), equivalent=True)

    with pytest.raises(RuntimeError, match="ducc: complex amplitudes not yet implemented"):
        create("effective_hamiltonian_constructor", "ducc").run(complex_wfn, full_ham, p_space)


# Each case: label, geometry, spin multiplicity, unrestricted flag, and the
# active and inactive spatial-MO index lists defining the active space.
_CASES = [
    ("lih_full", LIH, 1, False, list(range(6)), []),
    ("lih_cas22", LIH, 1, False, [1, 2], [0]),
    ("lih_cas23", LIH, 1, False, [1, 2, 3], [0]),
    ("lih_sparse", LIH, 1, False, [0, 3, 5], [1]),
    ("h2o_cas65", H2O, 1, False, [2, 3, 4, 5, 6], [0, 1]),
    ("oh_full", OH, 2, True, list(range(6)), []),
    ("oh_cas_fc2", OH, 2, True, [2, 3, 4, 5], [0, 1]),
    ("oh_cas_fc1", OH, 2, True, [1, 2, 3, 4, 5], [0]),
]


@pytest.mark.parametrize(
    ("label", "xyz", "multiplicity", "unrestricted", "active", "inactive"),
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_casci_equals_fci_at_level0(label, xyz, multiplicity, unrestricted, active, inactive):
    """CASCI on the full Hamiltonian == FCI on the DUCC level-0 active Hamiltonian."""
    out_ham, active_orbitals, n_active_a, n_active_b = _ducc_output(
        xyz, multiplicity, unrestricted, active, inactive, 0
    )

    # Reference CASCI on the standard-convention active Hamiltonian (full chemist
    # integrals): MACIS for restricted, PySCF direct_uhf for unrestricted.
    cas_ham = create("hamiltonian_constructor").run(active_orbitals)
    ci = _ci_unrestricted if unrestricted else _ci_restricted
    energy_casci = ci(cas_ham, n_active_a, n_active_b)

    # FCI on the DUCC output. Every level emits the spin-blocked container whose
    # same-spin block is the half-antisymmetrized representative, which a
    # conventional solver misreads, so it is diagonalized through the native
    # qubit mapper.
    energy_fci = _qubit_ground_state(out_ham)

    assert np.isclose(energy_casci, energy_fci, atol=ci_energy_tolerance), (
        f"{label}: CASCI={energy_casci:.10f} != FCI(DUCC0)={energy_fci:.10f}"
    )


# Closed-shell LiH active spaces with a real frozen core, shared by the dressed-output
# tests below. Each is run through both a restricted (RHF) and an unrestricted (UHF)
# reference; the unrestricted path exercises the spin-blocked (aaaa/bbbb/aabb) assembly.
_CLOSED_SHELL_CASES = [
    ("lih_cas22", LIH, [1, 2], [0]),
    ("lih_cas23", LIH, [1, 2, 3], [0]),
]


@pytest.mark.parametrize(
    ("label", "xyz", "active", "inactive"), _CLOSED_SHELL_CASES, ids=[c[0] for c in _CLOSED_SHELL_CASES]
)
def test_qubit_mapper_restricted_unrestricted_match_macis(label, xyz, active, inactive):
    """DUCC level-0 through the native qubit mapper.

    The restricted and unrestricted outputs give the same qubit ground-state energy,
    both equal to the MACIS total (core energy included).
    """
    out_r, active_orbitals, n_active_a, n_active_b = _ducc_output(xyz, 1, False, active, inactive, 0)
    out_u, *_ = _ducc_output(xyz, 1, True, active, inactive, 0)

    # MACIS reference on the standard-convention active Hamiltonian (full 8-fold
    # chemist integrals): total active-space energy, incl. core.
    cas_ham = create("hamiltonian_constructor").run(active_orbitals)
    energy_macis = _ci_restricted(cas_ham, n_active_a, n_active_b)

    energy_qubit_r = _qubit_ground_state(out_r)
    energy_qubit_u = _qubit_ground_state(out_u)

    # Restricted and unrestricted DUCC outputs give the same qubit ground state.
    assert np.isclose(energy_qubit_r, energy_qubit_u, atol=ci_energy_tolerance), (
        f"{label}: restricted={energy_qubit_r:.10f} != unrestricted={energy_qubit_u:.10f}"
    )
    # Both equal the MACIS total (the qubit ground state adds back the core energy).
    assert np.isclose(energy_qubit_r, energy_macis, atol=ci_energy_tolerance), (
        f"{label}: qubit(RHF)={energy_qubit_r:.10f} != MACIS={energy_macis:.10f}"
    )
    assert np.isclose(energy_qubit_u, energy_macis, atol=ci_energy_tolerance), (
        f"{label}: qubit(UHF)={energy_qubit_u:.10f} != MACIS={energy_macis:.10f}"
    )


# ── Levels 1-2: BCH-dressed effective Hamiltonian (generated BTAS backend) ──

# active = all orbitals -> no external space -> T_ext = 0 -> sigma = 0 -> bar{H} = H,
# so the level-0 identity (CASCI == FCI) must still hold at every BCH level.
_REDUCE_CASES = [
    ("lih_full_l1", LIH, 1, False, 1),
    ("lih_full_l2", LIH, 1, False, 2),
    ("oh_full_l1", OH, 2, True, 1),
    ("oh_full_l2", OH, 2, True, 2),
]


@pytest.mark.parametrize(
    ("label", "xyz", "multiplicity", "unrestricted", "level"),
    _REDUCE_CASES,
    ids=[c[0] for c in _REDUCE_CASES],
)
def test_level_gt0_reduces_to_bare_when_active_is_all(label, xyz, multiplicity, unrestricted, level):
    """With every orbital active, the BCH dressing collapses to the bare Hamiltonian."""
    wfn_hf = _scf(xyz, multiplicity, unrestricted)
    active = list(range(wfn_hf.get_orbitals().get_num_molecular_orbitals()))
    out_ham, active_orbitals, n_active_a, n_active_b = _ducc_output(
        xyz, multiplicity, unrestricted, active, [], level, wfn_hf=wfn_hf
    )

    cas_ham = create("hamiltonian_constructor").run(active_orbitals)
    ci = _ci_unrestricted if unrestricted else _ci_restricted
    energy_casci = ci(cas_ham, n_active_a, n_active_b)

    # The dressed (level > 0) output is spin-blocked regardless of the reference
    # type, so it is diagonalized through the qubit mapper (MACIS is restricted-only).
    energy_fci = _qubit_ground_state(out_ham)

    assert np.isclose(energy_casci, energy_fci, atol=ci_energy_tolerance), (
        f"{label}: CASCI={energy_casci:.10f} != FCI(DUCC{level})={energy_fci:.10f}"
    )


@pytest.mark.parametrize("level", [1, 2], ids=["level1", "level2"])
@pytest.mark.parametrize(
    ("label", "xyz", "active", "inactive"), _CLOSED_SHELL_CASES, ids=[c[0] for c in _CLOSED_SHELL_CASES]
)
def test_ducc_dressed_restricted_unrestricted_match(label, xyz, active, inactive, level):
    """Restricted and unrestricted DUCC dressings of a closed-shell system agree.

    Closed-shell LiH is dressed at level ``level`` through both the restricted (RHF ->
    single-block effective Hamiltonian) and unrestricted (UHF == RHF -> spin-blocked
    aaaa/bbbb/aabb effective Hamiltonian) paths, over an active space with a real frozen
    core (non-zero T_ext, so the BCH dressing is non-trivial). Feeding both dressed
    outputs to the native qubit mapper must give the same ground-state energy, confirming
    the unrestricted spin-blocked dressing is consistent with the restricted one.
    """
    out_r, *_ = _ducc_output(xyz, 1, False, active, inactive, level)
    out_u, *_ = _ducc_output(xyz, 1, True, active, inactive, level)
    energy_r = _qubit_ground_state(out_r)
    energy_u = _qubit_ground_state(out_u)
    assert np.isclose(energy_r, energy_u, atol=ci_energy_tolerance), (
        f"{label} level {level}: restricted={energy_r:.10f} != unrestricted={energy_u:.10f}"
    )


def _macis_single_block(out, active_orbitals, n_active_a, n_active_b):
    """MACIS energy of the restricted single-block form of a (possibly spin-blocked) DUCC output.

    A dressed (level > 0) output is spin-blocked (unrestricted), which MACIS rejects. The
    restricted single-block form -- the alpha one-body plus the opposite-spin block
    v = 2 g[aabb] -- is the representation MACIS re-antisymmetrizes internally, reproducing
    the pre-spin-blocking Hamiltonian. For a closed-shell system it is exact.
    """
    h1a = np.asarray(out.get_one_body_integrals()[0])
    v_ab = np.asarray(out.get_two_body_integrals()[1])  # opposite-spin (aabb) block
    n = round(h1a.size**0.5)
    container = CanonicalFourCenterHamiltonianContainer(
        h1a.reshape(n, n), v_ab.ravel(), active_orbitals, out.get_core_energy(), np.eye(0)
    )
    return _ci_restricted(Hamiltonian(container), n_active_a, n_active_b)


@pytest.mark.parametrize("level", [0, 1, 2], ids=["level0", "level1", "level2"])
@pytest.mark.parametrize(
    ("label", "xyz", "active", "inactive"), _CLOSED_SHELL_CASES, ids=[c[0] for c in _CLOSED_SHELL_CASES]
)
def test_ducc_restricted_macis_matches_independent_solvers(label, xyz, active, inactive, level):
    """MACIS agrees with two symmetry-agnostic solvers on the restricted DUCC output.

    For a closed-shell system at DUCC levels 0-2, the ground-state energy from MACIS (on the
    single-block form it re-antisymmetrizes), PySCF ``direct_uhf`` (reading the spin-blocked
    integrals as-is), and the native qubit mapper (which reads each spin channel independently,
    assuming no chemist 8-fold symmetry) must all agree. This confirms MACIS reads the
    half-antisymmetrized representative correctly rather than merely self-consistently.
    """
    out, ao, n_active_a, n_active_b = _ducc_output(xyz, 1, False, active, inactive, level)
    e_macis = _macis_single_block(out, ao, n_active_a, n_active_b)
    e_pyscf = _ci_unrestricted(out, n_active_a, n_active_b)
    e_qubit = _qubit_ground_state(out)
    assert np.isclose(e_macis, e_pyscf, atol=ci_energy_tolerance), (
        f"{label} L{level}: MACIS={e_macis:.10f} != pyscf_uhf={e_pyscf:.10f}"
    )
    assert np.isclose(e_macis, e_qubit, atol=ci_energy_tolerance), (
        f"{label} L{level}: MACIS={e_macis:.10f} != qubit_mapper={e_qubit:.10f}"
    )


# ── Absolute reference energies ──

# Closed-shell active spaces with a frozen core, so T_ext != 0 and the BCH dressing
# is non-trivial. Each uses a frontier window of active occupied and virtual orbitals
# about the Fermi level.
_REFERENCE_CASES = [*_CLOSED_SHELL_CASES, ("h2o_cas65", H2O, [2, 3, 4, 5, 6], [0, 1])]

# Total active-space energies from an independent DUCC implementation, evaluated
# on the same SCF orbitals and diagonalized with MACIS.
_REFERENCE_ENERGIES = {
    ("lih_cas22", 1): -7.8706560183,
    ("lih_cas22", 2): -7.8828224008,
    ("lih_cas23", 1): -7.8705694522,
    ("lih_cas23", 2): -7.8827929828,
    ("h2o_cas65", 1): -75.0074995411,
    ("h2o_cas65", 2): -75.0130624402,
}

# Cholesky decomposition of the reference two-electron integrals caps cross-code
# agreement well above the CI convergence tolerance, yet still far below the
# deviation a missing or mis-leveled dressing would produce.
_reference_energy_tolerance = 100 * ci_energy_tolerance


@pytest.mark.parametrize("level", [1, 2], ids=["level1", "level2"])
@pytest.mark.parametrize(("label", "xyz", "active", "inactive"), _REFERENCE_CASES, ids=[c[0] for c in _REFERENCE_CASES])
def test_dressed_matches_reference_energy(label, xyz, active, inactive, level):
    """The BCH-dressed active Hamiltonian reproduces an independent reference energy.

    An absolute check of the generated BCH equations at levels 1-2 over active spaces
    with a real frozen core: unlike the internal-consistency tests, a systematic error
    in the dressing cannot cancel here, because the reference comes from an independent
    implementation rather than another view of the same output.
    """
    out, ao, n_active_a, n_active_b = _ducc_output(xyz, 1, False, active, inactive, level)
    energy = _macis_single_block(out, ao, n_active_a, n_active_b)
    reference = _REFERENCE_ENERGIES[(label, level)]
    assert np.isclose(energy, reference, atol=_reference_energy_tolerance), (
        f"{label} L{level}: native={energy:.10f} != reference={reference:.10f}"
    )
