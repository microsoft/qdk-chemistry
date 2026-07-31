# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""Tests for the native DUCC ``effective_hamiltonian`` builder (``ducc_level`` 0-2).

At level 0 there is no BCH dressing, so the effective active-space Hamiltonian
must reproduce the bare active-space (CASCI) problem exactly: CASCI on the
original full Hamiltonian equals FCI on the DUCC output Hamiltonian.

Levels 1-2 add the Baker-Campbell-Hausdorff dressing, evaluated by the generated
BTAS equations. Two invariants are checked: (a) when the active space is the
whole orbital set there is no external space, so the dressing collapses to the
bare Hamiltonian and the level-0 identity still holds at every level; (b) the
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

import numpy as np
import pytest

pytest.importorskip("pyscf")

from pyscf.fci import direct_uhf

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
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
from qdk_chemistry.data.symmetry import spin_index_set
from qdk_chemistry.plugins.pyscf.coupled_cluster import PyscfCoupledClusterCalculator

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
        ca, cb = base.get_coefficients()
        ea, eb = base.get_energies()
        return Orbitals(
            coefficients_alpha=np.array(ca),
            coefficients_beta=np.array(cb),
            energies_alpha=np.array(ea),
            energies_beta=np.array(eb),
            ao_overlap=np.array(base.get_overlap_matrix()),
            basis_set=base.get_basis_set(),
            active_indices=active_idx,
            inactive_indices=inactive_idx,
        )
    return Orbitals(
        coefficients=np.array(base.get_coefficients()[0]),
        energies=np.array(base.get_energies()[0]),
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


def _ducc0_output(xyz, multiplicity, unrestricted, active, inactive):
    """Build the DUCC level-0 output Hamiltonian and its active electron counts."""
    wfn_hf = _scf(xyz, multiplicity, unrestricted)
    orbitals = wfn_hf.get_orbitals()
    nmo = np.array(orbitals.get_coefficients()[0]).shape[1]
    nocc_a, nocc_b = wfn_hf.get_total_num_electrons()
    full_ham = create("hamiltonian_constructor").run(orbitals)
    cc_wfn = _full_ccsd(full_ham, _hf_determinant(nmo, nocc_a, nocc_b, orbitals))
    active_orbitals = _active_orbitals(orbitals, nmo, active, inactive, unrestricted)
    builder = create("effective_hamiltonian", "ducc")
    builder.settings().set("ducc_level", 0)
    out_ham = builder.run(full_ham, cc_wfn, active_orbitals)
    n_active_a = sum(1 for p in active if p < nocc_a)
    n_active_b = sum(1 for p in active if p < nocc_b)
    return out_ham, n_active_a, n_active_b


# Each case: label, geometry, spin multiplicity, unrestricted flag, and the
# active and inactive spatial-MO index lists defining the active space.
_CASES = [
    ("lih_full", LIH, 1, False, list(range(6)), []),
    ("lih_cas22", LIH, 1, False, [1, 2], [0]),
    ("lih_cas23", LIH, 1, False, [1, 2, 3], [0]),
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
    wfn_hf = _scf(xyz, multiplicity, unrestricted)
    orbitals = wfn_hf.get_orbitals()
    nmo = np.array(orbitals.get_coefficients()[0]).shape[1]
    nocc_a, nocc_b = wfn_hf.get_total_num_electrons()
    full_ham = create("hamiltonian_constructor").run(orbitals)
    cc_wfn = _full_ccsd(full_ham, _hf_determinant(nmo, nocc_a, nocc_b, orbitals))

    active_orbitals = _active_orbitals(orbitals, nmo, active, inactive, unrestricted)
    n_active_a = sum(1 for p in active if p < nocc_a)
    n_active_b = sum(1 for p in active if p < nocc_b)

    # Reference CASCI on the standard-convention active Hamiltonian (full chemist
    # integrals): MACIS for restricted, PySCF direct_uhf for unrestricted.
    cas_ham = create("hamiltonian_constructor").run(active_orbitals)
    ci = _ci_unrestricted if unrestricted else _ci_restricted
    energy_casci = ci(cas_ham, n_active_a, n_active_b)

    builder = create("effective_hamiltonian", "ducc")
    builder.settings().set("ducc_level", 0)
    out_ham = builder.run(full_ham, cc_wfn, active_orbitals)

    # FCI on the DUCC output. Its same-spin block is the half-antisymmetrized
    # representative: MACIS reads it (restricted); direct_uhf does not, so the
    # unrestricted output is diagonalized through the native qubit mapper.
    energy_fci = _qubit_ground_state(out_ham) if unrestricted else _ci_restricted(out_ham, n_active_a, n_active_b)

    assert np.isclose(energy_casci, energy_fci, atol=1e-8), (
        f"{label}: CASCI={energy_casci:.10f} != FCI(DUCC0)={energy_fci:.10f}"
    )


def test_active_orbitals_must_be_subset_of_wavefunction():
    """The active orbitals must share the wavefunction's MO basis (subset assertion)."""
    wfn_hf = _scf(LIH, 1, False)
    orbitals = wfn_hf.get_orbitals()
    nmo = np.array(orbitals.get_coefficients()[0]).shape[1]
    nocc_a, nocc_b = wfn_hf.get_total_num_electrons()
    full_ham = create("hamiltonian_constructor").run(orbitals)
    cc_wfn = _full_ccsd(full_ham, _hf_determinant(nmo, nocc_a, nocc_b, orbitals))

    # Active orbitals with perturbed coefficients do not match the wavefunction's.
    bad = Orbitals(
        coefficients=np.array(orbitals.get_coefficients()[0]) * 1.5,
        energies=np.array(orbitals.get_energies()[0]),
        ao_overlap=np.array(orbitals.get_overlap_matrix()),
        basis_set=orbitals.get_basis_set(),
        active_indices=spin_index_set(nmo, [1, 2], [1, 2], equivalent=True),
    )
    builder = create("effective_hamiltonian", "ducc")
    builder.settings().set("ducc_level", 0)
    with pytest.raises(RuntimeError, match="subset"):
        builder.run(full_ham, cc_wfn, bad)


# Closed-shell LiH active spaces for the qubit-mapper equivalence test. Each is
# run through both a restricted (RHF) and an unrestricted (UHF) reference; the
# unrestricted path exercises the spin-blocked (aaaa/bbbb/aabb) assembly.
_QUBIT_CASES = [
    ("lih_cas22", LIH, [1, 2], [0]),
    ("lih_cas23", LIH, [1, 2, 3], [0]),
]


@pytest.mark.parametrize(("label", "xyz", "active", "inactive"), _QUBIT_CASES, ids=[c[0] for c in _QUBIT_CASES])
def test_qubit_mapper_restricted_unrestricted_match_macis(label, xyz, active, inactive):
    """DUCC level-0 through the native qubit mapper.

    The restricted and unrestricted outputs give the same qubit ground-state energy,
    both equal to the MACIS total (core energy included).
    """
    out_r, n_active_a, n_active_b = _ducc0_output(xyz, 1, False, active, inactive)
    out_u, _, _ = _ducc0_output(xyz, 1, True, active, inactive)

    # MACIS reference (restricted only): total active-space energy, incl. core.
    energy_macis = _ci_restricted(out_r, n_active_a, n_active_b)

    energy_qubit_r = _qubit_ground_state(out_r)
    energy_qubit_u = _qubit_ground_state(out_u)

    # Restricted and unrestricted DUCC outputs give the same qubit ground state.
    assert np.isclose(energy_qubit_r, energy_qubit_u, atol=1e-8), (
        f"{label}: restricted={energy_qubit_r:.10f} != unrestricted={energy_qubit_u:.10f}"
    )
    # Both equal the MACIS total (the qubit ground state adds back the core energy).
    assert np.isclose(energy_qubit_r, energy_macis, atol=1e-8), (
        f"{label}: qubit(RHF)={energy_qubit_r:.10f} != MACIS={energy_macis:.10f}"
    )
    assert np.isclose(energy_qubit_u, energy_macis, atol=1e-8), (
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
    orbitals = wfn_hf.get_orbitals()
    nmo = np.array(orbitals.get_coefficients()[0]).shape[1]
    nocc_a, nocc_b = wfn_hf.get_total_num_electrons()
    full_ham = create("hamiltonian_constructor").run(orbitals)
    cc_wfn = _full_ccsd(full_ham, _hf_determinant(nmo, nocc_a, nocc_b, orbitals))

    active = list(range(nmo))
    active_orbitals = _active_orbitals(orbitals, nmo, active, [], unrestricted)
    n_active_a = sum(1 for p in active if p < nocc_a)
    n_active_b = sum(1 for p in active if p < nocc_b)

    cas_ham = create("hamiltonian_constructor").run(active_orbitals)
    ci = _ci_unrestricted if unrestricted else _ci_restricted
    energy_casci = ci(cas_ham, n_active_a, n_active_b)

    builder = create("effective_hamiltonian", "ducc")
    builder.settings().set("ducc_level", level)
    out_ham = builder.run(full_ham, cc_wfn, active_orbitals)
    # The dressed (level > 0) output is spin-blocked regardless of the reference
    # type, so it is diagonalized through the qubit mapper (MACIS is restricted-only).
    energy_fci = _qubit_ground_state(out_ham)

    assert np.isclose(energy_casci, energy_fci, atol=1e-8), (
        f"{label}: CASCI={energy_casci:.10f} != FCI(DUCC{level})={energy_fci:.10f}"
    )


def _native_ducc_output(xyz, multiplicity, unrestricted, active, inactive, level):
    """Native DUCC level-`level` output Hamiltonian, the full Hamiltonian, and electron counts."""
    wfn_hf = _scf(xyz, multiplicity, unrestricted)
    orbitals = wfn_hf.get_orbitals()
    nmo = np.array(orbitals.get_coefficients()[0]).shape[1]
    nocc_a, nocc_b = wfn_hf.get_total_num_electrons()
    full_ham = create("hamiltonian_constructor").run(orbitals)
    cc_wfn = _full_ccsd(full_ham, _hf_determinant(nmo, nocc_a, nocc_b, orbitals))
    active_orbitals = _active_orbitals(orbitals, nmo, active, inactive, unrestricted)
    builder = create("effective_hamiltonian", "ducc")
    builder.settings().set("ducc_level", level)
    out_ham = builder.run(full_ham, cc_wfn, active_orbitals)
    return out_ham, full_ham, nocc_a, nocc_b


@pytest.mark.parametrize("level", [1, 2], ids=["level1", "level2"])
@pytest.mark.parametrize(("label", "xyz", "active", "inactive"), _QUBIT_CASES, ids=[c[0] for c in _QUBIT_CASES])
def test_ducc_dressed_restricted_unrestricted_match(label, xyz, active, inactive, level):
    """Restricted and unrestricted DUCC dressings of a closed-shell system agree.

    Closed-shell LiH is dressed at level ``level`` through both the restricted (RHF ->
    single-block effective Hamiltonian) and unrestricted (UHF == RHF -> spin-blocked
    aaaa/bbbb/aabb effective Hamiltonian) paths, over an active space with a real frozen
    core (non-zero T_ext, so the BCH dressing is non-trivial). Feeding both dressed
    outputs to the native qubit mapper must give the same ground-state energy, confirming
    the unrestricted spin-blocked dressing is consistent with the restricted one.
    """
    out_r, *_ = _native_ducc_output(xyz, 1, False, active, inactive, level)
    out_u, *_ = _native_ducc_output(xyz, 1, True, active, inactive, level)
    energy_r = _qubit_ground_state(out_r)
    energy_u = _qubit_ground_state(out_u)
    assert np.isclose(energy_r, energy_u, atol=1e-8), (
        f"{label} level {level}: restricted={energy_r:.10f} != unrestricted={energy_u:.10f}"
    )


def _restricted_ducc_output(xyz, active, inactive, level):
    """Restricted DUCC output, its active-space orbitals, and active electron counts."""
    wfn_hf = _scf(xyz, 1, False)
    orbitals = wfn_hf.get_orbitals()
    nmo = np.array(orbitals.get_coefficients()[0]).shape[1]
    nocc_a, nocc_b = wfn_hf.get_total_num_electrons()
    full_ham = create("hamiltonian_constructor").run(orbitals)
    cc_wfn = _full_ccsd(full_ham, _hf_determinant(nmo, nocc_a, nocc_b, orbitals))
    active_orbitals = _active_orbitals(orbitals, nmo, active, inactive, False)
    builder = create("effective_hamiltonian", "ducc")
    builder.settings().set("ducc_level", level)
    out = builder.run(full_ham, cc_wfn, active_orbitals)
    n_active_a = sum(1 for p in active if p < nocc_a)
    n_active_b = sum(1 for p in active if p < nocc_b)
    return out, active_orbitals, n_active_a, n_active_b


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


def _openfermion_ground_state(out, nelec):
    """Ground-state energy via OpenFermion Jordan-Wigner, reading the DUCC integrals as-is.

    ``hamiltonian_to_interaction_operator`` builds the spin-orbital operator from the output's
    stored spin blocks without assuming chemist 8-fold symmetry (it handles the DUCC
    dual-4-fold blocks), so the diagonalization is fully independent of the qdk qubit mapper.
    """
    import openfermion as of  # noqa: PLC0415

    from qdk_chemistry.plugins.openfermion.conversion import hamiltonian_to_interaction_operator  # noqa: PLC0415

    sparse = of.linalg.get_sparse_operator(hamiltonian_to_interaction_operator(out))
    energy, _ = of.linalg.jw_get_ground_state_at_particle_number(sparse, nelec)
    return float(energy)


@pytest.mark.parametrize("level", [0, 1, 2], ids=["level0", "level1", "level2"])
@pytest.mark.parametrize(("label", "xyz", "active", "inactive"), _QUBIT_CASES, ids=[c[0] for c in _QUBIT_CASES])
def test_ducc_restricted_macis_matches_independent_solvers(label, xyz, active, inactive, level):
    """MACIS agrees with two symmetry-agnostic solvers on the restricted DUCC output.

    For a closed-shell system at DUCC levels 0-2, the ground-state energy from MACIS (on the
    single-block form it re-antisymmetrizes), PySCF ``direct_uhf`` (reading the spin-blocked
    integrals as-is), and OpenFermion's Jordan-Wigner (reading the output integrals as-is, no
    symmetry assumption) must all agree. This confirms MACIS reads the half-antisymmetrized
    representative correctly rather than merely self-consistently.
    """
    pytest.importorskip("openfermion")
    out, ao, n_active_a, n_active_b = _restricted_ducc_output(xyz, active, inactive, level)
    e_macis = _macis_single_block(out, ao, n_active_a, n_active_b)
    e_pyscf = _ci_unrestricted(out, n_active_a, n_active_b)
    e_openfermion = _openfermion_ground_state(out, n_active_a + n_active_b)
    assert np.isclose(e_macis, e_pyscf, atol=1e-8), (
        f"{label} L{level}: MACIS={e_macis:.10f} != pyscf_uhf={e_pyscf:.10f}"
    )
    assert np.isclose(e_macis, e_openfermion, atol=1e-8), (
        f"{label} L{level}: MACIS={e_macis:.10f} != openfermion={e_openfermion:.10f}"
    )
