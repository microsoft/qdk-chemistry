"""Integration tests for the ExaChem CCSD calculator with T-amplitude read-in.

Both the ExaChem and PySCF coupled-cluster plugins implement the same
:class:`~qdk_chemistry.algorithms.DynamicalCorrelationCalculator` interface, so each
test runs them on an identical :class:`~qdk_chemistry.data.Ansatz` built from
qdk-chemistry's own SCF orbitals. The only difference between the two runs is the
coupled-cluster solver itself: ExaChem's Cholesky-decomposed TAMM implementation
versus PySCF's.

Because both consume the same MO coefficients there is no per-orbital phase
ambiguity, so the converged T1/T2 amplitudes are compared elementwise rather than
by magnitude. The two runs are not otherwise bit-identical: ExaChem is handed the
orbitals through its serial-IO restart files and rebuilds the integrals with its
own Libint2, so agreement is limited by that integral re-computation, by ExaChem's
Cholesky decomposition tolerance, and by the two codes' coupled-cluster
convergence thresholds. Both restricted (RHF) and unrestricted (UHF) references
are exercised, and an effective-core-potential case is checked against native PySCF.

Each ExaChem run goes through MPI, so those tests are marked ``slow`` and are skipped when ``ExaChem`` (found on
``PATH``), an MPI runtime, or PySCF is missing. The remaining checks need none of them.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pytest

import qdk_chemistry.plugins.exachem as exachem_plugin
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Ansatz, AOType, BasisSet, Structure
from qdk_chemistry.plugins.exachem import ccsd_calculator
from qdk_chemistry.plugins.exachem.cli import (
    ExachemNotFoundError,
    ExachemRunError,
    find_exachem_binary,
    find_mpi_launcher,
)
from qdk_chemistry.plugins.exachem.conversion import parse_ccsd_amplitudes_unrestricted
from qdk_chemistry.plugins.exachem.scf_export import (
    basis_name_to_exachem,
    ecp_name_to_exachem,
    write_qdk_basis_g94,
    write_qdk_ecp,
)

from .reference_tolerances import mp2_energy_tolerance, rdm_tolerance

if TYPE_CHECKING:
    from pathlib import Path

exachem_plugin.load()

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_EXACHEM_BINARY = shutil.which("ExaChem") or ""
_MPI_LAUNCHER = shutil.which("mpirun") or shutil.which("srun") or ""

try:
    from pyscf import cc, gto, scf

    import qdk_chemistry.plugins.pyscf as pyscf_plugin

    pyscf_plugin.load()

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

_requires_exachem = pytest.mark.skipif(
    not (_EXACHEM_BINARY and _MPI_LAUNCHER and PYSCF_AVAILABLE),
    reason="Requires ExaChem on PATH, an MPI runtime, and PySCF",
)

# With run_coupled_cluster's thresholds the codes agree to ~1e-9 Eh and ~1e-8 in amplitudes (ExaChem 45c192e8).
_energy_tolerance = 100 * mp2_energy_tolerance
# Amplitudes are wavefunction parameters and converge less tightly than the energy.
_amplitude_tolerance = 10 * rdm_tolerance

# ---------------------------------------------------------------------------
# Test molecules
# ---------------------------------------------------------------------------

H2 = "2\nH2\nH 0 0 0\nH 0 0 0.74\n"
LIH = "2\nLiH\nLi 0 0 0\nH 0 0 1.6\n"
H2O = "3\nH2O\nO 0 0 0.117790\nH 0 0.756950 -0.471161\nH 0 -0.756950 -0.471161\n"
OH = "2\nOH\nO 0 0 0\nH 0 0 0.97\n"
HI = "2\nHI\nH 0 0 0\nI 0 0 1.609\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_ansatz(xyz: str, basis: str, charge: int, multiplicity: int, scf_type: str) -> Ansatz:
    """Run qdk-chemistry SCF and wrap the result in an Ansatz.

    Args:
        xyz: Molecular geometry as an XYZ-format string.
        basis: Gaussian basis set name.
        charge: Molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        scf_type: SCF type, ``"restricted"`` or ``"unrestricted"``.

    Returns:
        An Ansatz pairing the qdk-chemistry Hamiltonian with its SCF wavefunction.

    """
    scf_solver = create("scf_solver")
    scf_solver.settings().set("scf_type", scf_type)
    _, wavefunction = scf_solver.run(Structure.from_xyz(xyz), charge, multiplicity, basis)

    hamiltonian = create("hamiltonian_constructor").run(wavefunction.get_orbitals())
    return Ansatz(hamiltonian, wavefunction)


def run_coupled_cluster(variant: str, ansatz: Ansatz):
    """Run a coupled-cluster calculator variant with amplitude storage enabled.

    Args:
        variant: The registered calculator name, e.g. ``"exachem_ccsd"``.
        ansatz: The reference Ansatz to correlate.

    Returns:
        The tuple ``(total_energy, wavefunction, bra_wavefunction)`` from the calculator.

    """
    calculator = create("dynamical_correlation_calculator", variant)
    calculator.settings().set("store_amplitudes", True)
    # Default thresholds leave up to ~1e-6 Eh between the codes, the size of the tolerance.
    if variant == "exachem_ccsd":
        calculator.settings().set("ccsd_threshold", 1e-9)
        calculator.settings().set("cd_diagtol", 1e-9)
    else:
        calculator.settings().set("conv_tol", 1e-10)
        calculator.settings().set("conv_tol_normt", 1e-8)
    return calculator.run(ansatz)


def assert_amplitudes_close(container, reference_container) -> None:
    """Assert two amplitude containers hold the same T1/T2 blocks.

    Args:
        container: The amplitude container under test.
        reference_container: The amplitude container to compare against.

    """
    assert container.has_t1_amplitudes()
    assert container.has_t2_amplitudes()

    for block, reference_block in zip(
        container.get_t1_amplitudes(), reference_container.get_t1_amplitudes(), strict=True
    ):
        np.testing.assert_allclose(np.asarray(block), np.asarray(reference_block), atol=_amplitude_tolerance)

    for block, reference_block in zip(
        container.get_t2_amplitudes(), reference_container.get_t2_amplitudes(), strict=True
    ):
        np.testing.assert_allclose(np.asarray(block), np.asarray(reference_block), atol=_amplitude_tolerance)


def assert_exachem_matches_pyscf(
    xyz: str,
    basis: str,
    charge: int = 0,
    multiplicity: int = 1,
    scf_type: str = "restricted",
) -> None:
    """Assert the ExaChem and PySCF CCSD plugins agree on the same Ansatz.

    Args:
        xyz: Molecular geometry as an XYZ-format string.
        basis: Gaussian basis set name.
        charge: Molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        scf_type: SCF type, ``"restricted"`` or ``"unrestricted"``.

    """
    ansatz = build_ansatz(xyz, basis, charge, multiplicity, scf_type)

    energy, wavefunction, bra_wavefunction = run_coupled_cluster("exachem_ccsd", ansatz)
    reference_energy, reference_wavefunction, _ = run_coupled_cluster("pyscf_coupled_cluster", ansatz)

    np.testing.assert_allclose(
        energy,
        reference_energy,
        atol=_energy_tolerance,
        err_msg=f"ExaChem CCSD energy ({energy:.10f}) differs from PySCF ({reference_energy:.10f})",
    )

    # The returned wavefunction carries the amplitudes; CCSD has no distinct bra.
    assert bra_wavefunction is None
    assert wavefunction.get_container_type() == "amplitude"
    assert_amplitudes_close(wavefunction.get_container(), reference_wavefunction.get_container())


def read_exachem_ecp(path: Path) -> dict[str, tuple[int, dict[str, list[tuple[int, float, float]]]]]:
    """Read an ExaChem ``.ecp`` file.

    Args:
        path: Path of the ``.ecp`` file.

    Returns:
        ``{element: (core electrons, {channel: [(radial power, exponent, coefficient), ...]})}``.

    """
    lines = path.read_text().splitlines()
    assert lines[0] == "ECP"
    assert lines[-1] == "END"
    ecps: dict[str, tuple[int, dict[str, list[tuple[int, float, float]]]]] = {}
    element = channel = ""
    for tokens in map(str.split, lines[1:-1]):
        if len(tokens) == 3 and tokens[1] == "nelec":
            element = tokens[0]
            ecps[element] = (int(tokens[2]), {})
        elif len(tokens) == 2:
            channel = tokens[1]
        else:
            ecps[element][1].setdefault(channel, []).append((int(tokens[0]), float(tokens[1]), float(tokens[2])))
    return ecps


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@_requires_exachem
class TestCcsdRestricted:
    """Compare the ExaChem and PySCF CCSD plugins for RHF references."""

    @pytest.mark.parametrize(
        ("xyz", "basis"),
        [
            pytest.param(H2, "sto-3g", id="h2_sto3g"),
            pytest.param(LIH, "sto-3g", id="lih_sto3g"),
            pytest.param(H2O, "sto-3g", id="h2o_sto3g"),
            pytest.param(H2O, "cc-pvdz", id="h2o_ccpvdz"),
            # ExaChem renames aug-* basis sets internally.
            pytest.param(H2O, "aug-cc-pvdz", id="h2o_augccpvdz"),
        ],
    )
    def test_energy_and_amplitudes_match_pyscf(self, xyz: str, basis: str):
        assert_exachem_matches_pyscf(xyz, basis)


@pytest.mark.slow
@_requires_exachem
class TestCcsdUnrestricted:
    """Compare the ExaChem and PySCF CCSD plugins for UHF references.

    These exercise the MSO-tensor spin-block decoding of the T amplitudes.
    """

    @pytest.mark.parametrize(
        ("xyz", "basis"),
        [
            pytest.param(OH, "sto-3g", id="oh_doublet_sto3g"),
            pytest.param(OH, "cc-pvdz", id="oh_doublet_ccpvdz"),
        ],
    )
    def test_energy_and_amplitudes_match_pyscf(self, xyz: str, basis: str):
        assert_exachem_matches_pyscf(xyz, basis, multiplicity=2, scf_type="unrestricted")


@pytest.mark.slow
@_requires_exachem
class TestCcsdEffectiveCorePotential:
    """Compare ExaChem with native PySCF for basis sets that carry ECPs.

    Native PySCF uses its own ECP library, so it independently checks the ECP ExaChem runs with.
    """

    @pytest.mark.parametrize("basis", ["def2-svp", "lanl2dz"])
    def test_energy_matches_native_pyscf(self, basis: str):
        energy, wavefunction, _ = run_coupled_cluster("exachem_ccsd", build_ansatz(HI, basis, 0, 1, "restricted"))

        molecule = gto.M(atom="H 0 0 0; I 0 0 1.609", basis=basis, ecp=basis, verbose=0)
        reference = cc.CCSD(scf.RHF(molecule).run(conv_tol=1e-10)).run(conv_tol=1e-8)

        np.testing.assert_allclose(energy, reference.e_tot, atol=_energy_tolerance)
        assert wavefunction.get_container().has_t2_amplitudes()


@pytest.mark.slow
@_requires_exachem
class TestActiveSpace:
    """Check that ExaChem correlates only the Ansatz's active orbitals."""

    def test_valence_active_space_matches_native_pyscf(self):
        _, wavefunction = create("scf_solver").run(Structure.from_xyz(H2O), 0, 1, "cc-pvdz")
        selector = create("active_space_selector", "qdk_valence")
        selector.settings().set("num_active_electrons", 8)
        selector.settings().set("num_active_orbitals", 20)
        active_wavefunction = selector.run(wavefunction)
        hamiltonian = create("hamiltonian_constructor").run(active_wavefunction.get_orbitals())

        energy, amplitudes, _ = run_coupled_cluster("exachem_ccsd", Ansatz(hamiltonian, active_wavefunction))

        molecule = gto.M(atom="\n".join(H2O.splitlines()[2:]), basis="cc-pvdz", verbose=0)
        # O 1s and the three highest virtuals lie outside the active space.
        reference = cc.CCSD(scf.RHF(molecule).run(conv_tol=1e-10), frozen=[0, 21, 22, 23]).run(conv_tol=1e-10)

        np.testing.assert_allclose(energy, reference.e_tot, atol=_energy_tolerance)
        assert amplitudes.get_container().has_t2_amplitudes()


class TestBasisExport:
    """Check how qdk-chemistry's basis set is handed to ExaChem."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("cc-pVDZ", "cc-pvdz"), ("aug-cc-pvdz", "ec-aug-cc-pvdz"), ("ahlrichs pvdz", "ahlrichs_pvdz")],
    )
    def test_basis_name_to_exachem(self, name: str, expected: str):
        assert basis_name_to_exachem(name) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("def2-tzvp", "def2-ecp"),
            ("lanl2dz", "lanl2dz_ecp"),
            ("lanl2dzdp", "lanl2dz_ecp"),
            ("cc-pvdz-pp", "cc-pvdz-pp"),
            ("aug-cc-pvdz-pp", "ec-aug-cc-pvdz-pp"),
        ],
    )
    def test_ecp_name_to_exachem(self, name: str, expected: str):
        assert ecp_name_to_exachem(name) == expected

    def test_rejects_custom_basis_sets(self):
        structure = Structure.from_xyz(H2O)
        basis_set = BasisSet.from_element_map({"O": "cc-pvdz", "H": "sto-3g"}, structure)
        _, wavefunction = create("scf_solver").run(structure, 0, 1, basis_set)
        ansatz = Ansatz(create("hamiltonian_constructor").run(wavefunction.get_orbitals()), wavefunction)

        with pytest.raises(ValueError, match="named basis set"):
            ccsd_calculator.ExachemCcsdCalculator().run(ansatz)

    def test_rejects_different_shells_on_atoms_of_one_element(self, tmp_path):
        structure = Structure.from_xyz(H2O)
        basis_set = BasisSet.from_index_map({0: "cc-pvdz", 1: "sto-3g", 2: "cc-pvdz"}, structure)

        with pytest.raises(ValueError, match="one basis per element"):
            write_qdk_basis_g94(basis_set, structure.get_atomic_symbols(), tmp_path, "sto-3g")

    def test_writes_qdk_ecp(self, tmp_path):
        structure = Structure.from_xyz(HI)
        basis_set = BasisSet.from_basis_name("def2-svp", structure)

        write_qdk_ecp(basis_set, structure.get_atomic_symbols(), tmp_path, "def2-ecp")

        ecps = read_exachem_ecp(tmp_path / "basis" / "def2-ecp.ecp")
        assert list(ecps) == ["I"]
        core_electrons, channels = ecps["I"]
        assert core_electrons == 28
        assert list(channels) == ["ul", "S", "P", "D"]
        for shell in basis_set.get_ecp_shells_for_atom(1):
            angular_momentum = shell.get_angular_momentum()
            channel = "ul" if angular_momentum < 0 else "SPD"[angular_momentum]
            expected = np.column_stack([shell.rpowers, shell.exponents, shell.coefficients])
            np.testing.assert_allclose(channels[channel], expected, rtol=1e-14)

    def test_rejects_different_ecps_on_atoms_of_one_element(self, tmp_path):
        structure = Structure.from_xyz("2\nI2\nI 0 0 0\nI 0 0 2.67\n")
        basis_set = BasisSet.from_index_map({0: "def2-svp", 1: "lanl2dz"}, structure)

        with pytest.raises(ValueError, match="one ECP per element"):
            write_qdk_ecp(basis_set, structure.get_atomic_symbols(), tmp_path, "def2-ecp")

    def test_writes_zero_local_term_for_ecps_without_one(self, tmp_path):
        structure = Structure.from_xyz(HI)
        library_basis = BasisSet.from_basis_name("def2-svp", structure)
        semi_local = [shell for shell in library_basis.get_ecp_shells() if shell.get_angular_momentum() >= 0]
        basis_set = BasisSet(
            "def2-svp",
            library_basis.get_shells(),
            "def2-svp",
            semi_local,
            list(library_basis.get_ecp_electrons()),
            structure,
            AOType.Spherical,
        )

        write_qdk_ecp(basis_set, structure.get_atomic_symbols(), tmp_path, "def2-ecp")

        _, channels = read_exachem_ecp(tmp_path / "basis" / "def2-ecp.ecp")["I"]
        assert list(channels) == ["ul", "S", "P", "D"]
        assert channels["ul"] == [(2, 1.0, 0.0)]


class TestAmplitudeParsing:
    """Check how ExaChem's amplitude files are read back."""

    def test_unrestricted_amplitudes_split_into_spin_blocks(self, tmp_path):
        noa, nob, nva, nvb = 2, 1, 1, 2
        occupied_spin = np.repeat([0, 1], [noa, nob])
        virtual_spin = np.repeat([0, 1], [nva, nvb])
        rng = np.random.default_rng(seed=7)
        # ExaChem writes only the spin-conserving entries of its spin-orbital tensors.
        t1 = rng.standard_normal((nva + nvb, noa + nob)) * (virtual_spin[:, None] == occupied_spin[None, :])
        conserving = (
            virtual_spin[:, None, None, None] + virtual_spin[None, :, None, None]
            == occupied_spin[None, None, :, None] + occupied_spin[None, None, None, :]
        )
        t2 = rng.standard_normal(conserving.shape) * conserving
        t1_path, t2_path = tmp_path / "t1.txt", tmp_path / "t2.txt"
        t1_path.write_text("".join(f"{a} {i} {t1[a, i]:.17g}\n" for a, i in np.argwhere(t1)) + "\n")
        t2_path.write_text("".join(f"{a} {b} {i} {j} {t2[a, b, i, j]:.17g}\n" for a, b, i, j in np.argwhere(t2)))

        va, vb, oa, ob = slice(None, nva), slice(nva, None), slice(None, noa), slice(noa, None)
        expected = (
            t1[va, oa].T,
            t1[vb, ob].T,
            t2[va, va, oa, oa].transpose(2, 3, 0, 1),
            t2[va, vb, oa, ob].transpose(2, 3, 0, 1),
            t2[vb, vb, ob, ob].transpose(2, 3, 0, 1),
        )
        blocks = parse_ccsd_amplitudes_unrestricted(t1_path, t2_path, noa, nob, nva, nvb)
        for block, expected_block in zip(blocks, expected, strict=True):
            np.testing.assert_array_equal(block, expected_block)


class TestErrors:
    """Check the errors raised when ExaChem is missing or its output cannot be used."""

    def test_missing_programs_raise_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PATH", str(tmp_path))

        with pytest.raises(ExachemNotFoundError, match="ExaChem binary"):
            find_exachem_binary()
        with pytest.raises(ExachemNotFoundError, match="MPI launcher"):
            find_mpi_launcher()

    @pytest.mark.parametrize(
        ("output", "match"),
        [
            pytest.param("", "SCF energy", id="no_scf_energy"),
            pytest.param("Total SCF energy = {scf:.12f}\n", "CCSD total energy", id="no_ccsd_energy"),
            pytest.param(
                "Total SCF energy = {scf:.12f}\nCCSD total energy / hartree = {scf:.12f}\n",
                "T-amplitude files",
                id="no_amplitude_files",
            ),
        ],
    )
    def test_unusable_output_raises_run_error(self, monkeypatch, output: str, match: str):
        ansatz = build_ansatz(H2, "sto-3g", 0, 1, "restricted")
        stdout = output.format(scf=ansatz.calculate_energy())
        monkeypatch.setattr(ccsd_calculator, "run_exachem", lambda *_args, **_kwargs: stdout)
        calculator = ccsd_calculator.ExachemCcsdCalculator()
        calculator.settings().set("store_amplitudes", True)

        with pytest.raises(ExachemRunError, match=match):
            calculator.run(ansatz)


class TestPluginLoading:
    """Check that the plugin registers its calculator only when ExaChem can run."""

    @pytest.mark.parametrize(
        ("on_path", "registered"),
        [
            pytest.param({"ExaChem", "mpirun"}, ["exachem_ccsd"], id="exachem_and_mpirun"),
            pytest.param({"mpirun"}, [], id="no_exachem"),
            pytest.param({"ExaChem"}, [], id="no_mpi_launcher"),
        ],
    )
    def test_registers_only_when_exachem_and_mpi_are_on_path(
        self, monkeypatch, on_path: set[str], registered: list[str]
    ):
        factories: list = []
        monkeypatch.setattr(exachem_plugin, "_loaded", False)
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in on_path else None)
        monkeypatch.setattr("qdk_chemistry.algorithms.register", factories.append)

        exachem_plugin.load()

        assert [factory().name() for factory in factories] == registered
