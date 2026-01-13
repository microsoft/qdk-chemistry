"""PySCF stability analysis module for QDK.

This module provides wavefunction stability analysis capabilities using the PySCF library
and integrates PySCF stability algorithms into the QDK framework.

The implementation supports both restricted (RHF, ROHF) and unrestricted (UHF)
wavefunction stability analysis including:

- Internal stability analysis (within the same wavefunction type)
- RHF external stability analysis (RHF -> UHF instabilities)

This module registers a `pyscf` stability checker with the QDK stability checker registry at
import time, making the functionality available via
`qdk_chemistry.algorithms.create('stability_checker', 'pyscf')`.

Requires: PySCF (the code uses `pyscf.lib`, `pyscf.scf`, `pyscf.soscf`,
and `pyscf.scf.stability` routines).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from pyscf import lib, scf
from pyscf.lib import logger as pyscf_logger
from pyscf.scf.stability import _gen_hop_rhf_external
from pyscf.soscf import newton_ah

from qdk_chemistry.algorithms import StabilityChecker
from qdk_chemistry.data import Settings, StabilityResult, Wavefunction
from qdk_chemistry.plugins.pyscf.conversion import orbitals_to_scf
from qdk_chemistry.utils import Logger

__all__ = ["PyscfStabilityChecker", "PyscfStabilitySettings"]


def _stability_preconditioner(dx: np.ndarray, e: float, hdiag: np.ndarray) -> np.ndarray:
    """Preconditioner for stability analysis eigenvalue problems.

    Directly from PySCF implementation.
    """
    hdiagd = hdiag - e
    hdiagd[abs(hdiagd) < 1e-8] = 1e-8
    return dx / hdiagd


def _stability_hessian(hop, x: np.ndarray) -> np.ndarray:
    """Stability analysis Hessian function.

    Directly from PySCF implementation.

    The result of hop(x) corresponds to a displacement that reduces
    gradients g. It is the vir-occ block of the matrix vector product
    (Hessian*x). The occ-vir block equals x2.T.conj(). The overall
    Hessian for internal rotation is x2 + x2.T.conj(). This is
    the reason we apply (.real * 2) below.
    """
    return hop(x).real * 2


def _rhf_internal(
    mf: scf.hf.SCF, with_symmetry: bool = True, nroots: int = 3, tol: float = 1e-4, verbose: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """RHF internal stability analysis.

    Modified from PySCF implementation.
    """
    Logger.trace_entering()
    log = pyscf_logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_rhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=with_symmetry)
    hdiag *= 2

    x0 = np.zeros_like(g)
    x0[g != 0] = 1.0 / hdiag[g != 0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[np.argmin(hdiag)] = 1
    e, v = lib.davidson(
        lambda x: _stability_hessian(hop, x),
        x0,
        lambda dx, e, _x0: _stability_preconditioner(dx, e, hdiag),
        tol=tol,
        nroots=nroots,
        verbose=log,
    )
    log.info("rhf_internal: lowest eigs of H = %s", e)
    return e, v


def _rhf_external(
    mf: scf.hf.SCF, with_symmetry: bool = True, nroots: int = 3, tol: float = 1e-4, verbose: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """RHF external stability analysis (RHF -> UHF).

    Modified from PySCF implementation.
    """
    Logger.trace_entering()
    log = pyscf_logger.new_logger(mf, verbose)
    # Do not consider real -> complex instability
    _, _, hop2, hdiag2 = _gen_hop_rhf_external(mf, with_symmetry)

    x0 = np.zeros_like(hdiag2)
    x0[hdiag2 > 1e-5] = 1.0 / hdiag2[hdiag2 > 1e-5]
    e, v = lib.davidson(
        hop2, x0, lambda dx, e, _x0: _stability_preconditioner(dx, e, hdiag2), tol=tol, nroots=nroots, verbose=log
    )
    log.info("rhf_external: lowest eigs of H = %s", e)
    return e, v


def _rohf_internal(
    mf: scf.rohf.ROHF, with_symmetry: bool = True, nroots: int = 3, tol: float = 1e-4, verbose: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """ROHF internal stability analysis.

    Modified from PySCF implementation.
    """
    Logger.trace_entering()
    log = pyscf_logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_rohf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=with_symmetry)
    hdiag *= 2

    x0 = np.zeros_like(g)
    x0[g != 0] = 1.0 / hdiag[g != 0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[np.argmin(hdiag)] = 1
    e, v = lib.davidson(
        lambda x: _stability_hessian(hop, x),
        x0,
        lambda dx, e, _x0: _stability_preconditioner(dx, e, hdiag),
        tol=tol,
        nroots=nroots,
        verbose=log,
    )
    log.info("rohf_internal: lowest eigs of H = %s", e)
    return e, v


def _rohf_external(mf: scf.rohf.ROHF, with_symmetry: bool = True, nroots: int = 3, tol: float = 1e-4) -> None:
    """ROHF external stability analysis (not implemented)."""
    raise NotImplementedError


def _uhf_internal(
    mf: scf.uhf.UHF, with_symmetry: bool = True, nroots: int = 3, tol: float = 1e-4, verbose: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """UHF internal stability analysis.

    Modified from PySCF implementation.
    """
    Logger.trace_entering()
    log = pyscf_logger.new_logger(mf, verbose)
    g, hop, hdiag = newton_ah.gen_g_hop_uhf(mf, mf.mo_coeff, mf.mo_occ, with_symmetry=with_symmetry)
    hdiag *= 2

    x0 = np.zeros_like(g)
    x0[g != 0] = 1.0 / hdiag[g != 0]
    if not with_symmetry:  # allow to break point group symmetry
        x0[np.argmin(hdiag)] = 1
    e, v = lib.davidson(
        lambda x: _stability_hessian(hop, x),
        x0,
        lambda dx, e, _x0: _stability_preconditioner(dx, e, hdiag),
        tol=tol,
        nroots=nroots,
        verbose=log,
    )
    log.info("uhf_internal: lowest eigs of H = %s", e)
    return e, v


class PyscfStabilitySettings(Settings):
    """Configuration settings for PySCF stability analysis.

    This class manages settings for wavefunction stability analysis procedures using
    PySCF. It inherits from the Settings base class and exposes the
    configurable options used by :class:`PyscfStabilityChecker`.

    Available settings:

    - internal: Whether to perform internal stability analysis (within the same wavefunction type).
    - external: Whether to perform external stability analysis (RHF -> UHF instabilities).
        Only supported for RHF wavefunctions. Will raise an error if enabled for ROHF or UHF.
    - with_symmetry: Whether to respect point group symmetry during stability analysis.
    - nroots: Number of eigenvalue roots to compute in the Davidson solver.
    - davidson_tolerance: Convergence threshold for the Davidson eigenvalue solver.
    - stability_tolerance: Threshold for determining stability from eigenvalues.
    - method: The electronic structure method ("hf" for Hartree-Fock or a DFT functional name).
    - xc_grid: Integer DFT integration grid density level passed to PySCF (0=coarse, 9=very fine).
    - pyscf_verbose: PySCF verbosity level for lib.davidson logging (0=silent, 4=info, 5=debug).

    Examples:
        >>> settings = PyscfStabilitySettings()
        >>> settings.get("nroots")
        3
        >>> settings.set("nroots", 5)
        >>> settings.set("external", False)  # Only internal stability
        >>> settings.set("method", "b3lyp")  # Use B3LYP DFT

    """

    def __init__(self):
        """Initialize the stability checker with default parameters."""
        Logger.trace_entering()
        super().__init__()
        self._set_default("internal", "bool", True)
        self._set_default("external", "bool", True)
        self._set_default("with_symmetry", "bool", False)
        self._set_default("nroots", "int", 3)
        self._set_default("davidson_tolerance", "double", 1e-8)
        self._set_default("stability_tolerance", "double", -1e-4)
        self._set_default("method", "string", "hf")
        self._set_default(
            "xc_grid", "int", 3, "Density functional integration grid level (0=coarse, 9=very fine)", list(range(10))
        )
        self._set_default("pyscf_verbose", "int", 4)


class PyscfStabilityChecker(StabilityChecker):
    """PySCF-based stability checker for quantum chemistry wavefunctions.

    This class implements wavefunction stability analysis using routines from PySCF.
    It supports multiple wavefunction types and can perform internal stability analysis of
    RHF, ROHF, UHF and external stability analysis (RHF -> UHF instability).

    Internal stability eigenvalues are rescaled to follow the convention used in the
    original stability analysis formulation (J. Chem. Phys. 66, 3045-3050 (1977)) and
    to be consistent with the QDK implementation:

    - RHF internal eigenvalues are scaled by 1/4
    - UHF internal eigenvalues are scaled by 1/2
    - RHF external eigenvalues are left unscaled

    Key behavior:
    - Automatically detects wavefunction type (RHF, ROHF, UHF) and applies appropriate analysis
    - Internal stability analysis is performed within the same wavefunction type
    - External stability analysis (RHF -> UHF) only supported for RHF wavefunctions
    - Raises an error for ROHF or UHF when external analysis is requested
    - Returns StabilityResult with separate internal and external stability information

    Raises:
        ValueError: If wavefunction is not a SlaterDeterminantContainer (container type "sd").

    Examples:
        >>> checker = PyscfStabilityChecker()
        >>> is_stable, result = checker.run(wavefunction)
        >>> print(f"Overall stable: {is_stable}")
        >>> print(f"Internal stable: {result.is_internal_stable()}")
        >>> print(f"External stable: {result.is_external_stable()}")

    """

    def __init__(self):
        """Initialize the PySCF stability checker with default settings."""
        Logger.trace_entering()
        super().__init__()
        self._settings = PyscfStabilitySettings()

    def _run_impl(self, wavefunction: Wavefunction) -> tuple[bool, StabilityResult]:
        """Perform wavefunction stability analysis using PySCF.

        This method analyzes the stability of the input wavefunction by computing
        eigenvalues of the electronic Hessian matrix. The analysis type (internal/rhf_external)
        and method (RHF/ROHF/UHF) are automatically determined from the wavefunction
        and settings.

        Args:
            wavefunction: The wavefunction to analyze for stability.

        Returns:
            A tuple containing a boolean indicating overall stability and a StabilityResult object
            with detailed analysis results.

        Raises:
            ValueError: If wavefunction is not a SlaterDeterminantContainer or if external
                stability analysis is requested for ROHF/UHF wavefunctions.

        """
        Logger.trace_entering()
        # Verify wavefunction compatibility: Only SlaterDeterminantContainer currently supported
        if wavefunction.get_container_type() != "sd":
            raise ValueError("Stability analysis currently only supports SlaterDeterminantContainer wavefunctions")

        # Extract settings
        do_internal = self._settings.get("internal")
        do_external = self._settings.get("external")
        with_symmetry = self._settings.get("with_symmetry")
        nroots = self._settings.get("nroots")
        alg_tol = self._settings.get("davidson_tolerance")
        stability_tol = self._settings.get("stability_tolerance")
        method = self._settings.get("method")
        xc_grid = self._settings.get("xc_grid")
        pyscf_verbose = self._settings.get("pyscf_verbose")

        # Get orbitals from wavefunction
        orbitals = wavefunction.get_orbitals()
        if orbitals is None:
            raise ValueError("Wavefunction must contain orbital information for stability analysis")

        # Get electron counts from the wavefunction
        nalpha, nbeta = wavefunction.get_total_num_electrons()

        # Create occupation arrays (assumes aufbau filling)
        num_molecular_orbitals = orbitals.get_num_molecular_orbitals()
        occ_alpha = np.zeros(num_molecular_orbitals)
        occ_beta = np.zeros(num_molecular_orbitals)
        occ_alpha[:nalpha] = 1.0
        occ_beta[:nbeta] = 1.0

        # Convert to PySCF SCF object
        mf = orbitals_to_scf(orbitals, occ_alpha, occ_beta, method=method)
        if method.lower() != "hf":
            mf.grids.level = xc_grid

        # Determine wavefunction type and perform appropriate stability analysis
        internal_eigenvalues_list: list = []
        internal_eigenvectors_list: list = []
        external_eigenvalues_list: list = []
        external_eigenvectors_list: list = []

        # Scale factors for internal eigenvalues so that reported values follow the
        # convention of J. Chem. Phys. 66, 3045-3050 (1977) and match the QDK backend.
        internal_scale_factor = 1.0

        if isinstance(mf, scf.rohf.ROHF):
            # ROHF stability analysis
            if do_internal:
                e, v = _rohf_internal(
                    mf, with_symmetry=with_symmetry, nroots=nroots, tol=alg_tol, verbose=pyscf_verbose
                )
                internal_eigenvalues_list.extend(e if isinstance(e, list | tuple | np.ndarray) else [e])
                internal_eigenvectors_list.extend(v if isinstance(v, list | tuple | np.ndarray) else [v])
                internal_scale_factor = 1.0
            # Raise error if external stability is requested for ROHF
            if do_external:
                raise ValueError(
                    "External stability analysis (RHF -> UHF) is not supported for ROHF wavefunctions. "
                    "Only internal stability analysis is supported."
                )

        elif isinstance(mf, scf.uhf.UHF):
            # UHF stability analysis
            if do_internal:
                e, v = _uhf_internal(mf, with_symmetry=with_symmetry, nroots=nroots, tol=alg_tol, verbose=pyscf_verbose)
                # Scale UHF internal eigenvalues by 1/2 to match the convention
                # used in the original stability analysis paper and the QDK backend.
                internal_scale_factor = 0.5
                internal_eigenvalues_list.extend(e if isinstance(e, list | tuple | np.ndarray) else [e])
                internal_eigenvectors_list.extend(v if isinstance(v, list | tuple | np.ndarray) else [v])
            # Raise error if external stability is requested for UHF
            if do_external:
                raise ValueError(
                    "External stability analysis (RHF -> UHF) is not supported for UHF wavefunctions. "
                    "Only internal stability analysis is supported."
                )

        else:
            # RHF stability analysis (default)
            if do_internal:
                e, v = _rhf_internal(mf, with_symmetry=with_symmetry, nroots=nroots, tol=alg_tol, verbose=pyscf_verbose)
                # Scale RHF internal eigenvalues by 1/4 to match the convention
                # used in the original stability analysis paper and the QDK backend.
                internal_scale_factor = 0.25
                internal_eigenvalues_list.extend(e if isinstance(e, list | tuple | np.ndarray) else [e])
                internal_eigenvectors_list.extend(v if isinstance(v, list | tuple | np.ndarray) else [v])
            if do_external:
                e, v = _rhf_external(mf, with_symmetry=with_symmetry, nroots=nroots, tol=alg_tol, verbose=pyscf_verbose)
                external_eigenvalues_list.extend(e if isinstance(e, list | tuple | np.ndarray) else [e])
                external_eigenvectors_list.extend(v if isinstance(v, list | tuple | np.ndarray) else [v])

        # Process results and create StabilityResult
        internal_stable = True
        external_stable = True

        # Convert to numpy arrays if we have results. Internal eigenvalues are
        # scaled (RHF by 1/4, UHF by 1/2) to follow the convention of
        # J. Chem. Phys. 66, 3045-3050 (1977) and to match the QDK backend.
        # External eigenvalues are left unscaled.
        if len(internal_eigenvalues_list) > 0:
            internal_eigenvalues = np.array(internal_eigenvalues_list) * internal_scale_factor
            internal_eigenvectors = np.array(internal_eigenvectors_list).T  # Transpose for proper shape
            # Check internal stability: all eigenvalues should be > stability_tol
            internal_stable = np.all(internal_eigenvalues > stability_tol)
        else:
            internal_eigenvalues = np.array([])
            internal_eigenvectors = np.array([]).reshape(0, 0)

        if len(external_eigenvalues_list) > 0:
            external_eigenvalues = np.array(external_eigenvalues_list)
            external_eigenvectors = np.array(external_eigenvectors_list).T  # Transpose for proper shape
            # Check external stability: all eigenvalues should be > stability_tol
            external_stable = np.all(external_eigenvalues > stability_tol)
        else:
            external_eigenvalues = np.array([])
            external_eigenvectors = np.array([]).reshape(0, 0)

        result = StabilityResult(
            internal_stable,
            external_stable,
            internal_eigenvalues,
            internal_eigenvectors,
            external_eigenvalues,
            external_eigenvectors,
        )

        return (result.is_stable(), result)

    def name(self) -> str:
        """Return the name for the stability checker."""
        Logger.trace_entering()
        return "pyscf"
