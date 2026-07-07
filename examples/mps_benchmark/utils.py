"""
Shared utilities for MPS benchmark resource estimation scripts.

Contains:
  - MPS tensor loading
  - Fake Hamiltonian generation (for QPE)
  - QRE result serialization (Pareto front extraction)
  - QRE estimation runner
"""

import glob
import json
import time

import numpy as np
from scipy.sparse import load_npz
from qdk.qre import estimate
from qdk.qre.instruction_ids import LATTICE_SURGERY
from qdk.qre.property_keys import DISTANCE, NUM_TS_PER_ROTATION, PHYSICAL_COMPUTE_QUBITS
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction
from qdk_chemistry.data import FactorizedHamiltonianContainer, ModelOrbitals
from qdk_chemistry.utils import Logger


# ============================================================================
# MPS Loading
# ============================================================================

def load_mps_tensors(path_prefix):
    """Load MPS tensors from compressed .npz files.

    Args:
        path_prefix: Path prefix for tensor files (e.g. "dir/tensors_compressed_").
            Files are expected as {path_prefix}0.npz, {path_prefix}1.npz, etc.

    Returns:
        MPSWavefunction instance.
    """
    path_prefix = str(path_prefix)
    n_tensors = len(glob.glob(path_prefix + "[0-9]*.npz"))
    if n_tensors == 0:
        raise FileNotFoundError(f"No tensors found at {path_prefix}*.npz")
    sparse_tensors = [load_npz(f"{path_prefix}{i}.npz") for i in range(n_tensors)]
    dense_tensors = [t.toarray() for t in sparse_tensors]
    return MPSWavefunction(dense_tensors)


# ============================================================================
# Fake Hamiltonian for QPE
# ============================================================================

def make_fake_factorized_hamiltonian(N, R, B, C, seed=42):
    """Create a fake FactorizedHamiltonianContainer with given (N, R, B, C).

    The actual tensor values are random but the dimensions match real data,
    which is sufficient for resource estimation (gate counts depend on dimensions).
    """
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((N, N))
    h1 = (h1 + h1.T) / 2

    u_matrices = np.zeros(R * B * N)
    for ri in range(R):
        for bi in range(B):
            v = rng.standard_normal(N)
            v /= np.linalg.norm(v)
            u_matrices[ri * B * N + bi * N : ri * B * N + (bi + 1) * N] = v

    w_matrices = rng.standard_normal(R * B * C) * 0.1
    wb_matrix = rng.standard_normal((R, C)) * 0.1
    orbitals = ModelOrbitals(N)
    inactive_fock = np.zeros((N, N))

    return FactorizedHamiltonianContainer(
        R, B, C, 0.0, u_matrices, w_matrices, h1, wb_matrix, inactive_fock, orbitals,
    )


# ============================================================================
# QRE Result Serialization
# ============================================================================

def extract_qre_extras(qre_result):
    """Extract extra data from the best (last) QRE result entry.

    Returns a dict with: compute_distance, compute_qubits, num_ts_per_rotation,
    factories (list), and pareto_front (from serialize_pareto).
    """
    r = qre_result[-1]
    extra = {
        "compute_distance": int(r.source[LATTICE_SURGERY].instruction[DISTANCE]),
        "compute_qubits": int(r.properties[PHYSICAL_COMPUTE_QUBITS]),
        "num_ts_per_rotation": int(r.properties[NUM_TS_PER_ROTATION]),
        "factories": [],
        "pareto_front": serialize_pareto(qre_result),
    }
    for fid, factory_result in r.factories.items():
        extra["factories"].append({
            "copies": factory_result.copies,
            "runs": factory_result.runs,
            "error_rate": factory_result.error_rate,
            "states": factory_result.states,
        })
    return extra


def serialize_pareto(qre_result):
    """Extract Pareto front from QRE result into a JSON-serializable list.

    Adds columns: compute_distance, compute qubits, num_ts_per_rotation, factories.
    """
    qre_result.add_column(
        "compute_distance",
        lambda entry: entry.source[LATTICE_SURGERY].instruction[DISTANCE],
    )
    qre_result.add_column(
        "compute qubits",
        lambda entry: entry.properties[PHYSICAL_COMPUTE_QUBITS],
    )
    qre_result.add_column(
        "num_ts_per_rotation",
        lambda entry: entry.properties[NUM_TS_PER_ROTATION],
    )
    qre_result.add_factory_summary_column()
    df = qre_result.as_frame()

    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (np.integer,)):
                record[col] = int(val)
            elif isinstance(val, (np.floating,)):
                record[col] = float(val)
            elif isinstance(val, (np.bool_,)):
                record[col] = bool(val)
            elif hasattr(val, "total_seconds"):
                record[col] = val.total_seconds()
            else:
                record[col] = val if isinstance(val, (int, float, bool, str, type(None))) else str(val)
        records.append(record)
    return records


def run_qre(circuit, name, architecture, isa_query, trace_query, max_error):
    """Run QRE v3 estimation and return a result dict with Pareto front.

    Args:
        circuit: A compiled circuit with .get_qre_application() method.
        name: Display name for the QRE run.
        architecture: QRE architecture (e.g. Majorana).
        isa_query: ISA query (e.g. ThreeAux * RoundBasedFactory).
        trace_query: Trace query (e.g. PSSPC * LatticeSurgery).
        max_error: Maximum allowed error budget.

    Returns:
        Dict with keys: physical_qubits, runtime_ns, compute_distance,
        compute_qubits, num_ts_per_rotation, factories, pareto_front.
    """
    Logger.info("   Running QRE v3...")
    t0 = time.perf_counter()
    app = circuit.get_qre_application()
    qre_result = estimate(app, architecture, isa_query, trace_query, max_error=max_error, name=name)
    r = qre_result[-1]
    t_qre = time.perf_counter() - t0

    Logger.info(f"   Physical qubits: {r.qubits:,}")
    Logger.info(f"   Runtime: {r.runtime}")
    Logger.info(f"   QRE time: {t_qre:.1f}s")

    extra = extract_qre_extras(qre_result)
    extra["physical_qubits"] = r.qubits
    extra["runtime_ns"] = r.runtime

    return extra


def print_qre_table(qre_result, label=""):
    """Print the full QRE Pareto table (for verbose output)."""
    df = qre_result.as_frame()
    Logger.info(f"\n   --- Full QRE Results{' (' + label + ')' if label else ''} ---")
    Logger.info(df.to_string(index=False))
    Logger.info("")


# ============================================================================
# JSON Save
# ============================================================================

def save_results(results, path):
    """Save results dict to JSON file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    Logger.info(f"   [saved to {path}]")
