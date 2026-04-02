"""QpeResult usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create-from-phase
from qdk_chemistry.data import QpeResult

# Construct a QpeResult from a measured phase fraction
result = QpeResult.from_phase_fraction(
    method="iterative",
    phase_fraction=0.423828125,
    evolution_time=0.1,
    bits_msb_first=(0, 1, 1, 0, 1, 1, 0, 0, 1, 0),
    bitstring_msb_first="0110110010",
    reference_energy=-1.137,
)
# end-cell-create-from-phase
################################################################################

################################################################################
# start-cell-inspect
# Inspect the result
print(f"Method: {result.method}")
print(f"Phase fraction: {result.phase_fraction:.6f}")
print(f"Phase angle: {result.phase_angle:.6f} rad")
print(f"Raw energy: {result.raw_energy:.8f} Ha")
print(f"Alias candidates: {result.branching}")
print(f"Resolved energy: {result.resolved_energy:.8f} Ha")
print(f"Measured bits: {result.bits_msb_first}")

# Full summary
print(result.get_summary())
# end-cell-inspect
################################################################################

################################################################################
# start-cell-alias
# Construct without a reference energy — no alias resolution
result_no_ref = QpeResult.from_phase_fraction(
    method="iterative",
    phase_fraction=0.423828125,
    evolution_time=0.1,
)

# All alias candidates are available in the branching tuple
for i, energy in enumerate(result_no_ref.branching):
    print(f"  Candidate {i}: {energy:.6f} Ha")

# Resolve later using a reference energy
from qdk_chemistry.utils.phase import resolve_energy_aliases

resolved = resolve_energy_aliases(
    result_no_ref.raw_energy,
    evolution_time=0.1,
    reference_energy=-1.137,
)
print(f"Resolved energy: {resolved:.8f} Ha")
# end-cell-alias
################################################################################

################################################################################
# start-cell-serialization
import os
import tempfile

tmpdir = tempfile.mkdtemp()

# Save to JSON
result.to_json_file(os.path.join(tmpdir, "result.qpe_result.json"))

# Load from JSON
loaded = QpeResult.from_json_file(os.path.join(tmpdir, "result.qpe_result.json"))

# Save to HDF5
result.to_hdf5_file(os.path.join(tmpdir, "result.qpe_result.h5"))

# Load from HDF5
loaded_h5 = QpeResult.from_hdf5_file(os.path.join(tmpdir, "result.qpe_result.h5"))
# end-cell-serialization
################################################################################
