# Migration regression fixtures

Real data files written by **qdk-chemistry 1.0.0** (schema version `0.1.0`),
used to validate the `qdk_chemistry.migrate` converter against ground-truth
output from the original serializer rather than hand-built fixtures.

All fixtures are small H2 systems, and the producing release is encoded in each
filename (``v1_0_0`` / ``v1_1_0``). The Cholesky and Sparse containers did not
exist at 1.0.0, so those two were produced with a 1.1.0 build; the rest are
H2 / STO-3G from 1.0.0:

- `h2_v1_0_0.orbitals.h5` — restricted orbitals with an embedded basis set (exercises
  the basis-set HDF5 sub-object extraction).
- `h2_v1_0_0.hamiltonian.h5` — a four-center Hamiltonian over the active space.
- `h2_v1_0_0_cas_rdm.wavefunction.h5` — a CAS wavefunction with spin-dependent active
  RDMs for a closed-shell system (only the alpha / opposite-spin channels are
  stored; the converter aliases the omitted beta channels).
- `h2_v1_0_0.ansatz.h5` — an Ansatz wrapping the Hamiltonian and a CAS wavefunction
  (both embedded payloads are migrated in place).
- `h2_v1_1_0_chol.hamiltonian.h5` — a Cholesky container (which stored the full
  four-center two-body tensor plus AO Cholesky vectors, not MO three-center
  vectors); the converter migrates it to a four-center container.
- `h2_v1_1_0_sparse.hamiltonian.h5` — a sparse lattice-model Hamiltonian
  (HDF5 packs the sparse indices into the bytes of a double via `pack_indices`).
- `reference.json` — spot-check values extracted with the 1.0.0 / 1.1.0 readers.

These were produced by running an SCF -> valence active space -> CASCI workflow
on H2 with the 1.0.0 Python package (the CAS wavefunction's spin-dependent RDMs
were attached explicitly). The Cholesky/Sparse fixtures require a 1.1.0 build
(Cholesky via the `qdk_cholesky` constructor with `store_cholesky_vectors=True`;
Sparse via a small lattice model). Regenerating requires the matching 1.x build.
