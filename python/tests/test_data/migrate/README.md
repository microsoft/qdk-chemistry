# Migration regression fixtures

Real data files written by **qdk-chemistry 1.0.0** (schema version `0.1.0`),
used to validate the `qdk_chemistry.migrate` converter against ground-truth
output from the original serializer rather than hand-built fixtures.

All fixtures are H2 / STO-3G (2 molecular orbitals) to keep the committed files
small:

- `h2.orbitals.h5` — restricted orbitals with an embedded basis set (exercises
  the basis-set HDF5 sub-object extraction).
- `h2.hamiltonian.h5` — a four-center Hamiltonian over the active space.
- `h2_cas_rdm.wavefunction.h5` — a CAS wavefunction with spin-dependent active
  RDMs for a closed-shell system (only the alpha / opposite-spin channels are
  stored; the converter aliases the omitted beta channels).
- `reference.json` — spot-check values extracted with the 1.0.0 reader.

These were produced by running an SCF -> valence active space -> CASCI workflow
on H2 with the 1.0.0 Python package (the CAS wavefunction's spin-dependent RDMs
were attached explicitly). Regenerating them requires a 1.0.0 build.
