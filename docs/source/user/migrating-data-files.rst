Migrating data files between serialization versions
===================================================

Each QDK/Chemistry data class versions its on-disk serialization schema
independently. A deserializer (:meth:`~qdk_chemistry.data.Orbitals.from_file`,
:meth:`~qdk_chemistry.data.Hamiltonian.from_file`,
:meth:`~qdk_chemistry.data.Wavefunction.from_file`,
:meth:`~qdk_chemistry.data.QpeResult.from_file`, and their ``from_json`` /
``from_hdf5`` counterparts) accepts **only** the serialization version the installed
library was built against. Loading a file written against an older version of that
class's schema raises an error that points back here.

The ``qdk_chemistry.migrate`` converter upgrades such a file to the serialization
version the installed library accepts. It migrates each data class point-for-point
along that class's own chain of serialization versions, and lives outside the core
data classes so that no legacy-schema knowledge leaks into the serialization code.

Command line
------------

The converter is shipped with the package and is the quickest way to upgrade a
single file:

.. code-block:: bash

   python -m qdk_chemistry.migrate old.hamiltonian.h5 new.hamiltonian.h5

The data type is taken from the ``name.type.ext`` filename convention
(``orbitals`` / ``hamiltonian`` / ``wavefunction`` / ``ansatz`` / ``qpe_result``) and the
serialization format from the file extension (``.json`` or ``.h5`` / ``.hdf5``).
The input and output formats may differ, so the same command also converts between
JSON and HDF5:

.. code-block:: bash

   python -m qdk_chemistry.migrate old.orbitals.json new.orbitals.h5

Python API
----------

The same conversion is available programmatically:

.. code-block:: python

   from qdk_chemistry import migrate

   migrate.convert_file("old.wavefunction.json", "new.wavefunction.json")

``migrate.convert_file`` raises ``migrate.MigrationError`` if the file is already
at the current serialization version, the source and destination are the same
file, the data type or format cannot be determined, or the conversion fails.

What is converted
-----------------

Each data class carries its own ordered migration steps. The steps currently
registered make the following changes:

- :class:`~qdk_chemistry.data.Orbitals` — molecular-orbital coefficients and
  energies are re-expressed as symmetry-blocked tensors. Active/inactive index
  sets, the AO overlap, and the basis set are carried across unchanged.
- :class:`~qdk_chemistry.data.Hamiltonian` — the four-center, Cholesky, and
  sparse containers all upgrade their integral storage to symmetry-blocked form.
- :class:`~qdk_chemistry.data.Wavefunction` — the single-determinant,
  complete-active-space, and selected-CI containers upgrade to the
  state-vector container; the MP2 and coupled-cluster containers upgrade to the
  amplitude container.
- :class:`~qdk_chemistry.data.Ansatz` — the embedded Hamiltonian and
  Wavefunction are each migrated through their own serialization-version chains.
- :class:`~qdk_chemistry.data.QpeResult` — the result fields are preserved while
   the obsolete evolution-time field is removed.

Data classes whose serialization schema has not changed (for example
:class:`~qdk_chemistry.data.Structure`, :class:`~qdk_chemistry.data.BasisSet`, and
the remaining qubit-level classes) load directly without conversion.

.. note::

   The converter does not support v1 ``TimeEvolutionUnitary`` files because they
   do not contain the scale required by the current
   :class:`~qdk_chemistry.data.UnitaryRepresentation` schema; regenerate these
   objects with v2.

Cholesky Hamiltonians
---------------------

An earlier Cholesky Hamiltonian container derived from the four-center container:
it stored the full four-center two-electron integrals and never persisted its
molecular-orbital three-center vectors. Such a container cannot be reconstructed
as a Cholesky representation, so the converter migrates it to a
:class:`~qdk_chemistry.data.CanonicalFourCenterHamiltonianContainer`, preserving
the integrals and dropping the now-unused AO Cholesky vectors. Re-run the
Cholesky decomposition from the orbitals if a Cholesky representation is needed
again.

A later Cholesky container stored the molecular-orbital three-center vectors
directly. Those vectors are the current Cholesky data model, so that container is
preserved as a :class:`~qdk_chemistry.data.CholeskyHamiltonianContainer`, with the
vectors re-expressed as a symmetry-blocked tensor. The two layouts are detected
automatically.

Supported formats
-----------------

Every supported data type converts between JSON and HDF5 in either direction;
the input and output formats are chosen independently from the file extensions.

Complex coefficients and complex active RDMs are not yet handled; such files
raise a ``migrate.MigrationError``.
