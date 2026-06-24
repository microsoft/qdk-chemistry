Migrating data files to the 2.0 schema
======================================

The on-disk serialization format for several QDK/Chemistry data classes changed
between the 1.x and 2.0 releases. To keep loading unambiguous, the 2.0
deserializers (:meth:`~qdk_chemistry.data.Orbitals.from_file`,
:meth:`~qdk_chemistry.data.Hamiltonian.from_file`,
:meth:`~qdk_chemistry.data.Wavefunction.from_file`, and their ``from_json`` /
``from_hdf5`` counterparts) accept **only** the current schema. Loading a file
written by an older release raises an error that points back here.

A standalone converter upgrades older files in place of guessing at load time.
It lives outside the core data classes so that no legacy-schema knowledge leaks
into the serialization code, and it is expected to be removed once 1.x files are
no longer in circulation.

Command line
------------

The converter is shipped with the package and is the quickest way to upgrade a
single file:

.. code-block:: bash

   python -m qdk_chemistry.migrate old.hamiltonian.h5 new.hamiltonian.h5

The data type is taken from the ``name.type.ext`` filename convention
(``orbitals`` / ``hamiltonian`` / ``wavefunction``) and the serialization format
from the file extension (``.json`` or ``.h5`` / ``.hdf5``). The input and output
formats may differ, so the same command also converts between JSON and HDF5:

.. code-block:: bash

   python -m qdk_chemistry.migrate old.orbitals.json new.orbitals.h5

Python API
----------

The same conversion is available programmatically:

.. code-block:: python

   from qdk_chemistry import migrate

   migrate.convert_file("old.wavefunction.json", "new.wavefunction.json")

``migrate.convert_file`` raises
``migrate.MigrationError`` if the file is already in the 2.0
schema, the data type or format cannot be determined, or the conversion fails.

What is converted
-----------------

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
  Wavefunction are each migrated in place.

Other data types (for example :class:`~qdk_chemistry.data.Structure`,
:class:`~qdk_chemistry.data.BasisSet`, and the qubit-level classes) did not
change format and load directly without conversion.

Cholesky Hamiltonians
---------------------

The 1.x release shipped a Cholesky Hamiltonian container that derived from the
four-center container: it stored the full four-center two-electron integrals and
never persisted its molecular-orbital three-center vectors. Such a container
cannot be reconstructed as a Cholesky representation, so the converter migrates
it to a
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
