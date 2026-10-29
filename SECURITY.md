<!-- BEGIN MICROSOFT SECURITY.MD V1.0.0 BLOCK -->

# Security

Microsoft takes the security of our software products and services seriously, which
includes all source code repositories in our GitHub organizations.

**Please do not report security vulnerabilities through public GitHub issues.**

For security reporting information, locations, contact information, and policies,
please review the latest guidance for Microsoft repositories at
[https://aka.ms/SECURITY.md](https://aka.ms/SECURITY.md).

<!-- END MICROSOFT SECURITY.MD BLOCK -->

## Secure Usage

### Data Serialization

QDK/Chemistry data classes (`Structure`, `Hamiltonian`, `Wavefunction`, `Orbitals`,
`BasisSet`, and others) support Python's
[pickle](https://docs.python.org/3/library/pickle.html) protocol for internal use
such as multiprocessing and deep copy. **Do not unpickle data from untrusted or
unverified sources**, as deserializing malicious pickle data can lead to arbitrary
code execution.

For data persistence and exchange, use the provided JSON and HDF5 serialization
methods (e.g., `to_json_file()` / `from_json_file()`, `to_hdf5_file()` /
`from_hdf5_file()`) instead.

### Shared Caches and Remote Jobs

The caller and deployment operator are responsible for deciding which producers
and writers to trust and for controlling write access. QDK/Chemistry does not
authenticate writers or verify their trustworthiness. JSON and HDF5 files,
cache filenames, content hashes, and job owner metadata do not establish that
trust. A content-hash comparison after loading does not protect the native
parser that reads the file.

A writer to a cache or job directory can forge results, including inline JSON
values, replace serialized objects, or select algorithms and settings for a
worker to execute. `is_shared=True` means storage is reachable from client and
worker machines; it does not verify access permissions or isolate mutually
untrusted tenants.

Operators are responsible for restricting write access to their chosen
submitters and workers, separating different trust groups, and protecting
input/output directories during transfer and execution. Applications accepting
less-trusted writers are responsible for authenticating manifests and payloads
before deserialization and protecting verification keys separately from the
writable data. QDK/Chemistry does not provide that authentication protocol.

Saved jobs retain backend configuration as metadata, but reconnect using current
runtime configuration. Callers choose that configuration or attach an explicitly
configured backend for a custom connection. Its trustworthiness is not verified;
it determines the destinations of authenticated requests. The saved job's
remaining metadata and payloads are not authenticated either.

### Executable Circuits

QIR is executable program input, not a sandboxed interchange format. The QDK
full-state executor requires `allow_external_qir=True` for directly supplied
QIR and QIR restored from serialized circuits. Enabling it permits QIR from
any external source without verifying provenance or safety. The caller is
responsible for deciding whether to trust the producer; malicious QIR can
compromise the executing process. This opt-in neither authenticates inputs nor
isolates their execution. OpenQASM and Q# factory circuits compiled internally
do not require the opt-in.

### Stub Generation

The optional import-time stub generator is resolved before the target package
directory is added to module lookup, and runs from a private temporary directory.
The caller and deployment operator are responsible for trusting installed
dependencies, interpreter search paths, and package code, and for controlling
who can modify them. The generator helper does not verify that trust or those
permissions; modifications can execute code with the importing application's
privileges.
