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

Load serialized objects and execute remote job manifests only from trusted
producers. JSON and HDF5 files, cache filenames, content hashes, and job owner
metadata do not authenticate their writers. A content-hash comparison after
loading does not protect the native parser that reads the file.

Every principal with write access to a cache or job directory must be trusted
to supply results, serialized objects, and algorithm settings. This includes
results stored inline in JSON manifests. `is_shared=True` means storage is
reachable from client and worker machines; it does not isolate mutually
untrusted tenants.

Restrict write access to the submitting application and its trusted workers.
Use separate storage and access permissions for different trust groups, and
protect input and output directories throughout transfer and execution.
Do not use a common writable cache as a security boundary between tenants.
Deployments that require less-trusted writers need authentication before
deserialization, including protected verification keys provisioned separately
from the writable data.

Saved jobs retain backend configuration as metadata, but reconnect using trusted
runtime configuration. Attach an explicitly configured backend when restoring
a custom connection. This does not authenticate the saved job's remaining
metadata or payloads.

### Executable Circuits

QIR is executable program input, not a sandboxed interchange format. The QDK
full-state executor requires `allow_external_qir=True` for directly supplied
QIR and QIR restored from serialized circuits. Enable it only for trusted
producers. This opt-in neither authenticates inputs nor isolates their execution.
OpenQASM and Q# factory circuits compiled internally do not require the opt-in.
