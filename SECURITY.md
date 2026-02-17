<!-- BEGIN MICROSOFT SECURITY.MD V1.0.0 BLOCK -->

# Security

Microsoft takes  and  othersthe security of our software products and services seriously, which
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
