---
name: qdk-chemistry-coding
version: 'v2.0.0'
description: 'Describes the Python API for discovering, configuring, and executing QDK Chemistry algorithms.'
---

# QDK Chemistry Python API

Algorithms are obtained from the registry:

- `available(algorithm_type)` returns registered implementations.
- `show_default(algorithm_type)` returns the registered default.
- `create(algorithm_type, algorithm_name, **settings)` creates an algorithm.
- `algorithm.settings()` returns its settings object.
- `algorithm.run(...)` executes it.
- `algorithm.hash(...)` computes the content hash for a call.
- `algorithm.on_remote(...)` returns a remote-bound algorithm.

Algorithm `run` signatures and data-class APIs define accepted inputs and
returned objects. Coordinates supplied to `Structure` are in Bohr. The
registry defines installed algorithm types, implementation names, settings,
and defaults.

## Settings

`settings()` returns typed setting entries. `get(name)` reads a value and
`set(name, value)` changes it before execution. Algorithm references encode a
nested algorithm selection and its nested settings. Validation errors identify
unknown setting names or incompatible values.

## Execution and Results

Each algorithm class defines its own `run(...)` signature. Inputs include data
objects and scalar configuration such as charge or multiplicity where required.
Returns may be a data object, a scalar, or a tuple. The API documentation for
the selected class defines the exact return shape.

Data objects support JSON or HDF5 serialization according to their class API.
Serialized files can be loaded and passed to later calls that accept the same
data type.

## Content Hashes

Algorithms and data objects expose deterministic content hashes. An algorithm
call hash includes the implementation, settings, and inputs. Cache backends use
that identity to associate a call with serialized outputs.

## Remote Binding

`on_remote(backend, **configuration)` retains the algorithm configuration and
routes execution through a remote backend. Blocking execution returns the same
result shape as local execution. Submitted execution returns a job handle whose
state and outputs can be retrieved later.
