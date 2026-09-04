# Test-only Q# sources

Q# drivers that exist purely to exercise the shipped `qdk_chemistry.utils.qsharp` project.
They live here rather than in `src/` because Q# `internal` is package-scoped, not a privacy
boundary: everything in the shipped project is reachable from Python on `context.code`,
`internal` included, so a driver left there is public API a user can call.

These are source, not fixture data, which is why they sit here rather than under
`tests/test_data/`.

## Adding a module

The `qsharp_test_context` fixture in `tests/conftest.py` globs `*.qs` from this directory
and evaluates each onto a fresh context, so a new file needs no registration. It does need:

- **The `.qs` extension, in this directory.** Nothing else is discovered.
- **A namespace under `QDKChemistry.TestUtils`.** The `qsharp_test_utils` fixture returns
  that namespace, so anything declared outside it is invisible to the tests.
- **A `Test` prefix on every declaration**, enforced by
  `test_qsharp_context.py::TestTestOnlySourcesAreNotShipped`.

Drivers may call `internal` callables of the shipped project, because `eval` runs them
inside that package. Ops compose only within the context they came from, so a test mixing
library and test-only Q# must take both from `qsharp_test_context`.
