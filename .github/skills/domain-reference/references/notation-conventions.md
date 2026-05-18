# Notation and Unit Conventions

## Unit Conventions

- **Coordinates**: All internal coordinates are in **Bohr** (atomic units of length). Angstrom conversion happens only at I/O boundaries (e.g., reading/writing XYZ files).
- **Energies**: All energies are in **Hartree** (atomic units of energy).

## Two-Body Integral Indexing

Two-body (two-electron) integrals use **chemist notation** `(ij|kl)`.

When stored as a flattened array, the index is computed as:

```
idx = i * norb³ + j * norb² + k * norb + l
```

where `norb` is the number of orbitals.

## Spin-Orbital Ordering

QDK uses **blocked** spin-orbital ordering:

```
[α₀, α₁, ..., αₙ, β₀, β₁, ..., βₙ]
```

All alpha spin-orbitals come first, followed by all beta spin-orbitals.

OpenFermion uses **interleaved** spin-orbital ordering:

```
[α₀, β₀, α₁, β₁, ..., αₙ, βₙ]
```

Alpha and beta spin-orbitals alternate.

**Conversion is required** at plugin boundaries — the OpenFermion plugin handles index permutation automatically when converting between representations.

## Pauli String Encoding

Pauli strings use **little-endian** qubit ordering: qubit 0 is the **rightmost** character in the string representation.

For example, in a 4-qubit system, the string `"IXYZ"` means:
- Qubit 0: `Z` (rightmost)
- Qubit 1: `Y`
- Qubit 2: `X`
- Qubit 3: `I` (leftmost)

This is consistent with standard circuit conventions where qubit 0 is at the bottom of circuit diagrams.

## 2-RDM Convention

Two-particle reduced density matrices (2-RDMs) use **chemist convention**:

```
Γ(p, q, r, s) = ⟨a†_p a†_r a_s a_q⟩
```

When stored as a flattened array, the index is computed as:

```
idx = p * n³ + q * n² + r * n + s
```

where `n` is the number of orbitals (or spin-orbitals, depending on context).
