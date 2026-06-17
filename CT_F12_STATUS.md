# CT-F12 Implementation Status

**Delivered:** F12-HF reference energy, validated to max precision against the
canonical transcorrelated reference (Comment on J. Chem. Phys. 136, 084107,
arXiv:2211.09685v2, Table I).

## Validation (neon atom, frozen-core, gamma = 1.5)

`f12_hf_scf_energy` builds the full dressed transcorrelated Hamiltonian over the
orbital basis and relaxes the closed-shell orbitals in its mean field. The
converged correction reproduces the reference F12-HF energies to the precision
they are tabulated:

| Basis        | E(F12-HF) - E_HF | Reference     | Error    |
|--------------|------------------|---------------|----------|
| aug-cc-pVDZ  | -0.1115550802    | -0.111555079  | -1.2e-9  |
| aug-cc-pVTZ  | -0.0428456402    | -0.042845640  | -2.2e-10 |
| aug-cc-pVQZ  | -0.0199399898    | -0.019939990  | +2.4e-10 |

All three bases match the Comment Table I to ~1e-9 (the references are quoted to
9 decimals), i.e. true max precision.

### Root cause of the former aDZ 2.4e-7 residual: spurious 1/2 in Eq. 27

An earlier version reproduced aDZ only to 2.4e-7 (aTZ/aQZ already at 1e-10). It
was decomposed to rule out HF orbitals (bit-identical to MPQC, +4.6e-10 at aDZ),
CABS (same aug-cc-pVXZ/OptRI; aDZ CABS space unambiguous, no truncation), geminal
integrals (libint `Operator::stg` = genuine Ten-no Slater factor, same backend as
MPQC), and SCF convergence/DIIS (the dressed H is Hermitian; MPQC's value is its
fully-converged minimum, 2.4e-7 *above* ours -- we converged lower). That
localized the difference to the dressed-Hamiltonian assembly.

The MPQC reference driver (`ValeevGroup/mpqc4` `test_uccf12.cpp`) revealed it: the
S intermediate of paper Eq. 27 carries a **spurious 1/2 prefactor**; MPQC removes
it under `S_prefactor_error=false`, giving the Comment's -128.607904810 (our prior
value matched MPQC's *uncorrected* -128.6079050465 to 1e-9). We had implemented
Eq. 27 verbatim; doubling the S intermediate (`ctf12_f12.cpp`, "remove the
spurious 1/2") closes aDZ to 1.2e-9 while leaving aTZ/aQZ unchanged (S is
negligible once the CABS is well-resolved). S enters only the virtual blocks of
C1bar/C2bar, so the first-order energy is unaffected.

## Key result: F12-HF is SCF-converged, not first-order

The "F12-HF" energy tabulated in the reference is the **self-consistent** energy
of the dressed transcorrelated Hamiltonian, not the first-order expectation
value `<HF|Hbar|HF>`. The geminal amplitudes are fixed from the original HF
orbitals; the dressed Hamiltonian `Hbar = H + [H,A] + 1/2[[F,A],A]` is built once
(paper Eqs. 14-28) and the orbitals are then relaxed by Roothaan SCF in its
mean field. The orbital relaxation (dominated by the U intermediate) closes the
~4e-4 gap between the first-order estimate and the reference.

## Milestones

### M0: Feasibility

- Papers vendored: Kong et al. 2012, arXiv Comment 2211.09685v2, Kedzuch et al. 2005
- Core physics equations identified and mapped

### M1: Infrastructure

- **Geminal integrals:** `stg_geminal_eri` (STG / STG x Coulomb over 4 distinct basis sets)
- **4-index MO transform:** `mo_transform_4index` with 4 independent coefficient matrices
- **Coulomb integrals:** `four_center_coulomb`, `kinetic_matrix`, `nuclear_matrix`
- **CABS construction:** SVD-based orthogonalization (`build_cabs`)
- **OptRI basis data:** aug-cc-pv{d,t,q}z-optri vendored from BSE

### M2: F12-HF physics (validated to max precision)

- **Intermediates:** V (Coulomb-dressed geminal), X (geminal metric), B (8-term
  approximation-C, Hermitized), all SP (cusp) coupled.
- **Dressed Hamiltonian:** C1bar / C2bar (paper Eqs. 15-22) including the U
  (Eqs. 21, 26) and S (Eqs. 22, 27) intermediates.
- **Self-consistent F12-HF:** Roothaan SCF with `(hbar, gbar)`; validated above.

## Public API (`ctf12_f12.hpp`)

- `build_intermediates(input)` -> diagonal V/X/B over the valence space.
- `f12_hf_energy(intermediates)` -> first-order `<HF|Hbar|HF> - E_HF`, the
  inexpensive "standard" estimate (selectable option).
- `f12_hf_scf_energy(input)` -> the self-consistent F12-HF correction (the
  quantity tabulated as "F12-HF" in the reference literature).

## Test coverage

- `test_ctf12_f12_hf.cpp`: F12-HF SCF correction for Ne / aug-cc-pV{D,T,Q}Z
  against the reference, plus the first-order "standard" estimate for aDZ.
- `test_geminal_eri.cpp`, `test_cabs.cpp`, `test_optri_basis.cpp`: integral,
  CABS, and OptRI-basis infrastructure.
- Python: factory, settings, backend registration.

## Remaining work (out of scope for this milestone)

### Multi-determinant extension

- Wire the dressed `hbar` / `gbar` into `ctf12_hamiltonian::_run_impl`.
- Emit a dressed `data::Hamiltonian` for CASCI/CASSCF/SCI references.
- Test with a multi-determinant `StateVectorContainer`.

## References

- Kong, Yanai, Shiozaki, J. Chem. Phys. **136**, 084107 (2012) -- CT-F12 theory
- arXiv:2211.09685v2 (Comment on Kong 2012) -- frozen-core conventions, reference energies
- Kedzuch, Milko, Noga, Int. J. Quantum Chem. **105**, 929 (2005) -- approximation C
- ValeevGroup/mpqc4 `tests/unit/test_uccf12.cpp` -- reference F12-HF SCF driver
