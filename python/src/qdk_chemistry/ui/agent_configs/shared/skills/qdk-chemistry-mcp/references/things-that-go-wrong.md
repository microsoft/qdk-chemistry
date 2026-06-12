# Things That Go Wrong

Real failure modes from the source code and examples, with what to do about them.

## General Error Recovery Principles

When a step fails, follow this order:

1. **Read the error message** — QDK Chemistry tools return specific, actionable messages. Don't skip them.
2. **Check inputs** — wrong coordinate units, missing files, and mismatched active spaces cause most failures.
3. **Try a simpler configuration first** — smaller basis set, smaller active space, fewer determinants. If that works, scale back up.
4. **Try an alternative method** — if HF doesn't converge, try UHF or a different initial guess. If CASCI fails, try a smaller active space.
5. **Report clearly** — state what failed, the error, what was tried, and what the user could change. Don't silently switch approaches.

Common recovery paths by failure type:

| Failure | First try | Then try |
|---------|-----------|----------|
| SCF won't converge | Different basis set (smaller) | Level shifting, damping, UHF instead of RHF |
| Active space too large | `qdk_valence` pre-filter | Reduce to frontier orbitals only |
| SCI doesn't converge | Increase `davidson_iterations` | Smaller active space |
| State prep circuit too deep | Fewer determinants (top 2-5) | Smaller active space |
| QPE takes too long | Fewer `num_bits`, fewer Trotter steps | Reduce active space upstream |

---

## Wrong Coordinate Units

**Symptom:** SCF energy is wildly wrong, or convergence fails immediately on a molecule that should be easy.

**Cause:** `create_structure` expects Bohr but received Angstrom. The molecule ends up ~1.89× too large.

**Fix:** Multiply Angstrom coordinates by 1.8897259886 to get Bohr. Use `convert_coordinates` if available. Always double-check — most chemistry papers, PDB files, and molecular editors use Angstrom.

---

## Unstable SCF Solution

**Symptom:** `run_stability_checker` reports instability. Downstream calculations give wrong energies.

**Cause:** SCF converged to a saddle point instead of the true minimum. Common for open-shell systems (O₂, radicals), transition metals, stretched bonds, and near-degenerate orbitals.

**Fix:**

1. Try a different SCF method (UHF ↔ ROHF)
2. Try a different initial guess
3. Use a different basis set
4. Report the instability — don't silently continue

**Always run stability check.** An unstable SCF cascades errors through every subsequent step.

---

## QPE With Invalid Defaults

**Symptom:** `run_phase_estimation` returns an error about `num_bits` or `evolution_time`.

**Cause:** The tool has intentionally invalid defaults: `num_bits=-1` and `evolution_time=0.0`. This is by design — these values depend on the problem and must be set explicitly.

**Fix:** Set both in the `settings` dict:

```json
{"num_bits": 10, "evolution_time": <computed_value>}
```

Use `compute_evolution_time()` (Python package) or compute from the Hamiltonian's spectral norm (MCP).

---

## AutoCAS on Bare SCF Wavefunction

**Symptom:** Active space selector fails or produces garbage when using `qdk_autocas` directly after SCF.

**Cause:** AutoCAS needs reduced density matrices (RDMs) from a multi-configuration calculation. A bare SCF wavefunction has no RDMs.

**Fix:** Either:

- Use `qdk_valence` for initial selection from SCF (no RDMs needed), OR
- Run SCI first with `calculate_one_rdm=True` and `calculate_two_rdm=True`, then use AutoCAS on the SCI result

---

## Confusing Resource Analysis with Energy Computation

**Symptom:** User asked "how many qubits" and got an energy number, or asked "compute the energy" and got only a gate count.

**Cause:** These are different endpoints that use different tools. Building a circuit and analyzing its resources (qubits, depth, T-count) is NOT the same as executing QPE to get an energy.

**Fix:** Listen to what the user actually asked:

- "qubits", "resources", "cost", "feasibility" → build circuit, extract resource profile, stop
- "energy", "eigenvalue", "QPE" → configure and run full phase estimation
- "SCF energy", "classical" → stop after classical calculation, no circuits

Don't switch between them. If one fails, report the failure — don't silently try the other as a fallback.

---

## Incomplete Resource Estimates

**Symptom:** Agent reports "12 qubits needed" as the resource estimate.

**Cause:** A qubit count alone is misleading. The actual computational cost is dominated by T-gates, not qubits.

**Fix:** Extract and present the full profile: logical qubits, circuit depth, total gate count, Clifford gates, T-count, T-depth. Present as a table. If some metrics aren't available, note which are missing — don't make up numbers.

---

## Convergence Failures by System Type

### Open-shell molecules (O₂, NO, radicals)

SCF may converge to the wrong spin state. Use UHF, verify spin multiplicity matches the physical ground state (e.g., O₂ is triplet, not singlet).

### Transition metal complexes (Fe, Co, Ni, Cu, Mn)

Near-degenerate d-orbitals cause SCF oscillation. Try level shifting, ROHF instead of UHF, or start with a smaller/simpler basis set.

### Stretched bonds (dissociation curves)

Beyond ~1.5× equilibrium bond distance, single-reference methods break down. Use UHF (not RHF), a large active space, and multi-reference methods.

### Near-degenerate systems (HOMO-LUMO gap < 1 eV)

SCF convergence failure or wrong orbital ordering. Try temperature smearing or an alternative initial guess.

---

## Basis Set Mismatches

| Basis | Speed | Quality | Appropriate for |
|---|---|---|---|
| STO-3G | Very fast | Qualitative | Testing workflows, debugging, quick checks |
| 6-31G / 6-31G* | Fast | Semi-quantitative | Surveys, large molecules |
| cc-pVDZ | Moderate | Good | Production calculations, main-group (used in most real examples) |
| cc-pVTZ | Slow | High | Benchmark accuracy, small molecules |
| aug-cc-pVDZ | Moderate | Good for anions | Diffuse character, anionic systems |

If the user asks for "accurate" results and the plan uses STO-3G, flag it — STO-3G is for testing, not production.

---

## Missing `calculate_mutual_information`

**Symptom:** Entanglement visualization (`visualize_orbital_entanglement`) fails or shows nothing after running SCI.

**Cause:** The SCI was run with `calculate_one_rdm=True` and `calculate_two_rdm=True` but NOT `calculate_mutual_information=True`. The RDMs are enough for AutoCAS, but the entanglement visualization specifically needs the mutual information matrix.

**Fix:** When running SCI for active space refinement, always set all three: `calculate_one_rdm=True`, `calculate_two_rdm=True`, `calculate_mutual_information=True`.

---

## File Already Exists

**Symptom:** Tool returns `{"status": "exists"}` instead of running.

**Cause:** The output file from a previous run already exists. Tools never silently overwrite.

**Fix:** Use a different output filename, or delete the existing file and re-run.
