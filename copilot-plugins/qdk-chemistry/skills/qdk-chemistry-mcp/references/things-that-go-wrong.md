# Things That Go Wrong

Real failure modes from the source code and examples, with what to do about them.

## General Error Recovery Principles

When a step fails, follow this order:

1. **Read the error message** — QDK Chemistry tools return specific, actionable messages. Don't skip them.
2. **Classify the failure** — distinguish invalid inputs or incompatible orbitals from a transient remote-service, transport, or artifact-retrieval failure.
3. **Make at least one recovery attempt** — correct a diagnosed input or method mismatch before rerunning; retry a transient remote operation once without changing the scientific settings.
4. **Try a simpler compatible configuration** — smaller basis set, smaller active space, or fewer determinants. Preserve any downstream requirement for restricted orbitals.
5. **Try an alternative compatible method** — change the initial guess or convergence settings before changing the reference type. Do not switch to UHF when the result must feed valence selection, ASCI, or AutoCAS.
6. **Report clearly** — state each attempt, job ID, exact error, and changed parameter. Declare a state inconclusive only after the required recovery attempt also fails.

Common recovery paths by failure type:

| Failure | First try | Then try |
|---------|-----------|----------|
| SCF won't converge before ASCI/AutoCAS | Different guess, damping, or level shifting while retaining restricted HF | Smaller basis set or active space |
| Active space too large | `qdk_valence` pre-filter | Reduce to frontier orbitals only |
| SCI doesn't converge | Increase `davidson_iterations` | Smaller active space |
| State prep circuit too deep | Fewer determinants (top 2-5) | Smaller active space |
| QPE takes too long | Fewer `num_bits`, fewer Trotter steps | Reduce active space upstream |
| Remote job fails transiently | Re-check logs/status, then resubmit once | Report both job IDs and errors |
| Remote result artifact is unavailable | Re-check the job and retry retrieval once | Resubmit the producing step once |

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

1. If the orbitals feed valence selection, ASCI, or AutoCAS, retain a restricted HF reference; use ROHF for `spin_multiplicity > 1`
2. Try a different initial guess
3. Try damping, level shifting, or a different basis set
4. Use UHF only for a path that does not require restricted orbitals downstream
5. Report the instability — don't silently continue

**Always run stability check.** An unstable SCF cascades errors through every subsequent step.

---

## Unrestricted Orbitals Sent to ASCI or AutoCAS

**Symptom:** `qdk_valence` reports `ValenceActiveSpaceSelector only supports restricted orbitals`, or a later ASCI/AutoCAS stage rejects an open-shell SCF wavefunction.

**Cause:** The default `run_scf` setting `scf_type="auto"` chooses UHF for `spin_multiplicity > 1`. UHF has separate alpha and beta orbitals, while this active-space pipeline requires one shared set of spatial orbitals.

**Fix:** Run a separate HF reference on the structure with:

```json
{"method": "hf", "scf_type": "restricted"}
```

For an open-shell state this produces ROHF; for a closed-shell state it produces RHF. Feed that wavefunction to valence selection and the ASCI/AutoCAS workflow. A preceding DFT geometry optimization may still provide the structure, but do not pass an unrestricted DFT or UHF wavefunction into the restricted-orbital pipeline.

If the failure already occurred, rerun SCF with these settings and retry the failed selector or MR stage at least once before declaring the state inconclusive.

---

## Remote Failures

**Symptom:** A remote job fails for an apparently recoverable service reason, or a succeeded job cannot be retrieved because an output artifact is temporarily unavailable.

**Fix:** A first basic remote failure is not a terminal scientific result.

1. Inspect `check_remote_job` status, logs, and error text.
2. For a succeeded job with a retrieval failure, retry `retrieve_remote_results` once for the same job ID. If the artifact is still absent, resubmit the producing `run_*` step once with the same scientific settings.
3. For a transient execution or service failure, resubmit the step once with the same scientific settings.
4. For a deterministic configuration error, correct the diagnosed issue first. Examples include regenerating an ROHF reference after a restricted-orbital error and verifying charge, electron count, and spin multiplicity after an invalid spin/charge error.
5. Stop after the recovery attempt repeats the failure or exposes a non-recoverable authentication, permission, quota, or unsupported-method error.

Record every attempt and job ID. Do not silently alter charge, multiplicity, basis, active space, or the requested scientific endpoint merely to make a retry pass.

---

## QPE With Invalid Defaults

**Symptom:** `run_phase_estimation` returns an error about `num_bits` or `evolution_time`.

**Cause:** The tool has intentionally invalid nested defaults: `settings.qpe_circuit_builder.num_bits=-1` and `settings.qpe_circuit_builder.unitary_builder.time=0.0`. This is by design — these values depend on the problem and must be set explicitly.

**Fix:** Set both in the nested `settings` dict:

```json
{
	"qpe_circuit_builder": {
		"num_bits": 12,
		"unitary_builder": {"algorithm_name": "trotter", "time": 1.0}
	}
}
```

There is no public `compute_evolution_time()` helper. Compute `evolution_time`
from target energy bounds or, by default, from the mapped qubit-Hamiltonian
coefficient 1-norm with `energy_window = 2 * sum(abs(coefficients))`. If no
campaign policy is provided, use the default QPE policy in
`qpe-and-state-prep.md`.

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

**Fix:** Present available logical circuit metrics from `get_circuit_stats` and physical-qubit/runtime/error Pareto points from `run_resource_estimation`. If metrics aren't available, identify them as missing instead of making up numbers.

---

## Convergence Failures by System Type

### Open-shell molecules (O₂, NO, radicals)

SCF may converge to the wrong spin state. Verify the requested multiplicity. Use ROHF when the wavefunction will feed valence selection, ASCI, or AutoCAS; reserve UHF for workflows that accept unrestricted orbitals.

### Transition metal complexes (Fe, Co, Ni, Cu, Mn)

Near-degenerate d-orbitals cause SCF oscillation. Try level shifting, ROHF instead of UHF, or start with a smaller/simpler basis set.

### Stretched bonds (dissociation curves)

Beyond ~1.5× equilibrium bond distance, single-reference methods break down. UHF can diagnose symmetry breaking, but an ASCI/AutoCAS path still needs a separate restricted HF reference plus a sufficiently large active space.

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
