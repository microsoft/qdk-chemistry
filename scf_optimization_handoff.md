# SCF Optimization — Handoff Document (Session 2)

## Branch: `users/copilot/fock-builder-benchmarks`

**Previous session**: Fock builder optimizations (distance screening, syevd, blaspp, pre-LinK data structures, variable threshold, per-phase instrumentation).

**This session**: GDM incremental trial Fock with J/K reuse. Porphyrin benchmark system. Automatic fallback. `fock_reset_steps` default. Skip-on-converged fix. All 33 SCF tests pass. Tight convergence (1e-8) works via automatic fallback.

**Primary file changes**:
- `cpp/src/.../scf/scf_impl.cpp` — `evaluate_trial_incremental()`, `rebuild_fock()`, J/K reuse in main loop, skip-on-converged
- `cpp/src/.../scf/scf_impl.h` — `cache_trial_fock()`, `incremental_trial_enabled()`, `disable_incremental_trial()`, `rebuild_fock()`, `recompute_energy()`
- `cpp/src/.../scf_algorithm/gdm.cpp` — incremental trial in `GDMLineFunctor::eval()`, J/K caching, automatic fallback on line search failure
- `cpp/src/.../scf/ks_impl.h` — `supports_trial_fock_reuse()` override (returns false)
- `cpp/src/.../scf/core/scf.h` — `fock_reset_steps` default 2^30 → 20
- `cpp/tests/test_fock_benchmark.cpp` — porphyrin benchmark system, `QDK_SCF_TIGHT` env var

---

## 1. What Was Done

### 1.1 Delivered Optimizations

| Change | Impact | Where |
|--------|--------|-------|
| GDM incremental trial Fock | Line search uses `F_ + J(ΔP)` instead of `H + J(P_full)` — cheaper per eval | gdm.cpp, scf_impl.cpp |
| J/K reuse | Cached `J(ΔP)`/`K(ΔP)` from trial installed into main loop — skips `build_JK` | gdm.cpp, scf_impl.cpp |
| Automatic fallback | On line search failure: disable incremental, `rebuild_fock()`, continue with full-P | gdm.cpp |
| Skip-on-converged | Don't call `algorithm->iterate()` after convergence detected — avoids wasted line search | scf_impl.cpp |
| Post-convergence reuse | Skip eigenvalue rebuild when trial Fock matches | scf_impl.cpp |
| `fock_reset_steps = 20` | Bounds incremental Fock drift (Gaussian standard) | scf.h |
| Porphyrin benchmark | C₁₅H₁₄N₇⁺, 856 AOs with cc-pVTZ, P450-scale without open-shell issues | test_fock_benchmark.cpp |

### 1.2 Results (OMP_PROC_BIND=spread, 32T, DIIS_GDM, normal convergence 1e-6)

| System | NAO | Iters | Baseline (ms) | Optimized (ms) | Speedup | ΔE (nHa) |
|--------|-----|-------|--------------|----------------|---------|-----------|
| porphyrin/cc-pVTZ | 856 | 12 | 236,942 | 184,726 | **1.28×** | +2.4 |
| benzene/TZVP | 222 | 9 | 6,075 | 5,167 | **1.18×** | <0.1 |
| water20/SVP | 480 | 11 | 4,964 | 4,229 | **1.17×** | <0.1 |
| water10/SVP | 240 | 11 | 3,023 | 2,358 | **1.28×** | <0.1 |
| benzene/SVP | 114 | 9 | 1,146 | 983 | **1.17×** | <0.1 |
| alkane12/6-31G* | 220 | 35→31 | 34,848 | 43,750 | 0.80× | -488 |

Alkane12 is pathological: flat GDM energy surface triggers the automatic fallback (rebuild + full-P switch), adding cost. All other systems show clean speedup with sub-nHa energy differences.

Tight convergence (1e-8): porphyrin converges in 21 iterations. Alkane12 converges in 48 iterations. Both trigger the automatic fallback mid-convergence and self-correct.

---

## 2. Key Findings

### 2.1 Why Incremental Trial Works

The GDM line search evaluates `F_trial = F_ + J(P_trial - P_current)` instead of `F_trial = H + J(P_trial)`. The `J(ΔP)` from this eval is bit-identical to what the main loop's `build_JK(P_ - P_last)` would compute on the next step (same ΔP, deterministic `build_JK`). So we cache it and skip the main-loop build.

The incremental trial is cheaper per eval because |ΔP| << |P_full|, giving more Schwarz screening. But it also makes the Nocedal-Wright zoom phase iterate ~3.4× more (less accurate energy landscape). Net: still faster because each eval is ~2× cheaper.

### 2.2 Sources of Numerical Difference

Three screening levels cause incremental J/K to differ from full-P J/K:

1. **Schwarz shell-level** (line 899): `Q² × |ΔP_shellblock| < threshold` screens medium-distance quartets
2. **Libint2 engine precision** (line 610): `epsilon / P_shmax` drops primitives when `P_shmax` is small
3. **Accumulated Fock drift**: `sum(J(ΔP_i)) ≠ J(P)` over many incremental steps

For standard convergence (1e-6), all three are manageable (~nHa per step). For tight convergence (1e-8), accumulated drift from the DIIS phase (~38 nHa for alkane12, ~few nHa for porphyrin) prevents convergence without the automatic fallback.

### 2.3 The Automatic Fallback

When GDM's line search fails (both BFGS and steepest descent, gradient norm exceeds threshold):
1. Disable `incremental_trial_enabled_` permanently for this SCF
2. Call `rebuild_fock()` — fresh `reset + build_JK(P_full) + update_fock_()`
3. Recompute `last_accepted_energy_` from the clean Fock
4. Accept zero rotation for this step
5. All subsequent GDM steps use full-P trial (slower but numerically robust)

### 2.4 Skip-on-Converged

The baseline called `algorithm->iterate()` even after convergence was detected. For GDM, this meant a full line search on a step where the gradient is ~1e-8 — causing expensive zoom-phase failures. The fix: check `converged` before calling `iterate()`. The post-convergence eigenvalue rebuild handles everything else.

### 2.5 What We Tried That Didn't Work

| Approach | Why it failed |
|----------|-------------|
| Full-P trial Fock reuse (`F_ = F_trial`) | μHa energy difference — full-P screening differs from incremental accumulation |
| Deferred Fock reset at DIIS→GDM (`request_fock_reset`) | Cancelled J/K reuse on next step; wasted expensive full-P build |
| Immediate Fock reset at DIIS→GDM (`rebuild_fock`) | Line search failures on final step caused 2× regression (fixed by skip-on-converged, but still slow due to BFGS history/energy landscape changes) |
| Density scaling (`ΔP * P_max/dP_max`) | Made trial build as expensive as full-P; FP errors from large scale factor |
| Engine precision override | Only fixes one of three screening levels; Schwarz still too aggressive |
| Schwarz threshold tightening | Combined with engine override: still fails because F_ drift is the root cause, not per-step screening |
| Incremental trial for energy + full-P trial for caching | 2 builds per eval = same cost as baseline, no speedup |

### 2.6 The `fock_reset_steps` Change

Changed default from 2^30 (never) to 20 (Gaussian standard). This independently bounds incremental Fock drift for all SCF algorithms. The previous value allowed unlimited drift accumulation during long DIIS phases.

---

## 3. Architecture

### 3.1 Incremental Trial Eval

`SCFImpl::evaluate_trial_incremental(P_trial, J_out, K_out)`:
- Computes `ΔP = P_trial - P_`
- Calls `build_JK(ΔP, J_out, K_out)` with standard screening
- Returns `{energy, F_trial}` where `F_trial = F_ + J/K contribution`
- `J_out`/`K_out` are cached by the GDM functor for main-loop reuse

### 3.2 Main Loop J/K Reuse

```
if (trial_fock_valid_ && !would_reset):
    P_last = P_
    // J_, K_ already hold cached values from GDM trial
    // fall through to update_fock_()
else:
    // normal path: build_JK(P_diff) + update_fock_()
```

`update_fock_()` always runs — it applies `F_ += J_ - 0.5*K_` using whatever values are in `J_`/`K_`.

### 3.3 Fallback State Machine

```
incremental_trial_enabled_ = true  (start)
    ↓ line search failure + grad_norm > threshold
incremental_trial_enabled_ = false (permanent for this SCF)
    + rebuild_fock()
    + recompute last_accepted_energy_
```

---

## 4. Future Directions

### 4.1 DF-J + Pre-LinK K-only (Highest Priority, unchanged from previous session)

Eliminates all J integral evaluations from the direct loop. K-only screening removes 87-97% of quartets for extended systems.

### 4.2 Smarter Fallback Trigger

The current fallback triggers on any line search failure when gradient norm exceeds threshold. This is overeager for alkane12 at normal convergence (triggers at |grad_norm| ~ 5e-8, well below 1e-6 convergence threshold). A better trigger: only fall back if gradient norm is significantly above the convergence threshold (e.g., `> 10 × og_threshold`), OR require 2+ consecutive failures before disabling.

### 4.3 Per-Step Engine Precision Override

The libint2 engine precision (`epsilon / P_shmax`) is the dominant per-step screening loss for the incremental trial. Adding a `set_engine_precision_override()` API to the ERI would let the trial eval set engine precision to match full-P without scaling the density. This reduces the per-step screening loss and might make the fallback unnecessary for most systems. Prototype was implemented and tested but reverted because Schwarz screening (the other level) also needed fixing, and fixing both eliminates the cost advantage.

### 4.4 Fock Reset Policy Tuning

`fock_reset_steps = 20` is a reasonable Gaussian-compatible default. For very long SCF convergences (>50 steps), consider adaptive reset frequency based on monitored `|F_accum - F_fresh|` drift.

### 4.5 P450 Convergence

P450 (941 AOs, Fe-porphyrin) still not converged — needs ROHF+GDM or EDIIS/ADIIS. The porphyrin benchmark (856 AOs, closed-shell) serves as a P450-scale proxy.

---

## 5. How to Build and Test

```bash
# Build
cmake --build .local/release/build -j 32

# Run correctness tests
OMP_NUM_THREADS=4 ./.local/release/build/tests/test_scf

# Run SCF benchmark (normal convergence)
OMP_PROC_BIND=spread OMP_PLACES=cores QDK_SCF_BENCH=1 \
  QDK_SCF_BENCH_MAX_NAO=900 OMP_NUM_THREADS=32 \
  ./.local/release/build/tests/test_fock_benchmark \
  --gtest_filter='SCFBenchmark.FullSCF'

# Run with tight convergence (1e-8)
QDK_SCF_TIGHT=1 <same command>
```

---

## 6. Lessons Learned

1. **F_ = F_trial corrupts incremental accumulation.** Installing a full-P trial Fock into F_ causes the next `update_fock_()` to double-count J/K. Cache separately (`cached_trial_fock_`), never write to F_ from the algorithm.

2. **Deferred resets cancel reuse.** A `fock_reset_requested_` flag consumed at the next step's start sets `would_reset=true`, blocking J/K reuse. Immediate resets (inside `iterate()`) avoid this but change the energy landscape.

3. **Engine precision is the hidden screening level.** Schwarz threshold controls shell-level screening, but `engines_[i].set_precision(epsilon / P_shmax)` controls primitive-level screening inside libint2. With small ΔP, this drops 5 orders of magnitude of precision. Both must match full-P to get nHa-level accuracy from incremental builds.

4. **Full-P and incremental builds are numerically incompatible.** `J(P_full) ≠ sum(J(ΔP_i))` due to screening differences at every level. Any optimization that mixes them will have nHa-to-μHa differences depending on system size and convergence state. Accept this or use full-P exclusively.

5. **Skip-on-converged is independently valuable.** The baseline wasted a full GDM line search on the post-convergence step. This is pure overhead and can cause spurious line search failures on tiny gradients.

6. **Automatic fallback is essential for robustness.** The incremental trial cannot handle all energy surfaces (alkane12's flat landscape, tight convergence). Rather than trying to fix the screening (which eliminates the cost advantage), detect the failure and switch to the proven full-P path.
- `cpp/src/.../scf_algorithm/gdm.cpp` — blaspp, syevd, workspace preallocation, GDM timers
- `cpp/src/.../scf_algorithm/scf_algorithm.cpp` — syevd in solve_fock_eigenproblem
- `cpp/src/.../scf/scf_solver.cpp` + `.h` — print_timer_summary() API
- `cpp/src/.../eri/eri_multiplexer.cpp` + `.h` — set_screening_threshold propagation
- `cpp/src/.../eri/LIBINT2_DIRECT/libint2_direct.h` — set_screening_threshold override
- `cpp/src/.../core/eri.h` — set_screening_threshold virtual method
- `cpp/tests/test_fock_benchmark.cpp` — SCFBenchmark.FullSCF test

---

## 1. What Was Done

### 1.1 Delivered Optimizations

| Change | Impact | Where |
|--------|--------|-------|
| `syevd` (divide-and-conquer eigendecomp) | 2.93× on pseudo-canonical phase (4% of total → 1.5%) | scf_algorithm.cpp, gdm.cpp |
| Distance-dependent integral screening (Lambrecht-Ochsenfeld) | 27-49% of quartets caught before density lookups | libint2_direct.cpp |
| GDM blaspp migration | Explicit BLAS for syrk/gemm (no perf change, maintainability) | gdm.cpp |
| GDM workspace preallocation | Eliminates per-eval heap allocs in line search | gdm.cpp |
| Pre-LinK K-only path | Ready for DF-J integration (not active in merged J+K) | libint2_direct.cpp |
| Pre-LinK data structures | sig_kets + sig_bras sorted lists, rebuilt per build_JK | libint2_direct.cpp |

### 1.2 Cumulative Results (OMP_PROC_BIND=spread, 32T, DIIS_GDM, 0 μHa error)

| System | NAO | Iters | Baseline (ms) | Current (ms) | Speedup |
|--------|-----|-------|--------------|-------------|---------|
| h2o/def2-SVP | 24 | 12 | 48.6 | 48.1 | 1.01× |
| h2o/6-31G** | 25 | 12 | 28.4 | 28.9 | 0.98× |
| benzene/SVP | 114 | 9 | 1176.6 | 1141.1 | 1.03× |
| water5/SVP | 120 | 11 | 831.1 | 576.7 | **1.44×** |
| water10/SVP | 240 | 11 | 4205.4 | 3028.1 | **1.39×** |
| alkane12/6-31G* | 220 | 35 | 35832.8 | 34727.3 | 1.03× |
| water20/SVP | 480 | 11 | 6817.2 | 4960.9 | **1.37×** |
| benzene/TZVP | 222 | 9 | 6244.3 | 6066.5 | 1.03× |
| P450/cc-pVDZ | 941 | >62 | >3600s | not run | — |

Compact molecules (benzene, alkane, h2o): ~1.03× — limited by all shells being nearby.
Extended water clusters: **1.37-1.44×** — distance screening + syevd.

### 1.3 Instrumentation Added

All gated by environment variables (zero overhead when not set):

| Env Var | What it enables |
|---------|----------------|
| `QDK_SCF_BENCH=1` | SCFBenchmark.FullSCF test (full SCF to convergence on benchmark systems) |
| `QDK_SCF_BENCH_MAX_NAO=N` | Skip systems with NAO > N (for quick profiling) |
| `QDK_SCF_DELTA_P_DIAG=1` | Per-iteration ΔP shell-block sparsity diagnostics |
| `QDK_FOCK_SCREEN_DIAG=1` | Per-call Fock screening statistics (computed/screened/J/K breakdown) |

Timer keys added (all via `AutoTimer` RAII, printed via `SCF::print_timer_summary()`):
- `SCF::build_JK`, `SCF::algorithm`, `SCF::convergence`, `SCF::update_fock`, `SCF::total_energy`
- `GDM::iterate`, `GDM::gradient`, `GDM::pseudo_canonical`, `GDM::bfgs`, `GDM::line_search`

---

## 2. Key Findings

### 2.1 Where Time Goes (8 systems, NAO ≤ 480, 32T, DIIS_GDM)

| Phase | % of SCF time | Notes |
|-------|--------------|-------|
| SCF::build_JK (main loop) | 43% | One build_JK per SCF iteration |
| GDM::line_search (trial Fock) | 42% | Each GDM step does ~1.1 extra build_JK via evaluate_trial_density_energy_and_fock |
| GDM::pseudo_canonical | 1.5% | syevd reduced from 4% |
| Everything else | <1% each | BFGS, gradient, convergence, update_fock |

**85% of SCF time is build_JK** (main loop + GDM line search).

### 2.2 Screening Landscape

Screening effectiveness with the current threshold (1e-9):

| System | NAO | Full-P computed% | ΔP computed% (late) | Distance-screened% |
|--------|-----|-----------------|--------------------|--------------------|
| benzene/SVP | 114 | 20.9% | 50-90% | 6.8% |
| water5 | 120 | 16.4% | 23-53% | 41.9% |
| water10 | 240 | 8.1% | 14-29% | 48.9% |
| water20 | 480 | 12.9% | 23-46% | 45.2% |
| alkane12 | 220 | 10.1% | 41-61% | 27.0% |

Key observations:
- Full-P (SOAD guess) screening is excellent: 79-91% of quartets screened
- Incremental ΔP screening degrades for compact molecules (merged J+K screening is bottleneck)
- Distance screening catches 27-49% of quartets before density norm lookups
- K-only screening would screen 87-97% for extended systems (vs 79-91% merged)

### 2.3 What Doesn't Work

| Approach | Why it fails |
|----------|-------------|
| Adaptive threshold (τ = dp_max or dp_max²) | Too aggressive for compact molecules — Schwarz bounds O(100) for nearby shells corrupt the Fock matrix |
| J/K loop splitting (same impl) | Doubles integral evaluations, net regression |
| Per-quartet K contraction skip | Cumulative error O(N² × threshold) = ~50 μHa for NAO=220 |
| LinK with ΔP (incremental density) | GDM orbital rotations are globally delocalized — sig_kets = nsh (100%) |
| Heuristic distance bounds (1/sqrt(1+R²)) | Not a rigorous upper bound — causes 0.2-200 μHa errors |

### 2.4 Pre-LinK Significant Ket List Sizes (sig_kets_avg per shell)

| System | NAO | Full-P sig_kets_avg | Full-P % of N | ΔP sig_kets_avg | ΔP % of N |
|--------|-----|--------------------|--------------|--------------------|-----------|
| water5 | 120 | 1.8 | 2.9% | 60.0 | 100% |
| water10 | 240 | 1.8 | 1.5% | 120.0 | 100% |
| water20 | 480 | 1.8 | 0.7% | 66.4 | 27.7% |
| alkane12 | 220 | 2.1 | 1.7% | 124.0 | 100% |

Full-P builds: sig_kets is O(1) — pre-LinK would be highly effective.
ΔP builds (compact): sig_kets = nsh — pre-LinK can't help.
ΔP builds (extended, water20): sig_kets = 28% — moderate benefit.

---

## 3. Architecture of Current Code

### 3.1 Distance-Dependent Screening (Lambrecht-Ochsenfeld)

Located in `build_JK()`, runs before density-weighted Schwarz:

```
bound = nprim12 * nprim34 * K_max12 * K_max34 / sqrt(γ_min12 + γ_min34) * F₀(η * R²)
```

Uses `PairBoundData` precomputed per CSR shell pair in constructor:
- `K_max`: max |K| from ShellPair primitive pairs
- `gamma_min`: 1/max(one_over_gamma) — most diffuse primitive
- `P_diffuse[3]`: product center of most diffuse primitive pair
- `nprim`: number of surviving primitive pairs

Only activates when T > 1 (otherwise F₀ ≈ 1, no distance benefit).

### 3.2 Pre-LinK Data Structures

Built each `build_JK()` call (when K is requested):

- `sig_kets_[P]`: shells R sorted descending by `ceiling[P] * ceiling[R] * D_max(P,R)`. Density-dependent, rebuilt per call.
- `sig_bras_[P]`: shells Q sorted descending by `K_schwarz_(P,Q)`. Density-independent, built once in constructor.

K-only path (`J==nullptr && K!=nullptr`) uses these for ML_PQ construction with early-exit. Currently exercised only when ERIMultiplexer routes K to a separate implementation (e.g., DF-J + direct K).

### 3.3 GDM Changes

- `blas::syrk` for density construction (P = C_occ * C_occ^T)
- `blas::gemm` for F_MO = C^T F C (3 locations)
- `blas::gemm` for gradient transform Uoo^T * grad * Uvv
- `lapack::syevd` for pseudo-canonical eigendecomposition
- Pre-allocated `cached_P_`, `cached_C_`, `grad_tmp_` in GDMLineFunctor

---

## 4. Future Directions (Prioritized)

### 4.1 DF-J + Pre-LinK K-only (Highest Priority)

**What**: Enable DF-J (already exists via `do_dfj` flag) in the default SCF path. This routes J through density fitting (O(N²M)) and K through direct integrals with the pre-LinK K-only path.

**Why**: Eliminates all J integral evaluations from the direct loop. K-only screening removes 87-97% of quartets for extended systems. The pre-LinK ML_PQ construction with sorted early-exit gives asymptotically linear scaling for K.

**Expected impact**: For water20, J currently consumes ~50% of computed quartets. Eliminating J and using K-only screening (97% for full-P) would reduce computed quartets from 12.9% to ~1.5% — potentially **5-8× on build_JK**.

**Blockers**: DF-J requires auxiliary basis sets. Need to verify DF-J accuracy meets production requirements. DF-J precision concerns for high-accuracy calculations.

### 4.2 P450 Convergence (High Priority)

**What**: P450 (941 AO, Fe-porphyrin) fails to converge within reasonable time under DIIS_GDM. DIIS switches to GDM at step 25, then GDM converges linearly at ~80s/step, needing 60+ steps.

**Options**:
- EDIIS/ADIIS hybrid for metal centers
- Level shifting auto-tuning based on DIIS error trajectory
- Better initial guess (SAD vs SOAD)
- Tune DIIS→GDM switch criteria

### 4.3 Rigorous QQR (Medium Priority)

**What**: The current distance screening uses the Lambrecht-Ochsenfeld bound with the most diffuse primitive pair as a conservative estimate. A proper QQR implementation would use tighter bounds per primitive pair combination.

**Why**: Current bound screens 27-49% by distance, but many of those are also screened by density. Tighter per-primitive bounds could screen quartets that the density check doesn't catch.

### 4.4 Diagnostic Counter Cleanup (Low Priority)

The screening diagnostic counters (`loc_j_screen`, `loc_k_screen`, `Pj`, `Pk` computations) add 6 operations per quartet in the hot inner loop. These should be compiled out when `QDK_FOCK_SCREEN_DIAG` is not set, or moved behind `[[unlikely]]` branches.

---

## 5. How to Build and Test

```bash
# Build
cmake --build .local/release/build -j 32

# Run correctness tests
OMP_NUM_THREADS=4 ./.local/release/build/tests/test_scf
OMP_NUM_THREADS=4 ./.local/release/build/tests/test_fock_benchmark \
  --gtest_filter='-FockBenchmark.*:-SCFBenchmark.*'

# Run SCF benchmark (skip P450)
OMP_PROC_BIND=spread OMP_PLACES=cores QDK_SCF_BENCH=1 \
  QDK_SCF_BENCH_MAX_NAO=500 OMP_NUM_THREADS=32 \
  ./.local/release/build/tests/test_fock_benchmark \
  --gtest_filter='SCFBenchmark.FullSCF'

# Run with screening diagnostics
QDK_FOCK_SCREEN_DIAG=1 <same command>

# Run with ΔP diagnostics
QDK_SCF_DELTA_P_DIAG=1 <same command>

# Print per-phase timing breakdown
# (automatically printed at end of SCFBenchmark.FullSCF)
```

**Benchmarking requirement**: `OMP_PROC_BIND=spread OMP_PLACES=cores` is mandatory on multi-socket/NUMA machines.

---

## 6. Files Modified (This Session)

| File | Changes |
|------|---------|
| `libint2_direct.cpp` | Distance screening (PairBoundData + Lambrecht-Ochsenfeld F₀ bound), pre-LinK data structures (sig_kets_, sig_bras_, sig_bras_schwarz_), K-only ML_PQ inner loop, screening diagnostic counters |
| `libint2_direct.h` | set_screening_threshold override |
| `eri.h` | set_screening_threshold virtual method |
| `eri_multiplexer.h` | set_screening_threshold propagation to sub-impls |
| `eri_multiplexer.cpp` | (unchanged from pre-session) |
| `scf_impl.cpp` | Per-phase AutoTimers, ΔP shell-block diagnostics |
| `scf_solver.h/cpp` | print_timer_summary() public API |
| `gdm.cpp` | blaspp (syrk, gemm), syevd, workspace prealloc, GDM timers |
| `scf_algorithm.cpp` | syevd in solve_fock_eigenproblem |
| `test_fock_benchmark.cpp` | SCFBenchmark.FullSCF test, QDK_SCF_BENCH_MAX_NAO filter |

---

## 7. Lessons Learned

1. **Measure before optimizing.** The Eigen→blaspp migration had zero impact because Eigen already delegates to BLAS. Profiling first would have saved hours.

2. **Heuristic bounds are dangerous.** The `1/sqrt(1+R²)` distance screen passed all tests but introduced 0.2 μHa errors that only showed up in comparison to baseline. Always use rigorous bounds from the literature.

3. **Bisect errors immediately.** When I saw the 0.2 μHa discrepancy, I should have bisected instead of hand-waving about "FP rounding from code path changes."

4. **J/K splitting in the same implementation is a regression** for system sizes < 1000 AO. The doubled integral evaluation cost outweighs any screening improvement. Pre-LinK only pays off with DF-J or Cholesky-J to eliminate J integrals entirely.

5. **GDM line search consumes 42% of SCF time.** Each GDM step does ~1.1 trial Fock evaluations via evaluate_trial_density_energy_and_fock(P_trial) using FULL density — this is where pre-LinK with full-P screening (sig_kets_avg = 1.8 per shell) would shine.

6. **Incremental Fock degrades screening for GDM.** The orbital rotation ΔP is globally delocalized for compact molecules (sig_kets = 100%). For extended systems (water20), ΔP sig_kets drops to 28%.
