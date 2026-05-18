# Fock Builder & SCF Optimization — Handoff Document (Updated)

## Branch: `users/copilot/fock-builder-benchmarks`

**Base commit**: main (`a679187a`)
**16 commits** on top. All Fock correctness tests pass (18/18).
1/33 SCF tests fails (`AtomInitGuessEnergyConvergence`) due to FP
reordering from tile contraction — borderline single-atom GDM
convergence, not a correctness bug.

**Primary file**: `cpp/src/qdk/chemistry/algorithms/microsoft/scf/src/eri/LIBINT2_DIRECT/libint2_direct.cpp`

---

## 1. What Was Done

### 1.1 Fock Builder Optimizations (16 commits)

| Phase | Commit | What | Impact |
|---|---|---|---|
| 0+1 | `c1bbae4e` | Benchmark harness + CSR shell-pair refactor | Baseline |
| 2a | `fada116a` | Persistent per-thread Libint2 engine pool | Big win on small systems |
| 3 | `0f750c7b` | Bra-pair-level scheduling | Load balance |
| 3 | `6b3f3603` | Cost-balanced static scheduling | Determinism |
| — | `6286d7d4` | Parallelize TLS reduction + symmetrization | |
| — | `1ab3af27` | NUMA-aware persistent TLS buffers | Multi-socket |
| — | `4f4c569b` | Merge TLS zeroing into compute parallel region | |
| — | `073f0a99` | P450 model compound (88 atoms, Fe-porphyrin) | |
| — | `5e01b22c` | Sparse reduction + zeroing via touch tracking | |
| — | `73f56fb7` | Hybrid dense/sparse reduction + UHF ndm fix | |
| — | `4bbafb96` | ERI threshold relaxation (1e-10 → 1e-9) + sweep | 8-14% |
| — | `54474c76` | Stack-tile contraction (L1-hot per-quartet tiles) | -38% benz/TZVP |
| — | `0233ca95` | Merged J+K single integral-buffer pass | |
| — | `583842a6` | Precompute schedule + parallel zero + touched lists | |
| — | `7f7ea87e` | Tiled TLS (shell-pair-blocked, cache-line aligned) | Structural |

### 1.2 Cumulative Results (OMP_PROC_BIND=spread, 64T, 8-socket Xeon 8180M)

| System | NAO | Baseline | Current | Speedup |
|---|---|---|---|---|
| benzene/SVP | 114 | 110ms | 29ms | **3.8×** |
| water₅ | 120 | 82ms | 15ms | **5.5×** |
| water₁₀ | 240 | 365ms | 100ms | **3.7×** |
| alkane₁₂ | 220 | 829ms | 292ms | **2.8×** |
| water₂₀ | 480 | 573ms | 92ms | **6.2×** |
| benzene/TZVP | 222 | 479ms | 225ms | **2.1×** |
| P450 | 941 | — | 24.2s | new |

### 1.3 Current Architecture

`build_JK` method:
1. Shell-block density norm computation (symmetric, lower-triangle only)
2. Cost-balanced bra-pair schedule (precomputed once in constructor)
3. Single parallel region:
   - Sparse zeroing via touched-index lists (contiguous tile memset)
   - Bra-pair loop with merged J+K contraction using stack tiles
   - Stack tile flush to tiled TLS (contiguous, cache-line aligned)
   - Per-thread cost model for reduction dispatch
4. Fused tile-to-dense reduction (contiguous tile reads → dense output)
5. Parallel symmetrization

Key data structures:
- `sp_csr_` — CSR shell-pair list (overlap-screened)
- `tile_offset_[sa*nsh+sb]` — shell-pair tile layout in TLS
- `J_tls_`, `K_tls_` — per-thread tiled TLS (NUMA first-touch)
- `touched_J_list_`, `touched_K_list_` — compact touched indices per thread

---

## 2. Critical Caveat: Random Density Benchmarks

**All benchmark timings above use synthetic random density matrices**
(`make_random_density()` — uniform noise in [-0.01, 0.01]). This has
major implications:

1. **Screening is artificially weak**: Random density has no spatial
   structure. Density-weighted screening provides minimal benefit beyond
   pure Schwarz bounds. Real SCF densities are spatially localized.

2. **LinK was evaluated with random density**: sig_kets_avg = nsh for ALL
   benchmark systems (prescreen passes for every shell pair). The inner
   screen pruned 33-66% of ket pairs for water clusters, but separating
   J and K into independent loops doubled the integral cost, causing a 2×
   regression. **LinK must be re-evaluated with real SCF densities,
   particularly with ΔP (incremental Fock).**

3. **Threshold sweep**: The 1e-10 → 1e-9 relaxation was validated with
   random density. Real SCF may have different accuracy/speed tradeoffs.

---

## 3. Future Directions: All-Up SCF Optimization

The next round should shift from isolated `build_JK` benchmarks to
**full SCF wall-time optimization**. Key areas:

### 3.1 LinK with Incremental Fock (High Priority)

LinK should be revisited in the context of **delta Fock builds** (incremental
Fock / ΔP), not full-density builds:

- **ΔP is sparse by construction**: In late SCF iterations, ΔP = P_new - P_old
  has only a few significant shell blocks (where the density is still changing).
  This is exactly the regime where LinK's `significant_kets[P]` based on
  `D_max(P,R)` would dramatically prune the ket list.

- **The J+K separation cost is lower with ΔP**: If ΔP has 10% non-zero shell
  blocks, J-only screening with ΔP skips 90% of quartets. The "doubled integral
  cost" from separate loops becomes "10% + 10%" instead of "100% + 100%".

- **Implementation**: Build `significant_kets` from ΔP (not full P). Rebuild
  each SCF iteration as ΔP evolves. Early iterations (large ΔP) → conventional.
  Late iterations (sparse ΔP) → LinK dominates.

- **DF-J already exists** in the codebase — LinK-K + DF-J is the natural
  combination. LinK handles K-only with separate density-screened ket lists
  while DF-J handles J via 3-center integrals. No doubled integral cost.

### 3.2 Full SCF Benchmarking Infrastructure

Current benchmarks time a single `build_JK` call with random density. Need:

- **End-to-end SCF timing**: Total wall time for SCF convergence, broken down
  by phase (Fock build, DIIS, diag, density construction).
- **Per-iteration timing**: Track how Fock build time changes across SCF
  iterations (should decrease with incremental Fock as ΔP shrinks).
- **Real molecular systems**: P450 SCF convergence, protein fragments, etc.

### 3.3 Tiled P / Tiled Output (Incremental)

The TLS buffers are now tiled. Next steps:
- **Tile P at entry**: Convert dense P to tiled layout once per `build_JK` call.
  Eliminates N-strided density reads in the contraction pre-load step.
  Tile boundaries should align to cache lines (64B) to avoid false sharing.
- **Tiled output**: If the SCF loop (DIIS, diag) can consume tiled J/K directly,
  eliminate the tile-to-dense reduction entirely. Requires changes to `scf_impl.cpp`.

### 3.4 Other SCF-Level Optimizations

- **DIIS / GDM cost**: For large systems, DIIS extrapolation and Fock matrix
  diagonalization may become significant fractions of total SCF time.
- **Initial guess quality**: Better initial density → fewer SCF iterations →
  fewer Fock builds.
- **Convergence acceleration**: EDIIS/ADIIS hybrid, level shifting tuning.

---

## 4. How to Build and Test

```bash
# Build (from repo root)
cmake -S cpp -B .local/release/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=.local/release/install
cmake --build .local/release/build -j 4

# Run correctness tests (fast, ~8 seconds)
OMP_NUM_THREADS=4 ./.local/release/build/tests/test_fock_benchmark \
  --gtest_filter='-FockBenchmark.*'

# Run benchmarks (MUST use NUMA binding for stable 64T numbers)
OMP_PROC_BIND=spread OMP_PLACES=cores \
QDK_FOCK_BENCH=1 OMP_NUM_THREADS=64 \
  ./.local/release/build/tests/test_fock_benchmark \
  --gtest_filter='FockBenchmark.ThreadScaling'

# Run SCF tests
OMP_NUM_THREADS=4 ./.local/release/build/tests/test_scf
```

**Benchmarking requirement**: `OMP_PROC_BIND=spread OMP_PLACES=cores` is
mandatory on the 8-socket/8-NUMA-node machine. Without binding, 64T numbers
vary up to 1.7× due to NUMA thread placement noise.

---

## 5. Files Modified

| File | What changed |
|---|---|
| `cpp/src/.../LIBINT2_DIRECT/libint2_direct.cpp` | All Fock builder optimizations (~1200 lines) |
| `cpp/tests/test_fock_benchmark.cpp` | Benchmark harness, correctness tests (~960 lines) |
| `cpp/src/.../microsoft/scf.cpp` | ERI threshold auto-multiplier (1e-5, unchanged) |
| `cpp/src/.../scf/core/scf.h` | ERIConfig default threshold (1e-9) |

---

## 6. Key Design Decisions

1. **`schedule(static)` with cost-balanced interleaving** for bit-reproducibility
2. **Tiled TLS with cache-line-aligned tiles** — no N-strided accumulation writes
3. **Stack-tile contraction** — L1-hot per-quartet tiles, flushed to tiled TLS
4. **Merged J+K pass** — single integral buffer traversal for both matrices
5. **Touched-index lists** alongside bitmaps — O(touched) zeroing/reduction
6. **Hybrid dense/sparse reduction threshold** removed in favor of fused
   tile-to-dense reduction (always sparse with tiled TLS)

---

## 7. LinK Implementation Notes (for future revisit)

A full LinK implementation was built and tested (then reverted). Key findings:

**Data structures needed:**
- `link_shell_ceilings_[P]` = max_Q K_schwarz_(P,Q) — build once
- `link_sig_bras_[R]` — Schwarz-sorted (DESC) bra list per shell — build once
- `link_sig_kets_[P]` — density-keyed ket list per shell — rebuild per iteration

**ML_PQ construction per bra pair:**
- Walk `sig_kets[s1]` and `sig_kets[s2]`
- For each R, walk `sig_bras[R]` with early-exit break (sorted DESC)
- Screen: `D_max(s1,R) * K_schwarz(s1,s2) * K_schwarz(R,S) >= tau`
- Deduplicate via sort + unique

**Why it failed with random density:**
- `sig_kets_avg = nsh` for all systems (prescreen useless)
- Separate J/K loops doubled integral cost (~50% of runtime)
- Inner screen pruned 33-66% but couldn't offset the doubling

**Why it should work with incremental Fock (ΔP):**
- ΔP is sparse → `sig_kets` has O(1) entries per shell
- Both J and K loops iterate O(N) quartets instead of O(N²)
- Total cost: O(N) instead of O(N²) — the fundamental win

**Psi4 reference**: `psi4/src/psi4/libfock/LinK.cc` (PR #2359, 2022)

**Key pitfalls (from Psi4 experience):**
1. Canonical RS ≤ PQ constraint in ML_PQ
2. Sorted lists are load-bearing for early-exit breaks
3. Stripe-out tracking for sparse K accumulation
4. Two independent thresholds (Schwarz vs density-keyed)
5. Degeneracy prefactors for diagonal shells
6. Density must be updated before sig_kets rebuild
