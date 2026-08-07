# Ground-state QPE tutorial TODO

## Review completion

- [ ] Process any additional substantive reviewer feedback using the grouped review workflow, followed by the full end-stage student and specialist review.

## Code changes

- [ ] Move some of the tutorial functions into the main code base (David)

## Output clarity

- [ ] Reduce QDK/Chemistry library logging that obscures instructional output in tutorial notebooks and scripts.
  - [ ] Audit the default log output for every required script and notebook.
  - [ ] Suppress library logs explicitly at script entry points and in notebook setup cells; do not mutate global logging at module import time.
  - [ ] Preserve tutorial-owned results, progress messages, and warnings while library logs are quiet.
  - [ ] Document a simple opt-in setting for students who need diagnostic QDK/Chemistry logs.
  - [ ] Validate quiet output through the script and notebook test paths.

## Visual design

- [ ] Add visuals directly to the Sphinx tutorial to reduce walls of text.
  - [x] Use the existing Sphinx Graphviz pipeline for simple conceptual diagrams: commit ``.dot`` sources under ``docs/source/_static/diagrams`` and include them with ``.. graphviz::``; do not add Mermaid, custom build steps, or another diagram dependency.
  - [x] Use committed static images only for quantitative plots that Graphviz cannot represent naturally; keep any reproducible generation script outside the Sphinx build.
  - [x] Start with an active/inactive/virtual orbital partition figure.
  - [x] Audit each chapter for concepts that genuinely benefit from a visual.
  - [x] Add an end-to-end Graphviz workflow on the landing page that connects the energy target, molecular model, active-space selection, qubit mapping, trial-state preparation, IQPE, and final reference comparison; use it to shorten or complement the existing six-stage prose roadmap.
  - [x] Add a reusable basis-function-to-multiconfigurational-wavefunction hierarchy in Chapter 2 and cross-reference it from the landing page.
  - [x] Add an orbital-entropy chart in Chapter 3 that shows the autoCAS selection gap across all candidate orbitals at once.
  - [ ] Add a Chapter 3 molecular-orbital image showing representative inactive, active, and virtual orbital isosurfaces from the existing visualization workflow.
  - [ ] Add a worked Jordan--Wigner parity-string figure in Chapter 4 that distinguishes fermionic modes, assigned qubits, and the lower-mode parity sign.
  - [ ] Add a Chapter 5 image comparing representative one- and multi-determinant state-preparation logical circuits exported from the existing circuit notebook.
  - [ ] Add a high-level single-iteration IQPE schematic in Chapter 6 before students inspect the fully decomposed circuit notebook.
  - [ ] Add an annotated Chapter 6 image of the rendered power-one IQPE circuit that identifies the readout ancilla, compute register, state preparation, feedback rotation, controlled evolution, and measurement.
  - [ ] Add a phase-grid number line in Chapter 6 that relates phase fraction, signed energy, grid spacing, reference alignment, and aliasing.
  - [ ] Consider a generated Chapter 1 plot showing the exponential sensitivity of equilibrium and rate predictions to free-energy error.
  - [ ] Consider a Chapter 6 energy-accounting figure that distinguishes the measured active energy, classically added core energy, reconstructed total, and CASCI reference.
  - [ ] Reuse the existing molecular-orbital and circuit workflows to export stable in-page assets; do not capture editor or widget UI screenshots.
  - [ ] Provide an ``:alt:`` description for every Graphviz directive and accessible captions/data descriptions for quantitative plots, then validate the rendered documentation.

## Low-priority cleanup

- [ ] Audit assumed-knowledge transitions for the mixed chemistry and quantum-computing audience; label prerequisite domains and balance refresher links without forcing every chapter to the same depth.
- [ ] Revisit nonblocking student-review friction only if later feedback confirms a learning obstacle: phase-kickback derivation, feedback recurrence, occupation-symbol-to-bit mapping, gate-family output, and the Chapter 3 coefficient-norm forward reference.
- [ ] Delete this TODO file after every item is complete.
