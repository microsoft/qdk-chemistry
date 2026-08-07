# Ground-state QPE tutorial TODO

## Review completion

- [ ] Process any additional substantive reviewer feedback using the grouped review workflow, followed by the full end-stage student and specialist review.

## Code changes

- [ ] Move some of the tutorial functions into the main code base (David)

## Visual design

- [ ] Add visuals directly to the Sphinx tutorial to reduce walls of text.
  - [x] Use the existing Sphinx Graphviz pipeline for simple conceptual diagrams: commit ``.dot`` sources under ``docs/source/_static/diagrams`` and include them with ``.. graphviz::``; do not add Mermaid, custom build steps, or another diagram dependency.
  - [ ] Use committed static images only for quantitative plots that Graphviz cannot represent naturally; keep any reproducible generation script outside the Sphinx build.
  - [ ] Start with an active/inactive/virtual orbital partition figure.
  - [x] Audit each chapter for concepts that genuinely benefit from a visual.
  - [x] Add an end-to-end Graphviz workflow on the landing page that connects the energy target, molecular model, active-space selection, qubit mapping, trial-state preparation, IQPE, and final reference comparison; use it to shorten or complement the existing six-stage prose roadmap.
  - [ ] Add a reusable basis-function-to-multiconfigurational-wavefunction hierarchy in Chapter 2 and cross-reference it from the landing page.
  - [ ] Add an orbital-entropy chart in Chapter 3 that shows the autoCAS selection gap across all candidate orbitals at once.
  - [ ] Add a worked Jordan--Wigner parity-string figure in Chapter 4 that distinguishes fermionic modes, assigned qubits, and the lower-mode parity sign.
  - [ ] Add a high-level single-iteration IQPE schematic in Chapter 6 before students inspect the fully decomposed circuit notebook.
  - [ ] Add a phase-grid number line in Chapter 6 that relates phase fraction, signed energy, grid spacing, reference alignment, and aliasing.
  - [ ] Consider a generated Chapter 1 plot showing the exponential sensitivity of equilibrium and rate predictions to free-energy error.
  - [ ] Consider a Chapter 6 energy-accounting figure that distinguishes the measured active energy, classically added core energy, reconstructed total, and CASCI reference.
  - [ ] Reuse existing molecular-orbital widgets and circuit notebooks where appropriate instead of committing duplicate screenshots.
  - [ ] Provide an ``:alt:`` description for every Graphviz directive and accessible captions/data descriptions for quantitative plots, then validate the rendered documentation.

## Low-priority cleanup

- [ ] Audit assumed-knowledge transitions for the mixed chemistry and quantum-computing audience; label prerequisite domains and balance refresher links without forcing every chapter to the same depth.
- [ ] Revisit nonblocking student-review friction only if later feedback confirms a learning obstacle: phase-kickback derivation, feedback recurrence, occupation-symbol-to-bit mapping, gate-family output, and the Chapter 3 coefficient-norm forward reference.
- [ ] Delete this TODO file after every item is complete.
