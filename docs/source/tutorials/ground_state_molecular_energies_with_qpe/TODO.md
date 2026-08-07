# Ground-state QPE tutorial TODO

## Review completion

- [ ] Process any additional substantive reviewer feedback using the grouped review workflow, followed by the full end-stage student and specialist review.

## Pre-merge issue migration

- [ ] Convert every remaining unchecked item in this file into a GitHub issue assigned to Nathan, recording any QDK/Chemistry 2.2 dependency explicitly.
- [ ] Delete this TODO file only after every remaining item has been represented by a GitHub issue.

GitHub issues created from this file should be assigned to Nathan for tracking. Issues A and B depend on David's QDK/Chemistry 2.2 work before Nathan's downstream tutorial cleanup can begin.

## Visual design

- [ ] Add visuals directly to the Sphinx tutorial to reduce walls of text.
  - [x] Use the existing Sphinx Graphviz pipeline for simple conceptual diagrams: commit ``.dot`` sources under ``docs/source/_static/diagrams`` and include them with ``.. graphviz::``; do not add Mermaid, custom build steps, or another diagram dependency.
  - [x] Use committed static images only for quantitative plots that Graphviz cannot represent naturally; keep any reproducible generation script outside the Sphinx build.
  - [x] Start with an active/inactive/virtual orbital partition figure.
  - [x] Audit each chapter for concepts that genuinely benefit from a visual.
  - [x] Add an end-to-end Graphviz workflow on the landing page that connects the energy target, molecular model, active-space selection, qubit mapping, trial-state preparation, IQPE, and final reference comparison; use it to shorten or complement the existing six-stage prose roadmap.
  - [x] Add a reusable basis-function-to-multiconfigurational-wavefunction hierarchy in Chapter 2 and cross-reference it from the landing page.
  - [x] Add labeled examples of atom-centered ``cc-pvdz`` basis functions and unlabeled example molecular orbitals to Chapter 2 using assets exported from the interactive visualization workflow.
  - [x] Add an orbital-entropy chart in Chapter 3 that shows the autoCAS selection gap across all candidate orbitals at once.
  - [x] Keep the selected-versus-excluded natural-orbital comparison interactive in the Chapter 3 notebook instead of duplicating it as a static Sphinx figure; prompt students to inspect at least one orbital from each group.
  - [x] Add a worked Jordan--Wigner parity-string figure in Chapter 4 that distinguishes fermionic modes, assigned qubits, and the lower-mode parity sign.
  - [x] Add a Chapter 5 image comparing representative one- and two-determinant state-preparation logical circuits rendered by the existing circuit notebook.
  - [x] Add a high-level single-iteration IQPE schematic in Chapter 6 before students inspect the fully decomposed circuit notebook.
  - [x] Add an annotated Chapter 6 image of the rendered power-one IQPE circuit that identifies the readout ancilla, compute register, state preparation, feedback rotation, controlled evolution, and measurement.
  - [ ] Add a phase-grid number line in Chapter 6 that relates phase fraction, signed energy, grid spacing, reference alignment, and aliasing.
  - [ ] Consider a Chapter 6 energy-accounting figure that distinguishes the measured active energy, classically added core energy, reconstructed total, and CASCI reference.
  - [ ] Reuse the existing circuit workflows to export stable in-page assets; do not capture editor or widget UI screenshots.
  - [ ] Provide an ``:alt:`` description for every Graphviz directive and accessible captions/data descriptions for quantitative plots, then validate the rendered documentation.

## Low-priority cleanup

- [ ] **Issue A — blocked by QDK/Chemistry 2.2; upstream implementation owner: David.** Remove the direct PySCF dependency from the ground-state QPE tutorial examples after native replacement functionality is released.
- [ ] **Issue B — blocked by QDK/Chemistry 2.2; upstream implementation owner: David.** Replace the functionality in ``tutorial_orbital_coordinates.py`` with the corresponding native QDK/Chemistry functionality, then remove the tutorial helper.
- [ ] **Blocked by issues A and B; owner: Nathan.** Clean up the tutorial Python scripts for pedagogical accessibility and determine which implementation details and learning activities should move from scripts into notebooks.
- [ ] Audit assumed-knowledge transitions for the mixed chemistry and quantum-computing audience; label prerequisite domains and balance refresher links without forcing every chapter to the same depth.
- [ ] Revisit nonblocking student-review friction only if later feedback confirms a learning obstacle: phase-kickback derivation, feedback recurrence, occupation-symbol-to-bit mapping, gate-family output, and the Chapter 3 coefficient-norm forward reference.
