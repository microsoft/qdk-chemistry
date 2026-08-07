# Ground-state QPE tutorial TODO

## Review completion

- [ ] Process any additional substantive reviewer feedback using the grouped review workflow.
- [ ] Run the end-stage review after substantive feedback is complete.

## Review carryover

- [ ] Evaluate near-zero-friction or one-click installation options in response to Conrad's participation-barrier concern.

## Code changes

- [ ] Move some of the tutorial functions into the main code base (David)

## CI

- [ ] Confirm PR #604 Build and Test jobs pass after merging the QDK/Chemistry 2.1.0 release.
- [x] Confirm the replacement macOS exact published tutorial baseline passes after the earlier runner-service failure.

## Visual design

- [ ] Add visuals directly to the Sphinx tutorial to reduce walls of text.
  - [ ] Start with an active/inactive/virtual orbital partition figure.
  - [ ] Audit each chapter for concepts that genuinely benefit from a visual.
  - [ ] Reuse existing molecular-orbital widgets and circuit notebooks where appropriate.
  - [ ] Provide accessible captions and alt text, then validate rendered documentation.

## Low-priority cleanup

- [ ] Audit assumed-knowledge transitions for the mixed chemistry and quantum-computing audience; label prerequisite domains and balance refresher links without forcing every chapter to the same depth.
- [ ] Evaluate whether to retain the mixed Python-script and Jupyter-notebook workflow or consolidate it; compare testability, reuse, interactivity, duplication, and reader experience, and improve terse output where interpretation remains unclear.
- [ ] Consider importing `PauliProductFormulaContainer` directly from `qdk_chemistry.data` in the tutorial test.
- [ ] Audit analogous concrete internal imports if time remains.
- [ ] Revisit nonblocking student-review friction only if later feedback confirms a learning obstacle: phase-kickback derivation, feedback recurrence, occupation-symbol-to-bit mapping, gate-family output, and the Chapter 3 coefficient-norm forward reference.
- [ ] Delete this TODO file after every item is complete.
