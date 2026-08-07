# Ground-state QPE tutorial TODO

## Review completion

- [x] Finish Caroline review using grouped, current-source decisions rather than applying generated-HTML edits mechanically.
  - [x] Complete objective and staleness audit.
  - [x] Review and implement accepted Chapter 2-4 clarity changes.
  - [x] Review redundant or stale Chapter 5-6 suggestions.
  - [x] Decide whether any landing-page ideas should be retained without adopting the wholesale rewrite.
  - [x] Record final dispositions and validate.
  - [x] Commit and prepare reply.
- [x] Finish Yingrong review feedback.
  - [x] Complete grouped current-source decisions and implement accepted wording.
  - [x] Run grouped validation.
  - [x] Commit and prepare reply.
- [ ] Finish Martin review feedback.
  - [x] Complete objective and staleness audit.
  - [x] Discuss and implement accepted current-source changes.
  - [x] Replace the undefined term `quantum advantage` with plain language.
  - [x] Define `phase bit` directly in Chapter 5 and avoid using the term earlier on the landing page.
  - [ ] Ask Martin to complete the truncated `occupations vary among the determinants ... is somewhat` comment in the wrap-up email.
  - [x] Run the bounded coherence gate and grouped validation.
  - [x] Commit and prepare reply.
- [ ] Process any additional substantive reviewer feedback using the grouped review workflow.
- [ ] Run the end-stage review after substantive feedback is complete.
- [ ] Monitor Conrad's re-review and pending QDK/Chemistry typesetting clarification.

## Review carryover

- [ ] Evaluate near-zero-friction or one-click installation options in response to Conrad's participation-barrier concern.

## Code changes

- [ ] Move some of the tutorial functions into the main code base (David)

## CI

- [ ] Confirm PR #604 Build and Test jobs pass after merging the QDK/Chemistry 2.1.0 release.
- [ ] Rerun the failed macOS tutorial-compatibility job after its workflow finishes. The runner failed before checkout because GitHub Actions could not download actions.

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
