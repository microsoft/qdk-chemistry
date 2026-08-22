# Ground-state QPE tutorial figures

The ground-state molecular-QPE tutorial uses Graphviz diagrams, generated plots,
orbital renders, and logical-circuit diagrams. Generation is a documentation
maintenance task and does not run during a Sphinx build.

## Generated plots

Run these commands from the repository root in an environment containing the
QDK/Chemistry documentation dependencies:

```console
python docs/source/_static/diagrams/generate_tutorial_qpe_phase_grid.py
python docs/source/_static/diagrams/generate_tutorial_qpe_orbital_entropy.py
```

Both scripts save their figures with transparent canvases. The phase-grid script
also regenerates `tutorial_qpe_phase_grid_table.rst`. Files ending in `.dot` are
rendered by Sphinx through Graphviz.

## Orbital renders

The original cube-generation, isocontour, orbital-index, and camera settings for
the two Chapter 2 orbital figures are not available. The transparent PNGs retain
the original dimensions and reproduce every original RGB pixel exactly when
composited on white. The opaque source assets are preserved in Git commit
`e128a7a40d7645dee8a0bbf854c10d823b79ee46`. Recover either source with
`git show`, then run the converter:

```console
git show e128a7a40d7645dee8a0bbf854c10d823b79ee46:docs/source/_static/diagrams/tutorial_qpe_atomic_basis_functions.png > /tmp/tutorial_qpe_atomic_basis_functions.png
python docs/source/_static/diagrams/convert_white_to_alpha.py /tmp/tutorial_qpe_atomic_basis_functions.png docs/source/_static/diagrams/tutorial_qpe_atomic_basis_functions.png
```

Repeat the command for `tutorial_qpe_example_molecular_orbitals.png`. The script
uses color-to-alpha unmixing instead of thresholding, so antialiased edges and
dark surface contours are preserved.

## Logical circuits

QDK/Chemistry obtains circuit data from QDK and displays it with
`qdk.widgets.Circuit`. The widget is owned by the
[`microsoft/qdk`](https://github.com/microsoft/qdk) repository and currently
exposes circuit JSON to an interactive `qsharp-widgets` SVG renderer. It does not
provide a supported Python or command-line SVG/PNG export API.

The committed SVGs contain no editor, notebook, browser, or widget controls.
They preserve the widget's natural circuit lengths, align both state-preparation
circuits to the same twelve wires, and retain all thirteen IQPE wires and
composite-operation labels. The assets are shared with
[`microsoft/qdk#3631`](https://github.com/microsoft/qdk/pull/3631).

Until QDK provides a supported exporter, the temporary maintenance workflow is:

1. Run the tested tutorial workflow to construct the circuit data.
2. Display that data with `qdk.widgets.Circuit`.
3. Copy the rendered circuit SVG element, not a screenshot or surrounding UI.
4. Remove editor-only collapse, expand, and drop-zone layers.
5. Normalize the SVG viewport and use transparent, foreground-relative styles.
6. Verify wire counts, operation labels, captions, and alt text before committing.

[`microsoft/qdk#3632`](https://github.com/microsoft/qdk/issues/3632) tracks
deterministic, headless export from circuit data.
