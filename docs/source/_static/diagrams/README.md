# Ground-state QPE tutorial figures

The ground-state molecular-QPE tutorial uses Graphviz diagrams, generated plots,
orbital renders, and logical-circuit diagrams. PNG regeneration is a
documentation-maintenance task outside the Sphinx build; Sphinx renders the
committed DOT sources through Graphviz during the build.

## Generated plots

Run these commands from the repository root in an environment containing the
QDK/Chemistry documentation dependencies:

```console
python docs/source/_static/diagrams/generate_tutorial_qpe_phase_grid.py
python docs/source/_static/diagrams/generate_tutorial_qpe_orbital_entropy.py
```

Both scripts save their figures as PNGs on light-gray canvases so fixed dark
labels remain readable in light and dark documentation themes. The phase-grid
script also regenerates `tutorial_qpe_phase_grid_table.rst`. Files ending in
`.dot` are rendered by Sphinx through Graphviz as transparent SVG assets.

## Orbital renders

The original cube-generation, isocontour, orbital-index, and camera settings for
the two Chapter 2 orbital figures are not available. Their original screenshots
are retained under `docs/figure_sources/ground_state_qpe`. The transparent PNGs
retain the original dimensions and reproduce every original RGB pixel exactly
when composited on white.

```console
python docs/source/_static/diagrams/generate_tutorial_qpe_screenshot_images.py
```

The generator uses color-to-alpha unmixing instead of thresholding, so
antialiased edges and dark surface contours are preserved.

## Logical circuits

The original widget screenshots are retained under
`docs/figure_sources/ground_state_qpe`. The screenshot generator replaces large
neutral background regions with light gray while preserving white labels and
small gate details. The resulting opaque PNGs remain readable against light and
dark documentation themes.

QDK/Chemistry obtains circuit data from QDK and displays it interactively with
`qdk.widgets.Circuit`. The widget does not currently provide a supported Python
or command-line SVG/PNG export API, so updating the source screenshots remains a
manual maintenance step.

[`microsoft/qdk#3632`](https://github.com/microsoft/qdk/issues/3632) tracks
deterministic, headless export from circuit data.
