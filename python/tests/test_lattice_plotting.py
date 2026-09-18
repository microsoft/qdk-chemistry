"""Small geometric checks for the shared example plotter."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qdk_chemistry.data import LatticeGeometry, LatticeGraph
from qdk_chemistry.utils.model_hamiltonians import kitaev_honeycomb_bond_flavors


@pytest.fixture(scope="module")
def plot_graph():
    """Import the example helper without changing the package or import path."""
    path = Path(__file__).resolve().parents[2] / "examples/benchmarks/lattice_plotting.py"
    spec = importlib.util.spec_from_file_location("lattice_plotting", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_lattice_graph


@pytest.mark.parametrize("model", ["square", "honeycomb", "periodic_chain"])
def test_positions_bonds_and_rotation(plot_graph, model):
    """Preserve real bond displacements, image multiplicity, and graph identity across plotting."""
    if model == "honeycomb":
        geometry = LatticeGeometry.honeycomb_plaquettes(4, 4)
        graph = LatticeGraph.from_geometry(geometry, shells=[1, 2, 3], bond_flavors=kitaev_honeycomb_bond_flavors())
    else:
        geometry = LatticeGeometry.square(4, 4) if model == "square" else LatticeGeometry.chain(2, periodic=True)
        graph = LatticeGraph.from_geometry(geometry, shells=[1, 2])
    before = graph.content_hash()
    rotation = np.array([[0, -1], [1, 0]])
    figure, axes = plot_graph(graph, rotation_degrees=90, flavor_labels={0: "X", 1: "Y", 2: "Z"})
    try:
        np.testing.assert_allclose(axes[0].collections[-1].get_offsets(), geometry.positions @ rotation.T, atol=1e-14)
        for shell, axis in zip(graph.selected_shells, axes, strict=True):
            actual = [
                tuple(np.round(segment, 12).ravel())
                for collection in axis.collections
                if collection.get_zorder() == 2
                for segment in collection.get_segments()
            ]
            expected = [
                np.array([geometry.positions[b.site_i], geometry.positions[b.site_i] + b.displacement]) @ rotation.T
                for b in graph.connections
                if b.bond_class.shell == shell
            ]
            assert sorted(actual) == sorted(tuple(np.round(segment, 12).ravel()) for segment in expected)
        assert graph.content_hash() == before
        if model == "honeycomb":
            assert graph.num_sites == 48
            assert sum(b.bond_class.shell == 1 for b in graph.connections) == 63
            assert [text.get_text() for text in figure.legends[0].texts][:3] == ["X", "Y", "Z"]
    finally:
        plt.close(figure)


def test_sites_only_and_missing_geometry(plot_graph):
    """Edgeless geometries still plot; adjacency alone cannot determine an embedding."""
    graph = LatticeGraph.from_geometry(LatticeGeometry.chain(3), shells=[])
    figure, axes = plot_graph(graph, vectors={"a": (1, 0)})
    assert len(axes) == 1
    assert len(axes[0].collections[-1].get_offsets()) == 3
    plt.close(figure)
    with pytest.raises(ValueError, match="geometry"):
        plot_graph(LatticeGraph.from_dense_matrix(np.zeros((2, 2))))
