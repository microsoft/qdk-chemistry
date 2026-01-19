"""End-to-end tests for sample notebooks and other sample workflows.

This module contains tests for notebooks and interoperability samples
(Pennylane, Q#) that are not covered by dedicated test modules.

See Also:
- test_sample_workflow_sci.py - Sparse-CI workflow tests
- test_sample_workflow_rdkit.py - RDKit geometry tests
- test_sample_workflow_qiskit.py - Qiskit IQPE tests

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest

################################################################################
# Sample notebook testing
################################################################################


@pytest.mark.xfail(reason="Skipping unimplemented examples/factory_list.ipynb test.")
def test_factory_list():
    """Test the examples/factory_list.ipynb script."""
    # TODO: Need to implement this test (see https://github.com/microsoft/qdk-chemistry/issues/196)
    raise NotImplementedError("TODO: add factory_list.ipynb test.")


@pytest.mark.xfail(reason="Skipping unimplemented examples/state_prep_energy.ipynb test.")
def test_state_prep_energy():
    """Test the examples/state_prep_energy.ipynb script."""
    # TODO: Need to implement this test (see https://github.com/microsoft/qdk-chemistry/issues/196)
    raise NotImplementedError("TODO: add state_prep_energy.ipynb test.")


################################################################################
# Pennylane interoperability sample testing
################################################################################


@pytest.mark.xfail(reason="Skipping unimplemented examples/interoperability/pennylane/qpe_no_trotter.py test.")
def test_pennylane_qpe_no_trotter():
    """Test the examples/interoperability/pennylane/qpe_no_trotter.py script."""
    # TODO: Need to implement this test (see https://github.com/microsoft/qdk-chemistry/issues/199)
    raise NotImplementedError("TODO: add pennylane/qpe_no_trotter.py test.")
