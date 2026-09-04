// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/// Test-only drivers for `QDKChemistry.Utils.PrepSelPrep`.
///
/// The Python test layer evaluates this file into a throwaway context (see
/// `tests/qsharp_test_sources.py`); it is never part of the shipped
/// `qdk_chemistry.utils.qsharp` project.
namespace QDKChemistry.TestUtils.PrepSelPrepTests {

    import QDKChemistry.Utils.PrepSelPrep.MakePrepSelPrepOp;

    /// # Summary
    /// One-system-qubit, one-ancilla block encoding used to drive block-encoding-agnostic
    /// schedules from a test.
    function TestMakeBlockEncodingOp(theta : Double) : (Qubit[] => Unit is Adj + Ctl) {
        MakePrepSelPrepOp(
            (ancilla) => Ry(theta, ancilla[0]),
            (ancilla, system) => Controlled Z(ancilla, system[0]),
            1
        )
    }
}
