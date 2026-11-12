// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.PrepSelPrepTests {

    import QDKChemistry.Utils.PrepSelPrep.MakePrepSelPrepOp;

    /// # Summary
    /// One-system-qubit, one-ancilla block encoding used to drive block-encoding-agnostic
    /// schedules from a test.
    function MakeTestBlockEncodingOp(theta : Double) : (Qubit[] => Unit is Adj + Ctl) {
        MakePrepSelPrepOp(
            (ancilla) => Ry(theta, ancilla[0]),
            (ancilla, system) => Controlled Z(ancilla, system[0]),
            1,
            1
        )
    }
}
