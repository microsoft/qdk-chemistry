// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.SkewTreeTests {
    import Std.Arrays.Zipped;
    import Std.Canon.ApplyToEachA;
    import Std.Canon.ApplyToEachCA;
    import Std.Core.Length;
    import Std.Diagnostics.CheckAllZero;
    import Std.TableLookup.Select;
    import QDKChemistry.Utils.SkewTree.SkewTreeLookup;

    /// Loads `data[address]` into `output` under `numControls` controls, copies it into `copy`, then erases `output`.
    function TestMakeSkewTreeCopyOp(
        data : Bool[][],
        fields : (Int, Int)[],
        numControls : Int
    ) : (Qubit[], Qubit[], Qubit[], Qubit[]) => Unit {
        (controls, address, output, copy) => {
            within {
                Controlled SkewTreeLookup(controls[0..numControls - 1], (data, fields, address, output));
            } apply {
                ApplyToEachCA(CNOT, Zipped(output, copy));
            }
        }
    }

    /// Whether a skew-tree load of `skewData` erased by `Select` over the full `2^n`-row `reference` returns every
    /// superposed address and control to zero, so any value or phase mismatch fails.
    operation TestSkewTreePhaseAgreement(
        skewData : Bool[][],
        reference : Bool[][],
        fields : (Int, Int)[],
        numAddressBits : Int,
        numControls : Int
    ) : Bool {
        use controls = Qubit[numControls];
        use address = Qubit[numAddressBits];
        use output = Qubit[Length(reference[0])];
        within {
            ApplyToEachA(H, controls + address);
        } apply {
            Controlled SkewTreeLookup(controls, (skewData, fields, address, output));
            Controlled Select(controls, (reference, address, output));
        }
        let agree = CheckAllZero(controls + address + output);
        ResetAll(controls + address + output);
        agree
    }

    /// Traces one lookup of `data` over `fields` under `numControls` controls for a costing regression.
    operation TestSkewTreeResourceProbe(data : Bool[][], fields : (Int, Int)[], numAddressBits : Int, numControls : Int) : Unit {
        use controls = Qubit[numControls];
        use address = Qubit[numAddressBits];
        use output = Qubit[Length(data[0])];
        Controlled SkewTreeLookup(controls, (data, fields, address, output));
    }
}
