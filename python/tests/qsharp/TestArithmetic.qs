// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.TestUtils.ArithmeticTests {
    import Std.Canon.ApplyXorInPlace;
    import Std.Convert.ResultArrayAsInt;
    import Std.Measurement.MeasureEachZ;
    import QDKChemistry.Utils.Arithmetic.AddConstant;

    /// Returns, for each little-endian starting value of a `width`-qubit register, the value
    /// after `AddConstant(constant)`, or after its adjoint when `inverse` is set.
    operation TestAddConstantTable(constant : Int, width : Int, inverse : Bool) : Int[] {
        mutable results : Int[] = [];
        for value in 0..(1 <<< width) - 1 {
            use target = Qubit[width];
            ApplyXorInPlace(value, target);
            if inverse {
                Adjoint AddConstant(constant, target);
            } else {
                AddConstant(constant, target);
            }
            set results += [ResultArrayAsInt(MeasureEachZ(target))];
            ResetAll(target);
        }
        results
    }
}
