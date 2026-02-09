// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
import Std.Math.ArcCos;
import Std.Math.PI;
import Std.Convert.IntAsDouble;
import Std.Arrays.Subarray;
import Std.Arrays.Mapped;


operation MakeIQPECircuit(
    statePrep : Qubit[] => Unit,
    repControlledEvolution : (Qubit, Qubit[]) => Unit,
    accumulatePhase : Double,
    control : Int,
    systems : Int[],
) : Result {
    use qs = Qubit[Length(systems) + 1];
    let control = qs[control];
    let system = Subarray(systems, qs);

    statePrep(system);

    within {
        H(control);
    } apply {
        if accumulatePhase > 0.0 or accumulatePhase < 0.0 {
            Rz(accumulatePhase, control);
        }
        repControlledEvolution(control, system);
    }
    ResetAll(system);
    return MResetZ(control);
}
