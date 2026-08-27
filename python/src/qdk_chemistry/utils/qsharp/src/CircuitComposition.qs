// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

namespace QDKChemistry.Utils.CircuitComposition {

    import Std.Arrays.Subarray;
    import Std.ResourceEstimation.BeginEstimateCaching;
    import Std.ResourceEstimation.EndEstimateCaching;
    import Std.ResourceEstimation.SingleVariant;

    /// Returns the controlled version of `op`, taking the control register as its first argument.
    function MakeControlledOp<'T>(op : 'T => Unit is Adj + Ctl) : ((Qubit[], 'T) => Unit is Adj + Ctl) {
        Controlled op
    }

    /// Applies `op` to `target` `power` times.
    operation ApplyRepeated<'T>(
        cacheName : String,
        op : 'T => Unit is Adj + Ctl,
        power : Int,
        target : 'T
    ) : Unit is Adj + Ctl {
        for _ in 1..power {
            if BeginEstimateCaching(cacheName, SingleVariant()) {
                op(target);
                EndEstimateCaching();
            }
        }
    }

    /// Returns an operation applying `op` `power` times.
    /// Parameters:
    /// - `cacheName`: A string used for caching the resource estimation
    function MakeRepeatedOp<'T>(
        cacheName : String,
        op : 'T => Unit is Adj + Ctl,
        power : Int
    ) : ('T => Unit is Adj + Ctl) {
        ApplyRepeated(cacheName, op, power, _)
    }

    /// Adapts a control-register operation to the single-control-qubit shape phase estimation takes.
    function MakeSingleControlOp<'T>(op : (Qubit[], 'T) => Unit is Adj + Ctl) : ((Qubit, 'T) => Unit is Adj + Ctl) {
        (control, target) => op([control], target)
    }

    /// Applies two operations sequentially on the same system register.
    operation ApplySequential(
        first : Qubit[] => Unit,
        second : Qubit[] => Unit,
        systems : Qubit[]
    ) : Unit {
        first(systems);
        second(systems);
    }

    /// Returns a composed operation that applies ``first`` and then ``second``.
    function MakeSequentialOp(first : Qubit[] => Unit, second : Qubit[] => Unit) : Qubit[] => Unit {
        ApplySequential(first, second, _)
    }

    /// Returns `op` wrapped so it prepares and restores its own trailing `numShared` ancillas.
    ///
    /// Use where `op` runs once; a caller invoking it repeatedly should hoist `prepareShared`.
    /// `numShared` of 0 leaves `op` unwrapped, since the trailing slice would be empty.
    function MakeSharedAncillaOp(
        op : Qubit[] => Unit is Adj + Ctl,
        prepareShared : Qubit[] => Unit is Adj + Ctl,
        numShared : Int
    ) : Qubit[] => Unit is Adj + Ctl {
        (qs) => {
            within {
                if numShared > 0 {
                    prepareShared(qs[Length(qs) - numShared...]);
                }
            } apply {
                op(qs);
            }
        }
    }

    /// Returns `op` wrapped so it allocates, prepares, and releases its own trailing ancillas.
    ///
    /// The counterpart to `MakeSharedAncillaOp`: that one leaves the ancillas in the caller's
    /// register, so the returned callable still spans `op`'s full width. This one hides them,
    /// so the callable takes only the leading register and can be handed to a caller whose
    /// layout has no slot for them. `numOwned` of 0 leaves `op` unwrapped.
    ///
    /// The ancillas are restored by the `within` block before release, so the wrapper stays
    /// Adj + Ctl and the visible register is safe to reflect about |0⟩.
    ///
    /// Cost note: `prepareOwned` runs on every invocation, so a caller that invokes the
    /// returned callable repeatedly (a walk step calls PREPARE and PREPARE† once each,
    /// repeated once per phase-estimation query) pays for the preparation each time. Where
    /// the caller already owns a persistent register of the same width, hoisting the
    /// preparation out and using `MakeSharedAncillaOp` is cheaper; this wrapper exists for
    /// callers whose register layout has no slot for the ancillas at all.
    operation ApplyWithOwnedAncillas(
        op : Qubit[] => Unit is Adj + Ctl,
        prepareOwned : Qubit[] => Unit is Adj + Ctl,
        numOwned : Int,
        qs : Qubit[]
    ) : Unit is Adj + Ctl {
        if numOwned == 0 {
            op(qs);
        } else {
            use owned = Qubit[numOwned];
            within {
                prepareOwned(owned);
            } apply {
                op(qs + owned);
            }
        }
    }

    /// Returns `op` wrapped so it owns its trailing `numOwned` ancillas. See `ApplyWithOwnedAncillas`.
    function MakeOwnedAncillaOp(
        op : Qubit[] => Unit is Adj + Ctl,
        prepareOwned : Qubit[] => Unit is Adj + Ctl,
        numOwned : Int
    ) : Qubit[] => Unit is Adj + Ctl {
        ApplyWithOwnedAncillas(op, prepareOwned, numOwned, _)
    }

    /// Returns the maximum element of the given array of integers.
    function MaxInt(values : Int[]) : Int {
        // Caller is responsible for not passing an empty array.
        mutable max = values[0];
        for idx in 1 .. Length(values) - 1 {
            let value = values[idx];
            if (value > max) {
                set max = value;
            }
        }
        return max;
    }

    /// Creates a circuit for sequentially applying two operations on the same target qubits.
    operation MakeSequentialCircuit(
        first : Qubit[] => Unit,
        second : Qubit[] => Unit,
        targets : Int[]
    ) : Unit {
        if (Length(targets) == 0) {
            // No target indices: do nothing.
            return ();
        } else {
            // Allocate enough qubits so that all indices in 'targets' are valid.
            let maxTarget = MaxInt(targets);
            use qs = Qubit[1 + maxTarget];
            ApplySequential(first, second, Subarray(targets, qs));
        }
    }
}
