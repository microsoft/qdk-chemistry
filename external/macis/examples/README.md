# MACIS Examples

## General Run Configuration: `standalone_driver`

To run an example through the standalone MACIS driver, run the following command:

```bash
OMP_NUM_THREADS=<NT> /path/to/macis/build/examples/standalone_driver inputs/<input>.inp
```

where `<NT>` is the number of threads to use and `<input>.inp` is the input file for the example.

### Complete Active Space (CAS)

Input: `4e4o_ethylene_2det.inp`

This example computes the ground state energy of the ethylene molecule using full configuration interaction within a
4e4o active space (CAS). The input file specifies the spin sector (2 alpha and 2 beta electrons) and the FCIDUMP file.
The expected output for this example is

```text
[standalone_driver] E(CI)     = -77.996417943349 Eh
[standalone_driver] WFN FILE
[determinants] Print leading determinants > 1.00e-02
[determinants]  -0.983719579911   2200
[determinants]   0.179017351346   2020
[determinants]   0.012205392326   0220
```

### Selected Configuration Interaction (SCI)

Input: `4e4o_ethylene_2det_sci.inp`

This example computes the selected configuration interaction (SCI) energy of the ethylene molecule in a 4e4o active
space using the configurations selected in `4e4o_ethylene_wfn1.txt`. N.B. This input was constructed to meet the needs
of the `prepq` workflow and is not a typical input for a standalone SCI calculation. In general, SCI would proceed by
iteratively selecting the configuations that are important to the energy of the system. This input is constructed to
"trick" the MACIS SCI machinery to just computing the energy from a manual (e.g. SQD) selection. The expected output for
this example is

```text
[standalone_driver] E(CI)     = -77.996196608303 Eh
[standalone_driver] WFN FILE
[determinants] Print leading determinants > 1.00e-02
[determinants]  -0.983794758483   2200
[determinants]   0.179298279920   2020
```

## Automatic Determination of SCI size for Chemical Accuracy: `auto_sci`

To run an example through the `auto_sci` MACIS driver, run the following command:

```bash
OMP_NUM_THREADS=<NT> /path/to/macis/build/examples/auto_sci inputs/<input>.inp
```

where `<NT>` is the number of threads to use and `<input>.inp` is the input file for the example.

### Ethylene

Input: `4e4o_ethylene_2det.inp`

This example determines the number of configurations required to achieve chemical accuracy for the ethylene molecule.
Expected relevant output:

```text
[standalone_driver] Found 2 determinants to achieve chemical accuracy
```
