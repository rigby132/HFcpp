# HFcpp
A simple implementation of the Hartree-Fock method.

## Dependencies
This project depends on the OpenMP and Eigen libraries.

## Example usage
`./hfcpp -s water.xyz -b 6-311G.d2k -n 5`

 Calculates the HF-energies for the molecule defined in `water.xyz` with the basis-set defined in `6-311G.d2k` and `5` occupied orbitals or 10 electrons.

`../hfcpp -s NH3.xyz -b ../6-311+g\*\*.d2k -n 5 -d nh3_d.cube -w nh3_w.cube -o 6 --buffer 4`

 Calculates the HF-energies for the molecule defined in `NH3.xyz` with the basis-set defined in `6-311+g\*\*.d2k` and `5` occupied orbitals or 10 electrons.
 It also creates both wavefunction and density files for the first `6` orbitals with an additional buffer of `4` Angstrom around the molecule.
