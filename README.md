# <img src="media/logo.png" height=120>

![CI](https://github.com/gator-program/gator/workflows/CI/badge.svg)
[![Anaconda-Server Badge](https://anaconda.org/gator/gator/badges/version.svg)](https://anaconda.org/gator/gator)


Gator is a program for computational spectroscopy and calculations
of molecular properties, currently using the algebraic diagrammatic construction (ADC) scheme for the polarization propagator.

## Installation

We recommend installation via the `conda` package manager
via the command
```
conda install -c gator gator
```
which will install Gator together with all its dependencies.
Afterwards, Gator can be run via
```
gator input.inp
```
or be alternatively imported as a Python module.

## Features
Calculations using ADC(2), ADC(2)-x, and ADC(3) are available via
[adcc](https://adc-connect.org), also with the core-valence separation (CVS) for computations in the X-ray region.
The [VeloxChem](https://veloxchem.org) program serves as SCF driver and provides all the necessary integrals.

Absorption cross sections are implemented using the complex
polarization propagator (CPP) approach via the
[respondo](https://github.com/gator-program/respondo) library.

Furthermore, Gator provides a hybrid OpenMP/MPI-parallelized implementation (HPC-QC) for
MP2 energies and ADC(2) excitation energies, which can be run on multiple cluster nodes using
distributed memory.

## Example Python Scripts

Gator can simply be used as a Python module, e.g., using the following
script for computing and plotting the X-ray absorption spectrum of water:
```Python
import gator
import matplotlib.pyplot as plt

water = gator.get_molecule("""
O   0.0    0.0  0.12
H  -0.7532 0.0 -0.475
H   0.7532 0.0 -0.475
""")

basis = gator.get_molecular_basis(water, '6-311++G**')
scf   = gator.run_scf(water, basis, verbose=False)
xas   = gator.run_adc(water, basis, scf, method='cvs-adc2', singlets=5, core_orbitals=1)

# plot spectrum
plt.figure(figsize=(4, 5))
xas.plot_spectrum()
plt.tight_layout()
plt.show()
```
In the background, Gator runs the SCF via VeloxChem and then dispatches the SCF
results to `adcc`, which computes the CVS-ADC(2) eigenstates. The resulting
spectrum can be directly visualized using the `plot_spectrum` function provided by `adcc`.
Of course, the calculation can also be run in a [Jupyter](https://jupyter.org) notebook.

## Example Input Files

An example input file for the computation of five singlet excited
states of water is shown below:
```
@jobs
task: adc
@end

@adc
method: adc2 
singlets: 5
@end

@method settings
basis: cc-pvdz
@end

@molecule
charge: 0
multiplicity: 1
units: au
xyz:  
O 0 0 0
H 0 0 1.795239827225189
H 1.693194615993441 0 -0.599043184453037
@end
```

This job will first run the SCF in VeloxChem and then hand over control
to `adcc`, which computes the ADC(2) eigenstates.

A more advanced example is the computation of the one-photon absorption
cross section with the CPP approach:
```
@jobs
task: adc
@end

@adc
method: cvs-adc2 (cpp)
frequencies: 20
damping: 0.001
core_orbitals: 1
@end

@method settings
basis: cc-pvdz
@end

@molecule
charge: 0
multiplicity: 1
units: au
xyz:  
O 0 0 0
H 0 0 1.795239827225189
H 1.693194615993441 0 -0.599043184453037
@end 
```

Here, the cross section is determined by solving the response equation
for the complex polarizability at a frequency of 20 a.u. with a damping parameter of 0.001.
This calculation also makes use of the CVS approximation to guarantee smooth convergence of the
response function in the X-ray regime.

### HPC-QC input file

To run the OpenMP/MPI-parallelized implementation for ADC(2) excitation energies
for water, the following input file is required:
```
@jobs
task: adc2
@end

@adc2
nstates: 5
@end

@method settings
basis: cc-pvdz
@end

@molecule
charge: 0
multiplicity: 1
units: au
xyz:  
O 0 0 0
H 0 0 1.795239827225189
H 1.693194615993441 0 -0.599043184453037
@end
```