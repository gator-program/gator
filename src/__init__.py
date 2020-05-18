# mpi4py
try:
    from mpi4py import MPI
except ImportError:
    raise ImportError('Unable to import mpi4py.MPI')

# Python classes
from .gatortask import GatorTask
from .scfdriver import ScfRestrictedDriver
from .mp2driver import Mp2Driver
from .adcdriver import AdcDriver

# Python functions
from .mpiutils import mpi_sanity_check

# Utility functions
from .gatorutils import get_molecule
from .gatorutils import get_molecular_basis
from .gatorutils import run_scf
from .gatorutils import run_adc

# Environment variable
import os
if 'OMP_NUM_THREADS' not in os.environ:
    import multiprocessing
    import sys
    ncores = multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(ncores)
    print('* Warning * Environment variable OMP_NUM_THREADS not set.',
          file=sys.stdout)
    print('* Warning * Setting OMP_NUM_THREADS to {:d}.'.format(ncores),
          file=sys.stdout)
