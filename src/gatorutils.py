from mpi4py import MPI
import sys
import os

from veloxchem import mpi_master
from veloxchem import OutputStream
from veloxchem import Molecule
from veloxchem import MolecularBasis
from veloxchem import ScfRestrictedDriver
from gator import AdcDriver


class MockTask:

    def __init__(self, mol, basis, comm, ostream):

        self.molecule = mol
        self.ao_basis = basis
        self.mpi_comm = comm
        self.ostream = ostream


def get_molecule(mol_string):
    """
    Initializes a molecule.

    :param mol_string:
        The string for the molecule.
    :return:
        The molecule.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == mpi_master():
        if '.xyz' == mol_string[-4:] and os.path.isfile(mol_string):
            mol = Molecule.read_xyz(mol_string)
        else:
            mol = Molecule.read_str(mol_string)
    else:
        mol = Molecule()

    mol.broadcast(rank, comm)

    return mol


def get_molecular_basis(mol, basis_label):
    """
    Initializes a basis set.

    :param mol:
        The molecule.
    :param basis_label:
        The name of the basis set.
    :return:
        The molecular basis set.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    ostream = OutputStream()

    if rank == mpi_master():
        basis = MolecularBasis.read(mol, basis_label, '.', ostream)
    else:
        basis = MolecularBasis()

    basis.broadcast(rank, comm)

    return basis


def get_bool(kwargs, key):

    if key in kwargs:
        if isinstance(kwargs[key], bool):
            return kwargs[key]
        elif isinstance(kwargs[key], str):
            return (kwargs[key].lower() in ['yes', 'y'])
    return False


def run_scf(mol, basis, **kwargs):
    """
    Runs SCF.

    :param mol:
        The molecule.
    :param basis:
        The molecular basis set.
    :return:
        The scf driver.
    """

    verbose = get_bool(kwargs, 'verbose')

    comm = MPI.COMM_WORLD
    if verbose:
        ostream = OutputStream(sys.stdout)
    else:
        ostream = OutputStream()

    scf_drv = ScfRestrictedDriver(comm, ostream)
    scf_drv.update_settings(kwargs)
    scf_drv.compute(mol, basis)
    scf_drv.ostream.flush()

    if not verbose:
        valstr = 'SCF '
        if scf_drv.is_converged:
            valstr += 'converged in '
        else:
            valstr += 'NOT converged in '
        valstr += '{:d} iterations.'.format(scf_drv.num_iter)
        print(valstr, file=sys.stdout)
        if scf_drv.is_converged:
            valstr = 'Total Energy: {:.10f} au'.format(scf_drv.get_scf_energy())
            print(valstr, file=sys.stdout)
        sys.stdout.flush()

    return scf_drv


def run_adc(mol, basis, scf_drv, **kwargs):
    """
    Runs ADC.

    :param mol:
        The molecule.
    :param basis:
        The molecular basis set.
    :param scf_drv:
        The scf driver.
    :return:
        The result of adcc.run_adc.
    """

    verbose = get_bool(kwargs, 'verbose')

    comm = MPI.COMM_WORLD
    ostream = OutputStream(sys.stdout)

    adc_drv = AdcDriver(comm, ostream)
    adc_drv.update_settings(kwargs, scf_drv)
    adc_drv.ostream.flush()

    task = MockTask(mol, basis, comm, ostream)
    return adc_drv.compute(task, scf_drv, verbose)
